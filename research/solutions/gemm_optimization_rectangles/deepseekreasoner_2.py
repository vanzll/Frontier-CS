import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride parameters
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if EVEN_K:
        # Compute without boundary checks when K is multiple of BLOCK_SIZE_K
        for k in range(0, K, BLOCK_SIZE_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b, out_dtype=tl.float32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    else:
        # Compute with boundary checks for arbitrary K
        for k in range(0, K, BLOCK_SIZE_K):
            k_remaining = K - k
            if k_remaining >= BLOCK_SIZE_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
            accumulator += tl.dot(a, b, out_dtype=tl.float32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Apply activation function
    if ACTIVATION:
        accumulator = gelu(accumulator)
    
    # Write back the block of C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Check dimensions
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {a.shape} @ {b.shape}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # Autotuned configurations
    configs = [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ]
    
    # Heuristic for choosing configuration
    def heuristic(config):
        # Prefer configurations that better tile the shapes
        m_blocks = triton.cdiv(M, config.kwargs['BLOCK_SIZE_M'])
        n_blocks = triton.cdiv(N, config.kwargs['BLOCK_SIZE_N'])
        total_blocks = m_blocks * n_blocks
        # Favor square-ish blocks for better utilization
        block_ratio = max(config.kwargs['BLOCK_SIZE_M'], config.kwargs['BLOCK_SIZE_N']) / \
                      min(config.kwargs['BLOCK_SIZE_M'], config.kwargs['BLOCK_SIZE_N'])
        # Favor larger K blocks for arithmetic intensity
        k_efficiency = min(K, config.kwargs['BLOCK_SIZE_K']) / config.kwargs['BLOCK_SIZE_K']
        # Combine factors (lower score is better)
        score = block_ratio * 0.3 + (1.0 / k_efficiency) * 0.7
        return score
    
    best_config = min(configs, key=heuristic)
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=best_config.kwargs['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=best_config.kwargs['BLOCK_SIZE_K'],
        GROUP_SIZE_M=best_config.kwargs['GROUP_SIZE_M'],
        EVEN_K=K % best_config.kwargs['BLOCK_SIZE_K'] == 0,
        ACTIVATION=True,
        num_stages=best_config.num_stages,
        num_warps=best_config.num_warps,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = """import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride parameters
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if EVEN_K:
        # Compute without boundary checks when K is multiple of BLOCK_SIZE_K
        for k in range(0, K, BLOCK_SIZE_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b, out_dtype=tl.float32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    else:
        # Compute with boundary checks for arbitrary K
        for k in range(0, K, BLOCK_SIZE_K):
            k_remaining = K - k
            if k_remaining >= BLOCK_SIZE_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
            accumulator += tl.dot(a, b, out_dtype=tl.float32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Apply activation function
    if ACTIVATION:
        accumulator = gelu(accumulator)
    
    # Write back the block of C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Check dimensions
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {a.shape} @ {b.shape}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # Autotuned configurations
    configs = [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ]
    
    # Heuristic for choosing configuration
    def heuristic(config):
        # Prefer configurations that better tile the shapes
        m_blocks = triton.cdiv(M, config.kwargs['BLOCK_SIZE_M'])
        n_blocks = triton.cdiv(N, config.kwargs['BLOCK_SIZE_N'])
        total_blocks = m_blocks * n_blocks
        # Favor square-ish blocks for better utilization
        block_ratio = max(config.kwargs['BLOCK_SIZE_M'], config.kwargs['BLOCK_SIZE_N']) / \
                      min(config.kwargs['BLOCK_SIZE_M'], config.kwargs['BLOCK_SIZE_N'])
        # Favor larger K blocks for arithmetic intensity
        k_efficiency = min(K, config.kwargs['BLOCK_SIZE_K']) / config.kwargs['BLOCK_SIZE_K']
        # Combine factors (lower score is better)
        score = block_ratio * 0.3 + (1.0 / k_efficiency) * 0.7
        return score
    
    best_config = min(configs, key=heuristic)
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=best_config.kwargs['BLOCK_SIZE_M'],
        BLOCK_SIZE_N=best_config.kwargs['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=best_config.kwargs['BLOCK_SIZE_K'],
        GROUP_SIZE_M=best_config.kwargs['GROUP_SIZE_M'],
        EVEN_K=K % best_config.kwargs['BLOCK_SIZE_K'] == 0,
        ACTIVATION=True,
        num_stages=best_config.num_stages,
        num_warps=best_config.num_warps,
    )
    return c
"""
        return {"code": code}