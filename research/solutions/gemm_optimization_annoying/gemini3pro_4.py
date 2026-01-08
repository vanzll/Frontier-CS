import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = r"""
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

def get_configs():
    configs = []
    # Optimized configs for L4 (Ada Lovelace)
    # Covering various block sizes to handle awkward shapes efficiently
    for num_stages in [3, 4, 5]:
        for num_warps in [4, 8]:
            # Balanced tiles
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=num_stages, num_warps=num_warps))
            configs.append(triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=num_stages, num_warps=num_warps))
            # Rectangular tiles
            configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=num_stages, num_warps=num_warps))
            configs.append(triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=num_stages, num_warps=num_warps))

    # Smaller tiles for very small or highly irregular shapes
    configs.append(triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2))
    
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Map program ids to block of C
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Masks for M and N dimensions (constant across K loop)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Iterate calculating a window of K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Calculate mask for K dimension
        mask_k = (k * BLOCK_K + offs_k) < K
        
        # Load A and B with masking
        # For A: valid if row < M and col < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        # For B: valid if row < K and col < N
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply GELU
    c = gelu(accumulator)

    # Store result
    c_ptrs = c_ptr + (stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :])
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Basic validation
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    M, K = a.shape
    K, N = b.shape
    
    # Alloc output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Grid definition
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c
"""
        return {"code": code}