import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 4},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 192, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 4},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 192, 'BLOCK_K': 64,  'GROUP_M': 4},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4},  num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID and tile coordinates with grouped ordering for better L2 locality
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size_m = GROUP_M
    group_tiles = num_pid_n * group_size_m
    group_id = pid // group_tiles
    first_pid_m = group_id * group_size_m
    pid_in_group = pid % group_tiles
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    k_iter = 0
    while k_iter < K:
        k_mask = offs_k[None, :] + k_iter < K
        a_mask = (offs_m[:, None] < M) & k_mask
        b_mask = (offs_n[None, :] < N) & k_mask.T

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # Apply GELU to accumulator
    acc = gelu(acc)

    # Write back
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, f"Incompatible shapes: {a.shape} @ {b.shape}"

    # Determine output dtype (keep input dtype for output)
    out_dtype = a.dtype if a.dtype == b.dtype else torch.promote_types(a.dtype, b.dtype)

    # Create output
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    # Strides in elements
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Grid with grouped ordering
    def grid(meta):
        BLOCK_M = meta['BLOCK_M']
        BLOCK_N = meta['BLOCK_N']
        GROUP_M = meta['GROUP_M']
        num_pid_m = triton.cdiv(M, BLOCK_M)
        num_pid_n = triton.cdiv(N, BLOCK_N)
        return (triton.cdiv(num_pid_m, GROUP_M) * num_pid_n * GROUP_M,)

    _matmul_gelu_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 4},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 192, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 4},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 192, 'BLOCK_K': 64,  'GROUP_M': 4},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4},  num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID and tile coordinates with grouped ordering for better L2 locality
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size_m = GROUP_M
    group_tiles = num_pid_n * group_size_m
    group_id = pid // group_tiles
    first_pid_m = group_id * group_size_m
    pid_in_group = pid % group_tiles
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    k_iter = 0
    while k_iter < K:
        k_mask = offs_k[None, :] + k_iter < K
        a_mask = (offs_m[:, None] < M) & k_mask
        b_mask = (offs_n[None, :] < N) & k_mask.T

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # Apply GELU to accumulator
    acc = gelu(acc)

    # Write back
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, f"Incompatible shapes: {a.shape} @ {b.shape}"

    out_dtype = a.dtype if a.dtype == b.dtype else torch.promote_types(a.dtype, b.dtype)
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    def grid(meta):
        BLOCK_M = meta['BLOCK_M']
        BLOCK_N = meta['BLOCK_N']
        GROUP_M = meta['GROUP_M']
        num_pid_m = triton.cdiv(M, BLOCK_M)
        num_pid_n = triton.cdiv(N, BLOCK_N)
        return (triton.cdiv(num_pid_m, GROUP_M) * num_pid_n * GROUP_M,)

    _matmul_gelu_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return c
'''
        return {"code": code}