import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 2}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 2}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 2}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 2}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn'],
)
@triton.jit
def matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_size_m = GROUP_M
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size_m
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    acc = gelu(acc)
    c = acc.to(OUT_DTYPE)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _to_tl_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError('Unsupported dtype: ' + str(dtype))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes")
    if a.device.type != 'cuda' or b.device.type != 'cuda':
        out = a @ b
        return out * 0.5 * (1.0 + torch.erf(out * 0.7071067811865476))

    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        a = a.to(torch.float32)
    if b.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        b = b.to(torch.float32)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out_dtype = a.dtype if a.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    out = torch.empty((M, N), device=a.device, dtype=out_dtype)

    grid = lambda META: (
        ((M + META['BLOCK_M'] - 1) // META['BLOCK_M']) * ((N + META['BLOCK_N'] - 1) // META['BLOCK_N']),
    )

    matmul_gelu_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        OUT_DTYPE=_to_tl_dtype(out.dtype),
    )
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = """
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 2}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 2}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 2}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 2}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn'],
)
@triton.jit
def matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_size_m = GROUP_M
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size_m
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    acc = gelu(acc)
    c = acc.to(OUT_DTYPE)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _to_tl_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError('Unsupported dtype: ' + str(dtype))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    \"""
    Matrix multiplication with GELU activation.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    \"""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes")
    if a.device.type != 'cuda' or b.device.type != 'cuda':
        out = a @ b
        return out * 0.5 * (1.0 + torch.erf(out * 0.7071067811865476))

    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        a = a.to(torch.float32)
    if b.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        b = b.to(torch.float32)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out_dtype = a.dtype if a.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    out = torch.empty((M, N), device=a.device, dtype=out_dtype)

    grid = lambda META: (
        ((M + META['BLOCK_M'] - 1) // META['BLOCK_M']) * ((N + META['BLOCK_N'] - 1) // META['BLOCK_N']),
    )

    matmul_gelu_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        OUT_DTYPE=_to_tl_dtype(out.dtype),
    )
    return out
"""
        return {"code": code}