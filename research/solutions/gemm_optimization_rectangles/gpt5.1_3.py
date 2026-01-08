import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m_broadcast = offs_m[:, None]
    offs_n_broadcast = offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)

        offs_k_broadcast_row = offs_k[None, :]
        offs_k_broadcast_col = offs_k[:, None]

        a_ptrs = a_ptr + offs_m_broadcast * stride_am + offs_k_broadcast_row * stride_ak
        b_ptrs = b_ptr + offs_k_broadcast_col * stride_bk + offs_n_broadcast * stride_bn

        a_mask = (offs_m_broadcast < M) & (offs_k_broadcast_row < K)
        b_mask = (offs_k_broadcast_col < K) & (offs_n_broadcast < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32)

        k += BLOCK_K

    acc = gelu(acc)

    if DTYPE == 0:
        acc_converted = acc.to(tl.float16)
    elif DTYPE == 1:
        acc_converted = acc
    else:
        acc_converted = acc.to(tl.bfloat16)

    c_ptrs = c_ptr + offs_m_broadcast * stride_cm + offs_n_broadcast * stride_cn
    c_mask = (offs_m_broadcast < M) & (offs_n_broadcast < N)
    tl.store(c_ptrs, acc_converted, mask=c_mask)


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
        raise ValueError("Inputs must be 2D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes for multiplication")
    if a.dtype != b.dtype:
        raise ValueError("Input tensors must have the same dtype")

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    if not a.is_cuda or not b.is_cuda:
        x = torch.matmul(a, b)
        return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))

    if a.dtype == torch.float16:
        dtype_id = 0
    elif a.dtype == torch.float32:
        dtype_id = 1
    elif a.dtype == torch.bfloat16:
        dtype_id = 2
    else:
        x = torch.matmul(a, b)
        return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        DTYPE=dtype_id,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": __file__}