import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32},
            num_stages=4,
            num_warps=8,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        k += BLOCK_K

    acc = gelu(acc)

    c = acc.to(OUT_DTYPE)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


_TORCH_TO_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.
    a: (M, K), b: (K, N)
    Returns: (M, N)
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes for multiplication")
    if a.device != b.device:
        raise ValueError("Input tensors must be on the same device")

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Fallback for unsupported devices or dtypes
    if (not a.is_cuda) or (not b.is_cuda):
        c = a @ b
        return c * 0.5 * (1.0 + torch.erf(c * 0.7071067811865476))

    if a.dtype != b.dtype or a.dtype not in _TORCH_TO_TRITON_DTYPE:
        c = a @ b
        return c * 0.5 * (1.0 + torch.erf(c * 0.7071067811865476))

    out_dtype = a.dtype
    C = torch.empty((M, N), device=a.device, dtype=out_dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    _matmul_kernel[grid](
        a,
        b,
        C,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        C.stride(0),
        C.stride(1),
        OUT_DTYPE=_TORCH_TO_TRITON_DTYPE[out_dtype],
    )
    return C


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": __file__}