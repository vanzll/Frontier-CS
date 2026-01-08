import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


configs = [
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
        num_warps=8,
        num_stages=4,
    ),
]


@triton.autotune(
    configs=configs,
    key=[
        'M',
        'N',
        'K',
        'a_stride_am',
        'a_stride_ak',
        'b_stride_bk',
        'b_stride_bn',
        'c_stride_cm',
        'c_stride_cn',
    ],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    a_stride_am,
    a_stride_ak,
    b_stride_bk,
    b_stride_bn,
    c_stride_cm,
    c_stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + (offs_m[:, None] * a_stride_am + offs_k[None, :] * a_stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * b_stride_bk + offs_n[None, :] * b_stride_bn)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] + k < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * a_stride_ak
        b_ptrs += BLOCK_K * b_stride_bk

    acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * c_stride_cm + offs_n[None, :] * c_stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N) with GELU activation applied.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Input tensors must be 2-dimensional")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match for matmul")
    if a.device.type != "cuda" or b.device.type != "cuda":
        # CPU or non-CUDA fallback
        out = a @ b
        return F.gelu(out)

    if a.dtype != b.dtype:
        b = b.to(a.dtype)

    M, K = a.shape
    Kb, N = b.shape
    if Kb != K:
        raise ValueError("Inner dimensions must match for matmul")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    a_stride_am, a_stride_ak = a.stride()
    b_stride_bk, b_stride_bn = b.stride()
    c_stride_cm, c_stride_cn = c.stride()

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a_stride_am,
        a_stride_ak,
        b_stride_bk,
        b_stride_bn,
        c_stride_cm,
        c_stride_cn,
    )

    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}