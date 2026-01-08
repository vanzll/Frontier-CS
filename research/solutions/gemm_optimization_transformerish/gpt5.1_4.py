import os
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32},
            num_stages=3, num_warps=4
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=3, num_warps=4
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=4, num_warps=2
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(c_ptrs, acc, mask=c_mask)


def _gelu_torch(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Inputs must be 2D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible dimensions for matmul")

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    if a.device.type != "cuda" or b.device.type != "cuda":
        return _gelu_torch(a @ b)

    if a.device != b.device:
        b = b.to(a.device)

    # Type promotion similar to torch.matmul
    if a.dtype != b.dtype:
        dtype = torch.result_type(a, b)
        a = a.to(dtype)
        b = b.to(dtype)
    dtype = a.dtype

    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        dtype = torch.float32

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N'])
    )

    matmul_gelu_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )

    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        path = os.path.abspath(__file__)
        with open(path, "r") as f:
            code = f.read()
        return {"code": code}