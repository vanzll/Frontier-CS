import torch
import triton
import triton.language as tl


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        # Large K, square-ish
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=4,
            num_warps=4,
        ),
        # Rectangular tiles
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32},
            num_stages=4,
            num_warps=8,
        ),
        # Smaller tiles / smaller K
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_m = tl.max_contiguous(offs_m, BLOCK_M)
    offs_n = tl.max_contiguous(offs_n, BLOCK_N)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = gelu(acc)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N) with GELU activation applied.
        On CUDA, uses a Triton kernel; otherwise falls back to PyTorch.
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("matmul only supports 2D tensors")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for matmul: {tuple(a.shape)} and {tuple(b.shape)}"
        )

    M, K = a.shape
    K_b, N = b.shape

    # Fallback to PyTorch for non-CUDA tensors or unsupported dtypes
    if (not a.is_cuda) or (not b.is_cuda):
        return F.gelu(a @ b)

    # Promote dtypes if necessary
    if a.dtype != b.dtype:
        common_dtype = torch.promote_types(a.dtype, b.dtype)
        a_mat = a.to(common_dtype)
        b_mat = b.to(common_dtype)
    else:
        common_dtype = a.dtype
        a_mat = a
        b_mat = b

    # Restrict to floating types; otherwise use fallback
    if common_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return F.gelu(a @ b)

    # Ensure tensors are on the same CUDA device
    if a_mat.device != b_mat.device:
        raise ValueError("Input tensors must be on the same device")

    c_mat = torch.empty((M, N), device=a_mat.device, dtype=common_dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _matmul_kernel[grid](
        a_mat,
        b_mat,
        c_mat,
        M,
        N,
        K,
        a_mat.stride(0),
        a_mat.stride(1),
        b_mat.stride(0),
        b_mat.stride(1),
        c_mat.stride(0),
        c_mat.stride(1),
    )

    return c_mat
'''
        return {"code": code}