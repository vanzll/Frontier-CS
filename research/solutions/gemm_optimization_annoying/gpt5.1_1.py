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
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
            num_stages=5,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_size_m = GROUP_M
    group_id = pid // (group_size_m * grid_n)
    first_pid_m = group_id * group_size_m
    pid_in_group = pid % (group_size_m * grid_n)
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k
        k_mask = k_offsets < K

        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + k_offsets[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + k_offsets[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )

        a_mask = mask_m[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    acc = gelu(acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensor")

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Input tensors must be 2D")

    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match for matrix multiplication")

    if a.device.type != "cuda" or b.device.type != "cuda":
        return torch.nn.functional.gelu(a @ b)

    if a.dtype != b.dtype:
        raise ValueError("Input tensors must have the same dtype")

    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        # Fallback for unsupported dtypes
        out = a.to(torch.float32) @ b.to(torch.float32)
        return torch.nn.functional.gelu(out).to(a.dtype)

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    a_ = a
    b_ = b

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stride_am, stride_ak = a_.stride()
    stride_bk, stride_bn = b_.stride()
    stride_cm, stride_cn = c.stride()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _matmul_gelu_kernel[grid](
        a_,
        b_,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
    )

    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}