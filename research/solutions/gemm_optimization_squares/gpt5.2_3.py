import os
import torch
import triton
import triton.language as tl

KERNEL_SRC = r"""
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=5),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=5),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=5),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
]

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
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

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_group = pid // (GROUP_M * num_pid_n)
    first_pid_m = pid_group * GROUP_M
    pid_in_group = pid % (GROUP_M * num_pid_n)
    pid_m = first_pid_m + (pid_in_group // num_pid_n)
    pid_n = pid_in_group % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tl.max_contiguous(offs_m, BLOCK_M)
    tl.max_contiguous(offs_n, BLOCK_N)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_mask = (k + offs_k) < K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_mask[None, :]),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_mask[:, None]) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes")
    if (not a.is_cuda) or (not b.is_cuda):
        c = a @ b
        return torch.nn.functional.gelu(c, approximate="none")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        c = a @ b
        return torch.nn.functional.gelu(c, approximate="none")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _matmul_gelu_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c
"""

exec(compile(KERNEL_SRC, "<triton_gemm_gelu>", "exec"), globals(), globals())


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": KERNEL_SRC}