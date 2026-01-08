import os
import textwrap

KERNEL_CODE = r"""
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _make_configs():
    cfgs = []
    def add(bm, bn, bk, gm, warps, stages):
        cfgs.append(triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm},
            num_warps=warps,
            num_stages=stages,
        ))

    add(128, 128, 32, 8, 8, 5)
    add(128, 128, 64, 8, 8, 4)
    add(128, 128, 128, 8, 8, 3)

    add(128, 64, 64, 8, 4, 4)
    add(64, 128, 64, 8, 4, 4)
    add(64, 64, 64, 8, 4, 4)

    add(256, 128, 64, 4, 8, 4)
    add(128, 256, 64, 4, 8, 4)
    add(256, 64, 64, 4, 8, 4)
    add(64, 256, 64, 4, 8, 4)

    return cfgs


@triton.autotune(
    configs=_make_configs(),
    key=["M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn"],
)
@triton.jit
def _matmul_gelu_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid - group_id * num_pid_in_group

    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    # K is constexpr; this loop becomes static/unrolled in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_mask[None, :]),
            other=0.0,
            eviction_policy="evict_last",
        )
        b = tl.load(
            b_ptrs,
            mask=(k_mask[:, None]) & (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += 1

    acc = gelu(acc)
    c = tl.cast(acc, OUT_DTYPE)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("CUDA tensors required")
    if a.dtype != b.dtype:
        raise ValueError("a and b must have same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported dtype")

    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError("incompatible shapes")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    out_dtype = tl.float16 if a.dtype == torch.float16 else (tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float32)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _matmul_gelu_kernel[grid](
        a, b, c,
        M=M, N=N, K=K,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
        OUT_DTYPE=out_dtype,
    )
    return c
"""

exec(KERNEL_CODE, globals(), globals())


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": KERNEL_CODE}