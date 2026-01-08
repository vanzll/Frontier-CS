import os
import math
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _make_configs():
    cfgs = []
    for bm, bn, bk, nw, ns, gm in [
        (128, 128, 32, 8, 4, 8),
        (128, 128, 64, 8, 5, 8),
        (128, 64, 32, 8, 4, 8),
        (128, 64, 64, 8, 5, 8),
        (64, 128, 32, 8, 4, 8),
        (64, 128, 64, 8, 5, 8),
        (64, 64, 32, 4, 4, 8),
        (64, 64, 64, 4, 5, 8),
        (64, 32, 32, 4, 3, 8),
        (32, 64, 32, 4, 3, 8),
        (32, 128, 32, 4, 4, 8),
        (128, 32, 32, 4, 4, 8),
    ]:
        cfgs.append(
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm},
                num_warps=nw,
                num_stages=ns,
            )
        )
    return cfgs


@triton.autotune(
    configs=_make_configs(),
    key=["M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn"],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    pid_group = pid // (group_size * num_pid_n)
    first_pid_m = pid_group * group_size
    group_pid_m = tl.minimum(num_pid_m - first_pid_m, group_size)
    pid_in_group = pid % (group_size * num_pid_n)
    pid_m = first_pid_m + (pid_in_group % group_pid_m)
    pid_n = pid_in_group // group_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0, cache_modifier=".ca")
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0, cache_modifier=".ca")
        acc = tl.dot(a, b, acc)
        k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = gelu(acc)
    out = tl.cast(out, OUT_DTYPE)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("matmul: inputs must be CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul: inputs must be 2D tensors")
    M, K_a = a.shape
    K_b, N = b.shape
    if K_a != K_b:
        raise ValueError(f"matmul: K mismatch, got {K_a} and {K_b}")

    if a.dtype != b.dtype:
        common = torch.promote_types(a.dtype, b.dtype)
        a = a.to(common)
        b = b.to(common)

    if a.dtype in (torch.float16, torch.bfloat16):
        out_dtype = a.dtype
    else:
        out_dtype = torch.float32

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _matmul_gelu_kernel[grid](
        a,
        b,
        c,
        M=M,
        N=N,
        K=K_a,
        stride_am=stride_am,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        stride_bn=stride_bn,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        OUT_DTYPE=tl.float16 if out_dtype == torch.float16 else (tl.bfloat16 if out_dtype == torch.bfloat16 else tl.float32),
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        path = None
        try:
            path = os.path.abspath(__file__)
        except Exception:
            path = None
        if path is not None and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return {"code": f.read()}
        return {"code": ""}