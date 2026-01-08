import os
import inspect
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _matmul_gelu_configs():
    return [
        triton.Config(
            {"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 4},
            num_warps=8,
            num_stages=5,
        ),
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 64, "GROUP_M": 4},
            num_warps=8,
            num_stages=4,
        ),
    ]


@triton.autotune(
    configs=_matmul_gelu_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_gelu_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid - group_id * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    k = 0
    while k < K:
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        k += BK
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    acc = gelu(acc.to(tl.float32))

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if OUT_DTYPE == 0:
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
    elif OUT_DTYPE == 1:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("matmul: inputs must be CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul: inputs must be rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul: incompatible shapes {tuple(a.shape)} x {tuple(b.shape)}")

    M, K = a.shape
    _, N = b.shape

    if a.dtype != b.dtype:
        b = b.to(dtype=a.dtype)

    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        a = a.to(dtype=torch.float16)
        b = b.to(dtype=torch.float16)

    if a.dtype == torch.float16:
        out_dtype_code = 0
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    elif a.dtype == torch.bfloat16:
        out_dtype_code = 1
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    else:
        out_dtype_code = 2
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),)
    _matmul_gelu_kernel[grid](
        a,
        b,
        c,
        M=M,
        N=N,
        K=K,
        stride_am=stride_am,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        stride_bn=stride_bn,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        OUT_DTYPE=out_dtype_code,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        try:
            path = __file__
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return {"code": f.read()}
        except Exception:
            pass
        try:
            import sys

            mod = sys.modules[__name__]
            return {"code": inspect.getsource(mod)}
        except Exception:
            return {"code": ""}