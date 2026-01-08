import os
import math
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _make_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=5,
        ),
    ]


@triton.autotune(
    configs=_make_autotune_configs(),
    key=["M", "N", "K", "OUT_DTYPE"],
    num_warmups=1,
    num_reps=1,
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M * grid_n
    pid_group = pid // group_size
    pid_in_group = pid - pid_group * group_size

    first_pid_m = pid_group * GROUP_M
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_M and EVEN_N and EVEN_K:
        tl.multiple_of(stride_ak, 8)
        tl.multiple_of(stride_bn, 8)
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        m_mask = offs_m < M
        n_mask = offs_n < N
        for k0 in range(0, K, BLOCK_K):
            k_mask = (k0 + offs_k) < K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    acc = gelu(acc)
    c = acc.to(OUT_DTYPE)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)} b={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device.type != "cuda":
        out = a @ b
        return torch.nn.functional.gelu(out)

    if a.dtype != b.dtype:
        raise TypeError(f"dtype mismatch: a.dtype={a.dtype} b.dtype={b.dtype}")

    if a.dtype == torch.float16:
        out_dtype = tl.float16
        out_torch_dtype = torch.float16
    elif a.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
        out_torch_dtype = torch.bfloat16
    elif a.dtype == torch.float32:
        out_dtype = tl.float32
        out_torch_dtype = torch.float32
    else:
        raise TypeError(f"unsupported dtype: {a.dtype}")

    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=out_torch_dtype)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    # Conservative defaults for divisibility; specialized per autotune config via constexpr args
    # EVEN_* are passed as constexpr to allow mask elimination in the common case.
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
        OUT_DTYPE=out_dtype,
        EVEN_M=False,
        EVEN_N=False,
        EVEN_K=False,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}