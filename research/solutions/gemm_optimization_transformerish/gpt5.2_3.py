import os
import tempfile
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=6),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: (args["M"] % args["BLOCK_M"]) == 0,
        "EVEN_N": lambda args: (args["N"] % args["BLOCK_N"]) == 0,
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_K"]) == 0,
    }
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    tl.multiple_of(K, 16)
    tl.multiple_of(N, 16)

    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    num_pid_in_group = group_size * grid_n
    pid_group = pid // num_pid_in_group
    first_pid_m = pid_group * group_size
    pid_in_group = pid - pid_group * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if EVEN_M:
        m_mask = tl.full((BLOCK_M,), True, tl.int1)
    else:
        m_mask = rm < M

    if EVEN_N:
        n_mask = tl.full((BLOCK_N,), True, tl.int1)
    else:
        n_mask = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        rk = k0 + tl.arange(0, BLOCK_K)

        if EVEN_K:
            a_mask = m_mask[:, None]
            b_mask = n_mask[None, :]
        else:
            k_mask = rk < K
            a_mask = m_mask[:, None] & k_mask[None, :]
            b_mask = k_mask[:, None] & n_mask[None, :]

        a = tl.load(
            a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        b = tl.load(
            b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=b_mask,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_last",
        )

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        k0 += BLOCK_K

    acc = gelu(acc)

    if OUT_DTYPE == tl.float16:
        out = acc.to(tl.float16)
    elif OUT_DTYPE == tl.bfloat16:
        out = acc.to(tl.bfloat16)
    else:
        out = acc.to(tl.float32)

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, out)
    else:
        tl.store(c_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("a and b must be CUDA tensors")
    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")

    M, K = a.shape
    _, N = b.shape

    if a.dtype == torch.float16:
        out_dtype = torch.float16
        out_tl = tl.float16
    elif a.dtype == torch.bfloat16:
        out_dtype = torch.bfloat16
        out_tl = tl.bfloat16
    elif a.dtype == torch.float32:
        out_dtype = torch.float32
        out_tl = tl.float32
    else:
        a = a.to(torch.float16)
        b = b.to(torch.float16)
        out_dtype = torch.float16
        out_tl = tl.float16

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

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
        OUT_DTYPE=out_tl,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": __file__}