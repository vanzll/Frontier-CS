import os
import math
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _gelu_torch(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


_configs = [
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
]


@triton.autotune(configs=_configs, key=["M", "N"])
@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_K"]) == 0,
    }
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
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M * grid_n
    pid_group = pid // group_size
    first_pid_m = pid_group * GROUP_M
    pid_in_group = pid - pid_group * group_size
    group_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid_in_group % group_m)
    pid_n = pid_in_group // group_m

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_block_ptr, boundary_check=(0,), padding_option="zero")
            b = tl.load(
                b_block_ptr,
                boundary_check=(1,),
                padding_option="zero",
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc=acc)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    else:
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc=acc)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    out = gelu(acc)
    out = tl.cast(out, OUT_DTYPE)

    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, out, boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("a and b must be torch.Tensor")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device.type != "cuda":
        c = a @ b
        return _gelu_torch(c)

    if a.dtype != b.dtype:
        common = torch.promote_types(a.dtype, b.dtype)
        if common not in (torch.float16, torch.bfloat16):
            c = (a.to(common) @ b.to(common))
            return _gelu_torch(c)
        a = a.to(common)
        b = b.to(common)

    if a.dtype not in (torch.float16, torch.bfloat16):
        c = a @ b
        return _gelu_torch(c)

    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    out_dtype = tl.float16 if a.dtype == torch.float16 else tl.bfloat16

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _matmul_gelu_kernel[grid](
        a,
        b,
        c,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        OUT_DTYPE=out_dtype,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}