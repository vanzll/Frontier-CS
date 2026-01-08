import os
import pathlib
import inspect
import sys
from typing import Dict, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _configs():
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=5,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=8,
            num_stages=5,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=8,
            num_stages=5,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
    ]


@triton.autotune(
    configs=_configs(),
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args, meta: (args["M"] % meta["BLOCK_M"]) == 0,
        "EVEN_N": lambda args, meta: (args["N"] % meta["BLOCK_N"]) == 0,
        "EVEN_K": lambda args, meta: (args["K"] % meta["BLOCK_K"]) == 0,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size_m = GROUP_M
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size_m
    pid_in_group = pid - group_id * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    a_blk = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_blk = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    tl.multiple_of(stride_ak, 1)
    tl.multiple_of(stride_bn, 1)

    k = 0
    if EVEN_K and EVEN_M and EVEN_N:
        while k < K:
            a = tl.load(a_blk)
            b = tl.load(b_blk)
            acc = tl.dot(a, b, acc)
            a_blk = tl.advance(a_blk, (0, BLOCK_K))
            b_blk = tl.advance(b_blk, (BLOCK_K, 0))
            k += BLOCK_K
    else:
        while k < K:
            a = tl.load(a_blk, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(b_blk, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(a, b, acc)
            a_blk = tl.advance(a_blk, (0, BLOCK_K))
            b_blk = tl.advance(b_blk, (BLOCK_K, 0))
            k += BLOCK_K

    acc = gelu(acc)

    c_blk = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if EVEN_M and EVEN_N:
        tl.store(c_blk, acc)
    else:
        tl.store(c_blk, acc, boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("matmul expects CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes")

    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

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
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        try:
            code = pathlib.Path(__file__).read_text()
            return {"code": code}
        except Exception:
            try:
                code = inspect.getsource(sys.modules[__name__])
                return {"code": code}
            except Exception:
                return {"program_path": os.path.abspath(__file__)}