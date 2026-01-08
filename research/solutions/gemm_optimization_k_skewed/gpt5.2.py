import os
import math
import inspect
import textwrap
from typing import Optional, Dict

import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _make_configs(block_m: int, block_n: int):
    cfgs = []
    # Small-K leaning
    cfgs.append(triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3))
    cfgs.append(triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4))
    cfgs.append(triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4))
    # Large-K leaning
    cfgs.append(triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=5))
    cfgs.append(triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=5))
    return cfgs


@triton.autotune(
    configs=_make_configs(128, 128),
    key=["K"],
)
@triton.jit
def _matmul_gelu_balanced(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUTPUT_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    # If group straddles the end, some pids can be out-of-range.
    if pid_m >= grid_m:
        return

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_blk = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_blk = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a = tl.load(a_blk, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_blk, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        a_blk = tl.advance(a_blk, (0, BLOCK_K))
        b_blk = tl.advance(b_blk, (BLOCK_K, 0))
        k += BLOCK_K

    acc = gelu(acc)

    if OUTPUT_FP16:
        out = acc.to(tl.float16)
    else:
        out = acc.to(tl.bfloat16)

    c_blk = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_blk, out, boundary_check=(0, 1))


@triton.autotune(
    configs=_make_configs(256, 64) + _make_configs(128, 64),
    key=["K"],
)
@triton.jit
def _matmul_gelu_tall_narrow(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUTPUT_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    if pid_m >= grid_m:
        return

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_blk = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_blk = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a = tl.load(a_blk, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_blk, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        a_blk = tl.advance(a_blk, (0, BLOCK_K))
        b_blk = tl.advance(b_blk, (BLOCK_K, 0))
        k += BLOCK_K

    acc = gelu(acc)

    if OUTPUT_FP16:
        out = acc.to(tl.float16)
    else:
        out = acc.to(tl.bfloat16)

    c_blk = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_blk, out, boundary_check=(0, 1))


@triton.autotune(
    configs=_make_configs(256, 128) + _make_configs(128, 128),
    key=["K"],
)
@triton.jit
def _matmul_gelu_tall_wide(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUTPUT_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    if pid_m >= grid_m:
        return

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_blk = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_blk = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a = tl.load(a_blk, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_blk, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        a_blk = tl.advance(a_blk, (0, BLOCK_K))
        b_blk = tl.advance(b_blk, (BLOCK_K, 0))
        k += BLOCK_K

    acc = gelu(acc)

    if OUTPUT_FP16:
        out = acc.to(tl.float16)
    else:
        out = acc.to(tl.bfloat16)

    c_blk = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_blk, out, boundary_check=(0, 1))


@triton.autotune(
    configs=_make_configs(64, 256) + _make_configs(64, 128),
    key=["K"],
)
@triton.jit
def _matmul_gelu_wide_short(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUTPUT_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    if pid_m >= grid_m:
        return

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_blk = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_blk = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a = tl.load(a_blk, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_blk, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        a_blk = tl.advance(a_blk, (0, BLOCK_K))
        b_blk = tl.advance(b_blk, (BLOCK_K, 0))
        k += BLOCK_K

    acc = gelu(acc)

    if OUTPUT_FP16:
        out = acc.to(tl.float16)
    else:
        out = acc.to(tl.bfloat16)

    c_blk = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_blk, out, boundary_check=(0, 1))


@triton.autotune(
    configs=_make_configs(128, 256) + _make_configs(128, 128),
    key=["K"],
)
@triton.jit
def _matmul_gelu_wide_tall(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUTPUT_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    if pid_m >= grid_m:
        return

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_blk = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_blk = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a = tl.load(a_blk, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_blk, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        a_blk = tl.advance(a_blk, (0, BLOCK_K))
        b_blk = tl.advance(b_blk, (BLOCK_K, 0))
        k += BLOCK_K

    acc = gelu(acc)

    if OUTPUT_FP16:
        out = acc.to(tl.float16)
    else:
        out = acc.to(tl.bfloat16)

    c_blk = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_blk, out, boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16):
        out = a @ b
        return torch.nn.functional.gelu(out, approximate="none")

    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    output_fp16 = (a.dtype == torch.float16)

    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)

    # Simple shape-based dispatch tailored to evaluation shapes
    if M >= 1024 and N <= 256:
        grid = (triton.cdiv(M, 256) * triton.cdiv(N, 64),)
        _matmul_gelu_tall_narrow[grid](
            a, b, c,
            M=M, N=N, K=K,
            stride_am=stride_am, stride_ak=stride_ak,
            stride_bk=stride_bk, stride_bn=stride_bn,
            stride_cm=stride_cm, stride_cn=stride_cn,
            OUTPUT_FP16=output_fp16,
            num_warps=8,
        )
        return c

    if M >= 1024 and N <= 1024:
        grid = (triton.cdiv(M, 256) * triton.cdiv(N, 128),)
        _matmul_gelu_tall_wide[grid](
            a, b, c,
            M=M, N=N, K=K,
            stride_am=stride_am, stride_ak=stride_ak,
            stride_bk=stride_bk, stride_bn=stride_bn,
            stride_cm=stride_cm, stride_cn=stride_cn,
            OUTPUT_FP16=output_fp16,
            num_warps=8,
        )
        return c

    if N >= 1024 and M <= 256:
        grid = (triton.cdiv(M, 64) * triton.cdiv(N, 256),)
        _matmul_gelu_wide_short[grid](
            a, b, c,
            M=M, N=N, K=K,
            stride_am=stride_am, stride_ak=stride_ak,
            stride_bk=stride_bk, stride_bn=stride_bn,
            stride_cm=stride_cm, stride_cn=stride_cn,
            OUTPUT_FP16=output_fp16,
            num_warps=8,
        )
        return c

    if N >= 1024:
        grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)
        _matmul_gelu_wide_tall[grid](
            a, b, c,
            M=M, N=N, K=K,
            stride_am=stride_am, stride_ak=stride_ak,
            stride_bk=stride_bk, stride_bn=stride_bn,
            stride_cm=stride_cm, stride_cn=stride_cn,
            OUTPUT_FP16=output_fp16,
            num_warps=8,
        )
        return c

    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    _matmul_gelu_balanced[grid](
        a, b, c,
        M=M, N=N, K=K,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
        OUTPUT_FP16=output_fp16,
        num_warps=8,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        try:
            path = os.path.abspath(__file__)
            if os.path.exists(path):
                return {"program_path": path}
        except Exception:
            pass
        # Fallback: return source code string
        try:
            src = inspect.getsource(inspect.getmodule(Solution))
            return {"code": src}
        except Exception:
            return {"code": ""}