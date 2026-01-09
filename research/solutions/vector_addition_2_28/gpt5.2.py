import os
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel_unrolled_nomask(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    pid = tl.program_id(0)

    tl.multiple_of(x_ptr, 128)
    tl.multiple_of(y_ptr, 128)
    tl.multiple_of(out_ptr, 128)

    block_start = pid * BLOCK * UNROLL
    tl.multiple_of(block_start, 256)

    r = tl.arange(0, BLOCK)
    tl.max_contiguous(r, BLOCK)

    # Unrolled, no mask (fast path for N divisible by BLOCK*UNROLL)
    tl.static_assert(UNROLL >= 1)
    for i in tl.static_range(UNROLL):
        offs = block_start + i * BLOCK + r
        x = tl.load(x_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first").to(tl.float32)
        y = tl.load(y_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first").to(tl.float32)
        z = x + y
        tl.store(out_ptr + offs, z, cache_modifier=".cg", eviction_policy="evict_first")


@triton.jit
def _add_kernel_unrolled_mask(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    pid = tl.program_id(0)

    tl.multiple_of(x_ptr, 128)
    tl.multiple_of(y_ptr, 128)
    tl.multiple_of(out_ptr, 128)

    block_start = pid * BLOCK * UNROLL
    r = tl.arange(0, BLOCK)
    tl.max_contiguous(r, BLOCK)

    for i in tl.static_range(UNROLL):
        offs = block_start + i * BLOCK + r
        m = offs < n_elements
        x = tl.load(x_ptr + offs, mask=m, other=0, cache_modifier=".cg", eviction_policy="evict_first").to(tl.float32)
        y = tl.load(y_ptr + offs, mask=m, other=0, cache_modifier=".cg", eviction_policy="evict_first").to(tl.float32)
        z = x + y
        tl.store(out_ptr + offs, z, mask=m, cache_modifier=".cg", eviction_policy="evict_first")


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda" or y.device.type != "cuda":
        return x + y
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.ndim != 1:
        x = x.reshape(-1)
        y = y.reshape(-1)
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    n = x.numel()
    out = torch.empty_like(x)

    # Tuned parameters for bandwidth on large vectors.
    BLOCK = 1024
    UNROLL = 4
    num_warps = 8
    num_stages = 1

    chunk = BLOCK * UNROLL
    if n % chunk == 0:
        grid = (n // chunk,)
        _add_kernel_unrolled_nomask[grid](x, y, out, BLOCK=BLOCK, UNROLL=UNROLL, num_warps=num_warps, num_stages=num_stages)
    else:
        grid = (triton.cdiv(n, chunk),)
        _add_kernel_unrolled_mask[grid](x, y, out, n, BLOCK=BLOCK, UNROLL=UNROLL, num_warps=num_warps, num_stages=num_stages)

    return out


KERNEL_CODE = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel_unrolled_nomask(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    pid = tl.program_id(0)

    tl.multiple_of(x_ptr, 128)
    tl.multiple_of(y_ptr, 128)
    tl.multiple_of(out_ptr, 128)

    block_start = pid * BLOCK * UNROLL
    tl.multiple_of(block_start, 256)

    r = tl.arange(0, BLOCK)
    tl.max_contiguous(r, BLOCK)

    for i in tl.static_range(UNROLL):
        offs = block_start + i * BLOCK + r
        x = tl.load(x_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first").to(tl.float32)
        y = tl.load(y_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first").to(tl.float32)
        z = x + y
        tl.store(out_ptr + offs, z, cache_modifier=".cg", eviction_policy="evict_first")


@triton.jit
def _add_kernel_unrolled_mask(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    pid = tl.program_id(0)

    tl.multiple_of(x_ptr, 128)
    tl.multiple_of(y_ptr, 128)
    tl.multiple_of(out_ptr, 128)

    block_start = pid * BLOCK * UNROLL
    r = tl.arange(0, BLOCK)
    tl.max_contiguous(r, BLOCK)

    for i in tl.static_range(UNROLL):
        offs = block_start + i * BLOCK + r
        m = offs < n_elements
        x = tl.load(x_ptr + offs, mask=m, other=0, cache_modifier=".cg", eviction_policy="evict_first").to(tl.float32)
        y = tl.load(y_ptr + offs, mask=m, other=0, cache_modifier=".cg", eviction_policy="evict_first").to(tl.float32)
        z = x + y
        tl.store(out_ptr + offs, z, mask=m, cache_modifier=".cg", eviction_policy="evict_first")


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda" or y.device.type != "cuda":
        return x + y
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.ndim != 1:
        x = x.reshape(-1)
        y = y.reshape(-1)
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    n = x.numel()
    out = torch.empty_like(x)

    BLOCK = 1024
    UNROLL = 4
    num_warps = 8
    num_stages = 1

    chunk = BLOCK * UNROLL
    if n % chunk == 0:
        grid = (n // chunk,)
        _add_kernel_unrolled_nomask[grid](x, y, out, BLOCK=BLOCK, UNROLL=UNROLL, num_warps=num_warps, num_stages=num_stages)
    else:
        grid = (triton.cdiv(n, chunk),)
        _add_kernel_unrolled_mask[grid](x, y, out, n, BLOCK=BLOCK, UNROLL=UNROLL, num_warps=num_warps, num_stages=num_stages)

    return out
'''


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": KERNEL_CODE}