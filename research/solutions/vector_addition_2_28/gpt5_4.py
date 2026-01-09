import os
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier="cg")
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0, cache_modifier="cg")
    tl.store(z_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("Input tensors must be contiguous")
    if x.device.type != y.device.type:
        raise ValueError("Input tensors must be on the same device")
    if x.device.type == "cuda":
        if x.dtype != y.dtype:
            raise ValueError("Input tensors must have the same dtype")
        z = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 4096
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)
        return z
    else:
        return x + y


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier="cg")
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0, cache_modifier="cg")
    tl.store(z_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("Input tensors must be contiguous")
    if x.device.type != y.device.type:
        raise ValueError("Input tensors must be on the same device")
    if x.device.type == "cuda":
        if x.dtype != y.dtype:
            raise ValueError("Input tensors must have the same dtype")
        z = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 4096
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)
        return z
    else:
        return x + y
'''
        return {"code": code}