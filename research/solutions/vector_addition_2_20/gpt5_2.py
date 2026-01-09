import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device"
    assert x.numel() == y.numel(), "Input tensors must have the same number of elements"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 4096
    grid = lambda META: (triton.cdiv(n_elements, BLOCK_SIZE),)
    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = """
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device"
    assert x.numel() == y.numel(), "Input tensors must have the same number of elements"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 4096
    grid = lambda META: (triton.cdiv(n_elements, BLOCK_SIZE),)
    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
    return out
"""
        return {"code": code}