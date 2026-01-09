import torch
import triton
import triton.language as tl


@triton.jit
def _vec_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    z = x + y
    tl.store(out_ptr + offsets, z, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "Shape mismatch"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    n = x.numel()
    # target size 1,048,576; support general case
    out = torch.empty_like(x)
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _vec_add_kernel[grid](
        x, y, out, n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=1
    )
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = """import torch
import triton
import triton.language as tl


@triton.jit
def _vec_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    z = x + y
    tl.store(out_ptr + offsets, z, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    \"""
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (1048576,)
        y: Input tensor of shape (1048576,)
    
    Returns:
        Output tensor of shape (1048576,) with x + y
    \"""
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "Shape mismatch"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _vec_add_kernel[grid](
        x, y, out, n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=1
    )
    return out
"""
        return {"code": code}