import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offsets, BLOCK_SIZE)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.

    Args:
        x: Input tensor of shape (16777216,)
        y: Input tensor of shape (16777216,)

    Returns:
        Output tensor of shape (16777216,) with x + y
    """
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("Input tensors must be CUDA tensors")
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    x_flat = x.view(-1)
    y_flat = y.view(-1)
    n_elements = x_flat.numel()

    out = torch.empty_like(x_flat)

    # Heuristic block size for large vs small vectors
    if n_elements >= (1 << 20):
        BLOCK_SIZE = 4096
        num_warps = 8
    else:
        BLOCK_SIZE = 1024
        num_warps = 4

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _add_kernel[grid](x_flat, y_flat, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=2)

    return out.view_as(x)


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        kernel_code = """
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offsets, BLOCK_SIZE)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    \"""
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (16777216,)
        y: Input tensor of shape (16777216,)
    
    Returns:
        Output tensor of shape (16777216,) with x + y
    \"""
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("Input tensors must be CUDA tensors")
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    x_flat = x.view(-1)
    y_flat = y.view(-1)
    n_elements = x_flat.numel()

    out = torch.empty_like(x_flat)

    if n_elements >= (1 << 20):
        BLOCK_SIZE = 4096
        num_warps = 8
    else:
        BLOCK_SIZE = 1024
        num_warps = 4

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _add_kernel[grid](x_flat, y_flat, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=2)

    return out.view_as(x)
"""
        return {"code": kernel_code}