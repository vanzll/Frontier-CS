import math
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, LOOPS: tl.constexpr):
    pid = tl.program_id(0)
    base_idx = pid * BLOCK_SIZE * LOOPS
    arange = tl.arange(0, BLOCK_SIZE)
    for i in range(LOOPS):
        offsets = base_idx + i * BLOCK_SIZE + arange
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    assert x.shape == y.shape and x.ndim == 1, "Inputs must be 1D tensors of the same shape"
    assert x.dtype == y.dtype, "Inputs must have the same dtype"

    n_elements = x.numel()
    out = torch.empty_like(x)

    # Tuned configuration for large vectors and L4 GPU
    BLOCK_SIZE = 4096
    LOOPS = 4
    grid = (triton.cdiv(n_elements, BLOCK_SIZE * LOOPS),)

    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, LOOPS=LOOPS, num_warps=8, num_stages=1)
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = r'''
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, LOOPS: tl.constexpr):
    pid = tl.program_id(0)
    base_idx = pid * BLOCK_SIZE * LOOPS
    arange = tl.arange(0, BLOCK_SIZE)
    for i in range(LOOPS):
        offsets = base_idx + i * BLOCK_SIZE + arange
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    assert x.shape == y.shape and x.ndim == 1, "Inputs must be 1D tensors of the same shape"
    assert x.dtype == y.dtype, "Inputs must have the same dtype"

    n_elements = x.numel()
    out = torch.empty_like(x)

    # Tuned configuration for large vectors and L4 GPU
    BLOCK_SIZE = 4096
    LOOPS = 4
    grid = (triton.cdiv(n_elements, BLOCK_SIZE * LOOPS),)

    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, LOOPS=LOOPS, num_warps=8, num_stages=1)
    return out
'''
        return {"code": code}