import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.multiple_of(offsets, BLOCK_SIZE)
    tl.max_contiguous(offsets, BLOCK_SIZE)

    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0)
    out_vals = x_vals + y_vals
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")
    if x.dtype != y.dtype:
        raise ValueError("Input tensors must have the same dtype.")
    if x.numel() != 1048576:
        # Still handle any size generically
        pass

    if x.is_cuda and y.is_cuda:
        n_elements = x.numel()
        out = torch.empty_like(x)
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
        return out
    else:
        return x + y


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.multiple_of(offsets, BLOCK_SIZE)
    tl.max_contiguous(offsets, BLOCK_SIZE)

    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0)
    out_vals = x_vals + y_vals
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")
    if x.dtype != y.dtype:
        raise ValueError("Input tensors must have the same dtype.")
    if x.is_cuda and y.is_cuda:
        n_elements = x.numel()
        out = torch.empty_like(x)
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
        return out
    else:
        return x + y
'''
        return {"code": code}