import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, EVEN_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    if EVEN_N:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        tl.store(out_ptr + offsets, x + y)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.device.type != 'cuda' or y.device.type != 'cuda':
        return x + y
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    assert x.shape == y.shape, "Inputs must have the same shape"
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    EVEN_N = (n % BLOCK_SIZE) == 0
    num_warps = 8
    num_stages = 2
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, EVEN_N=EVEN_N, num_warps=num_warps, num_stages=num_stages)
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = """
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, EVEN_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    if EVEN_N:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        tl.store(out_ptr + offsets, x + y)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.device.type != 'cuda' or y.device.type != 'cuda':
        return x + y
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    assert x.shape == y.shape, "Inputs must have the same shape"
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    EVEN_N = (n % BLOCK_SIZE) == 0
    num_warps = 8
    num_stages = 2
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, EVEN_N=EVEN_N, num_warps=num_warps, num_stages=num_stages)
    return out
"""
        return {"code": code}