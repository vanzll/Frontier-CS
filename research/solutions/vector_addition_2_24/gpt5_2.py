import torch
import triton
import triton.language as tl


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel_masked(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.multiple_of(BLOCK, 256)
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0)
    y = tl.load(y_ptr + offsets, mask=offsets < n_elements, other=0)
    tl.store(out_ptr + offsets, x + y, mask=offsets < n_elements)

@triton.jit
def _add_kernel_full(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.multiple_of(BLOCK, 256)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.

    Args:
        x: Input tensor of shape (16777216,)
        y: Input tensor of shape (16777216,)

    Returns:
        Output tensor of shape (16777216,) with x + y
    """
    assert x.shape == y.shape, "Input sizes must match"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    assert x.dtype == y.dtype, "Input dtypes must match"
    n_elements = x.numel()
    if x.device.type != y.device.type:
        raise ValueError("Input devices must match")
    if x.device.type != "cuda":
        return x + y

    out = torch.empty_like(x)

    # Tuned for large, bandwidth-bound vectors on NVIDIA L4
    BLOCK = 8192  # 2^13, divides 2^24 exactly -> avoids masking for the target size
    num_warps = 8
    num_stages = 4

    if n_elements % BLOCK == 0:
        grid = (n_elements // BLOCK,)
        _add_kernel_full[grid](x, y, out, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
    else:
        grid = (triton.cdiv(n_elements, BLOCK),)
        _add_kernel_masked[grid](x, y, out, n_elements, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
    return out
'''
        return {"code": code}