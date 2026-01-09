import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(out_ptr + offsets, z, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor
        y: Input tensor (same shape as x)
    
    Returns:
        Output tensor with x + y
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.dim() != 1:
        x = x.view(-1)
        y = y.view(-1)

    # Fallback to PyTorch if not on CUDA
    if x.device.type != "cuda" or y.device.type != "cuda":
        return x + y

    n_elements = x.numel()
    if n_elements == 0:
        return x + y

    out = torch.empty_like(x)

    BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return out


KERNEL_CODE = """import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(out_ptr + offsets, z, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    \"""
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor
        y: Input tensor (same shape as x)
    
    Returns:
        Output tensor with x + y
    \"""
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.dim() != 1:
        x = x.view(-1)
        y = y.view(-1)

    # Fallback to PyTorch if not on CUDA
    if x.device.type != "cuda" or y.device.type != "cuda":
        return x + y

    n_elements = x.numel()
    if n_elements == 0:
        return x + y

    out = torch.empty_like(x)

    BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return out
"""


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": KERNEL_CODE}