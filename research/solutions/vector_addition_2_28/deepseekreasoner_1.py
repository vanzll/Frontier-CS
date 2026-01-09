import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "Tensors must have same shape"
    assert x.is_cuda and y.is_cuda, "Tensors must be on GPU"
    assert x.is_contiguous() and y.is_contiguous(), "Tensors must be contiguous"
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    # Optimize block size for large vectors
    # For 2^28 elements, use 2048 threads per block to maximize occupancy
    # and reduce grid size overhead
    BLOCK_SIZE = 2048
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,  # 2048 threads / 32 = 64 warps per block
    )
    
    return out

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = """import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "Tensors must have same shape"
    assert x.is_cuda and y.is_cuda, "Tensors must be on GPU"
    assert x.is_contiguous() and y.is_contiguous(), "Tensors must be contiguous"
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    # Optimize for large vectors (2^28 elements)
    # Use large block size for better memory coalescing
    BLOCK_SIZE = 2048
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )
    
    return out
"""
        return {"code": code}