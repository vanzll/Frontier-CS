import torch
import triton
import triton.language as tl
import math

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * VEC_SIZE
    
    # Create vectorized offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE + tl.arange(0, VEC_SIZE)[None, :]
    offsets = tl.view(offsets, BLOCK_SIZE * VEC_SIZE)
    mask = offsets < n_elements
    
    # Vectorized loads
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute addition
    output = x + y
    
    # Vectorized store
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (268435456,)
        y: Input tensor of shape (268435456,)
    
    Returns:
        Output tensor of shape (268435456,) with x + y
    """
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Optimized configuration for L4 GPU
    # Use 1024 threads per block, vector size of 8 for maximum memory throughput
    BLOCK_SIZE = 1024
    VEC_SIZE = 8
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['VEC_SIZE']),)
    
    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC_SIZE=VEC_SIZE,
        num_warps=8,
        num_stages=4
    )
    
    return output

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": """
import torch
import triton
import triton.language as tl
import math

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * VEC_SIZE
    
    # Create vectorized offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE + tl.arange(0, VEC_SIZE)[None, :]
    offsets = tl.view(offsets, BLOCK_SIZE * VEC_SIZE)
    mask = offsets < n_elements
    
    # Vectorized loads
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute addition
    output = x + y
    
    # Vectorized store
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Optimized configuration for L4 GPU
    BLOCK_SIZE = 1024
    VEC_SIZE = 8
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['VEC_SIZE']),)
    
    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC_SIZE=VEC_SIZE,
        num_warps=8,
        num_stages=4
    )
    
    return output
"""}