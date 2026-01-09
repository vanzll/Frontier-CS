import torch
import triton
import triton.language as tl
import os

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_VECTORIZED_LOAD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if USE_VECTORIZED_LOAD:
        BLOCK_SIZE_VEC = BLOCK_SIZE * 4
        block_start = pid * BLOCK_SIZE_VEC
        
        offsets = block_start + tl.arange(0, BLOCK_SIZE_VEC)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    else:
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    def get_grid_size(block_size):
        return triton.cdiv(n_elements, block_size)
    
    BLOCK_SIZE = 1024
    USE_VECTORIZED_LOAD = True
    block_size = BLOCK_SIZE * 4 if USE_VECTORIZED_LOAD else BLOCK_SIZE
    grid_size = get_grid_size(block_size)
    
    add_kernel[grid_size](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_VECTORIZED_LOAD=USE_VECTORIZED_LOAD,
    )
    
    return output

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = """import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_VECTORIZED_LOAD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if USE_VECTORIZED_LOAD:
        BLOCK_SIZE_VEC = BLOCK_SIZE * 4
        block_start = pid * BLOCK_SIZE_VEC
        
        offsets = block_start + tl.arange(0, BLOCK_SIZE_VEC)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    else:
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    def get_grid_size(block_size):
        return triton.cdiv(n_elements, block_size)
    
    BLOCK_SIZE = 1024
    USE_VECTORIZED_LOAD = True
    block_size = BLOCK_SIZE * 4 if USE_VECTORIZED_LOAD else BLOCK_SIZE
    grid_size = get_grid_size(block_size)
    
    add_kernel[grid_size](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_VECTORIZED_LOAD=USE_VECTORIZED_LOAD,
    )
    
    return output
"""
        return {"code": code}