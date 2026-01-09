import torch
import triton
import triton.language as tl
import os

@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_VECTORIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    if USE_VECTORIZE:
        vector_size = 4
        offsets = block_start + tl.arange(0, BLOCK_SIZE, vector_size)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        result = x + y
        tl.store(output_ptr + offsets, result, mask=mask)
    else:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        result = x + y
        tl.store(output_ptr + offsets, result, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=1024,
        USE_VECTORIZE=True,
    )
    return output

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_VECTORIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    if USE_VECTORIZE:
        vector_size = 4
        offsets = block_start + tl.arange(0, BLOCK_SIZE, vector_size)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        result = x + y
        tl.store(output_ptr + offsets, result, mask=mask)
    else:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        result = x + y
        tl.store(output_ptr + offsets, result, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Heuristic for block size based on vector size
    if n_elements >= 2**28:  # Very large vectors
        block_size = 2048
        use_vectorize = True
    elif n_elements >= 2**24:  # Large vectors
        block_size = 1024
        use_vectorize = True
    else:  # Smaller vectors
        block_size = 512
        use_vectorize = False
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=block_size,
        USE_VECTORIZE=use_vectorize,
    )
    return output
'''
        return {"code": code}