import torch
import triton
import triton.language as tl
import math

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    assert x.shape == y.shape, "Input shapes must match"
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    if n_elements == 0:
        return out
    
    # Use optimal block size for memory-bound kernel
    # 128-1024 elements per block is typical for memory-bound operations
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": self._get_code()}
    
    def _get_code(self) -> str:
        import inspect
        return inspect.getsource(add)