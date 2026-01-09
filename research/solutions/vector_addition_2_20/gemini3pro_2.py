import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {
            "code": """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, 
    y_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs. 
    # Since input is guaranteed contiguous and vectors are medium sized,
    # large block sizes ensure efficient memory coalescing.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # 1024 is a standard block size for element-wise operations on modern GPUs
    # to maximize occupancy and memory bandwidth.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](
        x, y, output, 
        n_elements, 
        BLOCK_SIZE=1024, 
        num_warps=4
    )
    
    return output
"""
        }