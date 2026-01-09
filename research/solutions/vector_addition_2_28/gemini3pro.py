import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {
            "code": """import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions (though inputs are large, always good practice)
    mask = offsets < n_elements
    
    # Load data
    # Vectorized load of BLOCK_SIZE elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Pre-allocate output
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Define grid
    # 1D grid is sufficient for vector addition
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    # BLOCK_SIZE=1024 is typically optimal for element-wise operations on modern GPUs
    # num_warps=4 (default) or 8 provides enough parallelism to hide memory latency
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output
"""
        }