import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = r"""
import torch
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
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Block start index
    block_start = pid * BLOCK_SIZE
    
    # Offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds access
    mask = offsets < n_elements
    
    # Load data
    # Triton handles vectorization of loads automatically for contiguous data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Element-wise addition
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Pre-allocate output tensor
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Configuration optimized for bandwidth on large vectors
    # BLOCK_SIZE=1024 is standard for element-wise ops
    # num_warps=8 provides sufficient parallelism to hide memory latency
    BLOCK_SIZE = 1024
    num_warps = 8
    
    # Grid calculation
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output
"""
        return {"code": code}