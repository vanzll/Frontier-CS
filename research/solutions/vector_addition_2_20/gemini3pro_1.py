import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {
            "code": r"""
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
    """
    Triton kernel for element-wise vector addition.
    Processes BLOCK_SIZE elements per program instance.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data from x and y
    # Triton automatically vectorizes these loads for contiguous memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform addition
    output = x + y

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors using Triton.
    
    Args:
        x: Input tensor of shape (1048576,)
        y: Input tensor of shape (1048576,)
    
    Returns:
        Output tensor of shape (1048576,) with x + y
    """
    # Pre-allocate output tensor
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Block size of 1024 is generally optimal for element-wise operations
    # on modern GPUs to maximize occupancy and memory throughput.
    BLOCK_SIZE = 1024
    
    # Calculate grid size (number of blocks)
    # triton.cdiv calculates ceil(n_elements / BLOCK_SIZE)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    # num_warps=4 is standard for low-arithmetic-intensity kernels
    add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    
    return output
"""
        }