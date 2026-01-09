import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict with either:
        - {"code": "python_code_string"}
        - {"program_path": "path/to/kernel.py"}
        """
        code = """
import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    \"\"\"
    Triton kernel for element-wise vector addition.
    \"\"\"
    # Each program instance computes a block of BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)

    # Compute the offsets for the current block.
    # tl.arange creates a compile-time constant range.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load a block of data from the input tensors x and y.
    # Since n_elements is guaranteed to be a multiple of BLOCK_SIZE, no mask is needed
    # for memory operations, which can improve performance.
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    
    # Perform the element-wise addition.
    output = x + y
    
    # Store the resulting block back to the output tensor z.
    tl.store(z_ptr + offsets, output)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (1048576,)
        y: Input tensor of shape (1048576,)
    
    Returns:
        Output tensor of shape (1048576,) with x + y
    \"\"\"
    # Allocate the output tensor on the same device as the inputs.
    z = torch.empty_like(x)
    
    n_elements = x.numel()
    
    # --- Performance Tuning ---
    # For this memory-bound kernel, performance is dictated by memory bandwidth.
    # The most critical tuning parameter is BLOCK_SIZE. A larger BLOCK_SIZE
    # increases the size of memory operations, which helps saturate the memory bus.
    #
    # The vector size is 2^20. We choose a very large power of two that perfectly
    # divides the vector size. A value of 131072 (2^17) is chosen. This results
    # in a grid of 1,048,576 / 131,072 = 8 blocks. While the number of blocks is
    # small, the very large amount of work per block can effectively hide memory
    # latency and achieve maximum throughput on modern GPUs.
    BLOCK_SIZE = 131072
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the Triton kernel.
    _add_kernel[grid](
        x,
        y,
        z,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return z
"""
        return {"code": code}