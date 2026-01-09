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
        kernel_code = """
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
    \"\"\"
    Triton kernel for element-wise vector addition.
    \"\"\"
    # Each program instance computes a block of BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    
    # Calculate the offsets for the elements that this program instance will process.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds memory accesses. This is essential
    # for handling vector sizes that are not a multiple of the block size.
    # While not strictly necessary for N=2^24 and a power-of-2 BLOCK_SIZE,
    # it is a robust programming practice.
    mask = offsets < n_elements
    
    # Load the input vectors `x` and `y` from global memory into registers.
    # The mask ensures that we only load data within the vector's bounds.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the element-wise addition.
    output = x + y
    
    # Store the result from registers back to global memory.
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (16777216,)
        y: Input tensor of shape (16777216,)
    
    Returns:
        Output tensor of shape (16777216,) with x + y
    \"\"\"
    # Allocate the output tensor on the same device as the input.
    output = torch.empty_like(x)
    
    # Get the total number of elements in the vector.
    n_elements = x.numel()
    
    # For memory-bound operations on large vectors, using a large block size is optimal.
    # It maximizes memory throughput by fetching large, contiguous chunks of data,
    # which leads to better utilization of the memory bus and caches. It also
    # reduces the relative overhead of kernel launches.
    #
    # The value 131072 (2^17) is chosen. For an input size of 2^24, this creates
    # 2^24 / 2^17 = 128 blocks. This is a sufficient number of blocks to keep all
    # Streaming Multiprocessors (SMs) on a modern GPU like the NVIDIA L4 busy,
    # ensuring full hardware utilization.
    BLOCK_SIZE = 131072
    
    # The grid size is the number of blocks we need to launch. It's a 1D grid.
    # triton.cdiv ensures that we have enough blocks to cover all elements.
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the Triton kernel.
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
"""
        return {"code": kernel_code}