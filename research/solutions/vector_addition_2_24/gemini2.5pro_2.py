import triton
import triton.language as tl
import torch

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict with the Python code for the Triton kernel.
        """
        
        kernel_code = """
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Basic configurations
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        # Larger block sizes to maximize memory throughput on large vectors
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 65536}, num_warps=16),
    ],
    key=['n_elements'],
)
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
    This kernel is optimized for large, contiguous vectors where n_elements
    is a power of 2, allowing for unmasked memory operations.
    \"\"\"
    # Each program instance computes a block of BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    
    # Calculate the offsets for the current block.
    # Since n_elements (2^24) is a multiple of any power-of-2 BLOCK_SIZE,
    # we can use unmasked loads and stores for maximum efficiency.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load the input vectors from global memory.
    # These are vectorized loads, which is key for memory bandwidth.
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    
    # Perform the element-wise addition.
    output = x + y
    
    # Store the result back to global memory.
    # This is a vectorized store.
    tl.store(output_ptr + offsets, output)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Element-wise addition of two vectors using a Triton kernel.
    
    Args:
        x: Input tensor of shape (16777216,)
        y: Input tensor of shape (16777216,)
    
    Returns:
        Output tensor of shape (16777216,) with x + y
    \"\"\"
    # The problem specifies a fixed size of 2^24.
    n_elements = 16777216
    assert x.numel() == n_elements and y.numel() == n_elements, "Input tensors must have 2^24 elements"
    assert x.is_contiguous() and y.is_contiguous(), "Input tensors must be contiguous"

    # Allocate the output tensor on the same device as the inputs.
    output = torch.empty_like(x)
    
    # The grid is 1D. Each program instance in the grid handles one block of data.
    # The triton.cdiv utility function is used to compute the grid size,
    # which is ceil(n_elements / BLOCK_SIZE).
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the kernel. The autotuner will automatically select the best
    # configuration from the provided list based on empirical timing.
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
    )
    
    return output
"""
        return {"code": kernel_code}