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

@triton.autotune(
    configs=[
        # Basic configurations with varying block sizes
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        # Test more warps for larger block sizes
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
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
    Triton JIT kernel for element-wise vector addition.
    Each program instance computes a block of the output vector.
    \"\"\"
    # Get the program ID for this instance.
    pid = tl.program_id(axis=0)

    # Calculate the offsets for the block of data this program will process.
    # tl.arange(0, BLOCK_SIZE) creates a range of integers [0, 1, ..., BLOCK_SIZE-1].
    # This is a vector operation that happens in parallel.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create a mask to prevent out-of-bounds memory accesses.
    # This is crucial for the last block if n_elements is not a multiple of BLOCK_SIZE.
    mask = offsets < n_elements

    # Load the data from global memory.
    # The mask ensures that we only load valid data.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform the element-wise addition.
    output = x + y

    # Store the result back to global memory.
    # The mask ensures that we only write to valid memory locations.
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (1048576,)
        y: Input tensor of shape (1048576,)
    
    Returns:
        Output tensor of shape (1048576,) with x + y
    \"\"\"
    # Get the total number of elements in the input tensor.
    n_elements = x.numel()

    # Allocate the output tensor on the same device as the input.
    output = torch.empty_like(x)

    # Define the launch grid. The grid is 1D, and its size is the number of blocks.
    # triton.cdiv(a, b) computes ceil(a / b).
    # The autotuner will try different values for `meta['BLOCK_SIZE']`.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the Triton kernel.
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
    )

    return output
"""
        return {"code": code}