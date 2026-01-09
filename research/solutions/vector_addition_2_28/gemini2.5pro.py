import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict with a string of Python code for the Triton kernel.
        """
        code = """
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 65536}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 131072}, num_warps=16),
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
    '''
    Triton kernel for element-wise vector addition.
    '''
    # Each program instance computes a block of BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    
    # Calculate the offsets for the current block.
    # tl.arange creates a compile-time constant range of integers.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds memory accesses.
    # This is essential for the last block, which may be smaller than BLOCK_SIZE.
    mask = offsets < n_elements
    
    # Load BLOCK_SIZE elements from x and y tensors.
    # The mask ensures that we only load valid data.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the element-wise addition.
    output = x + y
    
    # Store the result back to the output tensor.
    # The mask ensures that we only write to valid memory locations.
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (268435456,)
        y: Input tensor of shape (268435456,)
    
    Returns:
        Output tensor of shape (268435456,) with x + y
    \"\"\"
    # Allocate the output tensor.
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # The launch grid specifies the number of parallel program instances.
    # Here, we use a 1D grid where each instance processes one block of data.
    # The grid size is calculated by dividing the total number of elements by
    # the block size and rounding up (ceiling division).
    # The `meta` dictionary contains the autotuner's selected configuration.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel.
    # The autotuner will benchmark the different configurations specified in `configs`
    # and automatically select the best-performing one.
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
    )
    
    return output
"""
        return {"code": code}