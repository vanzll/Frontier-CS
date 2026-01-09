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
        kernel_code = """import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 65536}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 131072}, num_warps=32),
    ],
    key=['n_elements'],
)
@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance computes a block of BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    
    # Calculate the offsets for the current block.
    # This gives us a range of indices to process.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds memory accesses.
    # This is important for the last block if n_elements is not a multiple of BLOCK_SIZE.
    mask = offsets < n_elements
    
    # Load the data from input tensors.
    # The mask ensures we only load valid data.
    # The loads are coalesced for maximum memory bandwidth.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the element-wise addition.
    output = x + y
    
    # Store the result back to the output tensor.
    # The mask ensures we only write to valid memory locations.
    tl.store(z_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (268435456,)
        y: Input tensor of shape (268435456,)
    
    Returns:
        Output tensor of shape (268435456,) with x + y
    \"\"\"
    n_elements = x.numel()
    
    # The problem guarantees x and y have the same shape and are contiguous.
    # We create an empty output tensor with the same shape as the inputs.
    z = torch.empty_like(x)
    
    # The grid is 1D, and its size is the number of blocks.
    # We use triton.cdiv to compute the ceiling division, ensuring
    # we have enough blocks to cover all elements.
    # The grid is defined as a lambda to be compatible with autotuning.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the kernel. The autotuner will pick the best BLOCK_SIZE.
    _add_kernel[grid](x, y, z, n_elements)
    
    return z
"""
        return {"code": kernel_code}