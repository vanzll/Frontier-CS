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
        # Provide a range of block sizes. For memory-bound operations like vector addition,
        # larger block sizes often yield better performance by maximizing memory throughput.
        # We test powers of 2, which are generally efficient.
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 65536}, num_warps=16),
    ],
    # The key for autotuning is the number of elements. Since this is fixed
    # for this problem, the autotuner will run once and cache the best result.
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
    Triton Kernel for element-wise Vector Addition.
    \"\"\"
    # Each program instance operates on a block of BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)

    # Calculate the offsets for the current block.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to safely handle the last block if n_elements is not
    # a multiple of BLOCK_SIZE. For this problem (N=2^20), the mask
    # will be all-true for our power-of-2 BLOCK_SIZE choices,
    # and the compiler will optimize it away.
    mask = offsets < n_elements

    # Load data from global memory.
    # The load is vectorized and masked to ensure memory safety.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform the element-wise addition.
    output = x + y

    # Store the result back to global memory.
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
    # Allocate the output tensor.
    output = torch.empty_like(x)
    n_elements = output.numel()

    # The launch grid is 1D, with a size equal to the number of blocks needed.
    # triton.cdiv ensures that we have enough blocks to cover all elements.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the kernel. The autotuner selects the best configuration.
    add_kernel[grid](x, y, output, n_elements)

    return output
"""
        return {"code": code}