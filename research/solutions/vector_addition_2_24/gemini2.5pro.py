class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict with either:
        - {"code": "python_code_string"}
        - {"program_path": "path/to/kernel.py"}
        """
        add_vector_kernel_code = """
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
    # Each program instance computes a block of the output.
    # The pid is the unique identifier for each instance.
    pid = tl.program_id(axis=0)

    # Calculate the offsets for the current block.
    # tl.arange() creates a vector like [0, 1, 2, ..., BLOCK_SIZE-1].
    # This is a block-level operation.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load a block of data from x and y.
    # The problem guarantees that n_elements is a power of 2, and we choose
    # BLOCK_SIZE to be a power of 2, so n_elements is perfectly divisible by BLOCK_SIZE.
    # This means we don't need boundary checks (masking), which can improve performance.
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)

    # Compute the element-wise sum. This is done in a vectorized manner.
    output = x + y

    # Store the result block back to the output tensor.
    tl.store(output_ptr + offsets, output)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (16777216,)
        y: Input tensor of shape (16777216,)
    
    Returns:
        Output tensor of shape (16777216,) with x + y
    \"\"\"
    # Allocate the output tensor.
    output = torch.empty_like(x)
    
    # Sanity check for contiguity, although guaranteed by the problem.
    assert x.is_contiguous() and y.is_contiguous() and output.is_contiguous()
    
    n_elements = output.numel()

    # The grid defines how many instances of the kernel we want to launch.
    # Each instance (or program) will process a block of data.
    # We use triton.cdiv to ensure we have enough blocks to cover all elements.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # BLOCK_SIZE is a key tuning parameter. For memory-bound operations on large vectors,
    # a larger block size is generally better as it increases the amount of work
    # per launch, improves memory access patterns, and amortizes overhead.
    # We choose a large power of 2 that divides n_elements (2^24).
    # 131072 = 2^17. This creates 2^24 / 2^17 = 128 programs, which is sufficient
    # to saturate a modern GPU like the L4.
    BLOCK_SIZE = 131072

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
        return {"code": add_vector_kernel_code}