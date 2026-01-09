import torch
import triton
import triton.language as tl
from typing import Optional
import os

@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def _add_kernel_vec4(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE, 4)
    mask = offsets + 3 < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    
    # Handle remaining elements with scalar operations
    rem_start = block_start + BLOCK_SIZE - (BLOCK_SIZE % 4)
    if rem_start < n_elements and pid == tl.num_programs(0) - 1:
        rem_offsets = rem_start + tl.arange(0, 4)
        rem_mask = rem_offsets < n_elements
        
        if tl.sum(rem_mask, axis=0) > 0:
            x_rem = tl.load(x_ptr + rem_offsets, mask=rem_mask, other=0.0)
            y_rem = tl.load(y_ptr + rem_offsets, mask=rem_mask, other=0.0)
            output_rem = x_rem + y_rem
            tl.store(output_ptr + rem_offsets, output_rem, mask=rem_mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (16777216,)
        y: Input tensor of shape (16777216,)
    
    Returns:
        Output tensor of shape (16777216,) with x + y
    """
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Heuristic for block size selection
    # For large vectors (2^24), use larger blocks for better memory throughput
    # but avoid excessive register usage
    if n_elements >= 2**20:  # 1M elements
        BLOCK_SIZE = 1024
        use_vec4 = True
    else:
        BLOCK_SIZE = 512
        use_vec4 = False
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    if use_vec4:
        _add_kernel_vec4[grid](
            x, y, output, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        _add_kernel[grid](
            x, y, output, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return output

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict with either:
        - {"code": "python_code_string"}
        - {"program_path": "path/to/kernel.py"}
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, "vector_addition_kernel.py")
        
        code = '''import torch
import triton
import triton.language as tl
from typing import Optional
import os

@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def _add_kernel_vec4(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE, 4)
    mask = offsets + 3 < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    
    # Handle remaining elements with scalar operations
    rem_start = block_start + BLOCK_SIZE - (BLOCK_SIZE % 4)
    if rem_start < n_elements and pid == tl.num_programs(0) - 1:
        rem_offsets = rem_start + tl.arange(0, 4)
        rem_mask = rem_offsets < n_elements
        
        if tl.sum(rem_mask, axis=0) > 0:
            x_rem = tl.load(x_ptr + rem_offsets, mask=rem_mask, other=0.0)
            y_rem = tl.load(y_ptr + rem_offsets, mask=rem_mask, other=0.0)
            output_rem = x_rem + y_rem
            tl.store(output_ptr + rem_offsets, output_rem, mask=rem_mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.
    
    Args:
        x: Input tensor of shape (16777216,)
        y: Input tensor of shape (16777216,)
    
    Returns:
        Output tensor of shape (16777216,) with x + y
    """
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Heuristic for block size selection
    # For large vectors (2^24), use larger blocks for better memory throughput
    # but avoid excessive register usage
    if n_elements >= 2**20:  # 1M elements
        BLOCK_SIZE = 1024
        use_vec4 = True
    else:
        BLOCK_SIZE = 512
        use_vec4 = False
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    if use_vec4:
        _add_kernel_vec4[grid](
            x, y, output, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        _add_kernel[grid](
            x, y, output, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return output
'''
        
        # Write the code to a file
        with open(output_path, 'w') as f:
            f.write(code)
        
        return {"program_path": output_path}