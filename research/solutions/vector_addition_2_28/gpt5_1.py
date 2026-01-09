import textwrap

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "Input shapes must match"
    assert x.dtype == y.dtype, "Input dtypes must match"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    n = x.numel()
    out = torch.empty_like(x)
    if n == 0:
        return out

    dtype = x.dtype
    # Heuristic: larger blocks for FP16/BF16 to better utilize bandwidth
    if dtype in (torch.float16, torch.bfloat16):
        BLOCK_SIZE = 8192
        num_warps = 8
    else:
        BLOCK_SIZE = 4096
        num_warps = 8

    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=1)
    return out
'''
        return {"code": textwrap.dedent(code)}