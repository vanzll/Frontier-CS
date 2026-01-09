import os
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    tl.multiple_of(base, 32)
    tl.max_contiguous(offs, 128)
    x = tl.load(x_ptr + offs, cache_modifier=".cg", eviction_policy="evict_last")
    y = tl.load(y_ptr + offs, cache_modifier=".cg", eviction_policy="evict_last")
    tl.store(out_ptr + offs, x + y, cache_modifier=".wb")


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not (x.is_cuda and y.is_cuda):
        raise ValueError("Inputs must be CUDA tensors")
    if x.dtype != y.dtype:
        raise ValueError("Input dtypes must match")
    if not (x.is_contiguous() and y.is_contiguous()):
        raise ValueError("Inputs must be contiguous")
    x1 = x.reshape(-1)
    y1 = y.reshape(-1)
    n = x1.numel()
    if n != (1 << 28):
        raise ValueError(f"Expected exactly {1<<28} elements, got {n}")
    if y1.numel() != n:
        raise ValueError("Input sizes must match")

    out = torch.empty_like(x1)
    BLOCK = 1024
    grid = (n // BLOCK,)
    _add_kernel[grid](x1, y1, out, BLOCK=BLOCK, num_warps=8, num_stages=4)
    return out.view_as(x)


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}