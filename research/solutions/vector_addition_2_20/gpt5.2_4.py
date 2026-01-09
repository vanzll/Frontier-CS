import os
import torch
import triton
import triton.language as tl

_N_ELEMENTS = 1 << 20
_BLOCK = 4096
_NUM_WARPS = 8
_NUM_STAGES = 2


@triton.jit
def _add_kernel(X_ptr, Y_ptr, Z_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)

    tl.multiple_of(start, BLOCK)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 128)

    x = tl.load(X_ptr + offs, cache_modifier=".cg")
    y = tl.load(Y_ptr + offs, cache_modifier=".cg")
    z = x + y
    tl.store(Z_ptr + offs, z, cache_modifier=".wb")


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("x and y must be torch.Tensor")
    if not (x.is_cuda and y.is_cuda):
        raise ValueError("x and y must be CUDA tensors")
    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype")
    if x.numel() != _N_ELEMENTS or y.numel() != _N_ELEMENTS:
        raise ValueError(f"x and y must have exactly {_N_ELEMENTS} elements")
    if x.dim() != 1 or y.dim() != 1:
        x = x.reshape(-1)
        y = y.reshape(-1)
        if x.numel() != _N_ELEMENTS or y.numel() != _N_ELEMENTS:
            raise ValueError(f"x and y must be vectors with exactly {_N_ELEMENTS} elements")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")

    out = torch.empty_like(x)
    grid = (_N_ELEMENTS // _BLOCK,)
    _add_kernel[grid](x, y, out, BLOCK=_BLOCK, num_warps=_NUM_WARPS, num_stages=_NUM_STAGES)
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}