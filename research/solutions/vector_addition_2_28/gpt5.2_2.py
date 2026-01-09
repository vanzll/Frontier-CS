import os
import torch
import triton
import triton.language as tl

N_ELEMENTS = 1 << 28
_DEFAULT_BLOCK = 8192
_DEFAULT_NUM_WARPS = 8
_DEFAULT_NUM_STAGES = 4


@triton.jit
def _add_kernel(x_ptr, y_ptr, z_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK

    tl.multiple_of(x_ptr + base, 16)
    tl.multiple_of(y_ptr + base, 16)
    tl.multiple_of(z_ptr + base, 16)

    offs = base + tl.arange(0, BLOCK)
    tl.max_contiguous(offs, 256)

    x = tl.load(x_ptr + offs, cache_modifier=".cg")
    y = tl.load(y_ptr + offs, cache_modifier=".cg")
    tl.store(z_ptr + offs, x + y, cache_modifier=".wb")


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("x and y must be torch.Tensors")
    if x.device.type != "cuda" or y.device.type != "cuda":
        return x + y
    if x.dtype != y.dtype:
        return x + y
    if x.numel() != y.numel():
        raise ValueError("x and y must have the same number of elements")
    if x.numel() != N_ELEMENTS:
        raise ValueError(f"Expected vectors of length {N_ELEMENTS}, got {x.numel()}")
    if not x.is_contiguous() or not y.is_contiguous():
        x = x.contiguous()
        y = y.contiguous()

    out = torch.empty_like(x)
    block = _DEFAULT_BLOCK
    grid = (N_ELEMENTS // block,)

    _add_kernel[grid](
        x,
        y,
        out,
        BLOCK=block,
        num_warps=_DEFAULT_NUM_WARPS,
        num_stages=_DEFAULT_NUM_STAGES,
    )
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}