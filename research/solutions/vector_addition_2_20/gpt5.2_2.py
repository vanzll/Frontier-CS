import os
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, x + y)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("Inputs must be CUDA tensors.")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D tensors.")
    if x.numel() != 1048576 or y.numel() != 1048576:
        raise ValueError("Input tensors must have exactly 1048576 elements.")
    if x.dtype != y.dtype:
        raise ValueError("Input tensors must have the same dtype.")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("Input tensors must be contiguous.")

    out = torch.empty_like(x)

    # Tuned for medium 1D vectors on modern NVIDIA GPUs (e.g., L4)
    BLOCK = 4096
    grid = (x.numel() // BLOCK,)

    _add_kernel[grid](
        x, y, out,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=2,
    )
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}