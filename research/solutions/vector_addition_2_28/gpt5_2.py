import os
import torch
import triton
import triton.language as tl


@triton.jit
def _vadd_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.device != y.device:
        raise ValueError("Input tensors must be on the same device")
    if x.dtype != y.dtype:
        raise ValueError("Input tensors must have the same dtype")
    if x.numel() != y.numel():
        raise ValueError("Input tensors must have the same number of elements")
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("Input tensors must be 1D vectors")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("Input tensors must be contiguous")

    n = x.numel()

    if not x.is_cuda:
        return x + y

    z = torch.empty_like(x)

    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    _vadd_kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)
    return z


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}