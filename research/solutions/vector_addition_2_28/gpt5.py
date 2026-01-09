import os
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.multiple_of(offsets, 16)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(out_ptr + offsets, z, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    if x.dtype != y.dtype:
        raise ValueError("Input tensors must have the same dtype")
    if x.device != y.device:
        raise ValueError("Input tensors must be on the same device")
    if not x.is_contiguous() or not y.is_contiguous():
        x = x.contiguous()
        y = y.contiguous()

    # CPU fallback
    if not x.is_cuda:
        return x + y

    n_elements = x.numel()
    out = torch.empty_like(x)

    # Choose block size and launch parameters optimized for large 1D bandwidth
    BLOCK_SIZE = 8192  # 32 KB of elements per program for fp32 -> 128 KB bytes load/store total
    num_warps = 8
    num_stages = 2

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages)
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}