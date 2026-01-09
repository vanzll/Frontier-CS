import os
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(out_ptr + offsets, z, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("x and y must be torch.Tensors")
    if x.numel() != y.numel():
        raise ValueError("x and y must have the same number of elements")
    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")

    n = x.numel()
    if n == 0:
        return torch.empty_like(x)

    # Fallback to PyTorch on CPU
    if not x.is_cuda:
        return x + y

    x_c = x.contiguous()
    y_c = y.contiguous()
    out = torch.empty_like(x_c)

    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    _add_kernel[grid](x_c, y_c, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=1)
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": __file__}