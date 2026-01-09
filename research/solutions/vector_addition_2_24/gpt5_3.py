import os
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.load(y_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    assert x.shape == y.shape, "Inputs must have the same shape"
    N = x.numel()
    out = torch.empty_like(x)

    if N >= (1 << 22):
        BLOCK = 8192
        num_warps = 8
    elif N >= (1 << 18):
        BLOCK = 4096
        num_warps = 4
    else:
        BLOCK = 1024
        num_warps = 2

    grid = (triton.cdiv(N, BLOCK),)
    _add_kernel[grid](x, y, out, N, BLOCK=BLOCK, num_warps=num_warps, num_stages=2)
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}