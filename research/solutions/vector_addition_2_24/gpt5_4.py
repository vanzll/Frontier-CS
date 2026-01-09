import os
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, VEC: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * VEC
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    for i in tl.static_range(VEC):
        idx = offsets + i * BLOCK_SIZE
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0)
        y = tl.load(y_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device"
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.is_contiguous() and y.is_contiguous(), "Input tensors must be contiguous"
    N = x.numel()
    out = torch.empty_like(x)

    # Heuristics for large vectors; tuned for bandwidth
    # Prefer larger per-program work to reduce launch overhead while keeping enough parallelism.
    if N >= (1 << 23):  # ~8M+
        BLOCK_SIZE = 2048
        VEC = 8
        num_warps = 8
        num_stages = 2
    elif N >= (1 << 20):  # ~1M+
        BLOCK_SIZE = 2048
        VEC = 4
        num_warps = 8
        num_stages = 2
    else:
        BLOCK_SIZE = 1024
        VEC = 4
        num_warps = 4
        num_stages = 1

    grid = (triton.cdiv(N, BLOCK_SIZE * VEC),)
    _add_kernel[grid](
        x, y, out, N,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC=VEC,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": os.path.abspath(__file__)}