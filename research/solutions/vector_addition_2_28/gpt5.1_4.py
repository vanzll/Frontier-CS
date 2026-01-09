import torch
import triton
import triton.language as tl
import textwrap


@triton.jit
def _vec_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.
    """
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    if x.device != y.device:
        raise ValueError("Input tensors must be on the same device")
    if not x.is_cuda:
        return x + y
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    N = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 4096
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _vec_add_kernel[grid](
        x, y, out, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        kernel_code = textwrap.dedent(
            '''\
import torch
import triton
import triton.language as tl


@triton.jit
def _vec_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.
    """
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    if x.device != y.device:
        raise ValueError("Input tensors must be on the same device")
    if not x.is_cuda:
        return x + y
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    N = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 4096
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _vec_add_kernel[grid](
        x, y, out, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return out
'''
        )
        return {"code": kernel_code}