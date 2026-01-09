import torch
import triton
import triton.language as tl


KERNEL_SOURCE = '''import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.

    Args:
        x: Input tensor
        y: Input tensor

    Returns:
        Output tensor with x + y
    """
    assert x.is_cuda and y.is_cuda, "Input tensors must be CUDA tensors."
    assert x.shape == y.shape, "Input tensors must have the same shape."

    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 2048

    grid = lambda META: ((n_elements + META["BLOCK_SIZE"] - 1) // META["BLOCK_SIZE"],)

    _add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )

    return out
'''


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.

    Args:
        x: Input tensor of shape (268435456,)
        y: Input tensor of shape (268435456,)

    Returns:
        Output tensor of shape (268435456,) with x + y
    """
    assert x.is_cuda and y.is_cuda, "Input tensors must be CUDA tensors."
    assert x.shape == y.shape, "Input tensors must have the same shape."

    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 2048

    grid = lambda META: ((n_elements + META["BLOCK_SIZE"] - 1) // META["BLOCK_SIZE"],)

    _add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )

    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": KERNEL_SOURCE}