import torch
import triton
import triton.language as tl

KERNEL_SOURCE = '''import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr: tl.pointer_type(dtype=tl.float32),
    y_ptr: tl.pointer_type(dtype=tl.float32),
    out_ptr: tl.pointer_type(dtype=tl.float32),
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two vectors.

    Args:
        x: Input tensor of shape (16777216,)
        y: Input tensor of shape (16777216,)

    Returns:
        Output tensor of shape (16777216,) with x + y
    """
    if x.device != y.device:
        raise ValueError("Input tensors must be on the same device")
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensor")
    if not x.is_cuda:
        # Fallback to PyTorch for non-CUDA tensors
        return x + y
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise TypeError("This kernel currently supports only float32 tensors")

    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _add_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
'''

exec(KERNEL_SOURCE, globals())


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": KERNEL_SOURCE}