import os
import textwrap
import torch
import triton
import triton.language as tl

N_ELEMENTS = 16777216


@triton.jit
def _add_kernel_unmasked(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    tl.multiple_of(block_start, 256)
    offs = block_start + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, x + y)


@triton.jit
def _add_kernel_masked(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.load(y_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda":
        return x + y
    out = torch.empty_like(x)
    n = x.numel()

    BLOCK = 4096
    num_warps = 8
    num_stages = 2

    if n == N_ELEMENTS and (n % BLOCK) == 0:
        grid = (n // BLOCK,)
        _add_kernel_unmasked[grid](x, y, out, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
    else:
        grid = (triton.cdiv(n, BLOCK),)
        _add_kernel_masked[grid](x, y, out, n, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
    return out


_KERNEL_CODE = textwrap.dedent(
    """
    import torch
    import triton
    import triton.language as tl

    N_ELEMENTS = 16777216

    @triton.jit
    def _add_kernel_unmasked(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK
        tl.multiple_of(block_start, 256)
        offs = block_start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(out_ptr + offs, x + y)

    @triton.jit
    def _add_kernel_masked(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        y = tl.load(y_ptr + offs, mask=mask, other=0)
        tl.store(out_ptr + offs, x + y, mask=mask)

    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cuda":
            return x + y
        out = torch.empty_like(x)
        n = x.numel()

        BLOCK = 4096
        num_warps = 8
        num_stages = 2

        if n == N_ELEMENTS and (n % BLOCK) == 0:
            grid = (n // BLOCK,)
            _add_kernel_unmasked[grid](x, y, out, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
        else:
            grid = (triton.cdiv(n, BLOCK),)
            _add_kernel_masked[grid](x, y, out, n, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
        return out
    """
).lstrip()


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": _KERNEL_CODE}