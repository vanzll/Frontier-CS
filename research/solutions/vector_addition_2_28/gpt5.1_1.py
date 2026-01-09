import torch
import triton
import triton.language as tl
import textwrap


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda" or y.device.type != "cuda":
        raise ValueError("Inputs must be CUDA tensors")
    if x.device != y.device:
        raise ValueError("Input tensors must be on the same device")
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")

    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    x_flat = x.view(-1)
    y_flat = y.view(-1)
    n_elements = x_flat.numel()

    out_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _add_kernel[grid](x_flat, y_flat, out_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

    return out_flat.view_as(x)


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent(
            """
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                out = x + y
                tl.store(out_ptr + offsets, out, mask=mask)


            def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if x.device.type != "cuda" or y.device.type != "cuda":
                    raise ValueError("Inputs must be CUDA tensors")
                if x.device != y.device:
                    raise ValueError("Input tensors must be on the same device")
                if x.shape != y.shape:
                    raise ValueError("Input tensors must have the same shape")

                if not x.is_contiguous():
                    x = x.contiguous()
                if not y.is_contiguous():
                    y = y.contiguous()

                x_flat = x.view(-1)
                y_flat = y.view(-1)
                n_elements = x_flat.numel()

                out_flat = torch.empty_like(x_flat)

                BLOCK_SIZE = 4096
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                _add_kernel[grid](x_flat, y_flat, out_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

                return out_flat.view_as(x)
            """
        )
        return {"code": code}