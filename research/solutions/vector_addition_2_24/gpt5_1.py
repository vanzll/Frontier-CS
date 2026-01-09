import os
import textwrap
import torch
import triton
import triton.language as tl


@triton.jit
def _vec_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, grid_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    while start < n_elements:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
        start += grid_size * BLOCK_SIZE


def _launch_vec_add(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
    n_elements = x.numel()
    assert y.numel() == n_elements
    assert x.is_cuda and y.is_cuda and out.is_cuda
    assert x.dtype == y.dtype == out.dtype
    assert x.is_contiguous() and y.is_contiguous() and out.is_contiguous()

    device_index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    sm_count = getattr(props, "multi_processor_count", 80)

    BLOCK_SIZE = 4096
    # limit programs to a multiple of SMs to reduce overhead; persistent loop covers all elements
    grid_target = sm_count * 8
    grid_size = min(triton.cdiv(n_elements, BLOCK_SIZE), grid_target)

    _vec_add_kernel[(grid_size,)](
        x, y, out, n_elements, grid_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )
    return out


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("Inputs must be torch.Tensor")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.device.type != "cuda" or y.device.type != "cuda":
        # Fallback to PyTorch on CPU or other devices
        return x + y
    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype")
    out = torch.empty_like(x)
    return _launch_vec_add(x, y, out)


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent(
            """
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def _vec_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, grid_size, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(axis=0)
                start = pid * BLOCK_SIZE
                while start < n_elements:
                    offsets = start + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    x = tl.load(x_ptr + offsets, mask=mask)
                    y = tl.load(y_ptr + offsets, mask=mask)
                    tl.store(out_ptr + offsets, x + y, mask=mask)
                    start += grid_size * BLOCK_SIZE

            def _launch_vec_add(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
                n_elements = x.numel()
                assert y.numel() == n_elements
                assert x.is_cuda and y.is_cuda and out.is_cuda
                assert x.dtype == y.dtype == out.dtype
                assert x.is_contiguous() and y.is_contiguous() and out.is_contiguous()

                device_index = x.device.index if x.device.index is not None else torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_index)
                sm_count = getattr(props, "multi_processor_count", 80)

                BLOCK_SIZE = 4096
                grid_target = sm_count * 8
                grid_size = min(triton.cdiv(n_elements, BLOCK_SIZE), grid_target)

                _vec_add_kernel[(grid_size,)](
                    x, y, out, n_elements, grid_size,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=8,
                    num_stages=1,
                )
                return out

            def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
                    raise TypeError("Inputs must be torch.Tensor")
                if x.shape != y.shape:
                    raise ValueError("x and y must have the same shape")
                if x.device.type != "cuda" or y.device.type != "cuda":
                    return x + y
                if x.dtype != y.dtype:
                    raise ValueError("x and y must have the same dtype")
                out = torch.empty_like(x)
                return _launch_vec_add(x, y, out)
            """
        )
        return {"code": code}