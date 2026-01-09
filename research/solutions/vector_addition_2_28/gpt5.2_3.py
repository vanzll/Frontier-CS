import os
import torch
import triton
import triton.language as tl

N_ELEMENTS_EXACT = 1 << 28


@triton.jit
def _add_kernel_nomask(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.multiple_of(x_ptr, 16)
    tl.multiple_of(y_ptr, 16)
    tl.multiple_of(out_ptr, 16)
    tl.max_contiguous(offs, BLOCK)

    x = tl.load(x_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first")
    y = tl.load(y_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first")
    tl.store(out_ptr + offs, x + y)


@triton.jit
def _add_kernel_masked(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.multiple_of(x_ptr, 16)
    tl.multiple_of(y_ptr, 16)
    tl.multiple_of(out_ptr, 16)
    tl.max_contiguous(offs, BLOCK)

    x = tl.load(x_ptr + offs, mask=mask, other=0, cache_modifier=".cg", eviction_policy="evict_first")
    y = tl.load(y_ptr + offs, mask=mask, other=0, cache_modifier=".cg", eviction_policy="evict_first")
    tl.store(out_ptr + offs, x + y, mask=mask)


def _pick_params(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 4096, 8, 4
    if dtype == torch.float32:
        return 2048, 8, 4
    if dtype == torch.float64:
        return 1024, 8, 3
    return 1024, 4, 3


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("x and y must be torch.Tensor")
    if not (x.is_cuda and y.is_cuda):
        raise ValueError("x and y must be CUDA tensors")
    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if x.numel() != y.numel():
        raise ValueError("x and y must have the same number of elements")

    n = x.numel()
    if n != N_ELEMENTS_EXACT:
        x = x.reshape(-1)
        y = y.reshape(-1)
    else:
        x = x.reshape(-1)
        y = y.reshape(-1)

    out = torch.empty_like(x)

    BLOCK, num_warps, num_stages = _pick_params(x.dtype)

    if n == N_ELEMENTS_EXACT and (n % BLOCK == 0):
        grid = (n // BLOCK,)
        _add_kernel_nomask[grid](x, y, out, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
    else:
        grid = (triton.cdiv(n, BLOCK),)
        _add_kernel_masked[grid](x, y, out, n, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)

    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        path = None
        try:
            path = os.path.abspath(__file__)
            with open(path, "r", encoding="utf-8") as f:
                return {"code": f.read()}
        except Exception:
            if path is not None:
                return {"program_path": path}
            raise