import os
import torch
import triton
import triton.language as tl

_N_FIXED = 16777216
_DEFAULT_BLOCK = 1024
_DEFAULT_NUM_WARPS = 8


@triton.jit
def _add_kernel_nomask(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
    tl.multiple_of(x_ptr, 16)
    tl.multiple_of(y_ptr, 16)
    tl.multiple_of(o_ptr, 16)

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, cache_modifier=".cg")
    y = tl.load(y_ptr + offs, cache_modifier=".cg")
    tl.store(o_ptr + offs, x + y)


@triton.jit
def _add_kernel_masked(x_ptr, y_ptr, o_ptr, n_elements, BLOCK: tl.constexpr):
    tl.multiple_of(x_ptr, 16)
    tl.multiple_of(y_ptr, 16)
    tl.multiple_of(o_ptr, 16)

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0, cache_modifier=".cg")
    y = tl.load(y_ptr + offs, mask=mask, other=0, cache_modifier=".cg")
    tl.store(o_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("x and y must be torch.Tensor")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.ndim != 1:
        raise ValueError("x and y must be 1D tensors")
    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype")
    if x.numel() != _N_FIXED:
        # Spec guarantees fixed size, but handle gracefully
        n = x.numel()
    else:
        n = _N_FIXED

    if not x.is_cuda or not y.is_cuda:
        return x + y

    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    out = torch.empty_like(x)

    BLOCK = _DEFAULT_BLOCK
    num_warps = _DEFAULT_NUM_WARPS

    if n % BLOCK == 0:
        grid = (n // BLOCK,)
        _add_kernel_nomask[grid](x, y, out, BLOCK=BLOCK, num_warps=num_warps, num_stages=1)
    else:
        grid = (triton.cdiv(n, BLOCK),)
        _add_kernel_masked[grid](x, y, out, n, BLOCK=BLOCK, num_warps=num_warps, num_stages=1)

    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        try:
            path = __file__
            if path and os.path.exists(path):
                return {"program_path": path}
        except Exception:
            pass
        # Fallback: provide minimal code string
        code = (
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n"
            "_N_FIXED=16777216\n"
            "_DEFAULT_BLOCK=1024\n"
            "_DEFAULT_NUM_WARPS=8\n"
            "@triton.jit\n"
            "def _add_kernel_nomask(x_ptr,y_ptr,o_ptr,BLOCK: tl.constexpr):\n"
            "    tl.multiple_of(x_ptr,16); tl.multiple_of(y_ptr,16); tl.multiple_of(o_ptr,16)\n"
            "    pid=tl.program_id(0)\n"
            "    offs=pid*BLOCK+tl.arange(0,BLOCK)\n"
            "    x=tl.load(x_ptr+offs, cache_modifier='.cg')\n"
            "    y=tl.load(y_ptr+offs, cache_modifier='.cg')\n"
            "    tl.store(o_ptr+offs, x+y)\n"
            "@triton.jit\n"
            "def _add_kernel_masked(x_ptr,y_ptr,o_ptr,n_elements,BLOCK: tl.constexpr):\n"
            "    tl.multiple_of(x_ptr,16); tl.multiple_of(y_ptr,16); tl.multiple_of(o_ptr,16)\n"
            "    pid=tl.program_id(0)\n"
            "    offs=pid*BLOCK+tl.arange(0,BLOCK)\n"
            "    mask=offs<n_elements\n"
            "    x=tl.load(x_ptr+offs, mask=mask, other=0, cache_modifier='.cg')\n"
            "    y=tl.load(y_ptr+offs, mask=mask, other=0, cache_modifier='.cg')\n"
            "    tl.store(o_ptr+offs, x+y, mask=mask)\n"
            "def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:\n"
            "    if x.shape!=y.shape or x.ndim!=1 or x.dtype!=y.dtype: raise ValueError('shape/dtype mismatch')\n"
            "    n = x.numel()\n"
            "    if not x.is_cuda or not y.is_cuda: return x+y\n"
            "    if not x.is_contiguous(): x=x.contiguous()\n"
            "    if not y.is_contiguous(): y=y.contiguous()\n"
            "    out=torch.empty_like(x)\n"
            "    BLOCK=_DEFAULT_BLOCK; num_warps=_DEFAULT_NUM_WARPS\n"
            "    if n % BLOCK == 0:\n"
            "        grid=(n//BLOCK,)\n"
            "        _add_kernel_nomask[grid](x,y,out,BLOCK=BLOCK,num_warps=num_warps,num_stages=1)\n"
            "    else:\n"
            "        grid=(triton.cdiv(n,BLOCK),)\n"
            "        _add_kernel_masked[grid](x,y,out,n,BLOCK=BLOCK,num_warps=num_warps,num_stages=1)\n"
            "    return out\n"
        )
        return {"code": code}