import torch
import triton
import triton.language as tl

_N_ELEMENTS = 1 << 24


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    tl.multiple_of(block_start, 256)
    offs = block_start + tl.arange(0, BLOCK)
    tl.max_contiguous(offs, 256)
    x = tl.load(x_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first")
    y = tl.load(y_ptr + offs, cache_modifier=".cg", eviction_policy="evict_first")
    tl.store(out_ptr + offs, x + y)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    if x.dtype in (torch.float16, torch.bfloat16):
        block = 4096
        num_warps = 8
    elif x.dtype == torch.float32:
        block = 2048
        num_warps = 4
    elif x.dtype == torch.float64:
        block = 1024
        num_warps = 4
    else:
        block = 2048
        num_warps = 4

    n = x.numel()
    grid = (n // block,)
    _add_kernel[grid](x, y, out, BLOCK=block, num_warps=num_warps, num_stages=2)
    return out


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = (
            "import torch\n"
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "_N_ELEMENTS = 1 << 24\n"
            "\n"
            "@triton.jit\n"
            "def _add_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    block_start = pid * BLOCK\n"
            "    tl.multiple_of(block_start, 256)\n"
            "    offs = block_start + tl.arange(0, BLOCK)\n"
            "    tl.max_contiguous(offs, 256)\n"
            "    x = tl.load(x_ptr + offs, cache_modifier='.cg', eviction_policy='evict_first')\n"
            "    y = tl.load(y_ptr + offs, cache_modifier='.cg', eviction_policy='evict_first')\n"
            "    tl.store(out_ptr + offs, x + y)\n"
            "\n"
            "def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:\n"
            "    out = torch.empty_like(x)\n"
            "    if x.dtype in (torch.float16, torch.bfloat16):\n"
            "        block = 4096\n"
            "        num_warps = 8\n"
            "    elif x.dtype == torch.float32:\n"
            "        block = 2048\n"
            "        num_warps = 4\n"
            "    elif x.dtype == torch.float64:\n"
            "        block = 1024\n"
            "        num_warps = 4\n"
            "    else:\n"
            "        block = 2048\n"
            "        num_warps = 4\n"
            "    n = x.numel()\n"
            "    grid = (n // block,)\n"
            "    _add_kernel[grid](x, y, out, BLOCK=block, num_warps=num_warps, num_stages=2)\n"
            "    return out\n"
        )
        return {"code": code}