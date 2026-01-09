import textwrap

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent("""
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def _vec_add_masked(x, y, out, n_elements, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offsets < n_elements
                x_vals = tl.load(x + offsets, mask=mask, other=0)
                y_vals = tl.load(y + offsets, mask=mask, other=0)
                tl.store(out + offsets, x_vals + y_vals, mask=mask)

            @triton.jit
            def _vec_add_nomask(x, y, out, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK + tl.arange(0, BLOCK)
                x_vals = tl.load(x + offsets)
                y_vals = tl.load(y + offsets)
                tl.store(out + offsets, x_vals + y_vals)

            def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                """
                Element-wise addition of two vectors.

                Args:
                    x: Input tensor of shape (16777216,)
                    y: Input tensor of shape (16777216,)

                Returns:
                    Output tensor of shape (16777216,) with x + y
                """
                if x.numel() == 0:
                    return torch.empty_like(x)
                assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
                assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
                assert x.shape == y.shape, "Input shapes must match"
                assert x.ndim == 1, "Inputs must be 1D tensors"

                n = x.numel()
                out = torch.empty_like(x)

                BLOCK = 4096
                NUM_WARPS = 8
                NUM_STAGES = 2

                if n % BLOCK == 0:
                    grid = (n // BLOCK,)
                    _vec_add_nomask[grid](x, y, out, BLOCK=BLOCK, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
                else:
                    grid = (triton.cdiv(n, BLOCK),)
                    _vec_add_masked[grid](x, y, out, n, BLOCK=BLOCK, num_warps=NUM_WARPS, num_stages=NUM_STAGES)

                return out
        """)
        return {"code": code}