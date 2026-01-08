import os
import textwrap

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent("""
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def gelu(x):
                return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

            @triton.autotune(
                configs=[
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=2, num_stages=3),
                    triton.Config({'BLOCK_M': 96,  'BLOCK_N': 192, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 192, 'BLOCK_N': 96,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 4}, num_warps=8, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 4}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=5),
                ],
                key=['M', 'N', 'K'],
            )
            @triton.jit
            def _matmul_kernel(
                A_ptr, B_ptr, C_ptr,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr,
            ):
                pid_m = tl.program_id(axis=0)
                pid_n = tl.program_id(axis=1)

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                k_remaining = K
                k_iter = 0
                while k_iter < K:
                    a_mask = (offs_m[:, None] < M) & (k_iter + offs_k[None, :] < K)
                    b_mask = (k_iter + offs_k[:, None] < K) & (offs_n[None, :] < N)

                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                    acc += tl.dot(a, b)

                    k_iter += BLOCK_K
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk

                acc = gelu(acc)

                c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
                c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                tl.store(c_ptrs, acc, mask=c_mask)

            def _get_blocking(M, N, K):
                # Provide a reasonable heuristic for launch grid sizes
                return

            def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
                assert a.shape[1] == b.shape[0], "Incompatible shapes for matmul"
                M, K = a.shape
                K2, N = b.shape
                # Accumulate in fp32 and output in fp32 for numerical stability/performance
                out_dtype = torch.float32

                # Compute strides in elements
                stride_am, stride_ak = a.stride()
                stride_bk, stride_bn = b.stride()

                c = torch.empty((M, N), device=a.device, dtype=out_dtype)

                # C strides
                stride_cm, stride_cn = c.stride()

                grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))

                # Kernel launch: let autotune pick best config per (M,N,K)
                _matmul_kernel[grid](
                    a, b, c,
                    M, N, K,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    stride_cm, stride_cn,
                )
                return c
        """)
        return {"code": code}