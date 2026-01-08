import os
import textwrap

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent('''
            import math
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def gelu(x):
                return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

            @triton.autotune(
                configs=[
                    triton.Config(
                        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                        num_warps=8, num_stages=4
                    ),
                    triton.Config(
                        {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                        num_warps=4, num_stages=4
                    ),
                    triton.Config(
                        {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                        num_warps=4, num_stages=4
                    ),
                    triton.Config(
                        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                        num_warps=4, num_stages=4
                    ),
                    triton.Config(
                        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
                        num_warps=8, num_stages=3
                    ),
                    triton.Config(
                        {'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4},
                        num_warps=8, num_stages=3
                    ),
                    triton.Config(
                        {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4},
                        num_warps=8, num_stages=3
                    ),
                ],
                key=['M', 'N', 'K'],
            )
            @triton.jit
            def _matmul_gelu_kernel(
                A_ptr, B_ptr, C_ptr,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr,
            ):
                pid = tl.program_id(axis=0)

                num_pid_m = tl.cdiv(M, BLOCK_M)
                num_pid_n = tl.cdiv(N, BLOCK_N)

                blocks_per_group = GROUP_M * num_pid_n
                group_id = pid // blocks_per_group
                first_pid_m = group_id * GROUP_M
                group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
                pid_in_group = pid % blocks_per_group
                pid_m = first_pid_m + (pid_in_group % group_size_m)
                pid_n = pid_in_group // group_size_m

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                k = 0
                while k < K:
                    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
                    b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)

                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                    acc += tl.dot(a, b, out_dtype=tl.float32)

                    k += BLOCK_K
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk

                acc = gelu(acc)

                c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
                c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

                # Store dtype is inferred from C_ptr's dtype
                tl.store(c_ptrs, acc, mask=c_mask)

            def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                assert a.is_cuda and b.is_cuda, "Input tensors must be CUDA tensors"
                assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D matrices"
                M, K = a.shape
                Kb, N = b.shape
                assert K == Kb, "Incompatible matrix dimensions"

                # Determine output dtype:
                # Accumulate in fp32, store in promote_types(a,b) to be numerically stable yet efficient
                out_dtype = torch.result_type(a.dtype, b.dtype)
                # We'll allocate C in fp32 for numerical stability within the kernel, then cast if needed
                # However, Triton kernel stores with dtype inferred from pointer; use fp32 buffer then cast
                c_tmp = torch.empty((M, N), device=a.device, dtype=torch.float32)

                grid = lambda META: (
                    triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
                )

                _matmul_gelu_kernel[grid](
                    a, b, c_tmp,
                    M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c_tmp.stride(0), c_tmp.stride(1),
                )

                if out_dtype != torch.float32:
                    return c_tmp.to(out_dtype)
                return c_tmp
        ''')
        return {"code": code}