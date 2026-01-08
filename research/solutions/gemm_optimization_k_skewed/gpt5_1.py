import os
import textwrap

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent(
            '''
            import math
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def gelu(x):
                return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

            def _get_configs():
                configs = []
                # A curated set of configurations to cover small/large K and varied M/N
                cfgs = [
                    # BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M
                    (64, 64, 32, 4, 3, 8),
                    (64, 128, 32, 4, 4, 8),
                    (128, 64, 32, 4, 4, 8),
                    (128, 128, 32, 8, 4, 8),
                    (64, 256, 32, 8, 4, 8),
                    (256, 64, 32, 8, 4, 8),

                    (64, 128, 64, 4, 5, 8),
                    (128, 64, 64, 4, 5, 8),
                    (128, 128, 64, 8, 5, 8),
                    (64, 256, 64, 8, 5, 8),
                    (256, 64, 64, 8, 5, 8),

                    (128, 128, 128, 8, 6, 8),
                    (64, 128, 128, 4, 6, 8),
                    (128, 64, 128, 4, 6, 8),
                ]
                for BM, BN, BK, NW, NS, GM in cfgs:
                    configs.append(
                        triton.Config(
                            {'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK, 'GROUP_M': GM},
                            num_stages=NS,
                            num_warps=NW
                        )
                    )
                return configs

            @triton.autotune(configs=_get_configs(), key=['M', 'N', 'K'])
            @triton.jit
            def _matmul_kernel(
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                ACT,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr,
            ):
                pid = tl.program_id(axis=0)

                num_pid_m = tl.cdiv(M, BLOCK_M)
                num_pid_n = tl.cdiv(N, BLOCK_N)
                group_size_m = GROUP_M

                group_id = pid // (group_size_m * num_pid_n)
                first_pid_m = group_id * group_size_m
                group_size_m = tl.minimum(num_pid_m - first_pid_m, group_size_m)
                pid_in_group = pid % (group_size_m * num_pid_n)
                pid_m = first_pid_m + (pid_in_group % group_size_m)
                pid_n = pid_in_group // group_size_m

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                k = 0
                while k < K:
                    k_remaining = K - k
                    k_mask = offs_k[None, :] < k_remaining
                    a_mask = (offs_m[:, None] < M) & k_mask
                    b_mask = k_mask.T & (offs_n[None, :] < N)

                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                    acc += tl.dot(a, b)

                    k += BLOCK_K
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk

                if ACT == 1:
                    acc = gelu(acc)

                c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
                c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                tl.store(c_ptrs, acc, mask=c_mask)

            def _promote_to_supported_dtype(t):
                if t.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    return t
                # Fallback: cast unsupported types to float32
                return t.to(torch.float32)

            def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                if a.ndim != 2 or b.ndim != 2:
                    raise ValueError("matmul expects 2D tensors")
                if a.shape[1] != b.shape[0]:
                    raise ValueError(f"Incompatible shapes: {a.shape} and {b.shape}")
                if not a.is_cuda or not b.is_cuda:
                    raise ValueError("Inputs must be CUDA tensors")

                a = _promote_to_supported_dtype(a)
                b = _promote_to_supported_dtype(b)

                M, K = a.shape
                Kb, N = b.shape
                assert K == Kb

                # Accumulate in fp32; output dtype follows accumulation dtype unless inputs are lower-precision
                acc_dtype = torch.float32
                out_dtype = torch.result_type(a.dtype, b.dtype)
                if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    out_dtype = torch.float32

                c = torch.empty((M, N), device=a.device, dtype=acc_dtype)

                grid = lambda META: (
                    triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
                )

                # Convert strides to element strides (PyTorch gives element strides already)
                _matmul_kernel[grid](
                    a, b, c,
                    M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    1,  # apply GELU activation
                )

                # Cast to the desired output dtype
                if out_dtype != acc_dtype:
                    c = c.to(out_dtype)
                return c

            __all__ = ["matmul", "gelu"]
            '''
        )
        return {"code": code}