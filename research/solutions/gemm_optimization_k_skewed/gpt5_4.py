import os
import textwrap
import torch
import triton
import triton.language as tl


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = textwrap.dedent(
            """
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def gelu(x):
                return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

            @triton.autotune(
                configs=[
                    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
                    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
                    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
                    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),

                    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
                    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
                    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
                    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),

                    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=5),
                    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=5),
                    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=5),
                    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=5),
                ],
                key=["M", "N", "K"],
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
                pid = tl.program_id(0)
                num_pid_m = tl.cdiv(M, BLOCK_M)
                num_pid_n = tl.cdiv(N, BLOCK_N)
                num_pid_in_group = GROUP_M * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_M
                pid_in_group = pid % num_pid_in_group
                pid_m = first_pid_m + (pid_in_group % GROUP_M)
                pid_n = pid_in_group // GROUP_M

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                k_iter = 0
                while k_iter < K:
                    k_mask = (k_iter + offs_k) < K
                    a_mask = (offs_m[:, None] < M) & k_mask[None, :]
                    b_mask = k_mask[:, None] & (offs_n[None, :] < N)

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

            def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                """
                Matrix multiplication with GELU activation.

                Args:
                    a: Input tensor of shape (M, K)
                    b: Input tensor of shape (K, N)

                Returns:
                    Output tensor of shape (M, N) with GELU activation applied
                """
                assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
                assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D matrices"
                M, K_a = a.shape
                K_b, N = b.shape
                assert K_a == K_b, "K dimensions must match"

                M = int(M)
                N = int(N)
                K = int(K_a)
                if M == 0 or N == 0 or K == 0:
                    out_dtype = torch.result_type(a, b)
                    return torch.empty((M, N), device=a.device, dtype=out_dtype)

                out_dtype = torch.result_type(a, b)
                c = torch.empty((M, N), device=a.device, dtype=out_dtype)

                # Ensure strides in elements
                stride_am = a.stride(0)
                stride_ak = a.stride(1)
                stride_bk = b.stride(0)
                stride_bn = b.stride(1)
                stride_cm = c.stride(0)
                stride_cn = c.stride(1)

                # Grid: fused 1D launch with grouping along M for L2 locality
                def grid(meta):
                    BM = meta["BLOCK_M"]
                    BN = meta["BLOCK_N"]
                    GM = meta["GROUP_M"]
                    grid_m = triton.cdiv(M, BM)
                    grid_n = triton.cdiv(N, BN)
                    return (int((grid_m + GM - 1) // GM * GM * grid_n),)

                _matmul_gelu_kernel[grid](
                    a, b, c,
                    M, N, K,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    stride_cm, stride_cn,
                )
                return c
            """
        )
        return {"code": code}