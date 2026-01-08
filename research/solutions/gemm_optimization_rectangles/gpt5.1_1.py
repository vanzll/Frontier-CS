import textwrap


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
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
                    triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
                ],
                key=['M', 'N', 'K'],
            )
            @triton.jit
            def matmul_gelu_kernel(
                a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            ):
                pid_m = tl.program_id(axis=0)
                pid_n = tl.program_id(axis=1)

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k in range(0, K, BLOCK_K):
                    k_ids = k + offs_k

                    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak)
                    b_ptrs = b_ptr + (k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn)

                    a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
                    b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)

                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                    acc += tl.dot(a, b)

                acc = gelu(acc)

                c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
                c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                tl.store(c_ptrs, acc, mask=c_mask)


            def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                if a.ndim != 2 or b.ndim != 2:
                    raise ValueError("Input tensors must be 2D")
                if a.shape[1] != b.shape[0]:
                    raise ValueError("Incompatible matrix dimensions for multiplication")
                if a.device.type != 'cuda' or b.device.type != 'cuda':
                    raise ValueError("Input tensors must be on CUDA device")
                if a.dtype != b.dtype:
                    raise ValueError("Input tensors must have the same dtype")

                M, K = a.shape
                Kb, N = b.shape

                if M == 0 or N == 0 or K == 0:
                    return torch.empty((M, N), device=a.device, dtype=a.dtype)

                stride_am, stride_ak = a.stride()
                stride_bk, stride_bn = b.stride()

                c = torch.empty((M, N), device=a.device, dtype=a.dtype)
                stride_cm, stride_cn = c.stride()

                grid = lambda META: (
                    triton.cdiv(M, META['BLOCK_M']),
                    triton.cdiv(N, META['BLOCK_N']),
                )

                matmul_gelu_kernel[grid](
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