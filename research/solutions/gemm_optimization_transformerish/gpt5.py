import os
import math
import textwrap
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4}, num_warps=8, num_stages=4),
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
    OTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    group_range = group_size * num_pid_n
    group_id = pid // group_range
    first_pid_m = group_id * group_size
    pid_in_group = pid % group_range
    pid_m = first_pid_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_remaining = K
    while k_remaining > 0:
        k_mask_a = (offs_k[None, :] + (K - k_remaining)) < K
        k_mask_b = (offs_k[:, None] + (K - k_remaining)) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & k_mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_remaining -= BLOCK_K

    acc = gelu(acc)

    if OTYPE == 0:
        out = acc.to(tl.float16)
    elif OTYPE == 1:
        out = acc.to(tl.bfloat16)
    else:
        out = acc

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _otype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    return 2


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D tensors"
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Incompatible dimensions"
    # Determine output dtype
    out_dtype = a.dtype if a.dtype == b.dtype else torch.promote_types(a.dtype, b.dtype)
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    # Strides in elements
    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _matmul_gelu_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        _otype_code(out_dtype),
    )
    return c


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
                    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
                    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
                    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
                    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
                    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=5),
                    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=5),
                    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
                    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4}, num_warps=8, num_stages=4),
                    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4}, num_warps=8, num_stages=4),
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
                OTYPE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr,
            ):
                pid = tl.program_id(axis=0)

                num_pid_m = tl.cdiv(M, BLOCK_M)
                num_pid_n = tl.cdiv(N, BLOCK_N)

                group_size = GROUP_M
                group_range = group_size * num_pid_n
                group_id = pid // group_range
                first_pid_m = group_id * group_size
                pid_in_group = pid % group_range
                pid_m = first_pid_m + (pid_in_group % group_size)
                pid_n = pid_in_group // group_size

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                k_remaining = K
                while k_remaining > 0:
                    k_mask_a = (offs_k[None, :] + (K - k_remaining)) < K
                    k_mask_b = (offs_k[:, None] + (K - k_remaining)) < K
                    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask_a, other=0.0)
                    b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & k_mask_b, other=0.0)
                    acc += tl.dot(a, b)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
                    k_remaining -= BLOCK_K

                acc = gelu(acc)

                if OTYPE == 0:
                    out = acc.to(tl.float16)
                elif OTYPE == 1:
                    out = acc.to(tl.bfloat16)
                else:
                    out = acc

                c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
                tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

            def _otype_code(dtype: torch.dtype) -> int:
                if dtype == torch.float16:
                    return 0
                if dtype == torch.bfloat16:
                    return 1
                return 2

            def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D tensors"
                assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
                M, K = a.shape
                Kb, N = b.shape
                assert K == Kb, "Incompatible dimensions"
                out_dtype = a.dtype if a.dtype == b.dtype else torch.promote_types(a.dtype, b.dtype)
                c = torch.empty((M, N), device=a.device, dtype=out_dtype)
                stride_am = a.stride(0)
                stride_ak = a.stride(1)
                stride_bk = b.stride(0)
                stride_bn = b.stride(1)
                stride_cm = c.stride(0)
                stride_cn = c.stride(1)

                def grid(meta):
                    return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

                _matmul_gelu_kernel[grid](
                    a, b, c,
                    M, N, K,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    stride_cm, stride_cn,
                    _otype_code(out_dtype),
                )
                return c
            """
        )
        return {"code": code}