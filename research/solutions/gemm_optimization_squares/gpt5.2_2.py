import math
import os
import sys

KERNEL_CODE = r'''
import math
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M * grid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid - group_id * group_size
    pid_m = first_pid_m + (pid_in_group // grid_n)
    pid_n = pid_in_group - (pid_in_group // grid_n) * grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_mask = (k + offs_k) < K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
            cache_modifier=".ca",
            eviction_policy="evict_last",
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
            cache_modifier=".ca",
            eviction_policy="evict_last",
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    out = gelu(acc)
    out = out.to(OUT_DTYPE)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out, mask=mask_c)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("matmul expects torch.Tensor inputs")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)} b={tuple(b.shape)}")
    if a.numel() == 0 or b.numel() == 0:
        m = a.shape[0]
        n = b.shape[1]
        return torch.empty((m, n), device=a.device, dtype=a.dtype)
    if not (a.is_cuda and b.is_cuda):
        x = a @ b
        return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))

    M, K = a.shape
    _, N = b.shape

    if a.dtype != b.dtype:
        # Minimal overhead: compute in promoted dtype if supported; else cast b to a.dtype
        dt = torch.promote_types(a.dtype, b.dtype)
        if dt in (torch.float16, torch.bfloat16, torch.float32):
            a_ = a.to(dt)
            b_ = b.to(dt)
            out = torch.empty((M, N), device=a.device, dtype=dt)
        else:
            a_ = a
            b_ = b.to(a.dtype)
            out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        a_ = a
        b_ = b
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    if out.dtype == torch.float16:
        out_tl = tl.float16
    elif out.dtype == torch.bfloat16:
        out_tl = tl.bfloat16
    else:
        out_tl = tl.float32

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _matmul_gelu_kernel[grid](
        a_, b_, out,
        M, N, K,
        a_.stride(0), a_.stride(1),
        b_.stride(0), b_.stride(1),
        out.stride(0), out.stride(1),
        OUT_DTYPE=out_tl,
    )
    return out
'''

exec(KERNEL_CODE, globals(), globals())


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": KERNEL_CODE}