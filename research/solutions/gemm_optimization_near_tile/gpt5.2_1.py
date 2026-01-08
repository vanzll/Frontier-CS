import os
import math
import torch
import triton
import triton.language as tl

KERNEL_SRC = r'''
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

_KERNEL_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_warps=8, num_stages=5),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 4}, num_warps=8, num_stages=3),
]

@triton.autotune(
    configs=_KERNEL_CONFIGS,
    key=['M_cat', 'N_cat', 'K_cat', 'A_layout', 'B_layout'],
)
@triton.jit
def _matmul_gelu_kernel_constk(
    a_ptr, b_ptr, c_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M_cat: tl.constexpr, N_cat: tl.constexpr, K_cat: tl.constexpr,
    A_layout: tl.constexpr, B_layout: tl.constexpr,
    K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid - group_id * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    tl.multiple_of(stride_ak, 1)
    tl.multiple_of(stride_bn, 1)
    tl.max_contiguous(offs_k, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.static_range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs + k * stride_ak, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0, cache_modifier=".ca")
        b = tl.load(b_ptrs + k * stride_bk, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0, cache_modifier=".ca")
        acc = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32) + acc

    acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.autotune(
    configs=_KERNEL_CONFIGS,
    key=['M_cat', 'N_cat', 'K_cat', 'A_layout', 'B_layout'],
)
@triton.jit
def _matmul_gelu_kernel_dynk(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M_cat: tl.constexpr, N_cat: tl.constexpr, K_cat: tl.constexpr,
    A_layout: tl.constexpr, B_layout: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid - group_id * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs + k * stride_ak, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0, cache_modifier=".ca")
        b = tl.load(b_ptrs + k * stride_bk, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0, cache_modifier=".ca")
        acc = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32) + acc
        k += BLOCK_K

    acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def _cat4(x, t0, t1, t2):
    if x <= t0:
        return 0
    if x <= t1:
        return 1
    if x <= t2:
        return 2
    return 3

def _layout_flag_2d(x):
    # 0: row-major (stride(1)==1), 1: col-major-ish (stride(0)==1), 2: other
    s0, s1 = x.stride(0), x.stride(1)
    if s1 == 1:
        return 0
    if s0 == 1:
        return 1
    return 2

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes")
    if a.device.type != "cuda" or b.device.type != "cuda":
        x = a @ b
        return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))
    if a.dtype != b.dtype:
        b = b.to(dtype=a.dtype)

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    M_cat = _cat4(M, 128, 256, 512)
    N_cat = _cat4(N, 128, 256, 512)
    K_cat = _cat4(K, 64, 128, 256)
    A_layout = _layout_flag_2d(a)
    B_layout = _layout_flag_2d(b)

    allow_tf32 = (a.dtype == torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    if K <= 512:
        _matmul_gelu_kernel_constk[grid](
            a, b, c,
            M, N,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            M_cat=M_cat, N_cat=N_cat, K_cat=K_cat,
            A_layout=A_layout, B_layout=B_layout,
            K=K,
            ALLOW_TF32=allow_tf32,
        )
    else:
        _matmul_gelu_kernel_dynk[grid](
            a, b, c,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            M_cat=M_cat, N_cat=N_cat, K_cat=K_cat,
            A_layout=A_layout, B_layout=B_layout,
            ALLOW_TF32=allow_tf32,
        )
    return c
'''

exec(KERNEL_SRC, globals())

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": KERNEL_SRC}