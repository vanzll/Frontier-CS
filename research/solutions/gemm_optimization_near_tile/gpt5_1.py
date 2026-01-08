import os
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),

        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),

        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),

        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=3, num_warps=16),
    ],
    key=['M', 'N', 'K', 'a_stride_am', 'a_stride_ak', 'b_stride_bk', 'b_stride_bn'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_stride_am, a_stride_ak,
    b_stride_bk, b_stride_bn,
    c_stride_cm, c_stride_cn,
    OUT_DTYPE: tl.constexpr,
    ACTIVATION_GELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    num_pid_in_group = group_size * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * a_stride_am + offs_k[None, :] * a_stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * b_stride_bk + offs_n[None, :] * b_stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        if EVEN_K and EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        if EVEN_K and EVEN_N:
            b = tl.load(b_ptrs)
        else:
            b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * a_stride_ak
        b_ptrs += BLOCK_K * b_stride_bk
        k += BLOCK_K

    if ACTIVATION_GELU:
        acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * c_stride_cm + offs_n[None, :] * c_stride_cn)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc.to(OUT_DTYPE))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(OUT_DTYPE), mask=c_mask)


def _torch_dtype_to_tl(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError("Unsupported dtype: {}".format(dtype))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert a.is_cuda and b.is_cuda
    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2, "Incompatible shapes"
    K = K1

    out_dtype = a.dtype
    assert out_dtype in (torch.float16, torch.bfloat16, torch.float32), "Unsupported dtype"
    assert b.dtype == a.dtype, "Input dtypes must match"

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    a_stride_am, a_stride_ak = a.stride()
    b_stride_bk, b_stride_bn = b.stride()
    c_stride_cm, c_stride_cn = c.stride()

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a_stride_am, a_stride_ak,
        b_stride_bk, b_stride_bn,
        c_stride_cm, c_stride_cn,
        OUT_DTYPE=_torch_dtype_to_tl(out_dtype),
        ACTIVATION_GELU=True,
        EVEN_M=(M % 1 == 0) and (M % 64 == 0 if 'BLOCK_M' not in {} else (M % 64 == 0)),  # placeholder, will be overridden by meta
        EVEN_N=(N % 1 == 0) and (N % 64 == 0 if 'BLOCK_N' not in {} else (N % 64 == 0)),
        EVEN_K=(K % 1 == 0) and (K % 32 == 0 if 'BLOCK_K' not in {} else (K % 32 == 0)),
    )

    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),

        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),

        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),

        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 4}, num_stages=3, num_warps=16),
    ],
    key=['M', 'N', 'K', 'a_stride_am', 'a_stride_ak', 'b_stride_bk', 'b_stride_bn'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_stride_am, a_stride_ak,
    b_stride_bk, b_stride_bn,
    c_stride_cm, c_stride_cn,
    OUT_DTYPE: tl.constexpr,
    ACTIVATION_GELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    num_pid_in_group = group_size * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * a_stride_am + offs_k[None, :] * a_stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * b_stride_bk + offs_n[None, :] * b_stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        if EVEN_K and EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        if EVEN_K and EVEN_N:
            b = tl.load(b_ptrs)
        else:
            b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * a_stride_ak
        b_ptrs += BLOCK_K * b_stride_bk
        k += BLOCK_K

    if ACTIVATION_GELU:
        acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * c_stride_cm + offs_n[None, :] * c_stride_cn)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc.to(OUT_DTYPE))
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(OUT_DTYPE), mask=c_mask)


def _torch_dtype_to_tl(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError("Unsupported dtype: {}".format(dtype))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert a.is_cuda and b.is_cuda
    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2, "Incompatible shapes"
    K = K1

    out_dtype = a.dtype
    assert out_dtype in (torch.float16, torch.bfloat16, torch.float32), "Unsupported dtype"
    assert b.dtype == a.dtype, "Input dtypes must match"

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    a_stride_am, a_stride_ak = a.stride()
    b_stride_bk, b_stride_bn = b.stride()
    c_stride_cm, c_stride_cn = c.stride()

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    even_m = (M % 128 == 0) or (M % 64 == 0) or (M % 256 == 0)
    even_n = (N % 128 == 0) or (N % 64 == 0) or (N % 256 == 0)
    # EVEN_K is provided per-config; pass True for common fast path when divisible by 32
    even_k = (K % 128 == 0) or (K % 64 == 0) or (K % 32 == 0)

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a_stride_am, a_stride_ak,
        b_stride_bk, b_stride_bn,
        c_stride_cm, c_stride_cn,
        OUT_DTYPE=_torch_dtype_to_tl(out_dtype),
        ACTIVATION_GELU=True,
        EVEN_M=even_m,
        EVEN_N=even_n,
        EVEN_K=even_k,
    )

    return c
'''
        return {"code": code}