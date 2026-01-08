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
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=["K", "stride_am", "stride_ak", "stride_bk", "stride_bn"],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid - group_id * group_size
    pid_m = first_pid_m + (pid_in_group // num_pid_n)
    pid_n = pid_in_group % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tl.multiple_of(offs_k, 8)
    tl.max_contiguous(offs_k, 16)
    tl.max_contiguous(offs_n, 8)
    tl.max_contiguous(offs_m, 8)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_mask = (k + offs_k) < K
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last")
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, eviction_policy="evict_last")

        acc += tl.dot(a, b)

        k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


def _gelu_ref(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes")
    if not (a.is_cuda and b.is_cuda):
        return _gelu_ref(a @ b)

    # Fast path: tensor-core friendly dtypes
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        return _gelu_ref(a @ b).to(dtype=a.dtype)

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _matmul_gelu_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


_KERNEL_CODE = r'''
import math
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=4),
    ],
    key=["K", "stride_am", "stride_ak", "stride_bk", "stride_bn"],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid - group_id * group_size
    pid_m = first_pid_m + (pid_in_group // num_pid_n)
    pid_n = pid_in_group % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tl.multiple_of(offs_k, 8)
    tl.max_contiguous(offs_k, 16)
    tl.max_contiguous(offs_n, 8)
    tl.max_contiguous(offs_m, 8)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_mask = (k + offs_k) < K
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last")
        b = tl.load(b_ptrs, mask=b_mask, other=0.0, eviction_policy="evict_last")

        acc += tl.dot(a, b)

        k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)

def _gelu_ref(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes")
    if not (a.is_cuda and b.is_cuda):
        return _gelu_ref(a @ b)

    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        return _gelu_ref(a @ b).to(dtype=a.dtype)

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _matmul_gelu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
'''


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": _KERNEL_CODE}