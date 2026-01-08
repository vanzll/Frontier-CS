import math
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["K", "TUNE_ID", "a_stride_am", "a_stride_ak", "b_stride_bk", "b_stride_bn", "c_stride_cm", "c_stride_cn"],
    warmup=2,
    rep=5,
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.int32,
    N: tl.int32,
    a_stride_am: tl.int32,
    a_stride_ak: tl.int32,
    b_stride_bk: tl.int32,
    b_stride_bn: tl.int32,
    c_stride_cm: tl.int32,
    c_stride_cn: tl.int32,
    TUNE_ID: tl.int32,
    K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _ = TUNE_ID

    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid - group_id * num_pid_in_group
    pid_m = first_pid_m + pid_in_group // grid_n
    pid_n = pid_in_group - (pid_in_group // grid_n) * grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if K % BLOCK_K == 0:
        for k in tl.static_range(0, K, BLOCK_K):
            a_ptrs = a_ptr + (offs_m[:, None] * a_stride_am + (k + offs_k)[None, :] * a_stride_ak)
            b_ptrs = b_ptr + ((k + offs_k)[:, None] * b_stride_bk + offs_n[None, :] * b_stride_bn)

            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)

            acc = tl.dot(a, b, acc=acc)
    else:
        for k in tl.static_range(0, K, BLOCK_K):
            k_ids = k + offs_k
            mask_k = k_ids < K

            a_ptrs = a_ptr + (offs_m[:, None] * a_stride_am + k_ids[None, :] * a_stride_ak)
            b_ptrs = b_ptr + (k_ids[:, None] * b_stride_bk + offs_n[None, :] * b_stride_bn)

            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            acc = tl.dot(a, b, acc=acc)

    acc = gelu(acc)

    c = tl.cast(acc, OUT_DTYPE)
    c_ptrs = c_ptr + offs_m[:, None] * c_stride_cm + offs_n[None, :] * c_stride_cn
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


def _tune_id(M: int, N: int) -> int:
    m_big = 1 if M >= 256 else 0
    n_big = 1 if N >= 256 else 0
    return m_big + (n_big << 1)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors a(M,K) and b(K,N)")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: a={tuple(a.shape)} b={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device.type != "cuda":
        c = a @ b
        return torch.nn.functional.gelu(c)

    if a.dtype != b.dtype:
        b = b.to(a.dtype)

    M, K = a.shape
    _, N = b.shape

    if a.dtype == torch.float16:
        out_dtype = tl.float16
    elif a.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif a.dtype == torch.float32:
        out_dtype = tl.float32
    else:
        raise TypeError(f"Unsupported dtype: {a.dtype}")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    tune = _tune_id(M, N)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _matmul_gelu_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        tune,
        K=K,
        OUT_DTYPE=out_dtype,
    )
    return c


_KERNEL_CODE = r'''
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["K", "TUNE_ID", "a_stride_am", "a_stride_ak", "b_stride_bk", "b_stride_bn", "c_stride_cm", "c_stride_cn"],
    warmup=2,
    rep=5,
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.int32,
    N: tl.int32,
    a_stride_am: tl.int32,
    a_stride_ak: tl.int32,
    b_stride_bk: tl.int32,
    b_stride_bn: tl.int32,
    c_stride_cm: tl.int32,
    c_stride_cn: tl.int32,
    TUNE_ID: tl.int32,
    K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    _ = TUNE_ID

    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid - group_id * num_pid_in_group
    pid_m = first_pid_m + pid_in_group // grid_n
    pid_n = pid_in_group - (pid_in_group // grid_n) * grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if K % BLOCK_K == 0:
        for k in tl.static_range(0, K, BLOCK_K):
            a_ptrs = a_ptr + (offs_m[:, None] * a_stride_am + (k + offs_k)[None, :] * a_stride_ak)
            b_ptrs = b_ptr + ((k + offs_k)[:, None] * b_stride_bk + offs_n[None, :] * b_stride_bn)

            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)

            acc = tl.dot(a, b, acc=acc)
    else:
        for k in tl.static_range(0, K, BLOCK_K):
            k_ids = k + offs_k
            mask_k = k_ids < K

            a_ptrs = a_ptr + (offs_m[:, None] * a_stride_am + k_ids[None, :] * a_stride_ak)
            b_ptrs = b_ptr + (k_ids[:, None] * b_stride_bk + offs_n[None, :] * b_stride_bn)

            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            acc = tl.dot(a, b, acc=acc)

    acc = gelu(acc)

    c = tl.cast(acc, OUT_DTYPE)
    c_ptrs = c_ptr + offs_m[:, None] * c_stride_cm + offs_n[None, :] * c_stride_cn
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


def _tune_id(M: int, N: int) -> int:
    m_big = 1 if M >= 256 else 0
    n_big = 1 if N >= 256 else 0
    return m_big + (n_big << 1)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors a(M,K) and b(K,N)")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: a={tuple(a.shape)} b={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device.type != "cuda":
        c = a @ b
        return torch.nn.functional.gelu(c)

    if a.dtype != b.dtype:
        b = b.to(a.dtype)

    M, K = a.shape
    _, N = b.shape

    if a.dtype == torch.float16:
        out_dtype = tl.float16
    elif a.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif a.dtype == torch.float32:
        out_dtype = tl.float32
    else:
        raise TypeError(f"Unsupported dtype: {a.dtype}")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    tune = _tune_id(M, N)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _matmul_gelu_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        tune,
        K=K,
        OUT_DTYPE=out_dtype,
    )
    return c
'''


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": _KERNEL_CODE}