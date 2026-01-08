import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),

        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4}, num_stages=4, num_warps=8),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4}, num_stages=5, num_warps=8),
    ],
    key=[
        "M",
        "N",
        "K",
        "a_stride_am",
        "a_stride_ak",
        "b_stride_bk",
        "b_stride_bn",
        "c_stride_cm",
        "c_stride_cn",
    ],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    a_stride_am,
    a_stride_ak,
    b_stride_bk,
    b_stride_bn,
    c_stride_cm,
    c_stride_cn,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rm[:, None] * a_stride_am + rk[None, :] * a_stride_ak
    b_ptrs = b_ptr + rk[:, None] * b_stride_bk + rn[None, :] * b_stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_remaining = K
    while k_remaining > 0:
        k_mask_a = rk[None, :] < k_remaining
        k_mask_b = rk[:, None] < k_remaining

        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b & (rn[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * a_stride_ak
        b_ptrs += BLOCK_K * b_stride_bk
        k_remaining -= BLOCK_K

    if ACTIVATION == "gelu":
        acc = gelu(acc)

    c_ptrs = c_ptr + rm[:, None] * c_stride_cm + rn[None, :] * c_stride_cn
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matmul"
    assert a.dtype == b.dtype, "Input dtypes must match"

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _matmul_gelu_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        ACTIVATION="gelu",
    )

    return out


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
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),

        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4}, num_stages=4, num_warps=8),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4}, num_stages=5, num_warps=8),
    ],
    key=[
        "M",
        "N",
        "K",
        "a_stride_am",
        "a_stride_ak",
        "b_stride_bk",
        "b_stride_bn",
        "c_stride_cm",
        "c_stride_cn",
    ],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    a_stride_am,
    a_stride_ak,
    b_stride_bk,
    b_stride_bn,
    c_stride_cm,
    c_stride_cn,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rm[:, None] * a_stride_am + rk[None, :] * a_stride_ak
    b_ptrs = b_ptr + rk[:, None] * b_stride_bk + rn[None, :] * b_stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_remaining = K
    while k_remaining > 0:
        k_mask_a = rk[None, :] < k_remaining
        k_mask_b = rk[:, None] < k_remaining

        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b & (rn[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * a_stride_ak
        b_ptrs += BLOCK_K * b_stride_bk
        k_remaining -= BLOCK_K

    if ACTIVATION == "gelu":
        acc = gelu(acc)

    c_ptrs = c_ptr + rm[:, None] * c_stride_cm + rn[None, :] * c_stride_cn
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matmul"
    assert a.dtype == b.dtype, "Input dtypes must match"

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _matmul_gelu_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        ACTIVATION="gelu",
    )

    return out
'''
        return {"code": code}