import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _to_triton_dtype(torch_dtype):
    if torch_dtype == torch.float16:
        return tl.float16
    if torch_dtype == torch.bfloat16:
        return tl.bfloat16
    if torch_dtype == torch.float32:
        return tl.float32
    # default to float32 for unsupported types
    return tl.float32


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 2}, num_stages=5, num_warps=16),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K", "a_stride_am", "a_stride_ak", "b_stride_bk", "b_stride_bn"],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_stride_am, a_stride_ak,
    b_stride_bk, b_stride_bn,
    c_stride_cm, c_stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # number of tile blocks
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # group tiles along M to maximize L2 reuse of B
    group_size = GROUP_M
    num_pid_in_group = group_size * grid_n
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
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        k += BLOCK_K
        a_ptrs += BLOCK_K * a_stride_ak
        b_ptrs += BLOCK_K * b_stride_bk

    # GELU activation on accumulator (fp32)
    acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * c_stride_cm + offs_n[None, :] * c_stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(OUT_DTYPE), mask=c_mask)


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
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication"
    M, K = a.shape
    Kb, N = b.shape
    assert Kb == K

    # Output dtype matches input A's dtype
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Strides for arbitrary layouts
    a_am, a_ak = a.stride(0), a.stride(1)
    b_bk, b_bn = b.stride(0), b.stride(1)
    c_cm, c_cn = out.stride(0), out.stride(1)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _matmul_gelu_kernel[grid](
        a, b, out,
        M, N, K,
        a_am, a_ak,
        b_bk, b_bn,
        c_cm, c_cn,
        _to_triton_dtype(out.dtype),
    )
    return out
'''
        return {"code": code}