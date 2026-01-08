import torch
import triton
import triton.language as tl


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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=8),

        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 2}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 2}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K', 'a_dtype', 'b_dtype'],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    a_dtype: tl.constexpr, b_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    pid_in_group = pid % group_size
    pid_m = group_id * GROUP_M + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        a_mask = (offs_m[:, None] < M) & (k_iter + offs_k[None, :] < K)
        b_mask = ((k_iter + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible shapes for matmul")

    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Inputs must be CUDA tensors")
    if a.device != b.device:
        raise ValueError("Inputs must be on the same device")

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Support common dtypes; convert unsupported ones to float32 for correctness
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    a_in = a
    b_in = b
    if a.dtype not in supported_dtypes:
        a_in = a.to(torch.float32)
    if b.dtype not in supported_dtypes:
        b_in = b.to(torch.float32)

    out_dtype = torch.result_type(a_in.dtype, b_in.dtype)
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _matmul_gelu_kernel[grid](
        a_in, b_in, c,
        M, N, K,
        a_in.stride(0), a_in.stride(1),
        b_in.stride(0), b_in.stride(1),
        c.stride(0), c.stride(1),
        a_in.dtype, b_in.dtype,
    )
    return c
'''
        return {"code": code}