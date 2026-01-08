import textwrap


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
            num_warps=4,
            num_stages=4,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group

    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32)

    acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Input tensors must be 2D")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Input tensors must be CUDA tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix dimensions")

    if a.dtype != b.dtype:
        raise ValueError("Input tensors must have the same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype: {a.dtype}")

    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError("Inner dimensions must match")

    out_dtype = torch.float32
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    if a.dtype != torch.float32:
        return c.to(a.dtype)
    return c
'''
        return {"code": textwrap.dedent(code)}