import os
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_M': 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 4},
            num_stages=4,
            num_warps=8,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_gelu_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
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
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    num_pid_in_group = group_size * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group // num_pid_n
    pid_n = pid_in_group % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_indices = k + offs_k

        a_ptrs = A_ptr + (
            offs_m[:, None] * stride_am + k_indices[None, :] * stride_ak
        )
        b_ptrs = B_ptr + (
            k_indices[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )

        a_mask = (offs_m[:, None] < M) & (k_indices[None, :] < K)
        b_mask = (k_indices[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32)

    c = gelu(acc)

    c_ptrs = C_ptr + (
        offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Input tensors must be 2D")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible shapes for matmul")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("Inputs must be CUDA tensors")
    if a.dtype != b.dtype:
        raise ValueError("Input tensors must have the same dtype")

    M, K = a.shape
    Kb, N = b.shape
    assert Kb == K

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    _matmul_gelu_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
    )

    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        try:
            program_path = os.path.abspath(__file__)
            return {"program_path": program_path}
        except NameError:
            import inspect
            import sys

            module = sys.modules[__name__]
            code = inspect.getsource(module)
            return {"code": code}