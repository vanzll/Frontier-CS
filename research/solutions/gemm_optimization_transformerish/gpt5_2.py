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
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8},
            num_warps=4,
            num_stages=4,
        ),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size_m = GROUP_M
    group_id = pid // (group_size_m * num_pid_n)
    first_pid_m = group_id * group_size_m
    group_size_m = tl.minimum(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid // group_size_m) % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = gelu(acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Inputs must be CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be 2D tensors")
    M, K1 = a.shape
    K2, N = b.shape
    if K1 != K2:
        raise ValueError("Inner dimensions must match for matmul")
    if a.dtype != b.dtype:
        raise ValueError("Input dtypes must match")
    if a.device != b.device:
        raise ValueError("Inputs must be on the same device")

    M, K = int(M), int(K1)
    N = int(N)

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _matmul_gelu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        src = []
        src.append("import torch")
        src.append("import triton")
        src.append("import triton.language as tl")
        src.append("")
        src.append("@triton.jit")
        src.append("def gelu(x):")
        src.append("    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))")
        src.append("")
        src.append("@triton.autotune(")
        src.append("    configs=[")
        src.append("        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),")
        src.append("        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),")
        src.append("        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),")
        src.append("        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=8, num_stages=4),")
        src.append("        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=8, num_stages=4),")
        src.append("        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),")
        src.append("        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),")
        src.append("        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),")
        src.append("    ],")
        src.append("    key=['M', 'N', 'K'],")
        src.append(")")
        src.append("@triton.jit")
        src.append("def _matmul_gelu_kernel(")
        src.append("    a_ptr, b_ptr, c_ptr,")
        src.append("    M, N, K,")
        src.append("    stride_am, stride_ak,")
        src.append("    stride_bk, stride_bn,")
        src.append("    stride_cm, stride_cn,")
        src.append("    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,")
        src.append("    GROUP_M: tl.constexpr,")
        src.append("):")
        src.append("    pid = tl.program_id(axis=0)")
        src.append("")
        src.append("    num_pid_m = tl.cdiv(M, BLOCK_M)")
        src.append("    num_pid_n = tl.cdiv(N, BLOCK_N)")
        src.append("")
        src.append("    group_size_m = GROUP_M")
        src.append("    group_id = pid // (group_size_m * num_pid_n)")
        src.append("    first_pid_m = group_id * group_size_m")
        src.append("    group_size_m = tl.minimum(num_pid_m - first_pid_m, group_size_m)")
        src.append("    pid_m = first_pid_m + (pid % group_size_m)")
        src.append("    pid_n = (pid // group_size_m) % num_pid_n")
        src.append("")
        src.append("    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
        src.append("    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)")
        src.append("    offs_k = tl.arange(0, BLOCK_K)")
        src.append("")
        src.append("    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)")
        src.append("    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)")
        src.append("")
        src.append("    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)")
        src.append("")
        src.append("    k = 0")
        src.append("    while k < K:")
        src.append("        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)")
        src.append("        b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)")
        src.append("        a = tl.load(a_ptrs, mask=a_mask, other=0.0)")
        src.append("        b = tl.load(b_ptrs, mask=b_mask, other=0.0)")
        src.append("")
        src.append("        acc += tl.dot(a, b)")
        src.append("")
        src.append("        k += BLOCK_K")
        src.append("        a_ptrs += BLOCK_K * stride_ak")
        src.append("        b_ptrs += BLOCK_K * stride_bk")
        src.append("")
        src.append("    acc = gelu(acc)")
        src.append("")
        src.append("    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)")
        src.append("    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)")
        src.append("    tl.store(c_ptrs, acc, mask=c_mask)")
        src.append("")
        src.append("def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:")
        src.append("    if not a.is_cuda or not b.is_cuda:")
        src.append("        raise ValueError('Inputs must be CUDA tensors')")
        src.append("    if a.ndim != 2 or b.ndim != 2:")
        src.append("        raise ValueError('Inputs must be 2D tensors')")
        src.append("    M, K1 = a.shape")
        src.append("    K2, N = b.shape")
        src.append("    if K1 != K2:")
        src.append("        raise ValueError('Inner dimensions must match for matmul')")
        src.append("    if a.dtype != b.dtype:")
        src.append("        raise ValueError('Input dtypes must match')")
        src.append("    if a.device != b.device:")
        src.append("        raise ValueError('Inputs must be on the same device')")
        src.append("")
        src.append("    M, K = int(M), int(K1)")
        src.append("    N = int(N)")
        src.append("")
        src.append("    c = torch.empty((M, N), device=a.device, dtype=a.dtype)")
        src.append("")
        src.append("    def grid(meta):")
        src.append("        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)")
        src.append("")
        src.append("    _matmul_gelu_kernel[grid](")
        src.append("        a, b, c,")
        src.append("        M, N, K,")
        src.append("        a.stride(0), a.stride(1),")
        src.append("        b.stride(0), b.stride(1),")
        src.append("        c.stride(0), c.stride(1),")
        src.append("    )")
        src.append("    return c")
        program = "\n".join(src)
        return {"code": program}