import os
import sys
import inspect

import torch
import triton
import triton.language as tl

# Ensure tl.extra.cuda.libdevice.erf exists, falling back to available implementations if necessary
try:
    _ = tl.extra.cuda.libdevice.erf  # type: ignore[attr-defined]
except Exception:
    class _ExtraContainer:
        pass
    if not hasattr(tl, "extra"):
        tl.extra = _ExtraContainer()  # type: ignore[attr-defined]
    if not hasattr(tl.extra, "cuda"):  # type: ignore[attr-defined]
        tl.extra.cuda = _ExtraContainer()  # type: ignore[attr-defined]
    if not hasattr(tl.extra.cuda, "libdevice"):  # type: ignore[attr-defined]
        tl.extra.cuda.libdevice = _ExtraContainer()  # type: ignore[attr-defined]
    if hasattr(tl, "libdevice") and hasattr(tl.libdevice, "erf"):  # type: ignore[attr-defined]
        tl.extra.cuda.libdevice.erf = tl.libdevice.erf  # type: ignore[attr-defined]
    elif hasattr(tl, "math") and hasattr(tl.math, "erf"):  # type: ignore[attr-defined]
        tl.extra.cuda.libdevice.erf = tl.math.erf  # type: ignore[attr-defined]
    else:
        def _erf_fallback(x):
            # Very rough fallback; should rarely be used.
            return x
        tl.extra.cuda.libdevice.erf = _erf_fallback  # type: ignore[attr-defined]


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8},
            num_stages=4, num_warps=8
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_stages=3, num_warps=4
        ),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    C_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    # Compute number of blocks along M and N
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    # Group programs by rows to improve L2/L1 cache utilization
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = (pid_in_group // group_size_m)

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Apply GELU activation in FP32
    acc = gelu(acc)

    # Cast to output dtype
    c_tile = acc.to(C_DTYPE)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c_tile, mask=c_mask)


def _to_triton_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported dtype for Triton matmul: {dtype}")


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible dimensions: a.shape={a.shape}, b.shape={b.shape}")

    # Fallback to PyTorch for unsupported devices/dtypes
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if (not a.is_cuda) or (not b.is_cuda) or (a.dtype not in supported_dtypes) or (b.dtype not in supported_dtypes):
        return torch.nn.functional.gelu(a @ b)

    # Ensure both tensors are on the same device
    if a.device != b.device:
        raise ValueError("Input tensors must be on the same device")

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    # Choose output dtype using PyTorch type promotion rules
    out_dtype = torch.result_type(a, b)
    if out_dtype not in supported_dtypes:
        # Fallback if promotion leads to unsupported dtype
        return torch.nn.functional.gelu(a @ b)

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    c_dtype_triton = _to_triton_dtype(out_dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        C_DTYPE=c_dtype_triton,
    )

    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        try:
            program_path = os.path.abspath(__file__)
            return {"program_path": program_path}
        except NameError:
            # __file__ might not exist in some execution environments; fall back to returning source code
            source = inspect.getsource(sys.modules[__name__])
            return {"code": source}