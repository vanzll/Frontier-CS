import os
import sys
import inspect
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    k = 0
    while k < K:
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] + k < K)

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a_block, b_block)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    acc = gelu(acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {a.shape} and {b.shape}")

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    # Move to a common CUDA device and dtype
    if not a.is_cuda and not b.is_cuda:
        device = torch.device("cuda")
    elif a.is_cuda and b.is_cuda:
        if a.device != b.device:
            raise ValueError("Input tensors must be on the same device")
        device = a.device
    else:
        # One tensor on CPU, one on CUDA: move CPU tensor to CUDA of the other
        device = a.device if a.is_cuda else b.device

    in_dtype = torch.promote_types(a.dtype, b.dtype)
    if in_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        in_dtype = torch.float32

    a_dev = a.to(device=device, dtype=in_dtype, non_blocking=True)
    b_dev = b.to(device=device, dtype=in_dtype, non_blocking=True)

    c_dev = torch.empty((M, N), device=device, dtype=in_dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _matmul_kernel[grid](
        a_dev, b_dev, c_dev,
        M, N, K,
        a_dev.stride(0), a_dev.stride(1),
        b_dev.stride(0), b_dev.stride(1),
        c_dev.stride(0), c_dev.stride(1),
    )

    # If original inputs were on CPU, move result back to CPU
    if not a.is_cuda and not b.is_cuda:
        return c_dev.to(a.device, non_blocking=False)
    return c_dev


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        path = None
        try:
            path = os.path.abspath(__file__)
            with open(path, "r") as f:
                code = f.read()
            return {"code": code}
        except Exception:
            module = sys.modules.get(__name__, None)
            if module is None:
                raise
            src = inspect.getsource(module)
            return {"code": src}