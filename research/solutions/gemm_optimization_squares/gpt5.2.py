import os
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _matmul_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=5),
    ]


@triton.autotune(
    configs=_matmul_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_gelu_fast_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    tl.multiple_of(M, BLOCK_M)
    tl.multiple_of(N, BLOCK_N)
    tl.multiple_of(K, BLOCK_K)

    group_id = pid // (GROUP_M * grid_n)
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_in_group = pid - group_id * (GROUP_M * grid_n)
    pid_m = first_pid_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    acc = gelu(acc)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc)


@triton.autotune(
    configs=_matmul_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_gelu_general_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_id = pid // (GROUP_M * grid_n)
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_in_group = pid - group_id * (GROUP_M * grid_n)
    pid_m = first_pid_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    acc = gelu(acc)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes")
    if not (a.is_cuda and b.is_cuda):
        c = a @ b
        return torch.nn.functional.gelu(c, approximate="none")
    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")

    M, K = a.shape
    _, N = b.shape

    if a.dtype in (torch.float16, torch.bfloat16):
        out_dtype = a.dtype
    elif a.dtype == torch.float32:
        out_dtype = torch.float32
    else:
        c = a @ b
        return torch.nn.functional.gelu(c, approximate="none")

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    use_fast = (M % 256 == 0) and (N % 256 == 0) and (K % 64 == 0)
    if use_fast:
        _matmul_gelu_fast_kernel[grid](
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
    else:
        _matmul_gelu_general_kernel[grid](
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
            path = __file__
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return {"code": f.read()}
        except Exception:
            pass
        return {"program_path": __file__}