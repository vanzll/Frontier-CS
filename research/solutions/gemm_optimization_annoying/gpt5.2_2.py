import os
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=8,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=8,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int32,
    stride_ak: tl.int32,
    stride_bk: tl.int32,
    stride_bn: tl.int32,
    stride_cm: tl.int32,
    stride_cn: tl.int32,
    OUT_DTYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    num_pid_in_group = group_size * grid_n
    pid_group = pid // num_pid_in_group
    first_pid_m = pid_group * group_size
    pid_in_group = pid - pid_group * num_pid_in_group

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
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
        k += BLOCK_K

    out = gelu(acc).to(OUT_DTYPE)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, out, boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: a={tuple(a.shape)} b={tuple(b.shape)}")
    if a.dtype != b.dtype:
        b = b.to(dtype=a.dtype)

    M, K = a.shape
    _, N = b.shape

    if a.dtype == torch.float16:
        out_dtype = tl.float16
        allow_tf32 = False
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    elif a.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
        allow_tf32 = False
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    elif a.dtype == torch.float32:
        out_dtype = tl.float32
        allow_tf32 = True
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    else:
        a = a.to(torch.float16)
        b = b.to(torch.float16)
        out_dtype = tl.float16
        allow_tf32 = False
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

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
        OUT_DTYPE=out_dtype,
        ALLOW_TF32=allow_tf32,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        try:
            return {"program_path": os.path.abspath(__file__)}
        except Exception:
            import inspect
            import sys

            mod = sys.modules[__name__]
            src = inspect.getsource(mod)
            return {"code": src}