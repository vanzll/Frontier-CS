import textwrap


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = r"""import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _configs_k64():
    cfgs = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4}, num_warps=8, num_stages=4),
    ]
    return cfgs


def _configs_k128():
    cfgs = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 4}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 4}, num_warps=8, num_stages=4),
    ]
    return cfgs


@triton.autotune(
    configs=_configs_k64(),
    key=["a_stride_am", "a_stride_ak", "b_stride_bk", "b_stride_bn"],
)
@triton.jit
def _matmul_gelu_k64(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    a_stride_am: tl.constexpr,
    a_stride_ak: tl.constexpr,
    b_stride_bk: tl.constexpr,
    b_stride_bn: tl.constexpr,
    c_stride_cm: tl.constexpr,
    c_stride_cn: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_m = tl.minimum(GROUP_M, num_pid_m)
    group_size = group_m * num_pid_n

    pid_group = pid // group_size
    first_pid_m = pid_group * group_m
    pid_in_group = pid - pid_group * group_size

    pid_n = pid_in_group // group_m
    pid_m = first_pid_m + (pid_in_group - pid_n * group_m)

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_block = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(a_stride_am, a_stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(b_stride_bk, b_stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
        acc = acc + tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))

    acc = gelu(acc)
    out = tl.cast(acc, OUT_DTYPE)

    c_block = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(c_stride_cm, c_stride_cn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block, out, boundary_check=(0, 1))


@triton.autotune(
    configs=_configs_k128(),
    key=["a_stride_am", "a_stride_ak", "b_stride_bk", "b_stride_bn"],
)
@triton.jit
def _matmul_gelu_k128(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    a_stride_am: tl.constexpr,
    a_stride_ak: tl.constexpr,
    b_stride_bk: tl.constexpr,
    b_stride_bn: tl.constexpr,
    c_stride_cm: tl.constexpr,
    c_stride_cn: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_m = tl.minimum(GROUP_M, num_pid_m)
    group_size = group_m * num_pid_n

    pid_group = pid // group_size
    first_pid_m = pid_group * group_m
    pid_in_group = pid - pid_group * group_size

    pid_n = pid_in_group // group_m
    pid_m = first_pid_m + (pid_in_group - pid_n * group_m)

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    a_block = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(a_stride_am, a_stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(b_stride_bk, b_stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
        acc = acc + tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))

    acc = gelu(acc)
    out = tl.cast(acc, OUT_DTYPE)

    c_block = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(c_stride_cm, c_stride_cn),
        offsets=(offs_m, offs_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block, out, boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: a={tuple(a.shape)}, b={tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: a={a.dtype}, b={b.dtype}")

    M, K = a.shape
    _, N = b.shape

    if a.numel() == 0 or b.numel() == 0:
        return torch.empty((M, N), device=a.device, dtype=a.dtype)

    if not a.is_cuda or not b.is_cuda:
        return torch.nn.functional.gelu(a @ b)

    if a.dtype == torch.float16:
        out_tl = tl.float16
        allow_tf32 = False
    elif a.dtype == torch.bfloat16:
        out_tl = tl.bfloat16
        allow_tf32 = False
    elif a.dtype == torch.float32:
        out_tl = tl.float32
        allow_tf32 = True
    else:
        raise ValueError(f"Unsupported dtype: {a.dtype}")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    a_stride_am, a_stride_ak = a.stride(0), a.stride(1)
    b_stride_bk, b_stride_bn = b.stride(0), b.stride(1)
    c_stride_cm, c_stride_cn = c.stride(0), c.stride(1)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    if K <= 96:
        _matmul_gelu_k64[grid](
            a,
            b,
            c,
            M=M,
            N=N,
            K=K,
            a_stride_am=a_stride_am,
            a_stride_ak=a_stride_ak,
            b_stride_bk=b_stride_bk,
            b_stride_bn=b_stride_bn,
            c_stride_cm=c_stride_cm,
            c_stride_cn=c_stride_cn,
            ALLOW_TF32=allow_tf32,
            OUT_DTYPE=out_tl,
        )
    else:
        _matmul_gelu_k128[grid](
            a,
            b,
            c,
            M=M,
            N=N,
            K=K,
            a_stride_am=a_stride_am,
            a_stride_ak=a_stride_ak,
            b_stride_bk=b_stride_bk,
            b_stride_bn=b_stride_bn,
            c_stride_cm=c_stride_cm,
            c_stride_cn=c_stride_cn,
            ALLOW_TF32=allow_tf32,
            OUT_DTYPE=out_tl,
        )

    return c
"""
        return {"code": textwrap.dedent(code)}