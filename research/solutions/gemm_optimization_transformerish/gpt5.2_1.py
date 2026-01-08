import os
import inspect
import textwrap
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.jit
def _matmul_gelu_fwd_kernel_contig(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    pid_group = pid // (group_size * grid_n)
    first_pid_m = pid_group * group_size
    pid_in_group = pid % (group_size * grid_n)
    pid_m = first_pid_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    a_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(offs_am, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, offs_bn),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.static_range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    acc = gelu(acc)
    out = tl.cast(acc, OUT_DTYPE)

    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(offs_am, offs_bn),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, out)


@triton.jit
def _matmul_gelu_fwd_kernel_generic(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    pid_group = pid // (group_size * grid_n)
    first_pid_m = pid_group * group_size
    pid_in_group = pid % (group_size * grid_n)
    pid_m = first_pid_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    a_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.static_range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    acc = gelu(acc)
    out = tl.cast(acc, OUT_DTYPE)

    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, out, boundary_check=(0, 1))


def _tl_dtype_from_torch(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    return None


def _select_meta(M: int, N: int, K: int):
    # Favor moderate tiles due to expensive GELU epilogue; keep warps high for tensor core utilization.
    # Evaluation shapes are multiples of 128 and K multiples of 32/64.
    BK = 32 if (K % 32 == 0) else 16
    if N >= 8192 and (N % 256 == 0):
        BN = 256
    else:
        BN = 128
    if M >= 8192 and (M % 256 == 0) and BN == 256:
        BM = 128
    else:
        BM = 128
    if BM == 128 and BN == 256:
        num_warps = 8
        num_stages = 4
    else:
        num_warps = 8
        num_stages = 5
    GROUP_M = 8
    return BM, BN, BK, GROUP_M, num_warps, num_stages


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes")
    if not (a.is_cuda and b.is_cuda):
        return F.gelu(a @ b, approximate="none")

    if a.dtype != b.dtype:
        # Fallback: torch handles mixed dtypes more broadly.
        return F.gelu(a @ b, approximate="none")

    if a.dtype not in (torch.float16, torch.bfloat16):
        return F.gelu(a @ b, approximate="none")

    M, K = a.shape
    _, N = b.shape

    out_dtype = a.dtype
    tl_out_dtype = _tl_dtype_from_torch(out_dtype)
    if tl_out_dtype is None:
        return F.gelu(a @ b, approximate="none")

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    BM, BN, BK, GROUP_M, num_warps, num_stages = _select_meta(M, N, K)

    # Fast path for contiguous row-major inputs and aligned shapes.
    # Also require grid_m divisible by GROUP_M to avoid out-of-bounds without masking.
    if a.is_contiguous() and b.is_contiguous():
        grid_m = M // BM if (M % BM == 0) else -1
        grid_n = N // BN if (N % BN == 0) else -1
        if (
            grid_m > 0
            and grid_n > 0
            and (K % BK == 0)
            and (grid_m % GROUP_M == 0)
        ):
            grid = (grid_m * grid_n,)
            _matmul_gelu_fwd_kernel_contig[grid](
                a,
                b,
                c,
                M,
                N,
                K=K,
                OUT_DTYPE=tl_out_dtype,
                BLOCK_M=BM,
                BLOCK_N=BN,
                BLOCK_K=BK,
                GROUP_M=GROUP_M,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return c

    # Generic Triton kernel for other layouts/shapes, but avoid compiling too many variants:
    # for uncommon K values, fallback to torch.
    if K not in (4096, 11008, 8192, 2048, 3072, 5120, 6144):
        return F.gelu(a @ b, approximate="none")

    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _matmul_gelu_fwd_kernel_generic[grid](
        a,
        b,
        c,
        M,
        N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        OUT_DTYPE=tl_out_dtype,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        GROUP_M=GROUP_M,
        num_warps=num_warps,
        num_stages=max(3, num_stages - 1),
    )
    return c


_SOLUTION_CODE_FALLBACK = textwrap.dedent(
    r"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.jit
def _matmul_gelu_fwd_kernel_contig(
    A_ptr, B_ptr, C_ptr,
    M, N,
    K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M
    pid_group = pid // (group_size * grid_n)
    first_pid_m = pid_group * group_size
    pid_in_group = pid % (group_size * grid_n)
    pid_m = first_pid_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    a_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(offs_am, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, offs_bn),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.static_range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    acc = gelu(acc)
    out = tl.cast(acc, OUT_DTYPE)
    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(offs_am, offs_bn),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, out)

def _tl_dtype_from_torch(dtype: torch.dtype):
    if dtype == torch.float16: return tl.float16
    if dtype == torch.bfloat16: return tl.bfloat16
    if dtype == torch.float32: return tl.float32
    return None

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (a.is_cuda and b.is_cuda): return F.gelu(a @ b, approximate="none")
    if a.ndim != 2 or b.ndim != 2: raise ValueError("a and b must be 2D")
    if a.shape[1] != b.shape[0]: raise ValueError("incompatible shapes")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        return F.gelu(a @ b, approximate="none")

    M, K = a.shape
    _, N = b.shape
    out_dtype = a.dtype
    tl_out = _tl_dtype_from_torch(out_dtype)
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    BM, BN, BK, GROUP_M = 128, 128, 32, 8
    if N >= 8192 and (N % 256 == 0): BN = 256
    if not (a.is_contiguous() and b.is_contiguous()):
        return F.gelu(a @ b, approximate="none")

    grid_m = M // BM if (M % BM == 0) else -1
    grid_n = N // BN if (N % BN == 0) else -1
    if grid_m <= 0 or grid_n <= 0 or (K % BK != 0) or (grid_m % GROUP_M != 0):
        return F.gelu(a @ b, approximate="none")

    grid = (grid_m * grid_n,)
    _matmul_gelu_fwd_kernel_contig[grid](
        a, b, c,
        M, N,
        K=K,
        OUT_DTYPE=tl_out,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_M=GROUP_M,
        num_warps=8,
        num_stages=5 if BN == 128 else 4,
    )
    return c
"""
).lstrip()


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        try:
            path = __file__
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return {"code": f.read()}
        except Exception:
            pass
        try:
            parts = [
                "import torch\nimport torch.nn.functional as F\nimport triton\nimport triton.language as tl\n\n",
                inspect.getsource(gelu) + "\n\n",
                inspect.getsource(_matmul_gelu_fwd_kernel_contig) + "\n\n",
                inspect.getsource(_matmul_gelu_fwd_kernel_generic) + "\n\n",
                inspect.getsource(_tl_dtype_from_torch) + "\n\n",
                inspect.getsource(_select_meta) + "\n\n",
                inspect.getsource(matmul) + "\n",
            ]
            return {"code": "".join(parts)}
        except Exception:
            return {"code": _SOLUTION_CODE_FALLBACK}