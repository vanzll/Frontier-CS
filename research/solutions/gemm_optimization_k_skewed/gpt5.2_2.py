import textwrap

KERNEL_CODE = textwrap.dedent(
    r"""
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

def _get_configs():
    return [
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 64,  "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 64,  "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),

        triton.Config({"BM": 128, "BN": 128, "BK": 64,  "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 64,  "BK": 64,  "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64,  "BN": 128, "BK": 64,  "GROUP_M": 8}, num_warps=4, num_stages=4),

        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=8, num_stages=5),
        triton.Config({"BM": 128, "BN": 64,  "BK": 128, "GROUP_M": 8}, num_warps=4, num_stages=5),
        triton.Config({"BM": 64,  "BN": 128, "BK": 128, "GROUP_M": 8}, num_warps=4, num_stages=5),
    ]

@triton.autotune(
    configs=_get_configs(),
    key=["K"],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)

    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    tl.multiple_of(offs_k, 16)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    k = 0
    while k < K:
        k_offs = k + offs_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_offs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        k += BK
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    acc = gelu(acc)
    out = tl.cast(acc, OUT_DTYPE)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out, mask=mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("matmul expects CUDA tensors")
    if a.device != b.device:
        raise ValueError("a and b must be on the same device")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes")

    M, K = a.shape
    _, N = b.shape

    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")

    if a.dtype == torch.float16:
        out_dtype = tl.float16
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    elif a.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    elif a.dtype == torch.float32:
        out_dtype = tl.float32
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported dtype: {a.dtype}")

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),)

    _matmul_gelu_kernel[grid](
        a, b, c,
        M=M, N=N, K=K,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
        OUT_DTYPE=out_dtype,
    )
    return c
"""
)

exec(KERNEL_CODE, globals(), globals())


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": KERNEL_CODE}