import os
import inspect
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _get_autotune_configs():
    cfgs = []
    for bm, bn, bk, warps, stages, group_m in [
        (128, 128, 32, 8, 4, 8),
        (128, 64, 32, 4, 4, 8),
        (64, 128, 32, 4, 4, 8),
        (64, 64, 32, 4, 3, 8),
        (128, 128, 64, 8, 5, 8),
        (128, 64, 64, 4, 5, 8),
    ]:
        cfgs.append(
            triton.Config(
                {"BM": bm, "BN": bn, "BK": bk, "GROUP_M": group_m},
                num_warps=warps,
                num_stages=stages,
            )
        )
    return cfgs


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_gelu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)

    group_size = GROUP_M
    num_pid_in_group = group_size * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size)
    pid_n = pid_in_group // group_size

    a = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0),
        block_shape=(BM, BK),
        order=(1, 0),
    )
    b = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN),
        block_shape=(BK, BN),
        order=(0, 1),
    )

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for _ in range(0, K, BK):
        a_tile = tl.load(a, boundary_check=(0, 1), padding_option="zero")
        b_tile = tl.load(b, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a_tile, b_tile)
        a = tl.advance(a, (0, BK))
        b = tl.advance(b, (BK, 0))

    acc = gelu(acc)
    out = acc.to(OUT_DTYPE)

    c = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN),
        block_shape=(BM, BN),
        order=(1, 0),
    )
    tl.store(c, out, boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise TypeError("a and b must be torch.Tensor")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.device.type != "cuda" or b.device.type != "cuda":
        c = a @ b
        return torch.nn.functional.gelu(c, approximate="none")

    M, K = a.shape
    K2, N = b.shape
    if K2 != K:
        raise ValueError(f"Incompatible shapes: a={a.shape}, b={b.shape}")

    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype not in (torch.float16, torch.bfloat16):
        c = a @ b
        return torch.nn.functional.gelu(c, approximate="none")

    out_dtype_tl = tl.float16 if a.dtype == torch.float16 else tl.bfloat16
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),)

    _matmul_gelu_kernel[grid](
        a,
        b,
        c,
        M=M,
        N=N,
        stride_am=stride_am,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        stride_bn=stride_bn,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        K=K,
        OUT_DTYPE=out_dtype_tl,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        try:
            p = os.path.abspath(__file__)
            if os.path.exists(p):
                return {"program_path": p}
        except Exception:
            pass
        try:
            src = inspect.getsource(inspect.getmodule(Solution))
            return {"code": src}
        except Exception:
            src = inspect.getsource(inspect.currentframe())
            return {"code": src}