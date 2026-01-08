import torch
import triton
import triton.language as tl
import inspect
import textwrap


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


def _gelu_torch(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4},
            num_stages=4,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
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
    FP16_OUTPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    group_id = pid // (grid_n * group_size)
    first_pid_m = group_id * group_size
    group_size_m = tl.minimum(grid_m - first_pid_m, group_size)
    pid_in_group = pid % (grid_n * group_size)
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_mask = k + offs_k < K
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32)

        k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = gelu(acc)

    if FP16_OUTPUT:
        c = c.to(tl.float16)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes")
    if a.device != b.device:
        raise ValueError("Input tensors must be on the same device")

    M, K = a.shape
    Kb, N = b.shape
    if K != Kb:
        raise ValueError("Inner dimensions must match")

    if not a.is_cuda or not b.is_cuda:
        return _gelu_torch(a @ b)

    if a.dtype != b.dtype:
        # Promote to float32 if dtypes differ
        a = a.to(torch.float32)
        b = b.to(torch.float32)

    out_dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)

    FP16_OUTPUT = out_dtype in (torch.float16, torch.bfloat16)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )

    _matmul_kernel[grid](
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
        FP16_OUTPUT=FP16_OUTPUT,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        gelu_src = textwrap.dedent(inspect.getsource(gelu))
        gelu_torch_src = textwrap.dedent(inspect.getsource(_gelu_torch))
        kernel_src = textwrap.dedent(inspect.getsource(_matmul_kernel))
        matmul_src = textwrap.dedent(inspect.getsource(matmul))

        code = "\n".join(
            [
                "import torch",
                "import triton",
                "import triton.language as tl",
                "",
                gelu_src.rstrip(),
                "",
                gelu_torch_src.rstrip(),
                "",
                kernel_src.rstrip(),
                "",
                matmul_src.rstrip(),
                "",
            ]
        )
        return {"code": code}