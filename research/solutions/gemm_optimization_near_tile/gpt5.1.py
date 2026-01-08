import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


_matmul_configs = [
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
        num_stages=3,
        num_warps=4,
    ),
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 2},
        num_stages=5,
        num_warps=8,
    ),
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 2},
        num_stages=5,
        num_warps=8,
    ),
    triton.Config(
        {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 2},
        num_stages=5,
        num_warps=8,
    ),
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 2},
        num_stages=5,
        num_warps=8,
    ),
]


def _make_matmul_kernel(dtype):
    @triton.autotune(
        configs=_matmul_configs,
        key=['M', 'N', 'K', 'a_stride_am', 'a_stride_ak', 'b_stride_bk', 'b_stride_bn'],
    )
    @triton.jit
    def kernel(
        a_ptr: tl.pointer_type(dtype),
        b_ptr: tl.pointer_type(dtype),
        c_ptr: tl.pointer_type(dtype),
        M, N, K,
        a_stride_am, a_stride_ak,
        b_stride_bk, b_stride_bn,
        c_stride_cm, c_stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        rm = offs_m < M
        rn = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_offsets = k + offs_k
            k_mask = k_offsets < K

            a_ptrs = a_ptr + offs_m[:, None] * a_stride_am + k_offsets[None, :] * a_stride_ak
            b_ptrs = b_ptr + k_offsets[:, None] * b_stride_bk + offs_n[None, :] * b_stride_bn

            a_mask = rm[:, None] & k_mask[None, :]
            b_mask = k_mask[:, None] & rn[None, :]

            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            acc += tl.dot(a, b)

        c_tile = gelu(acc).to(dtype)

        c_ptrs = c_ptr + offs_m[:, None] * c_stride_cm + offs_n[None, :] * c_stride_cn
        c_mask = rm[:, None] & rn[None, :]
        tl.store(c_ptrs, c_tile, mask=c_mask)

    return kernel


_matmul_kernel_fp16 = _make_matmul_kernel(tl.float16)
_matmul_kernel_bf16 = _make_matmul_kernel(tl.bfloat16)
_matmul_kernel_fp32 = _make_matmul_kernel(tl.float32)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be 2D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes for multiplication")
    if a.device.type != 'cuda' or b.device.type != 'cuda':
        raise ValueError("Inputs must be CUDA tensors")
    if a.dtype != b.dtype:
        raise ValueError("Input dtypes must match")

    M, K = a.shape
    Kb, N = b.shape
    if Kb != K:
        raise ValueError("Incompatible matrix shapes for multiplication")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    a_stride_am, a_stride_ak = a.stride()
    b_stride_bk, b_stride_bn = b.stride()
    c_stride_cm, c_stride_cn = c.stride()

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    if a.dtype == torch.float16:
        kernel = _matmul_kernel_fp16
    elif a.dtype == torch.bfloat16:
        kernel = _matmul_kernel_bf16
    elif a.dtype == torch.float32:
        kernel = _matmul_kernel_fp32
    else:
        raise TypeError("Unsupported dtype: {}".format(a.dtype))

    kernel[grid](
        a, b, c,
        M, N, K,
        a_stride_am, a_stride_ak,
        b_stride_bk, b_stride_bn,
        c_stride_cm, c_stride_cn,
    )

    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"program_path": __file__}