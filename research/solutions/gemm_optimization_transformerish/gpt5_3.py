import os
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_gelu_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_remaining = K - k_iter
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        k_iter += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply GELU in float32
    acc = gelu(acc)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not (a.is_cuda and b.is_cuda):
        raise ValueError("Inputs must be CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be 2D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix shapes")

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Supported dtypes: float16, bfloat16, float32
    if a.dtype != b.dtype:
        raise ValueError("Input dtypes must match")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Unsupported dtype: {}".format(a.dtype))

    out_dtype = a.dtype
    # Triton kernel will store in float32; cast if needed after
    # However triton supports storing float32 into destination pointer expecting float32 dtype
    # To store as original dtype, we allocate output in the same dtype and rely on implicit cast via tl.store
    # acc (float32) will be cast to output dtype automatically by tl.store.

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    # Strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Grid function for autotuning
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    allow_tf32 = (a.dtype == torch.float32)

    _matmul_gelu_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        allow_tf32,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        try:
            return {"program_path": os.path.abspath(__file__)}
        except NameError:
            # Fallback when __file__ is not defined (e.g., in interactive environments)
            import inspect
            return {"code": inspect.getsource(type(self).__module__)}