import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 2}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 2}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    A, B, C,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    if SPLIT_K > 1:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    
    A_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        a = tl.load(A_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K * SPLIT_K, other=0.0)
        b = tl.load(B_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K * SPLIT_K, other=0.0)
        accumulator += tl.dot(a, b)
        A_ptrs += BLOCK_K * SPLIT_K * stride_ak
        B_ptrs += BLOCK_K * SPLIT_K * stride_bk
    
    if SPLIT_K > 1:
        accumulator = accumulator.to(tl.float32)
        C_ptrs = C + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
        tl.atomic_add(C_ptrs, accumulator, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))
    else:
        c = gelu(accumulator).to(A.dtype.element_ty)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(C_ptrs, c, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Both inputs must be 2D tensors")
    
    device = a.device
    dtype = a.dtype
    
    if device.type != 'cuda':
        raise ValueError("Input tensors must be on CUDA device")
    
    c = torch.empty((M, N), device=device, dtype=dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        META['SPLIT_K'],
    )
    
    _matmul_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        M, N, K,
        SPLIT_K=1,
    )
    
    return c

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": open(__file__).read()}