import torch

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {
            "code": r"""
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    """
    Applies GELU activation function.
    Uses the approximation: 0.5 * x * (1 + erf(x / sqrt(2)))
    1/sqrt(2) is approximately 0.7071067811865476
    """
    return x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))

@triton.autotune(
    configs=[
        # Configs for large matrices / large K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # Configs optimized for small K (less pipeline stages needed, smaller tiles)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing the matrix multiplication C = A x B with GELU activation.
    """
    pid = tl.program_id(axis=0)
    
    # Grid dimensions
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzling to improve L2 Cache Hit Rate
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute offsets for A and B
    # Use modulo M/N to ensure memory safety for unmasked loads (padding behavior)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Initial pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Calculate remaining elements in K dimension for masking
        k_remaining = K - k * BLOCK_K
        
        # We must mask K dimension to avoid including garbage data in the sum
        # Ideally, if K is a multiple of BLOCK_K, mask is all True, but logic is generic
        load_mask = offs_k < k_remaining
        
        # Load blocks of A and B
        a = tl.load(a_ptrs, mask=load_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=load_mask[:, None], other=0.0)
        
        # Accumulate dot product
        accumulator = tl.dot(a, b, accumulator)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply GELU activation to the result
    accumulator = gelu(accumulator)
    
    # Convert to output precision (float16)
    c = accumulator.to(tl.float16)
    
    # Compute offsets for C
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Mask for C store to ensure we only write within bounds
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    """
    # Check constraints
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"K dimension mismatch: {K} != {K2}"
    
    # Allocating output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Define grid function
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    
    return c
"""
        }