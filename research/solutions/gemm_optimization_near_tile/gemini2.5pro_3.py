import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        kernel_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    \"\"\"
    GELU activation function, as specified in the problem.
    This is equivalent to x * Phi(x) where Phi is the CDF of the normal distribution.
    \"\"\"
    # Using the constant for 1/sqrt(2)
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        # Basic configurations for various tile sizes and aspect ratios
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        # Configurations with larger K blocks for K-bound problems
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 8}),
        # Configurations with fewer stages (less register pressure)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 4}),
        # More balanced configurations
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 8}),
        # Configuration for larger shapes
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn'],
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
    num_stages: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouping thread blocks for better L2 cache locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create offsets for the M, N, and K dimensions
    offs_m_arange = tl.arange(0, BLOCK_M)
    offs_n_arange = tl.arange(0, BLOCK_N)
    offs_k_arange = tl.arange(0, BLOCK_K)

    offs_m = pid_m * BLOCK_M + offs_m_arange
    offs_n = pid_n * BLOCK_N + offs_n_arange
    
    # Pointers to the first elements of the A and B blocks
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k_arange[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_arange[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize accumulator with float32 for precision
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over the K dimension, with software pipelining managed by num_stages
    for k in range(0, K, BLOCK_K):
        # Create masks to handle tail cases in the K dimension
        a_mask = (k + offs_k_arange[None, :]) < K
        b_mask = (k + offs_k_arange[:, None]) < K
        
        # Load tiles from A and B matrices
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Perform matrix multiplication and accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers to the next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Apply GELU activation to the accumulated result
    c = gelu(accumulator)

    # Pointers to the C matrix for storing the result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    
    # Create masks to handle tail cases in M and N dimensions
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Store the final result, casting back to the output tensor's dtype
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Matrix multiplication with GELU activation.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    \"\"\"
    M, K = a.shape
    _, N = b.shape
    
    # Allocate the output tensor, matching the input dtype
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define the grid for launching the kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    # Launch the Triton kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c
"""
        return {"code": kernel_code}