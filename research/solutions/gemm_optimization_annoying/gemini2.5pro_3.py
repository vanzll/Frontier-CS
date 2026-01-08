import torch
import triton
import triton.language as tl

# The code string to be returned by the solve method
# This contains the full implementation of the matmul kernel
KERNEL_CODE = """
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    \"\"\"
    GELU activation function as specified in the problem.
    This uses the error function (erf) from the CUDA libdevice library.
    \"\"\"
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476)) # 0.707... = 1/sqrt(2)

@triton.autotune(
    configs=[
        # Basic configurations with varying block sizes
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        
        # Configurations with smaller block sizes for K, potentially better for smaller K dimensions
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}),

        # Configurations with smaller block sizes for M and N, good for non-multiple-of-large-block dimensions
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 4}),

        # Configurations with minimal warps/stages, can be efficient for some shapes
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 2}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    \"\"\"
    Triton kernel for matrix multiplication with GELU activation.
    - Uses a 1D launch grid and grouping to improve L2 cache locality.
    - Handles arbitrary matrix shapes and strides using masking.
    - Accumulates in float32 for precision.
    - Applies GELU activation before storing the result.
    \"\"\"
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    
    # Grouping programs for L2 cache reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    # Create block-level offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create pointers to the first elements of the blocks
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize accumulator with zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Define masks for handling non-multiples of block sizes
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & ((offs_k[None, :] + k) < K)
        b_mask = ((offs_k[:, None] + k) < K) & (offs_n[None, :] < N)

        # Load blocks of A and B from global memory
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Perform matrix multiplication for the blocks and accumulate the result
        accumulator += tl.dot(a, b, allow_tf32=True)

        # Advance pointers for the next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply GELU activation to the accumulated result
    result = gelu(accumulator)

    # Define pointers and masks for storing the final result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Store the result block to global memory
    tl.store(c_ptrs, result, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Matrix multiplication with GELU activation.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    \"\"\"
    # Shape validation
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions for matmul"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on a CUDA device"
    assert a.dtype == b.dtype, "Input tensors must have the same dtype"

    M, K = a.shape
    K, N = b.shape

    # Create the output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define the launch grid
    # The grid is 1D, and the kernel internally maps the 1D program ID to 2D block coordinates
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch the kernel
    gemm_gelu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )

    return c
"""

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict with either:
        - {"code": "python_code_string"}
        - {"program_path": "path/to/kernel.py"}
        """
        return {"code": KERNEL_CODE}