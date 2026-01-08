import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = """
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    \"\"\"
    GeLU activation function, as specified in the problem.
    This function is applied element-wise to the output of the matrix multiplication.
    \"\"\"
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        # Basic configurations, good for a wide range of sizes
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_stages': 4, 'num_warps': 4}),
        
        # Configurations with larger BLOCK_SIZE_K, helpful for larger K dimensions
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_stages': 4, 'num_warps': 4}),

        # Configurations with even larger tiles, good for very large matrices on modern GPUs
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 8}),
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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    
    # This kernel computes a block of the output matrix C.
    # The grid is 2D, with each program instance handling one block.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute pointers for the first tile of A and B
    # `offs_m` and `offs_n` are the row and column offsets for the C tile
    # Since all evaluation shapes are multiples of 256, and our block sizes
    # are powers of two <= 256, no boundary checks (masking) are needed.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize accumulator with zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension, accumulating the product of A and B tiles
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles of A and B from global memory
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # Perform the matrix multiplication of the tiles and accumulate the result
        # `allow_tf32=True` enables the use of Tensor Cores for float32 inputs on Ampere+ GPUs
        accumulator += tl.dot(a, b, allow_tf32=True)
        
        # Advance pointers to the next K-block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    # Apply the GELU activation function to the accumulated result
    c_tile = gelu(accumulator)

    # Compute pointers to the C output tile
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    
    # Store the final result tile back to global memory
    tl.store(c_ptrs, c_tile)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Matrix multiplication with GELU activation.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    \"\"\"
    # Check for valid dimensions
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions for multiplication"
    
    M, K = a.shape
    _, N = b.shape
    
    # Allocate the output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define the grid for launching the kernel
    # The grid is a 2D grid of thread blocks.
    # Each block computes a tile of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # The `meta` argument is provided by the autotuner to get the chosen block sizes.
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    # Launch the Triton kernel
    # The autotuner will try all configs and pick the best one for the given M, N, K.
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c
"""
        return {"code": code.strip()}