class Solution:
    def solve(self, spec_path: str = None) -> dict:
        kernel_code_string = """
import torch
import triton
import triton.language as tl
import os

# This is a common workaround for Triton to find the CUDA libdevice path in some environments.
if "LIBDEVICE_PATH" not in os.environ:
    for path in [
        "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
        "/opt/cuda/nvvm/libdevice/libdevice.10.bc"
    ]:
        if os.path.exists(path):
            os.environ["LIBDEVICE_PATH"] = path
            break
if "LIBDEVICE_PATH" in os.environ:
    tl.extra.cuda.LIBDEVICE_PATH = os.environ["LIBDEVICE_PATH"]

@triton.jit
def gelu(x):
    \"\"\"
    GELU activation function as specified in the problem.
    This uses the error function (erf) for a smooth approximation.
    \"\"\"
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        # Basic configurations with balanced block sizes
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 4}),
        
        # Configurations biased towards larger M (tall C matrix)
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 4}),

        # Configurations biased towards larger N (wide C matrix)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 4}),
        
        # Configurations with larger K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 4}),

        # Configurations for smaller problems (smaller K, fewer stages/warps)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_stages': 2, 'num_warps': 4}),

        # Some high-performance configs from tutorials
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K', 'a_stride_m', 'a_stride_k', 'b_stride_k', 'b_stride_n'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_stride_m, a_stride_k,
    b_stride_k, b_stride_n,
    c_stride_m, c_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    \"\"\"
    Triton kernel for GEMM with GELU activation.
    This kernel is optimized using block tiling, autotuning, and grouped ordering for L2 cache efficiency.
    \"\"\"
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # Re-order program IDs to improve L2 cache performance by grouping blocks.
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Pointers to the first tile of A and B
    rm_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk_offsets = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (rm_offsets[:, None] * a_stride_m + rk_offsets[None, :] * a_stride_k)
    b_ptrs = b_ptr + (rk_offsets[:, None] * b_stride_k + rn_offsets[None, :] * b_stride_n)

    # Accumulator initialization
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Compute K offsets for masking
        k_offsets = k * BLOCK_K + rk_offsets
        
        # Create masks for safe loading from A and B
        a_mask = (rm_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (rn_offsets[None, :] < N)
        
        # Load tiles from global memory
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Perform matrix multiplication and accumulate
        accumulator += tl.dot(a_tile, b_tile, allow_tf32=True)

        # Advance pointers to the next tile in the K dimension
        a_ptrs += BLOCK_K * a_stride_k
        b_ptrs += BLOCK_K * b_stride_k

    # Apply GELU activation to the accumulated result
    accumulator = gelu(accumulator)
    
    # Cast the result to the output tensor's data type
    c_tile = accumulator.to(c_ptr.dtype.element_ty)
    
    # Pointers and masks for storing the final result to C
    c_ptrs = c_ptr + rm_offsets[:, None] * c_stride_m + rn_offsets[None, :] * c_stride_n
    c_mask = (rm_offsets[:, None] < M) & (rn_offsets[None, :] < N)
    
    tl.store(c_ptrs, c_tile, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Matrix multiplication with GELU activation.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    \"\"\"
    assert a.shape[1] == b.shape[0], "Matrix multiplication requires inner dimensions to match."
    
    M, K = a.shape
    _, N = b.shape
    
    # Allocate the output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define the grid for launching the kernel
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    # Launch the Triton kernel
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c
"""
        return {"code": kernel_code_string}