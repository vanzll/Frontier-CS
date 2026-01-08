import torch
import triton
import triton.language as tl

_KERNEL_CODE = """
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    \"\"\"
    GELU activation function as specified.
    This is equivalent to `x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))`
    but uses the libdevice implementation for compatibility.
    \"\"\"
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.autotune(
    configs=[
        # Basic square tiles
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),

        # Configurations for tall/skinny matrices (M > N)
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),

        # Configurations for short/wide matrices (M < N)
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),

        # Other general-purpose configurations
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        
        # Configs with larger K blocking
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2),
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    \"\"\"
    Triton kernel for GEMM with GELU activation.
    - Tiled matrix multiplication for performance.
    - Autotuned over block sizes, stages, and warps.
    - Grouped launch schedule to improve L2 cache hit rate for the B matrix.
    - Handles arbitrary matrix shapes and strides with masking.
    - Fuses GELU activation to avoid a separate kernel launch.
    \"\"\"
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Re-order program ID for better L2 cache performance (grouped scheduling)
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Pointers to the first element of the block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs_base = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs_base = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator for the output block, initialized to zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute pointers for the current K-block
        k_offset = k * BLOCK_SIZE_K
        a_ptrs = a_ptrs_base + k_offset * stride_ak
        b_ptrs = b_ptrs_base + k_offset * stride_bk
        
        # Create masks to handle matrix boundaries
        a_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
        b_mask = ((k_offset + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        
        # Load A and B tiles from global memory
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Perform the matrix multiplication and accumulate the result
        accumulator += tl.dot(a, b)

    # Apply GELU activation
    accumulator = gelu(accumulator)

    # Cast accumulator to the output tensor's data type
    c = accumulator.to(c_ptr.dtype.element_ty)

    # Write the result block back to global memory
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
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
    assert a.shape[1] == b.shape[0], "incompatible dimensions for matrix multiplication"
    assert a.is_cuda and b.is_cuda, "input tensors must be on a CUDA device"
    
    M, K = a.shape
    _, N = b.shape
    
    # Allocate the output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define the grid for kernel launch
    # The grid is 1D and remapped inside the kernel for better L2 cache locality
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch the kernel
    gemm_gelu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
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
        return {"code": _KERNEL_CODE}