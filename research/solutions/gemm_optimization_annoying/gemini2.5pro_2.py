import torch
import triton
import triton.language as tl
import inspect

# Required GELU implementation from the problem specification
@triton.jit
def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    This is an approximation of the exact GELU function.
    """
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        # Basic configurations covering different tile shapes
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        
        # Larger tile configurations
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        
        # Smaller configurations, potentially better for awkward shapes
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton Kernel for Matrix Multiplication with fused GELU activation.
    - It uses tiling to maximize SRAM usage and minimize global memory access.
    - It uses grouping of thread blocks to improve L2 cache locality.
    - It accumulates in float32 for precision and then casts to output type.
    - It handles arbitrary matrix shapes (non-power-of-two) using masking.
    - `allow_tf32=True` is used for `tl.dot` to leverage Tensor Cores on supported hardware.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Re-order program IDs for better L2 cache performance (grouping)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    # Pointers to the first element of the tiles
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k_init = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_init[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_init[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator with zeros. Use float32 for precision.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate offsets for the current K-block
        current_k_offsets = k * BLOCK_SIZE_K + offs_k_init

        # Load tiles of a and b from global memory
        # Masking handles cases where matrix dimensions are not multiples of block sizes
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (current_k_offsets[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(current_k_offsets[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        
        # Perform matrix multiplication on tiles
        accumulator += tl.dot(a, b, allow_tf32=True)
        
        # Advance pointers to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    # Apply GELU activation
    # Cast accumulator to the same type as output tensor before storing
    c = gelu(accumulator.to(c_ptr.dtype.element_ty))

    # Pointers to the output tile
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Mask for storing the final result tile
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
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define the grid for launching the kernel
    def grid(meta):
        # The grid is 1D, and the kernel internally re-arranges it into a 2D grid with grouping
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)

    # Launch the Triton kernel
    gemm_gelu_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        M, N, K,
    )
    
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict containing the source code of the solution.
        The evaluator will execute this file and call this method.
        """
        # Get the source code of the current Python module
        module_source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        return {"code": module_source}