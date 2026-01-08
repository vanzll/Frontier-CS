import torch
import triton
import triton.language as tl


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        python_code_string = r"""
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    """
    GELU activation function, as specified in the problem.
    This uses the erf approximation.
    """
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.autotune(
    configs=[
        # Basic configurations with varying block sizes and stages
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 16, 'num_stages': 4, 'num_warps': 4}),
        
        # Configurations with a larger K block size
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8, 'num_stages': 3, 'num_warps': 8}),
        
        # Configurations with a smaller K block size
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 16, 'num_stages': 3, 'num_warps': 4}),

        # Configuration without grouping for comparison
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 1, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn', 'stride_cm', 'stride_cn'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    """
    Triton kernel for matrix multiplication with GELU activation.
    - Tiled execution to leverage shared memory and reduce global memory traffic.
    - Autotuned over different block sizes, stages, and warps.
    - Handles arbitrary matrix sizes and memory layouts (strides).
    - Accumulates in fp32 for precision, casts to output dtype before storing.
    - Implements grouping of thread blocks for better L2 cache locality.
    """
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # Re-order program IDs to improve L2 cache performance of matrix A
    width = GROUP_M * grid_n
    group_id = pid // width
    group_rank = pid % width
    group_offset = group_id * GROUP_M
    pid_m = group_offset + (group_rank // grid_n)
    pid_n = group_rank % grid_n

    # Pointers to the first blocks of A and B
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize accumulator with zeros
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, handling tail K dimension
        a_mask = (offs_m[:, None] < M) & ((k * BLOCK_K + offs_k[None, :]) < K)
        b_mask = ((k * BLOCK_K + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Perform the matrix multiplication on the blocks
        accumulator += tl.dot(a, b, allow_tf32=True)

        # Advance the pointers to the next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply GELU activation to the accumulated result
    c_activated = gelu(accumulator)
    
    # Cast back to the output tensor's dtype before storing
    c_activated = c_activated.to(c_ptr.dtype.element_ty)

    # Pointers and mask for writing the final result block to C
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c_activated, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.
    
    This function serves as a wrapper for the Triton JIT kernel. It sets up
    the problem, allocates the output tensor, and launches the kernel.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    """
    M, K = a.shape
    _, N = b.shape
    
    assert a.shape[1] == b.shape[0], "incompatible matrix dimensions"
    assert a.is_cuda and b.is_cuda, "input tensors must be on a CUDA device"
    
    # Allocate the output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define the grid for kernel launch
    # The grid is 1D, and the kernel re-arranges it into a 2D grid with grouping
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

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
        return {"code": python_code_string}