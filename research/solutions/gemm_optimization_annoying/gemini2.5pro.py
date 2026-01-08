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
    GELU activation function, as required by the problem specification.
    \"\"\"
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        # Basic configurations for a variety of shapes
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        
        # Configurations with larger K block size for deep K dimensions
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        
        # Configurations targeting wider N blocks (often good for memory layout)
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        
        # Configurations targeting taller M blocks
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        
        # A "best guess" configuration based on typical modern GPU GEMM tuning
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Re-order grid for better L2 cache locality (grouped scheduling).
    # The grid is divided into groups of GROUP_SIZE_M rows.
    # Programs in the same group are scheduled together to improve reuse of B matrix tiles.
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    
    # Each program computes a tile of C. Find its location in the grid.
    pid_m = group_id * GROUP_SIZE_M + (pid % width) // grid_n
    pid_n = (pid % width) % grid_n

    # Create pointers for the first blocks of A and B.
    # We will advance these pointers as we loop over the K dimension.
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator with zeros.
    # Accumulator is in float32 to maintain precision during summation.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in blocks of BLOCK_SIZE_K.
    for k in range(0, K, BLOCK_SIZE_K):
        # Boundary checks for K are handled by masking.
        current_k_offs = k + offs_k
        
        # Load the next blocks of A and B from global memory.
        # Masking is used to handle matrices where dimensions are not
        # perfectly divisible by the block sizes.
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (current_k_offs[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(current_k_offs[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        
        # Perform the matrix multiplication on the loaded blocks
        # and accumulate the result. tl.dot uses Tensor Cores.
        accumulator += tl.dot(a, b)

        # Advance the pointers to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Apply the GELU activation function to the final accumulator.
    accumulator = gelu(accumulator)

    # Write the final result block to C.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Convert accumulator to the output dtype before storing.
    tl.store(c_ptrs, accumulator.to(C.dtype.element_ty), mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Matrix multiplication with GELU activation.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    \"\"\"
    # Check constraints for the kernel.
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions for multiplication"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on a CUDA device"
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous for best performance"
    
    M, K = a.shape
    K, N = b.shape
    
    # Allocate the output tensor.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define the grid for launching the kernel.
    # Each program instance computes a block of the output matrix C.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch the Triton kernel.
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c
"""
        return {"code": kernel_code}