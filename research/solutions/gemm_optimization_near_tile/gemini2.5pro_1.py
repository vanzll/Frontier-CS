import torch
import triton
import triton.language as tl

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict with either:
        - {"code": "python_code_string"}
        - {"program_path": "path/to/kernel.py"}
        """
        matmul_kernel_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    \"\"\"
    Computes the GeLU activation function as specified.
    \"\"\"
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        # Basic configurations for various shapes
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 5, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 5, 'num_warps': 8}),
        
        # Configurations with smaller K block size
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 4}),

        # Configurations with larger K block size
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 4}),
        
        # Configurations for smaller matrices
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
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
):
    \"\"\"
    Triton kernel for matrix multiplication with GELU activation.
    This kernel is autotuned for various tile sizes and configurations.
    It uses a 2D grid of thread blocks and handles arbitrary matrix shapes
    and memory layouts through masking and stride parameters.
    \"\"\"
    # -----------------------------------------------------------
    # Map program ids to tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # -----------------------------------------------------------
    # Create offsets for the tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # -----------------------------------------------------------
    # Initialize pointers to A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Initialize accumulator with float32 for precision
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -----------------------------------------------------------
    # Main loop over the K dimension
    for k in range(0, K, BLOCK_K):
        # Create a mask for the K dimension to handle non-multiples of BLOCK_K
        k_mask = (k + offs_k) < K
        
        # Load tiles from A and B, applying masks for boundary conditions
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        
        # Perform the matrix multiplication on the tiles
        accumulator += tl.dot(a, b)
        
        # Advance pointers for the next iteration of the K loop
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # -----------------------------------------------------------
    # Apply GELU activation to the accumulator
    accumulator = gelu(accumulator)

    # -----------------------------------------------------------
    # Write the result to the C matrix
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Cast accumulator to output dtype and store
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Matrix multiplication with GELU activation.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    \"\"\"
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    
    M, K = a.shape
    _, N = b.shape
    
    # Allocate output tensor
    # The output dtype must match the input dtype for correctness checks.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Grid for kernel launch using a 2D grid of thread blocks
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    # Launch the autotuned kernel
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c
"""
        return {"code": matmul_kernel_code}