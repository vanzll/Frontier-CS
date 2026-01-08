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
    GELU activation function as specified in the problem.
    This implementation uses the erf function from libdevice.
    \"\"\"
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.autotune(
    configs=[
        # Basic configurations for various tile shapes
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_stages': 4, 'num_warps': 4}),
        
        # Configurations with larger block sizes for larger matrices
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8}),

        # Configurations with more stages for better latency hiding
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'num_stages': 5, 'num_warps': 4}),

        # Configurations with larger K block size
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'num_stages': 3, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_gelu_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    \"\"\"
    Triton kernel for matrix multiplication with GELU activation.
    This kernel is autotuned for various block sizes and scheduling parameters.
    It handles arbitrary matrix shapes using boundary checks.
    \"\"\"
    # Each program instance computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of C.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Create pointers to the first tiles of A and B for this program instance.
    a_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0)
    )

    # Initialize the accumulator with zeros. Use float32 for precision.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension, tiling it by BLOCK_SIZE_K.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles from A and B from global memory.
        # boundary_check handles non-multiple-of-block-size dimensions.
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        
        # Perform the matrix multiplication on the tiles.
        # allow_tf32=True enables TensorFloat-32 for float32 inputs on compatible hardware.
        accumulator += tl.dot(a, b, allow_tf32=True)
        
        # Advance pointers to the next K-dimension tile.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    # Apply the GELU activation function to the accumulator.
    accumulator = gelu(accumulator)
    
    # Cast the result back to the output tensor's data type.
    c = accumulator.to(C_ptr.dtype.element_ty)

    # Create pointer to the C matrix tile and store the result.
    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Matrix multiplication with GELU activation.
    
    This function serves as a wrapper for the Triton JIT kernel. It prepares
    the tensors, defines the launch grid, and calls the optimized kernel.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    \"\"\"
    # Input validation
    assert a.shape[1] == b.shape[0], "incompatible matrix dimensions"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on a CUDA device"
    
    M, K = a.shape
    _, N = b.shape
    
    # Create the output tensor on the same device and with the same dtype as input 'a'.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # The launch grid is 2D. Each program in the grid computes one tile of C.
    # The grid size is determined by the autotuner's chosen block sizes.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch the Triton kernel.
    _matmul_gelu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c
"""
        return {"code": code}