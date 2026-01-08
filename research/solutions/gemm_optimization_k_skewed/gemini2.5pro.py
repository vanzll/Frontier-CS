import torch
import triton
import triton.language as tl
import inspect


@triton.jit
def gelu(x):
    """
    GeLU activation function, as required by the problem specification.
    This is the exact implementation of the fast GeLU approximation.
    x * 0.5 * (1.0 + erf(x / sqrt(2)))
    """
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        # Basic configurations, balanced for general purpose
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_stages': 4, 'num_warps': 4}),
        
        # Configurations potentially good for large matrices
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'num_stages': 3, 'num_warps': 8}),

        # Configurations for small K dimension, using larger K tiles
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 256, 'num_stages': 3, 'num_warps': 4}),

        # Configurations for large K dimension, using smaller K tiles and more stages
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_stages': 5, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'num_stages': 4, 'num_warps': 4}),

        # Additional configurations for diversity
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K', 'IS_EVEN_K_32', 'IS_EVEN_K_64'],
)
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_EVEN_K_32: tl.constexpr, IS_EVEN_K_64: tl.constexpr
):
    """
    Triton kernel for GEMM with GeLU activation.
    This kernel is autotuned for various tile sizes, number of stages, and warps.
    It handles arbitrary matrix shapes and memory layouts through masking and strides.
    """
    # 1. Thread block identification
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 2. Pointers for the first tile
    # Pointers for block of A
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    # Pointers for block of B
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    # Pointers for K dimension
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. Accumulator initialization
    # Use float32 for accumulator to maintain precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 4. Main loop over K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate current K offset
        k_offset = k * BLOCK_SIZE_K
        
        # Create masks for K dimension to handle cases where K is not a multiple of BLOCK_SIZE_K
        k_mask = (k_offset + offs_k) < K
        
        # Load tiles of A and B from global memory with padding (other=0.0)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        
        # Perform matrix multiplication and accumulate
        accumulator += tl.dot(a, b, allow_tf32=True)
        
        # Advance pointers for the next iteration of the K loop
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 5. Apply GELU activation
    accumulator = gelu(accumulator)

    # 6. Store the result tile to C
    # Pointers for block of C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Create masks for M and N dimensions to handle cases where they are not multiples of block sizes
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Safely store the result
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication with GELU activation.
    
    This function serves as a wrapper for the Triton JIT kernel `gemm_kernel`.
    It sets up the problem size, allocates the output tensor, defines the launch grid,
    and calls the optimized kernel.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    """
    # Ensure inputs are on the same CUDA device
    assert a.device.type == 'cuda' and b.device.type == 'cuda', "Inputs must be CUDA tensors"
    assert a.device == b.device, "Inputs must be on the same device"

    # Get matrix dimensions
    M, K = a.shape
    _, N = b.shape

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define the launch grid for the kernel
    # The grid is 1D, and each program instance computes one tile of the output matrix C.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch the kernel
    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # Pass constants for autotuner hint
        IS_EVEN_K_32=(K % 32 == 0),
        IS_EVEN_K_64=(K % 64 == 0),
    )

    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dict with the Python code for the solution.
        The evaluator will execute this code.
        """
        # This approach uses `inspect` to reliably get the source code of the current module.
        # It's a standard pattern in programming challenges that require code submission as a string.
        source_code = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        return {"code": source_code}