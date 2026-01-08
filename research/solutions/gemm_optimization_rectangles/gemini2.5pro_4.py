import torch
import triton
import triton.language as tl

# The code for the kernel and its wrapper is defined in this string.
# This string is returned by the `solve` method of the `Solution` class.
# It is also executed to define the `matmul` function in the current module's namespace,
# as required by the evaluation environment.
_GEMM_KERNEL_CODE = """
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    \"\"\"
    Required GELU implementation using the error function (erf).
    This is called on the accumulator before storing the result.
    \"\"\"
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.autotune(
    configs=[
        # Configurations targeting tall/skinny matrices (M >> N)
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 8}),
        
        # Configurations targeting short/wide matrices (N >> M)
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 8}),
        
        # Configurations for more square-like matrices
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        
        # High-performance configurations inspired by Triton's own benchmarks
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    \"\"\"
    Triton kernel for GEMM with GELU activation. It is autotuned for various block sizes and 
    configurations. It uses a 1D launch grid with grouping to improve L2 cache locality, which is
    beneficial for large matrices. TF32 is enabled on the dot product for performance on Ampere+ GPUs.
    \"\"\"
    # -----------------------------------------------------------
    # Map program ids to blocks of C
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouping for L2 cache locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_m_block = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_block = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_range = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m_block[:, None] * stride_am + offs_k_range[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_range[:, None] * stride_bk + offs_n_block[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # -----------------------------------------------------------
    # Loop over K in blocks of BLOCK_SIZE_K
    for k in range(0, K, BLOCK_SIZE_K):
        # Boundary checks
        a_mask = (offs_m_block[:, None] < M) & ((k + offs_k_range)[None, :] < K)
        b_mask = ((k + offs_k_range)[:, None] < K) & (offs_n_block[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix-multiply-accumulate
        accumulator += tl.dot(a, b, allow_tf32=True)
        
        # Advance pointers to next blocks in K dimension
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Apply GELU activation and store the result
    result = gelu(accumulator)
    
    c_ptrs = c_ptr + (offs_m_block[:, None] * stride_cm + offs_n_block[None, :] * stride_cn)
    c_mask = (offs_m_block[:, None] < M) & (offs_n_block[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Matrix multiplication with GELU activation.
    
    This function acts as a wrapper for the Triton JIT kernel. It validates
    inputs, allocates the output tensor, and launches the kernel.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N) with GELU activation applied
    \"\"\"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Incompatible dimensions"
    
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 1D launch grid to enable grouping for L2 cache locality.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch the autotuned kernel.
    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
"""

# Execute the kernel code string to define the functions in the module's scope
exec(_GEMM_KERNEL_CODE)

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns a dictionary containing the Python code for the GEMM kernel
        and its wrapper function as a string.
        """
        return {"code": _GEMM_KERNEL_CODE}