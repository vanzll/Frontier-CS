import torch
import triton
import triton.language as tl

# Final implementation of the Solution class
class Solution:
    def solve(self, spec_path: str = None) -> dict:
        """
        Returns the Python code for the GEMM kernel as a string.
        """
        # The entire Python code for the solution is embedded as a string.
        # This includes all necessary imports, the GELU function, the Triton kernel,
        # the matmul wrapper, and this Solution class itself.
        # The evaluator will execute this file and call the solve() method.

        # It's a common pattern in such platforms to read the current file's content.
        # However, embedding the code directly is more robust and self-contained.
        code = """
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    \"\"\"
    GELU activation function using the error function (erf), as specified.
    \"\"\"
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        # Basic configurations with varying block sizes
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        
        # Configurations with smaller BLOCK_SIZE_K for potentially better parallelism
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 8}),
        
        # Configurations with larger tile sizes
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    \"\"\"
    Triton kernel for matrix multiplication with fused GELU activation.
    Computes C = GELU(A * B).
    - Tiling is used to load blocks of A and B into SRAM.
    - A 1D launch grid is mapped to a 2D grid of blocks in C.
    - Grouped ordering is used to improve L2 cache reuse for matrix A.
    - The accumulation is done in float32 for precision.
    - Boundary checks are performed with masking for all loads and stores.
    \"\"\"
    # -----------------------------------------------------------
    # Map program ids to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouped ordering for L2 cache locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Pre-calculate masks for M and N dimensions, which are loop-invariant
    m_mask = offs_m < M
    n_mask = offs_n < N

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Mask for the K dimension, which changes each iteration
        k_remaining = K - k * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining
        
        # Combine with M/N masks for safe loading
        a_load_mask = m_mask[:, None] & k_mask[None, :]
        b_load_mask = k_mask[:, None] & n_mask[None, :]
        
        a = tl.load(a_ptrs, mask=a_load_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_load_mask, other=0.0)
        
        accumulator += tl.dot(a, b)
        
        # Advance pointers to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply GELU activation to the accumulator
    activated_accumulator = gelu(accumulator)
    c = activated_accumulator.to(c_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C.
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = m_mask[:, None] & n_mask[None, :]
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
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    M, K = a.shape
    _, N = b.shape
    
    # Allocates output tensor C
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 1D launch grid. The kernel itself handles the 2D mapping.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch the kernel
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c
    
class Solution:
    def solve(self, spec_path: str = None) -> dict:
        # This implementation simply returns the source code of the functions above.
        # A more complex implementation might involve dynamically generating code
        # or loading from a separate file.
        import inspect
        
        # Get the source code of all relevant functions and classes in this scope
        # and concatenate them.
        functions_to_include = [gelu, _matmul_kernel, matmul, Solution]
        full_code = "import torch\\nimport triton\\nimport triton.language as tl\\n\\n"
        for func in functions_to_include:
            full_code += inspect.getsource(func)
            full_code += "\\n\\n"
            
        # In a real contest, returning the file content or a direct string is better.
        # The provided string at the top level of this solve method is a robust way.
        # This dynamic approach is just for demonstration.
        # The outer string being returned will be used instead.
        return {} # This part is not executed due to the top-level return.

"""
        # The actual Solution class to be submitted is inside the 'code' string.
        # This outer class is just a wrapper to fulfill the execution environment's contract.
        # The 'code' key will contain the full, self-contained Python script.
        return {"code": code.strip()}