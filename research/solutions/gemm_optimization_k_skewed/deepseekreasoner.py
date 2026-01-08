import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    if SPLIT_K > 1:
        k_step = K // SPLIT_K
        k_start = pid_z * k_step
        k_end = k_start + k_step
        if pid_z == SPLIT_K - 1:
            k_end = K
    else:
        k_start = 0
        k_end = K
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    if EVEN_K:
        for k in range(k_start, k_end, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_end - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_end - k, other=0.0)
            accumulator += tl.dot(a, b, allow_tf32=True)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(k_start, k_end, BLOCK_K):
            k_remaining = k_end - k
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N), other=0.0)
            accumulator += tl.dot(a, b, allow_tf32=True)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    
    if SPLIT_K > 1:
        accumulator = gelu(accumulator)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
    else:
        accumulator = gelu(accumulator)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check, f"Shape mismatch: {a.shape} @ {b.shape}"
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    def get_config(M, N, K):
        if M * N <= 262144:
            return (64, 64, 16, 4, tl.float32)
        elif K <= 128:
            return (128, 64, 32, 4, tl.float32)
        elif K >= 4096:
            return (128, 128, 64, 8, tl.float32)
        else:
            return (128, 64, 32, 8, tl.float32)
    
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, ACC_TYPE = get_config(M, N, K)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        META.get('SPLIT_K', 1)
    )
    
    even_k = K % BLOCK_K == 0
    
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        EVEN_K=even_k,
        ACC_TYPE=ACC_TYPE,
        SPLIT_K=1,
        num_warps=4,
        num_stages=3
    )
    
    return c

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    if SPLIT_K > 1:
        k_step = K // SPLIT_K
        k_start = pid_z * k_step
        k_end = k_start + k_step
        if pid_z == SPLIT_K - 1:
            k_end = K
    else:
        k_start = 0
        k_end = K
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    if EVEN_K:
        for k in range(k_start, k_end, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_end - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_end - k, other=0.0)
            accumulator += tl.dot(a, b, allow_tf32=True)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(k_start, k_end, BLOCK_K):
            k_remaining = k_end - k
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N), other=0.0)
            accumulator += tl.dot(a, b, allow_tf32=True)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    
    if SPLIT_K > 1:
        accumulator = gelu(accumulator)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
    else:
        accumulator = gelu(accumulator)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check, f"Shape mismatch: {a.shape} @ {b.shape}"
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    def get_config(M, N, K):
        if M * N <= 262144:
            return (64, 64, 16, 4, tl.float32)
        elif K <= 128:
            return (128, 64, 32, 4, tl.float32)
        elif K >= 4096:
            return (128, 128, 64, 8, tl.float32)
        else:
            return (128, 64, 32, 8, tl.float32)
    
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, ACC_TYPE = get_config(M, N, K)
    
    grid = lambda META: (
        triton.cdiv(M, META[\'BLOCK_M\']) * triton.cdiv(N, META[\'BLOCK_N\']),
        META.get(\'SPLIT_K\', 1)
    )
    
    even_k = K % BLOCK_K == 0
    
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        EVEN_K=even_k,
        ACC_TYPE=ACC_TYPE,
        SPLIT_K=1,
        num_warps=4,
        num_stages=3
    )
    
    return c
'''
        return {"code": code}