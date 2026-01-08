import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    if SPLIT_K > 1:
        K_PER_SPLIT = tl.cdiv(K, SPLIT_K)
        k_start = pid_z * K_PER_SPLIT
        k_end = tl.minimum(k_start + K_PER_SPLIT, K)
    else:
        k_start = 0
        k_end = K
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)
    
    for k in tl.range(k_start, k_end, BLOCK_SIZE_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = k_end - k
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        
        accumulator += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = gelu(accumulator.to(tl.float32))
    
    if SPLIT_K > 1:
        output_ptr = c_ptr + pid_z * M * N
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = output_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

@triton.jit
def reduce_kernel(
    c_ptr, output_ptr,
    M, N,
    stride_cm, stride_cn,
    stride_om, stride_on,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(SPLIT_K):
        ptr = c_ptr + k * M * N
        c_ptrs = ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        accumulator += tl.load(c_ptrs, mask=mask, other=0.0)
    
    c = gelu(accumulator)
    output_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(output_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Incompatible dimensions"
    
    a = a.contiguous()
    b = b.contiguous()
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    def get_config():
        if M * N <= 8192 * 8192:
            configs = [
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
            ]
        else:
            configs = [
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            ]
        
        for config in configs:
            config.kwargs['EVEN_K'] = K % (config.kwargs['BLOCK_SIZE_K'] * 1) == 0
            config.kwargs['ACC_TYPE'] = tl.float32 if a.dtype == torch.float32 else tl.float16
        
        return configs
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        1,
    )
    
    best_config = None
    best_time = float('inf')
    
    for config in get_config():
        try:
            kernel = matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                **config.kwargs,
            )
            
            torch.cuda.synchronize()
            
            if kernel.best_config:
                best_config = config
                break
        except:
            continue
    
    if best_config is None:
        best_config = get_config()[0]
    
    SPLIT_K = 1
    if K > 2048 and M * N > 524288:
        SPLIT_K = min(4, triton.cdiv(K, 512))
    
    if SPLIT_K > 1:
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            SPLIT_K,
        )
        
        intermediate = torch.empty((SPLIT_K, M, N), device=a.device, dtype=a.dtype)
        
        matmul_kernel[grid](
            a, b, intermediate,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            intermediate.stride(1), intermediate.stride(2),
            SPLIT_K=SPLIT_K,
            **best_config.kwargs,
        )
        
        reduce_grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        reduce_kernel[reduce_grid](
            intermediate, c,
            M, N,
            intermediate.stride(1), intermediate.stride(2),
            c.stride(0), c.stride(1),
            SPLIT_K=SPLIT_K,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
        )
    else:
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            1,
        )
        
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            SPLIT_K=1,
            **best_config.kwargs,
        )
    
    return c

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    if SPLIT_K > 1:
        K_PER_SPLIT = tl.cdiv(K, SPLIT_K)
        k_start = pid_z * K_PER_SPLIT
        k_end = tl.minimum(k_start + K_PER_SPLIT, K)
    else:
        k_start = 0
        k_end = K
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)
    
    for k in tl.range(k_start, k_end, BLOCK_SIZE_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = k_end - k
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        
        accumulator += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = gelu(accumulator.to(tl.float32))
    
    if SPLIT_K > 1:
        output_ptr = c_ptr + pid_z * M * N
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = output_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

@triton.jit
def reduce_kernel(
    c_ptr, output_ptr,
    M, N,
    stride_cm, stride_cn,
    stride_om, stride_on,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(SPLIT_K):
        ptr = c_ptr + k * M * N
        c_ptrs = ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        accumulator += tl.load(c_ptrs, mask=mask, other=0.0)
    
    c = gelu(accumulator)
    output_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(output_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Incompatible dimensions"
    
    a = a.contiguous()
    b = b.contiguous()
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    def get_config():
        if M * N <= 8192 * 8192:
            configs = [
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
            ]
        else:
            configs = [
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            ]
        
        for config in configs:
            config.kwargs['EVEN_K'] = K % (config.kwargs['BLOCK_SIZE_K'] * 1) == 0
            config.kwargs['ACC_TYPE'] = tl.float32 if a.dtype == torch.float32 else tl.float16
        
        return configs
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        1,
    )
    
    best_config = None
    best_time = float('inf')
    
    for config in get_config():
        try:
            kernel = matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                **config.kwargs,
            )
            
            torch.cuda.synchronize()
            
            if kernel.best_config:
                best_config = config
                break
        except:
            continue
    
    if best_config is None:
        best_config = get_config()[0]
    
    SPLIT_K = 1
    if K > 2048 and M * N > 524288:
        SPLIT_K = min(4, triton.cdiv(K, 512))
    
    if SPLIT_K > 1:
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            SPLIT_K,
        )
        
        intermediate = torch.empty((SPLIT_K, M, N), device=a.device, dtype=a.dtype)
        
        matmul_kernel[grid](
            a, b, intermediate,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            intermediate.stride(1), intermediate.stride(2),
            SPLIT_K=SPLIT_K,
            **best_config.kwargs,
        )
        
        reduce_grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        reduce_kernel[reduce_grid](
            intermediate, c,
            M, N,
            intermediate.stride(1), intermediate.stride(2),
            c.stride(0), c.stride(1),
            SPLIT_K=SPLIT_K,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
        )
    else:
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            1,
        )
        
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            SPLIT_K=1,
            **best_config.kwargs,
        )
    
    return c
'''
        return {"code": code}