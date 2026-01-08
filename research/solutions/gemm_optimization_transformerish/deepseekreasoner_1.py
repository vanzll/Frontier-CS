import torch
import triton
import triton.language as tl
import os

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
    USE_FP16_ACC: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    
    if GROUP_M > 1:
        num_pid_in_group = GROUP_M * grid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(grid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32 if USE_FP16_ACC else tl.float16)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    if USE_FP16_ACC:
        accumulator = accumulator.to(tl.float16)
    
    c = gelu(accumulator)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} @ {b.shape}"
    M, K = a.shape
    _, N = b.shape
    
    a = a.contiguous()
    b = b.contiguous() if b.stride(0) == 1 or b.stride(1) == 1 else b.contiguous()
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    def get_configs():
        configs = []
        for block_m in [32, 64, 128]:
            for block_n in [32, 64, 128]:
                for block_k in [32, 64, 128]:
                    for group_m in [1, 4, 8]:
                        for use_fp16_acc in [False, True]:
                            if block_m * block_n <= 256 * 256:
                                configs.append(triton.Config({
                                    'BLOCK_M': block_m,
                                    'BLOCK_N': block_n,
                                    'BLOCK_K': block_k,
                                    'GROUP_M': group_m,
                                    'EVEN_K': K % block_k == 0,
                                    'USE_FP16_ACC': use_fp16_acc,
                                }, num_stages=4, num_warps=8))
        return configs
    
    def get_best_config():
        if M <= 64 and N <= 64:
            return triton.Config({
                'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32,
                'GROUP_M': 8, 'EVEN_K': K % 32 == 0, 'USE_FP16_ACC': False
            }, num_stages=4, num_warps=4)
        elif M <= 128 and N <= 128:
            return triton.Config({
                'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32,
                'GROUP_M': 4, 'EVEN_K': K % 32 == 0, 'USE_FP16_ACC': True
            }, num_stages=4, num_warps=8)
        else:
            return triton.Config({
                'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
                'GROUP_M': 4, 'EVEN_K': K % 64 == 0, 'USE_FP16_ACC': True
            }, num_stages=4, num_warps=8)
    
    config = get_best_config()
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        **config.kwargs
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
    USE_FP16_ACC: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    
    if GROUP_M > 1:
        num_pid_in_group = GROUP_M * grid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(grid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32 if USE_FP16_ACC else tl.float16)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    if USE_FP16_ACC:
        accumulator = accumulator.to(tl.float16)
    
    c = gelu(accumulator)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} @ {b.shape}"
    M, K = a.shape
    _, N = b.shape
    
    a = a.contiguous()
    b = b.contiguous() if b.stride(0) == 1 or b.stride(1) == 1 else b.contiguous()
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    def get_best_config():
        if M <= 64 and N <= 64:
            return triton.Config({
                'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32,
                'GROUP_M': 8, 'EVEN_K': K % 32 == 0, 'USE_FP16_ACC': False
            }, num_stages=4, num_warps=4)
        elif M <= 128 and N <= 128:
            return triton.Config({
                'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32,
                'GROUP_M': 4, 'EVEN_K': K % 32 == 0, 'USE_FP16_ACC': True
            }, num_stages=4, num_warps=8)
        else:
            return triton.Config({
                'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
                'GROUP_M': 4, 'EVEN_K': K % 64 == 0, 'USE_FP16_ACC': True
            }, num_stages=4, num_warps=8)
    
    config = get_best_config()
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        **config.kwargs
    )
    
    return c
'''
        return {"code": code}