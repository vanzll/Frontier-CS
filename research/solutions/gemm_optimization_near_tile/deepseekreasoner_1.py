import torch
import triton
import triton.language as tl
from typing import Dict

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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    if EVEN_K:
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_bn[None, :] < N), other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    c = gelu(accumulator).to(a_ptr.dtype.element_ty)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    def autotune_configs():
        configs = []
        for block_m in [64, 128, 256]:
            for block_n in [64, 128, 256]:
                for block_k in [32, 64, 128]:
                    for group_m in [8]:
                        for even_k in [False, True]:
                            if block_k <= K:
                                configs.append(triton.Config({
                                    'BLOCK_M': block_m,
                                    'BLOCK_N': block_n,
                                    'BLOCK_K': block_k,
                                    'GROUP_M': group_m,
                                    'EVEN_K': even_k,
                                    'USE_TMA': False,
                                }, num_warps=4 if block_n <= 64 else 8))
        return configs
    
    kernel = _matmul_kernel
    if hasattr(triton, 'autotune'):
        kernel = triton.autotune(
            configs=autotune_configs(),
            key=['M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn']
        )(_matmul_kernel)
    
    kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

class Solution:
    def solve(self, spec_path: str = None) -> Dict:
        code = '''
import torch
import triton
import triton.language as tl
from typing import Dict

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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    if EVEN_K:
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_bn[None, :] < N), other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    c = gelu(accumulator).to(a_ptr.dtype.element_ty)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (triton.cdiv(M, META[\'BLOCK_M\']) * triton.cdiv(N, META[\'BLOCK_N\']),)
    
    def autotune_configs():
        configs = []
        for block_m in [64, 128, 256]:
            for block_n in [64, 128, 256]:
                for block_k in [32, 64, 128]:
                    for group_m in [8]:
                        for even_k in [False, True]:
                            if block_k <= K:
                                configs.append(triton.Config({
                                    \'BLOCK_M\': block_m,
                                    \'BLOCK_N\': block_n,
                                    \'BLOCK_K\': block_k,
                                    \'GROUP_M\': group_m,
                                    \'EVEN_K\': even_k,
                                    \'USE_TMA\': False,
                                }, num_warps=4 if block_n <= 64 else 8))
        return configs
    
    kernel = _matmul_kernel
    if hasattr(triton, \'autotune\'):
        kernel = triton.autotune(
            configs=autotune_configs(),
            key=[\'M\', \'N\', \'K\', \'stride_am\', \'stride_ak\', \'stride_bk\', \'stride_bn\']
        )(_matmul_kernel)
    
    kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
'''
        return {"code": code}