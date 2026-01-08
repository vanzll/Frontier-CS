import torch
import triton
import triton.language as tl
from typing import Optional, Dict, Any
import json
import os

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 4, 'SPLIT_K': 1}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'STAGES': 4, 'SPLIT_K': 1}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 2}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 2}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 4}, num_stages=3, num_warps=4),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, STAGES: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    pid_k = tl.program_id(axis=1)
    K_BLOCK = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_BLOCK
    k_end = tl.minimum(k_start + K_BLOCK, K)
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(k_start, k_end, BLOCK_K * STAGES):
        a = tl.zeros((BLOCK_M, BLOCK_K), dtype=A.dtype.element_ty)
        b = tl.zeros((BLOCK_K, BLOCK_N), dtype=B.dtype.element_ty)
        
        for stage in range(STAGES):
            k_cur = k + stage * BLOCK_K
            if k_cur < k_end:
                mask_a = (offs_am[:, None] < M) & ((k_cur + offs_k[None, :]) < K)
                mask_b = ((k_cur + offs_k[:, None]) < K) & (offs_bn[None, :] < N)
                
                a_stage = tl.load(a_ptrs + k_cur * stride_ak, mask=mask_a, other=0.0)
                b_stage = tl.load(b_ptrs + k_cur * stride_bk, mask=mask_b, other=0.0)
                
                a += a_stage
                b += b_stage
            else:
                a += tl.zeros((BLOCK_M, BLOCK_K), dtype=A.dtype.element_ty)
                b += tl.zeros((BLOCK_K, BLOCK_N), dtype=B.dtype.element_ty)
        
        accumulator += tl.dot(a, b, allow_tf32=True)
    
    accumulator = accumulator.to(C.dtype.element_ty)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    if SPLIT_K == 1:
        tl.store(c_ptrs, gelu(accumulator), mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], f"Shapes incompatible: {a.shape} @ {b.shape}"
    M, K = a.shape
    K, N = b.shape
    
    a = a.contiguous()
    b = b.contiguous()
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        META['SPLIT_K'],
    )
    
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c

class Solution:
    def solve(self, spec_path: Optional[str] = None) -> Dict[str, Any]:
        if spec_path:
            with open(spec_path, 'r') as f:
                spec = json.load(f)
            output_path = spec.get('output_path', 'kernel.py')
            with open(output_path, 'w') as f:
                f.write(self._get_code())
            return {"program_path": output_path}
        return {"code": self._get_code()}
    
    def _get_code(self) -> str:
        import inspect
        return inspect.getsource(__main__) if '__main__' in globals() else """
import torch
import triton
import triton.language as tl
from typing import Optional, Dict, Any
import json
import os

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 4, 'SPLIT_K': 1}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'STAGES': 4, 'SPLIT_K': 1}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 2}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 2}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'STAGES': 3, 'SPLIT_K': 4}, num_stages=3, num_warps=4),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, STAGES: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    pid_k = tl.program_id(axis=1)
    K_BLOCK = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_BLOCK
    k_end = tl.minimum(k_start + K_BLOCK, K)
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(k_start, k_end, BLOCK_K * STAGES):
        a = tl.zeros((BLOCK_M, BLOCK_K), dtype=A.dtype.element_ty)
        b = tl.zeros((BLOCK_K, BLOCK_N), dtype=B.dtype.element_ty)
        
        for stage in range(STAGES):
            k_cur = k + stage * BLOCK_K
            if k_cur < k_end:
                mask_a = (offs_am[:, None] < M) & ((k_cur + offs_k[None, :]) < K)
                mask_b = ((k_cur + offs_k[:, None]) < K) & (offs_bn[None, :] < N)
                
                a_stage = tl.load(a_ptrs + k_cur * stride_ak, mask=mask_a, other=0.0)
                b_stage = tl.load(b_ptrs + k_cur * stride_bk, mask=mask_b, other=0.0)
                
                a += a_stage
                b += b_stage
            else:
                a += tl.zeros((BLOCK_M, BLOCK_K), dtype=A.dtype.element_ty)
                b += tl.zeros((BLOCK_K, BLOCK_N), dtype=B.dtype.element_ty)
        
        accumulator += tl.dot(a, b, allow_tf32=True)
    
    accumulator = accumulator.to(C.dtype.element_ty)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    if SPLIT_K == 1:
        tl.store(c_ptrs, gelu(accumulator), mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], f"Shapes incompatible: {a.shape} @ {b.shape}"
    M, K = a.shape
    K, N = b.shape
    
    a = a.contiguous()
    b = b.contiguous()
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        META['SPLIT_K'],
    )
    
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c

class Solution:
    def solve(self, spec_path: Optional[str] = None) -> Dict[str, Any]:
        if spec_path:
            with open(spec_path, 'r') as f:
                spec = json.load(f)
            output_path = spec.get('output_path', 'kernel.py')
            with open(output_path, 'w') as f:
                f.write(self._get_code())
            return {"program_path": output_path}
        return {"code": self._get_code()}
    
    def _get_code(self) -> str:
        import inspect
        return inspect.getsource(__main__) if '__main__' in globals() else ''
"""