import torch
import triton
import triton.language as tl

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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_ACC: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K * SPLIT_K)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    pid_k = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    if EVEN_K:
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b, out_dtype=ACC_TYPE)
        for k in range(1, SPLIT_K):
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b, out_dtype=ACC_TYPE)
    else:
        for k in range(0, SPLIT_K):
            k_remaining = K - k * BLOCK_K
            if k_remaining >= BLOCK_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b, out_dtype=ACC_TYPE)
            else:
                a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
                accumulator += tl.dot(a, b, out_dtype=ACC_TYPE)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    
    c = accumulator.to(c_ptr.dtype.element_ty)
    c = gelu(c)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask)
    else:
        if USE_ACC:
            tl.atomic_add(c_ptrs, c, mask=mask)
        else:
            tl.store(c_ptrs, c, mask=mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Shapes {a.shape} and {b.shape} cannot be multiplied"
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    if min(M, N, K) <= 0:
        return c
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        META['SPLIT_K']
    )
    
    def get_configs():
        configs = []
        block_mn_candidates = [16, 32, 64, 128, 256]
        block_k_candidates = [32, 64, 128, 256]
        
        for bm in block_mn_candidates:
            for bn in block_mn_candidates:
                for bk in block_k_candidates:
                    for split_k in [1, 2, 4, 8]:
                        for group_m in [1, 4, 8, 16]:
                            if bm * bk + bn * bk <= 65536:
                                configs.append({
                                    'BLOCK_M': bm,
                                    'BLOCK_N': bn,
                                    'BLOCK_K': bk,
                                    'GROUP_M': group_m,
                                    'SPLIT_K': split_k,
                                    'ACC_TYPE': tl.float32 if a.dtype == torch.float16 else tl.float32,
                                    'EVEN_K': K % (bk * split_k) == 0,
                                    'USE_ACC': split_k > 1
                                })
        return configs
    
    configs = get_configs()
    
    compiled_kernel = triton.autotune(configs, key=['M', 'N', 'K'])(matmul_kernel)
    
    compiled_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACC_TYPE=tl.float32 if a.dtype == torch.float16 else tl.float32,
        EVEN_K=K % 128 == 0,
        USE_ACC=False
    )
    
    return c

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        return {"code": __import__('inspect').getsource(__import__(__name__))}