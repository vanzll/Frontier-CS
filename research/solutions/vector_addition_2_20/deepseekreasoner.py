import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    out = x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def add_kernel_tiled(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    tile_id = pid // TILE_SIZE
    lane_id = pid % TILE_SIZE
    
    block_start = tile_id * BLOCK_SIZE * TILE_SIZE + lane_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    out = x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    if n_elements <= 0:
        return out
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    if n_elements <= 131072:
        BLOCK_SIZE = 1024
        add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    elif n_elements <= 524288:
        BLOCK_SIZE = 2048
        add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=16)
    else:
        BLOCK_SIZE = 4096
        add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=32)
    
    return out

class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = '''import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    out = x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit  
def add_kernel_vectorized(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    out = x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def add_kernel_optimized(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    out = x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    if n_elements <= 0:
        return out
    
    if n_elements == 1048576:
        BLOCK_SIZE = 4096
        NUM_WARPS = 32
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        add_kernel_optimized[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS)
    else:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        if n_elements <= 131072:
            BLOCK_SIZE = 1024
            add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        elif n_elements <= 524288:
            BLOCK_SIZE = 2048
            add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=16)
        else:
            BLOCK_SIZE = 4096
            add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=32)
    
    return out'''
        
        return {"code": code}