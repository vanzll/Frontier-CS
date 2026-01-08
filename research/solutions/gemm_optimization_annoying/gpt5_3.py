import os
import textwrap
import torch
import triton
import triton.language as tl


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 96, 'BLOCK_N': 160, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 160, 'BLOCK_N': 96, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_gelu_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ACT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M
    num_pid_in_group = group_size * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size
    group_size_m = tl.minimum(num_pid_m - first_pid_m, group_size)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_mask_m = offs_m < M
    b_mask_n = offs_n < N

    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = tl.cdiv(K, BLOCK_K)
    for _ in range(0, k_iter):
        k_remaining = K - _ * BLOCK_K
        k_mask = offs_k < k_remaining
        a = tl.load(A_ptrs, mask=a_mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & b_mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Apply activation (GELU)
    if ACT == 1:
        acc = gelu(acc)

    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, acc, mask=a_mask_m[:, None] & b_mask_n[None, :])


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be 2D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible shapes for matmul")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Inputs must be CUDA tensors")

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    # Promote dtypes for compute accuracy; accumulate/store in float32
    a_ = a
    b_ = b
    if a_.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        a_ = a_.to(torch.float32)
    if b_.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        b_ = b_.to(torch.float32)

    # We store output in float32 per spec tolerance and stability
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    _matmul_gelu_kernel[grid](
        a_, b_, c,
        M, N, K,
        a_.stride(0), a_.stride(1),
        b_.stride(0), b_.stride(1),
        c.stride(0), c.stride(1),
        ACT=1,
    )
    return c


class Solution:
    def solve(self, spec_path: str = None) -> dict:
        kernel_code = textwrap.dedent(
            """
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def gelu(x):
                return x * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(x * 0.7071067811865476))


            @triton.autotune(
                configs=[
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=5),
                    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=5),
                    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
                    triton.Config({'BLOCK_M': 96, 'BLOCK_N': 160, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
                    triton.Config({'BLOCK_M': 160, 'BLOCK_N': 96, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
                ],
                key=['M', 'N', 'K'],
            )
            @triton.jit
            def _matmul_gelu_kernel(
                A_ptr, B_ptr, C_ptr,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                ACT: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr,
            ):
                pid = tl.program_id(0)

                num_pid_m = tl.cdiv(M, BLOCK_M)
                num_pid_n = tl.cdiv(N, BLOCK_N)

                group_size = GROUP_M
                num_pid_in_group = group_size * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size
                group_size_m = tl.minimum(num_pid_m - first_pid_m, group_size)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                a_mask_m = offs_m < M
                b_mask_n = offs_n < N

                A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
                B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                k_iter = tl.cdiv(K, BLOCK_K)
                for _ in range(0, k_iter):
                    k_remaining = K - _ * BLOCK_K
                    k_mask = offs_k < k_remaining
                    a = tl.load(A_ptrs, mask=a_mask_m[:, None] & k_mask[None, :], other=0.0)
                    b = tl.load(B_ptrs, mask=k_mask[:, None] & b_mask_n[None, :], other=0.0)
                    acc += tl.dot(a, b)
                    A_ptrs += BLOCK_K * stride_ak
                    B_ptrs += BLOCK_K * stride_bk

                if ACT == 1:
                    acc = gelu(acc)

                C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
                tl.store(C_ptrs, acc, mask=a_mask_m[:, None] & b_mask_n[None, :])


            def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                if a.ndim != 2 or b.ndim != 2:
                    raise ValueError("Inputs must be 2D matrices")
                if a.shape[1] != b.shape[0]:
                    raise ValueError("Incompatible shapes for matmul")
                if not a.is_cuda or not b.is_cuda:
                    raise ValueError("Inputs must be CUDA tensors")

                M, K = a.shape
                Kb, N = b.shape
                assert K == Kb

                a_ = a
                b_ = b
                if a_.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    a_ = a_.to(torch.float32)
                if b_.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    b_ = b_.to(torch.float32)

                c = torch.empty((M, N), device=a.device, dtype=torch.float32)

                grid = lambda META: (
                    triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
                )

                _matmul_gelu_kernel[grid](
                    a_, b_, c,
                    M, N, K,
                    a_.stride(0), a_.stride(1),
                    b_.stride(0), b_.stride(1),
                    c.stride(0), c.stride(1),
                    ACT=1,
                )
                return c
            """
        )
        return {"code": kernel_code}


# If someone imports this module directly, also expose a working matmul for convenience.
# This mirrors the code returned by Solution.solve to make the module directly usable.
# Note: The evaluator primarily uses the returned code string.
# The following ensures functionality if they import and call matmul from this module.

# No-op to avoid redefining in environments where code is executed multiple times.
if "direct_module_matmul_defined" not in globals():
    direct_module_matmul_defined = True

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=5),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=5),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 96, 'BLOCK_N': 160, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
            triton.Config({'BLOCK_M': 160, 'BLOCK_N': 96, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _matmul_gelu_kernel_direct(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        ACT: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)

        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)

        group_size = GROUP_M
        num_pid_in_group = group_size * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size
        group_size_m = tl.minimum(num_pid_m - first_pid_m, group_size)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_mask_m = offs_m < M
        b_mask_n = offs_n < N

        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        k_iter = tl.cdiv(K, BLOCK_K)
        for _ in range(0, k_iter):
            k_remaining = K - _ * BLOCK_K
            k_mask = offs_k < k_remaining
            a = tl.load(A_ptrs, mask=a_mask_m[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(B_ptrs, mask=k_mask[:, None] & b_mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)
            A_ptrs += BLOCK_K * stride_ak
            B_ptrs += BLOCK_K * stride_bk

        if ACT == 1:
            acc = gelu(acc)

        C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(C_ptrs, acc, mask=a_mask_m[:, None] & b_mask_n[None, :])

    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Inputs must be 2D matrices")
        if a.shape[1] != b.shape[0]:
            raise ValueError("Incompatible shapes for matmul")
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")

        M, K = a.shape
        Kb, N = b.shape
        assert K == Kb

        a_ = a
        b_ = b
        if a_.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            a_ = a_.to(torch.float32)
        if b_.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            b_ = b_.to(torch.float32)

        c = torch.empty((M, N), device=a.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )

        _matmul_gelu_kernel_direct[grid](
            a_, b_, c,
            M, N, K,
            a_.stride(0), a_.stride(1),
            b_.stride(0), b_.stride(1),
            c.stride(0), c.stride(1),
            ACT=1,
        )
        return c