"""Fused RMSNorm + per-row symmetric INT8 quantisation (Triton), registered as ``serving::rmsnorm_int8``.

For each row x of an activation matrix [M, N]:

    y     = x * rsqrt(mean(x^2) + eps) * weight          (fp32 inside the kernel)
    scale = max(|y|) / 127
    q     = clamp(rint(y * 127 / max|y|), -127, 127)      (int8; rint = round half to even)

returns (q int8 [M, N], scale fp32 [M]); ``q * scale[:, None]`` approximates y. A row of zeros gives q = 0 and scale = 0.

Why fuse: unfused, RMSNorm writes y (16-bit) to memory, an abs-max pass reads it, and the quantise pass reads it again. The fused
kernel reads x once and writes only int8 + one scale per row, so the memory traffic is the floor for this operation.

Limits (also asserted at the call site): one program handles one whole row held in registers, so N <= MAX_N; no backward (the
quantiser is not differentiable and this is an inference kernel); the weight is a plain [N] vector (no bias, no residual add).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

MAX_N = 16384
QMAX = 127.0


def reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain PyTorch implementation of the same maths (fp32 throughout); also the baseline for the benchmark."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    amax = y.abs().amax(-1, keepdim=True)
    inv = torch.where(amax > 0, QMAX / amax, torch.zeros_like(amax))
    q = torch.round(y * inv).clamp_(-QMAX, QMAX).to(torch.int8)
    return q, (amax / QMAX).squeeze(-1)


@triton.jit
def _rmsnorm_int8_kernel(x_ptr, w_ptr, q_ptr, s_ptr, N, stride_x, stride_q, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / N + eps) * w
    amax = tl.max(tl.abs(y), axis=0)
    inv = tl.where(amax > 0, 127.0 / amax, 0.0)
    q = libdevice.rint(y * inv)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(q_ptr + row * stride_q + cols, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, amax / 127.0)


def _launch_config(n: int) -> tuple[int, int]:
    block = triton.next_power_of_2(n)
    return block, (4 if block <= 2048 else 8 if block <= 8192 else 16)


def _check(x: torch.Tensor, weight: torch.Tensor) -> None:
    if not x.is_cuda or x.device != weight.device:
        raise ValueError("x and weight must be on the same CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"x must be fp16, bf16 or fp32, got {x.dtype}")
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"weight must have shape ({x.shape[-1]},), got {tuple(weight.shape)}")
    if x.shape[-1] > MAX_N:
        raise ValueError(f"hidden size {x.shape[-1]} > {MAX_N}: a whole row no longer fits in one program")


@torch.library.custom_op("serving::rmsnorm_int8", mutates_args=())
def rmsnorm_int8(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    _check(x, weight)
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    w = weight if weight.is_contiguous() else weight.contiguous()
    m = x2.shape[0]
    q = torch.empty((m, n), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    if m:
        block, warps = _launch_config(n)
        _rmsnorm_int8_kernel[(m,)](x2, w, q, s, n, x2.stride(0), q.stride(0), eps, BLOCK_N=block, num_warps=warps)
    return q.reshape(x.shape), s.reshape(x.shape[:-1])


@rmsnorm_int8.register_fake
def _(x, weight, eps=1e-6):
    return torch.empty_like(x, dtype=torch.int8), x.new_empty(x.shape[:-1], dtype=torch.float32)
