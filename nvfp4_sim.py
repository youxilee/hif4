"""NVFP4 输入数据模拟器（按公开权威配方实现）。

依据：
- NVIDIA cuDNN Block Scaling 文档：NVFP4 配方为
      scale  = E4M3_round_up(amax / vmax_E2M1)     # vmax = 6
      output = E2M1_round_to_even(values / scale)
  其中 E2M1 值集为 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}；
- "Four Over Six" (2512.02010)：scale = amax / 4 的变体（保留头部余量）。

用法：
    from nvfp4_sim import nvfp4_encode
    carrier, scale = nvfp4_encode(dense, mode="amax6")   # 或 "amax4"/"pow2"
"""

from __future__ import annotations

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# E4M3: 1 符号 + 4 指数（偏置 7）+ 3 尾数；最小正规数 2^-6，最大有限值 448。
E4M3_MIN_NORMAL = 2.0**-6
E4M3_MAX = 448.0


def e4m3_round_up(x: torch.Tensor) -> torch.Tensor:
    """把正数向上取整到最近的 E4M3 可表示值。"""
    x = torch.nan_to_num(x, nan=1.0, posinf=E4M3_MAX, neginf=E4M3_MIN_NORMAL)
    x = x.clamp_min(E4M3_MIN_NORMAL).clamp_max(E4M3_MAX)
    exponent = torch.floor(torch.log2(x))
    mantissa = x / torch.pow(2.0, exponent)  # [1, 2)
    m_code = torch.ceil((mantissa - 1.0) * 8.0)
    carry = m_code >= 8.0
    exponent = exponent + carry.to(exponent.dtype)
    m_code = torch.where(carry, torch.zeros_like(m_code), m_code)
    value = torch.pow(2.0, exponent.clamp(-6, 8)) * (1.0 + m_code / 8.0)
    return value.clamp_max(E4M3_MAX)


def e2m1_round_to_even(r: torch.Tensor) -> torch.Tensor:
    """把比值舍入到最近的 E2M1 值；平局时取尾数位为偶者。"""
    a = r.abs().contiguous()
    idx = torch.bucketize(a, E2M1).clamp(1, len(E2M1) - 1)
    lo = E2M1[idx - 1]
    hi = E2M1[idx]
    d_lo = a - lo
    d_hi = hi - a
    tie = d_lo == d_hi
    lo_even = ((idx - 1) % 2) == 0  # 偶索引 = 尾数位 0
    pick_lo = torch.where(tie, lo_even, d_lo < d_hi)
    value = torch.where(pick_lo, lo, hi)
    return value * torch.sign(r)


def nvfp4_encode(x: torch.Tensor, mode: str = "amax6"):
    """把稠密张量编码为 (carrier, scale) NVFP4 对（16 元素一块）。

    mode="amax6"：标准 NVFP4，scale = E4M3_round_up(amax/6)；
    mode="amax4"：Four-Over-Six 变体，scale = E4M3_round_up(amax/4)。
    mode="pow2"：功率 2 scale = 2^ceil(log2(amax/6))（E8M0 风格，旧实验用）。
    """
    x = x.detach().to(torch.float32)
    flat = x.reshape(*x.shape[:-1], -1, 16)
    amax = flat.abs().amax(-1).clamp_min(1e-8)
    if mode == "amax6":
        scale = e4m3_round_up(amax / 6.0)
    elif mode == "amax4":
        scale = e4m3_round_up(amax / 4.0)
    elif mode == "pow2":
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0))).clamp_min(1e-8)
    else:
        raise ValueError(f"unknown mode: {mode}")
    ratio = flat / scale.unsqueeze(-1)
    carrier = e2m1_round_to_even(ratio)
    return carrier.reshape_as(x), scale
