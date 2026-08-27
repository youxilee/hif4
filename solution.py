"""HiF4 solution for the 2026 Huawei algorithm competition.

The implementation keeps the official HiF4 conversion as an explicit fallback,
selects calibration-gated equivalent scaling/reordering/block-matrix transforms,
and applies bounded scale/hierarchy refinement to difficult blocks. All
calibration states are plain CPU data.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Union

import torch


_NVFP4_BLOCK_SIZE = 16
_HIF4_BLOCK_SIZE = 64
_E6M2_MIN = 2.0**-48
_E6M2_MAX = 49152.0
_HIF4_MAX_INNER = 7.0
_BF16_ONE_SEVENTH = 0.142578125
_EPS = 1.0e-12

_LINEAR_STATS_TOKENS = 4096
_LINEAR_EVAL_TOKENS = 128
_LINEAR_WEIGHT_EVAL_ROWS = 256
_ATTN_STATS_TOKENS = 4096
_ATTN_EVAL_TOKENS = 128

# Tunable calibration/refinement knobs.  Values are deliberately
# conservative so worst-case calibration and dynamic-quantization time stay
# bounded on large models; the sweep harness overrides them to find a better
# accuracy/runtime trade-off.
_WEIGHT_SMOOTH_ALPHAS = (0.25, 0.50, 0.75)
# 宽层（FFN 的 fc/proj，输入或输出 ≥ 2048）用更细的 alpha 网格：
# 通道多、候选统计更稳，细网格实测 +fc 0.0012 / +proj 0.0044；
# 窄层（q/k/v/o，768）细网格反而过拟合（-0.0003~-0.0021），保持 3 档。
_WEIGHT_SMOOTH_ALPHAS_WIDE = (0.25, 0.375, 0.50, 0.625, 0.75)
_WIDE_LAYER_MIN_DIM = 2048
# Matrix SmoothQuant candidates.  After the diagonal scale and optional
# permutation, apply the same orthogonal block transform to X and W.  In the
# usual X @ W convention this is a block-diagonal S on X and S^{-1} on W;
# orthogonality makes S^{-1}=S^T, so the row-major weight carrier can use the
# same right transform.  Only the winning block size enters dynamic state.
_BLOCK_SMOOTH_ALLOWED_SIZES = (4, 8, 16, 32, 64)
_BLOCK_SMOOTH_SIZES = (4, 8, 16)
# 下投影层（proj，out_features < in_features）额外允许更大 block：
# 输入通道多、块内相关性更强，32/64 能更充分摊平重尾（GPT-2 即 FFN 下投影 3072→768，
# 实测 proj +0.0115；对 o/fc 放宽会轻微退化，故只给 proj）。
_BLOCK_SMOOTH_PROJ_SIZES = (4, 8, 16, 32, 64)
_BLOCK_SMOOTH_SEEDS = (0, 1, 2, 3)
# 其余层只枚举 seed 0（v2.0 已验证的绝对索引图案）。旧公式 seed 0/1/2 完全
# 退化、seed 3 近退化，等价于只有 seed 0；多样化 seed 仅 proj 使用，窄层
# 细搜索会过拟合（v1.8 教训，8-batch 实测 fc -0.0037）。
_BLOCK_SMOOTH_NARROW_SEEDS = (0,)
# Evaluation-only override used by the sweep harness.  Zero keeps the guarded
# production behavior; 4/8/16 forces that size while still choosing its best
# deterministic sign seed on calibration data.
_BLOCK_SMOOTH_FORCE_SIZE = 0
_BLOCK_SMOOTH_MIN_IMPROVEMENT = 0.005
_BLOCK_SMOOTH_WORST_TOLERANCE = 0.005
_WEIGHT_REFINE_ERROR_THRESHOLD = 1.0e-7
_WEIGHT_REFINE_ACCEPT_MARGIN = 0.005
_WEIGHT_REFINE_MAX_RATIO_SMALL = 1.0
_WEIGHT_REFINE_MAX_RATIO_LARGE = 1.0
_WEIGHT_REFINE_MAX_BLOCKS = 65_536

# 候选评估与落地量化器对齐（v2.5）：Linear block-S 搜索用 offsets + importance +
# refine 后的端到端输出打分，消除“候选用标准 HiF4、落地用精确 refine”的目标偏差。
# ratio 0.5 保留候选判别力（v2.4 Attention 经验：full refine 会抹平差异）且成本减半。
_LINEAR_CANDIDATE_REFINE_RATIO = 0.5
_LINEAR_CANDIDATE_REFINE_BLOCKS = 8_192

_ACTIVATION_REFINE_ERROR_THRESHOLD = 1.0e-7
_ACTIVATION_REFINE_ACCEPT_MARGIN = 0.02
_ACTIVATION_REFINE_MAX_RATIO = 0.70
_ACTIVATION_REFINE_MAX_BLOCKS = 32_768

_QK_SMOOTH_ALPHAS = (0.25, 0.50)
_WEIGHT_SMOOTH_RMS = True
_QK_SMOOTH_RMS = True
_ATTN_CENTER_MODES = (0, 2)
# Rank refinement by absolute block error (True) or normalized error (False).
_REFINE_RANK_BY_ABSOLUTE = True
_ATTN_REFINE_ERROR_THRESHOLD = 1.0e-7
_Q_REFINE_ACCEPT_MARGIN = 0.03
_Q_REFINE_MAX_RATIO = 0.60
_Q_REFINE_MAX_BLOCKS = 16_384
_K_REFINE_ACCEPT_MARGIN = 0.03
_K_REFINE_MAX_RATIO = 0.70
_K_REFINE_MAX_BLOCKS = 24_576
_V_REFINE_ACCEPT_MARGIN = 0.01
_V_REFINE_MAX_RATIO = 0.60
_V_REFINE_MAX_BLOCKS = 24_576

# Attention Q/K 等价块正交变换（v2.3）：每个 head 内 Q′=QS、K′=KS⁻ᵀ
# （S 正交 → S⁻ᵀ=S），QKᵀ 在量化前严格不变，把块内能量摊平后再量化以降低
# logits 误差。Q 与 K 必须共享同一 (size, seed)，否则等价性破坏。
# head_dim=64 可整除 4/8/16。候选用端到端 causal softmax 输出 MSE 门控。
_ATTN_BLOCK_SMOOTH_SIZES = (4, 8, 16)
_ATTN_BLOCK_SMOOTH_SEEDS = (0, 1, 2, 3)
# 门控阈值按端到端输出 MSE 的尺度设定：score +0.001 ≈ m_h 相对下降 ~0.16%，
# 故 min_mean_improvement 用 0.001 量级（原逐算子 proxy 用的是 0.01，量纲不同）。
_ATTN_BLOCK_MIN_IMPROVEMENT = 0.001
_ATTN_BLOCK_WORST_TOLERANCE = 0.005
_ATTN_SMOOTH_MIN_IMPROVEMENT = 0.001
_ATTN_SMOOTH_WORST_TOLERANCE = 0.01
_ATTN_PERM_MIN_IMPROVEMENT = 0.001
_ATTN_PERM_WORST_TOLERANCE = 0.005
# v2.4：候选评估用"最终量化器"（offsets + refine + 候选 importance）而不是
# 标准 HiF4，消除候选用标准量化器评、落地却用精确量化器的目标不一致。
# refine 比例取 0.5 而非 1.0：只 refine 最难的半数块，反而比全量 refine 判别力
# 更好（全量把候选输出拉平，丢掉了"哪些候选更受益于 refine"的信号），且省一半
# 耗时；8 批实测 0.4961 vs 0.4954（ratio 1.0）。accept/block 上限与落地一致。
_ATTN_CANDIDATE_REFINE_RATIO = 0.5
_ATTN_CANDIDATE_ACCEPT_MARGIN = 0.03
_ATTN_CANDIDATE_REFINE_BLOCKS = 24_576

_SMOOTH_SCALE_MIN = 1.0 / 8.0
_SMOOTH_SCALE_MAX = 8.0
_QK_SMOOTH_MIN = 1.0 / 16.0
_QK_SMOOTH_MAX = 16.0

# Importance weights are mean-normalized; a floor keeps every channel at least
# this fraction of the mean.  Without it, calibrated importance can be ~0 for
# whole blocks (e.g. outlier-heavy Q/K data), their weighted losses vanish, and
# the scale search drifts on numerical noise while degrading the unweighted
# reconstruction of those blocks.
_IMPORTANCE_FLOOR = 0.05

# E6M2 code offsets.  Offset +2 is roughly the E6M2 analogue of the
# alternative 1.5x scale mode seen in microscaling scale search.
#
# Exact-solve analysis: the standard amax/7 code frequently rounds DOWN
# (clipping the block peak), and all other 4-element subgroups sit on
# arbitrary log phases, so the useful neighborhood is +1..+3 (~1.25x..2x
# scale).  Negative offsets shrink the representable range and almost never
# win on the common NVFP4 input regimes; -1 is kept as a fallback for
# overshooting codes.  -2 is kept for weights (calibration-only cost) as
# insurance for finer-grained input encodings where it wins up to a few
# percent of blocks, but dropped from the per-sample dynamic path.
_DYNAMIC_OFFSETS = (-1, 1, 2, 3)
_WEIGHT_OFFSETS = (-2, -1, 1, 2, 3)

# v2.6: X/W 联合残差补偿。标准 refine 只最小化各块重建损失；联合 refine
# 额外把激活量化交叉项纳入候选评估，直接在校准数据上最小化输出误差
# m_h = ||X·Wᵀ − Q(X)·Q(W)ᵀ||²（对块位置做 Gauss-Seidel，逐步单调）。
# 8-batch 实测 6 个线性算子 72/72 层全胜（fc +0.06/proj +0.08/q +0.04/
# o +0.07/k +0.05/v +0.05），换 calib/test 划分增益基本不变。
_JOINT_REFINE_ENABLED = True
_JOINT_REFINE_ITER = 3
_JOINT_REFINE_MIN_TOKENS = 8

# The per-block scale error over E6M2 codes is locally unimodal, so if the
# best fixed-window offset lands on a window edge, the true optimum may lie
# outside the window.  Extend the search beyond the winning edge (only for
# blocks that actually hit the edge) by up to this many extra codes.
_REFINE_EDGE_EXTENSION = True
_REFINE_EDGE_EXTEND_STEPS = 2

# Data-driven per-layer refine budgets: instead of a global hand-tuned ratio,
# calibration estimates the block-loss distribution and stores the smallest
# refine fraction that captures a target share of the total weighted loss.
_DATA_DRIVEN_RATIO = True
# 时间预算允许放宽后，把损失覆盖目标从 0.99 提到 0.999：实际 refine
# 比例从 ~0.95 提到 ~0.99（几乎全量），8 批测试上 7 类得分全部为正
# （attn +0.0018，其余 +0.0001~0.0003），动态耗时 +约 5%。
_RATIO_CAPTURE_TARGET = 0.99
_RATIO_MIN = 0.10

# Weight quantization can use the full per-block activation covariance as a
# quadratic loss (true output-MSE weighting) instead of the diagonal
# per-channel importance.  This is calibration-only: the Gram/covariance
# never enters a dynamic state, so the 4096-node state limit is unaffected.
_WEIGHT_QUADRATIC = True
_WEIGHT_QUADRATIC_MAX_FEATURES = 4096

# Dynamic activation quantization can also use the full weight Gram (W^T W)
# as quadratic error weights.  Unlike Q/K covariances (estimated from a few
# calibration tokens), the weight Gram is static and well conditioned, so the
# same machinery that helped weight quantization should transfer.  Only the
# per-4-group 4x4 blocks are stored in the state (~4*channels elements).
_ACTIVATION_QUADRATIC = True
# Gram state is ~4*channels elements; cap so the single stored tensor stays
# within 4096 elements even under a strict element-count reading of the state
# node limit.  Wide layers (e.g. FFN down-projection, 3072) fall back to the
# diagonal importance automatically.
_ACTIVATION_QUADRATIC_MAX_FEATURES = 1024


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = _NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize an NVFP4 carrier/scale pair to BF16."""

    if not torch.is_tensor(quant_float) or not torch.is_tensor(scale_float):
        raise TypeError("quant_float and scale_float must be torch.Tensor")
    if quant_float.ndim < 1:
        raise ValueError("quant_float must have at least one dimension")
    c = int(quant_float.shape[-1])
    if c % blk_size != 0:
        raise ValueError(
            f"Last dim {c} is not divisible by NVFP4 block size {blk_size}"
        )
    expected_scale_shape = tuple(quant_float.shape[:-1]) + (c // blk_size,)
    if tuple(scale_float.shape) != expected_scale_shape:
        raise ValueError(
            f"scale_float shape {tuple(scale_float.shape)} does not match "
            f"expected {expected_scale_shape}"
        )

    x = quant_float.unflatten(-1, (-1, blk_size))
    result = x * scale_float.unsqueeze(-1)
    return result.flatten(-2, -1).to(torch.bfloat16)


def _dequantize_nvfp4_float32(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
) -> torch.Tensor:
    """Match the supplied BF16 dequantizer, then use FP32 for optimization."""

    return dequantize_nvfp4(quant_float, scale_float).to(torch.float32)


def _sample_rows(x: torch.Tensor, limit: int) -> torch.Tensor:
    """Deterministically sample at most ``limit`` rows without random state."""

    rows = int(x.shape[0])
    if rows <= limit:
        return x
    step = max(1, (rows + limit - 1) // limit)
    return x[::step][:limit]


def _safe_positive_vector(x: torch.Tensor, length: int) -> torch.Tensor:
    """Return a finite, positive FP32 vector of the requested length."""

    y = x.detach().to(dtype=torch.float32).reshape(-1)
    if int(y.numel()) != length:
        raise ValueError(f"Expected vector of length {length}, got {y.numel()}")
    return torch.nan_to_num(
        y, nan=1.0, posinf=1.0, neginf=1.0
    ).clamp_min(_EPS)


def _normalize_importance(
    importance: Optional[torch.Tensor],
    length: int,
) -> Optional[torch.Tensor]:
    if importance is None:
        return None
    w = importance.detach().to(dtype=torch.float32).reshape(-1)
    if int(w.numel()) != length:
        raise ValueError(
            f"Expected importance of length {length}, got {w.numel()}"
        )
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mean = w.mean()
    if float(mean) <= _EPS:
        return torch.ones_like(w)
    return (w / mean).clamp_min(_IMPORTANCE_FLOOR)


def _standard_block_losses(
    dense: torch.Tensor,
    importance: Optional[torch.Tensor],
) -> torch.Tensor:
    """Per-block importance-weighted squared error of standard HiF4."""

    prefix = tuple(int(v) for v in dense.shape[:-1])
    channels = int(dense.shape[-1])
    if channels % _HIF4_BLOCK_SIZE != 0:
        raise ValueError(
            f"Last dim {channels} is not divisible by HiF4 block size 64"
        )
    blocks = channels // _HIF4_BLOCK_SIZE

    x = torch.nan_to_num(
        dense.detach().to(torch.float32),
        nan=0.0,
        posinf=_E6M2_MAX * _HIF4_MAX_INNER,
        neginf=-_E6M2_MAX * _HIF4_MAX_INNER,
    )
    x_grouped = x.reshape(*prefix, blocks, 8, 2, 4)
    x_abs = x_grouped.abs()
    max4 = x_abs.amax(dim=-1)
    max8 = max4.amax(dim=-1)
    amax = max8.amax(dim=-1)
    _, standard_scale = _standard_e6m2_scale(amax)

    e2 = max8 >= (4.0 * standard_scale[..., None])
    scale_lv2 = 1.0 + e2.to(torch.float32)
    e3 = max4 >= (
        2.0 * standard_scale[..., None, None] * scale_lv2[..., None]
    )
    scale_lv3 = 1.0 + e3.to(torch.float32)
    denominator = (
        standard_scale[..., None, None, None]
        * scale_lv2[..., None, None]
        * scale_lv3[..., None]
    )
    mantissa = (
        torch.round(x_abs * (4.0 / denominator)).clamp_(0.0, 7.0) * 0.25
    )

    channel_importance = _normalize_importance(importance, channels)
    if channel_importance is None:
        weighted_error = (x_abs - mantissa * denominator).square()
    else:
        weighted_error = (
            (x_abs - mantissa * denominator).square()
            * channel_importance.reshape(*([1] * len(prefix)), blocks, 8, 2, 4)
        )
    return weighted_error.sum(dim=(-1, -2, -3)).reshape(-1)


def _loss_capture_ratio(
    losses: torch.Tensor,
    *,
    target: float,
    ratio_min: float,
) -> float:
    """Smallest fraction of the largest-loss blocks covering ``target`` of the
    total loss.  This converts the per-block loss tail into a refine budget."""

    losses = losses.detach().to(torch.float32).reshape(-1)
    total = float(losses.sum())
    if total <= _EPS:
        return float(ratio_min)
    sorted_descending = torch.sort(losses, descending=True).values
    cumulative = torch.cumsum(sorted_descending, dim=0)
    k = int((cumulative < float(target) * total).sum()) + 1
    return float(
        min(1.0, max(float(ratio_min), k / max(1, int(losses.numel()))))
    )


def _flat_group_gram(cov: torch.Tensor, channels: int) -> torch.Tensor:
    """Extract per-4-group 4x4 block-diagonal quadratic weights as a flat
    ``[channels // 4, 4, 4]`` tensor (the only part the solver needs)."""

    blocks = channels // _HIF4_BLOCK_SIZE
    g = cov.reshape(blocks, 64, blocks, 64)
    g = torch.diagonal(g, dim1=0, dim2=2).permute(2, 0, 1)
    g = g.reshape(blocks, 16, 4, 16, 4)
    g = torch.diagonal(g, dim1=1, dim2=3).permute(0, 3, 1, 2)
    return g.reshape(blocks * 16, 4, 4)


def _identity_permutation(length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(length, dtype=torch.int64, device=device)


def _hierarchy_aware_permutation(
    first_range: torch.Tensor,
    second_range: torch.Tensor,
) -> torch.Tensor:
    """Cluster similarly scaled channels for the 64/8/4 HiF4 hierarchy.

    The two ranges describe the paired operands of an exactly equivalent
    transform (X/W or Q/K). Log-domain median normalization makes the
    ordering insensitive to the operands' unrelated global units.
    """

    if tuple(first_range.shape) != tuple(second_range.shape):
        raise ValueError("Paired channel ranges must have identical shapes")
    log_first = torch.log2(first_range.to(torch.float32).clamp_min(_EPS))
    log_second = torch.log2(second_range.to(torch.float32).clamp_min(_EPS))
    log_first = log_first - torch.median(log_first)
    log_second = log_second - torch.median(log_second)
    pressure = torch.maximum(log_first, log_second).reshape(-1)
    if int(pressure.numel()) == 0:
        return torch.empty(0, dtype=torch.int64, device=pressure.device)
    if float(pressure.max() - pressure.min()) < 0.25:
        return _identity_permutation(int(pressure.numel()), pressure.device)
    return torch.argsort(pressure, descending=True)


def _headwise_hierarchy_permutation(
    q_range: torch.Tensor,
    k_range: torch.Tensor,
) -> torch.Tensor:
    """Return a local feature permutation for each paired Q/KV head."""

    if q_range.ndim != 2 or tuple(q_range.shape) != tuple(k_range.shape):
        raise ValueError("Headwise Q/K ranges must have shape [heads, head_dim]")
    q_log = torch.log2(q_range.to(torch.float32).clamp_min(_EPS))
    k_log = torch.log2(k_range.to(torch.float32).clamp_min(_EPS))
    q_log = q_log - q_log.median(dim=-1, keepdim=True).values
    k_log = k_log - k_log.median(dim=-1, keepdim=True).values
    pressure = torch.maximum(q_log, k_log)
    permutation = torch.argsort(pressure, dim=-1, descending=True)

    spread = pressure.amax(dim=-1) - pressure.amin(dim=-1)
    identity = torch.arange(
        int(pressure.shape[-1]), dtype=torch.int64, device=pressure.device
    ).expand_as(permutation)
    return torch.where(spread[:, None] >= 0.25, permutation, identity)


def _flatten_head_permutation(local_permutation: torch.Tensor) -> torch.Tensor:
    heads, head_dim = map(int, local_permutation.shape)
    base = torch.arange(
        heads, dtype=torch.int64, device=local_permutation.device
    )[:, None] * head_dim
    return (local_permutation.to(torch.int64) + base).reshape(-1)


def _candidate_is_safe(
    candidate: tuple[float, tuple[float, ...]],
    baseline: tuple[float, tuple[float, ...]],
    *,
    min_mean_improvement: float,
    worst_tolerance: float,
) -> bool:
    candidate_mean, candidate_cases = candidate
    baseline_mean, baseline_cases = baseline
    if not math.isfinite(candidate_mean):
        return False
    if candidate_mean > baseline_mean * (1.0 - min_mean_improvement):
        return False
    if len(candidate_cases) != len(baseline_cases):
        return False
    for current, reference in zip(candidate_cases, baseline_cases):
        if current > reference * (1.0 + worst_tolerance) + 1.0e-8:
            return False
    return True


def _center_attention_k(
    dense: torch.Tensor,
    num_heads: int,
    head_dim: int,
    center_mode: int,
) -> torch.Tensor:
    """Apply a token-invariant K shift; softmax(QK^T) is unchanged."""

    mode = int(center_mode)
    if mode == 0:
        return dense
    if dense.ndim != 2 or int(dense.shape[0]) <= 0:
        raise ValueError("Attention centering expects a non-empty 2D tensor")
    if int(dense.shape[1]) != int(num_heads) * int(head_dim):
        raise ValueError("Invalid dimensions for attention centering")
    grouped = dense.reshape(-1, int(num_heads), int(head_dim))
    if mode == 2:
        center = 0.5 * (
            grouped.amax(dim=0, keepdim=True)
            + grouped.amin(dim=0, keepdim=True)
        )
    else:
        raise ValueError("Unsupported attention center mode")
    return (grouped - center).reshape_as(dense)


def _e6m2_encode_nearest(value: torch.Tensor) -> torch.Tensor:
    """Encode non-negative FP32 values into finite unsigned E6M2 codes.

    Codes 0..254 are finite and monotonic.  Code 255 is NaN and is never
    produced.  Round-to-nearest-even is inherited from ``torch.round``.
    """

    x = torch.nan_to_num(
        value.detach().to(torch.float32),
        nan=_E6M2_MIN,
        posinf=_E6M2_MAX,
        neginf=_E6M2_MIN,
    ).clamp(min=_E6M2_MIN, max=_E6M2_MAX)

    exponent = torch.floor(torch.log2(x))
    base = torch.pow(2.0, exponent)
    mantissa_field = torch.round((x / base - 1.0) * 4.0).to(torch.int64)

    carry = mantissa_field >= 4
    exponent = exponent + carry.to(exponent.dtype)
    mantissa_field = torch.where(
        carry, torch.zeros_like(mantissa_field), mantissa_field
    ).clamp(min=0, max=3)

    exponent_field = (exponent.to(torch.int64) + 48).clamp(min=0, max=63)
    code = exponent_field * 4 + mantissa_field
    return code.clamp(min=0, max=254).to(torch.int16)


def _e6m2_decode(code: torch.Tensor) -> torch.Tensor:
    c = code.to(torch.int64).clamp(min=0, max=254)
    exponent_field = torch.bitwise_right_shift(c, 2)
    mantissa_field = torch.bitwise_and(c, 3)
    exponent = exponent_field.to(torch.float32) - 48.0
    return torch.pow(2.0, exponent) * (
        1.0 + mantissa_field.to(torch.float32) * 0.25
    )


def _standard_e6m2_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the official amax/7 base scale with a BF16 intermediate."""

    high_precision_scale = (
        amax.to(torch.bfloat16) * _BF16_ONE_SEVENTH
    ).to(torch.float32)
    code = _e6m2_encode_nearest(high_precision_scale)
    return code, _e6m2_decode(code)


def _offsets_as_tuple(offsets: Optional[Iterable[int]]) -> tuple[int, ...]:
    ordered = [0]
    if offsets is None:
        return (0,)
    if torch.is_tensor(offsets):
        values = offsets.detach().to("cpu").reshape(-1).tolist()
    else:
        values = list(offsets)
    for raw in values:
        value = int(raw)
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _solve_exact_hierarchy(
    x_abs: torch.Tensor,
    scale: torch.Tensor,
    importance: Optional[torch.Tensor],
    sign: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exactly solve lv2/lv3 for fixed scales using three loss tables.

    Args:
        x_abs: ``[num_blocks, 8, 2, 4]`` absolute values.
        scale: ``[num_blocks]`` finite E6M2 values.
        importance: optional tensor with the same shape as ``x_abs``.
        sign: ``[num_blocks, 8, 2, 4]`` signs (required with ``group_gram``).
        group_gram: ``[num_blocks, 8, 2, 4, 4]`` per-group quadratic weights;
            when given, the loss is the quadratic form ``delta^T G delta``
            instead of the diagonal per-channel weighted squares.
    """

    losses: list[torch.Tensor] = []
    mantissas: list[torch.Tensor] = []

    for total_exponent in (0, 1, 2):
        local_scale = scale[..., None, None, None] * float(1 << total_exponent)
        mant_code = torch.round(x_abs * (4.0 / local_scale)).clamp_(0.0, 7.0)
        mantissa = mant_code * 0.25
        if group_gram is not None:
            delta = sign * (x_abs - mantissa * local_scale)
            losses.append(
                torch.einsum(
                    "...abi,...abij,...abj->...ab", delta, group_gram, delta
                )
            )
        else:
            error = (x_abs - mantissa * local_scale).square()
            if importance is not None:
                error = error * importance
            losses.append(error.sum(dim=-1))
        mantissas.append(mantissa)

    loss_0, loss_1, loss_2 = losses
    choose_01 = loss_1 < loss_0
    choose_12 = loss_2 < loss_1

    cost_e2_0 = torch.minimum(loss_0, loss_1).sum(dim=-1)
    cost_e2_1 = torch.minimum(loss_1, loss_2).sum(dim=-1)
    e2 = cost_e2_1 < cost_e2_0
    e3 = torch.where(e2[..., None], choose_12, choose_01)

    block_loss = torch.where(e2, cost_e2_1, cost_e2_0).sum(dim=-1)
    total_exponent = e2.to(torch.int64)[..., None] + e3.to(torch.int64)

    # [..., 3, 4]，指数维固定在倒数第二维（批量时随输入维度自然后移）。
    mantissa_stack = torch.stack(mantissas, dim=-2)
    gather_index = total_exponent[..., None, None].expand(
        *total_exponent.shape, 1, 4
    )
    gather_dim = mantissa_stack.ndim - 2  # 指数维：4D 输入为 3，批量时随维度后移
    mantissa = torch.gather(
        mantissa_stack, gather_dim, gather_index
    ).squeeze(-2)

    scale_lv2 = 1.0 + e2.to(torch.float32)
    scale_lv3 = 1.0 + e3.to(torch.float32)
    return block_loss, scale_lv2, scale_lv3, mantissa


def _pack_hif4_params(
    prefix: tuple[int, ...],
    blocks: int,
    scale_factor: torch.Tensor,
    scale_lv2: torch.Tensor,
    scale_lv3: torch.Tensor,
    sign: torch.Tensor,
    mantissa: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # Canonical zero: it is numerically irrelevant, but avoids relying on a
    # checker accepting sign=+/-1 when the final mantissa is zero.
    sign_out = sign.reshape(*prefix, blocks, 8, 2, 4)
    mantissa_out = mantissa.reshape(*prefix, blocks, 8, 2, 4)
    sign_out = torch.where(
        mantissa_out == 0.0, torch.zeros_like(sign_out), sign_out
    )
    return {
        "scale_factor": scale_factor.reshape(*prefix, blocks, 1, 1, 1),
        "scale_lv2": scale_lv2.reshape(*prefix, blocks, 8, 1, 1),
        "scale_lv3": scale_lv3.reshape(*prefix, blocks, 8, 2, 1),
        "sign": sign_out,
        "mant": mantissa_out,
    }


def _dense_to_hif4(
    dense: torch.Tensor,
    *,
    importance: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
    search_offsets: Optional[Union[Sequence[int], torch.Tensor]] = None,
    error_threshold: float = 0.0,
    accept_margin: float = 0.0,
    max_refine_ratio: float = 0.0,
    max_refine_blocks: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Quantize a dense tensor into valid HiF4 parameters."""

    if group_gram is not None:
        expected_gram_shape = dense.shape[:-1] + (
            dense.shape[-1] // 64,
            8,
            2,
            4,
            4,
        )
        if tuple(group_gram.shape) != tuple(expected_gram_shape):
            raise ValueError(
                f"group_gram shape {tuple(group_gram.shape)} does not match "
                f"expected {expected_gram_shape}"
            )

    if dense.ndim < 1:
        raise ValueError("dense must have at least one dimension")
    prefix = tuple(int(v) for v in dense.shape[:-1])
    channels = int(dense.shape[-1])
    if channels % _HIF4_BLOCK_SIZE != 0:
        raise ValueError(
            f"Last dim {channels} is not divisible by HiF4 block size 64"
        )
    blocks = channels // _HIF4_BLOCK_SIZE

    x = torch.nan_to_num(
        dense.detach().to(torch.float32),
        nan=0.0,
        posinf=_E6M2_MAX * _HIF4_MAX_INNER,
        neginf=-_E6M2_MAX * _HIF4_MAX_INNER,
    )
    x_grouped = x.reshape(*prefix, blocks, 8, 2, 4)
    x_abs = x_grouped.abs()
    sign = torch.sign(x_grouped)

    max4 = x_abs.amax(dim=-1)
    max8 = max4.amax(dim=-1)
    amax = max8.amax(dim=-1)
    standard_code, standard_scale = _standard_e6m2_scale(amax)

    e2 = max8 >= (4.0 * standard_scale[..., None])
    scale_lv2 = 1.0 + e2.to(torch.float32)
    e3 = max4 >= (
        2.0 * standard_scale[..., None, None] * scale_lv2[..., None]
    )
    scale_lv3 = 1.0 + e3.to(torch.float32)

    denominator = (
        standard_scale[..., None, None, None]
        * scale_lv2[..., None, None]
        * scale_lv3[..., None]
    )
    mantissa = (
        torch.round(x_abs * (4.0 / denominator)).clamp_(0.0, 7.0) * 0.25
    )

    offsets = _offsets_as_tuple(search_offsets)
    refine_ratio = max(0.0, min(float(max_refine_ratio), 1.0))
    if refine_ratio <= 0.0 or len(offsets) == 0:
        return _pack_hif4_params(
            prefix,
            blocks,
            standard_scale,
            scale_lv2,
            scale_lv3,
            sign,
            mantissa,
        )

    channel_importance = _normalize_importance(importance, channels)
    if channel_importance is not None:
        channel_importance = channel_importance.to(x.device)
    if group_gram is not None:
        delta = sign * (x_abs - mantissa * denominator)
        weighted_error = torch.einsum(
            "...abi,...abij,...abj->...ab", delta, group_gram, delta
        )
        weighted_energy = x_abs.square()
        importance_view = None
    elif channel_importance is None:
        weighted_error = (x_abs - mantissa * denominator).square()
        weighted_energy = x_abs.square()
        importance_view = None
    else:
        importance_view = channel_importance.reshape(
            *([1] * len(prefix)), blocks, 8, 2, 4
        )
        weighted_error = (x_abs - mantissa * denominator).square() * importance_view
        weighted_energy = x_abs.square() * importance_view

    loss_reduce_dims = (-1, -2) if group_gram is not None else (-1, -2, -3)
    standard_loss = weighted_error.sum(dim=loss_reduce_dims)
    energy = weighted_energy.sum(dim=(-1, -2, -3))
    normalized_error = standard_loss / (energy + _EPS)

    flat_norm = normalized_error.reshape(-1)
    flat_loss = standard_loss.reshape(-1)
    hard_mask = flat_norm > float(error_threshold)
    hard_indices = torch.nonzero(hard_mask, as_tuple=False).reshape(-1)
    if int(hard_indices.numel()) == 0:
        return _pack_hif4_params(
            prefix,
            blocks,
            standard_scale,
            scale_lv2,
            scale_lv3,
            sign,
            mantissa,
        )

    total_blocks = int(flat_norm.numel())
    refine_cap = max(1, int(math.ceil(total_blocks * refine_ratio)))
    if max_refine_blocks is not None:
        refine_cap = min(refine_cap, max(1, int(max_refine_blocks)))
    if int(hard_indices.numel()) > refine_cap:
        if _REFINE_RANK_BY_ABSOLUTE:
            # Rank by the block's absolute (importance-weighted) reconstruction
            # error, i.e. its true contribution to the output MSE, instead of
            # the normalized error: under a fixed refinement budget this
            # greedily maximizes the total MSE reduction (and hence the
            # competition score).
            hard_indices = torch.topk(flat_loss, k=refine_cap, largest=True).indices
        else:
            hard_indices = torch.topk(flat_norm, k=refine_cap, largest=True).indices

    x_flat = x_abs.reshape(-1, 8, 2, 4)
    x_hard = x_flat.index_select(0, hard_indices)
    standard_loss_hard = standard_loss.reshape(-1).index_select(0, hard_indices)
    standard_code_hard = standard_code.reshape(-1).index_select(0, hard_indices)

    best_loss = standard_loss_hard.clone()
    best_scale = standard_scale.reshape(-1).index_select(0, hard_indices).clone()
    best_lv2 = scale_lv2.reshape(-1, 8).index_select(0, hard_indices).clone()
    best_lv3 = scale_lv3.reshape(-1, 8, 2).index_select(0, hard_indices).clone()
    best_mantissa = mantissa.reshape(-1, 8, 2, 4).index_select(
        0, hard_indices
    ).clone()
    sign_hard = sign.reshape(-1, 8, 2, 4).index_select(0, hard_indices)
    group_gram_hard = (
        None
        if group_gram is None
        else group_gram.reshape(-1, 8, 2, 4, 4).index_select(0, hard_indices)
    )
    best_offset = torch.zeros(
        int(hard_indices.numel()), dtype=torch.int64, device=x.device
    )

    if channel_importance is None:
        importance_hard = None
    else:
        block_importance = channel_importance.reshape(blocks, 8, 2, 4)
        channel_block_ids = torch.remainder(hard_indices, blocks)
        importance_hard = block_importance.index_select(0, channel_block_ids)

    # 全部 offset 一次性批量求解：把 [N] 块沿 offset 维展开成 [K, N]，
    # 一次精确求解后按块取 argmin。标准 code（offset 0）必须保留在候选里：
    # 阈值式 lv2/lv3 与精确解不等价（真实数据约半数块有更低损失），
    # offset 0 会把 hard 块的 lv2/lv3 升级为精确解。
    offset_values = torch.tensor(
        [int(o) for o in offsets], dtype=torch.int64, device=x.device
    )
    expanded_codes = (
        standard_code_hard.to(torch.int64).unsqueeze(0)
        + offset_values.unsqueeze(1)
    ).clamp(min=0, max=254)
    candidate_scales = _e6m2_decode(expanded_codes)
    num_offsets = int(offset_values.numel())
    x_expanded = x_hard.unsqueeze(0).expand(
        num_offsets, -1, -1, -1, -1
    )
    sign_expanded = sign_hard.unsqueeze(0).expand(
        num_offsets, -1, -1, -1, -1
    )
    importance_expanded = (
        None
        if importance_hard is None
        else importance_hard.unsqueeze(0).expand(
            num_offsets, -1, -1, -1, -1
        )
    )
    gram_expanded = (
        None
        if group_gram_hard is None
        else group_gram_hard.unsqueeze(0).expand(
            num_offsets, -1, -1, -1, -1, -1
        )
    )
    all_losses, all_lv2, all_lv3, all_mantissa = _solve_exact_hierarchy(
        x_expanded,
        candidate_scales,
        importance_expanded,
        sign_expanded,
        gram_expanded,
    )
    best_k = all_losses.argmin(dim=0)
    hard_arange = torch.arange(
        int(hard_indices.numel()), device=x.device
    )
    candidate_loss = all_losses[best_k, hard_arange]
    candidate_scale = candidate_scales[best_k, hard_arange]
    candidate_lv2 = all_lv2[best_k, hard_arange]
    candidate_lv3 = all_lv3[best_k, hard_arange]
    candidate_mantissa = all_mantissa[best_k, hard_arange]

    improve = candidate_loss < best_loss
    best_loss = torch.where(improve, candidate_loss, best_loss)
    best_scale = torch.where(improve, candidate_scale, best_scale)
    best_lv2 = torch.where(improve[:, None], candidate_lv2, best_lv2)
    best_lv3 = torch.where(improve[:, None, None], candidate_lv3, best_lv3)
    best_mantissa = torch.where(
        improve[:, None, None, None], candidate_mantissa, best_mantissa
    )
    best_offset = torch.where(
        improve, offset_values[best_k], best_offset
    )

    if _REFINE_EDGE_EXTENSION and len(offsets) > 1:
        lo_offset = int(offsets[0])
        hi_offset = int(offsets[-1])

        def extend_edge(edge: int, direction: int) -> None:
            nonlocal best_loss, best_scale
            nonlocal best_lv2, best_lv3, best_mantissa, best_offset
            mask = best_offset == edge
            for _ in range(_REFINE_EDGE_EXTEND_STEPS):
                if not bool(mask.any()):
                    return
                edge_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
                target = edge + direction
                if target < -254 or target > 254:
                    return
                edge_code = (
                    standard_code_hard.index_select(0, edge_indices).to(
                        torch.int64
                    )
                    + target
                ).clamp(min=0, max=254)
                edge_scale = _e6m2_decode(edge_code)
                edge_importance = (
                    None
                    if importance_hard is None
                    else importance_hard.index_select(0, edge_indices)
                )
                edge_loss, edge_lv2, edge_lv3, edge_mantissa = (
                    _solve_exact_hierarchy(
                        x_hard.index_select(0, edge_indices),
                        edge_scale,
                        edge_importance,
                        sign_hard.index_select(0, edge_indices),
                        (
                            None
                            if group_gram_hard is None
                            else group_gram_hard.index_select(0, edge_indices)
                        ),
                    )
                )
                improve = edge_loss < best_loss.index_select(0, edge_indices)
                improved = edge_indices[improve]
                if int(improved.numel()) == 0:
                    return
                best_loss.index_copy_(
                    0, improved, edge_loss[improve]
                )
                best_scale.index_copy_(0, improved, edge_scale[improve])
                best_lv2.index_copy_(0, improved, edge_lv2[improve])
                best_lv3.index_copy_(0, improved, edge_lv3[improve])
                best_mantissa.index_copy_(
                    0, improved, edge_mantissa[improve]
                )
                best_offset.index_copy_(
                    0,
                    improved,
                    torch.full_like(best_offset[improved], target),
                )
                edge = target
                mask = best_offset == target

        extend_edge(hi_offset, +1)
        extend_edge(lo_offset, -1)

    margin = max(0.0, min(float(accept_margin), 0.99))
    accept = best_loss <= ((1.0 - margin) * standard_loss_hard)
    if not bool(torch.any(accept)):
        return _pack_hif4_params(
            prefix,
            blocks,
            standard_scale,
            scale_lv2,
            scale_lv3,
            sign,
            mantissa,
        )

    selected_indices = hard_indices[accept]
    out_scale = standard_scale.reshape(-1).clone()
    out_lv2 = scale_lv2.reshape(-1, 8).clone()
    out_lv3 = scale_lv3.reshape(-1, 8, 2).clone()
    out_mantissa = mantissa.reshape(-1, 8, 2, 4).clone()

    out_scale.index_copy_(0, selected_indices, best_scale[accept])
    out_lv2.index_copy_(0, selected_indices, best_lv2[accept])
    out_lv3.index_copy_(0, selected_indices, best_lv3[accept])
    out_mantissa.index_copy_(0, selected_indices, best_mantissa[accept])

    return _pack_hif4_params(
        prefix,
        blocks,
        out_scale,
        out_lv2,
        out_lv3,
        sign,
        out_mantissa,
    )


def _nvfp4_to_hif4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    *,
    multiplier: Optional[torch.Tensor] = None,
    permutation: Optional[torch.Tensor] = None,
    block_smooth_size: int = 0,
    block_smooth_seed: int = 0,
    center_mode: int = 0,
    center_num_heads: Optional[int] = None,
    center_head_dim: Optional[int] = None,
    importance: Optional[torch.Tensor] = None,
    group_gram: Optional[torch.Tensor] = None,
    search_offsets: Optional[Union[Sequence[int], torch.Tensor]] = None,
    error_threshold: float = 0.0,
    accept_margin: float = 0.0,
    max_refine_ratio: float = 0.0,
    max_refine_blocks: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    dense = _dequantize_nvfp4_float32(quant_float, scale_float)
    channels = int(dense.shape[-1])
    if int(center_mode) != 0:
        if center_num_heads is None or center_head_dim is None:
            raise ValueError("Attention centering requires head metadata")
        dense = _center_attention_k(
            dense,
            int(center_num_heads),
            int(center_head_dim),
            int(center_mode),
        )
    if multiplier is not None:
        scale = _safe_positive_vector(multiplier, channels).to(dense.device)
        dense.mul_(scale.reshape(*([1] * (dense.ndim - 1)), channels))
    if permutation is not None:
        order = permutation.detach().to(
            device=dense.device, dtype=torch.int64
        ).reshape(-1)
        if int(order.numel()) != channels:
            raise ValueError("Permutation width does not match tensor width")
        dense = dense.index_select(-1, order)
    if int(block_smooth_size) != 0:
        dense = _block_hadamard_transform(
            dense, int(block_smooth_size), int(block_smooth_seed)
        )
    gram = None
    if group_gram is not None:
        gram = group_gram.detach().to(
            device=dense.device, dtype=torch.float32
        )
        expected = (channels // 4, 4, 4)
        if tuple(gram.shape) != expected:
            raise ValueError(
                f"group_gram shape {tuple(gram.shape)} does not match "
                f"expected {expected}"
            )
        blocks = channels // _HIF4_BLOCK_SIZE
        gram = gram.reshape(blocks, 8, 2, 4, 4).unsqueeze(0).expand(
            int(dense.shape[0]), blocks, 8, 2, 4, 4
        )
    return _dense_to_hif4(
        dense,
        importance=importance,
        group_gram=gram,
        search_offsets=search_offsets,
        error_threshold=error_threshold,
        accept_margin=accept_margin,
        max_refine_ratio=max_refine_ratio,
        max_refine_blocks=max_refine_blocks,
    )


def _dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    dense = (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    )
    return dense.flatten(start_dim=-4, end_dim=-1)


def _smooth_scale(
    activation_amax: torch.Tensor,
    weight_amax: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    d = (activation_amax + _EPS).pow(alpha) / (
        weight_amax + _EPS
    ).pow(1.0 - alpha)
    d = torch.nan_to_num(
        d, nan=1.0, posinf=_SMOOTH_SCALE_MAX, neginf=_SMOOTH_SCALE_MIN
    )
    d = d.clamp(min=_SMOOTH_SCALE_MIN, max=_SMOOTH_SCALE_MAX)
    # A global normalization prevents an arbitrary overall scale drift while
    # retaining the relative channel smoothing.
    geometric_mean = torch.exp(torch.log(d).mean())
    return (d / geometric_mean).clamp(
        min=_SMOOTH_SCALE_MIN, max=_SMOOTH_SCALE_MAX
    )


def _hadamard_matrix(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a normalized Sylvester Hadamard matrix (size 4/8/16)."""

    n = int(size)
    if n not in _BLOCK_SMOOTH_ALLOWED_SIZES:
        raise ValueError(
            "block_smooth_size must be one of "
            f"{_BLOCK_SMOOTH_ALLOWED_SIZES}, got {n}"
        )
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while int(h.shape[0]) < n:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0
        )
    return h * (1.0 / math.sqrt(float(n)))


def _block_signs(
    channels: int,
    size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic per-channel signs for the block-orthogonal transform.

    seed 0 保留 v2.0 已验证的绝对通道索引图案（与块尺寸无关）；seed>0 用
    独立 PRNG 生成每块恰好半正半负的随机置乱，与 (size, seed) 强相关。
    修复旧公式 seed 0/1/2 完全退化、seed 3 仅差一个块的缺陷（当时整个
    block-S 搜索实际只测了一个全局符号图案）。
    """

    blocks = channels // size
    if int(seed) == 0:
        indices = torch.arange(channels, dtype=torch.int64, device=device)
        bits = (indices * 1_103_515_245 + 12_345).bitwise_and(1 << 30)
        return torch.where(bits == 0, 1.0, -1.0).to(dtype=dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(0x9E3779B97F4A7C15 + int(seed) * 2_654_435_761 + size)
    # 每块一个均匀随机排列，前 size//2 个位置 +1，其余 -1 → 每块恰好半正半负，
    # 保证每块都被真实打乱（不出现整块同号导致白做）。
    rank = torch.rand(blocks, size, device=device, generator=generator).argsort(
        dim=1
    )
    return torch.where(
        rank < size // 2,
        torch.tensor(1.0, dtype=dtype, device=device),
        torch.tensor(-1.0, dtype=dtype, device=device),
    )


def _block_hadamard_transform(
    dense: torch.Tensor,
    block_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """Apply a deterministic signed orthogonal transform to feature blocks.

    The signs scramble channels within each block before the Hadamard, avoiding
    concentrating positively correlated channels in the DC coefficient.  Signs
    are rebuilt from ``(channels, block_size, seed)``, so calibration and
    dynamic quantization only share ``block_size`` and a small integer
    ``seed``.
    """

    size = int(block_size)
    if size == 0:
        return dense
    channels = int(dense.shape[-1])
    if channels % size != 0:
        raise ValueError(
            f"Feature width {channels} is not divisible by block size {size}"
        )
    signs = _block_signs(channels, size, int(seed), dense.device, dense.dtype)
    grouped = dense.reshape(*dense.shape[:-1], channels // size, size)
    grouped = grouped * signs.reshape(channels // size, size)
    h = _hadamard_matrix(size, dense.device, dense.dtype)
    return torch.matmul(grouped, h).reshape_as(dense)


def _block_average(moment: torch.Tensor, size: int) -> torch.Tensor:
    """Flatten a per-channel moment to its block-mean after an orthogonal rotation.

    After a block-Hadamard the per-channel diagonal importance is no longer the
    right weight (energy spreads inside each block), so refine importance uses
    the block mean instead.  ``size <= 0`` is a no-op.
    """

    if int(size) <= 0:
        return moment
    return moment.reshape(-1, int(size)).mean(dim=-1, keepdim=True).expand(
        -1, int(size)
    ).reshape(-1)


def _linear_pair_transform(
    dense: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int = 0,
    *,
    weight_side: bool,
) -> torch.Tensor:
    """Apply one side of the exactly equivalent Linear transform."""

    scale = d if weight_side else d.reciprocal()
    transformed = (dense * scale.unsqueeze(0)).index_select(-1, permutation)
    return _block_hadamard_transform(
        transformed, block_smooth_size, block_smooth_seed
    )


def _transformed_second_moment(
    second_moment: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int = 0,
) -> torch.Tensor:
    """Diagonal covariance after scale/permutation/block rotation.

    Without a full covariance the diagonal after a normalized Hadamard is the
    mean variance of each block.  The full covariance path below is still used
    for the quadratic weight solver once a candidate has been selected.
    """

    moment = (second_moment / d.square()).index_select(0, permutation)
    size = int(block_smooth_size)
    if size != 0:
        moment = moment.reshape(-1, size).mean(dim=-1, keepdim=True).expand(
            -1, size
        ).reshape(-1)
    return moment


def _transformed_covariance(
    covariance: torch.Tensor,
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int,
    block_smooth_seed: int = 0,
) -> torch.Tensor:
    """Full activation covariance after the equivalent transform."""

    scale = d.reciprocal().to(dtype=covariance.dtype)
    cov = covariance * scale.unsqueeze(0) * scale.unsqueeze(1)
    cov = cov.index_select(0, permutation).index_select(1, permutation)
    size = int(block_smooth_size)
    if size != 0:
        cov = _block_hadamard_transform(cov, size, block_smooth_seed)
        cov = _block_hadamard_transform(
            cov.t(), size, block_smooth_seed
        ).t()
    return cov


def _linear_candidate_metrics(
    weight: torch.Tensor,
    activation_second_moment: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int = 0,
    block_smooth_seed: int = 0,
) -> tuple[float, tuple[float, ...]]:
    """Score an equivalent Linear transform from operand-side statistics."""

    channels = int(weight.shape[1])
    order = permutation.to(device=weight.device, dtype=torch.int64).reshape(-1)
    if int(order.numel()) != channels:
        raise ValueError("Linear candidate permutation has an invalid width")

    weight_smooth = _linear_pair_transform(
        weight,
        d,
        order,
        block_smooth_size,
        block_smooth_seed,
        weight_side=True,
    )
    h_x = _transformed_second_moment(
        activation_second_moment, d, order, block_smooth_size
    )
    weight_params = _dense_to_hif4(weight_smooth)
    weight_hat = _dequantize_hif4(weight_params)

    weight_error = (
        (weight_smooth - weight_hat).square() * h_x.unsqueeze(0)
    ).sum()
    weight_energy = (weight_smooth.square() * h_x.unsqueeze(0)).sum()
    weight_score = weight_error / (weight_energy + _EPS)

    h_w = _normalize_importance(weight_hat.square().sum(dim=0), channels)
    if h_w is None:
        h_w = torch.ones(channels, dtype=torch.float32, device=weight.device)

    case_scores: list[float] = []
    for sample in activation_samples:
        smooth = _linear_pair_transform(
            sample,
            d,
            order,
            block_smooth_size,
            block_smooth_seed,
            weight_side=False,
        )
        params = _dense_to_hif4(smooth)
        reconstructed = _dequantize_hif4(params)
        error = ((smooth - reconstructed).square() * h_w.unsqueeze(0)).sum()
        energy = (smooth.square() * h_w.unsqueeze(0)).sum()
        score = torch.nan_to_num(
            weight_score + error / (energy + _EPS),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )
        case_scores.append(float(score))

    if not case_scores:
        case_scores.append(float(torch.nan_to_num(weight_score, nan=1.0e30)))
    mean_score = sum(case_scores) / float(len(case_scores))
    return mean_score, tuple(case_scores)


def _linear_output_candidate_metrics(
    weight: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
    block_smooth_size: int = 0,
    block_smooth_seed: int = 0,
    activation_second_moment: Optional[torch.Tensor] = None,
    use_final_quantizer: bool = False,
) -> tuple[float, tuple[float, ...]]:
    """Score a transform by the actual sampled Linear output error.

    Operand-local reconstruction error is a useful cheap proxy for diagonal
    smoothing, but it misses cancellation between activation and weight errors
    after a non-diagonal transform.  Block-S candidates therefore use the
    end-to-end sampled objective that the competition ultimately measures.

    Deployment refines both operands with offsets + importance + refine, so
    ranking candidates with the plain HiF4 quantizer is an objective mismatch.
    When use_final_quantizer is set, weight candidates use _WEIGHT_OFFSETS with
    activation-second-moment importance and activation candidates use
    _DYNAMIC_OFFSETS with weight-gram importance, mirroring deployment (v2.5).
    """

    channels = int(weight.shape[1])
    order = permutation.to(device=weight.device, dtype=torch.int64).reshape(-1)
    weight_transformed = _linear_pair_transform(
        weight,
        d,
        order,
        block_smooth_size,
        block_smooth_seed,
        weight_side=True,
    )
    if use_final_quantizer:
        h_x = _normalize_importance(
            _transformed_second_moment(
                activation_second_moment, d, order, block_smooth_size
            ),
            channels,
        )
        weight_params = _dense_to_hif4(
            weight_transformed,
            importance=h_x,
            search_offsets=_WEIGHT_OFFSETS,
            error_threshold=_WEIGHT_REFINE_ERROR_THRESHOLD,
            accept_margin=_WEIGHT_REFINE_ACCEPT_MARGIN,
            max_refine_ratio=_LINEAR_CANDIDATE_REFINE_RATIO,
            max_refine_blocks=_LINEAR_CANDIDATE_REFINE_BLOCKS,
        )
    else:
        weight_params = _dense_to_hif4(weight_transformed)
    weight_hat = _dequantize_hif4(weight_params)

    h_w = _normalize_importance(weight_hat.square().sum(dim=0), channels)
    if h_w is None:
        h_w = torch.ones(channels, dtype=torch.float32, device=weight.device)

    case_scores: list[float] = []
    for sample in activation_samples:
        activation_transformed = _linear_pair_transform(
            sample,
            d,
            order,
            block_smooth_size,
            block_smooth_seed,
            weight_side=False,
        )
        if use_final_quantizer:
            activation_params = _dense_to_hif4(
                activation_transformed,
                importance=h_w,
                search_offsets=_DYNAMIC_OFFSETS,
                error_threshold=_ACTIVATION_REFINE_ERROR_THRESHOLD,
                accept_margin=_ACTIVATION_REFINE_ACCEPT_MARGIN,
                max_refine_ratio=_LINEAR_CANDIDATE_REFINE_RATIO,
                max_refine_blocks=_LINEAR_CANDIDATE_REFINE_BLOCKS,
            )
        else:
            activation_params = _dense_to_hif4(activation_transformed)
        activation_hat = _dequantize_hif4(activation_params)
        reference = activation_transformed.mm(weight_transformed.t())
        reconstructed = activation_hat.mm(weight_hat.t())
        score = (reference - reconstructed).square().sum() / (
            reference.square().sum() + _EPS
        )
        case_scores.append(
            float(
                torch.nan_to_num(
                    score, nan=1.0e30, posinf=1.0e30, neginf=1.0e30
                )
            )
        )
    if not case_scores:
        return 1.0e30, (1.0e30,)
    return sum(case_scores) / float(len(case_scores)), tuple(case_scores)


def _cpu_state_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        x.detach().to(device="cpu", dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).contiguous()


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """Calibrate SmoothQuant state and quantize a static Linear weight."""

    if not isinstance(calib_activation_list, list) or not calib_activation_list:
        raise ValueError("calib_activation_list must be a non-empty list")

    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale)
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor [out_features, in_features]")
    out_features, in_features = map(int, weight.shape)
    if in_features % _HIF4_BLOCK_SIZE != 0:
        raise ValueError("in_features must be divisible by 64")

    sum_square = torch.zeros(in_features, dtype=torch.float32, device=weight.device)
    activation_amax = torch.zeros_like(sum_square)
    token_count = 0
    activation_samples: list[torch.Tensor] = []
    use_quadratic = (
        _WEIGHT_QUADRATIC
        and in_features <= _WEIGHT_QUADRATIC_MAX_FEATURES
    )
    if use_quadratic:
        cov_sum = torch.zeros(
            in_features, in_features, dtype=torch.float32, device=weight.device
        )

    for pair in calib_activation_list:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each calibration activation must be a (quant, scale) pair")
        activation = _dequantize_nvfp4_float32(pair[0], pair[1])
        if activation.ndim != 2 or int(activation.shape[1]) != in_features:
            raise ValueError("Calibration activation shape is incompatible with weight")
        stats_sample = _sample_rows(activation, _LINEAR_STATS_TOKENS)
        sum_square += stats_sample.square().sum(dim=0)
        if use_quadratic:
            cov_sum += stats_sample.t().mm(stats_sample)
        activation_amax = torch.maximum(
            activation_amax, stats_sample.abs().amax(dim=0)
        )
        token_count += int(stats_sample.shape[0])
        activation_samples.append(
            _sample_rows(activation, _LINEAR_EVAL_TOKENS).clone()
        )

    activation_second_moment = sum_square / float(max(token_count, 1))
    weight_amax = weight.abs().amax(dim=0)
    activation_rms = torch.sqrt(activation_second_moment.clamp_min(_EPS))
    weight_rms = torch.sqrt((weight * weight).mean(dim=0).clamp_min(_EPS))

    identity_d = torch.ones(
        in_features, dtype=torch.float32, device=weight.device
    )
    identity_perm = _identity_permutation(in_features, weight.device)
    smooth_candidates = [identity_d]
    smooth_alphas = (
        _WEIGHT_SMOOTH_ALPHAS_WIDE
        if (
            in_features >= _WIDE_LAYER_MIN_DIM
            or out_features >= _WIDE_LAYER_MIN_DIM
        )
        else _WEIGHT_SMOOTH_ALPHAS
    )
    for alpha in smooth_alphas:
        smooth_candidates.append(
            _smooth_scale(activation_amax, weight_amax, alpha)
        )
        if _WEIGHT_SMOOTH_RMS:
            smooth_candidates.append(
                _smooth_scale(activation_rms, weight_rms, alpha)
            )

    # Candidate search touches only sampled output rows.  The selected
    # transform is then applied to the full Weight exactly once.
    weight_sample = _sample_rows(weight, _LINEAR_WEIGHT_EVAL_ROWS)
    baseline_metrics = _linear_candidate_metrics(
        weight_sample,
        activation_second_moment,
        activation_samples,
        identity_d,
        identity_perm,
    )
    best_metrics = baseline_metrics
    best_d = identity_d
    best_perm = identity_perm
    best_block_smooth_size = 0
    best_block_smooth_seed = 0

    for candidate_index, candidate_d in enumerate(smooth_candidates):
        candidate_permutations = [identity_perm]
        sorted_perm = _hierarchy_aware_permutation(
            activation_amax / candidate_d,
            weight_amax * candidate_d,
        )
        if not torch.equal(sorted_perm, identity_perm):
            candidate_permutations.append(sorted_perm)

        for candidate_perm in candidate_permutations:
            if candidate_index == 0 and torch.equal(candidate_perm, identity_perm):
                continue
            metrics = _linear_candidate_metrics(
                weight_sample,
                activation_second_moment,
                activation_samples,
                candidate_d,
                candidate_perm,
            )
            uses_reordering = not torch.equal(candidate_perm, identity_perm)
            if (
                metrics[0] < best_metrics[0]
                and _candidate_is_safe(
                    metrics,
                    baseline_metrics,
                    min_mean_improvement=0.02 if uses_reordering else 0.01,
                    worst_tolerance=0.005 if uses_reordering else 0.02,
                )
            ):
                best_metrics = metrics
                best_d = candidate_d
                best_perm = candidate_perm

    # Matrix SmoothQuant extension: within the channel groups selected above,
    # try non-diagonal block transforms of size 4/8/16.  The transform is a
    # deterministic signed Hadamard, hence exactly orthogonal and represented
    # in dynamic state by two small integers rather than a dense matrix.
    force_block_size = int(_BLOCK_SMOOTH_FORCE_SIZE)
    if force_block_size:
        candidate_block_sizes = (force_block_size,)
        candidate_seeds = _BLOCK_SMOOTH_SEEDS
    else:
        # 下投影（out < in，即 GPT-2 的 proj）允许更大 block 族 + 多样化 seed；
        # 其余层保持 4/8/16 与 seed-0 图案（v1.8 教训：窄层细搜索过拟合，
        # 8-batch 实测多样化 seed 在 fc -0.0037/o -0.0012）。proj 扩展只在
        # 基础 _BLOCK_SMOOTH_SIZES 非空时生效，否则继承 set_config 的禁用。
        is_proj = out_features < in_features and bool(_BLOCK_SMOOTH_SIZES)
        candidate_block_sizes = (
            _BLOCK_SMOOTH_PROJ_SIZES if is_proj else _BLOCK_SMOOTH_SIZES
        )
        candidate_seeds = (
            _BLOCK_SMOOTH_SEEDS if is_proj else _BLOCK_SMOOTH_NARROW_SEEDS
        )
    # v2.5 逐算子诊断：refine 落地量化器排序在 proj（GELU 后激活、结构稀疏）
    # 显著更好（8-batch 12 层中 7 层胜，净 +0.0080），在 fc（LN 后激活、平滑）
    # 却更差（净 -0.0071，且会把 fc 误判回 identity）。因此只在 down-projection
    # 上启用最终量化器排序，其余算子保持标准量化器。
    use_refined_block_ranking = out_features < in_features and bool(
        _BLOCK_SMOOTH_SIZES
    )
    forced_choice: Optional[tuple[tuple[float, tuple[float, ...]], int, int]] = None
    block_baseline_metrics = _linear_output_candidate_metrics(
        weight_sample,
        activation_samples,
        best_d,
        best_perm,
        activation_second_moment=activation_second_moment,
        use_final_quantizer=use_refined_block_ranking,
    )
    block_best_metrics = block_baseline_metrics
    if candidate_block_sizes:
        for candidate_size in candidate_block_sizes:
            size = int(candidate_size)
            if size <= 0 or in_features % size != 0:
                continue
            for candidate_seed in candidate_seeds:
                seed = int(candidate_seed)
                block_metrics = _linear_output_candidate_metrics(
                    weight_sample,
                    activation_samples,
                    best_d,
                    best_perm,
                    size,
                    seed,
                    activation_second_moment=activation_second_moment,
                    use_final_quantizer=use_refined_block_ranking,
                )
                if force_block_size:
                    if (
                        forced_choice is None
                        or block_metrics[0] < forced_choice[0][0]
                    ):
                        forced_choice = (block_metrics, size, seed)
                    continue
                if (
                    block_metrics[0] < block_best_metrics[0]
                    and _candidate_is_safe(
                        block_metrics,
                        block_baseline_metrics,
                        min_mean_improvement=_BLOCK_SMOOTH_MIN_IMPROVEMENT,
                        worst_tolerance=_BLOCK_SMOOTH_WORST_TOLERANCE,
                    )
                ):
                    block_best_metrics = block_metrics
                    best_block_smooth_size = size
                    best_block_smooth_seed = seed
    if forced_choice is not None:
        _, best_block_smooth_size, best_block_smooth_seed = forced_choice

    weight_smooth = _linear_pair_transform(
        weight,
        best_d,
        best_perm,
        best_block_smooth_size,
        best_block_smooth_seed,
        weight_side=True,
    )
    h_x_smooth = _transformed_second_moment(
        activation_second_moment,
        best_d,
        best_perm,
        best_block_smooth_size,
    )
    weight_group_gram = None
    if use_quadratic:
        gram = _transformed_covariance(
            cov_sum / float(max(token_count, 1)),
            best_d,
            best_perm,
            best_block_smooth_size,
            best_block_smooth_seed,
        )
        blocks = in_features // _HIF4_BLOCK_SIZE
        weight_group_gram = _flat_group_gram(gram, in_features).reshape(
            blocks, 8, 2, 4, 4
        ).unsqueeze(0).expand(
            int(weight.shape[0]), blocks, 8, 2, 4, 4
        )
    weight_params = _dense_to_hif4(
        weight_smooth,
        importance=h_x_smooth,
        group_gram=weight_group_gram,
        search_offsets=_WEIGHT_OFFSETS,
        error_threshold=_WEIGHT_REFINE_ERROR_THRESHOLD,
        accept_margin=_WEIGHT_REFINE_ACCEPT_MARGIN,
        max_refine_ratio=(
            _WEIGHT_REFINE_MAX_RATIO_SMALL
            if int(weight.numel()) <= 4_194_304
            else _WEIGHT_REFINE_MAX_RATIO_LARGE
        ),
        max_refine_blocks=_WEIGHT_REFINE_MAX_BLOCKS,
    )

    weight_hat = _dequantize_hif4(weight_params)
    activation_importance = _normalize_importance(
        weight_hat.square().sum(dim=0), in_features
    )
    if activation_importance is None:
        activation_importance = torch.ones_like(best_d)

    permutation_state = None
    if not torch.equal(best_perm, identity_perm):
        permutation_state = best_perm.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
    smooth_inv_state = None
    if not torch.equal(best_d, identity_d):
        smooth_inv_state = _cpu_state_tensor(best_d.reciprocal())

    if _DATA_DRIVEN_RATIO:
        loss_parts = []
        for sample in activation_samples:
            transformed = sample.to(dtype=torch.float32)
            if smooth_inv_state is not None:
                transformed = transformed * smooth_inv_state.reshape(1, -1)
            if permutation_state is not None:
                transformed = transformed.index_select(-1, permutation_state)
            if best_block_smooth_size != 0:
                transformed = _block_hadamard_transform(
                    transformed,
                    best_block_smooth_size,
                    best_block_smooth_seed,
                )
            loss_parts.append(
                _standard_block_losses(transformed, activation_importance)
            )
        activation_ratio = _loss_capture_ratio(
            torch.cat(loss_parts),
            target=_RATIO_CAPTURE_TARGET,
            ratio_min=_RATIO_MIN,
        )
    else:
        activation_ratio = _ACTIVATION_REFINE_MAX_RATIO

    activation_gram_state = None
    if (
        _ACTIVATION_QUADRATIC
        and in_features <= _ACTIVATION_QUADRATIC_MAX_FEATURES
    ):
        gram = weight_smooth.t().mm(weight_smooth)
        activation_gram_state = _cpu_state_tensor(
            _flat_group_gram(gram, in_features)
        )

    activation_state = {
        "smooth_inv": smooth_inv_state,
        "permutation": permutation_state,
        "block_smooth_size": int(best_block_smooth_size),
        "block_smooth_seed": int(best_block_smooth_seed),
        "importance": _cpu_state_tensor(activation_importance),
        "gram": activation_gram_state,
        "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
        "error_threshold": _ACTIVATION_REFINE_ERROR_THRESHOLD,
        "accept_margin": _ACTIVATION_REFINE_ACCEPT_MARGIN,
        "max_refine_ratio": float(activation_ratio),
        "max_refine_blocks": _ACTIVATION_REFINE_MAX_BLOCKS,
        "in_features": int(in_features),
        "version": 3,
    }
    if _JOINT_REFINE_ENABLED:
        weight_params = _joint_refine_weight_params(
            weight_params, weight_smooth, calib_activation_list, activation_state
        )
    return {
        "weight_params": weight_params,
        "activation_state": activation_state,
    }


@torch.no_grad()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    if not isinstance(activation_state, dict):
        raise TypeError("activation_state must be a dict")
    channels = int(activation_quant.shape[-1])
    if channels != int(activation_state.get("in_features", -1)):
        raise ValueError("Activation hidden size does not match calibration state")
    return _nvfp4_to_hif4(
        activation_quant,
        activation_scale,
        multiplier=activation_state["smooth_inv"],
        permutation=activation_state["permutation"],
        block_smooth_size=int(activation_state.get("block_smooth_size", 0)),
        block_smooth_seed=int(activation_state.get("block_smooth_seed", 0)),
        importance=activation_state["importance"],
        group_gram=activation_state.get("gram"),
        search_offsets=activation_state["offsets"],
        error_threshold=float(activation_state["error_threshold"]),
        accept_margin=float(activation_state["accept_margin"]),
        max_refine_ratio=float(activation_state["max_refine_ratio"]),
        max_refine_blocks=int(activation_state["max_refine_blocks"]),
    )


@torch.no_grad()
def _joint_refine_weight_params(
    weight_params: dict[str, torch.Tensor],
    weight_smooth: torch.Tensor,
    calib_activation_list: list,
    activation_state: dict,
) -> dict[str, torch.Tensor]:
    """X/W joint residual compensation (v2.6).

    标准 refine 最小化各块重建损失（Cx 加权）；本后处理额外把激活量化误差
    的交叉项纳入候选评估，直接最小化校准数据上的输出误差
    ``m_h = ||X·Wᵀ − Q(X)·Q(W)ᵀ||²``（这正是 score 度量的 m_h）。对块位置做
    Gauss-Seidel：块 b 的候选 δ 用 ΔE = +2a_bᵀδ + δᵀG_bδ 判定
    （a_b = XQᵀ·res、G_b = XQᵀXQ，delta = wq_old − cand），接受 ΔE<0，
    每步残差增量更新，保证逐步单调。8-batch 实测 6 个线性算子 72/72 层全胜。
    """
    out_features, in_features = map(int, weight_smooth.shape)
    blocks = in_features // _HIF4_BLOCK_SIZE
    if blocks <= 0 or not calib_activation_list:
        return weight_params
    device = weight_smooth.device
    smooth_inv = activation_state.get("smooth_inv")
    permutation = activation_state.get("permutation")
    bs = int(activation_state.get("block_smooth_size", 0))
    seed = int(activation_state.get("block_smooth_seed", 0))
    xt_list, xq_list = [], []
    for quant, scale in calib_activation_list:
        x = _dequantize_nvfp4_float32(quant, scale).to(torch.float32)
        if smooth_inv is not None:
            x = x * smooth_inv.reshape(1, -1)
        if permutation is not None:
            x = x.index_select(-1, permutation)
        if bs:
            x = _block_hadamard_transform(x, bs, seed)
        xt_list.append(x)
        xq_list.append(
            _dequantize_hif4(
                hif4_dynamic_quantize_activation(quant, scale, activation_state)
            )
        )
    if sum(int(x.shape[0]) for x in xt_list) < _JOINT_REFINE_MIN_TOKENS:
        return weight_params
    xt = torch.cat(xt_list, 0)
    xq = torch.cat(xq_list, 0)
    wt = weight_smooth
    out = out_features
    T = int(xt.shape[0])
    offsets = torch.tensor(_WEIGHT_OFFSETS, dtype=torch.int64, device=device)
    wt_abs = wt.abs().reshape(out, blocks, 8, 2, 4)
    amax = wt_abs.amax(dim=(-1, -2, -3))
    standard_code, _ = _standard_e6m2_scale(amax)
    xqb = xq.reshape(T, blocks, 64)
    wq = _dequantize_hif4(weight_params)
    res = xt @ wt.T - xq @ wq.T
    # 预计算每块位置：Gram 与各 offset 候选块（仅依赖 xq / wt，跨迭代不变）。
    gram_blocks = [xqb[:, b].T @ xqb[:, b] for b in range(blocks)]
    cand_blocks: list[torch.Tensor] = []
    for b in range(blocks):
        cands = []
        for o in offsets.tolist():
            cand_code = (standard_code[:, b] + o).clamp(0, 254)
            cand_scale = _e6m2_decode(cand_code)
            _, lv2, lv3, mant = _solve_exact_hierarchy(
                wt_abs[:, b], cand_scale, None, weight_params["sign"][:, b], None
            )
            lv2 = lv2.reshape(out, 8, 1, 1)
            lv3 = lv3.reshape(out, 8, 2, 1)
            mant = mant.reshape(out, 8, 2, 4)
            cands.append(
                (weight_params["sign"][:, b] * mant * lv3 * lv2
                 * cand_scale[:, None, None, None]).reshape(out, 64)
            )
        cand_blocks.append(torch.stack(cands, 0))  # [n_off, out, 64]
    arange = torch.arange(out, device=device)
    n_off = len(offsets)
    for _ in range(_JOINT_REFINE_ITER):
        any_change = False
        for b in range(blocks):
            A = xqb[:, b].T @ res
            wq_b = wq[:, b * 64:(b + 1) * 64]
            best_dE = torch.zeros(out, dtype=wt.dtype, device=device)
            best_idx = torch.zeros(out, dtype=torch.int64, device=device)
            for k in range(n_off):
                delta = wq_b - cand_blocks[b][k]
                dE = (
                    2.0 * (A.T * delta).sum(dim=1)
                    + (delta * (delta @ gram_blocks[b])).sum(dim=1)
                )
                better = dE < best_dE
                best_dE = torch.where(better, dE, best_dE)
                best_idx = torch.where(
                    better, torch.full_like(best_idx, k), best_idx
                )
            accept = best_dE < 0.0
            if not bool(accept.any()):
                continue
            any_change = True
            cand = cand_blocks[b][best_idx, arange]
            # 先在写 wq 前捕获旧块并计算增量（wq_b 是 wq 的视图，写后即失效）。
            delta_applied = torch.where(
                accept[:, None], wq_b - cand, torch.zeros_like(wq_b)
            )
            wq[:, b * 64:(b + 1) * 64] = torch.where(
                accept[:, None], cand, wq_b
            )
            res = res + xqb[:, b] @ delta_applied.T
            cand_code = (standard_code[:, b] + offsets[best_idx]).clamp(0, 254)
            cand_scale = _e6m2_decode(cand_code)
            _, lv2, lv3, mant = _solve_exact_hierarchy(
                wt_abs[:, b], cand_scale, None, weight_params["sign"][:, b], None
            )
            lv2 = lv2.reshape(out, 8, 1, 1)
            lv3 = lv3.reshape(out, 8, 2, 1)
            mant = mant.reshape(out, 8, 2, 4)
            weight_params["scale_factor"][:, b, 0, 0, 0] = torch.where(
                accept, cand_scale, weight_params["scale_factor"][:, b, 0, 0, 0]
            )
            weight_params["scale_lv2"][:, b] = torch.where(
                accept[:, None, None, None],
                lv2, weight_params["scale_lv2"][:, b],
            )
            weight_params["scale_lv3"][:, b] = torch.where(
                accept[:, None, None, None],
                lv3, weight_params["scale_lv3"][:, b],
            )
            weight_params["mant"][:, b] = torch.where(
                accept[:, None, None, None],
                mant, weight_params["mant"][:, b],
            )
        if not any_change:
            break
    weight_params["sign"] = torch.where(
        weight_params["mant"] == 0.0,
        torch.zeros_like(weight_params["sign"]),
        weight_params["sign"],
    )
    return weight_params


def _causal_attention_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Causal GQA attention output from dense [T, C] Q/K/V tensors.

    Mirrors the evaluation harness operator exactly (causal softmax over source
    positions, GQA K/V shared across each q-head group), so candidate scoring
    and the final competition score measure the same function.  Lives in
    solution.py because the submission must be standalone.
    """

    seq = int(q.shape[0])
    group = q_num_heads // kv_num_heads
    qh = q.reshape(seq, q_num_heads, head_dim)
    kh = k.reshape(seq, kv_num_heads, head_dim).repeat_interleave(group, dim=1)
    vh = v.reshape(seq, kv_num_heads, head_dim).repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->tsh", qh, kh) / math.sqrt(float(head_dim))
    mask = torch.triu(
        torch.full((seq, seq), float("-inf"), device=scores.device), 1
    ).unsqueeze(-1)
    probs = torch.softmax(scores + mask, dim=1)
    out = torch.einsum("tsh,shd->thd", probs, vh)
    return out.reshape(seq, q_num_heads * head_dim)


def _smooth_qk_scale(
    q_peak: torch.Tensor,
    k_peak: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    d = (k_peak + _EPS).pow(alpha) / (q_peak + _EPS).pow(1.0 - alpha)
    return torch.nan_to_num(
        d, nan=1.0, posinf=_QK_SMOOTH_MAX, neginf=_QK_SMOOTH_MIN
    ).clamp(min=_QK_SMOOTH_MIN, max=_QK_SMOOTH_MAX)


def _attention_candidate_metrics(
    q_samples: Sequence[torch.Tensor],
    k_samples: Sequence[torch.Tensor],
    v_samples: Sequence[torch.Tensor],
    d_kv: torch.Tensor,
    q_second_moment: torch.Tensor,
    k_effective_second_moment: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_permutation: torch.Tensor,
    k_permutation: torch.Tensor,
    center_mode: int,
    block_smooth_size: int = 0,
    block_smooth_seed: int = 0,
    use_final_quantizer: bool = True,
) -> tuple[float, tuple[float, ...]]:
    """End-to-end causal softmax output MSE for Q/K transform candidates.

    v2.3 replaced the old per-operator Q/K reconstruction proxy with the real
    ``softmax(AV)`` output error against the exact reference (V kept exact so the
    score isolates the Q/K contribution).  v2.4 makes the ranking quantizer the
    *final* deployed one (offsets + bounded refine + the candidate's own
    importance), so candidate selection and the competition score no longer
    disagree on which quantizer matters.  Q and K share the same block-orthogonal
    (size, seed), which preserves QK^T exactly before quantization.
    """

    group_size = q_num_heads // kv_num_heads
    d_q = d_kv.repeat_interleave(group_size, dim=0)
    d_k = d_kv.reciprocal()
    q_order = q_permutation.to(dtype=torch.int64, device=d_kv.device).reshape(-1)
    k_order = k_permutation.to(dtype=torch.int64, device=d_kv.device).reshape(-1)

    q_second_kv = q_second_moment.reshape(
        kv_num_heads, group_size, head_dim
    ).mean(dim=1)
    h_k = k_effective_second_moment * d_k.square()
    h_q = q_second_kv * d_kv.square()
    h_k_for_q = _normalize_importance(
        h_k.repeat_interleave(group_size, dim=0)
        .reshape(-1)
        .index_select(0, q_order),
        q_num_heads * head_dim,
    )
    h_q_for_k = _normalize_importance(
        h_q.reshape(-1).index_select(0, k_order),
        kv_num_heads * head_dim,
    )
    if h_k_for_q is None:
        h_k_for_q = torch.ones(q_num_heads * head_dim, dtype=torch.float32)
    if h_q_for_k is None:
        h_q_for_k = torch.ones(kv_num_heads * head_dim, dtype=torch.float32)
    if block_smooth_size:
        h_k_for_q = _block_average(h_k_for_q, block_smooth_size)
        h_q_for_k = _block_average(h_q_for_k, block_smooth_size)

    def quantize(
        dense: torch.Tensor,
        importance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not use_final_quantizer:
            return _dense_to_hif4(dense)
        return _dense_to_hif4(
            dense,
            importance=importance,
            search_offsets=_DYNAMIC_OFFSETS,
            error_threshold=_ATTN_REFINE_ERROR_THRESHOLD,
            accept_margin=_ATTN_CANDIDATE_ACCEPT_MARGIN,
            max_refine_ratio=_ATTN_CANDIDATE_REFINE_RATIO,
            max_refine_blocks=_ATTN_CANDIDATE_REFINE_BLOCKS,
        )

    ref_outputs = [
        _causal_attention_output(qs, ks, vs, q_num_heads, kv_num_heads, head_dim)
        for qs, ks, vs in zip(q_samples, k_samples, v_samples)
    ]

    case_scores: list[float] = []
    for q_sample, k_sample, v_sample, ref_out in zip(
        q_samples, k_samples, v_samples, ref_outputs
    ):
        q_smooth = _block_hadamard_transform(
            (q_sample * d_q.reshape(1, -1)).index_select(-1, q_order),
            block_smooth_size,
            block_smooth_seed,
        )
        k_centered = _center_attention_k(
            k_sample, kv_num_heads, head_dim, center_mode
        )
        k_smooth = _block_hadamard_transform(
            (k_centered * d_k.reshape(1, -1)).index_select(-1, k_order),
            block_smooth_size,
            block_smooth_seed,
        )
        q_hat = _dequantize_hif4(quantize(q_smooth, h_k_for_q))
        k_hat = _dequantize_hif4(quantize(k_smooth, h_q_for_k))
        out_h = _causal_attention_output(
            q_hat, k_hat, v_sample, q_num_heads, kv_num_heads, head_dim
        )
        case_scores.append(float(((out_h - ref_out).square()).mean()))

    if not case_scores:
        return 1.0e30, (1.0e30,)
    return sum(case_scores) / float(len(case_scores)), tuple(case_scores)


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Calibrate static Smooth-QK and output-sensitive Q/K weights."""

    if not isinstance(calib_qkv_list, list) or not calib_qkv_list:
        raise ValueError("calib_qkv_list must be a non-empty list")
    if q_num_heads <= 0 or kv_num_heads <= 0 or head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")
    q_channels = q_num_heads * head_dim
    kv_channels = kv_num_heads * head_dim
    if q_channels % 64 != 0 or kv_channels % 64 != 0:
        raise ValueError("Flattened Q/K/V dimensions must be divisible by 64")

    q_sum_square = torch.zeros(q_num_heads, head_dim, dtype=torch.float32)
    k_sum_square = torch.zeros(kv_num_heads, head_dim, dtype=torch.float32)
    k_mid_sum_square = torch.zeros_like(k_sum_square)
    q_peak_square = torch.zeros_like(q_sum_square)
    k_peak_square = torch.zeros_like(k_sum_square)
    k_mid_peak_square = torch.zeros_like(k_sum_square)
    q_token_count = 0
    k_token_count = 0
    sample_count = 0
    q_samples: list[torch.Tensor] = []
    k_samples: list[torch.Tensor] = []
    v_samples: list[torch.Tensor] = []

    for sample in calib_qkv_list:
        if not isinstance(sample, dict) or set(sample.keys()) != {"q", "k", "v"}:
            raise ValueError("Each attention calibration sample must contain q/k/v")
        q = _dequantize_nvfp4_float32(*sample["q"])
        k = _dequantize_nvfp4_float32(*sample["k"])
        if not isinstance(sample["v"], (tuple, list)) or len(sample["v"]) != 2:
            raise ValueError("V calibration data must be an NVFP4 pair")
        v_quant, v_scale = sample["v"]
        if not torch.is_tensor(v_quant) or not torch.is_tensor(v_scale):
            raise TypeError("V calibration pair must contain tensors")
        if q.ndim != 2 or k.ndim != 2 or v_quant.ndim != 2:
            raise ValueError("Q/K/V calibration tensors must be 2D")
        if int(q.shape[1]) != q_channels:
            raise ValueError("Q calibration width does not match head metadata")
        if int(k.shape[1]) != kv_channels or int(v_quant.shape[1]) != kv_channels:
            raise ValueError("K/V calibration width does not match head metadata")
        expected_v_scale_shape = (int(v_quant.shape[0]), kv_channels // 16)
        if tuple(v_scale.shape) != expected_v_scale_shape:
            raise ValueError("V calibration scale shape is invalid")
        if int(q.shape[0]) != int(k.shape[0]) or int(k.shape[0]) != int(v_quant.shape[0]):
            raise ValueError("Q/K/V in a calibration sample must share seq_len")

        q_stats = _sample_rows(q, _ATTN_STATS_TOKENS).reshape(
            -1, q_num_heads, head_dim
        )
        k_stats = _sample_rows(k, _ATTN_STATS_TOKENS).reshape(
            -1, kv_num_heads, head_dim
        )
        k_mid_stats = _center_attention_k(
            k_stats.reshape(-1, kv_channels),
            kv_num_heads,
            head_dim,
            2,
        ).reshape(-1, kv_num_heads, head_dim)
        q_sum_square += q_stats.square().sum(dim=0)
        k_sum_square += k_stats.square().sum(dim=0)
        k_mid_sum_square += k_mid_stats.square().sum(dim=0)
        q_peak_square += q_stats.abs().amax(dim=0).square()
        k_peak_square += k_stats.abs().amax(dim=0).square()
        k_mid_peak_square += k_mid_stats.abs().amax(dim=0).square()
        q_token_count += int(q_stats.shape[0])
        k_token_count += int(k_stats.shape[0])
        sample_count += 1
        q_samples.append(_sample_rows(q, _ATTN_EVAL_TOKENS).clone())
        k_samples.append(_sample_rows(k, _ATTN_EVAL_TOKENS).clone())
        v_dense = _dequantize_nvfp4_float32(v_quant, v_scale)
        v_samples.append(_sample_rows(v_dense, _ATTN_EVAL_TOKENS).clone())

    q_second_moment = q_sum_square / float(max(q_token_count, 1))
    k_second_moment = k_sum_square / float(max(k_token_count, 1))
    k_mid_second_moment = k_mid_sum_square / float(max(k_token_count, 1))
    q_peak = torch.sqrt(q_peak_square / float(max(sample_count, 1)))
    k_peak = torch.sqrt(k_peak_square / float(max(sample_count, 1)))
    k_mid_peak = torch.sqrt(k_mid_peak_square / float(max(sample_count, 1)))

    group_size = q_num_heads // kv_num_heads
    q_peak_kv = q_peak.reshape(kv_num_heads, group_size, head_dim).amax(dim=1)
    identity_d = torch.ones(
        kv_num_heads,
        head_dim,
        dtype=torch.float32,
        device=q_second_moment.device,
    )
    local_identity = torch.arange(
        head_dim, dtype=torch.int64, device=q_second_moment.device
    )[None, :].expand(kv_num_heads, -1)
    k_identity_perm = _flatten_head_permutation(local_identity)
    q_identity_perm = _flatten_head_permutation(
        local_identity.repeat_interleave(group_size, dim=0)
    )

    baseline_metrics = _attention_candidate_metrics(
        q_samples,
        k_samples,
        v_samples,
        identity_d,
        q_second_moment,
        k_second_moment,
        q_num_heads,
        kv_num_heads,
        head_dim,
        q_identity_perm,
        k_identity_perm,
        0,
    )
    best_metrics = baseline_metrics
    best_d = identity_d
    best_center_mode = 0
    best_q_perm = q_identity_perm
    best_k_perm = k_identity_perm

    # Midrange K-centering is an exact softmax invariance.  First select the
    # centering/smoothing pair with identity ordering, then test one hierarchy-
    # aware ordering for the selected pair to bound calibration time.
    for center_mode in _ATTN_CENTER_MODES:
        if center_mode in (2, 3):
            effective_second = k_mid_second_moment
            effective_peak = k_mid_peak
        else:
            effective_second = k_second_moment
            effective_peak = k_peak
        q_rms_kv = torch.sqrt(
            q_second_moment.reshape(
                kv_num_heads, group_size, head_dim
            ).mean(dim=1).clamp_min(_EPS)
        )
        k_rms = torch.sqrt(effective_second.clamp_min(_EPS))
        smooth_candidates = [identity_d]
        for alpha in _QK_SMOOTH_ALPHAS:
            smooth_candidates.append(
                _smooth_qk_scale(q_peak_kv, effective_peak, alpha)
            )
            if _QK_SMOOTH_RMS:
                smooth_candidates.append(
                    _smooth_qk_scale(q_rms_kv, k_rms, alpha)
                )
        for candidate_index, candidate_d in enumerate(smooth_candidates):
            if center_mode == 0 and candidate_index == 0:
                continue
            metrics = _attention_candidate_metrics(
                q_samples,
                k_samples,
                v_samples,
                candidate_d,
                q_second_moment,
                effective_second,
                q_num_heads,
                kv_num_heads,
                head_dim,
                q_identity_perm,
                k_identity_perm,
                center_mode,
            )
            if (
                metrics[0] < best_metrics[0]
                and _candidate_is_safe(
                    metrics,
                    baseline_metrics,
                    min_mean_improvement=_ATTN_SMOOTH_MIN_IMPROVEMENT,
                    worst_tolerance=_ATTN_SMOOTH_WORST_TOLERANCE,
                )
            ):
                best_metrics = metrics
                best_d = candidate_d
                best_center_mode = center_mode

    selected_k_peak = (
        k_mid_peak if best_center_mode in (2, 3) else k_peak
    )
    selected_k_second = (
        k_mid_second_moment if best_center_mode in (2, 3) else k_second_moment
    )
    local_permutation = _headwise_hierarchy_permutation(
        q_peak_kv * best_d,
        selected_k_peak * best_d.reciprocal(),
    )
    candidate_k_perm = _flatten_head_permutation(local_permutation)
    candidate_q_perm = _flatten_head_permutation(
        local_permutation.repeat_interleave(group_size, dim=0)
    )
    if not torch.equal(candidate_k_perm, k_identity_perm):
        permutation_metrics = _attention_candidate_metrics(
            q_samples,
            k_samples,
            v_samples,
            best_d,
            q_second_moment,
            selected_k_second,
            q_num_heads,
            kv_num_heads,
            head_dim,
            candidate_q_perm,
            candidate_k_perm,
            best_center_mode,
        )
        if (
            permutation_metrics[0] < best_metrics[0]
            and _candidate_is_safe(
                permutation_metrics,
                baseline_metrics,
                min_mean_improvement=_ATTN_PERM_MIN_IMPROVEMENT,
                worst_tolerance=_ATTN_PERM_WORST_TOLERANCE,
            )
        ):
            best_metrics = permutation_metrics
            best_q_perm = candidate_q_perm
            best_k_perm = candidate_k_perm

    # 等价块正交变换：Q/K 共享同一 (size, seed)，QK^T 在量化前严格不变，
    # 只在已选定的 d/置换之上枚举块大小×符号 seed，用端到端输出损失门控。
    best_block_size = 0
    best_block_seed = 0
    if _ATTN_BLOCK_SMOOTH_SIZES:
        for bsize in _ATTN_BLOCK_SMOOTH_SIZES:
            if head_dim % int(bsize) != 0:
                continue
            for bseed in _ATTN_BLOCK_SMOOTH_SEEDS:
                block_metrics = _attention_candidate_metrics(
                    q_samples,
                    k_samples,
                    v_samples,
                    best_d,
                    q_second_moment,
                    selected_k_second,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    best_q_perm,
                    best_k_perm,
                    best_center_mode,
                    int(bsize),
                    int(bseed),
                )
                if (
                    block_metrics[0] < best_metrics[0]
                    and _candidate_is_safe(
                        block_metrics,
                        baseline_metrics,
                        min_mean_improvement=_ATTN_BLOCK_MIN_IMPROVEMENT,
                        worst_tolerance=_ATTN_BLOCK_WORST_TOLERANCE,
                    )
                ):
                    best_metrics = block_metrics
                    best_block_size = int(bsize)
                    best_block_seed = int(bseed)

    d_q = best_d.repeat_interleave(group_size, dim=0)
    d_k = best_d.reciprocal()
    q_second_kv = q_second_moment.reshape(
        kv_num_heads, group_size, head_dim
    ).mean(dim=1)
    h_k = selected_k_second * d_k.square()
    h_q = q_second_kv * best_d.square()
    h_k_for_q = h_k.repeat_interleave(group_size, dim=0).reshape(-1)
    h_q_for_k = h_q.reshape(-1)
    h_k_for_q = _normalize_importance(
        h_k_for_q.index_select(0, best_q_perm), q_channels
    )
    h_q_for_k = _normalize_importance(
        h_q_for_k.index_select(0, best_k_perm), kv_channels
    )
    if h_k_for_q is None:
        h_k_for_q = torch.ones(q_channels, dtype=torch.float32)
    if h_q_for_k is None:
        h_q_for_k = torch.ones(kv_channels, dtype=torch.float32)
    # 块旋转后逐通道对角重要性不再准确（能量在块内摊平），refine 用块均值加权。
    if best_block_size:
        h_k_for_q = _block_average(h_k_for_q, best_block_size)
        h_q_for_k = _block_average(h_q_for_k, best_block_size)

    q_flat = d_q.reshape(-1)
    k_flat = d_k.reshape(-1)

    def q_transform(sample: torch.Tensor) -> torch.Tensor:
        return _block_hadamard_transform(
            (sample * q_flat.reshape(1, -1)).index_select(-1, best_q_perm),
            best_block_size,
            best_block_seed,
        )

    def k_transform(sample: torch.Tensor) -> torch.Tensor:
        return _block_hadamard_transform(
            (
                _center_attention_k(
                    sample, kv_num_heads, head_dim, int(best_center_mode)
                )
                * k_flat.reshape(1, -1)
            ).index_select(-1, best_k_perm),
            best_block_size,
            best_block_seed,
        )

    if _DATA_DRIVEN_RATIO:
        q_ratio = _loss_capture_ratio(
            torch.cat(
                [_standard_block_losses(q_transform(s), h_k_for_q)
                 for s in q_samples]
            ),
            target=_RATIO_CAPTURE_TARGET,
            ratio_min=_RATIO_MIN,
        )
        k_ratio = _loss_capture_ratio(
            torch.cat(
                [_standard_block_losses(k_transform(s), h_q_for_k)
                 for s in k_samples]
            ),
            target=_RATIO_CAPTURE_TARGET,
            ratio_min=_RATIO_MIN,
        )
        v_ratio = _loss_capture_ratio(
            torch.cat(
                [_standard_block_losses(s, None) for s in v_samples]
            ),
            target=_RATIO_CAPTURE_TARGET,
            ratio_min=_RATIO_MIN,
        )
    else:
        q_ratio = _Q_REFINE_MAX_RATIO
        k_ratio = _K_REFINE_MAX_RATIO
        v_ratio = _V_REFINE_MAX_RATIO

    q_permutation_state = None
    k_permutation_state = None
    if not torch.equal(best_k_perm, k_identity_perm):
        q_permutation_state = best_q_perm.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
        k_permutation_state = best_k_perm.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
    q_multiplier_state = None
    k_multiplier_state = None
    if not torch.equal(best_d, identity_d):
        q_multiplier_state = _cpu_state_tensor(d_q.reshape(-1))
        k_multiplier_state = _cpu_state_tensor(d_k.reshape(-1))

    q_state = {
        "multiplier": q_multiplier_state,
        "permutation": q_permutation_state,
        "block_smooth_size": int(best_block_size),
        "block_smooth_seed": int(best_block_seed),
        "importance": _cpu_state_tensor(h_k_for_q),
        "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
        "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
        "accept_margin": _Q_REFINE_ACCEPT_MARGIN,
        "max_refine_ratio": float(q_ratio),
        "max_refine_blocks": _Q_REFINE_MAX_BLOCKS,
        "num_heads": int(q_num_heads),
        "head_dim": int(head_dim),
        "version": 3,
    }
    k_state = {
        "multiplier": k_multiplier_state,
        "permutation": k_permutation_state,
        "center_mode": int(best_center_mode),
        "block_smooth_size": int(best_block_size),
        "block_smooth_seed": int(best_block_seed),
        "importance": _cpu_state_tensor(h_q_for_k),
        "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
        "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
        "accept_margin": _K_REFINE_ACCEPT_MARGIN,
        "max_refine_ratio": float(k_ratio),
        "max_refine_blocks": _K_REFINE_MAX_BLOCKS,
        "num_heads": int(kv_num_heads),
        "head_dim": int(head_dim),
        "version": 3,
    }
    v_state = {
        "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
        "importance": None,
        "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
        "accept_margin": _V_REFINE_ACCEPT_MARGIN,
        "max_refine_ratio": float(v_ratio),
        "max_refine_blocks": _V_REFINE_MAX_BLOCKS,
        "num_heads": int(kv_num_heads),
        "head_dim": int(head_dim),
        "version": 2,
    }
    return {"q_state": q_state, "k_state": k_state, "v_state": v_state}


def _check_attention_state(
    state: Any,
    num_heads: int,
    head_dim: int,
    name: str,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise TypeError(f"{name}_state must be a dict")
    if int(state.get("num_heads", -1)) != int(num_heads):
        raise ValueError(f"{name} head count does not match calibration state")
    if int(state.get("head_dim", -1)) != int(head_dim):
        raise ValueError(f"{name} head_dim does not match calibration state")
    return state


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(q_state, q_num_heads, head_dim, "q")
    if int(q_quant.shape[-1]) != q_num_heads * head_dim:
        raise ValueError("Q width does not match q_num_heads * head_dim")
    return _nvfp4_to_hif4(
        q_quant,
        q_scale,
        multiplier=state["multiplier"],
        permutation=state["permutation"],
        block_smooth_size=int(state.get("block_smooth_size", 0)),
        block_smooth_seed=int(state.get("block_smooth_seed", 0)),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )


@torch.no_grad()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(k_state, kv_num_heads, head_dim, "k")
    if int(k_quant.shape[-1]) != kv_num_heads * head_dim:
        raise ValueError("K width does not match kv_num_heads * head_dim")
    return _nvfp4_to_hif4(
        k_quant,
        k_scale,
        multiplier=state["multiplier"],
        permutation=state["permutation"],
        center_mode=int(state["center_mode"]),
        center_num_heads=kv_num_heads,
        center_head_dim=head_dim,
        block_smooth_size=int(state.get("block_smooth_size", 0)),
        block_smooth_seed=int(state.get("block_smooth_seed", 0)),
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )


@torch.no_grad()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    state = _check_attention_state(v_state, kv_num_heads, head_dim, "v")
    if int(v_quant.shape[-1]) != kv_num_heads * head_dim:
        raise ValueError("V width does not match kv_num_heads * head_dim")
    return _nvfp4_to_hif4(
        v_quant,
        v_scale,
        importance=state["importance"],
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )
