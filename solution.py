"""HiF4 solution for the 2026 Huawei algorithm competition.

The implementation keeps the official HiF4 conversion as an explicit fallback,
selects calibration-gated equivalent scaling/reordering transforms, and applies
bounded scale/hierarchy refinement to difficult blocks. All calibration states
are plain CPU data.
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
_WEIGHT_REFINE_ERROR_THRESHOLD = 1.0e-7
_WEIGHT_REFINE_ACCEPT_MARGIN = 0.005
_WEIGHT_REFINE_MAX_RATIO_SMALL = 1.0
_WEIGHT_REFINE_MAX_RATIO_LARGE = 1.0
_WEIGHT_REFINE_MAX_BLOCKS = 65_536

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

# Permutation search bases.  The initial hierarchy-aware ordering combines the
# paired operands via max(log range); real-data diagnostics show the operand
# with the larger quantization burden (usually the weight/K side) often yields
# a better single-sided ordering.  Each basis is evaluated with the exact
# paired metric and accepted only when it clears the same safety gate as the
# smoothing candidates.
_PERMUTATION_BASES = True


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


def _range_permutation(ranges: torch.Tensor) -> torch.Tensor:
    """1D argsort of log ranges; identity when the log spread is negligible."""

    log_r = torch.log2(ranges.to(torch.float32).clamp_min(_EPS))
    log_r = log_r - torch.median(log_r)
    flat = log_r.reshape(-1)
    if float(flat.max() - flat.min()) < 0.25:
        return _identity_permutation(int(flat.numel()), flat.device)
    return torch.argsort(flat, descending=True)


def _headwise_range_permutation(ranges: torch.Tensor) -> torch.Tensor:
    """Per-head argsort of log ranges (ranges: [heads, head_dim])."""

    log_r = torch.log2(ranges.to(torch.float32).clamp_min(_EPS))
    log_r = log_r - log_r.median(dim=-1, keepdim=True).values
    spread = log_r.amax(dim=-1) - log_r.amin(dim=-1)
    identity = torch.arange(
        int(ranges.shape[-1]), dtype=torch.int64, device=ranges.device
    ).expand_as(ranges)
    ordered = torch.argsort(log_r, dim=-1, descending=True)
    return torch.where(spread[:, None] >= 0.25, ordered, identity)


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
        local_scale = scale[:, None, None, None] * float(1 << total_exponent)
        mant_code = torch.round(x_abs * (4.0 / local_scale)).clamp_(0.0, 7.0)
        mantissa = mant_code * 0.25
        if group_gram is not None:
            delta = sign * (x_abs - mantissa * local_scale)
            losses.append(
                torch.einsum(
                    "nabi,nabij,nabj->nab", delta, group_gram, delta
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

    # [N, 8, 2, 3, 4], gather the mantissa matching k=e2+e3.
    mantissa_stack = torch.stack(mantissas, dim=3)
    gather_index = total_exponent[..., None, None].expand(-1, -1, -1, 1, 4)
    mantissa = torch.gather(mantissa_stack, 3, gather_index).squeeze(3)

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

    for offset in offsets:
        candidate_code = (
            standard_code_hard.to(torch.int64) + int(offset)
        ).clamp(min=0, max=254)
        candidate_scale = _e6m2_decode(candidate_code)
        candidate_loss, candidate_lv2, candidate_lv3, candidate_mantissa = (
            _solve_exact_hierarchy(
                x_hard,
                candidate_scale,
                importance_hard,
                sign_hard,
                group_gram_hard,
            )
        )

        improve = candidate_loss < best_loss
        best_loss = torch.where(improve, candidate_loss, best_loss)
        best_scale = torch.where(improve, candidate_scale, best_scale)
        best_lv2 = torch.where(improve[:, None], candidate_lv2, best_lv2)
        best_lv3 = torch.where(improve[:, None, None], candidate_lv3, best_lv3)
        best_mantissa = torch.where(
            improve[:, None, None, None], candidate_mantissa, best_mantissa
        )
        best_offset = torch.where(
            improve,
            torch.full_like(best_offset, int(offset)),
            best_offset,
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


def _linear_candidate_metrics(
    weight: torch.Tensor,
    activation_second_moment: torch.Tensor,
    activation_samples: Sequence[torch.Tensor],
    d: torch.Tensor,
    permutation: torch.Tensor,
) -> tuple[float, tuple[float, ...]]:
    """Score an equivalent Linear transform from operand-side statistics."""

    channels = int(weight.shape[1])
    order = permutation.to(device=weight.device, dtype=torch.int64).reshape(-1)
    if int(order.numel()) != channels:
        raise ValueError("Linear candidate permutation has an invalid width")

    weight_smooth = (weight * d.unsqueeze(0)).index_select(-1, order)
    h_x = (activation_second_moment / d.square()).index_select(0, order)
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
        smooth = (sample / d.unsqueeze(0)).index_select(-1, order)
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
    for alpha in _WEIGHT_SMOOTH_ALPHAS:
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

    # 置换基扩展：同一平滑 d 下比较 weight-only / activation-only 排序，
    # 诊断显示单侧排序常优于 max(log range) 组合排序。
    if _PERMUTATION_BASES:
        basis_ranges = {
            "w_amax": weight_amax * best_d,
            "x_amax": activation_amax / best_d,
            "w_rms": weight_rms * best_d,
            "x_rms": activation_rms / best_d,
        }
        seen = {tuple(best_perm.tolist())}
        for bname, b_range in basis_ranges.items():
            b_perm = _range_permutation(b_range)
            if torch.equal(b_perm, identity_perm):
                continue
            if tuple(b_perm.tolist()) in seen:
                continue
            seen.add(tuple(b_perm.tolist()))
            b_metrics = _linear_candidate_metrics(
                weight_sample,
                activation_second_moment,
                activation_samples,
                best_d,
                b_perm,
            )
            if (
                b_metrics[0] < best_metrics[0]
                and _candidate_is_safe(
                    b_metrics,
                    baseline_metrics,
                    min_mean_improvement=0.02,
                    worst_tolerance=0.005,
                )
            ):
                best_metrics = b_metrics
                best_perm = b_perm

    weight_smooth = (weight * best_d.unsqueeze(0)).index_select(
        -1, best_perm
    )
    h_x_smooth = (activation_second_moment / best_d.square()).index_select(
        0, best_perm
    )
    weight_group_gram = None
    if use_quadratic:
        gram = cov_sum / float(max(token_count, 1))
        d = best_d.to(gram.dtype)
        gram = gram / d.unsqueeze(0) / d.unsqueeze(1)
        gram = gram.index_select(0, best_perm).index_select(1, best_perm)
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
        "importance": _cpu_state_tensor(activation_importance),
        "gram": activation_gram_state,
        "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
        "error_threshold": _ACTIVATION_REFINE_ERROR_THRESHOLD,
        "accept_margin": _ACTIVATION_REFINE_ACCEPT_MARGIN,
        "max_refine_ratio": float(activation_ratio),
        "max_refine_blocks": _ACTIVATION_REFINE_MAX_BLOCKS,
        "in_features": int(in_features),
        "version": 2,
    }
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
        importance=activation_state["importance"],
        group_gram=activation_state.get("gram"),
        search_offsets=activation_state["offsets"],
        error_threshold=float(activation_state["error_threshold"]),
        accept_margin=float(activation_state["accept_margin"]),
        max_refine_ratio=float(activation_state["max_refine_ratio"]),
        max_refine_blocks=int(activation_state["max_refine_blocks"]),
    )


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
    d_kv: torch.Tensor,
    q_second_moment: torch.Tensor,
    k_effective_second_moment: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    q_permutation: torch.Tensor,
    k_permutation: torch.Tensor,
    center_mode: int,
) -> tuple[float, tuple[float, ...]]:
    """Q/K quantization proxy with GQA-aligned equivalent transforms."""

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
    h_k_for_q = h_k.repeat_interleave(group_size, dim=0).reshape(-1)
    h_q_for_k = h_q.reshape(-1)
    h_k_for_q = h_k_for_q.index_select(0, q_order)
    h_q_for_k = h_q_for_k.index_select(0, k_order)
    h_k_for_q = _normalize_importance(h_k_for_q, q_num_heads * head_dim)
    h_q_for_k = _normalize_importance(h_q_for_k, kv_num_heads * head_dim)
    if h_k_for_q is None or h_q_for_k is None:
        raise RuntimeError("Attention importance construction failed")

    case_scores: list[float] = []
    for q_sample, k_sample in zip(q_samples, k_samples):
        q_smooth = (q_sample * d_q.reshape(1, -1)).index_select(
            -1, q_order
        )
        k_centered = _center_attention_k(
            k_sample, kv_num_heads, head_dim, center_mode
        )
        k_smooth = (k_centered * d_k.reshape(1, -1)).index_select(
            -1, k_order
        )
        q_hat = _dequantize_hif4(_dense_to_hif4(q_smooth))
        k_hat = _dequantize_hif4(_dense_to_hif4(k_smooth))

        q_error = (
            (q_smooth - q_hat).square() * h_k_for_q.reshape(1, -1)
        ).sum()
        q_energy = (q_smooth.square() * h_k_for_q.reshape(1, -1)).sum()
        k_error = (
            (k_smooth - k_hat).square() * h_q_for_k.reshape(1, -1)
        ).sum()
        k_energy = (k_smooth.square() * h_q_for_k.reshape(1, -1)).sum()
        score = torch.nan_to_num(
            q_error / (q_energy + _EPS) + k_error / (k_energy + _EPS),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )
        case_scores.append(float(score))

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
                    min_mean_improvement=0.01,
                    worst_tolerance=0.02,
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
                min_mean_improvement=0.02,
                worst_tolerance=0.005,
            )
        ):
            best_metrics = permutation_metrics
            best_q_perm = candidate_q_perm
            best_k_perm = candidate_k_perm

    # 置换基扩展：单侧排序（Q-only / K-only）常优于 max(log range) 组合。
    if _PERMUTATION_BASES:
        basis_ranges = {
            "q_amax": q_peak_kv * best_d,
            "k_amax": selected_k_peak * best_d.reciprocal(),
        }
        seen = {tuple(best_k_perm.tolist())}
        for bname, b_range in basis_ranges.items():
            b_local = _headwise_range_permutation(b_range)
            b_k_perm = _flatten_head_permutation(b_local)
            if torch.equal(b_k_perm, k_identity_perm):
                continue
            if tuple(b_k_perm.tolist()) in seen:
                continue
            seen.add(tuple(b_k_perm.tolist()))
            b_q_perm = _flatten_head_permutation(
                b_local.repeat_interleave(group_size, dim=0)
            )
            b_metrics = _attention_candidate_metrics(
                q_samples,
                k_samples,
                best_d,
                q_second_moment,
                selected_k_second,
                q_num_heads,
                kv_num_heads,
                head_dim,
                b_q_perm,
                b_k_perm,
                best_center_mode,
            )
            if (
                b_metrics[0] < best_metrics[0]
                and _candidate_is_safe(
                    b_metrics,
                    baseline_metrics,
                    min_mean_improvement=0.02,
                    worst_tolerance=0.005,
                )
            ):
                best_metrics = b_metrics
                best_q_perm = b_q_perm
                best_k_perm = b_k_perm

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

    q_flat = d_q.reshape(-1)
    k_flat = d_k.reshape(-1)

    def q_transform(sample: torch.Tensor) -> torch.Tensor:
        return (sample * q_flat.reshape(1, -1)).index_select(-1, best_q_perm)

    def k_transform(sample: torch.Tensor) -> torch.Tensor:
        return (
            _center_attention_k(
                sample, kv_num_heads, head_dim, int(best_center_mode)
            )
            * k_flat.reshape(1, -1)
        ).index_select(-1, best_k_perm)

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
        "importance": _cpu_state_tensor(h_k_for_q),
        "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
        "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
        "accept_margin": _Q_REFINE_ACCEPT_MARGIN,
        "max_refine_ratio": float(q_ratio),
        "max_refine_blocks": _Q_REFINE_MAX_BLOCKS,
        "num_heads": int(q_num_heads),
        "head_dim": int(head_dim),
        "version": 2,
    }
    k_state = {
        "multiplier": k_multiplier_state,
        "permutation": k_permutation_state,
        "center_mode": int(best_center_mode),
        "importance": _cpu_state_tensor(h_q_for_k),
        "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
        "error_threshold": _ATTN_REFINE_ERROR_THRESHOLD,
        "accept_margin": _K_REFINE_ACCEPT_MARGIN,
        "max_refine_ratio": float(k_ratio),
        "max_refine_blocks": _K_REFINE_MAX_BLOCKS,
        "num_heads": int(kv_num_heads),
        "head_dim": int(head_dim),
        "version": 2,
    }
    v_state = {
        "offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),
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
        search_offsets=state["offsets"],
        error_threshold=float(state["error_threshold"]),
        accept_margin=float(state["accept_margin"]),
        max_refine_ratio=float(state["max_refine_ratio"]),
        max_refine_blocks=int(state["max_refine_blocks"]),
    )
