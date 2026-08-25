"""solution.py 的离线自测脚本。

只依赖 CPU 版 PyTorch，不联网、不用 GPU：

    /opt/anaconda3/bin/python3 test_solution.py

覆盖 solution.py 的全部公开入口，并用“先施加平滑/置换/居中变换、
再与 HiF4 反量化结果对比”的方式做数值校验。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solution as s  # noqa: E402

# ---- CPU 上跑得动、又尽量贴近真实模型的形状 -----------------------------
OUT_FEATURES = 512          # Linear 输出通道
IN_FEATURES = 1024          # Linear 输入通道（须能被 64 整除）
Q_HEADS, KV_HEADS, HEAD_DIM = 8, 4, 128
Q_CHANNELS = Q_HEADS * HEAD_DIM
KV_CHANNELS = KV_HEADS * HEAD_DIM
CALIB_SAMPLES = 3           # 标定样本数
CALIB_TOKENS = 2048         # 标定序列长度
DYNAMIC_TOKENS = 128        # 动态量化序列长度

# HiF4 是 64 元素块的 4-bit 格式，重建误差几个百分点是正常的；
# 阈值放宽到 15% 是为了避免随机数据下的偶发抖动。
MAX_REL_ERR = 0.15

torch.manual_seed(20260825)


def make_nvfp4(rows: int, cols: int, seed: int):
    """生成形状合规的随机 NVFP4 (quant, scale) 对。"""
    g = torch.Generator().manual_seed(seed)
    carriers = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    quant = carriers[torch.randint(0, len(carriers), (rows, cols), generator=g)]
    scale = torch.rand(rows, cols // 16, generator=g) * 0.15 + 0.005
    return quant, scale


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """归一化相对误差（L2 / L2）。"""
    denom = a.square().sum().clamp_min(1e-12)
    return float(((a - b).square().sum() / denom).sqrt())


def apply_state_transform(
    dense: torch.Tensor,
    *,
    multiplier: torch.Tensor | None = None,
    permutation: torch.Tensor | None = None,
    center_mode: int = 0,
    heads: int | None = None,
    head_dim: int | None = None,
) -> torch.Tensor:
    """复现量化前施加的等价变换，作为反量化结果的对照基准。"""
    x = dense
    if center_mode:
        x = s._center_attention_k(x, heads, head_dim, center_mode)
    if multiplier is not None:
        x = x * multiplier.reshape(1, -1)
    if permutation is not None:
        x = x.index_select(-1, permutation)
    return x


def test_dequantize_nvfp4() -> None:
    quant, scale = make_nvfp4(DYNAMIC_TOKENS, IN_FEATURES, 1)
    dense = s.dequantize_nvfp4(quant, scale)
    assert dense.shape == (DYNAMIC_TOKENS, IN_FEATURES)
    assert dense.dtype == torch.bfloat16
    assert torch.allclose(
        dense.to(torch.float32),
        quant * scale.unsqueeze(-1).repeat(1, 1, 16).reshape_as(quant),
        rtol=0.02,
        atol=1e-6,
    )

    # scale 形状错误时应直接报错
    bad_scale = torch.rand(DYNAMIC_TOKENS, IN_FEATURES // 16 + 1)
    try:
        s.dequantize_nvfp4(quant, bad_scale)
    except ValueError:
        pass
    else:
        raise AssertionError("dequantize_nvfp4 未拒绝错误的 scale 形状")
    print("dequantize_nvfp4              OK")


def test_weight_and_activation() -> None:
    w_quant, w_scale = make_nvfp4(OUT_FEATURES, IN_FEATURES, 2)
    calib = [
        make_nvfp4(CALIB_TOKENS, IN_FEATURES, 100 + i) for i in range(CALIB_SAMPLES)
    ]

    result = s.hif4_calibration_and_quantize_weight(w_quant, w_scale, calib)
    state = result["activation_state"]
    assert set(result.keys()) == {"weight_params", "activation_state"}

    weight_hat = s._dequantize_hif4(result["weight_params"]).to(torch.float32)
    weight_dense = s._dequantize_nvfp4_float32(w_quant, w_scale)
    err = rel_err(weight_dense, weight_hat)
    assert err < MAX_REL_ERR, f"weight 重建误差过大: {err:.3f}"
    print(f"weight 校准+量化             OK  rel_err={err:.4f}")

    # 同样输入再跑一遍，标定结果必须确定
    result2 = s.hif4_calibration_and_quantize_weight(w_quant, w_scale, calib)
    for key in result2["weight_params"]:
        assert torch.equal(
            result["weight_params"][key], result2["weight_params"][key]
        ), f"weight_params[{key}] 结果不确定"
    for key in ("smooth_inv", "permutation", "importance", "offsets"):
        a, b = state[key], result2["activation_state"][key]
        assert (a is None and b is None) or torch.equal(a, b), (
            f"activation_state[{key}] 结果不确定"
        )
    print("标定确定性                      OK")

    for i, (quant, scale) in enumerate(calib):
        out = s.hif4_dynamic_quantize_activation(quant, scale, state)
        ref = apply_state_transform(
            s._dequantize_nvfp4_float32(quant, scale),
            multiplier=state["smooth_inv"],
            permutation=state["permutation"],
        )
        hat = s._dequantize_hif4(out).to(torch.float32)
        err = rel_err(ref, hat)
        assert err < MAX_REL_ERR, f"activation 样本 {i} 重建误差过大: {err:.3f}"
    print(f"动态激活量化 x{CALIB_SAMPLES}              OK")

    # 空标定列表应报错
    try:
        s.hif4_calibration_and_quantize_weight(w_quant, w_scale, [])
    except ValueError:
        pass
    else:
        raise AssertionError("空标定列表未被拒绝")


def test_attention() -> None:
    calib_qkv = []
    for i in range(CALIB_SAMPLES):
        q, qs = make_nvfp4(CALIB_TOKENS, Q_CHANNELS, 200 + 3 * i)
        k, ks = make_nvfp4(CALIB_TOKENS, KV_CHANNELS, 201 + 3 * i)
        v, vs = make_nvfp4(CALIB_TOKENS, KV_CHANNELS, 202 + 3 * i)
        calib_qkv.append({"q": (q, qs), "k": (k, ks), "v": (v, vs)})

    attn = s.hif4_calibration_attention(
        calib_qkv, Q_HEADS, KV_HEADS, HEAD_DIM
    )
    assert set(attn.keys()) == {"q_state", "k_state", "v_state"}

    q, qs = make_nvfp4(DYNAMIC_TOKENS, Q_CHANNELS, 300)
    k, ks = make_nvfp4(DYNAMIC_TOKENS, KV_CHANNELS, 301)
    v, vs = make_nvfp4(DYNAMIC_TOKENS, KV_CHANNELS, 302)

    q_state = attn["q_state"]
    out = s.hif4_dynamic_quantize_q(q, qs, Q_HEADS, HEAD_DIM, q_state)
    ref = apply_state_transform(
        s._dequantize_nvfp4_float32(q, qs),
        multiplier=q_state["multiplier"],
        permutation=q_state["permutation"],
    )
    err = rel_err(ref, s._dequantize_hif4(out).to(torch.float32))
    assert err < MAX_REL_ERR, f"Q 重建误差过大: {err:.3f}"
    print(f"Q   动态量化                  OK  rel_err={err:.4f}")

    k_state = attn["k_state"]
    out = s.hif4_dynamic_quantize_k(k, ks, KV_HEADS, HEAD_DIM, k_state)
    ref = apply_state_transform(
        s._dequantize_nvfp4_float32(k, ks),
        multiplier=k_state["multiplier"],
        permutation=k_state["permutation"],
        center_mode=int(k_state["center_mode"]),
        heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )
    err = rel_err(ref, s._dequantize_hif4(out).to(torch.float32))
    assert err < MAX_REL_ERR, f"K 重建误差过大: {err:.3f}"
    print(f"K   动态量化                  OK  rel_err={err:.4f}")

    v_state = attn["v_state"]
    out = s.hif4_dynamic_quantize_v(v, vs, KV_HEADS, HEAD_DIM, v_state)
    err = rel_err(
        s._dequantize_nvfp4_float32(v, vs),
        s._dequantize_hif4(out).to(torch.float32),
    )
    assert err < MAX_REL_ERR, f"V 重建误差过大: {err:.3f}"
    print(f"V   动态量化                  OK  rel_err={err:.4f}")

    # 缺 v 的样本应报错
    bad = [{"q": (q, qs), "k": (k, ks)}]
    try:
        s.hif4_calibration_attention(bad, Q_HEADS, KV_HEADS, HEAD_DIM)
    except ValueError:
        pass
    else:
        raise AssertionError("缺少 v 的标定样本未被拒绝")


def main() -> int:
    try:
        test_dequantize_nvfp4()
        test_weight_and_activation()
        test_attention()
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {exc}")
        return 1
    print("\n全部离线测试通过（纯 CPU，无网络）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
