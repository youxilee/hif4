"""非 GPT-2 几何的冒烟验证（真实 GPT-2 数据重排）。

本地没有 LLaMA 权重，用真实 GPT-2 激活重排成关键几何差异：
  1. head_dim 64→128：head 两两合并，HiF4 64 块变成"半个 head"，
     验证 V 重要性/逐 head 置换/居中在块≠head 时仍正常；
     再按 GQA（6 q-head × 2 kv-head）组合。
  2. 宽 Linear：两层 fc/proj 权重与激活沿输入通道拼接，
     1536 / 6144 通道，验证 Gram 回退对角与大规模置换。

用法：
    /opt/anaconda3/bin/python3 shape_smoke.py [--layers 12] [--seq 128]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solution as s  # noqa: E402
from real_data_eval import (  # noqa: E402
    causal_attn,
    collect_real_data,
    score_attention,
    score_linear,
    to_gqa_kv,
)
from nvfp4_sim import nvfp4_encode  # noqa: E402


def merge_heads(x: torch.Tensor, qh: int, hd: int) -> torch.Tensor:
    """把 [seq, qh*hd] 的相邻 head 两两合并为 [seq, (qh//2)*(hd*2)]。"""
    return x.reshape(-1, qh // 2, 2, hd).reshape(-1, (qh // 2) * hd * 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--seq", type=int, default=128)
    args = ap.parse_args()

    model, weights, calib, test, qh, hd = collect_real_data(
        args.layers, args.seq, 2, 2
    )
    L = len(weights)
    hidden = model.config.n_embd

    # ---- 1) head_dim=128 + GQA 的 Attention 冒烟 ----
    hd2 = hd * 2
    qh2 = qh // 2
    kvh2 = 2
    attn_scores = []
    for i in range(L):
        qkv_calib = []
        for b in range(2):
            dense = calib["qkv"][b * L + i].reshape(-1, 3 * hidden)
            q_, k_, v_ = dense.chunk(3, dim=-1)
            qm = merge_heads(q_, qh, hd)          # [seq, qh2*hd2]
            km = merge_heads(k_, qh, hd)
            vm = merge_heads(v_, qh, hd)
            kg = to_gqa_kv(km, qh2, kvh2, hd2)    # [seq, kvh2*hd2]
            vg = to_gqa_kv(vm, qh2, kvh2, hd2)
            qkv_calib.append(
                {"q": nvfp4_encode(qm, "amax6"),
                 "k": nvfp4_encode(kg, "amax6"),
                 "v": nvfp4_encode(vg, "amax6")}
            )
        att = s.hif4_calibration_attention(qkv_calib, qh2, kvh2, hd2)
        qkv_test = []
        for b in range(2):
            dense = test["qkv"][b * L + i].reshape(-1, 3 * hidden)
            q_, k_, v_ = dense.chunk(3, dim=-1)
            qm = merge_heads(q_, qh, hd)
            km = merge_heads(k_, qh, hd)
            vm = merge_heads(v_, qh, hd)
            qkv_test.append(
                (nvfp4_encode(qm, "amax6"),
                 nvfp4_encode(to_gqa_kv(km, qh2, kvh2, hd2), "amax6"),
                 nvfp4_encode(to_gqa_kv(vm, qh2, kvh2, hd2), "amax6"))
            )
        attn_scores.append(
            score_attention(qkv_test, att["q_state"], att["k_state"],
                            att["v_state"], qh2, kvh2, hd2)
        )
    print(f"attn  hd={hd2} qh={qh2} kvh={kvh2}: "
          f"mean={sum(attn_scores)/L:.4f} min={min(attn_scores):.4f} "
          f"max={max(attn_scores):.4f}")

    # ---- 2) 宽 Linear：两层拼接（fc 1536-in，proj 6144-in）----
    for name, pair in (("fc", (0, 1)), ("proj", (0, 1))):
        i0, i1 = pair
        w = torch.cat([weights[i0][name], weights[i1][name]], dim=1)
        calib_pairs = []
        test_pairs = []
        for b in range(2):
            x_c = torch.cat(
                [calib["act"][name][b * L + i0],
                 calib["act"][name][b * L + i1]], dim=1
            )
            x_t = torch.cat(
                [test["act"][name][b * L + i0],
                 test["act"][name][b * L + i1]], dim=1
            )
            calib_pairs.append(nvfp4_encode(x_c, "amax6"))
            test_pairs.append(nvfp4_encode(x_t, "amax6"))
        w_pair = nvfp4_encode(w, "amax6")
        res = s.hif4_calibration_and_quantize_weight(*w_pair, calib_pairs)
        sc = score_linear(w_pair, test_pairs,
                          res["activation_state"], res["weight_params"])
        print(f"linear {name:5s} in={w.shape[1]} out={w.shape[0]}: "
              f"score={sc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
