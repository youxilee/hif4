"""真实数据参数扫描：复用 real_data_eval 的数据与打分口径。

真实 GPT-2 数据只加载/编码一次，每个配置仅重跑 solution.py 的
校准 + 打分，用于比较 Q/K/V/激活/权重的 refine 比例、offset、
居中模式、平滑 alpha 等参数组合。

用法：
    /opt/anaconda3/bin/python3 sweep_eval.py [--layers 12] [--seq 128]
        [--calib 2] [--test 2] [--mode amax6] [--only 0,1,3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solution as s  # noqa: E402
from real_data_eval import (  # noqa: E402
    collect_real_data,
    score_attention,
    score_linear,
)
from nvfp4_sim import nvfp4_encode  # noqa: E402


def make_configs() -> list[tuple[str, dict]]:
    """(名字, 覆盖项)。每个配置从 DEFAULT 出发，只覆盖列出的键。

    基准行固定为 best（当前最佳版本），所有百分比都相对 best 计算；
    old 仅保留作历史参考，不参与对比。
    """

    old = {
        "_WEIGHT_SMOOTH_RMS": False,
        "_QK_SMOOTH_RMS": False,
        "_REFINE_RANK_BY_ABSOLUTE": False,
        "_ATTN_CENTER_MODES": (0, 2),
    }
    best = {
        "_WEIGHT_SMOOTH_RMS": True,
        "_QK_SMOOTH_RMS": True,
        "_REFINE_RANK_BY_ABSOLUTE": True,
        "_ATTN_CENTER_MODES": (0, 2),
        "_REFINE_EDGE_EXTENSION": True,
        "_DATA_DRIVEN_RATIO": True,
        "_RATIO_CAPTURE_TARGET": 0.99,
        "_WEIGHT_QUADRATIC": True,
        "_ACTIVATION_QUADRATIC": True,
        "_WEIGHT_SMOOTH_ALPHAS": (0.25, 0.50, 0.75),
        "_Q_REFINE_MAX_RATIO": 0.60,
        "_K_REFINE_MAX_RATIO": 0.70,
        "_V_REFINE_MAX_RATIO": 0.60,
        "_ACTIVATION_REFINE_MAX_RATIO": 0.70,
    }
    new = {
        "_WEIGHT_SMOOTH_RMS": True,
        "_QK_SMOOTH_RMS": True,
        "_REFINE_RANK_BY_ABSOLUTE": True,
        "_ATTN_CENTER_MODES": (0, 2, 3),
    }
    return [
        ("best", best),                      # 当前最佳版本（基准）
        ("old", old),                      # 代码改动前的行为
        ("no_quad", {**best, "_WEIGHT_QUADRATIC": False}),
        ("no_act_quad", {**best, "_ACTIVATION_QUADRATIC": False}),
    ]


def apply_config(overrides: dict) -> None:
    defaults = {
        "_WEIGHT_OFFSETS": (-2, -1, 1, 2, 3),
        "_DYNAMIC_OFFSETS": (-1, 1, 2, 3),
        "_WEIGHT_REFINE_MAX_RATIO_SMALL": 1.0,
        "_WEIGHT_REFINE_MAX_RATIO_LARGE": 1.0,
        "_WEIGHT_REFINE_ACCEPT_MARGIN": 0.005,
        "_ACTIVATION_REFINE_MAX_RATIO": 0.30,
        "_ACTIVATION_REFINE_ACCEPT_MARGIN": 0.02,
        "_Q_REFINE_MAX_RATIO": 0.25,
        "_K_REFINE_MAX_RATIO": 0.35,
        "_V_REFINE_MAX_RATIO": 0.30,
        "_Q_REFINE_ACCEPT_MARGIN": 0.03,
        "_K_REFINE_ACCEPT_MARGIN": 0.03,
        "_V_REFINE_ACCEPT_MARGIN": 0.01,
        "_ATTN_CENTER_MODES": (0, 2),
        "_QK_SMOOTH_ALPHAS": (0.25, 0.50),
        "_WEIGHT_SMOOTH_ALPHAS": (0.25, 0.50),
        "_WEIGHT_SMOOTH_RMS": False,
        "_QK_SMOOTH_RMS": False,
        "_REFINE_RANK_BY_ABSOLUTE": False,
        "_REFINE_EDGE_EXTENSION": False,
        "_REFINE_EDGE_EXTEND_STEPS": 2,
        "_DATA_DRIVEN_RATIO": False,
        "_WEIGHT_QUADRATIC": False,
        "_ACTIVATION_QUADRATIC": False,
        "_RATIO_CAPTURE_TARGET": 0.95,
        "_RATIO_MIN": 0.10,
        "_LINEAR_EVAL_TOKENS": 128,
        "_ATTN_EVAL_TOKENS": 128,
        "_IMPORTANCE_FLOOR": 0.05,
    }
    for key, value in {**defaults, **overrides}.items():
        if not hasattr(s, key):
            raise KeyError(f"solution has no constant {key}")
        setattr(s, key, value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--calib", type=int, default=2)
    ap.add_argument("--test", type=int, default=2)
    ap.add_argument("--mode", default="amax6", choices=["amax6", "amax4", "pow2"])
    ap.add_argument("--only", default=None, help="逗号分隔的配置序号")
    args = ap.parse_args()

    configs = make_configs()
    if args.only is not None:
        indices = [int(x) for x in args.only.split(",")]
        configs = [configs[i] for i in indices]

    t0 = time.time()
    model, weights, calib, test, qh, hd = collect_real_data(
        args.layers, args.seq, args.calib, args.test
    )
    print(f"数据准备 {time.time()-t0:.1f}s  layers={args.layers} "
          f"heads={qh}x{hd} mode={args.mode}")

    L = len(weights)
    hidden = model.config.n_embd
    names = ("q", "k", "v", "o", "fc", "proj")

    # NVFP4 编码只做一次，所有配置共享同一输入。
    enc = {}
    enc["w"] = {
        i: {name: nvfp4_encode(weights[i][name], args.mode) for name in names}
        for i in range(L)
    }
    enc["calib_act"] = {
        i: {
            name: [
                nvfp4_encode(calib["act"][name][b * L + i], args.mode)
                for b in range(args.calib)
            ]
            for name in names
        }
        for i in range(L)
    }
    enc["test_act"] = {
        i: {
            name: [
                nvfp4_encode(test["act"][name][b * L + i], args.mode)
                for b in range(args.test)
            ]
            for name in names
        }
        for i in range(L)
    }
    enc["calib_qkv"] = {}
    enc["test_qkv"] = {}
    for i in range(L):
        enc["calib_qkv"][i] = []
        for b in range(args.calib):
            dense = calib["qkv"][b * L + i].reshape(-1, 3 * hidden)
            q_, k_, v_ = dense.chunk(3, dim=-1)
            enc["calib_qkv"][i].append(
                {"q": nvfp4_encode(q_, args.mode),
                 "k": nvfp4_encode(k_, args.mode),
                 "v": nvfp4_encode(v_, args.mode)}
            )
        enc["test_qkv"][i] = []
        for b in range(args.test):
            dense = test["qkv"][b * L + i].reshape(-1, 3 * hidden)
            q_, k_, v_ = dense.chunk(3, dim=-1)
            enc["test_qkv"][i].append(
                (nvfp4_encode(q_, args.mode),
                 nvfp4_encode(k_, args.mode),
                 nvfp4_encode(v_, args.mode))
            )

    print(f"NVFP4 编码完成 {time.time()-t0:.1f}s")

    rows: list[dict] = []
    for cfg_name, overrides in configs:
        apply_config(overrides)
        tc = time.time()
        lin = {name: [] for name in names}
        attn = []
        for i in range(L):
            for name in names:
                res = s.hif4_calibration_and_quantize_weight(
                    *enc["w"][i][name], enc["calib_act"][i][name]
                )
                lin[name].append(
                    score_linear(
                        enc["w"][i][name], enc["test_act"][i][name],
                        res["activation_state"], res["weight_params"],
                    )
                )
            att_state = s.hif4_calibration_attention(
                enc["calib_qkv"][i], qh, qh, hd
            )
            attn.append(
                score_attention(
                    enc["test_qkv"][i], att_state["q_state"],
                    att_state["k_state"], att_state["v_state"], qh, hd,
                )
            )
        row = {"cfg": cfg_name, "time": time.time() - tc}
        for name in names:
            row[name] = sum(lin[name]) / len(lin[name])
        row["attn"] = sum(attn) / len(attn)
        rows.append(row)
        print(f"[{cfg_name}] {time.time()-tc:.1f}s  " + "  ".join(
            f"{k}={v:.4f}" for k, v in row.items() if k not in ("cfg", "time")
        ), flush=True)

    base = next(r for r in rows if r["cfg"] == "best")
    print("\n相对 best（当前最佳）的变化:")
    for row in rows[1:]:
        parts = []
        if row["cfg"] in ("best", "old"):
            continue
        for key in ("q", "k", "v", "o", "fc", "proj", "attn"):
            delta = row[key] - base[key]
            parts.append(f"{key}{delta:+.4f}")
        print(f"  {row['cfg']:18s} " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
