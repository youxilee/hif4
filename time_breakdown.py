"""单层时间拆解：校准 vs 每样本动态量化，用于估算比赛 5 分钟预算。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solution as s  # noqa: E402
from real_data_eval import collect_real_data  # noqa: E402
from nvfp4_sim import nvfp4_encode  # noqa: E402


def main() -> int:
    model, weights, calib, test, qh, hd = collect_real_data(12, 128, 2, 2)
    L = len(weights)
    hidden = model.config.n_embd

    # 单层、单 Linear（fc）的校准时间
    w_pair = nvfp4_encode(weights[0]["fc"], "amax6")
    calib_pairs = [nvfp4_encode(calib["act"]["fc"][i], "amax6") for i in range(2)]
    t0 = time.perf_counter()
    res = s.hif4_calibration_and_quantize_weight(*w_pair, calib_pairs)
    t_wcal = time.perf_counter() - t0

    # 每样本激活动态量化时间
    x_pair = nvfp4_encode(test["act"]["fc"][2], "amax6")
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        s.hif4_dynamic_quantize_activation(*x_pair, res["activation_state"])
        times.append(time.perf_counter() - t0)
    t_act = min(times)

    # 注意力校准 + 每样本 Q/K/V
    qkv_calib = []
    for b in range(2):
        dense = calib["qkv"][b * L].reshape(-1, 3 * hidden)
        q_, k_, v_ = dense.chunk(3, dim=-1)
        qkv_calib.append(
            {"q": nvfp4_encode(q_, "amax6"),
             "k": nvfp4_encode(k_, "amax6"),
             "v": nvfp4_encode(v_, "amax6")}
        )
    t0 = time.perf_counter()
    att = s.hif4_calibration_attention(qkv_calib, qh, qh, hd)
    t_acal = time.perf_counter() - t0

    dense = test["qkv"][1 * L].reshape(-1, 3 * hidden)
    q_, k_, v_ = dense.chunk(3, dim=-1)
    qp, ks, vp = nvfp4_encode(q_, "amax6"), nvfp4_encode(k_, "amax6"), nvfp4_encode(v_, "amax6")
    for name, fn, args in (
        ("q", s.hif4_dynamic_quantize_q, (*qp, qh, hd, att["q_state"])),
        ("k", s.hif4_dynamic_quantize_k, (*ks, qh, hd, att["k_state"])),
        ("v", s.hif4_dynamic_quantize_v, (*vp, qh, hd, att["v_state"])),
    ):
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            fn(*args)
            ts.append(time.perf_counter() - t0)
        print(f"dynamic {name}: {min(ts)*1000:.1f} ms/sample")

    print(f"weight calib (fc, 1 层, 2 calib): {t_wcal*1000:.1f} ms")
    print(f"activation dynamic: {t_act*1000:.1f} ms/sample")
    print(f"attn calib (1 层, 2 calib): {t_acal*1000:.1f} ms")

    # 外推：N 组 × L 层
    for n_groups in (3, 5, 10):
        calib_total = n_groups * L * (t_wcal + t_acal)
        # 每测试样本：6 个 Linear 激活 + Q/K/V
        per_sample = L * (6 * t_act + 3 * min(times))
        for n_test in (4, 16, 64):
            total = calib_total + n_groups * n_test * per_sample
            print(f"  N={n_groups} test={n_test}: 校准 {calib_total:.1f}s "
                  f"+ 动态 {total-calib_total:.1f}s = {total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
