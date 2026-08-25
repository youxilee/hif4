"""端到端离线评测：TinyTransformer + HiF4 转换。

模拟比赛流程：模型权重/激活以权威 NVFP4 (carrier, scale) 对给出，
reference 路径直接反量化 NVFP4 前向；HiF4 路径先走 solution.py 的
校准/动态量化再反量化前向。对比两条路径的最终 logits 误差。

用法：
    /opt/anaconda3/bin/python3 e2e_eval.py [--outlier] [--layers 2] [--seq 128]

可选 --outlier 让权重带离群通道（考验重要性权重鲁棒性）。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvfp4_sim import nvfp4_encode  # noqa: E402
import solution as s  # noqa: E402


def nvfp4_decode(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return s.dequantize_nvfp4(q, scale).to(torch.float32)


def flat2d(t: torch.Tensor) -> torch.Tensor:
    """把 [batch, seq, width] 压成 [seq, width]（标定 API 需要 2D）。"""
    return t.reshape(-1, t.shape[-1])


def hif4_dequant(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return s._dequantize_hif4(params).to(torch.float32)


def _normalize_no_floor(importance, length):
    """复刻无重要性地板的旧版归一化（用于对比原始配置）。"""
    if importance is None:
        return None
    w = importance.detach().to(dtype=torch.float32).reshape(-1)
    if int(w.numel()) != length:
        raise ValueError(f"Expected importance of length {length}, got {w.numel()}")
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mean = w.mean()
    if float(mean) <= 1.0e-12:
        return torch.ones_like(w)
    return w / mean


class Layer:
    def __init__(
        self,
        hidden: int,
        ff: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        generator: torch.Generator,
        outlier: bool,
    ):
        rn = lambda *shape: torch.randn(*shape, generator=generator)  # noqa: E731
        self.wq = rn(hidden, hidden) * 0.02
        self.wk = rn(kv_heads * head_dim, hidden) * 0.02
        self.wv = rn(kv_heads * head_dim, hidden) * 0.02
        self.wo = rn(hidden, hidden) * 0.02
        self.w1 = rn(ff, hidden) * 0.02
        self.w2 = rn(hidden, ff) * 0.02
        if outlier:
            k = max(1, hidden // 100)
            for w in (self.wq, self.wk, self.wv, self.wo, self.w1):
                w[:, :k] *= 25.0  # 离群输入通道
            self.w2[:, : max(1, ff // 100)] *= 25.0
        self.ln1_w = torch.ones(hidden)
        self.ln1_b = torch.zeros(hidden)
        self.ln2_w = torch.ones(hidden)
        self.ln2_b = torch.zeros(hidden)


class TinyTransformer:
    def __init__(
        self,
        hidden: int = 256,
        layers: int = 2,
        q_heads: int = 4,
        kv_heads: int = 2,
        head_dim: int = 64,
        ff: int = 1024,
        vocab: int = 512,
        seed: int = 0,
        outlier: bool = False,
    ):
        g = torch.Generator().manual_seed(seed)
        rn = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
        self.hidden = hidden
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.embed = rn(vocab, hidden) * 0.5
        if outlier:
            self.embed[: max(1, vocab // 100)] *= 30.0
        self.out_proj = rn(hidden, vocab) * 0.05
        self.layers = [
            Layer(hidden, ff, q_heads, kv_heads, head_dim, g, outlier)
            for _ in range(layers)
        ]

    def _ln(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + 1e-5) * w + b

    def _attn(self, q, k, v) -> torch.Tensor:
        B, T, _ = q.shape
        qh, kvh, hd = self.q_heads, self.kv_heads, self.head_dim
        q = q.reshape(B, T, qh, hd).transpose(1, 2)
        k = k.reshape(B, T, kvh, hd).transpose(1, 2).repeat_interleave(qh // kvh, dim=1)
        v = v.reshape(B, T, kvh, hd).transpose(1, 2).repeat_interleave(qh // kvh, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(hd)
        att = torch.softmax(scores, dim=-1)
        return (att @ v).transpose(1, 2).reshape(B, T, qh * hd)

    def forward(
        self,
        tokens: torch.Tensor,
        mode: str,
        wpairs: dict,
        st: dict,
        intermediates: list | None = None,
    ) -> torch.Tensor:
        """mode='ref'：NVFP4 反量化前向；mode='hif4'：HiF4 转换后前向。"""
        B, T = tokens.shape
        x = self.embed[tokens].to(torch.float32)
        for i, layer in enumerate(self.layers):
            x = self._ln(x, layer.ln1_w, layer.ln1_b)

            if mode == "ref":
                x_wq = nvfp4_decode(*nvfp4_encode(x))
                x_wk = nvfp4_decode(*nvfp4_encode(x))
                x_wv = nvfp4_decode(*nvfp4_encode(x))
                q = nvfp4_decode(*nvfp4_encode(x_wq @ layer.wq.T))
                k = nvfp4_decode(*nvfp4_encode(x_wk @ layer.wk.T))
                v = nvfp4_decode(*nvfp4_encode(x_wv @ layer.wv.T))
                attn_out = self._attn(q, k, v)
                x = x + nvfp4_decode(*nvfp4_encode(attn_out)) @ layer.wo.T
                x = self._ln(x, layer.ln2_w, layer.ln2_b)
                h = nvfp4_decode(*nvfp4_encode(x)) @ layer.w1.T
                act = F.gelu(h)
                act = nvfp4_decode(*nvfp4_encode(act))
                x = x + act @ layer.w2.T
            else:
                Wq = hif4_dequant(st["wp"]["wq"][i])
                Wk = hif4_dequant(st["wp"]["wk"][i])
                Wv = hif4_dequant(st["wp"]["wv"][i])
                Wo = hif4_dequant(st["wp"]["wo"][i])
                W1 = hif4_dequant(st["wp"]["w1"][i])
                W2 = hif4_dequant(st["wp"]["w2"][i])

                # 比赛 API 的激活/Q/K/V 都是 2D [seq, channels]，先展平再恢复
                x_wq = hif4_dequant(
                    s.hif4_dynamic_quantize_activation(
                        *nvfp4_encode(flat2d(x)), st["act"]["wq"][i]
                    )
                ).reshape(B, T, self.hidden)
                q = hif4_dequant(
                    s.hif4_dynamic_quantize_q(
                        *nvfp4_encode(flat2d(x_wq @ Wq.T)), self.q_heads,
                        self.head_dim,
                        st["q"][i],
                    )
                ).reshape(B, T, self.hidden)
                k = hif4_dequant(
                    s.hif4_dynamic_quantize_k(
                        *nvfp4_encode(flat2d(x_wq @ Wk.T)), self.kv_heads,
                        self.head_dim,
                        st["k"][i],
                    )
                ).reshape(B, T, self.kv_heads * self.head_dim)
                v = hif4_dequant(
                    s.hif4_dynamic_quantize_v(
                        *nvfp4_encode(flat2d(x_wq @ Wv.T)), self.kv_heads,
                        self.head_dim,
                        st["v"][i],
                    )
                ).reshape(B, T, self.kv_heads * self.head_dim)
                attn_out = self._attn(q, k, v)
                x = x + hif4_dequant(
                    s.hif4_dynamic_quantize_activation(
                        *nvfp4_encode(flat2d(attn_out)), st["act"]["wo"][i]
                    )
                ).reshape(B, T, self.hidden) @ Wo
                x = self._ln(x, layer.ln2_w, layer.ln2_b)
                h = hif4_dequant(
                    s.hif4_dynamic_quantize_activation(
                        *nvfp4_encode(flat2d(x)), st["act"]["w1"][i]
                    )
                ).reshape(B, T, self.hidden) @ W1.T
                act = F.gelu(h)
                act = hif4_dequant(
                    s.hif4_dynamic_quantize_activation(
                        *nvfp4_encode(flat2d(act)), st["act"]["w2"][i]
                    )
                ).reshape(B, T, self.layers[0].w1.shape[0])
                x = x + act @ W2.T

            if intermediates is not None:
                intermediates.append(x.clone())
        return x @ self.out_proj


def collect_calibration(model: TinyTransformer, token_batches: list[torch.Tensor]):
    """在 reference 前向下收集各层输入激活（稠密）。"""
    x_ln1, x_ln2, attn_out, gelu_out, qkv = [], [], [], [], []
    with torch.no_grad():
        for tokens in token_batches:
            x = model.embed[tokens].to(torch.float32)
            xl1, xl2, ao, go, qkv_b = [], [], [], [], []
            for layer in model.layers:
                x = model._ln(x, layer.ln1_w, layer.ln1_b)
                xl1.append(x)
                q = nvfp4_decode(*nvfp4_encode(x @ layer.wq.T))
                k = nvfp4_decode(*nvfp4_encode(x @ layer.wk.T))
                v = nvfp4_decode(*nvfp4_encode(x @ layer.wv.T))
                qkv_b.append((q, k, v))
                attn = model._attn(q, k, v)
                ao.append(attn)
                x = x + nvfp4_decode(*nvfp4_encode(attn)) @ layer.wo.T
                x = model._ln(x, layer.ln2_w, layer.ln2_b)
                xl2.append(x)
                act = F.gelu(nvfp4_decode(*nvfp4_encode(x)) @ layer.w1.T)
                act = nvfp4_decode(*nvfp4_encode(act))
                go.append(act)
                x = x + act @ layer.w2.T
            x_ln1.append(xl1)
            x_ln2.append(xl2)
            attn_out.append(ao)
            gelu_out.append(go)
            qkv.append(qkv_b)
    return x_ln1, x_ln2, attn_out, gelu_out, qkv


def calibrate(model: TinyTransformer, wpairs: dict, calib):
    x_ln1, x_ln2, attn_out, gelu_out, qkv = calib
    L = len(model.layers)
    st = {"wp": {k: [] for k in ("wq", "wk", "wv", "wo", "w1", "w2")},
          "act": {k: [] for k in ("wq", "wk", "wv", "wo", "w1", "w2")},
          "q": [], "k": [], "v": []}
    for i in range(L):
        for name in ("wq", "wk", "wv"):
            w_q, w_s = wpairs[name][i]
            acts = [nvfp4_encode(flat2d(x[i])) for x in x_ln1]
            res = s.hif4_calibration_and_quantize_weight(w_q, w_s, acts)
            st["wp"][name].append(res["weight_params"])
            st["act"][name].append(res["activation_state"])
        # W_o 的输入是 attention 输出
        w_q, w_s = wpairs["wo"][i]
        res = s.hif4_calibration_and_quantize_weight(
            w_q, w_s,
            [nvfp4_encode(flat2d(attn_out[b][i])) for b in range(len(attn_out))],
        )
        st["wp"]["wo"].append(res["weight_params"])
        st["act"]["wo"].append(res["activation_state"])
        w_q, w_s = wpairs["w1"][i]
        res = s.hif4_calibration_and_quantize_weight(
            w_q, w_s, [nvfp4_encode(flat2d(x[i])) for x in x_ln2]
        )
        st["wp"]["w1"].append(res["weight_params"])
        st["act"]["w1"].append(res["activation_state"])
        w_q, w_s = wpairs["w2"][i]
        res = s.hif4_calibration_and_quantize_weight(
            w_q, w_s, [nvfp4_encode(flat2d(x[i])) for x in gelu_out],
        )
        st["wp"]["w2"].append(res["weight_params"])
        st["act"]["w2"].append(res["activation_state"])

        qkv_list = [
            {"q": nvfp4_encode(flat2d(qkv[b][i][0])),
             "k": nvfp4_encode(flat2d(qkv[b][i][1])),
             "v": nvfp4_encode(flat2d(qkv[b][i][2]))}
            for b in range(len(qkv))
        ]
        att = s.hif4_calibration_attention(
            qkv_list, model.q_heads, model.kv_heads, model.head_dim
        )
        st["q"].append(att["q_state"])
        st["k"].append(att["k_state"])
        st["v"].append(att["v_state"])
    return st


def metrics(logits_ref: torch.Tensor, logits_h: torch.Tensor) -> dict:
    rel = float((logits_ref - logits_h).norm() / logits_ref.norm().clamp_min(1e-9))
    mae = float((logits_ref - logits_h).abs().mean())
    cos = float(F.cosine_similarity(logits_ref.reshape(-1), logits_h.reshape(-1), dim=0))
    p = F.softmax(logits_ref, -1).clamp_min(1e-9)
    q = F.softmax(logits_h, -1).clamp_min(1e-9)
    kl = float((q * (q.log() - p.log())).sum(-1).mean())
    agree = float((logits_ref.argmax(-1) == logits_h.argmax(-1)).float().mean())
    return {"rel_l2": rel, "mae": mae, "cos": cos, "kl": kl, "top1": agree}


def set_config(name: str, real_normalize) -> None:
    if name == "original":
        s._WEIGHT_OFFSETS = (-1, 1, 2)
        s._DYNAMIC_OFFSETS = (-1, 2)
        s._WEIGHT_REFINE_MAX_RATIO_SMALL = 0.2
        s._WEIGHT_REFINE_MAX_RATIO_LARGE = 0.1
        s._ACTIVATION_REFINE_MAX_RATIO = 0.1
        s._Q_REFINE_MAX_RATIO = 0.08
        s._K_REFINE_MAX_RATIO = 0.12
        s._V_REFINE_MAX_RATIO = 0.1
        s._WEIGHT_SMOOTH_ALPHAS = (0.25, 0.50)
        s._WEIGHT_SMOOTH_RMS = False
        s._QK_SMOOTH_RMS = False
        s._REFINE_RANK_BY_ABSOLUTE = False
        s._REFINE_EDGE_EXTENSION = False
        s._DATA_DRIVEN_RATIO = False
        s._OFFSET_SELECTION = False
        s._WEIGHT_QUADRATIC = False
        s._ATTN_QUADRATIC = False
        s._ATTN_CENTER_MODES = (0, 2)
        s._normalize_importance = _normalize_no_floor
    else:
        s._WEIGHT_OFFSETS = (-2, -1, 1, 2, 3)
        s._DYNAMIC_OFFSETS = (-1, 1, 2, 3)
        s._WEIGHT_REFINE_MAX_RATIO_SMALL = 1.0
        s._WEIGHT_REFINE_MAX_RATIO_LARGE = 1.0
        s._ACTIVATION_REFINE_MAX_RATIO = 0.5
        s._Q_REFINE_MAX_RATIO = 0.4
        s._K_REFINE_MAX_RATIO = 0.5
        s._V_REFINE_MAX_RATIO = 0.4
        s._WEIGHT_SMOOTH_ALPHAS = (0.25, 0.50, 0.75)
        s._WEIGHT_SMOOTH_RMS = True
        s._QK_SMOOTH_RMS = True
        s._REFINE_RANK_BY_ABSOLUTE = True
        s._REFINE_EDGE_EXTENSION = True
        s._DATA_DRIVEN_RATIO = True
        s._OFFSET_SELECTION = False
        s._WEIGHT_QUADRATIC = True
        s._ATTN_QUADRATIC = False
        s._ATTN_CENTER_MODES = (0, 2)
        s._normalize_importance = real_normalize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--calib-batches", type=int, default=3)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--outlier", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = TinyTransformer(layers=args.layers, outlier=args.outlier)
    wpairs = {
        name: [nvfp4_encode(getattr(layer, name)) for layer in model.layers]
        for name in ("wq", "wk", "wv", "wo", "w1", "w2")
    }

    g = torch.Generator().manual_seed(1)
    calib_tokens = [
        torch.randint(0, model.embed.shape[0], (1, args.seq), generator=g)
        for _ in range(args.calib_batches)
    ]
    eval_tokens = [
        torch.randint(0, model.embed.shape[0], (1, args.seq), generator=g)
        for _ in range(args.eval_batches)
    ]

    calib = collect_calibration(model, calib_tokens)
    real_normalize = s._normalize_importance
    print(f"model: layers={args.layers} hidden={model.hidden} "
          f"heads={model.q_heads}/{model.kv_heads}x{model.head_dim} "
          f"outlier={args.outlier}")
    print(f"{'config':<10}{'rel_l2':>9}{'mae':>10}{'cos':>9}{'kl':>10}{'top1':>8}")

    results = {}
    layer_diag = {}
    with torch.no_grad():
        for cfg in ("original", "current"):
            set_config(cfg, real_normalize)
            st = calibrate(model, wpairs, calib)
            refs, hifs = [], []
            ref_ints, hif_ints = [], []
            for t in eval_tokens:
                refs.append(model.forward(t, "ref", wpairs, st, ref_ints))
                hifs.append(model.forward(t, "hif4", wpairs, st, hif_ints))
            m = metrics(torch.cat(refs), torch.cat(hifs))
            results[cfg] = m
            print(f"{cfg:<10}{m['rel_l2']:>9.4f}{m['mae']:>10.4f}{m['cos']:>9.4f}"
                  f"{m['kl']:>10.5f}{m['top1']:>8.3f}")
            # 逐层隐藏态误差（每个 batch 一层一张，取平均）
            layer_diag[cfg] = [
                float(sum(
                    (r - h).norm() / r.norm().clamp_min(1e-9)
                    for r, h in zip(
                        (ref_ints[b * args.layers + i] for b in range(len(eval_tokens))),
                        (hif_ints[b * args.layers + i] for b in range(len(eval_tokens))),
                    )
                ) / len(eval_tokens))
                for i in range(args.layers)
            ]

    a, b = results["original"], results["current"]
    print(f"\ncurrent vs original: rel_l2 {a['rel_l2']:.4f} -> {b['rel_l2']:.4f} "
          f"({(b['rel_l2']/a['rel_l2']-1)*100:+.1f}%), "
          f"top1 {a['top1']:.3f} -> {b['top1']:.3f}")
    print("\nper-layer hidden-state rel error:")
    for i in range(args.layers):
        print(f"  layer {i}: original={layer_diag['original'][i]:.4f} "
              f"current={layer_diag['current'][i]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
