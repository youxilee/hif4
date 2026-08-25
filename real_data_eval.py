"""真实数据端到端评测：GPT-2 权重/激活 + 比赛口径评分。

用真实预训练 GPT-2（12 层，hidden 768）跑真实文本前向，抓取每个
Linear 的输入激活与 Q/K/V 稠密张量，按 NVIDIA cuDNN 权威配方编码为
NVFP4 (E2M1 carrier + E4M3 scale) 对，再走
solution.py 的校准/动态量化，按任务书口径计算：
    Linear 得分 = (MSE_STD - MSE_PLAYER) / MSE_STD
    Attn   得分 = (MSE_STD - MSE_PLAYER) / MSE_STD
其中 STD 为标准 HiF4 启发式转换，PLAYER 为 solution.py 的完整方案。

用法：
    /opt/anaconda3/bin/python3 real_data_eval.py [--layers 12] [--seq 128]
        [--mode amax6|amax4|pow2] [--config both|original|current]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvfp4_sim import nvfp4_encode  # noqa: E402
import solution as s  # noqa: E402

MODEL_DIR = "/private/tmp/gpt2"

TEXT = (
    "The history of science is a history of measurement and precision. "
    "Modern language models learn to predict the next word from vast amounts of text. "
    "Attention mechanisms allow a model to focus on the most relevant parts of a sequence. "
    "Quantization reduces numerical precision to accelerate inference on specialized hardware. "
    "A four bit format stores each value using only four bits, trading range for speed. "
    "The transformer architecture has become the backbone of modern natural language processing. "
    "Neural networks are trained with gradient descent and backpropagation. "
    "The embedding layer maps discrete tokens into continuous vector representations. "
    "Layer normalization stabilizes training by normalizing activations across hidden units. "
    "Residual connections help gradients flow through deep networks during optimization. "
    "Softmax converts raw logits into a probability distribution over the vocabulary. "
    "Matrix multiplication is the fundamental operation in deep learning inference. "
    "Huawei's Ascend hardware provides efficient support for low precision computation. "
    "The calibration phase collects statistics from representative input data. "
    "Online quantization must be fast because it runs for every token during inference. "
    "A block scale is shared by a group of values to reduce metadata overhead. "
    "Outliers can dominate the dynamic range and reduce the accuracy of block quantization. "
    "Smoothing redistributes magnitude between activations and weights to improve precision. "
    "Permuting channels keeps exactly equivalent operations while improving quantization. "
    "The exact solver enumerates all valid exponent combinations for a given scale. "
    "Testing on held out data measures how well the quantization generalizes. "
    "The competition evaluates both linear layers and attention projections separately. "
    "Time limits require the quantization algorithm to be efficient as well as accurate. "
    "Small language models can still exhibit the same outlier patterns as large ones. "
    "Understanding the distribution of values is the key to effective quantization. "
    "Every experiment should be reproducible with a fixed random seed. "
    "The final score is the relative improvement over a standard baseline. "
    "Careful numerical analysis reveals why some scale choices outperform others. "
    "This paragraph provides diverse natural text for capturing real activation statistics. "
    "The quick brown fox jumps over the lazy dog while machines learn to reason. "
)

def std_hif4(dense: torch.Tensor) -> torch.Tensor:
    return s._dequantize_hif4(
        s._dense_to_hif4(dense, search_offsets=())
    ).to(torch.float32)


def causal_attn(Q, K, V, qh: int, hd: int) -> torch.Tensor:
    B, T, _ = Q.shape
    q = Q.reshape(B, T, qh, hd).transpose(1, 2)
    k = K.reshape(B, T, qh, hd).transpose(1, 2)
    v = V.reshape(B, T, qh, hd).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / math.sqrt(hd)
    mask = torch.triu(torch.full((T, T), float("-inf"), device=scores.device), 1)
    scores = scores + mask
    att = torch.softmax(scores, -1)
    return (att @ v).transpose(1, 2).reshape(B, T, qh * hd)


def set_config(kind: str) -> None:
    if kind == "original":
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
        s._ATTN_CENTER_MODES = (0, 2)
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
        s._ATTN_CENTER_MODES = (0, 2)


def collect_real_data(layers: int, seq: int, n_calib: int, n_test: int):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    tok = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model.eval()
    ids = tok(TEXT, return_tensors="pt", truncation=True, max_length=4096)["input_ids"][0]
    n = ids.numel()
    n_batches = n_calib + n_test
    total = n_batches * seq
    if total > n:
        ids = ids.repeat((total + n - 1) // n)[:total]

    hidden = model.config.n_embd
    qh = model.config.n_head
    hd = hidden // qh
    blocks = model.transformer.h[:layers]

    weights = []  # 每层: {name: [out, in] float32}
    for layer in blocks:
        w = {}
        attn_w = layer.attn.c_attn.weight.detach().t().float()  # [3*hidden, hidden]
        w["q"] = attn_w[:hidden].clone()
        w["k"] = attn_w[hidden:2 * hidden].clone()
        w["v"] = attn_w[2 * hidden:].clone()
        w["o"] = layer.attn.c_proj.weight.detach().t().float()
        w["fc"] = layer.mlp.c_fc.weight.detach().t().float()
        w["proj"] = layer.mlp.c_proj.weight.detach().t().float()
        weights.append(w)

    calib = {"act": {k: [] for k in ("q", "k", "v", "o", "fc", "proj")},
             "qkv": []}
    test = {"act": {k: [] for k in ("q", "k", "v", "o", "fc", "proj")},
            "qkv": []}

    def run2(store, b_start: int, b_end: int):
        captured = {i: {} for i in range(len(blocks))}
        handles = []
        for i, layer in enumerate(blocks):
            def make(i, key):
                def fn(mod, inp, out):
                    captured[i][key] = inp[0].detach().float()
                return fn

            def make_cattn(i):
                def fn(mod, inp, out):
                    captured[i]["attn_in"] = inp[0].detach().float()
                    captured[i]["attn_raw"] = out[0].detach().float()
                return fn

            handles.append(layer.attn.c_attn.register_forward_hook(make_cattn(i)))
            handles.append(layer.attn.c_proj.register_forward_hook(make(i, "attn_proj_in")))
            handles.append(layer.mlp.c_fc.register_forward_hook(make(i, "fc_in")))
            handles.append(layer.mlp.c_proj.register_forward_hook(make(i, "proj_in")))
        for b in range(b_start, b_end):
            inp = ids[b * seq:(b + 1) * seq][None]
            with torch.no_grad():
                model(inp)
            for i, layer in enumerate(blocks):
                x_in = captured[i]["attn_in"]
                attn_out = captured[i]["attn_proj_in"]
                ln2_in = captured[i]["fc_in"]
                gelu_out = captured[i]["proj_in"]
                qkv_dense = captured[i]["attn_raw"]
                flat = lambda t: t.reshape(-1, t.shape[-1])
                store["act"]["q"].append(flat(x_in))
                store["act"]["k"].append(flat(x_in))
                store["act"]["v"].append(flat(x_in))
                store["act"]["o"].append(flat(attn_out))
                store["act"]["fc"].append(flat(ln2_in))
                store["act"]["proj"].append(flat(gelu_out))
                store["qkv"].append(qkv_dense)
        for h in handles:
            h.remove()

    run2(calib, 0, n_calib)
    run2(test, n_calib, n_calib + n_test)
    return model, weights, calib, test, qh, hd


def score_linear(w_pair, act_pairs, act_state, wp):
    scores = []
    for x_pair in act_pairs:
        X_ref = s._dequantize_nvfp4_float32(*x_pair)
        W_ref = s._dequantize_nvfp4_float32(*w_pair)
        y_ref = X_ref @ W_ref.T
        y_std = std_hif4(X_ref) @ std_hif4(W_ref).T
        X_h = s._dequantize_hif4(
            s.hif4_dynamic_quantize_activation(*x_pair, act_state)
        ).to(torch.float32)
        W_h = s._dequantize_hif4(wp).to(torch.float32)
        y_h = X_h @ W_h.T
        m_std = float(((y_std - y_ref).square()).mean())
        m_h = float(((y_h - y_ref).square()).mean())
        scores.append((m_std - m_h) / m_std)
    return sum(scores) / len(scores)


def score_attention(qkv_pairs, q_state, k_state, v_state, qh, hd):
    scores = []
    for q_pair, k_pair, v_pair in qkv_pairs:
        Q_ref = s._dequantize_nvfp4_float32(*q_pair)
        K_ref = s._dequantize_nvfp4_float32(*k_pair)
        V_ref = s._dequantize_nvfp4_float32(*v_pair)
        A_ref = causal_attn(Q_ref[None], K_ref[None], V_ref[None], qh, hd)
        A_std = causal_attn(
            std_hif4(Q_ref)[None], std_hif4(K_ref)[None], std_hif4(V_ref)[None], qh, hd
        )
        Q_h = s._dequantize_hif4(
            s.hif4_dynamic_quantize_q(*q_pair, qh, hd, q_state)
        ).to(torch.float32)
        K_h = s._dequantize_hif4(
            s.hif4_dynamic_quantize_k(*k_pair, qh, hd, k_state)
        ).to(torch.float32)
        V_h = s._dequantize_hif4(
            s.hif4_dynamic_quantize_v(*v_pair, qh, hd, v_state)
        ).to(torch.float32)
        A_h = causal_attn(Q_h[None], K_h[None], V_h[None], qh, hd)
        m_std = float(((A_std - A_ref).square()).mean())
        m_h = float(((A_h - A_ref).square()).mean())
        scores.append((m_std - m_h) / m_std)
    return sum(scores) / len(scores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--calib", type=int, default=2)
    ap.add_argument("--test", type=int, default=2)
    ap.add_argument("--mode", default="amax6", choices=["amax6", "amax4", "pow2"])
    ap.add_argument("--config", default="both", choices=["both", "original", "current"])
    args = ap.parse_args()

    model, weights, calib, test, qh, hd = collect_real_data(
        args.layers, args.seq, args.calib, args.test
    )
    L = len(weights)
    hidden = model.config.n_embd

    results = {}
    for cfg in (("original", "current") if args.config == "both" else (args.config,)):
        set_config(cfg)
        lin = {"q": [], "k": [], "v": [], "o": [], "fc": [], "proj": []}
        attn_scores = []
        for i in range(L):
            for name in ("q", "k", "v", "o", "fc", "proj"):
                w_pair = nvfp4_encode(weights[i][name], args.mode)
                calib_pairs = [
                    nvfp4_encode(calib["act"][name][b * L + i], args.mode)
                    for b in range(args.calib)
                ]
                res = s.hif4_calibration_and_quantize_weight(
                    *w_pair, calib_pairs
                )
                test_pairs = [
                    nvfp4_encode(test["act"][name][b * L + i], args.mode)
                    for b in range(args.test)
                ]
                lin[name].append(
                    score_linear(w_pair, test_pairs,
                                 res["activation_state"], res["weight_params"])
                )
            # attention：q/k/v 用各自层的校准数据
            qkv_calib = []
            for b in range(args.calib):
                dense = calib["qkv"][b * L + i].reshape(-1, 3 * hidden)
                q_, k_, v_ = dense.chunk(3, dim=-1)
                qkv_calib.append(
                    {"q": nvfp4_encode(q_, args.mode),
                     "k": nvfp4_encode(k_, args.mode),
                     "v": nvfp4_encode(v_, args.mode)}
                )
            att = s.hif4_calibration_attention(qkv_calib, qh, qh, hd)
            qkv_test = []
            for b in range(args.test):
                dense = test["qkv"][b * L + i].reshape(-1, 3 * hidden)
                q_, k_, v_ = dense.chunk(3, dim=-1)
                qkv_test.append(
                    (nvfp4_encode(q_, args.mode),
                     nvfp4_encode(k_, args.mode),
                     nvfp4_encode(v_, args.mode))
                )
            attn_scores.append(
                score_attention(qkv_test, att["q_state"], att["k_state"],
                                att["v_state"], qh, hd)
            )
        results[cfg] = (lin, attn_scores)

    print(f"GPT-2 layers={L} hidden={hidden} heads={qh}x{hd} mode={args.mode} "
          f"seq={args.seq} calib={args.calib} test={args.test}")
    for cfg, (lin, attn_scores) in results.items():
        print(f"\n[{cfg}] Linear 得分（按层平均）:")
        for name in ("q", "k", "v", "o", "fc", "proj"):
            v = lin[name]
            print(f"  {name:5s} mean={sum(v)/len(v):.4f}  "
                  f"min={min(v):.4f} max={max(v):.4f}")
        print(f"  Attention mean={sum(attn_scores)/len(attn_scores):.4f} "
              f"min={min(attn_scores):.4f} max={max(attn_scores):.4f}")

    if len(results) == 2:
        lo, co = results["original"][0], results["current"][0]
        ao, ac = results["original"][1], results["current"][1]
        print("\ncurrent vs original:")
        for name in ("q", "k", "v", "o", "fc", "proj"):
            a, b = sum(lo[name]) / len(lo[name]), sum(co[name]) / len(co[name])
            print(f"  Linear {name:5s}: {a:.4f} -> {b:.4f} ({(b/a-1)*100:+.1f}%)")
        a, b = sum(ao) / len(ao), sum(ac) / len(ac)
        print(f"  Attn       : {a:.4f} -> {b:.4f} ({(b/a-1)*100:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
