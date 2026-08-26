"""Probe block-orthogonal SmoothQuant transforms on real GPT-2 Linear data.

This script deliberately does not change solution.py.  It reuses the diagonal
scale and permutation selected by the current calibration, then appends a
normalized block-Hadamard transform of size 4, 8, or 16 to both operands.  For
row-vector Linear convention y = X @ W.T, an orthogonal H preserves the result:

    (X @ H) @ (W @ H).T = X @ W.T.

The probe uses the standard (non-refined) HiF4 converter for every candidate so
that it isolates the value of the transform rather than scale-search settings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvfp4_sim import nvfp4_encode  # noqa: E402
from real_data_eval import collect_real_data, set_config  # noqa: E402
import solution as s  # noqa: E402


NAMES = ("q", "k", "v", "o", "fc", "proj")
BLOCK_SIZES = (4, 8, 16)


def hadamard(size: int, *, device: torch.device) -> torch.Tensor:
    if size < 1 or size & (size - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    while int(h.shape[0]) < size:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0
        )
    return h / float(size) ** 0.5


def block_rotate(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    size = int(h.shape[0])
    if int(x.shape[-1]) % size:
        raise ValueError("Feature width must be divisible by the block size")
    shape = x.shape
    return (x.reshape(-1, shape[-1] // size, size) @ h).reshape(shape)


def std_hif4(x: torch.Tensor) -> torch.Tensor:
    return s._dequantize_hif4(s._dense_to_hif4(x, search_offsets=())).float()


def mse_output(x: torch.Tensor, w: torch.Tensor, ref: torch.Tensor) -> float:
    return float(((std_hif4(x) @ std_hif4(w).T - ref).square()).mean())


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--calib", type=int, default=2)
    ap.add_argument("--test", type=int, default=2)
    ap.add_argument("--mode", default="amax6", choices=("amax6", "amax4", "pow2"))
    args = ap.parse_args()

    set_config("current")
    _, weights, calib, test, _, _ = collect_real_data(
        args.layers, args.seq, args.calib, args.test
    )
    scores = {name: {"diag": [], **{f"h{b}": [] for b in BLOCK_SIZES}}
              for name in NAMES}

    for layer in range(args.layers):
        for name in NAMES:
            w_pair = nvfp4_encode(weights[layer][name], args.mode)
            calib_pairs = [
                nvfp4_encode(calib["act"][name][batch * args.layers + layer], args.mode)
                for batch in range(args.calib)
            ]
            result = s.hif4_calibration_and_quantize_weight(
                *w_pair, calib_pairs
            )
            state = result["activation_state"]
            w = s._dequantize_nvfp4_float32(*w_pair)
            channels = int(w.shape[-1])
            inv_d = state["smooth_inv"]
            if inv_d is None:
                inv_d = torch.ones(channels, dtype=torch.float32)
            inv_d = inv_d.to(w.device)
            d = inv_d.reciprocal()
            perm = state["permutation"]
            if perm is None:
                perm = torch.arange(channels, dtype=torch.int64)
            perm = perm.to(w.device)
            w_base = (w * d.reshape(1, -1)).index_select(-1, perm)
            hs = {b: hadamard(b, device=w.device) for b in BLOCK_SIZES}
            w_rot = {b: block_rotate(w_base, hs[b]) for b in BLOCK_SIZES}

            for batch in range(args.test):
                x_pair = nvfp4_encode(
                    test["act"][name][batch * args.layers + layer], args.mode
                )
                x = s._dequantize_nvfp4_float32(*x_pair)
                ref = x @ w.T
                std_mse = mse_output(x, w, ref)
                x_base = (x * inv_d.reshape(1, -1)).index_select(-1, perm)
                candidates = {"diag": mse_output(x_base, w_base, ref)}
                for b in BLOCK_SIZES:
                    candidates[f"h{b}"] = mse_output(
                        block_rotate(x_base, hs[b]), w_rot[b], ref
                    )
                for key, value in candidates.items():
                    scores[name][key].append((std_mse - value) / std_mse)

        print(f"layer {layer + 1}/{args.layers} complete", flush=True)

    print("score = improvement over standard HiF4; delta is relative to diagonal")
    for name in NAMES:
        means = {
            key: sum(values) / len(values) for key, values in scores[name].items()
        }
        parts = [f"diag={means['diag']:.4f}"]
        for b in BLOCK_SIZES:
            key = f"h{b}"
            parts.append(f"{key}={means[key]:.4f} ({means[key]-means['diag']:+.4f})")
        print(f"{name:>4s}  " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
