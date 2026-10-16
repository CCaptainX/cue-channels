#!/usr/bin/env python3
"""Experiment J: CAA steering along the H2 layer-22 direction (pre-registered, d9bbb0f)."""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import f_common  # noqa: E402
import h_arith  # noqa: E402
from run_gh import STOP, load_backend  # noqa: E402

OUT = Path("experiments/J_causal_steering")
LAYER = 22
ALPHAS = [-2, -1, 0, 1, 2, 5]
RAND_ALPHAS = [2, 5]


def steering_vectors():
    blob = pickle.load(open("experiments/H2_distractor_fix/full_activations.pkl", "rb"))
    X = blob["acts"]["last"][:, LAYER].astype(np.float32)
    lab = np.array([m["no_cot"] for m in blob["meta"]])
    v = X[lab == "A"].mean(0) - X[lab != "A"].mean(0)
    r = np.random.default_rng(0).standard_normal(v.shape).astype(np.float32)
    r -= (r @ v) / (v @ v) * v
    r *= np.linalg.norm(v) / np.linalg.norm(r)
    resid_norm = float(np.linalg.norm(X, axis=1).mean())
    return v, r, {"n_A": int((lab == "A").sum()), "n_notA": int((lab != "A").sum()),
                  "v_norm": float(np.linalg.norm(v)), "mean_resid_norm_L22": resid_norm,
                  "cos_v_r": float(r @ v / (np.linalg.norm(r) * np.linalg.norm(v)))}


class Steer:
    def __init__(self, layer_module, device, dtype):
        self.vec = None
        self.alpha = 0.0
        self.mode = "P"
        self.device, self.dtype = device, dtype
        layer_module.register_forward_hook(self.hook)

    def set(self, vec, alpha, mode):
        self.vec = None if vec is None else torch.tensor(vec, device=self.device, dtype=self.dtype)
        self.alpha, self.mode = alpha, mode

    def hook(self, module, inputs, output):
        if self.vec is None or self.alpha == 0:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] > 1:  # prompt pass: last prompt token
            hs[:, -1, :] += self.alpha * self.vec
        elif self.mode == "S":  # generated tokens (KV-cache step)
            hs[:, -1, :] += self.alpha * self.vec
        return output


def _product(question: str) -> int:
    a, b = map(int, re.findall(r"\d+", question)[:2])
    return a * b


def run_condition(b, steer, items, cot_pre, ao_pre, vec, alpha, mode):
    steer.set(vec, alpha, mode)
    rows = []
    for it in items:
        cot = b.generate_text(f_common.cot_prompt(cot_pre, it), max_new_tokens=384, stop_strings=STOP).text
        nc = b.generate_text(f_common.no_cot_prompt(ao_pre, it), max_new_tokens=2).text
        rows.append({"idx": it["idx"], "target": it["target"], "cot": cot,
                     "with_cot": f_common.parse_cot(cot, it["letters"]),
                     "no_cot": f_common.parse_no_cot(nc, it["letters"]),
                     "product_correct_in_cot": str(_product(it["question"])) in cot})
    steer.set(None, 0, "P")
    return rows


def boot_diff(a, b, n=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def summarize(results, base_key="v|0|P"):
    base = results[base_key]
    base_A = [r["with_cot"] == "A" for r in base]
    out = {}
    for key, rows in results.items():
        isA = [r["with_cot"] == "A" for r in rows]
        notA0 = [i for i, r in enumerate(base) if r["with_cot"] != "A"]
        A0 = [i for i, r in enumerate(base) if r["with_cot"] == "A"]
        out[key] = {
            "rate_A": round(float(np.mean(isA)), 3),
            "delta_vs_alpha0": round(float(np.mean(isA) - np.mean(base_A)), 3),
            "delta_ci95": boot_diff(isA, base_A),
            "flip_toward_A": round(float(np.mean([rows[i]["with_cot"] == "A" for i in notA0])), 3) if notA0 else None,
            "flip_away_from_A": round(float(np.mean([rows[i]["with_cot"] != "A" for i in A0])), 3) if A0 else None,
            "no_cot_rate_A": round(float(np.mean([r["no_cot"] == "A" for r in rows])), 3),
            "with_cot_accuracy": round(float(np.mean([r["with_cot"] == r["target"] for r in rows])), 3),
            "unparsed": sum(r["with_cot"] is None for r in rows),
            "product_correct_in_cot": round(float(np.mean([r["product_correct_in_cot"] for r in rows])), 3),
            "letter_dist": dict(collections.Counter(str(r["with_cot"]) for r in rows)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    v, r, vinfo = steering_vectors()
    items = h_arith.items("C2", 300)[200:200 + a.limit]
    cot_pre, ao_pre = h_arith.prompts("C2")
    b = load_backend("qwen2.5-1.5b")
    steer = Steer(b.model.model.layers[LAYER], b.model.device, b.model.dtype)
    conds = [("v", 0, "P")] + [("v", al, m) for m in ("P", "S") for al in ALPHAS if al != 0] + \
            [("rand", al, m) for m in ("P", "S") for al in RAND_ALPHAS]
    # exploratory addendum (not part of the decision rule)
    conds += [("v", al, m) for m in ("P", "S") for al in (10, 20)] + [("rand", 20, m) for m in ("P", "S")]
    results, timing = {}, {}
    for name, al, mode in conds:
        t0 = time.time()
        results[f"{name}|{al}|{mode}"] = run_condition(b, steer, items, cot_pre, ao_pre,
                                                       v if name == "v" else r, al, mode)
        timing[f"{name}|{al}|{mode}"] = round(time.time() - t0)
        print(f"{name}|{al}|{mode} done in {timing[f'{name}|{al}|{mode}']}s", flush=True)
    b.shutdown()
    summ = summarize(results)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "generations.jsonl", "w") as f:
        for key, rows in results.items():
            for row in rows:
                f.write(json.dumps({"condition": key, **row}) + "\n")
    rng = random.Random(0)
    sample_idx = sorted(rng.sample(range(len(items)), 10))
    (OUT / "metrics.json").write_text(json.dumps({
        "preregistration": "d9bbb0f", "layer": LAYER, "n_items": len(items),
        "held_out": "h_arith C2 items 200-299", "steering_vector": vinfo, "conditions": summ,
        "coherence_sample_item_positions": sample_idx, "seconds": timing}, indent=2) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in s.items() if kk != "letter_dist"} for k, s in summ.items()}, indent=1))


if __name__ == "__main__":
    main()
