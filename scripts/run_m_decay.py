#!/usr/bin/env python3
"""Experiment M: mid-CoT probe decay (pre-registered, 3742d74)."""

from __future__ import annotations

import collections
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import f_common  # noqa: E402
import h_arith  # noqa: E402
from run_e_layer_sweep import bootstrap_ci  # noqa: E402
from run_full_f_protocol import cv_correct  # noqa: E402
from run_gh import load_backend  # noqa: E402

from cue_channels.probes.train import _make_pipeline  # noqa: E402

OUT = Path("experiments/M_mid_cot_decay")
LAYER = 22
ANCHORS = ["last_prompt", "first_cot", "first_eq", "pre_digit", "digit", "post_digit",
           "paren", "letter", "after_letter"]


def transfer_probe():
    blob = pickle.load(open("experiments/H2_distractor_fix/full_activations.pkl", "rb"))
    X = blob["acts"]["last"][:, LAYER].astype(np.float32)
    y = np.array([m["no_cot"] for m in blob["meta"]])
    keep = np.array([v is not None for v in y])
    return _make_pipeline(C=0.1, seed=0).fit(X[keep], y[keep])


def char_anchors(cot: str, product: int) -> dict:
    a = {"first_cot": 0}
    eq = cot.find("=")
    a["first_eq"] = eq if eq >= 0 else None
    m = re.search(rf"=\s*{product}(?!\d)", cot)
    if m:
        d0 = m.end() - len(str(product))
        a.update(pre_digit=d0 - 1, digit=d0, post_digit=m.end() - 1)
    else:
        a.update(pre_digit=None, digit=None, post_digit=None)
    m2 = re.search(r"answer is:?\s*\(([A-E])\)", cot)
    if m2:
        a.update(paren=m2.start(1) - 1, letter=m2.start(1), after_letter=m2.end(1))
    else:
        a.update(paren=None, letter=None, after_letter=None)
    return a


def main():
    gens = {}
    for line in open("experiments/J_causal_steering/generations.jsonl"):
        r = json.loads(line)
        if r["condition"] == "v|0|P":
            gens[r["idx"]] = r
    items = h_arith.items("C2", 300)[200:]
    cot_pre, _ = h_arith.prompts("C2")
    probe = transfer_probe()
    classes = list(probe.classes_)
    b = load_backend("qwen2.5-1.5b")
    tok = b.tokenizer
    per_item = []
    for it in items:
        g = gens[it["idx"]]
        prompt = f_common.cot_prompt(cot_pre, it)
        p_ids = tok(prompt)["input_ids"]
        enc = tok(prompt + g["cot"], return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        prefix_ok = ids[:len(p_ids)] == p_ids
        with torch.no_grad():
            h = b.model(torch.tensor([ids], device=b.model.device),
                        output_hidden_states=True).hidden_states[LAYER + 1][0].float().cpu().numpy()
        start = len(p_ids) - 1  # last prompt token = index 0
        H = h[start:]
        tok_char = [o[0] - len(prompt) for o in offs[start:]]  # char start within CoT (negative for prompt token)
        product = int(re.findall(r"\d+", it["question"])[0]) * int(re.findall(r"\d+", it["question"])[1])
        ca = char_anchors(g["cot"], product)

        def tok_at(c, tok_char=tok_char):
            if c is None:
                return None
            idx = [i for i, s in enumerate(tok_char) if s <= c]
            return idx[-1] if idx else None

        anchors = {"last_prompt": 0, **{k: tok_at(v) for k, v in ca.items()}}
        probs = probe.predict_proba(H)
        per_item.append({"idx": it["idx"], "label": g["no_cot"], "with_cot": g["with_cot"], "H": H, "anchors": anchors,
                         "pred": [classes[j] for j in probs.argmax(1)],
                         "p_label": [float(p[classes.index(g["no_cot"])]) if g["no_cot"] in classes else None
                                     for p in probs],
                         "tokens": [tok.decode([t]) for t in ids[start:]], "prefix_ok": prefix_ok})
    b.shutdown()

    labels = [r["label"] for r in per_item]
    maj = max(collections.Counter(labels).values()) / len(labels)
    max_len = max(len(r["pred"]) for r in per_item)
    curve = []
    for t in range(max_len):
        c = np.array([r["pred"][t] == r["label"] for r in per_item if t < len(r["pred"])])
        curve.append({"t": t, "n": int(len(c)), "acc": round(float(c.mean()), 3), "ci95": bootstrap_ci(c)})
    anchor_res = {}
    for a in ANCHORS:
        rows = [r for r in per_item if r["anchors"].get(a) is not None]
        c = np.array([r["pred"][r["anchors"][a]] == r["label"] for r in rows])
        X = np.stack([r["H"][r["anchors"][a]] for r in rows])
        y = np.array([r["label"] for r in rows])
        cc, _ = cv_correct(X, y, 0)
        anchor_res[a] = {"n": len(rows), "transfer_acc": round(float(c.mean()), 3), "transfer_ci95": bootstrap_ci(c),
                         "per_position_cv_acc": round(float(cc.mean()), 3), "per_position_ci95": bootstrap_ci(cc),
                         "mean_token_index": round(float(np.mean([r["anchors"][a] for r in rows])), 1)}
    pos_cv = []
    for t in range(max_len):
        rows = [r for r in per_item if t < len(r["pred"])]
        if len(rows) < 50:
            break
        cc, _ = cv_correct(np.stack([r["H"][t] for r in rows]), np.array([r["label"] for r in rows]), 0)
        pos_cv.append({"t": t, "n": len(rows), "acc": round(float(cc.mean()), 3)})

    pre, post = anchor_res["pre_digit"], anchor_res["post_digit"]
    overwrite = (pre["transfer_acc"] >= 0.60 and post["transfer_acc"] <= pre["transfer_acc"] - 0.15
                 and post["transfer_ci95"][1] < pre["transfer_ci95"][0])
    persists = all(v["transfer_acc"] >= 0.60 for v in anchor_res.values()) and \
        all(c["acc"] >= 0.60 for c in curve if c["n"] >= 50)
    noise = all(curve[t]["acc"] <= maj + 0.05 for t in (1, 2, 3))
    verdict = ("overwrite confirmed" if overwrite else "signal persists" if persists
               else "signal was always noise" if noise else "mixed")
    out = {"preregistration": "3742d74", "n_items": len(per_item), "majority": round(maj, 3),
           "prefix_tokenization_ok": sum(r["prefix_ok"] for r in per_item),
           "anchors": anchor_res, "curve_transfer": curve, "curve_per_position_cv": pos_cv,
           "decision": {"overwrite_confirmed": bool(overwrite), "signal_persists": bool(persists),
                        "always_noise": bool(noise), "verdict": verdict}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1) + "\n")
    ex = per_item[0]
    (OUT / "example_annotated.json").write_text(json.dumps(
        {"idx": ex["idx"], "label": ex["label"], "with_cot": ex["with_cot"], "anchors": ex["anchors"],
         "tokens": [{"t": i, "token": tk, "pred": p, "p_label": None if q is None else round(q, 3)}
                    for i, (tk, p, q) in enumerate(zip(ex["tokens"], ex["pred"], ex["p_label"], strict=True))]},
        indent=1) + "\n")
    with open(OUT / "per_item_curves.jsonl", "w") as f:
        for r in per_item:
            f.write(json.dumps({"idx": r["idx"], "label": r["label"], "anchors": r["anchors"], "pred": r["pred"],
                                "p_label": [None if q is None else round(q, 3) for q in r["p_label"]]}) + "\n")
    print(json.dumps({"majority": out["majority"], "anchors": anchor_res, "decision": out["decision"],
                      "prefix_ok": out["prefix_tokenization_ok"]}, indent=1))
    print("curve:", [(c["t"], c["acc"], c["n"]) for c in curve[:60]])


if __name__ == "__main__":
    main()
