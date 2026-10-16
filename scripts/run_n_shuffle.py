#!/usr/bin/env python3
"""Experiment N: option-order reversal (pre-registered, ecb491d)."""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import f_common  # noqa: E402
import h_arith  # noqa: E402
from run_e_layer_sweep import bootstrap_ci  # noqa: E402
from run_gh import load_backend  # noqa: E402
from run_k_per_token import ANCHORS, anchor_positions  # noqa: E402

from cue_channels.probes.train import _make_pipeline  # noqa: E402

OUT = Path("experiments/N_shuffle_options")
LAYER = 22
L = "ABCDE"


def reverse_item(it):
    q = it["input"].split("\nOptions:")[0]
    vals = re.findall(r"^\([A-E]\) (\S+)", it["input"], re.M)[::-1]
    opts = "\n".join(f"({ltr}) {v}" for ltr, v in zip(L, vals, strict=True))
    return {**it, "input": f"{q}\nOptions:\n{opts}", "target": L[4 - L.index(it["target"])]}


def slot(anchor):
    if anchor.startswith(("open_", "value_end_")):
        return L.index(anchor[-1])
    if anchor in ("A:", "think", "last"):
        return 5
    return -1  # pre-options anchors (Q, question_end, Options)


def collect(b, pre, ao_pre, items, gen_no_cot):
    acts, no_cot = [], []
    for it in items:
        pos, full = anchor_positions(b, pre, it)
        ids = b.encode_raw(full)
        with torch.no_grad():
            hs = torch.stack([h[0] for h in b.model(ids, output_hidden_states=True).hidden_states[1:]])
        acts.append(hs[:, [pos[a] for a in ANCHORS]].float().cpu().numpy().astype(np.float16))
        if gen_no_cot:
            no_cot.append(f_common.parse_no_cot(b.generate_text(f_common.no_cot_prompt(ao_pre, it),
                                                                max_new_tokens=2).text, it["letters"]))
    return np.stack(acts), no_cot  # [N, 28, 16, D]


def crossfit_probs(X, y, seed=0):
    from sklearn.model_selection import StratifiedKFold

    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = _make_pipeline(C=0.1, seed=seed).fit(X[tr], y[tr])
        pr = clf.predict_proba(X[te])
        cls = list(clf.classes_)
        p[te] = [pr[i, cls.index(y[j])] if y[j] in cls else 0 for i, j in enumerate(te)]
    return p


def heat(A, y):
    from run_full_f_protocol import cv_correct

    return np.array([[float(cv_correct(A[:, lyr, a].astype(np.float32), y, 0)[0].mean())
                      for a in range(A.shape[2])] for lyr in range(A.shape[1])])


def r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 0.0
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), float(slope)


def main():
    items = h_arith.items("C2", 200)
    rev = [reverse_item(it) for it in items]
    pre, ao_pre = h_arith.prompts("C2")
    gens = [json.loads(line) for line in open("experiments/H2_distractor_fix/generations.jsonl")]
    nc_orig = [g["no_cot"] for g in gens]
    b = load_backend("qwen2.5-1.5b")
    A_o, _ = collect(b, pre, ao_pre, items, False)
    A_r, nc_rev = collect(b, pre, ao_pre, rev, True)
    b.shutdown()
    res = {}
    per = {}
    for name, A, lab, its in (("original", A_o, nc_orig, items), ("reversed", A_r, nc_rev, rev)):
        keep = np.array([x is not None for x in lab])
        y = np.array([x for x in lab if x is not None])
        P = np.full((len(lab), len(ANCHORS)), np.nan)
        for a in range(len(ANCHORS)):
            P[keep, a] = crossfit_probs(A[keep][:, LAYER, a].astype(np.float32), y)
        H = heat(A[keep], y)
        res[name] = {"n": int(keep.sum()), "no_cot_acc": round(float(np.mean([x == it["target"]
                                                                              for x, it in zip(lab, its, strict=True)])), 3),
                     "no_cot_dist": dict(collections.Counter(str(x) for x in lab)),
                     "acc_by_layer_anchor": np.round(H, 3).tolist(),
                     "L22_acc_by_anchor": dict(zip(ANCHORS, np.round(H[LAYER], 3).tolist(), strict=True))}
        per[name] = P
    # per-item measures
    rows = []
    for i, (it, rt) in enumerate(zip(items, rev, strict=True)):
        po, pr = per["original"][i], per["reversed"][i]
        if np.isnan(po).any() or np.isnan(pr).any():
            continue
        ok = nc_orig[i] == it["target"] and nc_rev[i] == rt["target"]

        def peak(p):
            return slot(ANCHORS[int(np.argmax(p))]), ANCHORS[int(np.argmax(p))]

        def onset(p):
            for a, v in zip(ANCHORS, p, strict=True):
                if v >= 0.5:
                    return slot(a)
            return 6

        rows.append({"i": i, "c_orig": L.index(it["target"]), "c_rev": L.index(rt["target"]), "main": ok,
                     "peak_orig": peak(po)[0], "peak_rev": peak(pr)[0], "peak_anchor_orig": peak(po)[1],
                     "peak_anchor_rev": peak(pr)[1], "onset_orig": onset(po), "onset_rev": onset(pr)})
    main_rows = [r for r in rows if r["main"]]
    out = {"preregistration": "ecb491d", "layer": LAYER, "versions": res, "n_items_with_both": len(rows),
           "n_main": len(main_rows)}
    for meas in ("peak", "onset"):
        dc = [r["c_rev"] - r["c_orig"] for r in main_rows]
        dm = [r[f"{meas}_rev"] - r[f"{meas}_orig"] for r in main_rows]
        rsq, slope = r2(dc, dm)
        mode_o = collections.Counter(r[f"{meas}_orig"] for r in main_rows).most_common(1)[0]
        mode_r = collections.Counter(r[f"{meas}_rev"] for r in main_rows).most_common(1)[0]
        same_mode = mode_o[0] == mode_r[0]
        verdict = ("tracks the correct answer" if rsq >= 0.6 and slope > 0 else
                   "structural" if rsq < 0.2 and same_mode else "mixed")
        # within-version relation: does the measure follow the correct slot?
        r_o, s_o = r2([r["c_orig"] for r in main_rows], [r[f"{meas}_orig"] for r in main_rows])
        r_r, s_r = r2([r["c_rev"] for r in main_rows], [r[f"{meas}_rev"] for r in main_rows])
        out[meas] = {"R2_delta": round(rsq, 3), "slope_delta": round(slope, 3),
                     "modal_orig": {"slot": mode_o[0], "share": round(mode_o[1] / len(main_rows), 3)},
                     "modal_rev": {"slot": mode_r[0], "share": round(mode_r[1] / len(main_rows), 3)},
                     "within_version_R2": {"original": round(r_o, 3), "reversed": round(r_r, 3)},
                     "within_version_slope": {"original": round(s_o, 3), "reversed": round(s_r, 3)},
                     "slot_dist_orig": dict(collections.Counter(r[f"{meas}_orig"] for r in main_rows)),
                     "slot_dist_rev": dict(collections.Counter(r[f"{meas}_rev"] for r in main_rows)),
                     "verdict": verdict}
    out["peak_anchor_dist"] = {
        "original": dict(collections.Counter(r["peak_anchor_orig"] for r in main_rows)),
        "reversed": dict(collections.Counter(r["peak_anchor_rev"] for r in main_rows))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1) + "\n")
    with open(OUT / "per_item.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "versions"}, indent=1))
    for v in ("original", "reversed"):
        print(v, res[v]["n"], res[v]["no_cot_acc"], res[v]["no_cot_dist"], res[v]["L22_acc_by_anchor"])
    _ = bootstrap_ci


if __name__ == "__main__":
    main()
