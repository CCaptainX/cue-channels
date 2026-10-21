#!/usr/bin/env python3
"""Exploratory (NOT pre-registered) robustness checks on the cached H2 activations, added after review.

1. Regularization sweep: probe accuracy by layer for C in {0.01, 0.1, 1, 10} (seed 0, same CV as run_full_f_protocol).
2. 2x2 breakdown at the headline cell (no-CoT-answer probe, layer 22, C = 0.1, seed 0): probe correct x model correct.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_full_f_protocol import _splits  # noqa: E402

from cue_channels.probes.train import _make_pipeline  # noqa: E402

D = Path("experiments/H2_distractor_fix")
data = pickle.load(open(D / "full_activations.pkl", "rb"))
X_all = data["acts"]["last"].astype(np.float32)  # [N, L, D]
meta = data["meta"]


def cv_pred(X, y, C, seed=0):
    splits, _ = _splits(X, y, seed, None)
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        pred[te] = _make_pipeline(C=C, seed=seed).fit(X[tr], y[tr]).predict(X[te])
    return pred


out = {"note": "exploratory, not pre-registered; seed 0 only", "C_sweep": {}}
for tname, key in (("no_cot_answer", "no_cot"), ("with_cot_answer", "with_cot")):
    keep = np.array([m[key] is not None for m in meta])
    y = np.array([m[key] for m in meta], dtype=object)[keep]
    X = X_all[keep]
    res = {}
    for C in (0.01, 0.1, 1.0, 10.0):
        accs = [float((cv_pred(X[:, lyr], y, C) == y).mean()) for lyr in range(X.shape[1])]
        pk = int(np.argmax(accs))
        res[str(C)] = {"by_layer": [round(a, 3) for a in accs], "peak_layer": pk, "peak_acc": round(accs[pk], 3),
                       "acc_L21": round(accs[21], 3), "acc_L22": round(accs[22], 3)}
        print(tname, C, "peak", pk, round(accs[pk], 3), "L22", round(accs[22], 3), flush=True)
    out["C_sweep"][tname] = {"n": int(keep.sum()), "results": res}

keep = np.array([m["no_cot"] is not None for m in meta])
y = np.array([m["no_cot"] for m in meta], dtype=object)[keep]
truth = np.array([m["target"] for m in meta], dtype=object)[keep]
pred = cv_pred(X_all[keep][:, 22], y, 0.1)
pc, mc = pred == y, y == truth
tab = {"model_correct": {"n": int(mc.sum()), "probe_acc": round(float(pc[mc].mean()), 3)},
       "model_wrong": {"n": int((~mc).sum()), "probe_acc": round(float(pc[~mc].mean()), 3)},
       "probe_predicts_truth_when_model_wrong": round(float((pred[~mc] == truth[~mc]).mean()), 3)}
out["breakdown_no_cot_L22"] = tab
print(tab)
(D / "exploratory").mkdir(exist_ok=True)
json.dump(out, open(D / "exploratory/robustness.json", "w"), indent=1)
