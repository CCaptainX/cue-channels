#!/usr/bin/env python3
"""Exploratory (NOT pre-registered): 1,000-permutation max-over-layers null for the two H2 headline cells.

The registered analysis used 100 permutations (p floor 0.0099). Same CV, C = 0.1, seed 0.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_full_f_protocol import cv_correct, max_over_layers_perm  # noqa: E402

D = Path("experiments/H2_distractor_fix")
data = pickle.load(open(D / "full_activations.pkl", "rb"))
X_all = data["acts"]["last"].astype(np.float32)
meta = data["meta"]
m = json.loads((D / "metrics.json").read_text())
out = {"note": "exploratory, not pre-registered; 1000 label permutations, max over 28 layers, seed 0", "targets": {}}
for tname, key, tk in (("no_cot_answer", "no_cot", "A"), ("with_cot_answer", "with_cot", "B")):
    keep = np.array([x[key] is not None for x in meta])
    y = np.array([x[key] for x in meta], dtype=object)[keep]
    Xl = X_all[keep]
    pk = m["targets"][tk]["peak_layer"]
    obs = float(cv_correct(Xl[:, pk], y, 0)[0].mean())
    null = max_over_layers_perm(Xl, y, n_perm=1000)
    p = (1 + int((null >= obs).sum())) / (1 + len(null))
    out["targets"][tname] = {"peak_layer": pk, "observed": round(obs, 3), "null_max": round(float(null.max()), 3),
                             "null_p95": round(float(np.percentile(null, 95)), 3), "p": round(p, 4), "n_perm": 1000}
    print(tname, out["targets"][tname], flush=True)
json.dump(out, open(D / "exploratory/permutation_1000.json", "w"), indent=1)
