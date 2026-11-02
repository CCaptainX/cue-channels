#!/usr/bin/env python3
"""I3 gate G4: options-only baseline (pre-registered, 02fb86d)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import i3_tasks as T  # noqa: E402
from i_pattern_baseline import feats as num_feats  # noqa: E402
from i_pattern_baseline import score  # noqa: E402

OUT = Path("experiments/I3_memorizable_tasks")


def feats(opts: list[str]) -> np.ndarray:
    base = num_feats(opts)
    alpha = np.array(sorted(range(5), key=lambda i: opts[i].lower())).argsort()
    upper = [sum(ch.isupper() for ch in o) for o in opts]
    return np.column_stack([base, alpha, upper])


def evaluate(task: str) -> dict:
    its = T.items(task, 5000, seed_offset=9000)
    X = np.stack([feats(it["option_values"]) for it in its])
    y = np.array(["ABCDE".index(it["target"]) for it in its])
    res = {"n_items": len(its),
           "logistic_regression": score(X, y, lambda: make_pipeline(StandardScaler(),
                                                                     LogisticRegression(max_iter=5000))),
           "random_forest": score(X, y, lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1)),
           "rule_median": round(float(np.mean([int(np.argsort(X[i, :, 0])[2]) == y[i] for i in range(len(y))])), 3)}
    res["best"] = max(res["logistic_regression"], res["random_forest"], res["rule_median"])
    res["G4_pass"] = res["best"] <= 0.30
    return res


if __name__ == "__main__":
    out = {t: evaluate(t) for t in T.TASKS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pattern_baseline.json").write_text(json.dumps(out, indent=2) + "\n")
    for t, r in out.items():
        print(t, r)
