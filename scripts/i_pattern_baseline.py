#!/usr/bin/env python3
"""I gate G4: options-only baseline per task (pre-registered, be4fee1)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import i_tasks as T  # noqa: E402

OUT = Path("experiments/I_generalization")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def first_num(s: str) -> float:
    m = _NUM.search(s.replace(",", ""))
    return float(m.group()) if m else np.nan


def feats(opts: list[str]) -> np.ndarray:
    v = np.array([first_num(o) for o in opts])
    fill = np.nanmedian(v) if np.isfinite(v).any() else 0.0
    v = np.where(np.isfinite(v), v, fill)
    ranks = v.argsort().argsort()
    med = np.median(v)
    rows = []
    for i in range(5):
        d = np.sort(np.abs(np.delete(v, i) - v[i]))
        rows.append([v[i], ranks[i], *d, d.mean(), float(v[i] == med), len(opts[i])])
    return np.array(rows)


def score(X, y, make):
    correct = 0
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        ytr = np.zeros((len(tr), 5))
        ytr[np.arange(len(tr)), y[tr]] = 1
        m = make().fit(X[tr].reshape(-1, X.shape[2]), ytr.reshape(-1))
        p = m.predict_proba(X[te].reshape(-1, X.shape[2]))[:, 1].reshape(len(te), 5)
        correct += int((p.argmax(1) == y[te]).sum())
    return round(correct / len(y), 3)


def evaluate(task: str) -> dict:
    if task in ("add_carry", "mod"):
        its = T.items(task, 5000, seed_offset=9000)
    else:
        pool = T.gsm8k_pool() if task == "gsm8k_easy" else T.aqua_pool()
        its = T.items(task, len(pool), seed_offset=9000)
    X = np.stack([feats(it["option_values"]) for it in its])
    y = np.array(["ABCDE".index(it["target"]) for it in its])
    res = {
        "n_items": len(its),
        "logistic_regression": score(X, y, lambda: make_pipeline(StandardScaler(),
                                                                  LogisticRegression(max_iter=5000))),
        "random_forest": score(X, y, lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1)),
        "rule_median": round(float(np.mean([int(np.argsort(X[i, :, 0])[2]) == y[i]
                                             for i in range(len(y))])), 3),
        "majority_letter_rate": round(float(np.bincount(y, minlength=5).max() / len(y)), 3),
    }
    res["best"] = max(res["logistic_regression"], res["random_forest"], res["rule_median"])
    res["G4_pass"] = res["best"] <= 0.30
    return res


def main() -> None:
    out = {t: evaluate(t) for t in T.TASKS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pattern_baseline.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
