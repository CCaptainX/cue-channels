#!/usr/bin/env python3
"""H2 step 0: can the correct option be predicted from the 5 option values alone?

Pre-registered in experiments/H2_distractor_fix/preregistration.md (1ee5528).
"""

from __future__ import annotations

import json
import random
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
import h_arith  # noqa: E402

OUT = Path("experiments/H2_distractor_fix")
OFFS = {-10, -2, -1, 1, 2, 10}


def gen(variant: str, seed: int, n: int) -> list[tuple[list[int], int]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        it = h_arith._item(variant, rng)
        vals = [int(v) for v in re.findall(r"^\([A-E]\) (\d+)", it["input"], re.M)]
        rows.append((vals, "ABCDE".index(it["target"])))
    return rows


def option_features(vals: list[int]) -> np.ndarray:
    v = np.array(vals, dtype=float)
    ranks = v.argsort().argsort()
    med = np.median(v)
    feats = []
    for i in range(5):
        d = np.sort(np.abs(np.delete(v, i) - v[i]))
        feats.append([v[i], ranks[i], *d, d.mean(), float(v[i] == med)])
    return np.array(feats)


def scorer_cv(rows, make_model) -> float:
    X = np.stack([option_features(v) for v, _ in rows])  # [n, 5, F]
    y = np.array([t for _, t in rows])
    correct = 0
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        Xtr = X[tr].reshape(-1, X.shape[2])
        ytr = np.zeros((len(tr), 5))
        ytr[np.arange(len(tr)), y[tr]] = 1
        m = make_model().fit(Xtr, ytr.reshape(-1))
        p = m.predict_proba(X[te].reshape(-1, X.shape[2]))[:, 1].reshape(len(te), 5)
        correct += int((p.argmax(1) == y[te]).sum())
    return correct / len(rows)


def rule_median(rows) -> float:
    return float(np.mean([int(np.argsort(v)[2]) == t for v, t in rows]))


def rule_centre(rows, seed=0) -> float:
    rng = random.Random(seed)
    hits = 0
    for v, t in rows:
        c = [i for i in range(5) if all((v[j] - v[i]) in OFFS for j in range(5) if j != i)]
        pick = c[0] if len(c) == 1 else rng.choice(c if c else list(range(5)))
        hits += pick == t
    return hits / len(rows)


def evaluate(rows) -> dict:
    res = {
        "logistic_regression": scorer_cv(rows, lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000))),
        "random_forest": scorer_cv(rows, lambda: RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1)),
        "rule_median": rule_median(rows),
        "rule_centre_of_offsets": rule_centre(rows),
    }
    res = {k: round(v, 3) for k, v in res.items()}
    res["best"] = max(res.values())
    res["letter_dist"] = {ltr: int(sum(t == i for _, t in rows)) for i, ltr in enumerate("ABCDE")}
    return res


def main() -> None:
    h2 = evaluate(gen("C2", 9201, 5000))
    h = evaluate(gen("C", 9101, 5000))
    gate = h2["best"] <= 0.30
    out = {"n_items_each": 5000, "chance": 0.2, "H2_items": h2, "H_items_positive_control": h,
           "gate_H2_best_le_0.30": gate, "gate": "PASS" if gate else "FAIL",
           "detector_sensitive_on_H": h["best"] >= 0.95}
    # token check: the shifted position (q_end - 5) on real H2 prompts
    import f_common
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    pre, _ = h_arith.prompts("C2")
    toks = {}
    for it in h_arith.items("C2", 200):
        q_end = len(tok(f_common.question_end_prefix(pre, it))["input_ids"]) - 1
        ids = tok(f_common.cot_prompt(pre, it))["input_ids"]
        toks[tok.decode([ids[q_end - 5]])] = toks.get(tok.decode([ids[q_end - 5]]), 0) + 1
    out["shifted_token_identity_on_200_H2_prompts"] = toks
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pattern_baseline.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
