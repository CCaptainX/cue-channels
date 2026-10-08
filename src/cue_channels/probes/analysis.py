"""Shared probe analysis for experiments B and D (pre-CoT decodability with controls)."""

from __future__ import annotations

import math
import re

import numpy as np

from cue_channels.probes.train import train_probe


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(centre - half, 3), round(centre + half, 3))


def grouped_cv_accuracy(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5, seed: int = 0, C: float = 0.1
) -> dict:
    """Grouped k-fold: every prompt is tested exactly once. Returns accuracy + Wilson 95% CI."""
    from sklearn.model_selection import GroupKFold

    from cue_channels.probes.train import _make_pipeline

    rng = np.random.default_rng(seed)
    # Shuffle group order so folds aren't contiguous question ids.
    uniq = np.unique(groups)
    perm = dict(zip(uniq, rng.permutation(len(uniq)), strict=True))
    g = np.array([perm[x] for x in groups])
    correct = 0
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            correct += int((y[te] == y[tr][0]).sum())
            continue
        clf = _make_pipeline(C=C, seed=seed).fit(X[tr], y[tr])
        correct += int((clf.predict(X[te]) == y[te]).sum())
    n = len(y)
    return {"accuracy": round(correct / n, 3), "ci95": wilson_ci(correct, n), "k": correct, "n": n}


def _nums(rows: list[dict]):
    opt_a = np.array([int(re.search(r"\(A\) (-?\d+)", r["prompt"]).group(1)) for r in rows])
    opt_b = np.array([int(re.search(r"\(B\) (-?\d+)", r["prompt"]).group(1)) for r in rows])
    first = np.array([int(re.search(r"What is (\d+)", r["prompt"]).group(1)) for r in rows])
    return opt_a, opt_b, first


def analyze_condition(
    acts: dict[int, np.ndarray],
    rows: list[dict],
    model_answers: list[str | None] | None = None,
    layer: int | None = None,
    seed: int = 0,
) -> dict:
    """Correct-letter probe (headline), controls, layer sweep, own-answer probe."""
    y = np.array([1 if r["label"] == "A" else 0 for r in rows])
    groups = np.array([r["qid"] for r in rows])
    layer = layer if layer is not None else len(acts) // 2
    X = acts[layer]

    split_accs = [
        train_probe(X, y, groups=groups, seed=s)[1]["test_accuracy"] for s in range(5)
    ]
    cv = grouped_cv_accuracy(X, y, groups, seed=seed)
    shuffled = grouped_cv_accuracy(X, np.random.default_rng(seed).permutation(y), groups, seed=seed)

    opt_a, opt_b, first = _nums(rows)
    positive = {
        "option_A_gt_option_B": grouped_cv_accuracy(X, (opt_a > opt_b).astype(int), groups)[
            "accuracy"
        ],
        "first_operand_gt_50": grouped_cv_accuracy(X, (first > 50).astype(int), groups)[
            "accuracy"
        ],
    }
    sweep = {
        lyr: grouped_cv_accuracy(acts[lyr], y, groups, seed=seed)["accuracy"]
        for lyr in sorted(acts)
    }

    out = {
        "layer": layer,
        "correct_letter_probe": {
            "grouped_5fold_cv": cv,
            "split_80_20_5_seeds": {
                "mean": round(float(np.mean(split_accs)), 3),
                "std": round(float(np.std(split_accs)), 3),
                "values": [round(a, 3) for a in split_accs],
            },
        },
        "shuffled_label_control": shuffled,
        "positive_controls": positive,
        "correct_letter_by_layer": sweep,
    }

    if model_answers is not None:
        ans = list(model_answers)
        parsed = [i for i, a in enumerate(ans) if a in ("A", "B")]
        behavior = {
            "n": len(ans),
            "parsed": len(parsed),
            "correct": sum(ans[i] == rows[i]["label"] for i in parsed),
            "accuracy_all": round(sum(ans[i] == rows[i]["label"] for i in parsed) / len(ans), 3),
            "frac_answer_A_of_parsed": (
                round(sum(ans[i] == "A" for i in parsed) / len(parsed), 3) if parsed else None
            ),
        }
        out["behavior"] = behavior
        if len(parsed) >= 40:
            idx = np.array(parsed)
            y_own = np.array([1 if ans[i] == "A" else 0 for i in parsed])
            if 0 < y_own.mean() < 1:
                own_cv = grouped_cv_accuracy(X[idx], y_own, groups[idx], seed=seed)
                out["own_answer_probe"] = {
                    **own_cv,
                    "majority_baseline": round(float(max(y_own.mean(), 1 - y_own.mean())), 3),
                    "by_layer": {
                        lyr: grouped_cv_accuracy(acts[lyr][idx], y_own, groups[idx], seed=seed)[
                            "accuracy"
                        ]
                        for lyr in sorted(acts)
                    },
                }
    return out
