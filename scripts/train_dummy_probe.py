#!/usr/bin/env python3
"""Commit B probe: train on pre-CoT activations to predict which letter is correct.

Headline uses the middle layer with an 80/20 split grouped by question. Also reports a
per-layer sweep, a shuffled-label control (which should land near 50%), and positive controls
for facts stated in the prompt (which should be decodable if extraction works). Runs for each
prompt format present: cot (dummy_activations.pkl) and direct (dummy_activations_direct.pkl).
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from cue_channels.probes.train import train_probe


def analyze(blob: dict, rows: list[dict], layer: int | None, seed: int):
    acts: dict[int, np.ndarray] = blob["activations"]
    y = np.array([1 if r["label"] == "A" else 0 for r in rows])
    groups = np.array([r["qid"] for r in rows])
    layer = layer if layer is not None else len(acts) // 2

    probe, headline = train_probe(acts[layer], y, groups=groups, seed=seed)
    seed_accs = [
        train_probe(acts[layer], y, groups=groups, seed=s)[1]["test_accuracy"] for s in range(5)
    ]
    rng = np.random.default_rng(seed)
    _, control = train_probe(acts[layer], rng.permutation(y), groups=groups, seed=seed)

    opt_a = np.array([int(re.search(r"\(A\) (\d+)", r["prompt"]).group(1)) for r in rows])
    opt_b = np.array([int(re.search(r"\(B\) (\d+)", r["prompt"]).group(1)) for r in rows])
    first = np.array([int(re.search(r"What is (\d+)", r["prompt"]).group(1)) for r in rows])
    positive = {
        "option_A_gt_option_B": (opt_a > opt_b).astype(int),
        "first_operand_gt_50": (first > 50).astype(int),
    }
    positive_acc = {
        k: round(train_probe(acts[layer], v, groups=groups, seed=seed)[1]["test_accuracy"], 3)
        for k, v in positive.items()
    }
    sweep = {
        lyr: round(train_probe(acts[lyr], y, groups=groups, seed=seed)[1]["test_accuracy"], 3)
        for lyr in sorted(acts)
    }

    # The model's own answer (when it was generated for every prompt): this is the decision
    # actually made at this position, as opposed to the correct answer.
    own = None
    answers = blob.get("model_answers") or []
    if len(answers) == len(rows) and all(a in ("A", "B") for a in answers):
        y_own = np.array([1 if a == "A" else 0 for a in answers])
        own_sweep = {
            lyr: round(
                train_probe(acts[lyr], y_own, groups=groups, seed=seed)[1]["test_accuracy"], 3
            )
            for lyr in sorted(acts)
        }
        own = {
            "frac_answer_A": round(float(y_own.mean()), 3),
            "test_accuracy_at_layer": own_sweep[layer],
            "majority_baseline": round(float(max(y_own.mean(), 1 - y_own.mean())), 3),
            "test_accuracy_by_layer": own_sweep,
        }
    best = max(sweep, key=sweep.get)
    metrics = {
        "format": blob.get("format", "cot"),
        "system_prompt": blob.get("system"),
        "behavior": blob.get("behavior"),
        "layer": layer,
        "headline": headline,
        "test_accuracy_over_5_split_seeds": {
            "mean": round(float(np.mean(seed_accs)), 3),
            "std": round(float(np.std(seed_accs)), 3),
            "values": [round(a, 3) for a in seed_accs],
        },
        "shuffled_label_control_test_accuracy": round(control["test_accuracy"], 3),
        "model_own_answer_probe": own,
        "positive_controls_test_accuracy": positive_acc,
        "test_accuracy_by_layer": sweep,
        "best_layer": {"layer": best, "test_accuracy": sweep[best],
                       "note": "selected post hoc on the test split; optimistic"},
    }
    return probe, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="experiments/B_probe")
    ap.add_argument("--layer", type=int, default=None, help="default: n_layers // 2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.dir)
    rows = [json.loads(line) for line in open(d / "labels.jsonl")]
    results, probes = {}, {}
    files = {"cot": "dummy_activations.pkl", "direct": "dummy_activations_direct.pkl"}
    for fmt, fname in files.items():
        if not (d / fname).exists():
            continue
        with open(d / fname, "rb") as f:
            blob = pickle.load(f)
        probes[fmt], results[fmt] = analyze(blob, rows, args.layer, args.seed)
        model = blob["model"]

    metrics = {
        "model": model,
        "position": "last_prompt_token (before any generated token)",
        "task": "2-option two-digit addition; label = letter of the correct option",
        "n_prompts": len(rows),
        "split": "80/20 grouped by question (both orderings of a question stay together)",
        "probe": "StandardScaler + LogisticRegression(C=0.1)",
        "results": results,
    }
    with open(d / "probe_v1.pkl", "wb") as f:
        pickle.dump(probes, f)
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for fmt, m in results.items():
        print(f"[{fmt}] behavior={m['behavior']}")
        print(f"[{fmt}] L{m['layer']} test acc={m['headline']['test_accuracy']:.3f} "
              f"(5 seeds {m['test_accuracy_over_5_split_seeds']['mean']:.3f}"
              f"±{m['test_accuracy_over_5_split_seeds']['std']:.3f}), "
              f"shuffled={m['shuffled_label_control_test_accuracy']}, "
              f"positive={m['positive_controls_test_accuracy']}")
        print(f"[{fmt}] by layer: {m['test_accuracy_by_layer']}")
        if m["model_own_answer_probe"]:
            print(f"[{fmt}] own-answer probe: {m['model_own_answer_probe']}")


if __name__ == "__main__":
    main()
