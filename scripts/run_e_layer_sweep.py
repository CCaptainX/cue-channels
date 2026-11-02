#!/usr/bin/env python3
"""Experiment E: pre-registered late-layer pre-CoT probe on BBH logical_deduction_three_objects.

Implements experiments/E_layer_sweep/preregistration.md exactly (commit 8647c89).
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
import re
import time
from pathlib import Path

import numpy as np

from cue_channels.config import RunConfig
from cue_channels.eval.cue_tasks import SYSTEM as COT_SYSTEM
from cue_channels.models import LocalBackend

DIRECT_SYSTEM = "Reply with only the letter of the correct option, formatted as 'Answer: <letter>'."
E_DIR = Path("experiments/E_layer_sweep")
DATA = E_DIR / "data/logical_deduction_three_objects.json"
LETTERS = "ABC"
LATE = list(range(21, 28))
PRIMARY_LAYER = 24
_PARSE = re.compile(r"(?:Answer:\s*|answer is:?\s*)\**\(?([A-Z])\)?", re.IGNORECASE)
_CATS = ("books", "birds", "golfers", "vehicles", "fruits")
DEVIATIONS: set[str] = set()


def parse(text: str) -> str | None:
    m = _PARSE.findall(text)
    if not m:
        return None
    x = m[-1].upper()
    return x if x in LETTERS else None


def load_items() -> list[dict]:
    data = json.loads(DATA.read_text())["examples"]
    idx = sorted(random.Random(0).sample(range(len(data)), 150))
    items = []
    for i in idx:
        e = data[i]
        cat = next(c for c in _CATS if c in e["input"])
        items.append({"idx": i, "input": e["input"], "target": e["target"].strip("()"),
                      "category": cat})
    return items


def collect(backend: LocalBackend, items: list[dict], n_layers: int) -> dict:
    layers = list(range(n_layers))
    out = {"cot_acts": {k: [] for k in layers}, "direct_acts": {k: [] for k in layers},
           "cot_text": [], "direct_text": []}
    for i, it in enumerate(items):
        for kind, system, max_new in (("cot", COT_SYSTEM, 512), ("direct", DIRECT_SYSTEM, 8)):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": it["input"]}]
            for lyr, v in backend.get_activations(msgs, layers).items():
                out[f"{kind}_acts"][lyr].append(v)
            out[f"{kind}_text"].append(backend.generate(msgs, max_new_tokens=max_new).text)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(items)}", flush=True)
    for kind in ("cot", "direct"):
        out[f"{kind}_acts"] = {k: np.stack(v) for k, v in out[f"{kind}_acts"].items()}
    return out


def cv_predictions(X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Stratified 5-fold CV (shuffle, seed 0); per-item correctness, each item tested once."""
    from sklearn.model_selection import KFold, StratifiedKFold

    from cue_channels.probes.train import _make_pipeline

    _, counts = np.unique(y, return_counts=True)
    if counts.min() >= 5:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    else:  # deviation from the pre-registration; recorded in metrics
        DEVIATIONS.add("plain KFold used for a probe whose rarest class had < 5 items")
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    correct = np.zeros(len(y), dtype=bool)
    for tr, te in splitter.split(X, y):
        if len(np.unique(y[tr])) < 2:
            correct[te] = y[te] == y[tr][0]
            continue
        clf = _make_pipeline(C=0.1, seed=seed).fit(X[tr], y[tr])
        correct[te] = clf.predict(X[te]) == y[te]
    return correct


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n_boot)]
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def probe(acts: dict[int, np.ndarray], labels: list[str | None], layers: list[int]) -> dict:
    keep = np.array([lab is not None for lab in labels])
    y = np.array([lab for lab in labels if lab is not None])
    res = {"n": int(keep.sum()), "excluded_unparsed": int((~keep).sum()),
           "label_counts": dict(collections.Counter(y.tolist())),
           "majority_rate": round(max(collections.Counter(y.tolist()).values()) / len(y), 3),
           "by_layer": {}}
    for lyr in layers:
        c = cv_predictions(acts[lyr][keep], y)
        res["by_layer"][lyr] = {"acc": round(float(c.mean()), 3), "ci95": bootstrap_ci(c)}
    return res


def permutation_null(X: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    accs = np.array([cv_predictions(X, rng.permutation(y)).mean() for _ in range(n_perm)])
    return {"n_perm": n_perm, "mean": round(float(accs.mean()), 3),
            "p95": round(float(np.percentile(accs, 95)), 3), "accs": accs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    items = load_items()
    cfg = RunConfig(model="qwen2.5-1.5b")
    pkl = E_DIR / "e_activations.pkl"
    t0 = time.time()
    if args.reuse and pkl.exists():
        blob = pickle.load(open(pkl, "rb"))
    else:
        backend = LocalBackend(cfg.model_config, cfg)
        backend.load()
        blob = collect(backend, items, cfg.model_config.n_layers)
        blob["device"] = backend.device
        backend.shutdown()
        blob["seconds"] = round(time.time() - t0)
        pickle.dump(blob, open(pkl, "wb"))

    cot_ans = [parse(t) for t in blob["cot_text"]]
    dir_ans = [parse(t) for t in blob["direct_text"]]
    truth = [it["target"] for it in items]
    n = len(items)
    no_cot_acc = sum(a == t for a, t in zip(dir_ans, truth, strict=True)) / n
    cot_acc = sum(a == t for a, t in zip(cot_ans, truth, strict=True)) / n
    both = [(d, c) for d, c in zip(dir_ans, cot_ans, strict=True) if d and c]
    behavior = {
        "n": n,
        "no_cot_acc": round(no_cot_acc, 3),
        "no_cot_acc_ci95": bootstrap_ci(np.array([a == t for a, t in zip(dir_ans, truth,
                                                                        strict=True)])),
        "with_cot_acc": round(cot_acc, 3),
        "with_cot_acc_ci95": bootstrap_ci(np.array([a == t for a, t in zip(cot_ans, truth,
                                                                          strict=True)])),
        "no_cot_unparsed": dir_ans.count(None),
        "with_cot_unparsed": cot_ans.count(None),
        "no_cot_letter_dist": dict(collections.Counter(str(a) for a in dir_ans)),
        "with_cot_letter_dist": dict(collections.Counter(str(a) for a in cot_ans)),
        "truth_letter_dist": dict(collections.Counter(truth)),
        "no_cot_vs_with_cot_agreement": round(sum(d == c for d, c in both) / len(both), 3)
        if both else None,
    }
    all_layers = sorted(blob["cot_acts"])

    primary = probe(blob["cot_acts"], dir_ans, all_layers)
    cot_target = probe(blob["cot_acts"], cot_ans, all_layers)
    truth_target = probe(blob["cot_acts"], truth, all_layers)
    upper_ref = probe(blob["direct_acts"], dir_ans, all_layers)
    cats = [it["category"] for it in items]
    pos_ctrl = probe(blob["cot_acts"], cats, [PRIMARY_LAYER])["by_layer"][PRIMARY_LAYER]

    keep = np.array([a is not None for a in dir_ans])
    perm = permutation_null(blob["cot_acts"][PRIMARY_LAYER][keep],
                            np.array([a for a in dir_ans if a is not None]))
    p24 = primary["by_layer"][PRIMARY_LAYER]
    perm_p = float((np.sum(perm["accs"] >= p24["acc"]) + 1) / (perm["n_perm"] + 1))
    perm.pop("accs")
    perm["p_value"] = round(perm_p, 4)

    threshold_chance = max(primary["majority_rate"], perm["p95"])
    cond1 = p24["acc"] >= no_cot_acc - 0.05
    cond2 = p24["ci95"][0] > threshold_chance
    verdict = "SUPPORTED" if (cond1 and cond2) else "NOT SUPPORTED"

    metrics = {
        "preregistration": "experiments/E_layer_sweep/preregistration.md (commit 8647c89)",
        "model": cfg.model_config.hf_id,
        "task": "BBH logical_deduction_three_objects, N=150 (seed 0), zero-shot, chat template",
        "device": blob.get("device"),
        "runtime_seconds": blob.get("seconds"),
        "behavior": behavior,
        "primary": {
            "probe": "CoT-prompt activations @ L24 -> model's own no-CoT letter",
            "acc": p24["acc"], "ci95": p24["ci95"],
            "no_cot_acc": round(no_cot_acc, 3),
            "rule1_probe_ge_no_cot_minus_0.05": cond1,
            "majority_rate": primary["majority_rate"],
            "permutation": perm,
            "chance_threshold": threshold_chance,
            "rule2_ci_lower_gt_chance_threshold": cond2,
            "verdict": verdict,
        },
        "late_layers_primary_probe": {lyr: primary["by_layer"][lyr] for lyr in LATE},
        "positive_control_category_L24": pos_ctrl,
        "secondary": {
            "cot_acts_to_with_cot_letter": cot_target,
            "cot_acts_to_ground_truth": truth_target,
            "direct_acts_to_own_no_cot_letter_upper_ref": upper_ref,
            "primary_all_layers": primary,
        },
        "deviations": sorted(DEVIATIONS),
    }
    (E_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with open(E_DIR / "generations.jsonl", "w") as f:
        for it, c, d, ct, dt in zip(items, cot_ans, dir_ans, blob["cot_text"], blob["direct_text"],
                                    strict=True):
            f.write(json.dumps({"idx": it["idx"], "target": it["target"], "no_cot": d,
                                "with_cot": c, "no_cot_text": dt, "cot_text": ct}) + "\n")
    print(json.dumps({"behavior": behavior, "primary": metrics["primary"]}, indent=2))
    for name, r in (("PRIMARY own no-CoT", primary), ("with-CoT letter", cot_target),
                    ("ground truth", truth_target), ("UPPER direct acts", upper_ref)):
        print(name, {lyr: r["by_layer"][lyr]["acc"] for lyr in LATE}, "maj", r["majority_rate"])
    print("pos ctrl", pos_ctrl)


if __name__ == "__main__":
    main()
