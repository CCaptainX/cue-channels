#!/usr/bin/env python3
"""Experiment D: is B's pre-CoT null caused by post-training ("distillation")?

Re-runs B's probe on the base model Qwen/Qwen2.5-1.5B with 4-shot raw-text prompts, on the
exact 200 prompts from experiments/B_probe/labels.jsonl. To separate model from prompt format,
the Instruct model is run on the *identical* raw 4-shot prompts too (2 models × 2 formats).
Formats: cot ("Reasoning:" then "Answer: X") and direct ("Answer:" immediately).
Activations: last prompt token (just before the first generated token), all layers.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import time
from pathlib import Path

import numpy as np

from cue_channels.config import RunConfig
from cue_channels.models import LocalBackend
from cue_channels.probes.analysis import analyze_condition, wilson_ci

MODELS = {"base": "qwen2.5-1.5b-base", "instruct": "qwen2.5-1.5b"}


def make_exemplars(exclude: set[tuple[int, int]], seed: int = 1234) -> list[dict]:
    """4 addition exemplars disjoint from the B questions; correct letters A, B, B, A."""
    rng = random.Random(seed)
    out = []
    for letter in "ABBA":
        while True:
            a, b = rng.randint(12, 89), rng.randint(12, 89)
            if (a, b) not in exclude:
                break
        exclude.add((a, b))
        c = a + b
        w = c + rng.choice([-10, -1, 1, 10])
        opts = (c, w) if letter == "A" else (w, c)
        out.append({"a": a, "b": b, "sum": c, "opts": opts, "label": letter})
    return out


def exemplar_text(ex: dict, fmt: str) -> str:
    q = f"Question: What is {ex['a']} + {ex['b']}?\n(A) {ex['opts'][0]}\n(B) {ex['opts'][1]}\n"
    if fmt == "cot":
        return (f"{q}Reasoning: {ex['a']} + {ex['b']} = {ex['sum']}. "
                f"{ex['sum']} is option ({ex['label']}).\nAnswer: {ex['label']}\n\n")
    return f"{q}Answer: {ex['label']}\n\n"


def build_prompt(row: dict, exemplars: list[dict], fmt: str) -> str:
    shots = "".join(exemplar_text(e, fmt) for e in exemplars)
    q = row["prompt"].replace("What is", "Question: What is", 1)
    return f"{shots}{q}\n" + ("Reasoning:" if fmt == "cot" else "Answer:")


_ANS = re.compile(r"Answer:\s*\(?([AB])\b")


def parse(fmt: str, completion: str) -> str | None:
    if fmt == "direct":
        m = re.match(r"\s*\(?([AB])\b", completion)
        return m.group(1) if m else None
    # Only the first "Answer:" counts; base models keep going with new questions.
    m = _ANS.search(completion)
    return m.group(1) if m else None


def run_condition(backend: LocalBackend, rows, exemplars, fmt, n_layers) -> dict:
    layers = list(range(n_layers))
    acts = {lyr: [] for lyr in layers}
    answers, samples = [], []
    for i, r in enumerate(rows):
        prompt = build_prompt(r, exemplars, fmt)
        for lyr, v in backend.get_activations_text(prompt, layers).items():
            acts[lyr].append(v)
        gen = backend.generate_text(prompt, max_new_tokens=48 if fmt == "cot" else 3)
        answers.append(parse(fmt, gen.text))
        if i < 3:
            samples.append({"prompt_tail": prompt[-120:], "completion": gen.text[:200]})
    return {"acts": {k: np.stack(v) for k, v in acts.items()}, "answers": answers,
            "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b-dir", default="experiments/B_probe")
    ap.add_argument("--out", default="experiments/D_base_vs_instruct")
    ap.add_argument("--models", default="base,instruct")
    ap.add_argument("--reuse", action="store_true", help="reuse saved activation pickles")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(Path(args.b_dir) / "labels.jsonl")]
    used = {tuple(map(int, re.search(r"What is (\d+) \+ (\d+)", r["prompt"]).groups()))
            for r in rows}
    exemplars = make_exemplars(set(used))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results, timing, samples = {}, {}, {}
    for name in args.models.split(","):
        key = MODELS[name]
        cfg = RunConfig(model=key)
        backend = None
        for fmt in ("cot", "direct"):
            pkl = out / f"{name}_{fmt}_activations.pkl"
            if args.reuse and pkl.exists():
                blob = pickle.load(open(pkl, "rb"))
            else:
                if backend is None:
                    backend = LocalBackend(cfg.model_config, cfg)
                    backend.load()
                t0 = time.time()
                blob = run_condition(backend, rows, exemplars, fmt, cfg.model_config.n_layers)
                blob["seconds"] = round(time.time() - t0)
                blob["device"] = backend.device
                pickle.dump(blob, open(pkl, "wb"))
            cond = f"{name}_{fmt}"
            timing[cond] = blob.get("seconds")
            samples[cond] = blob["samples"]
            results[cond] = analyze_condition(blob["acts"], rows, blob["answers"])
            r = results[cond]
            print(f"[{cond}] probe={r['correct_letter_probe']['grouped_5fold_cv']} "
                  f"behavior={r['behavior']} pos={r['positive_controls']} "
                  f"own={r.get('own_answer_probe', {}).get('accuracy')}", flush=True)
        if backend is not None:
            backend.shutdown()

    # B's chat-template Instruct result, recomputed with the same grouped-CV statistic.
    b_metrics = json.loads((Path(args.b_dir) / "metrics.json").read_text())["results"]["cot"]
    b_recomputed = None
    b_pkl = Path(args.b_dir) / "dummy_activations.pkl"
    if b_pkl.exists():
        b_blob = pickle.load(open(b_pkl, "rb"))
        b_full = analyze_condition(b_blob["activations"], rows)
        b_recomputed = b_full["correct_letter_probe"]
        results["instruct_chat_cot_from_B"] = b_full

    base = results.get("base_cot", {})
    cv = base.get("correct_letter_probe", {}).get("grouped_5fold_cv", {})
    beh = base.get("behavior", {})
    d_beh = results.get("base_direct", {}).get("behavior", {})
    summary = {
        "base_probe_acc": cv.get("accuracy"),
        "base_probe_ci": cv.get("ci95"),
        "instruct_probe_acc": b_metrics["test_accuracy_over_5_split_seeds"]["mean"],
        "instruct_probe_acc_source": "experiments/B_probe/metrics.json (chat template, 80/20 x5)",
        "instruct_probe_acc_grouped_cv_recomputed": (
            b_recomputed["grouped_5fold_cv"] if b_recomputed else None
        ),
        "instruct_same_raw_prompt_probe": (
            results.get("instruct_cot", {}).get("correct_letter_probe", {}).get("grouped_5fold_cv")
        ),
        "base_control_probe_acc": base.get("positive_controls"),
        "base_shuffled_control_acc": base.get("shuffled_label_control", {}).get("accuracy"),
        "base_direct_answer_acc": d_beh.get("accuracy_all"),
        "base_direct_answer_ci": (
            wilson_ci(d_beh["correct"], d_beh["n"]) if d_beh else None
        ),
        "base_direct_frac_A": d_beh.get("frac_answer_A_of_parsed"),
        "base_cot_accuracy": beh.get("accuracy_all"),
        "base_cot_accuracy_ci": wilson_ci(beh["correct"], beh["n"]) if beh else None,
        "base_cot_parsed": beh.get("parsed"),
    }
    metrics = {
        "experiment": "D: base vs instruct pre-CoT probe (distillation hypothesis)",
        "questions": "identical 200 prompts from experiments/B_probe/labels.jsonl",
        "prompting": "4-shot raw text (exemplars disjoint from the 200; letters A,B,B,A)",
        "exemplars": [exemplar_text(e, "cot") for e in exemplars],
        "position": "last prompt token ('Reasoning:' for cot, 'Answer:' for direct)",
        "probe": "StandardScaler + LogisticRegression(C=0.1); headline = grouped 5-fold CV "
                 "over all 200 (Wilson 95% CI); 80/20 x5 seeds also reported (B's statistic)",
        "layer": 14,
        "summary": summary,
        "conditions": results,
        "seconds": timing,
        "sample_completions": samples,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
