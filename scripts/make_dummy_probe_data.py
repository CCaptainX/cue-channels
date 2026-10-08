#!/usr/bin/env python3
"""Commit B data: 200 two-option arithmetic prompts, labelled by which letter is correct.

Each of 100 questions appears twice, once with the correct answer in (A) and once in (B),
so the label is balanced and not tied to question identity. Activations are taken at the
last prompt token (before any CoT token) at every layer.

``--format cot`` (default) uses the step-by-step system prompt; ``--format direct`` asks for
the letter only, so the model has to commit to an answer at that position. Comparing the two
tells extraction bugs apart from "the model hasn't decided yet".
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np

from cue_channels.config import DEFAULT_MODEL, RunConfig
from cue_channels.eval.cue_tasks import SYSTEM, parse_answer
from cue_channels.models import LocalBackend

DIRECT_SYSTEM = "Reply with only the letter of the correct option, formatted as 'Answer: <letter>'."


def make_questions(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    seen, qs = set(), []
    while len(qs) < n:
        a, b = rng.randint(12, 89), rng.randint(12, 89)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        correct = a + b
        wrong = correct + rng.choice([-11, -10, -9, -2, -1, 1, 2, 9, 10, 11])
        qs.append({"qid": len(qs), "question": f"What is {a} + {b}?", "correct": correct,
                   "wrong": wrong})
    return qs


def to_rows(qs: list[dict]) -> list[dict]:
    rows = []
    for q in qs:
        for label in ("A", "B"):
            opts = (q["correct"], q["wrong"]) if label == "A" else (q["wrong"], q["correct"])
            text = f"{q['question']}\n(A) {opts[0]}\n(B) {opts[1]}"
            rows.append({"qid": q["qid"], "label": label, "prompt": text})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-questions", type=int, default=100)
    ap.add_argument("--out", default="experiments/B_probe")
    ap.add_argument("--format", choices=["cot", "direct"], default="cot")
    ap.add_argument("--behavior-n", type=int, default=40,
                    help="prompts to also generate answers for (behavioral accuracy check)")
    args = ap.parse_args()
    system = SYSTEM if args.format == "cot" else DIRECT_SYSTEM

    rows = to_rows(make_questions(args.n_questions))
    cfg = RunConfig(model=args.model)
    backend = LocalBackend(cfg.model_config, cfg)
    backend.load()
    layers = list(range(cfg.model_config.n_layers))
    acts: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    n_correct = n_parsed = 0
    model_answers: list[str | None] = []
    for i, r in enumerate(rows):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}]
        for layer, vec in backend.get_activations(msgs, layers).items():
            acts[layer].append(vec)
        if i < args.behavior_n:
            max_new = 400 if args.format == "cot" else 8
            ans = parse_answer(backend.generate(msgs, max_new_tokens=max_new).text)
            model_answers.append(ans)
            n_parsed += ans is not None
            n_correct += ans == r["label"]
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}")
    backend.shutdown()
    behavior = {"format": args.format, "n": min(args.behavior_n, len(rows)),
                "parsed": n_parsed, "correct": n_correct}
    print("behavior:", behavior)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stacked = {layer: np.stack(v) for layer, v in acts.items()}
    suffix = "" if args.format == "cot" else "_direct"
    with open(out / f"dummy_activations{suffix}.pkl", "wb") as f:
        pickle.dump({"model": cfg.model_config.hf_id, "position": "last_prompt_token",
                     "format": args.format, "system": system, "behavior": behavior,
                     "model_answers": model_answers,
                     "activations": stacked}, f)
    with open(out / "labels.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"saved {len(rows)} rows, {len(layers)} layers, d={stacked[0].shape[1]}")


if __name__ == "__main__":
    main()
