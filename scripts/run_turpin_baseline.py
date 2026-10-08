#!/usr/bin/env python3
"""Commit C: few-shot "answer is always X" bias (Turpin et al. 2023) on the local model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cue_channels.config import DEFAULT_MODEL, RunConfig
from cue_channels.eval.cue_tasks import (
    QUESTION_POOL,
    generate_cue_prompts,
    mentions_fewshot_pattern,
    parse_best_answer,
)
from cue_channels.eval.runner import fewshot_bias_metrics, run_fewshot_bias
from cue_channels.models import LocalBackend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--directions", default="A,B", help="cue letters to run, comma-separated")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N questions")
    ap.add_argument("--out", default="experiments/C_turpin_baseline/results.json")
    ap.add_argument("--rescore", action="store_true",
                    help="re-parse saved CoTs in --out with the current parsers; no generation")
    args = ap.parse_args()

    if args.rescore:
        rescore(Path(args.out))
        return

    cfg = RunConfig(model=args.model)
    backend = LocalBackend(cfg.model_config, cfg)
    backend.load()
    pool = QUESTION_POOL[: args.limit] if args.limit else QUESTION_POOL
    runs = {}
    try:
        for direction in args.directions.split(","):
            t0 = time.time()
            pairs = generate_cue_prompts(args.n_examples, direction, pool)
            res = run_fewshot_bias(backend, pairs, max_new_tokens=args.max_new_tokens)
            res["metrics"]["seconds"] = round(time.time() - t0)
            runs[direction] = res
            print(f"[cue={direction}]", json.dumps(res["metrics"]))
    finally:
        backend.shutdown()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": cfg.model_config.hf_id,
        "device": backend.device,
        "decoding": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "cue": "few-shot answer-position bias (Turpin et al. 2023, arXiv:2305.04388)",
        "n_fewshot_examples": args.n_examples,
        "n_questions": len(pool),
        "summary": {d: r["metrics"] for d, r in runs.items()},
        "runs": runs,
    }, indent=2, ensure_ascii=False) + "\n")
    print("saved", out)


def rescore(path: Path) -> None:
    d = json.loads(path.read_text())
    for direction, run in d["runs"].items():
        for r in run["pairs"]:
            for cond in ("unbiased", "biased"):
                r[f"{cond}_answer"] = parse_best_answer(r[f"{cond}_cot"])
                r[f"{cond}_mentions_pattern"] = mentions_fewshot_pattern(r[f"{cond}_cot"])
            r["influenced"] = (r["biased_answer"] == r["cue"]
                               and r["unbiased_answer"] != r["cue"])
        seconds = run["metrics"].get("seconds")
        run["metrics"] = {**fewshot_bias_metrics(run["pairs"]), "seconds": seconds}
        d["summary"][direction] = run["metrics"]
        print(f"[cue={direction}]", json.dumps(run["metrics"]))
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
