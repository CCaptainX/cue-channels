#!/usr/bin/env python3
"""Run the toy cue eval locally. Downloads model weights on first run."""

from __future__ import annotations

import argparse
import json

from cue_channels.config import DEFAULT_MODEL, MODEL_REGISTRY, RunConfig
from cue_channels.eval.cue_tasks import CueChannel, CueType, toy_items
from cue_channels.eval.runner import run_eval, summarize
from cue_channels.models import make_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--out", default="results/")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--layers", type=int, nargs="*", help="layers to cache (default: none)")
    args = ap.parse_args()

    cfg = RunConfig(model=args.model, max_new_tokens=args.max_new_tokens, output_dir=args.out)
    backend = make_backend(cfg)
    backend.load()
    try:
        records = run_eval(
            backend,
            toy_items(),
            channels=list(CueChannel),
            cue_types=list(CueType),
            probe_layers=args.layers,
            out_dir=f"{args.out.rstrip('/')}/{args.model}",
        )
    finally:
        backend.shutdown()
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
