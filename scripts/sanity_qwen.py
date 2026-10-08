#!/usr/bin/env python3
"""Commit A sanity run: load the default model, generate a CoT, and grab mid-layer activations."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from cue_channels.config import DEFAULT_MODEL, RunConfig
from cue_channels.models import LocalBackend

PROMPT = (
    "Q: If a book has 300 pages and I read 50 pages a day, how many days will it take "
    "to finish? Let's think step by step:"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="experiments/A_sanity")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    cfg = RunConfig(model=args.model, max_new_tokens=args.max_new_tokens)
    backend = LocalBackend(cfg.model_config, cfg)
    t0 = time.time()
    backend.load()
    load_s = time.time() - t0

    messages = [{"role": "user", "content": PROMPT}]
    layer = cfg.model_config.n_layers // 2

    t0 = time.time()
    gen = backend.generate(messages)
    gen_s = time.time() - t0
    acts = backend.get_residual_stream(messages, layer)  # [seq, d_model]
    backend.shutdown()

    assert acts.shape[1] == cfg.model_config.d_model, acts.shape
    assert np.isfinite(acts).all(), "non-finite activations"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = cfg.model_config.key.replace("qwen2.5-", "qwen_")
    with open(out / f"{tag}_activations.pkl", "wb") as f:
        pickle.dump(
            {"model": cfg.model_config.hf_id, "layer": layer, "prompt": PROMPT,
             "messages": messages, "continuation": gen.text, "activations": acts},
            f,
        )

    last = acts[-1]
    summary = {
        "model": cfg.model_config.hf_id,
        "device": backend.device,
        "layer": layer,
        "n_layers": cfg.model_config.n_layers,
        "activations_shape": list(acts.shape),
        "activations_dtype": str(acts.dtype),
        "model_dtype": "bfloat16" if backend.device != "cpu" else "float32",
        "mean": float(acts.mean()),
        "std": float(acts.std()),
        "abs_max": float(np.abs(acts).max()),
        "last_token_norm": float(np.linalg.norm(last)),
        "prompt_tokens": gen.prompt_tokens,
        "completion_tokens": gen.completion_tokens,
        "load_seconds": round(load_s, 1),
        "generate_seconds": round(gen_s, 1),
        "prompt": PROMPT,
        "continuation": gen.text,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "continuation"}, indent=2))
    print("\n--- continuation ---\n" + gen.text)


if __name__ == "__main__":
    main()
