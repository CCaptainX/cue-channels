#!/usr/bin/env python3
"""Experiments G and H (pre-registration experiments/G_scale_7b/preregistration.md, c1f8159).

  run_gh.py load-check --model qwen2.5-7b
  run_gh.py pilot --model qwen2.5-7b --suite bbh --out experiments/G_scale_7b
  run_gh.py pilot --model qwen2.5-1.5b --suite arith --out experiments/H_arithmetic
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import f_common  # noqa: E402
import h_arith  # noqa: E402
from run_e_layer_sweep import bootstrap_ci  # noqa: E402

from cue_channels.config import RunConfig  # noqa: E402
from cue_channels.models import LocalBackend  # noqa: E402

STOP = ["\nQ:"]


def suite_tasks(suite: str) -> list[str]:
    return f_common.TASKS if suite == "bbh" else ["A", "B", "C"]


def task_items(suite: str, task: str, n: int) -> list[dict]:
    return f_common.load_items(task, 200)[:n] if suite == "bbh" else h_arith.items(task, n)


def task_prompts(suite: str, task: str) -> tuple[str, str]:
    return f_common.load_prompts(task) if suite == "bbh" else h_arith.prompts(task)


def load_backend(model: str) -> LocalBackend:
    cfg = RunConfig(model=model)
    b = LocalBackend(cfg.model_config, cfg)
    b.load()
    return b


def load_check(model: str, out: Path) -> None:
    import torch

    t0 = time.time()
    b = load_backend(model)
    load_s = time.time() - t0
    prompt = "Q: What is 2 + 2?\nA: Let's think step by step."
    ids = b.encode_raw(prompt)
    with torch.no_grad():
        o = b.model(ids, output_hidden_states=True)
    hs = o.hidden_states
    acts = b.get_activations_text(prompt, [0, len(hs) - 2])
    gen = b.generate_text(prompt, max_new_tokens=24, stop_strings=STOP)
    cfg = b.model.config
    res = {
        "model": b.model_config.hf_id, "device": b.device, "dtype": str(b.model.dtype),
        "env_PYTORCH_MPS_HIGH_WATERMARK_RATIO": __import__("os").environ.get(
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
        "n_layers": cfg.num_hidden_layers, "d_model": cfg.hidden_size,
        "n_hidden_states": len(hs), "hidden_state_shape": list(hs[-1].shape),
        "activation_vector_shapes": {k: list(v.shape) for k, v in acts.items()},
        "finite": bool(all(np.isfinite(v).all() for v in acts.values())),
        "load_seconds": round(load_s, 1),
        "mps_allocated_gb": round(torch.mps.current_allocated_memory() / 1e9, 2)
        if b.device == "mps" else None,
        "sample_generation": gen.text,
    }
    assert res["n_hidden_states"] == res["n_layers"] + 1
    assert all(s == [res["d_model"]] for s in res["activation_vector_shapes"].values())
    b.shutdown()
    out.mkdir(parents=True, exist_ok=True)
    (out / "load_check.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


def pilot(model: str, suite: str, out: Path) -> None:
    b = load_backend(model)
    results = {}
    for task in suite_tasks(suite):
        t0 = time.time()
        its = task_items(suite, task, 30)
        cot_pre, ao_pre = task_prompts(suite, task)
        no_cot, cot, texts = [], [], []
        for it in its:
            no_cot.append(f_common.parse_no_cot(
                b.generate_text(f_common.no_cot_prompt(ao_pre, it), max_new_tokens=2).text,
                it["letters"]))
            t = b.generate_text(f_common.cot_prompt(cot_pre, it), max_new_tokens=384,
                                stop_strings=STOP).text
            texts.append(t)
            cot.append(f_common.parse_cot(t, it["letters"]))
        truth = [it["target"] for it in its]
        nc = np.array([a == t for a, t in zip(no_cot, truth, strict=True)])
        wc = np.array([a == t for a, t in zip(cot, truth, strict=True)])
        dist = collections.Counter(str(a) for a in no_cot)
        top = max((v for k, v in dist.items() if k != "None"), default=0) / len(its)
        g1, g2, g3 = 0.55 <= nc.mean() <= 0.85, top <= 0.60, wc.mean() >= 0.85
        results[task] = {
            "n": len(its), "n_options": len(its[0]["letters"]),
            "no_cot_acc": round(float(nc.mean()), 3), "no_cot_ci95": bootstrap_ci(nc),
            "with_cot_acc": round(float(wc.mean()), 3), "with_cot_ci95": bootstrap_ci(wc),
            "no_cot_letter_dist": dict(dist),
            "with_cot_letter_dist": dict(collections.Counter(str(a) for a in cot)),
            "truth_dist": dict(collections.Counter(truth)),
            "top_letter_share": round(top, 3),
            "no_cot_unparsed": no_cot.count(None), "with_cot_unparsed": cot.count(None),
            "G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "pass": bool(g1 and g2 and g3),
            "seconds": round(time.time() - t0), "cot_samples": texts[:2],
        }
        r = results[task]
        print(f"[{task}] noCoT {r['no_cot_acc']} {r['no_cot_ci95']} | CoT {r['with_cot_acc']} "
              f"{r['with_cot_ci95']} | top {r['top_letter_share']} {r['no_cot_letter_dist']} | "
              f"G1 {g1} G2 {g2} G3 {g3} unparsed {r['with_cot_unparsed']} ({r['seconds']}s)",
              flush=True)
    b.shutdown()
    passing = [t for t in results if results[t]["pass"]]

    def gap(t: str) -> tuple[float, int]:
        return (results[t]["with_cot_acc"] - results[t]["no_cot_acc"], results[t]["n_options"])

    sel = max(passing, key=gap) if passing else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot.json").write_text(json.dumps(
        {"model": model, "suite": suite, "tasks": results, "passing": passing, "selected": sel},
        indent=2) + "\n")
    print("passing:", passing, "selected:", sel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load-check", "pilot"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", choices=["bbh", "arith"], default="bbh")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "load-check":
        load_check(a.model, Path(a.out))
    else:
        pilot(a.model, a.suite, Path(a.out))


if __name__ == "__main__":
    main()
