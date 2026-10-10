#!/usr/bin/env python3
"""Experiment F, step 1: pilot three BBH tasks (30 items each) against the pre-registered gates."""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from f_common import (  # noqa: E402
    F_DIR,
    TASKS,
    cot_prompt,
    load_items,
    load_prompts,
    no_cot_prompt,
    parse_cot,
    parse_no_cot,
)
from run_e_layer_sweep import bootstrap_ci  # noqa: E402

from cue_channels.config import RunConfig  # noqa: E402
from cue_channels.models import LocalBackend  # noqa: E402


def main() -> None:
    cfg = RunConfig(model="qwen2.5-1.5b")
    backend = LocalBackend(cfg.model_config, cfg)
    backend.load()
    results = {}
    for task in TASKS:
        t0 = time.time()
        items = load_items(task)[:30]
        cot_pre, ao_pre = load_prompts(task)
        no_cot, cot, cot_texts = [], [], []
        for it in items:
            no_cot.append(parse_no_cot(
                backend.generate_text(no_cot_prompt(ao_pre, it), max_new_tokens=2).text,
                it["letters"]))
            t = backend.generate_text(cot_prompt(cot_pre, it), max_new_tokens=384).text
            cot_texts.append(t)
            cot.append(parse_cot(t, it["letters"]))
        truth = [it["target"] for it in items]
        nc = np.array([a == t for a, t in zip(no_cot, truth, strict=True)])
        wc = np.array([a == t for a, t in zip(cot, truth, strict=True)])
        dist = collections.Counter(str(a) for a in no_cot)
        top = max((v for k, v in dist.items() if k != "None"), default=0) / len(items)
        g1 = 0.55 <= nc.mean() <= 0.85
        g2 = top <= 0.60
        g3 = wc.mean() >= 0.85
        results[task] = {
            "n": len(items), "n_options": len(items[0]["letters"]),
            "no_cot_acc": round(float(nc.mean()), 3), "no_cot_ci95": bootstrap_ci(nc),
            "with_cot_acc": round(float(wc.mean()), 3), "with_cot_ci95": bootstrap_ci(wc),
            "no_cot_letter_dist": dict(dist),
            "with_cot_letter_dist": dict(collections.Counter(str(a) for a in cot)),
            "truth_dist": dict(collections.Counter(truth)),
            "top_letter_share": round(top, 3),
            "no_cot_unparsed": no_cot.count(None), "with_cot_unparsed": cot.count(None),
            "G1_no_cot_in_[0.55,0.85]": bool(g1), "G2_top_letter_le_0.60": bool(g2),
            "G3_with_cot_ge_0.85": bool(g3), "pass": bool(g1 and g2 and g3),
            "seconds": round(time.time() - t0),
            "cot_samples": cot_texts[:2],
        }
        r = results[task]
        verdict = "PASS" if r["pass"] else "FAIL"
        print(f"[{task}] no-CoT {r['no_cot_acc']} {r['no_cot_ci95']} | "
              f"with-CoT {r['with_cot_acc']} {r['with_cot_ci95']} | "
              f"top {r['top_letter_share']} | dist {r['no_cot_letter_dist']} | "
              f"G1 {g1} G2 {g2} G3 {g3} -> {verdict} ({r['seconds']}s)", flush=True)
    backend.shutdown()
    passing = [t for t in TASKS if results[t]["pass"]]
    selected = None
    if passing:
        def gap(t: str) -> tuple[float, int]:
            return (results[t]["with_cot_acc"] - results[t]["no_cot_acc"], results[t]["n_options"])

        selected = max(passing, key=gap)
    out = {"tasks": results, "passing": passing, "selected": selected}
    (F_DIR / "pilot.json").write_text(json.dumps(out, indent=2) + "\n")
    print("passing:", passing, "selected:", selected)


if __name__ == "__main__":
    main()
