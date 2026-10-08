#!/usr/bin/env python3
"""Experiment E2: E with BBH standard 3-shot raw-text prompts and a no-CoT pilot gate.

Implements experiments/E2_layer_sweep_3shot/preregistration.md (commit c918f6e), which inherits
experiments/E_layer_sweep/preregistration.md (0f7353a). Probe/statistics code is reused from E.

  python scripts/run_e2_layer_sweep.py --pilot   # 30-item no-CoT gate
  python scripts/run_e2_layer_sweep.py --full    # N=150, only if the gate passed
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_e_layer_sweep import (  # noqa: E402
    LATE,
    PRIMARY_LAYER,
    bootstrap_ci,
    load_items,
    permutation_null,
    probe,
)

from cue_channels.config import RunConfig  # noqa: E402
from cue_channels.models import LocalBackend  # noqa: E402

E2_DIR = Path("experiments/E2_layer_sweep_3shot")
COT_FILE = E2_DIR / "data/logical_deduction_three_objects_cot.txt"
LETTERS = "ABC"
_FIRST_ANS = re.compile(r"answer is \(?([A-C])\)?")


def load_prompts() -> tuple[str, str]:
    """(cot_prefix, answer_only_prefix) built verbatim from the BBH 3-shot CoT file."""
    txt = COT_FILE.read_text()
    body = txt.split("-----\n", 1)[1].strip()  # drop canary header
    description, *blocks = re.split(r"\n\n(?=Q: )", body)
    cot = description + "\n\n" + "\n\n".join(b.strip() for b in blocks)
    ao_blocks = []
    for b in blocks:
        q, a = b.split("\nA: ", 1)
        letter = re.search(r"So the answer is \(([A-C])\)\.", a).group(1)
        ao_blocks.append(f"{q}\nA: ({letter})")
    return cot, description + "\n\n" + "\n\n".join(ao_blocks)


def cot_prompt(prefix: str, item: dict) -> str:
    return f"{prefix}\n\nQ: {item['input']}\nA: Let's think step by step."


def no_cot_prompt(prefix: str, item: dict) -> str:
    return f"{prefix}\n\nQ: {item['input']}\nA: ("


def parse_no_cot(text: str) -> str | None:
    t = text.strip()
    return t[0] if t and t[0] in LETTERS else None


def parse_cot(text: str) -> str | None:
    m = _FIRST_ANS.search(text)
    return m.group(1) if m else None


def run_pilot(backend: LocalBackend, items: list[dict], ao_prefix: str) -> dict:
    pilot = items[:30]
    answers, texts = [], []
    for it in pilot:
        g = backend.generate_text(no_cot_prompt(ao_prefix, it), max_new_tokens=2).text
        texts.append(g)
        answers.append(parse_no_cot(g))
    truth = [it["target"] for it in pilot]
    correct = np.array([a == t for a, t in zip(answers, truth, strict=True)])
    acc = float(correct.mean())
    dist = collections.Counter(str(a) for a in answers)
    top_letter, top_n = max(((k, v) for k, v in dist.items() if k != "None"),
                            key=lambda kv: kv[1], default=(None, 0))
    gate = 0.55 <= acc <= 0.85
    return {
        "n": len(pilot),
        "item_idx": [it["idx"] for it in pilot],
        "no_cot_acc": round(acc, 3),
        "no_cot_acc_ci95": bootstrap_ci(correct),
        "letter_dist": dict(dist),
        "truth_dist": dict(collections.Counter(truth)),
        "unparsed": answers.count(None),
        "top_letter": top_letter,
        "top_letter_share": round(top_n / len(pilot), 3),
        "bias_flag_gt_0.60": top_n / len(pilot) > 0.60,
        "gate_band": [0.55, 0.85],
        "gate": "PASS" if gate else "FAIL",
        "completions": texts,
    }


def run_full(backend: LocalBackend, items: list[dict], cot_prefix: str, ao_prefix: str,
             n_layers: int) -> dict:
    layers = list(range(n_layers))
    out = {"cot_acts": {k: [] for k in layers}, "direct_acts": {k: [] for k in layers},
           "cot_text": [], "direct_text": []}
    for i, it in enumerate(items):
        for kind, prompt, max_new in (("cot", cot_prompt(cot_prefix, it), 256),
                                      ("direct", no_cot_prompt(ao_prefix, it), 2)):
            for lyr, v in backend.get_activations_text(prompt, layers).items():
                out[f"{kind}_acts"][lyr].append(v)
            out[f"{kind}_text"].append(backend.generate_text(prompt, max_new_tokens=max_new).text)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(items)}", flush=True)
    for kind in ("cot", "direct"):
        out[f"{kind}_acts"] = {k: np.stack(v) for k, v in out[f"{kind}_acts"].items()}
    return out


def analyze(blob: dict, items: list[dict]) -> dict:
    cot_ans = [parse_cot(t) for t in blob["cot_text"]]
    dir_ans = [parse_no_cot(t) for t in blob["direct_text"]]
    truth = [it["target"] for it in items]
    n = len(items)
    dir_correct = np.array([a == t for a, t in zip(dir_ans, truth, strict=True)])
    cot_correct = np.array([a == t for a, t in zip(cot_ans, truth, strict=True)])
    no_cot_acc = float(dir_correct.mean())
    both = [(d, c) for d, c in zip(dir_ans, cot_ans, strict=True) if d and c]
    behavior = {
        "n": n,
        "no_cot_acc": round(no_cot_acc, 3), "no_cot_acc_ci95": bootstrap_ci(dir_correct),
        "with_cot_acc": round(float(cot_correct.mean()), 3),
        "with_cot_acc_ci95": bootstrap_ci(cot_correct),
        "no_cot_unparsed": dir_ans.count(None), "with_cot_unparsed": cot_ans.count(None),
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
    pos_ctrl = probe(blob["cot_acts"], [it["category"] for it in items],
                     [PRIMARY_LAYER])["by_layer"][PRIMARY_LAYER]

    keep = np.array([a is not None for a in dir_ans])
    perm = permutation_null(blob["cot_acts"][PRIMARY_LAYER][keep],
                            np.array([a for a in dir_ans if a is not None]))
    p24 = primary["by_layer"][PRIMARY_LAYER]
    perm["p_value"] = round(float((np.sum(perm["accs"] >= p24["acc"]) + 1) /
                                  (perm["n_perm"] + 1)), 4)
    perm.pop("accs")
    rule1 = p24["acc"] >= no_cot_acc - 0.05
    floor = max(primary["majority_rate"], perm["p95"])
    rule2 = p24["ci95"][0] > floor
    floor_restated = max(0.333, primary["majority_rate"])
    return {
        "behavior": behavior,
        "primary": {
            "probe": "CoT-prompt activations @ L24 -> model's own no-CoT letter",
            "acc": p24["acc"], "ci95": p24["ci95"], "no_cot_acc": round(no_cot_acc, 3),
            "rule1_probe_ge_no_cot_minus_0.05": rule1,
            "majority_rate": primary["majority_rate"], "permutation": perm,
            "rule2_floor_E_wording": floor, "rule2_ci_lower_gt_floor": rule2,
            "rule2_restated_floor_max_0.333_majority": floor_restated,
            "rule2_restated_pass": p24["ci95"][0] > floor_restated,
            "verdict": "SUPPORTED" if (rule1 and rule2) else "NOT SUPPORTED",
        },
        "late_layers_primary_probe": {lyr: primary["by_layer"][lyr] for lyr in LATE},
        "positive_control_category_L24": pos_ctrl,
        "secondary": {
            "cot_acts_to_with_cot_letter": cot_target,
            "cot_acts_to_ground_truth": truth_target,
            "direct_acts_to_own_no_cot_letter_upper_ref": upper_ref,
            "primary_all_layers": primary,
        },
        "_answers": (dir_ans, cot_ans),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    items = load_items()
    cot_prefix, ao_prefix = load_prompts()
    cfg = RunConfig(model="qwen2.5-1.5b")

    if args.pilot:
        backend = LocalBackend(cfg.model_config, cfg)
        backend.load()
        t0 = time.time()
        res = run_pilot(backend, items, ao_prefix)
        backend.shutdown()
        res["seconds"] = round(time.time() - t0)
        res["device"] = backend.device
        (E2_DIR / "pilot.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "completions"}, indent=2))
        return

    pilot = json.loads((E2_DIR / "pilot.json").read_text())
    if pilot["gate"] != "PASS":
        sys.exit(f"pilot gate {pilot['gate']}; full run not allowed by the pre-registration")
    pkl = E2_DIR / "e2_activations.pkl"
    t0 = time.time()
    if args.reuse and pkl.exists():
        blob = pickle.load(open(pkl, "rb"))
    else:
        backend = LocalBackend(cfg.model_config, cfg)
        backend.load()
        blob = run_full(backend, items, cot_prefix, ao_prefix, cfg.model_config.n_layers)
        blob["device"] = backend.device
        backend.shutdown()
        blob["seconds"] = round(time.time() - t0)
        pickle.dump(blob, open(pkl, "wb"))
    res = analyze(blob, items)
    dir_ans, cot_ans = res.pop("_answers")
    metrics = {
        "preregistration": "E2 c918f6e (inherits E 0f7353a)",
        "model": cfg.model_config.hf_id,
        "task": "BBH logical_deduction_three_objects, N=150 (seed 0), BBH 3-shot, raw text",
        "device": blob.get("device"), "runtime_seconds": blob.get("seconds"),
        "pilot": {k: v for k, v in pilot.items() if k != "completions"},
        **res,
    }
    (E2_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with open(E2_DIR / "generations.jsonl", "w") as f:
        for it, d, c, dt, ct in zip(items, dir_ans, cot_ans, blob["direct_text"], blob["cot_text"],
                                    strict=True):
            f.write(json.dumps({"idx": it["idx"], "target": it["target"], "no_cot": d,
                                "with_cot": c, "no_cot_text": dt, "cot_text": ct}) + "\n")
    print(json.dumps({"behavior": res["behavior"], "primary": res["primary"]}, indent=2))
    for name, key in (("PRIMARY", "primary_all_layers"),
                      ("with-CoT", "cot_acts_to_with_cot_letter"),
                      ("truth", "cot_acts_to_ground_truth"),
                      ("UPPER", "direct_acts_to_own_no_cot_letter_upper_ref")):
        r = res["secondary"][key]
        print(name, {lyr: r["by_layer"][lyr]["acc"] for lyr in LATE}, "maj", r["majority_rate"])
    print("pos ctrl", res["positive_control_category_L24"])


if __name__ == "__main__":
    main()
