#!/usr/bin/env python3
"""Experiment K: per-token x per-layer probe maps (pre-registered, 1d8ad88)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import f_common  # noqa: E402
import h_arith  # noqa: E402
import i3_tasks  # noqa: E402
from run_e_layer_sweep import bootstrap_ci  # noqa: E402
from run_full_f_protocol import cv_correct  # noqa: E402
from run_gh import load_backend  # noqa: E402

OUT = Path("experiments/K_per_token_sweep")
H2_DIR = Path("experiments/H2_distractor_fix")
ANCHORS = ["Q", "question_end", "Options"] + [f"{k}_{L}" for L in "ABCDE" for k in ("open", "value_end")] + \
          ["A:", "think", "last"]
TASKS = {
    "H2_mult": {"items": lambda: h_arith.items("C2", 200), "prompts": lambda: h_arith.prompts("C2"),
                "gens": "experiments/H2_distractor_fix/generations.jsonl", "grouped": False, "contiguous": True},
    "I3_unit_conversions": {"items": lambda: i3_tasks.items("unit_conversions", 200),
                            "prompts": lambda: i3_tasks.prompts("unit_conversions"),
                            "gens": "experiments/I3_memorizable_tasks/unit_conversions/generations.jsonl",
                            "grouped": True, "contiguous": False},
}


def anchor_positions(b, pre, it):
    def end_of(text):
        return b.encode_raw(text).shape[1] - 1

    base = pre + "\n\nQ: "
    inp = it["input"]
    q = inp.split("\nOptions:")[0]
    pos = {"Q": end_of(pre + "\n\nQ"), "question_end": end_of(base + q), "Options": end_of(base + q + "\nOptions")}
    for L in "ABCDE":
        o = inp.index(f"({L})")
        pos[f"open_{L}"] = end_of(base + inp[:o + 2])
        nl = inp.find("\n", o)
        pos[f"value_end_{L}"] = end_of(base + (inp if nl == -1 else inp[:nl]))
    pos["A:"] = end_of(base + inp + "\nA")
    pos["think"] = end_of(base + inp + "\nA: Let's think")
    full = f_common.cot_prompt(pre, it)
    pos["last"] = end_of(full)
    return pos, full


def collect(task):
    cfg = TASKS[task]
    b = load_backend("qwen2.5-1.5b")
    pre, _ = cfg["prompts"]()
    its = cfg["items"]()
    anc, contig, toks, checks = [], [], None, []
    for i, it in enumerate(its):
        pos, full = anchor_positions(b, pre, it)
        ids = b.encode_raw(full)
        with torch.no_grad():
            hs = torch.stack([h[0] for h in b.model(ids, output_hidden_states=True).hidden_states[1:]])  # [L,T,D]
        idx = [pos[a] for a in ANCHORS]
        anc.append(hs[:, idx].float().cpu().numpy().astype(np.float16))
        if cfg["contiguous"]:
            contig.append(hs[:, pos["Q"]:].float().cpu().numpy().astype(np.float16))
            if toks is None:
                toks = [b.tokenizer.decode([t]) for t in ids[0, pos["Q"]:].tolist()]
        if i < 3:
            checks.append({a: b.tokenizer.decode([ids[0, pos[a]].item()]) for a in ANCHORS})
    b.shutdown()
    lens = {c.shape[1] for c in contig}
    return {"anchors": np.stack(anc), "contig": np.stack(contig) if contig and len(lens) == 1 else None,
            "contig_tokens": toks, "anchor_token_checks": checks, "items": its}


def labels(task, its):
    gens = [json.loads(line) for line in open(TASKS[task]["gens"])]
    assert len(gens) == len(its) and all(g["target"] == it["target"] for g, it in zip(gens, its, strict=True))
    return [g["no_cot"] for g in gens]


def probe_map(X, y, groups):
    from joblib import Parallel, delayed

    L, P = X.shape[1], X.shape[2]

    def one(lyr, p):
        c, _ = cv_correct(X[:, lyr, p].astype(np.float32), y, 0, groups)
        return lyr, p, float(c.mean()), bootstrap_ci(c)

    res = Parallel(n_jobs=-1)(delayed(one)(lyr, p) for lyr in range(L) for p in range(P))
    acc = np.zeros((L, P))
    ci = {}
    for lyr, p, a, c in res:
        acc[lyr, p] = a
        ci[f"{lyr},{p}"] = c
    return acc, ci


def _parse_paths(extra=None):
    """Input/output directories; defaults reproduce the committed layout (run from repo root)."""
    import argparse

    global OUT, H2_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="output directory (default: %(default)s)")
    ap.add_argument("--h2-dir", default=str(H2_DIR),
                    help="H2 run providing generations.jsonl (default: %(default)s)")
    if extra:
        extra(ap)
    a = ap.parse_args()
    OUT, H2_DIR = Path(a.out), Path(a.h2_dir)
    return a


def main():
    _parse_paths()
    TASKS["H2_mult"]["gens"] = str(H2_DIR / "generations.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for task, cfg in TASKS.items():
        t0 = time.time()
        d = collect(task)
        lab = labels(task, d["items"])
        keep = np.array([x is not None for x in lab])
        y = np.array([x for x in lab if x is not None])
        groups = None
        if cfg["grouped"]:
            groups = np.array([it["group"] for it, k in zip(d["items"], keep, strict=True) if k])
        t1 = time.time()
        acc, ci = probe_map(d["anchors"][keep], y, groups)
        res = {"n": int(keep.sum()), "majority": round(float(np.bincount(np.unique(y, return_inverse=True)[1]).max()
                                                          / len(y)), 3),
               "anchors": ANCHORS, "anchor_token_checks": d["anchor_token_checks"],
               "anchor_acc": np.round(acc, 3).tolist(), "anchor_ci95": ci,
               "collect_seconds": round(t1 - t0)}
        late = acc[18:]
        li, pi = np.unravel_index(late.argmax(), late.shape)
        res["best_late"] = {"layer": int(li + 18), "anchor": ANCHORS[pi], "acc": round(float(late.max()), 3)}
        if d["contig"] is not None:
            cacc, _ = probe_map(d["contig"][keep], y, groups)
            res["contig_tokens"] = d["contig_tokens"]
            res["contig_acc"] = np.round(cacc, 3).tolist()
            cl = cacc[18:]
            ci_, cp = np.unravel_index(cl.argmax(), cl.shape)
            res["contig_best_late"] = {"layer": int(ci_ + 18), "pos": int(cp), "token": d["contig_tokens"][cp],
                                       "from_end": int(cp - cacc.shape[1]), "acc": round(float(cl.max()), 3)}
        res["probe_seconds"] = round(time.time() - t1)
        out[task] = res
        print(task, res["best_late"], res.get("contig_best_late"), res["collect_seconds"], res["probe_seconds"],
              flush=True)
    # decision
    a, b = out["H2_mult"], out["I3_unit_conversions"]
    region = {x: ("options" if x.startswith(("open_", "value_end_")) else
                  "answer_span" if x in ("A:", "think", "last") else "question") for x in ANCHORS}
    ra, rb = region[a["best_late"]["anchor"]], region[b["best_late"]["anchor"]]
    acc_a, acc_b = np.array(a["anchor_acc"]), np.array(b["anchor_acc"])
    ia, ib = ANCHORS.index(a["best_late"]["anchor"]), ANCHORS.index(b["best_late"]["anchor"])
    la, lb = a["best_late"]["layer"], b["best_late"]["layer"]
    margin_a = acc_a[la, ia] - acc_a[la, ib]
    margin_b = acc_b[lb, ib] - acc_b[lb, ia]
    out["decision"] = {"H2_best_region": ra, "UC_best_region": rb, "regions_differ": ra != rb,
                       "H2_margin_over_UC_anchor": round(float(margin_a), 3),
                       "UC_margin_over_H2_anchor": round(float(margin_b), 3),
                       "task_dependent_confirmed": bool(ra != rb and margin_a >= 0.10 and margin_b >= 0.10)}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["decision"], indent=2))


if __name__ == "__main__":
    main()
