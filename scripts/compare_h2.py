#!/usr/bin/env python3
"""Compare a reproduction of H2 against the committed results.

    python scripts/compare_h2.py experiments/H2_distractor_fix results/h2_repro

Behavior and the options-only baseline should match exactly (greedy decoding, fixed seeds).
Probe accuracies can move slightly with hardware/library numerics; the tolerance below is the
seed-stability window reported in the post (seed-to-seed SD up to 0.046). Verdicts must match.
"""

import json
import sys
from pathlib import Path

TOL = 0.046
ref_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
ref = json.loads((ref_dir / "metrics.json").read_text())
new = json.loads((new_dir / "metrics.json").read_text())
rows, ok = [], True


def check(name, a, b, tol):
    global ok
    good = (a == b) if tol is None else abs(float(a) - float(b)) <= tol
    ok &= good
    rows.append(f"{'OK  ' if good else 'DIFF'} {name:45s} committed={a}  reproduced={b}")


for k in ("no_cot_acc", "with_cot_acc", "A_eq_B_rate"):
    if k in ref.get("behavior", {}):
        check(f"behavior.{k}", ref["behavior"][k], new["behavior"][k], 0.0)
for t in ("A", "B"):
    r, n = ref["targets"][t], new["targets"][t]
    check(f"target {t}: peak layer", r["peak_layer"], n["peak_layer"], 1)
    rl = r["by_layer"][str(r["peak_layer"])]
    nl = n["by_layer"][str(r["peak_layer"])]
    check(f"target {t}: acc at committed peak (seed 0)", rl["acc"], nl["acc"], TOL)
    check(f"target {t}: 5-seed mean at committed peak", rl["seed_mean"], nl["seed_mean"], TOL)
    check(f"target {t}: verdict", r["verdict"], n["verdict"], None)
pb_ref = ref_dir / "pattern_baseline.json"
pb_new = new_dir / "pattern_baseline.json"
if pb_ref.exists() and pb_new.exists():
    a, b = json.loads(pb_ref.read_text()), json.loads(pb_new.read_text())
    check("options-only best (H2 items)", a["H2_items"]["best"], b["H2_items"]["best"], 0.0)
    check("options-only best (H items, positive control)", a["H_items_positive_control"]["best"],
          b["H_items_positive_control"]["best"], 0.0)
print("\n".join(rows))
print("\nREPRODUCED" if ok else "\nMISMATCH (see DIFF rows)")
sys.exit(0 if ok else 1)
