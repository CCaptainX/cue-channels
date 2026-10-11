#!/usr/bin/env python3
"""Figure 4 for experiment F: no-CoT letter distribution per pilot task vs ground truth."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# usage: plot_f_pilot.py [experiment_dir] [model label]
F_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/F_layer_sweep")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen2.5-1.5B-Instruct"
pilot = json.loads((F_DIR / "pilot.json").read_text())
tasks = list(pilot["tasks"])
short = {"hyperbaton": "hyperbaton (2-way)",
         "tracking_shuffled_objects_three_objects": "tracking_shuffled_3 (3-way)",
         "date_understanding": "date_understanding (6-way)"}

fig, axes = plt.subplots(1, len(tasks), figsize=(11, 3.4), constrained_layout=True)
for ax, t in zip(axes, tasks, strict=True):
    r = pilot["tasks"][t]
    letters = "ABCDEF"[: r["n_options"]]
    model = [r["no_cot_letter_dist"].get(x, 0) for x in letters]
    truth = [r["truth_dist"].get(x, 0) for x in letters]
    xs = range(len(letters))
    ax.bar([x - 0.2 for x in xs], model, 0.4, label="model no-CoT answer", color="#3b6ea8")
    ax.bar([x + 0.2 for x in xs], truth, 0.4, label="ground truth", color="#bbbbbb")
    ax.axhline(0.6 * r["n"], color="#c0392b", ls=":", lw=1, label="60% bias limit (G2)")
    ax.set_xticks(list(xs), list(letters))
    sub = f"no-CoT acc {r['no_cot_acc']:.2f}, top {r['top_letter_share']:.0%}"
    ax.set_title(f"{short.get(t, t)}\n{sub}", fontsize=9)
    ax.set_ylim(0, r["n"])
    ax.set_ylabel("items (of 30)")
axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle(f"{F_DIR.name} pilot: no-CoT letter distribution ({MODEL}, BBH 3-shot)", fontsize=10)
out = F_DIR / "figures/4_no_cot_letter_distribution.png"
fig.savefig(out, dpi=300)
print(out)
