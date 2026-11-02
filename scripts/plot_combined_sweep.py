#!/usr/bin/env python3
"""Overlay last-prompt-token layer sweeps from several runs (H2 + I3 + I4).

usage: plot_combined_sweep.py OUT.png "label=path/to/metrics.json" ...
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "legend.fontsize": 7.5})
COLORS = ["#1f4e79", "#c55a11", "#548235", "#7030a0", "#bf9000", "#2e75b6", "#7f7f7f"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]


def main(out, specs):
    runs = []
    for s in specs:
        label, path = s.split("=", 1)
        runs.append((label, json.load(open(path))))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, tgt, title in zip(axes, "AB", ("Target A: no-CoT answer", "Target B: with-CoT answer"), strict=True):
        ax.axvspan(19, 25, color="#dddddd", alpha=0.5, linewidth=0, label="pre-registered peak window (L19–25)")
        ax.axhline(0.55, color="#999999", ls=":", linewidth=1, label="pre-registered bar (0.55)")
        ax.axhline(0.20, color="#444444", ls=":", linewidth=0.8, label="chance (0.20)")
        for k, (label, m) in enumerate(runs):
            t = m["targets"][tgt]
            n = m["n_layers"]
            y = np.array([t["by_layer"][str(i)]["seed_mean"] for i in range(n)])
            pk = t["peak_layer"]
            ax.plot(range(n), y, color=COLORS[k % 7], ls=STYLES[k % 7], linewidth=1.6,
                    label=f"{label} (peak L{pk}: {t['peak']['acc']:.2f})")
            ax.plot(pk, t["by_layer"][str(pk)]["seed_mean"], "o", color=COLORS[k % 7], ms=4)
        ax.set_title(title)
        ax.set_xlabel("layer (28 blocks)")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    axes[0].set_ylabel("last-prompt-token probe accuracy\n(5-fold CV, 5-seed mean)")
    fig.suptitle("Pre-CoT probe accuracy by layer across tasks and models", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
