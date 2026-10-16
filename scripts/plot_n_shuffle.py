#!/usr/bin/env python3
"""Plots for experiment N."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = Path("experiments/N_shuffle_options")
m = json.loads((D / "metrics.json").read_text())
rows = [json.loads(x) for x in open(D / "per_item.jsonl")]
main = [r for r in rows if r["main"]]
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "serif", "font.size": 9})
rng = np.random.default_rng(0)
SL = ["pre", "A", "B", "C", "D", "E", "post"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
for v, c, col in (("orig", "c_orig", "#1f4e79"), ("rev", "c_rev", "#c55a11")):
    x = np.array([r[c] for r in main]) + rng.uniform(-0.15, 0.15, len(main))
    y = np.array([r[f"peak_{v}"] for r in main]) + rng.uniform(-0.15, 0.15, len(main))
    ax.scatter(x, y, s=10, alpha=0.5, color=col, label="original" if v == "orig" else "reversed")
    means = [np.mean([r[f"peak_{v}"] for r in main if r[c] == k]) for k in range(5)]
    ax.plot(range(5), means, "-o", color=col, lw=1.6)
ax.plot([0, 4], [0, 4], "k--", lw=0.8, label="peak = correct slot")
ax.set_xticks(range(5), list("ABCDE"))
ax.set_yticks(range(-1, 6), SL)
ax.set_xlabel("slot of the correct option in that version")
ax.set_ylabel("per-item peak position (L22, cross-fitted)")
ax.set_title("Peak position vs. correct slot (lines = means)", fontsize=10)
ax.legend(fontsize=7.5)
ax = axes[1]
dc = np.array([r["c_rev"] - r["c_orig"] for r in main])
dp = np.array([r["peak_rev"] - r["peak_orig"] for r in main])
ax.scatter(dc + rng.uniform(-0.2, 0.2, len(dc)), dp + rng.uniform(-0.2, 0.2, len(dp)), s=10, alpha=0.5, color="#548235")
s, b = np.polyfit(dc, dp, 1)
xs = np.linspace(-4, 4, 10)
ax.plot(xs, s * xs + b, color="#548235", lw=1.6, label=f"fit: slope {s:.2f}, R² {m['peak']['R2_delta']:.2f}")
ax.plot(xs, xs, "k--", lw=0.8, label="perfect tracking (slope 1)")
ax.set_xlabel("Δ correct slot (reversed − original)")
ax.set_ylabel("Δ peak position")
ax.set_title(f"Registered test: R² = {m['peak']['R2_delta']:.2f} → '{m['peak']['verdict']}' (n = {m['n_main']})", fontsize=10)
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(D / "figures/peak_vs_correct_slot.png", dpi=300)
plt.close(fig)
labels = ["Q", "q end", "Opts", "(A", "A val", "(B", "B val", "(C", "C val", "(D", "D val", "(E", "E val", "A:", "think", "last"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, v in zip(axes, ("original", "reversed"), strict=True):
    H = np.array(m["versions"][v]["acc_by_layer_anchor"])
    im = ax.imshow(H, aspect="auto", origin="lower", cmap="viridis", vmin=0.2, vmax=0.85)
    ax.set_xticks(range(len(labels)), labels, fontsize=6.5, rotation=60)
    ax.set_title(f"{v} option order (no-CoT acc {m['versions'][v]['no_cot_acc']:.2f})", fontsize=10)
axes[0].set_ylabel("layer")
fig.colorbar(im, ax=axes, label="Target A probe accuracy (5-fold CV)")
fig.suptitle("N: population probe accuracy by anchor × layer — nearly identical under reversal", fontsize=10.5)
fig.savefig(D / "figures/heatmaps_original_vs_reversed.png", dpi=300, bbox_inches="tight")
print("ok")
