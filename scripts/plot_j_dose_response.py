#!/usr/bin/env python3
"""Dose-response plot for experiment J."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = Path("experiments/J_causal_steering")
m = json.loads((D / "metrics.json").read_text())["conditions"]
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
alphas = [-2, -1, 0, 1, 2, 5, 10, 20]
for ax, mode, title in ((axes[0], "P", "P: last prompt token only (primary)"),
                        (axes[1], "S", "S: last prompt token + every generated token")):
    def key(a, name="v", mode=mode):
        return f"{name}|{a}|{'P' if a == 0 else mode}"
    xs = [a for a in alphas if key(a) in m]
    for letter, col in zip("ABCDE", ("#1f4e79", "#c55a11", "#548235", "#7030a0", "#7f7f7f"), strict=True):
        ys = [m[key(a)]["letter_dist"].get(letter, 0) / 100 for a in xs]
        ax.plot(xs, ys, "-o", color=col, ms=3, linewidth=1.6 if letter == "A" else 0.9,
                label=f"answer {letter}" + (" (steered toward)" if letter == "A" else ""))
    acc = [m[key(a)]["with_cot_accuracy"] for a in xs]
    ax.plot(xs, acc, "k--", linewidth=1, label="with-CoT accuracy")
    rx = [a for a in (2, 5, 20) if f"rand|{a}|{mode}" in m]
    ax.plot(rx, [m[f"rand|{a}|{mode}"]["rate_A"] for a in rx], "x", color="#c00000", ms=7,
            label="random direction: share A")
    ax.axvspan(-2.3, 5.3, color="#eeeeee", zorder=0, label="pre-registered α range")
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xticks(alphas, [str(a) for a in alphas])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("α (multiples of the layer-22 mean-difference vector, ‖v‖ = 3.7)")
    ax.set_title(title, fontsize=9.5)
axes[0].set_ylabel("share of 100 held-out items")
axes[1].legend(fontsize=7, loc="center right")
fig.suptitle("J: steering along the H2 probe direction does not move the with-CoT answer", fontsize=10.5)
fig.tight_layout()
fig.savefig(D / "figures/dose_response.png", dpi=300)
print("ok")
