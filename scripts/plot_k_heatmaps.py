#!/usr/bin/env python3
"""Heatmaps for experiment K from experiments/K_per_token_sweep/metrics.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = Path("experiments/K_per_token_sweep")
m = json.loads((D / "metrics.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 8})
labels = {"Q": "Q", "question_end": "question\nend", "Options": "Options", "A:": "A:", "think": "think",
          "last": "last\n(.)"}


def lab(a):
    if a.startswith("open_"):
        return f"({a[-1]}"
    if a.startswith("value_end_"):
        return f"{a[-1]} val"
    return labels[a]


for task, title in (("H2_mult", "H2: a × b (10–20)"), ("I3_unit_conversions", "I3: unit conversions")):
    r = m[task]
    acc = np.array(r["anchor_acc"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(acc, aspect="auto", origin="lower", cmap="viridis", vmin=0.2, vmax=0.85)
    ax.set_xticks(range(len(r["anchors"])), [lab(a) for a in r["anchors"]], fontsize=7)
    ax.set_yticks(range(0, 28, 3))
    ax.set_ylabel("layer")
    ax.set_xlabel("prompt anchor (test question region)")
    b = r["best_late"]
    ax.plot(r["anchors"].index(b["anchor"]), b["layer"], "r*", ms=10)
    for x in (2.5, 12.5):
        ax.axvline(x, color="white", lw=0.8, ls="--")
    ax.set_title(f"{title}: Target A probe accuracy by anchor × layer\n"
                 f"best (layers ≥ 18): {lab(b['anchor']).replace(chr(10), ' ')} @ L{b['layer']} = {b['acc']:.2f}; "
                 f"majority {r['majority']:.2f}", fontsize=9)
    fig.colorbar(im, ax=ax, label="5-fold CV accuracy")
    fig.tight_layout()
    fig.savefig(D / f"figures/heatmap_{task}.png", dpi=300)
    plt.close(fig)
    if "contig_acc" in r:
        c = np.array(r["contig_acc"])
        toks = [t.replace("\n", "↵") for t in r["contig_tokens"]]
        fig, ax = plt.subplots(figsize=(11, 4.2))
        im = ax.imshow(c, aspect="auto", origin="lower", cmap="viridis", vmin=0.2, vmax=0.85)
        ax.set_xticks(range(len(toks)), toks, fontsize=5.5, rotation=90)
        ax.set_yticks(range(0, 28, 3))
        ax.set_ylabel("layer")
        cb = r["contig_best_late"]
        ax.plot(cb["pos"], cb["layer"], "r*", ms=9)
        best = f"'{cb['token']}' ({cb['from_end']} from end) @ L{cb['layer']} = {cb['acc']:.2f}"
        ax.set_title(f"{title}: every token of the test question (constant length) × layer; best (layers ≥ 18): {best}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, label="5-fold CV accuracy")
        fig.tight_layout()
        fig.savefig(D / f"figures/heatmap_{task}_all_tokens.png", dpi=300)
        plt.close(fig)
print("ok")
