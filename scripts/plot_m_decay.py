#!/usr/bin/env python3
"""Plots for experiment M."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = Path("experiments/M_mid_cot_decay")
m = json.loads((D / "metrics.json").read_text())
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
names = list(m["anchors"])
lab = {"last_prompt": "last prompt\ntoken", "first_cot": "1st CoT\ntoken", "first_eq": "1st '='",
       "pre_digit": "before\nproduct", "digit": "product\n1st digit", "post_digit": "product\nlast digit",
       "paren": "'answer is ('", "letter": "letter", "after_letter": "after\nletter"}
x = range(len(names))
SERIES = (("transfer_acc", "transfer_ci95", "#1f4e79", "transfer probe (trained on H2 last token) — primary"),
          ("per_position_cv_acc", "per_position_ci95", "#c55a11", "probe retrained at each anchor (CV) — secondary"))
for key, ci, col, name in SERIES:
    y = [m["anchors"][a][key] for a in names]
    lo = [m["anchors"][a][ci][0] for a in names]
    hi = [m["anchors"][a][ci][1] for a in names]
    ax.plot(x, y, "-o", color=col, ms=4, label=name)
    ax.fill_between(x, lo, hi, color=col, alpha=0.15)
ax.axhline(m["majority"], color="#555555", ls=":", label=f"majority ({m['majority']:.2f})")
ax.axhline(0.2, color="#999999", ls="--", linewidth=0.8, label="chance (0.20)")
ax.set_xticks(list(x), [lab[a] for a in names], fontsize=7.5)
ax.set_ylim(0, 1)
ax.set_ylabel("Target A accuracy at L22 (100 held-out items, 95% CI)")
ax.set_title("By CoT structural anchor", fontsize=10)
ax.legend(fontsize=7, loc="upper right")
ax = axes[1]
c = m["curve_transfer"]
ax.plot([p["t"] for p in c if p["n"] >= 50], [p["acc"] for p in c if p["n"] >= 50], color="#1f4e79", linewidth=1.2,
        label="transfer probe")
pc = m["curve_per_position_cv"]
ax.plot([p["t"] for p in pc], [p["acc"] for p in pc], color="#c55a11", linewidth=1.2, label="per-position CV probe")
for a in ("first_eq", "pre_digit", "paren"):
    ax.axvline(m["anchors"][a]["mean_token_index"], color="#888888", linewidth=0.7, ls="--")
    ax.text(m["anchors"][a]["mean_token_index"] + 0.5, 0.93, lab[a].replace("\n", " "), fontsize=7, rotation=90,
            va="top")
ax.axhline(m["majority"], color="#555555", ls=":")
ax.set_ylim(0, 1)
ax.set_xlabel("CoT token index (0 = last prompt token; items with ≥ 50 contributing)")
ax.set_title("By absolute token index (anchor lines = mean positions)", fontsize=10)
ax.legend(fontsize=7)
fig.suptitle("M: pre-CoT answer-letter information at L22 across the chain of thought", fontsize=10.5)
fig.tight_layout()
fig.savefig(D / "figures/decay_curve.png", dpi=300)
ex = json.loads((D / "example_annotated.json").read_text())
toks = ex["tokens"][:70]
fig, ax = plt.subplots(figsize=(12, 2.4))
ax.bar(range(len(toks)), [t["p_label"] for t in toks], color=["#1f4e79" if t["pred"] == ex["label"] else "#bbbbbb"
                                                            for t in toks])
ax.set_xticks(range(len(toks)), [t["token"].replace("\n", "↵") for t in toks], rotation=90, fontsize=6)
ax.set_ylim(0, 1)
ax.set_ylabel("transfer-probe P(label)")
ax.set_title(f"Example item {ex['idx']}: no-CoT answer {ex['label']}, with-CoT answer {ex['with_cot']} "
             f"(blue = probe's top prediction equals the label)", fontsize=9)
fig.tight_layout()
fig.savefig(D / "figures/example_annotated.png", dpi=300)
print("ok")
