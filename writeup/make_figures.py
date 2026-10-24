#!/usr/bin/env python3
"""Figures for the LW draft, built from the committed metrics files.

Every figure is laid out for a single column (~6.25 in wide in the Doc): two-panel figures are
stacked vertically rather than side by side, so text is not shrunk when the Doc scales them.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import h_arith  # noqa: E402

OUT = ROOT / "writeup/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 2.0,
})
LW = 2.0
BAND = 0.25
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#1f4e79", "#c55a11", "#548235", "#7030a0", "#7f7f7f"
EXP = ROOT / "experiments"


def save(fig, name):
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT / name)
    plt.close(fig)


H2 = json.loads((EXP / "H2_distractor_fix/metrics.json").read_text())
PB = json.loads((EXP / "H2_distractor_fix/pattern_baseline.json").read_text())

# ---- fig 1: layer sweep on the multiplication benchmark
fig, ax = plt.subplots(figsize=(10, 6))
xs = np.arange(28)
for key, color, ls, label in (("A", BLUE, "-", "no-CoT target"),
                              ("B", ORANGE, "--", "with-CoT target")):
    t = H2["targets"][key]
    # line = 5-seed mean (what the peak layer was selected on); band = seed-0 bootstrap 95% CI
    acc = np.array([t["by_layer"][str(i)]["seed_mean"] for i in xs])
    lo = np.array([t["by_layer"][str(i)]["ci95"][0] for i in xs])
    hi = np.array([t["by_layer"][str(i)]["ci95"][1] for i in xs])
    ax.plot(xs, acc, ls, color=color, lw=LW, label=label)
    ax.fill_between(xs, lo, hi, color=color, alpha=BAND, lw=0)
    pk = t["peak_layer"]
    pm, p0 = t["by_layer"][str(pk)]["seed_mean"], t["by_layer"][str(pk)]["acc"]
    ax.plot(pk, pm, "o", color=color, ms=8)
    xy_text = (7.5, 0.86) if key == "A" else (19.5, 0.40)
    ax.annotate(f"peak L{pk}: {p0:.3f} (seed 0)\n5-seed mean {pm:.3f}", (pk, pm),
                xytext=xy_text, color=color, fontsize=13,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.2))
ax.axhline(0.20, color=GRAY, ls=":", lw=LW, label="chance (0.20)")
ax.axhline(H2["targets"]["A"]["majority"], color=BLUE, ls="-.", lw=1.5, alpha=0.7,
           label=f"no-CoT-target majority baseline ({H2['targets']['A']['majority']:.2f})")
ax.set_xlim(-0.5, 27.5)
ax.set_ylim(0, 1)
ax.set_xlabel("layer (Qwen 2.5 1.5B Instruct, 28 blocks)")
ax.set_ylabel("probe accuracy (5-fold CV)")
ax.set_title("Pre-CoT probe accuracy by layer, multiplication benchmark (N = 200)")
ax.legend(loc="lower right", frameon=True, framealpha=0.95)
save(fig, "fig1_layer_sweep.png")

# ---- fig 2: options-only baseline
fig, ax = plt.subplots(figsize=(10, 6))
vals = [PB["H_items_positive_control"]["best"], PB["H2_items"]["best"]]
labels = ["original distractors\n(correct ± 1, 2, 10)", "corrected distractors\n(other a×b products)"]
bars = ax.bar(labels, vals, color=[GRAY, BLUE], width=0.5)
for b, v in zip(bars, vals, strict=True):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=15)
ax.axhline(0.20, color=ORANGE, ls=":", lw=LW, label="chance (0.20)")
ax.set_ylim(0, 1.12)
ax.set_ylabel("accuracy picking the correct letter")
ax.set_title("Options-only classifier (no model, no question):\ncan the option values alone identify the answer?")
ax.legend(loc="upper right")
save(fig, "fig2_pattern_baseline.png")

# ---- fig 3: causal steering test, dose-response (panels stacked)
J = json.loads((EXP / "J_causal_steering/metrics.json").read_text())["conditions"]
alphas = [-2, -1, 0, 1, 2, 5, 10, 20]
fig, axes = plt.subplots(2, 1, figsize=(10, 11), sharex=True)
for ax, mode, title in ((axes[0], "P", "Steering at the last prompt token only (primary)"),
                        (axes[1], "S", "Steering at the last prompt token and every generated token")):
    def key(a, name="v", mode=mode):
        return f"{name}|{a}|{'P' if a == 0 else mode}"
    x = [a for a in alphas if key(a) in J]
    for letter, col in zip("ABCDE", (BLUE, ORANGE, GREEN, PURPLE, GRAY), strict=True):
        ys = [J[key(a)]["letter_dist"].get(letter, 0) / 100 for a in x]
        ax.plot(x, ys, "-o", color=col, ms=6, lw=3.0 if letter == "A" else 1.5,
                label=f"answer {letter}" + (" (steered toward)" if letter == "A" else ""))
    ax.plot(x, [J[key(a)]["with_cot_accuracy"] for a in x], "k--", lw=LW, label="with-CoT accuracy")
    rx = [a for a in (2, 5, 20) if f"rand|{a}|{mode}" in J]
    ax.plot(rx, [J[f"rand|{a}|{mode}"]["rate_A"] for a in rx], "x", color="#c00000", ms=11, mew=2.5,
            label="random direction: share A")
    ax.axvspan(-2.3, 5.3, color="#eeeeee", zorder=0, label="pre-registered α range")
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xticks(alphas, [str(a) for a in alphas])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of 100 held-out items")
    ax.set_title(title)
axes[1].set_xlabel("α (multiples of the layer-22 steering vector, ‖v‖ = 3.7)")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.11), frameon=False)
fig.suptitle("The causal steering test: the with-CoT answer does not move")
save(fig, "fig3_J_dose_response.png")

# ---- fig 4: mid-CoT decay test (panels stacked)
M = json.loads((EXP / "M_mid_cot_decay/metrics.json").read_text())
names = list(M["anchors"])
lab = {"last_prompt": "last\nprompt", "first_cot": "1st CoT\ntoken", "first_eq": "1st\n'='",
       "pre_digit": "before\nproduct", "digit": "product\n1st digit", "post_digit": "product\nlast digit",
       "paren": "'answer\nis ('", "letter": "letter", "after_letter": "after\nletter"}
fig, axes = plt.subplots(2, 1, figsize=(10, 11))
ax = axes[0]
x = np.arange(len(names))
for key, ci, col, name in (("transfer_acc", "transfer_ci95", BLUE, "registered: last-token probe applied at each anchor"),
                           ("per_position_cv_acc", "per_position_ci95", ORANGE, "post hoc: probe retrained at each anchor")):
    y = [M["anchors"][a][key] for a in names]
    lo = [M["anchors"][a][ci][0] for a in names]
    hi = [M["anchors"][a][ci][1] for a in names]
    ax.plot(x, y, "-o" if col == BLUE else "--s", color=col, ms=7, lw=LW, label=name)
    ax.fill_between(x, lo, hi, color=col, alpha=BAND)
ax.axhline(M["majority"], color="#555555", ls=":", lw=LW, label=f"majority ({M['majority']:.2f})")
ax.axhline(0.2, color="#999999", ls="--", lw=1.5, label="chance (0.20)")
ax.set_xticks(x, [lab[a] for a in names])
ax.set_ylim(0, 1)
ax.set_ylabel("no-CoT answer accuracy\n(layer 22, 95% CI)")
ax.set_title("By structural anchor in the chain of thought")
ax.legend(loc="upper right", fontsize=11.5)
ax = axes[1]
c = M["curve_transfer"]
ax.plot([p["t"] for p in c if p["n"] >= 50], [p["acc"] for p in c if p["n"] >= 50], color=BLUE, lw=LW,
        label="registered: last-token probe")
pc = M["curve_per_position_cv"]
ax.plot([p["t"] for p in pc], [p["acc"] for p in pc], color=ORANGE, lw=LW, ls="--", label="post hoc: retrained probe")
for a in ("first_eq", "pre_digit", "paren"):
    t = M["anchors"][a]["mean_token_index"]
    ax.axvline(t, color="#888888", lw=1.2, ls="--")
    ax.text(t + 0.6, 0.97, lab[a].replace("\n", " "), fontsize=12, rotation=90, va="top")
ax.axhline(M["majority"], color="#555555", ls=":", lw=LW)
ax.set_ylim(0, 1)
ax.set_xlabel("CoT token index (0 = last prompt token)")
ax.set_ylabel("no-CoT answer accuracy\n(layer 22)")
ax.set_title("By token index (dashed lines = mean anchor positions)")
ax.legend(loc="upper right")
fig.suptitle("The mid-CoT decay test: the pre-CoT answer drops out during the CoT")
save(fig, "fig4_M_decay.png")

# ---- fig 5: option-reversal test, peak position vs correct slot (panels stacked)
N = json.loads((EXP / "N_shuffle_options/metrics.json").read_text())
rows = [json.loads(x) for x in open(EXP / "N_shuffle_options/per_item.jsonl")]
main = [r for r in rows if r["main"]]
rng = np.random.default_rng(0)
SL = ["pre", "A", "B", "C", "D", "E", "post"]
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
ax = axes[0]
for v, c, col in (("orig", "c_orig", BLUE), ("rev", "c_rev", ORANGE)):
    xj = np.array([r[c] for r in main]) + rng.uniform(-0.15, 0.15, len(main))
    yj = np.array([r[f"peak_{v}"] for r in main]) + rng.uniform(-0.15, 0.15, len(main))
    ax.scatter(xj, yj, s=22, alpha=0.3, color=col, label="original order" if v == "orig" else "reversed order")
    means = [np.mean([r[f"peak_{v}"] for r in main if r[c] == k]) for k in range(5)]
    ax.plot(range(5), means, "-o", color=col, lw=2.5, ms=8)
ax.plot([0, 4], [0, 4], "k--", lw=1.5, label="peak = correct slot")
ax.set_xticks(range(5), list("ABCDE"))
ax.set_yticks(range(-1, 6), SL)
ax.set_xlabel("slot of the correct option")
ax.set_ylabel("per-item peak position\n(layer 22, cross-fitted)")
ax.set_title("Post hoc: peak position vs. correct slot (lines = means)")
ax.legend(loc="upper left")
ax = axes[1]
dc = np.array([r["c_rev"] - r["c_orig"] for r in main])
dp = np.array([r["peak_rev"] - r["peak_orig"] for r in main])
ax.scatter(dc + rng.uniform(-0.2, 0.2, len(dc)), dp + rng.uniform(-0.2, 0.2, len(dp)), s=22, alpha=0.3, color=GREEN)
s, b = np.polyfit(dc, dp, 1)
xl = np.linspace(-4, 4, 10)
ax.plot(xl, s * xl + b, color=GREEN, lw=2.5, label=f"fit: slope {s:.2f}, R² {N['peak']['R2_delta']:.2f}")
ax.plot(xl, xl, "k--", lw=1.5, label="perfect tracking (slope 1)")
ax.set_xlabel("change in correct slot (reversed − original)")
ax.set_ylabel("change in peak position")
ax.set_title(f"Registered test: R² = {N['peak']['R2_delta']:.2f} → '{N['peak']['verdict']}' (n = {N['n_main']})")
ax.legend(loc="upper left")
fig.suptitle("The option-reversal test: does the peak follow the correct option?")
save(fig, "fig5_N_peak_vs_slot.png")

# ---- fig 6: option-reversal test, anchor x layer heatmaps (panels stacked)
labels = ["Q", "q end", "Opts", "(A", "A val", "(B", "B val", "(C", "C val", "(D", "D val", "(E", "E val",
          "A:", "think", "last"]
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
for ax, v in zip(axes, ("original", "reversed"), strict=True):
    Hm = np.array(N["versions"][v]["acc_by_layer_anchor"])
    im = ax.imshow(Hm, aspect="auto", origin="lower", cmap="viridis", vmin=0.2, vmax=0.85)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_ylabel("layer")
    ax.grid(False)
    ax.set_title(f"{v} option order (no-CoT accuracy {N['versions'][v]['no_cot_acc']:.2f})")
fig.suptitle("The option-reversal test (post hoc): probe accuracy by position × layer")
fig.tight_layout(pad=1.5)
fig.colorbar(im, ax=axes, label="no-CoT answer probe accuracy (5-fold CV)", shrink=0.8, pad=0.03)
fig.savefig(OUT / "fig6_N_heatmaps.png")
plt.close(fig)


# ---- fig 7 (appendix): distractor examples, real items from each generator
def first_item(variant, question):
    for it in h_arith.items(variant, 200):
        if it["question"] == question:
            return it
    raise LookupError(question)


ex = {"H": first_item("C", "What is 11 * 15?"), "H2": first_item("C2", "What is 11 * 15?")}
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
for ax, (name, it) in zip(axes, ex.items(), strict=True):
    opts = re.findall(r"^\(([A-E])\) (\d+)", it["input"], re.M)
    ax.axis("off")
    title = ("H: correct ± {1, 2, 10}\n(answer identifiable\nfrom options alone)" if name == "H"
             else "H2: other a×b products\nfrom [10, 20]²\n(no structural tell)")
    ax.set_title(title, fontsize=15)
    ax.text(0.05, 0.86, "What is 11 × 15?", fontsize=17, transform=ax.transAxes, weight="bold")
    for k, (ltr, v) in enumerate(opts):
        correct = ltr == it["target"]
        diff = "" if correct else f"   ({int(v) - 165:+d})" if name == "H" else ""
        ax.text(0.10, 0.70 - k * 0.14, f"({ltr}) {v}{diff}", fontsize=16, transform=ax.transAxes,
                color=BLUE if correct else "#222222", weight="bold" if correct else "normal")
        if correct:
            ax.text(0.62, 0.70 - k * 0.14, "← correct", fontsize=14, color=BLUE, transform=ax.transAxes)
save(fig, "fig4_distractor_examples.png")
print(sorted(p.name for p in OUT.glob("*.png")))
