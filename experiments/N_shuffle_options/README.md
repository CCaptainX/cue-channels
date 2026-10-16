# N: option-order reversal: does the pre-CoT signal track the correct option's position? Registered verdict: MIXED

Pre-registration: [`preregistration.md`](preregistration.md) (`ecb491d`). Numbers: [`metrics.json`](metrics.json); per-item measures:
[`per_item.jsonl`](per_item.jsonl); figures: `figures/peak_vs_correct_slot.png`, `figures/heatmaps_original_vs_reversed.png`.
H2's 200 items in original and reversed option order; L22; K's 16 anchors; cross-fitted (5-fold) probe probabilities per item.
Runtime ≈ 10 min. No-CoT accuracy: original 0.740 (committed H2 answers), reversed 0.685 (fresh greedy answers; "D" 40%).
Main analysis: **108 items** whose no-CoT answer is correct in both versions.

## Registered test
| Measure | R² of Δposition on Δcorrect-slot | slope | modal position (orig / rev) | Verdict |
|---|---|---|---|---|
| **Peak (primary)** | **0.293** | 0.44 | post-options 29% / slot E 38% | **mixed** (0.2 ≤ R² < 0.6) |
| Onset (secondary) | 0.000 | 0.00 | slot B / slot A | mixed by rule (R² < 0.2, no shared mode). Effectively uninformative: cross-fitted probabilities often pass 0.5 early by chance |

## What the data show
- **Structural component.** At the population level the two versions are nearly identical: accuracy builds along the option list and peaks at
  "A:" in both (original "(E" 0.71, "A:" 0.745; reversed 0.715, 0.795), wherever the correct option sits. Most items peak late (slot E or after).
- **Content component.** Per item, the peak moves part of the way with the correct option (slope 0.44 where perfect tracking is 1). The
  mean peak slot rises steadily with the correct slot in both versions (original: correct A → peak 2.1, B → 3.1, C → 3.7, D → 4.1, E → 4.2).
- **Post hoc (not pre-registered):** the peak lands at or after the item's own correct option on **93.5%** (original) and **91.7%** (reversed)
  of items, vs **84%** expected if peaks were unrelated to the correct slot (shuffled-label permutation, 5,000 draws; p = 0.0002 / 0.001).
  Only about 15% peak exactly at the correct slot.

## Interpretation
The answer letter is decodable from the residual stream **only after the correct option has been read**, a content-dependent gate, and the
signal then **keeps building through the remaining options and peaks at the structural answer position ("A:")**, a structural sink. That is
the "mixed" outcome: neither pure tracking (which would put the peak at the correct option) nor pure structure (which would ignore it).
Together with K, M and J: the model recognizes its answer among the options as it reads them, accumulates that toward the end of the prompt,
does not carry it into the arithmetic CoT (M), and does not need it to produce the CoT answer (J).
