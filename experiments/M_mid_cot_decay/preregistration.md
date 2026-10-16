# M: pre-registration — mid-CoT probe decay on H2 items

**Registered:** 2026-10-16T05:37:07Z (UTC), before any M activations are computed. Model/task/prompts as H2 (`1ee5528`/`0b9dd69`). Items: the 100
held-out C2 items (200–299) used in J. **CoTs and labels are reused from J's α = 0 condition** (`experiments/J_causal_steering/generations.jsonl`,
condition `v|0|P`, commit `e9882de`): greedy with-CoT text and the no-CoT answer (Target A) for each item. One forward pass over
prompt + CoT with `output_hidden_states` gives the block-22 residual at every CoT position (the same computation as during generation).

## Probes
- **Primary, transfer probe (as specified):** the H2 Target-A probe at L22 (StandardScaler + multinomial LR, C = 0.1), trained on H2's 200
  last-prompt-token activations and labels, applied unchanged to held-out activations at each CoT position. Accuracy vs held-out Target A,
  bootstrap 95% CI over items.
- **Secondary, per-position probe:** at each aligned position, a probe trained and evaluated within the held-out set (5-fold stratified CV),
  which asks whether the information is present at all, in any direction.

## Positions
Index 0 = last prompt token; t = t-th generated token. Absolute-index curve uses items whose CoT reaches t. **Structural anchors per item:**
first CoT token; first "=" token; **pre-digit** = the token immediately before the first digit of the first occurrence of the correct
product as a standalone number following "= " (the final result); **digit** = that first digit; **post-digit** = the last digit of that
product; the "(" before the answer letter; the answer letter; the token after it. Items where the product string is not found are excluded
from anchor statistics (count reported).

## Hypothesis
Transfer-probe accuracy stays near its last-prompt-token level for the first CoT tokens, then drops to ≤ 0.50 at or around the result digit.

## Decision rules (transfer probe)
Let chance = majority rate of held-out Target A.
- **Overwrite confirmed** iff acc(pre-digit) ≥ 0.60 **and** acc(post-digit) ≤ acc(pre-digit) − 0.15 with non-overlapping 95% CIs.
- **Signal persists** iff accuracy ≥ 0.60 at every anchor and at every absolute index with ≥ 50 items.
- **Signal was always noise** iff accuracy is ≤ chance + 0.05 at all of absolute indices 1, 2 and 3.
- Otherwise **mixed**, described explicitly. The per-position probe is reported alongside (not decisive).
Hard cap 3 h.
