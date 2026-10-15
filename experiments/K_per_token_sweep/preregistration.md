# K: pre-registration — per-token × per-layer probe maps for H2 and I3 unit_conversions

**Registered:** 2026-10-15T23:22:30Z (UTC), before any K activations are collected. Items, prompts and Target A labels are those of the committed
runs: H2 (`experiments/H2_distractor_fix/generations.jsonl`, N = 200) and I3 unit_conversions
(`experiments/I3_memorizable_tasks/unit_conversions/generations.jsonl`, N = 200). One fresh forward pass per item, CoT prompt, no
generation; activations at block outputs 0–27.

## Positions
Prompts are 3-shot; only the **test-question region** is mapped. Because unit-conversion prompts vary in length, the primary map uses
**structural anchors**, located by tokenizing prompt prefixes: (1) "Q" of the test question, (2) last token of the question text,
(3) "Options" token, (4–13) for each letter L ∈ {A..E}: the "(L" token and the last token of option L's value, (14) the "A" of
"A: Let's", (15) "think", (16) the last prompt token ("."). H2 prompts have constant length, so H2 additionally gets a **full contiguous
map** of every token from the test "Q:" to the last token.

## Probe
H2 hyperparameters (StandardScaler + multinomial LR, C = 0.1), Target A, 5-fold CV, **seed 0 only** (cost); stratified for H2,
StratifiedGroupKFold by fact for unit_conversions (as I3). Bootstrap 95% CI per cell.

## Hypothesis
H2: the best cell over layers ≥ 18 lies at the last prompt token (anchor 16) or within the final "A: Let's think step by step." span.
unit_conversions: the best cell over layers ≥ 18 lies in the option region (anchors 4–13), at or adjacent to the "(E" anchor.

## Decision rule
**Task-dependent positional structure confirmed** iff (a) the two tasks' best anchors (layers ≥ 18) differ in region (option region vs
answer span), **and** (b) for each task, its best cell exceeds the same layer's accuracy at the *other* task's best anchor by ≥ 0.10.
The pre-registered by-eye criterion (heatmaps visibly different) is also reported. If (a) fails, the I3 option-E observation is called
not distinctive. Hard cap 3 h.
