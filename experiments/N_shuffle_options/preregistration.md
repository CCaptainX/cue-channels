# N: pre-registration — does the pre-CoT signal track the correct option's position? (option-order reversal)

**Registered:** 2026-10-16T05:37:07Z (UTC), before any N activations or generations. Items: H2's 200 C2 items. **Original** = the committed H2 prompts;
**reversed** = the same items with the five option values in reverse order, relettered (A)–(E) (correct option index c → 4 − c). Same
exemplars and prompt layout. Target A: original = H2's committed no-CoT answers; reversed = fresh greedy no-CoT answers on reversed
prompts (2 tokens). Activations at block 22 at K's 16 anchors (`9256836`) for both versions.

## Per-item position measures (per version, per item)
At each anchor, a probe (H2 hyperparameters) is fit with 5-fold stratified CV; each item gets its cross-fitted predicted probability for
its own Target-A letter. Option anchors are indexed by option slot (0 = A … 4 = E; "(L" and "L value end" both map to slot L).
- **Peak (primary, as specified):** the anchor with the maximal cross-fitted probability; mapped to an option slot if it is an option
  anchor, otherwise to 5 (post-options: "A:", "think", last).
- **Onset (secondary):** the first anchor, in prompt order, whose probability ≥ 0.5; mapped the same way (5 if only after the options; 6 if never).
Items whose Target A ≠ the correct option in either version are analysed separately (main analysis: items whose no-CoT answer equals the
correct option in both versions, so "position of the correct option" = "position of the decoded letter").

## Hypothesis
The peak position in the reversed version tracks the correct option's new position: regressing Δpeak = peak_rev − peak_orig on
Δc = c_rev − c_orig across items gives **R² ≥ 0.6** (positive slope).

## Decision rules (applied to peak; onset reported with the same rules)
- **Tracks the correct answer:** R² ≥ 0.6 with positive slope.
- **Structural:** R² < 0.2 and the modal peak position is the same anchor in both versions (e.g. always "A:" or always "(E").
- **Mixed:** 0.2 ≤ R² < 0.6 (or R² < 0.2 without a shared modal position).
Population-level heatmaps (accuracy by anchor × layer) for both versions are also produced. Hard cap 2 h.
