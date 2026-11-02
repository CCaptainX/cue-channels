# M: mid-CoT probe decay on H2 items

## At a glance

- **Name in the write-up:** The mid-CoT decay test (Section 5, Figure 4).
- **Hypothesis:** Registered rules on the last-token probe across CoT tokens (overwrite / persists / noise). Verdict: "signal was always noise".
- **Pre-registration:** `6cb35be` · **Results:** `da18d77`
- **Outputs:** `metrics.json` in this directory.
- **Expected runtime:** about 5 min (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_m_decay.py --h2-dir results/h2_repro   # needs H2 activations and J's generations (committed)
python scripts/plot_m_decay.py
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Pre-registration: [`preregistration.md`](preregistration.md) (`6cb35be`). Numbers: [`metrics.json`](metrics.json); per-item curves:
[`per_item_curves.jsonl`](per_item_curves.jsonl); figures: `figures/decay_curve.png`, `figures/example_annotated.png`. 100 held-out C2 items
(200–299) with J's α = 0 greedy CoTs (`00c70ac`); prompt tokenization matched the generation prefix on 100/100 items. Runtime ≈ 5 min.

## Registered verdict: "signal was always noise", but the label is misleading (see below)
The rule fires because the **transfer probe** (trained on H2's last-prompt-token activations, applied unchanged to every CoT position) is
at or below the majority rate (0.32) at CoT tokens 1, 2 and 3 (0.16, 0.28, 0.18). But at the position it was trained for, the last prompt
token, it reads **0.62** [0.52, 0.71] on these held-out items. What the rule actually detected is that **the learned direction does not
transfer to other token positions**, not that there was never a signal. Neither "overwrite" (needs ≥ 0.60 just before the product) nor
"persists" fires.

| CoT anchor (mean token index) | Transfer probe [95% CI] | Probe retrained at this anchor (5-fold CV) [95% CI] |
|---|---|---|
| last prompt token (0) | **0.62** [0.52, 0.71] | 0.58 [0.48, 0.68] |
| first CoT token (1) | 0.16 [0.09, 0.23] | 0.43 [0.33, 0.53] |
| first "=" (13.7) | 0.24 | 0.30 [0.21, 0.39] |
| just before the product (43.6) | 0.21 | **0.32** [0.23, 0.41] |
| product first digit (44.6) | 0.18 | 0.31 |
| product last digit (46.6) | 0.17 | **0.48** [0.37, 0.58] |
| "answer is (" (53.4) | 0.44 [0.35, 0.53] | **0.61** [0.51, 0.71] |
| answer letter / after (54–55) | 0.23 | 0.58–0.59 |
Majority of held-out Target A = 0.32; chance 0.20. Product found in 99/100 CoTs.

## What happens to the information
- **The pre-CoT letter information is not carried through the CoT.** Retrained probes fall from 0.58 at the last prompt token to the
  majority rate by the first "=" and stay there through the arithmetic, so the letter isn't linearly present while the product is being worked out.
- **It reappears after the product is written** (0.48 at the product's last digit) and peaks as the model is about to name the option
  (0.61 at "answer is ("). The letter is *re-derived* from the computed product, not retrieved from the pre-CoT state.
- This is not the registered "overwrite at the result digit" pattern. The pre-CoT signal is gone **from the first CoT token**, long before
  the digit, and the letter returns only after the computation. It fits J: the pre-CoT direction is readable but the CoT does not use it.
- Caveats: N = 100 (CIs about ±0.10); Target A is the no-CoT answer, while the late-CoT letter is the with-CoT answer, which matches Target A on
  about 74% of items, so late-CoT accuracy is bounded near that agreement; one layer (22).
