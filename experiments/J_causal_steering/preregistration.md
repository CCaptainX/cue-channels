# J: pre-registration — CAA causal steering along the H2 layer-22 direction

**Registered:** 2026-10-15T23:22:30Z (UTC), before any steering run. Model, task generator (`h_arith` variant C2), prompts and parsing as H2
(`1ee5528` → `0b9dd69`). Reference: Rimsky et al. (2023) CAA; Cox et al. (2026) steering protocol and orthogonal control.

## Question
Is the layer-22 residual direction that H2's probe reads causally load-bearing on the model's eventual (with-CoT) answer?

## Steering vector
H2's probe is 5-class, so the steering direction is the **CAA mean-difference vector at block-22 output, last prompt token**, computed on
H2's 200 probe items (`experiments/H2_distractor_fix/full_activations.pkl`, Target A = no-CoT letter):
v = mean(x | no-CoT answer = A) − mean(x | no-CoT answer ≠ A). **α is a multiplier on v** (α = 1 adds exactly one mean difference).
Random-direction control: a fixed random direction (seed 0) orthogonal to v and rescaled to ‖v‖.

## Held-out items
Items 200–299 of the C2 stream (`h_arith.items("C2", 300)[200:]`), disjoint from the 200 used to compute v. N = 100.

## Interventions (hook on block 22's output, added in model dtype)
- **P (primary, as specified):** add α·v at the **last prompt token only** (the prompt forward pass); generated tokens untouched.
- **S (secondary, Cox et al. protocol):** add α·v at the last prompt token **and at every generated token**.
- α ∈ {−2, −1, 0, +1, +2, +5} for v, under both P and S. Random-direction control at α ∈ {+2, +5} under P and S.
- For every condition: greedy CoT generation (384 tokens max, stop at "\nQ:") → with-CoT answer; also the no-CoT answer (2 tokens,
  answer-only prompt) under the same intervention (secondary).

## Metrics
- **rate_A(α)** = share of the 100 items whose with-CoT answer is "A". **Δ(α) = rate_A(α) − rate_A(0)**, paired bootstrap 95% CI
  (2,000 resamples over items).
- flip-toward-A(α) = among items whose α=0 answer ≠ A, share answering A at α. flip-away(α<0) = among items with α=0 answer = A,
  share no longer A. Per-letter answer distribution at every α. Accuracy vs ground truth. Unparsed count.
- **Coherence:** 10 randomly chosen items (seed 0) at α = +2 and α = 0 (condition P) are read in full. An output is coherent if it
  states a multiplication and a final option in well-formed text. Also automatic: share of CoTs containing a correct "a × b = c" line.

## Decision rule (primary: condition P)
**Steering is causal** iff Δ(+2) > 0.15 **and** its paired-bootstrap CI excludes 0 **and** ≥ 8 of 10 inspected α = +2 outputs are
coherent. Dose-response (reported, not decisive): Δ monotone non-decreasing over α ∈ {−2, −1, 0, +1, +2, +5}; symmetry: flip-away at −2
vs flip-toward at +2. The random-direction Δ at the same α is reported as the generic-perturbation baseline; if it is within 0.05 of the
v result, the report says the effect is not direction-specific. Condition S is evaluated with the same rule and reported separately.

## Limits
Hard cap 5 h for J.
