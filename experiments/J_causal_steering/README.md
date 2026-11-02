# J: CAA causal steering along the H2 layer-22 direction: NOT SUPPORTED

## At a glance

- **Name in the write-up:** The causal steering test (Section 5, Section 6 (strongest steering null), Figure 3).
- **Hypothesis:** Steering at layer 22 along the mean-difference "A" direction (α = +2) raises "A" answers by > 0.15. Not supported.
- **Pre-registration:** `d310a5b` (+ addendum `c4a9e51`) · **Results:** `00c70ac`
- **Outputs:** `metrics.json`, `generations.jsonl` in this directory.
- **Expected runtime:** about 3 h 10 min (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_j_steering.py --h2-dir results/h2_repro   # needs H2 activations: run scripts/reproduce_h2.sh first
python scripts/plot_j_dose_response.py
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Pre-registration: [`preregistration.md`](preregistration.md) (`d310a5b`), with a pre-run exploratory addendum (`c4a9e51`). Full numbers:
[`metrics.json`](metrics.json); every output: [`generations.jsonl`](generations.jsonl); plot: [`figures/dose_response.png`](figures/dose_response.png).
Qwen 2.5 1.5B Instruct; 100 held-out C2 items (200–299, disjoint from the 200 used for the vector); runtime ≈ 3 h 10 min.

**Steering vector:** CAA mean difference at block-22 output, last prompt token, no-CoT answer = A (18 items) vs ≠ A (182 items):
‖v‖ = 3.72 vs mean residual norm 115.6. α multiplies v. Random control: same norm, orthogonal to v.

| Condition | share answering A (with CoT) | Δ vs α = 0 [paired 95% CI] | with-CoT accuracy | no-CoT share A | unparsed |
|---|---|---|---|---|---|
| α = 0 | 0.230 | — | 0.990 | 0.090 | 0 |
| P, α = −2 / −1 / +1 | 0.220 / 0.230 / 0.230 | −0.010 / 0 / 0 | 0.98–0.99 | 0.08–0.09 | 0–2 |
| **P, α = +2 (primary)** | **0.230** | **+0.000 [0.000, 0.000]** | 0.990 | 0.100 | 0 |
| P, α = +5 | 0.230 | +0.000 | 0.990 | 0.100 | 0 |
| S, α = +2 / +5 | 0.230 / 0.230 | 0 / 0 | 0.99 / 1.00 | 0.10 / 0.10 | 0 |
| random, P / S, α = +2, +5 | 0.22–0.23 | −0.01 to 0 | 0.97–0.99 | 0.06–0.09 | 1–2 |
| *exploratory* P, α = +10 / +20 | 0.230 / 0.230 | 0 / 0 | 0.99 / 1.00 | 0.10 / 0.12 | 0 |
| *exploratory* S, α = +10 / **+20** | 0.230 / **0.200** | 0 / −0.030 [−0.08, 0.02] | 1.00 / **0.740** | 0.10 / 0.12 | 0 / 13 |
| *exploratory* random S, α = +20 | 0.180 | −0.050 [−0.10, −0.01] | 0.870 | 0.04 | 7 |

**Decision rule:** Δ(+2) = 0.000, not > 0.15, so **steering is not causal** on the registered grid. Coherence condition met: 10/10
randomly sampled α = +2 outputs are well-formed (they state the multiplication, compute it correctly and pick an option); automatic check:
the correct product appears in 99% of CoTs at α = +2. Dose-response: flat across α ∈ {−2 … +5} and through α = +10. Symmetry: flip-away at
α = −2 is 0.043 (1 of 23 A-answers), flip-toward at +2 is 0.

**The intervention is applied:** only 38/100 CoTs at P, α = +2 are word-for-word identical to α = 0 (14/100 under S), so the text
changes, but the answer does not. The model re-derives the product in its CoT and picks the matching option. At α = +20 on every token
(about 64% of the residual norm), arithmetic degrades ("160 + 60 = 220"), the model drifts off-task, and accuracy falls to 0.74, but answers
do not move toward A (0.200) and the random direction does the same or worse. That is generic disruption, not steering.

## Interpretation
On this task the pre-CoT answer direction is **readable but not load-bearing** for the CoT answer. The CoT is doing real work, recomputing
the product, which matches the task's behaviour (no-CoT 74%, with-CoT 98.5%). This is the opposite regime from the one where Cox et al.
(2026) found steering flipped answers (tasks where CoT changed few answers). Caveats: the vector comes from only 18 "A" items; it's a
mean-difference direction, not a trained probe weight; one layer; one task. No-CoT answers show at most a 0.03 drift toward A.
