# H2: H with pattern-proof distractors. Scenario 1: H's result replicates

## At a glance

- **Name in the write-up:** The multiplication benchmark (headline result) (Sections 3–4, Figures 1–2).
- **Hypothesis:** With pattern-proof distractors, the model's answer is linearly decodable before the CoT (no-CoT target L22 0.705; with-CoT target L21 0.667).
- **Pre-registration:** `7c41311` · **Results:** `63dd6cb` (options-only baseline), `2ac5425` (results); exploratory: `8b0c762`, `013fdfe`
- **Outputs:** `metrics.json`, `pattern_baseline.json`, `exploratory/` in this directory.
- **Expected runtime:** about 25 min (collection 12, analysis 9) (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
bash scripts/reproduce_h2.sh results/h2_repro   # baseline + full protocol + comparison; writes only to results/
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Pre-registration [`preregistration.md`](preregistration.md) (`7c41311`) → pattern baseline [`pattern_baseline.md`](pattern_baseline.md)
(`63dd6cb`) → this results commit. (Naming note: H's own full-run commit `533dbff` was titled "H2 results"; this directory is the
new distractor-fix experiment.) Full numbers: [`metrics.json`](metrics.json); pilot: [`pilot.json`](pilot.json); generations:
[`generations.jsonl`](generations.jsonl). Qwen2.5-1.5B-Instruct, a × b (10–20), 5 options whose 4 distractors are **other products**.
Runtime: pilot 2 min, collection 12 min, analysis 8.5 min.

## Gates
- **Pattern-only baseline:** best option-values-only scorer 0.206 (chance 0.20, gate ≤ 0.30). **PASS.** The same detector scores 1.000 on H's items.
- **Pilot (30):** no-CoT 0.700 [0.533, 0.867], with-CoT 0.900, top letter 27%. **PASS.**

## Behavior (N = 200)
no-CoT **0.740** [0.680, 0.800] · with-CoT **0.985** [0.965, 1.000] (2 unparsed) · no-CoT letters D 56, E 53, B 45, C 28, A 18 (max 28%).

## Pre-registered verdicts

| | Target A (no-CoT answer) | Target B (with-CoT answer) |
|---|---|---|
| Peak layer | **L22** | **L21** |
| Acc @ peak, seed 0 [95% CI] | **0.705** [0.645, 0.765] | **0.667** [0.596, 0.727] |
| 5-seed mean ± SD | 0.730 ± 0.019 | 0.674 ± 0.013 |
| Rule floor | majority 0.280 | chance 0.200 (majority 0.247) |
| Random-Gaussian probe | 0.246 ± 0.021 | 0.236 ± 0.038 |
| Max-over-layers permutation | p = 0.0099 (null p95 0.335) | p = 0.0099 (null p95 0.298) |
| Layer 24 | 0.680 | 0.641 |
| **Rule** | **SUPPORTED** | **SUPPORTED** (also vs majority) |
| Positional bias ruled out (formal rule) | No: shifted token not beaten by 0.10 | No: same |

Seed SD across all (layer, target) cells: 0.005–0.046.

## The option-E comparison

| Peak layer | H (leaky distractors) | **H2 (pattern-proof)** |
|---|---|---|
| Target A: last token / "(E" token | 0.695 / 0.730 | **0.705 / 0.710** |
| Target B: last token / "(E" token | 0.665 / **0.835** | **0.667 / 0.702** |
| Position-only (A / B) | 0.385 / 0.230 (= majority) | 0.280 / 0.242 (≈ majority) |
| Mean-pooled all tokens (A / B) | 0.455 / 0.575 | 0.445 / 0.621 |

**H's E-token advantage on Target B (+0.17) vanished (+0.035, CIs overlap).** It was the model reading the distractor
pattern while listing options. The **last-token result did not move** (0.665 → 0.667 for B; 0.695 → 0.705 for A). Under the
pre-registered definition, "E-token better" holds for neither target.

The formal positional rule still fails only because it requires the last token to beat the shifted token by ≥ 0.10. With the leak
gone, both positions carry about the same information, and position alone carries none (≈ majority). That's the "spread across the
late question" reading anticipated in F's pre-registration, not a positional confound.

## Scenario: 1 (replicates)
Both targets are supported at the last prompt token, and the E-token is not better for either. Layer profile as in H: baseline through
about L12, a gradual rise to L19, then a **jump at L20–L21** (A 0.49 → 0.72, B 0.41 → 0.67), with a plateau to L27 (`figures/1_layer_sweep.png`).

## Interpretation and limits
- On a task inside the model's no-CoT competence (74%), the answer the model will give, with or without CoT, is linearly
  readable from the residual stream **before any CoT token, from layer ~20**, at roughly its no-CoT accuracy (capability bound).
- Target B ≈ the correct letter here (CoT 98.5% correct), so "decodes the with-CoT answer" and "decodes the correct answer" can't be
  separated on this task; they'd need a task where CoT changes answers.
- Wide product distractors permit **magnitude estimation**. That's a real (approximate) computation, not an option-pattern leak, but it
  means "pre-computes the product" should be stated as "has enough pre-CoT information to pick the product".
- One model (1.5B), one task, N = 200, a single prompt template, correlational (no causal intervention).
