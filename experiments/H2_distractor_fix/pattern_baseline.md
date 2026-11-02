# H2 step 0: pattern-only baseline: PASS

Pre-registration: [`preregistration.md`](preregistration.md) (`7c41311`). Script: `scripts/h2_pattern_baseline.py`. Raw numbers:
[`pattern_baseline.json`](pattern_baseline.json). 5,000 generated items per rule, **option values only** (no model, no question text),
5-fold CV, per-option scorers with argmax over the 5 options.

| Predict the correct letter from the 5 option values | H2 items (product distractors) | H items (±1/±2/±10), positive control |
|---|---|---|
| Logistic regression (per-option features) | 0.205 | 0.607 |
| Random forest (300 trees) | 0.202 | **1.000** |
| Rule: median value | 0.196 | 0.607 |
| Rule: centre of offsets | 0.206 | **1.000** |
| **Best** | **0.206** | **1.000** |

Chance = 0.20; gate ≤ 0.30. **PASS.** The detector is sensitive: it fully recovers H's leak and finds nothing in H2's items.
Letter balance (5,000 H2 items): A 991, B 967, C 1046, D 1009, E 987.

Tokenization check on the 200 real H2 prompts: the shifted probe position (question end − 5) is **"(E" on 200/200**, the same token as in H,
so the option-E comparison is like for like.
