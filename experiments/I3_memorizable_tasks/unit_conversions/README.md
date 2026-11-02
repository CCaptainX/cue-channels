# I3 / unit_conversions: full run (Qwen 2.5 1.5B Instruct)

Pre-registration: [`../preregistration.md`](../preregistration.md) (`02fb86d`); pilot `4a6498d`. N = 200, layers 0–27, 5 seeds,
**StratifiedGroupKFold by fact**, option-E token located directly. Full numbers: [`metrics.json`](metrics.json); figures in `figures/`.
Runtime: collection 671 s, analysis 635 s.

Behaviour: no-CoT **0.545** [0.475, 0.615], with-CoT **0.955** [0.925, 0.98]; no-CoT letters {'D': 77, 'B': 42, 'A': 21, 'C': 33, 'E': 27}.

| | Peak | Acc @ peak (seed 0) [95% CI] | 5-seed mean ± SD | Majority | Gaussian | Perm p | H2-rule verdict | Option-E token | Position-only |
|---|---|---|---|---|---|---|---|---|---|
| Target A | L22 | **0.465** [0.395, 0.535] | 0.421 ± 0.033 | 0.385 | 0.351 | 0.0099 | supported | 0.755 [0.695, 0.810] | 0.380 |
| Target B | L22 | **0.289** [0.228, 0.355] | 0.269 ± 0.024 | 0.218 | 0.239 | 0.1188 | not supported | 0.472 [0.401, 0.538] | 0.234 |

**I3 hypothesis: NOT MET.** At the last prompt token Target A peaks at L22 with 0.465 — above majority (H2 rule technically
supported, p = 0.0099) but below the pre-registered 0.55; Target B is not supported (0.289, p = 0.119).

**Signal present at a different position.** At the option-E token the same probes read **0.755** (A) and **0.472** (B), far above the
last token and far above chance, with the options-only baseline at 0.199 (no leak) and position-only at the majority rate. On this task the
answer is decodable while the options are being listed but is largely gone by the final prompt token. Per the pre-registration this is
"signal present, criteria not met" at the registered position. Behaviour: no-CoT 0.545 on the full 200 (pilot 0.600), with-CoT 0.955,
no-CoT letter D 38.5%.
