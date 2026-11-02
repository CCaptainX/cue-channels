# I3 / times_tables: full run (Qwen 2.5 1.5B Instruct)

Pre-registration: [`../preregistration.md`](../preregistration.md) (`5c3e806`); pilot `b91b174`. N = 200, layers 0–27, 5 seeds,
**StratifiedGroupKFold by fact**, option-E token located directly. Full numbers: [`metrics.json`](metrics.json); figures in `figures/`.
Runtime: collection 382 s, analysis 257 s.

Behaviour: no-CoT **0.855** [0.805, 0.905], with-CoT **1.000** [1.0, 1.0]; no-CoT letters {'D': 46, 'B': 45, 'E': 36, 'C': 39, 'A': 34}.

| | Peak | Acc @ peak (seed 0) [95% CI] | 5-seed mean ± SD | Majority | Gaussian | Perm p | H2-rule verdict | Option-E token | Position-only |
|---|---|---|---|---|---|---|---|---|---|
| Target A | L21 | **0.780** [0.720, 0.835] | 0.792 ± 0.013 | 0.230 | 0.213 | 0.0099 | supported | 0.760 [0.700, 0.820] | 0.240 |
| Target B | L21 | **0.735** [0.670, 0.795] | 0.754 ± 0.017 | 0.265 | 0.237 | 0.0099 | supported | 0.850 [0.795, 0.900] | 0.275 |

**I3 hypothesis (Target A peak in [19, 25], ≥ 0.55, H2 rule): MET** — L21, 0.780.

- Profile matches H2: near baseline in early/middle layers, a jump around L19–21, plateau to L27 (`figures/1_layer_sweep.png`).
- Positional rule (pre-registered) fails as in H2: the option-E token is not beaten by 0.10. On Target B the option-E token is *higher*
  (0.850 vs 0.735). The options-only baseline is at chance (0.207), so this is not a distractor leak; with CoT the model is 100% correct,
  so Target B = the correct letter, and the model appears to register which listed option matches while reading the options. Reported as an
  observation, not a confound.
