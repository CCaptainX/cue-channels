# I4 pilot: Qwen 2.5 7B Instruct on I's multi-step tasks: 0 of 4 pass; no full runs

Pre-registration: [`preregistration.md`](preregistration.md) (`a54d1a6`). Load check: [`load_check.json`](load_check.json) (`aaa65aa`; bf16 MPS,
15.5 GB, 28 layers, d_model 3584). Raw numbers: [`pilot.json`](pilot.json). Same items and prompts as I (`8a6de72` / `e094845`). G4 is
model-independent and carried over from I (`../I_generalization/pattern_baseline.json`). Pilot runtime 32 min.

| Task | G1 no-CoT [95% CI] (0.55–0.85) | G3 with-CoT (≥ 0.85) | G2 top letter (≤ 60%) | G4 options-only (≤ 0.30) | Result |
|---|---|---|---|---|---|
| add_carry | **0.333** [0.167, 0.500] | 1.000 | 33% | 0.208 | fail (G1) |
| mod | **0.233** [0.100, 0.400] | 0.967 | 33% | 0.204 | fail (G1) |
| gsm8k_easy | **0.433** [0.267, 0.600] | 0.967 | 33% | 0.235 | fail (G1) |
| aqua_easy | **0.400** [0.233, 0.567] | **0.567** | 50% | **0.354** | fail (G1, G3, G4) |

**Outcome: no task passed, so no I4 full runs.** Per the pre-registration, "I4 no" here means **untested**, not refuted: 7B does not reach
no-CoT competence on these multi-step tasks, so there is no pre-formed decision to probe.

Compared with 1.5B (I): 7B's constant-"D" default without CoT is gone (top letter 33% vs 50–97%), and with-CoT accuracy rises (mod 0.900 →
0.967, GSM8K 0.800 → 0.967), but no-CoT accuracy barely moves (carry-addition 0.333 → 0.333, mod 0.267 → 0.233, GSM8K 0.333 → 0.433).
Scale buys CoT competence on these tasks, not one-pass competence.

`figures/combined_layer_sweep_all.png` overlays every run that reached the probe stage: H2 and the two I3 tasks (no I4 task did).
