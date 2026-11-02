# I4: pre-registration — Qwen 2.5 7B Instruct on I's multi-step tasks (fires after I3 regardless of outcome)

**Registered:** 2026-10-15T12:58:17Z (UTC), before any I4 model run. Tasks, items, prompts, distractors and gates are exactly those of experiment I
(`experiments/I_generalization/preregistration.md`, `be4fee1`; generator `scripts/i_tasks.py` as committed in `0eb4fcb`, including the
disclosed GSM8K sampler fix): add_carry, mod, gsm8k_easy, aqua_easy. AQuA failed G4 (options-only 0.354) in I; G4 is model-independent,
so aqua_easy cannot pass and is piloted only for completeness.

## Model
`Qwen/Qwen2.5-7B-Instruct`, precision ladder as G1 (`f3ce9d4`): bf16 MPS → bf16 MPS with watermark ratio 0.0 → stop and report.
28 layers, d_model 3584 expected; recorded in `load_check.json`.

## Hypothesis
For every task passing the gates on 7B: **Target A** peaks (5-seed mean argmax, layers 0–27) in **[19, 25]** with seed-0 accuracy
**≥ 0.55** and H2's rule (A: > majority + 0.05, CI lower > majority, permutation p < 0.05). Target B reported with H2's rule.
If a task passes the gates (so 7B solves it partly without CoT) but Target A fails the hypothesis, that is evidence the late-layer
finding is about retrievability rather than pre-CoT decision formation in general.

## Phases
1. Load check (commit). 2. Pilot: 30 items per task, gates G1–G4 as I. 3. Full runs (N = 200, as H2; CV stratified by item, since I's
items are not repeated) on every passing task. Random-Gaussian baseline dimension = 3584.

## Joint I3 + I4 verdict scenarios
"I4 yes" iff at least one I4 task runs and every I4 task that runs meets the hypothesis; "I4 no" otherwise (including 0 passing).
1. I3 yes, I4 yes → generalizes across retrievable tasks and multi-step tasks with competence.
2. I3 yes, I4 no → specific to retrievable answers (multi-step answers form during CoT, or 7B lacks no-CoT competence).
3. I3 no, I4 yes → capability × scale rather than retrievability.
4. I3 no, I4 no → H2 is specific to 2-digit multiplication on 1.5B as far as these tests reach.
Where "no" comes from **no task passing the gates** rather than a failed probe, the report says so explicitly: that outcome means
"untested", not "refuted", and the scenario label is given with that qualifier.

## Limits
Combined I3 + I4 hard cap: 12 h wall time; at the cap, save partials and stop.
