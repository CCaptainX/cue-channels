# I: pre-registration — does H2's late-layer pre-CoT signal generalize across tasks?

**Registered:** 2026-10-15T05:51:17Z (UTC). Committed before any I data is downloaded or generated and before any model run.
Methodology ancestors: H (`c1f8159` pre-reg → `8cb14c3` results) and H2 (`1ee5528` → `db4fad4` → `0b9dd69`).
Everything not stated here is identical to H2: model (`Qwen/Qwen2.5-1.5B-Instruct`, bf16, MPS, greedy), raw-text 3-shot
prompt layout and parsing, stop at `\nQ:`, probe (StandardScaler + multinomial LR, C = 0.1), stratified 5-fold CV, 5 seeds,
2,000-resample bootstrap CI, majority / chance / random-Gaussian baselines, 100-permutation max-over-layers test, position-only probe.

## Research question
Does H2's result — the model's answer linearly decodable before CoT from about layer 20 at about its no-CoT accuracy — hold on
other tasks the model is competent at, or is it specific to two-digit multiplication?

## Hypothesis (per task that runs)
**Generalizes** iff for **both** Target A (no-CoT answer) and Target B (with-CoT answer): (i) the peak layer (argmax of the 5-seed
mean over layers 0–27) is in **[19, 25]**; (ii) seed-0 accuracy at the peak is **≥ 0.55**; (iii) H2's decision rule holds (A: acc >
majority + 0.05 and CI lower bound > majority; B: acc > chance + 0.10 and CI lower bound > chance) and the max-over-layers
permutation p < 0.05.

## Candidate tasks (all 5 options (A)–(E), chance 0.20)
Generated tasks use `random.Random(seed)` for items and `random.Random(seed + 1000)` for 3 exemplars (correct letters forced to B, A, C;
exemplar questions excluded from the item stream). Option order for generated tasks is a uniform shuffle.

1. **add_carry** (seed 301): "What is a + b?", a, b ~ U{10..99} conditioned on (a mod 10) + (b mod 10) ≥ 10. **Distractors:** 4 distinct
   values a′ + b′ with (a′, b′) drawn by the same rule (carry condition included), ≠ correct. Worked step: "a + b = s."
2. **mod** (seed 302): "What is (a + b) mod m?", a, b ~ U{10..99}, m ~ U{7, 11, 13}. **Distractors:** 4 distinct residues drawn uniformly
   from {0..m−1} \ {correct}. Worked steps: "a + b = s. s = q × m + r, so (a + b) mod m = r."
3. **gsm8k_easy**: GSM8K train (`github.com/openai/grade-school-math`, `grade_school_math/data/train.jsonl`), filtered to items whose final
   answer (after "####") is an integer in [10, 99] and whose question has ≤ 2 sentences (count of `[.?!]` followed by whitespace or
   end ≤ 2). **Distractors:** 4 distinct answers of *other* filtered items (≠ correct), sampled uniformly — same distribution as the
   correct answer (the H2 rule; ±small offsets are not used because they recreate H's leak). Sampling seed 303. Exemplars: 3 filtered items
   (seed 1303) with their GSM8K solutions as worked steps (calculator annotations `<<…>>` removed).
4. **aqua_easy**: AQuA-RAT train (`github.com/google-deepmind/AQuA`, `train.json`), native 5 options and native correct letter,
   reformatted from "A)x" to "(A) x" only. Filtered to items with all 5 options present, rationale ≤ 3 non-empty lines, question ≤ 60 words.
   Sampling seed 304. Exemplars: 3 filtered items (seed 1304) with their rationale as worked steps (native letters kept; not forced).

Item sets: the first 30 items of each stream are the pilot; the full run is the first 200.

## Phase I1: pilot gates (all four must hold for a task to run)
G1 no-CoT accuracy ∈ [0.55, 0.85]; G2 no single no-CoT letter > 60%; G3 with-CoT accuracy ≥ 0.85 (all on 30 items);
**G4 options-only baseline ≤ 0.30**: the H2 detector (per-option features: first number in the option, rank, sorted absolute
differences to the other options, mean difference, is-median, option string length; logistic regression and random forest,
5-fold CV, argmax over options) plus the median rule, best of these. Generated tasks: 5,000 items from an independent seed (seed + 9000).
GSM8K / AQuA: all filtered items (finite pool, size reported).
- **0 tasks pass → stop and report** (no improvising). **≥ 1 pass → I2 on every passing task.**

## Phase I2: full run per passing task
N = 200, layers 0–27, Targets A and B, 5 seeds, all H2 baselines and controls. **Option-E sanity token:** the token where "(E" begins
in the CoT prompt, located directly (not by fixed offset; for H2's items this is the same token). Hard cap 5 h for I1 + I2.

## Verdict scenarios (over the tasks that ran)
1. **Generalizes:** every task that ran meets the hypothesis.
2. **Boundary:** at least one meets it and at least one does not → compare task class / no-CoT accuracy / CoT gain.
3. **Task-specific:** none meets it → H2's result may be specific to multiplication; the task-competence interpretation is re-examined.
A task where the probe signal is clearly present but peaks outside [19, 25] or below 0.55 is reported as "signal present, criteria not met".

## Deliverables
`pilot.md`; per task `<task>/README.md` and `<task>/metrics.json`; a combined layer-sweep figure overlaying H2 and each I task;
commits: preregistration → I1 pilot → one per I2 run; a one-page Kindle summary.
