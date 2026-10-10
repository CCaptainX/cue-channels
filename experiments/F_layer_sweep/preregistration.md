# F: pre-registration — full layer sweep, Target A vs Target B, positional-bias checks

**Registered:** 2026-10-10T23:31:32Z (UTC). No F task data has been downloaded or inspected yet. Tasks are pinned by upstream URL
here, and file hashes get recorded in the pilot commit.

## Questions

1. Where (which layer) does the pre-CoT decision form, separately for **Target A** (the model's no-CoT answer) and
   **Target B** (the model's with-CoT answer)?
2. Is the probe reading something genuine, or positional / question-length signal?
3. Is any finding stable across seeds and baselines?

## Model and prompts (as in E2)

`Qwen/Qwen2.5-1.5B-Instruct`, bf16 on MPS, greedy, raw-text BBH **3-shot** prompts taken verbatim from
`cot-prompts/<task>.txt` at github.com/suzgunmirac/BIG-Bench-Hard (main), canary header removed.
- **CoT prompt:** `<description>\n\n<3 CoT exemplars>\n\nQ: <input>\nA: Let's think step by step.`, 384 new tokens,
  answer = the **first** "answer is (X)".
- **No-CoT prompt:** same exemplars with the reasoning removed (`A: (X)`), ending `Q: <input>\nA: (`, 2 new tokens,
  answer = the first character if it's a valid option letter.
- Unparsed answers count as wrong for accuracy and are excluded from the probe whose target they are (count reported).

## Step 1: pilot (task selection)

- **Candidates:** `hyperbaton` (2-way), `tracking_shuffled_objects_three_objects` (3-way), `date_understanding` (multi-way),
  from `bbh/<task>.json` at the same repo.
- **Sample per task:** `random.Random(0).sample(range(len(examples)), 200)`, sorted. The pilot uses the first 30 of these
  (exemplars must not appear among them; checked by exact match).
- **Gates (all three must hold):** (G1) no-CoT accuracy ∈ [0.55, 0.85]; (G2) no single letter > 60% of no-CoT answers;
  (G3) with-CoT accuracy ≥ 0.85. Point estimates on 30 items, CIs reported.
- **If several pass:** choose the largest (with-CoT − no-CoT) accuracy gap; ties go to the task with more options.
- **If none pass:** write `pilot.md`, commit it, report back and stop. No full run, no other task.

## Step 2: full experiment (only if a task passes)

- N = 200 (the seed-0 sample), every item run in both formats.
- **Activations** (CoT prompt, one forward pass per item, all 28 layers): (i) the **last prompt token**, PRIMARY;
  (ii) the token **5 positions before the question end**, where question end = the last token of `<input>`;
  (iii) the **mean over all prompt tokens**; (iv) secondary: the mean over the question's tokens only.
- **Probes:** StandardScaler + multinomial LogisticRegression(C = 0.1, max_iter = 5000). Stratified 5-fold CV (plain
  KFold if a class has < 5 items, and recorded). **Probe A** target = no-CoT letter; **Probe B** target = with-CoT letter.
- **Per (layer, target):** pooled CV accuracy with 2,000-resample bootstrap 95% CI (seed 0); **5 seeds** (`random_state`
  0–4 for fold shuffling and LR) → mean ± SD; majority-class baseline; chance = 1/K (K = number of options);
  **random-Gaussian baseline** (N × 1536 i.i.d. N(0, 1) features, same labels, same CV, 5 seeds).
- **Peak layer** (per target) = argmax over layers 0–27 of the 5-seed mean accuracy. Because that's a selection over 28
  layers, a **max-over-layers permutation test** (100 label permutations, max accuracy across layers each time) gives a
  selection-corrected p-value. The fixed layer 24 (E's primary) is also reported.
- **Positional checks**, at each target's peak layer, accuracy with bootstrap CI:
  1. *Length-balanced subset:* items whose prompt token-length is within ±2 of the median. Report n; if n < 60 it's
     labelled underpowered.
  2. *Position-only probe:* features [prompt_length, question_end_token_index], same probe and CV.
  3. *Shifted-token probe:* activations (ii).
  4. *Mean-pooled probe:* activations (iii), plus (iv) as secondary.

## Decision rules

- **Target A supported** iff probe_A@peak > majority_A + 0.05 **and** the CI lower bound > majority_A.
- **Target B supported** iff probe_B@peak > chance + 0.10 **and** the CI lower bound > chance. Also reported, not
  decisive: the same test against majority_B.
- **Positional bias ruled out** (per target) iff probe@peak exceeds **both** the position-only probe and the
  shifted-token probe by ≥ 0.10 with non-overlapping 95% CIs. If this fails, the report says which comparison
  failed. A shifted-token match means the signal is spread across the question, not that it's positional.
- **Verdict labels:** "supported" or "not supported", plus "inconclusive" when the selection-corrected permutation
  p ≥ 0.05 while the uncorrected rule passes.

## Figures

`figures/`: (1) layer sweep (A solid, B dashed, CI shading, chance / majority / Gaussian lines), 300 DPI; (2) per-item
agreement of probe-A vs probe-B predictions at the peak layer; (3) bar chart of the positional checks; (4) no-CoT
letter distribution.

## Limits

Hard cap 3 h wall time. No 7B, no causal patching, no task change without checking back.
