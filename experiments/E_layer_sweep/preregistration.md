# E: pre-registration — late-layer pre-CoT probe on BBH logical deduction

**Registered:** 2026-10-08T19:47:58Z (UTC). Committed before any model run on this task. No activations, generations or
probe results for E existed when this was written. The only thing inspected was the dataset's format and
label balance (below).

## Motivation

In B and D, a probe at the pre-registered layer 14 could not decode the answer before CoT (≈ chance). D then found,
post hoc, a weak rise in layers 21–27 in every condition. E tests that late-layer signal prospectively, on a task
the model can partly solve without CoT, so there is a pre-formed decision to find.

## Hypothesis

At layers 21–27 of Qwen2.5-1.5B-Instruct, a linear probe on last-prompt-token activations predicts the model's
answer letter at an accuracy at least equal to the model's own no-CoT accuracy on the task (within 0.05). If so, the
model forms an answer in activations before the CoT, in late layers only, and B's layer-14 null was a layer-choice artifact.

## Data

- BIG-Bench Hard `logical_deduction_three_objects` (Suzgun et al.), file `data/logical_deduction_three_objects.json`
  from github.com/suzgunmirac/BIG-Bench-Hard (main), sha256 `e590fae72e3a6041fbd23bd8ff6b74baf44ca563f6ba40a041c55e7b867c1681`.
  250 items, 3 options; targets (A) 80, (B) 86, (C) 84.
- **N = 150**, drawn with `random.Random(0).sample(range(250), 150)`, sorted. Fixed now; no replacement or filtering.

## Model and prompts

- `Qwen/Qwen2.5-1.5B-Instruct`, bf16 on MPS, greedy decoding, same chat template as B.
- **Zero-shot** (not BBH's 3-shot), so the format matches B. The user turn is the BBH `input` verbatim.
- **CoT prompt:** system = B's CoT system prompt ("Think step by step, then give your final answer on the last line as 'Answer: <letter>'."). max_new_tokens = 512.
- **No-CoT prompt:** system = B's direct prompt ("Reply with only the letter of the correct option, formatted as 'Answer: <letter>'."). max_new_tokens = 8.
- **Parsing:** last `Answer: X` / `answer is (X)` with X ∈ {A, B, C}; otherwise unparsed (reported, never imputed).

## Activations

Residual stream after blocks 21–27 (plus all other layers, reported as secondary), at the **last prompt token**
(the final token of the assistant header, before any generated token), for **both** prompts:
- **CoT-prompt activations: PRIMARY.** This is the state before the CoT is emitted.
- No-CoT-prompt activations: secondary. At this position the model is about to emit the letter itself, so it's an upper reference, not a pre-CoT test.

## Probe

StandardScaler + multinomial LogisticRegression(C = 0.1, max_iter = 5000), as in B/D. **Stratified 5-fold CV**
(shuffle, seed 0). Every item is tested once, giving pooled accuracy with a **95% CI from 2,000 bootstrap resamples** of
the per-item correctness. Items with an unparsed target label are excluded from that probe (count reported).

## Primary metric and decision rule

- **Primary probe:** CoT-prompt activations at **layer 24** → the model's **own no-CoT answer letter**.
- `no_cot_acc` = the model's no-CoT accuracy against BBH ground truth on the 150 items (unparsed counts as wrong).
- **SUPPORTED iff both:**
  1. probe_acc@24 ≥ no_cot_acc − 0.05, and
  2. the bootstrap 95% CI lower bound of probe_acc@24 > max(majority-class rate of the probe's labels, 95th
     percentile of the permutation null). Chance is not 0.5 for 3 options, so this replaces the "CI excludes 0.5" wording.
- Otherwise **NOT SUPPORTED**. Layers 21–23 and 25–27 are reported with CIs but don't change the verdict.

## Secondary analyses (reported, not used for the verdict)

1. CoT-prompt activations → the model's **with-CoT** answer letter (layers 21–27).
2. CoT-prompt activations → **ground-truth** letter.
3. No-CoT-prompt activations → own no-CoT letter (upper reference).
4. Full 28-layer sweep for every probe.
5. Behavior: with-CoT accuracy, no-CoT accuracy, no-CoT letter distribution (A-bias), agreement between no-CoT and
   with-CoT answers.

## Required controls

- **Positive control:** probe at layer 24 (CoT-prompt activations) for a fact written in the prompt, the object category
  (books / birds / golfers / vehicles / fruits, 5-way). Expected ≥ 0.95.
- **Permutation test:** primary probe retrained on 200 label permutations → null distribution; report its mean and
  95th percentile and the permutation p-value.

## Known limitations, stated in advance

- The decision rule compares probe accuracy on the model's *own* letter with the model's accuracy against *ground
  truth*. These are different quantities, so a high own-answer probe can pass even if no_cot_acc is low. The
  secondary analyses (ground-truth probe, majority baseline, permutation null) are there to interpret that.
- One task, one model, N = 150, zero-shot. The published no-CoT accuracy for this model on this task wasn't verified
  in advance; if no_cot_acc falls outside 0.60–0.80 that is reported, but the analysis isn't changed.
