# E: pre-registered late-layer pre-CoT probe (BBH logical deduction, 3 objects)

## At a glance

- **Name in the LW post:** The letter-bias trap run (Section 4 (letter-bias trap), Appendix A).
- **Hypothesis:** A late-layer (L21–27) pre-CoT probe on BBH logical_deduction_three_objects decodes the eventual answer above baseline.
- **Pre-registration:** `0f7353a` · **Results:** `2bc86bc`
- **Outputs:** `metrics.json` in this directory.
- **Expected runtime:** about 7 min (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_e_layer_sweep.py
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Pre-registration: [`preregistration.md`](preregistration.md), committed in `0f7353a` before any model run.
Results: `metrics.json` (verdict and all probes), `generations.jsonl` (every CoT and no-CoT output).
Activation pickle gitignored. Runtime: 8 min 15 s on MPS.

## Verdict: NOT SUPPORTED, but the test was uninformative

| Pre-registered rule | Value | Pass? |
|---|---|---|
| (1) probe_acc@L24 ≥ no_cot_acc − 0.05 | 0.731 ≥ 0.340 − 0.05 | yes |
| (2) CI lower bound > max(majority rate, permutation p95) | 0.654 vs max(0.769, 0.723) = 0.769 | **no** |

**The task precondition failed.** The design assumed the model gets 60–80% without CoT. On this zero-shot
setup Qwen2.5-1.5B-Instruct is **at chance with and without CoT**:

| Behavior (N = 150) | Accuracy (95% bootstrap CI) | Letter distribution |
|---|---|---|
| no-CoT | **0.340** [0.267, 0.420] | B 100, A 23, C 7, unparsed 20 (all bare "B") |
| with-CoT | **0.333** [0.260, 0.407] | A 69, B 47, C 32, unparsed 2 |
| ground truth | n/a | A 48, B 52, C 50 |

No-CoT and with-CoT answers agree on only 35.9% of items. The CoTs are fluent but logically wrong: one
item argues an option is incorrect and then selects it. So there is no reliable decision to decode, and the
probe's main target, the model's own no-CoT letter, is mostly "B" (77% of parsed answers).

## Probe results: CoT-prompt activations, last prompt token, stratified 5-fold CV, bootstrap 95% CI

| Layer | → own no-CoT letter (PRIMARY) | → with-CoT letter | → ground truth |
|---|---|---|---|
| 21 | 0.723 [0.646, 0.800] | 0.392 | 0.353 |
| 22 | 0.738 [0.662, 0.815] | 0.385 | 0.413 |
| 23 | 0.746 [0.669, 0.823] | 0.419 | 0.340 |
| **24** | **0.731 [0.654, 0.808]** | 0.459 | 0.340 |
| 25 | 0.708 [0.631, 0.785] | 0.459 | 0.273 |
| 26 | 0.700 [0.615, 0.777] | 0.473 | 0.247 |
| 27 | 0.677 [0.592, 0.762] | 0.459 | 0.200 |
| majority baseline | 0.769 | 0.466 | 0.347 |

- **Permutation null** (primary, L24, 200 permutations): mean 0.666, p95 0.723, p = 0.040. The probe beats
  shuffled labels marginally but never beats always predicting "B". The permutation mean sits below the majority rate
  because a regularized probe trained on shuffled labels doesn't fully collapse onto the majority class.
- **Positive control** (object category, 5-way, L24): **1.000** [1.000, 1.000]. Extraction works.
- **Upper reference** (no-CoT-prompt activations → own no-CoT letter): 0.769–0.792 across L21–27, which is also
  ≈ majority. Even right before the model answers, the letter isn't cleanly decodable here. One reason: the probed
  token sits about 3 tokens *before* the letter (the model writes "Answer: " first), whereas in D the raw prompt
  ended at "Answer:". Compare D's 0.96–0.98.

## Comparison with B (layer 14) and D (late layers)

| | B (addition, L14) | D (addition, L21–27, post hoc) | E (BBH logic, L24, pre-registered) |
|---|---|---|---|
| no-CoT accuracy | 57% | 60–63% | 34% (chance 33%) |
| with-CoT accuracy | 20/20 parsed | 100% | 33% |
| pre-CoT probe → correct answer | 0.495 | ≈ 0.58–0.63 | 0.20–0.41 (≈ chance 0.35) |
| pre-CoT probe → own no-CoT answer | n/a | n/a | 0.68–0.75 (below majority 0.77) |

E neither confirms nor refutes D's post-hoc late-layer signal: on a task the model can't do at all, there is
nothing to find. It does extend the capability-bound pattern: probe accuracy for the correct answer tracks what
the model can actually compute, and that is chance here.

## Why the precondition failed (design notes, owned)

- **Zero-shot was my choice** (to match B's format). BBH's published numbers use the standard 3-shot CoT prompts,
  and small models depend heavily on those exemplars. The "55–65% no-CoT" expectation came from that setting.
- The no-CoT accuracy expectation wasn't checked with a pilot before registering. The pre-registration said a
  miss would be reported, not fixed, and it hasn't been fixed.

## Post-hoc sensitivity (not pre-registered)

Counting the 20 bare-letter no-CoT outputs as answers: no-CoT accuracy 0.400, "B" share 0.80, majority rate 0.80.
The verdict is unchanged.

## What a valid E2 needs

1. **A pilot gate, pre-registered:** run no-CoT on 30 items first, and proceed only if the no-CoT accuracy CI sits
   inside 0.55–0.85.
2. The BBH standard 3-shot prompts (or a task the 1.5B model demonstrably solves, such as D's addition at 2-shot).
3. Probe at the token right before the answer letter (end the prompt with "Answer:") for the upper reference.
4. Balanced own-answer labels, or report balanced accuracy, because a 77–80% single-letter bias swamps accuracy.
