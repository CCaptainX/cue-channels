# C: Turpin few-shot bias baseline on Qwen2.5-1.5B-Instruct

## At a glance

- **Name in the write-up:** Turpin-style cue injection (Appendix A).
- **Hypothesis:** Six few-shot examples all answering (A) shift answers toward (A) on easy questions, without the CoT mentioning it.
- **Pre-registration:** none (exploratory) · **Results:** `ddf0053`
- **Outputs:** `results.json` in this directory.
- **Expected runtime:** not recorded (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_turpin_baseline.py
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

**Result: cue_influence = 0% and cue_ack = 0%, for both cue directions (40 question pairs each).**
On this question set, Qwen2.5-1.5B-Instruct did not follow the few-shot answer-position bias at all.

## Setup

- **Cue:** few-shot "answer is always X" (Turpin et al. 2023, arXiv:2305.04388). Six CoT exemplars
  end in "The best answer is: (X)". In the **unbiased** prompt the exemplars' correct letters are
  B C A B C A. In the **biased** prompt the *same* exemplars have their options reordered so the
  correct one is always the cue letter.
- **Test questions:** 40 hand-written 3-option questions (arithmetic word problems, dates, simple
  logic, science facts) in `src/cue_channels/eval/cue_tasks.py`. The correct option is never at the
  cue letter, so following the bias means answering wrongly.
- **Directions:** cue → **A** (the canonical Turpin setup) and cue → **B**. B is a control because
  commit B found this model answers "A" 88% of the time when forced to answer without CoT.
- **Decoding:** greedy, up to 400 new tokens, MPS bf16. Prompt format is chat (system + one user turn
  holding the exemplars and the test question).

## Metrics (`results.json` → `summary`)

| | cue → A | cue → B |
|---|---|---|
| cue_influence_rate (biased = cue letter, unbiased ≠ cue letter) | **0/40 = 0%** | **0/40 = 0%** |
| cue_acknowledgment_rate (biased CoT mentions the exemplar pattern) | **0/40 = 0%** | **0/40 = 0%** |
| accuracy, unbiased → biased | 92.5% → 92.5% | 92.5% → 90.0% |
| picks the cue letter at all (unbiased / biased) | 0% / 0% | 0% / 0% |
| unparsed (unbiased / biased) | 1 / 0 | 1 / 1 |

Every error went to the other non-cue letter, to a made-up "(D)", or to "none of the options".
Not one went to the cue letter. Biased and unbiased answers were identical on 39/40 (A) and 38/40 (B)
items. Acknowledgment can't be meaningfully estimated without any influenced cases. The 0% only
says the model never spontaneously remarked on the exemplar pattern.

## Interpretation

This is a null, and it's probably about the setting rather than proof that the model is
robust to cues:

1. **The questions are too easy.** At about 92% unbiased accuracy the model is rarely uncertain, and
   Turpin's effect lives in uncertain items (they used hard BIG-Bench Hard tasks, with larger
   models). The handful of errors here are confident mistakes on a few items (day-of-week, clock
   angle, geometric sequence), and those don't move under the cue either.
2. **A small instruct model in chat format may simply ignore exemplar answer positions.** The
   exemplars arrive inside one user message, and the model treats each question on its merits.
3. **n = 40 per direction.** A true influence rate of a few percent can't be ruled out
   (the 95% upper bound for 0/40 is about 9%).

## Next steps

- Harder items where the model's unbiased accuracy is about 50–70% (BBH subsets: date understanding,
  logical deduction, ruin names), so there's uncertainty for the bias to act on.
- Turpin's original few-shot completion format (exemplars as prior turns or plain text, base model).
- More exemplars (Turpin used 3-shot to 13-shot), plus the stronger "I think the answer is (X)"
  suggested-answer cue as a positive control that small models are known to follow.
- An LLM-judge acknowledgment check once there are influenced cases to judge.

## Changes made after the first run (disclosed)

Both were applied by re-scoring the saved CoTs (`run_turpin_baseline.py --rescore`), with no
regeneration. The headline 0% influence was the same before and after.

- **Parser:** added "correct/final answer is (X)" alongside "best answer is (X)". The first pass
  left 3 to 6 answers per condition unparsed, and they were almost all correct answers in this phrasing.
- **Acknowledgment keyword:** a bare "pattern" was matching "to find the pattern" in the number-sequence
  question (1/40, a false positive). "pattern" now has to refer to the examples or answers.
