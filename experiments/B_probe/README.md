# B: linear probe pipeline on synthetic data

## At a glance

- **Name in the LW post:** The initial layer-14 null (Appendix A).
- **Hypothesis:** A linear probe at layer 14, last prompt token, decodes which of two options is correct on two-option addition prompts.
- **Pre-registration:** none (exploratory) · **Results:** `d2e16f1`
- **Outputs:** `metrics.json` in this directory.
- **Expected runtime:** not recorded (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/make_dummy_probe_data.py
python scripts/train_dummy_probe.py
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

**Model:** Qwen/Qwen2.5-1.5B-Instruct (bf16, MPS) · **Position:** last prompt token, before any generated token
**Data:** 100 two-digit addition questions with two options. Each appears twice, with the correct answer once in (A) and once in (B), for 200 prompts (`labels.jsonl`).
**Probe:** StandardScaler + LogisticRegression(C=0.1), 80/20 split grouped by question.
Full numbers are in `metrics.json`. Activation and probe pickles are gitignored. To reproduce:
`scripts/make_dummy_probe_data.py [--format direct --behavior-n 200]`, then `scripts/train_dummy_probe.py`.

## Result: the correct letter is *not* decodable before the answer (≈ chance)

| prompt format | model behavior | probe → correct letter (L14, 5 split seeds) | shuffled-label control |
|---|---|---|---|
| CoT ("think step by step") | 20/20 correct when parsed (20 of 40 hit the 400-token cap) | **0.51 ± 0.06** | 0.60 |
| direct ("letter only") | 114/200 = 57% correct; answers "A" 88% of the time | **0.55 ± 0.06** | 0.65 |

Test sets have 40 prompts, so ±0.08 is noise (the shuffled controls land at 0.60 to 0.65). The layer sweep in `metrics.json` shows no layer that is clearly above chance.

The brief expected 70–90%. Before accepting ~50% I checked whether extraction is broken:

- **Positive controls pass.** On the same vectors, probes decode facts stated in the prompt: "option A > option B" at 1.00 and "first operand > 50" at 0.95/0.93. The activations carry prompt content, and the pipeline works.
- **The decision the model actually makes is decodable.** In direct mode, a probe for the model's *own* output letter gets 0.975 at L14. The majority baseline is 0.88 because of the A bias, so this control is positive but modest.

## Interpretation

At the last prompt token, Qwen2.5-1.5B has not computed which option is correct. Without CoT it mostly guesses, with a strong "A" position bias. With CoT it gets the answer right, so the computation happens *during* the CoT. On this easy arithmetic task there is no evidence of a pre-CoT decision. That is the expected situation for the "faithful" side of H3a, and it means pre-CoT decodability has to be tested on cued items where a shortcut answer exists (commit C), not on raw arithmetic.

## Caveats

- n_test = 40 per split, so treat differences under ~0.1 as noise.
- One template, one model, a single token position. Probing later positions inside the CoT is the natural next step.
