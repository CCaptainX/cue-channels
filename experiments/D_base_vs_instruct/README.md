# D: does post-training ("distillation") cause B's pre-CoT null?

## At a glance

- **Name in the write-up:** The externalization ("distillation") hypothesis test (Appendix A).
- **Hypothesis:** If post-training taught the model to externalize computation, the base model shows more pre-CoT signal than Instruct.
- **Pre-registration:** none (exploratory) · **Results:** `c01d6c7`
- **Outputs:** `metrics.json` in this directory.
- **Expected runtime:** about 9 min (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_base_vs_instruct.py   # uses B's prompts (experiments/B_probe/labels.jsonl)
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

**Verdict: H2. The base model shows the same null, so the distillation hypothesis is rejected.**
At the pre-registered layer 14, the correct letter is undecodable before CoT in both the pretrained-only
model (0.540) and the Instruct model (0.500 on the identical prompt, 0.495 in B's chat format), even though
both solve every item with CoT (200/200). The two models behave almost identically.

## Hypothesis

Qwen2.5-1.5B-Instruct's post-training on CoT traces taught it to externalize computation, so it doesn't
pre-decide. A base model of the same size and pretraining should then show **higher** pre-CoT decodability
(H1: base ≥ 0.70, Instruct ≈ 0.51). If base ≈ chance too, distillation is not the mechanism (H2).

## Protocol (held constant vs B)

- **Questions:** the exact 200 prompts from `experiments/B_probe/labels.jsonl` (100 additions × both option orders).
- **Models:** `Qwen/Qwen2.5-1.5B` (base) vs `Qwen/Qwen2.5-1.5B-Instruct`. Same architecture, tokenizer and pretraining.
- **Prompt:** 4-shot raw text. The exemplars are additions disjoint from the 200, with correct letters A, B, B, A.
  The CoT format ends `Reasoning:` (exemplars: "x + y = s. s is option (L).\nAnswer: L"). The direct format ends `Answer:`.
- **Format confound removed:** the Instruct model also ran on the *identical* raw 4-shot prompt (2 × 2 design).
  B's chat-template run is re-scored with the same statistic for reference.
- **Position / layer:** last prompt token, all 28 layers. The headline is L14, as in B.
- **Probe:** StandardScaler + LogisticRegression (C = 0.1). The headline uses **grouped 5-fold CV over all 200**:
  both orderings of a question share a fold, every prompt is tested once, and a Wilson 95% CI is reported.
  B's 80/20 × 5 statistic is also in `metrics.json`.
- **Runtime:** MPS bf16, about 9 minutes total (CoT conditions about 200 s each, direct about 50 s each).

## Results (L14 unless noted)

| | Base, raw 4-shot | Instruct, raw 4-shot | Instruct, chat (B) |
|---|---|---|---|
| **Probe → correct letter** | **0.540** [0.471, 0.608] | **0.500** [0.431, 0.569] | **0.495** [0.426, 0.564] |
| Shuffled-label control | 0.520 | 0.515 | 0.485 |
| Positive control: option A > B | 0.985 | 0.985 | 0.980 |
| Positive control: first operand > 50 | 0.980 | 0.980 | 0.945 |
| CoT accuracy | **200/200** [0.981, 1.0] | 200/200 | 20/20 parsed (of 40) |
| Direct (no-CoT) accuracy | **59.5%** [0.526, 0.661] | 62.5% | 57.0% |
| Direct: share answering "A" | 43.5% | 57.5% | 88% |
| Direct: probe → model's own letter, L14 | 0.685 (majority 0.565) | 0.695 (majority 0.575) | 0.975 (majority 0.88) |
| Direct: probe → model's own letter, L21–27 | **0.955–0.965** | **0.975–0.98** | n/a |

**Verdict-matrix check:** base CoT accuracy is 100%, far above chance, so the test is meaningful.
The base probe is 0.540, not ≥ 0.70, so it lands in the "same null" cell.

## A second, post-hoc observation: a weak late-layer signal in every condition

The layer sweep shows correct-letter accuracy near 0.5 through about L14, then rising in late layers in **all**
conditions:

| mean accuracy | L0–13 | L21–27 | L27 |
|---|---|---|---|
| base CoT | 0.508 | **0.630** | 0.675 |
| Instruct CoT (raw) | 0.534 | 0.583 | 0.655 |
| Instruct CoT (chat, B) | 0.517 | 0.584 | 0.630 |
| base direct | 0.547 | 0.615 | 0.690 |
| Instruct direct (raw) | 0.519 | 0.636 | 0.745 |

For 28 layers of pure chance at n = 200, the best layer's 95th/99th percentile is 0.605/0.620, and the rise
here is smooth and consistent across conditions, so it is probably real. It is post hoc, though: L14 was the
pre-registered layer. Its size (about 0.6–0.7) roughly matches the models' **no-CoT accuracy (about 0.6)**. That is
what the capability-bound reading predicts: before the CoT, the model holds the answer only as well as it can
compute it in one pass. Base sits about 0.05 above Instruct in late layers under CoT. That is the direction H1
predicts, but it is well inside the CIs and far below the 0.70 threshold.

## Interpretation

- **Distillation is not what makes this null.** Removing post-training changes neither the L14 result nor the
  behavioral profile (both 100% with CoT, about 60% without).
- **The capability bound fits better.** Both models can't reliably do these additions in one forward pass. Pre-CoT
  decodability is at chance in middle layers and weak (≈ no-CoT accuracy) in late layers.
- **The direct-mode own-answer probe is now a strong positive control.** It reads the model's actual decision at
  0.96–0.98 from about L21 with near-balanced labels, much stronger than B's 0.975 vs 0.88 majority.
  Decisions live in late layers here, so L14 was a weak choice of layer for any decision readout.
- **B's 88% "A" bias was a chat-format effect.** On the raw few-shot prompt, Instruct says "A" 57.5% of the time and base 43.5%.

## Caveats

- **"Undistilled" is approximate.** Qwen2.5's base pretraining reportedly includes synthetic math and reasoning
  data generated by earlier Qwen models, so the base model isn't free of teacher-generated CoT. A cleaner
  contrast would be a base model with documented organic-only pretraining, or an earlier base checkpoint.
- One easy task, a single token position and 1.5B scale. The late-layer signal and the base-vs-Instruct gap
  both need a pre-registered test (layers 21–27 fixed in advance) on new items.

## Reproduce

`python scripts/run_base_vs_instruct.py` (about 9 min on M-series MPS; downloads `Qwen/Qwen2.5-1.5B`, about 3 GB).
`--reuse` re-scores the saved activation pickles (gitignored) without regenerating.
