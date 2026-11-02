# H: F's full protocol on arithmetic (a × b) with Qwen2.5-1.5B-Instruct

## At a glance

- **Name in the write-up:** The leaky first multiplication run (Section 4 (distractor-pattern correction), Appendix D).
- **Hypothesis:** On two-digit multiplication the pre-CoT answer is linearly decodable at the last prompt token (distractors were later found to leak).
- **Pre-registration:** `f3ce9d4` · **Results:** `f7df7bd` (pilot), `533dbff` (results)
- **Outputs:** `metrics.json` in this directory.
- **Expected runtime:** about 30 min (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_gh.py pilot --model qwen2.5-1.5b --suite arith --out experiments/H_arithmetic
python scripts/run_full_f_protocol.py --model qwen2.5-1.5b --suite arith --task C --out experiments/H_arithmetic
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Pre-registration: [`../G_scale_7b/preregistration.md`](../G_scale_7b/preregistration.md) (`f3ce9d4`, inherits F `5f12575`).
Dispatched automatically after the G pilot failed (`905adfd`). Pilot: [`pilot.md`](pilot.md) (`f7df7bd`); variant C was selected.
Full numbers: [`metrics.json`](metrics.json). Generations: [`generations.jsonl`](generations.jsonl). Activations gitignored.
Runtime: 18 min collection + 10 min analysis (cap 2 h).

## Behavior (N = 200)

no-CoT **0.725** [0.665, 0.785] · with-CoT **0.995** [0.985, 1.000] · no-CoT letters: D 77 (38.5%), E 38, B 32, C 32, A 21 · 0 unparsed.

## Pre-registered verdicts

| | Target A (no-CoT answer) | Target B (with-CoT answer) |
|---|---|---|
| Peak layer (5-seed mean) | **L26** | **L21** |
| Probe @ peak, seed 0 [95% CI] | **0.695** [0.635, 0.760] | **0.665** [0.605, 0.725] |
| 5-seed mean ± SD | 0.676 ± 0.037 | 0.706 ± 0.025 |
| Floor in the rule | majority 0.385 | chance 0.200 (majority 0.230) |
| Random-Gaussian probe | 0.335 ± 0.024 | 0.200 ± 0.047 |
| Max-over-layers permutation (100) | p = 0.0099, null p95 0.370 | p = 0.0099, null p95 0.305 |
| Layer 24 (E's fixed layer) | 0.670 [0.605, 0.735] | 0.655 [0.590, 0.720] |
| **Rule** | **SUPPORTED** | **SUPPORTED** (also vs majority) |
| **Positional bias ruled out?** | **No**: beats position-only, but not the shifted token | **No**: same |

Seed SD across all layers and targets: 0.005–0.041.

## Positional checks (at each target's peak layer)

| Probe input | Target A | Target B |
|---|---|---|
| Last prompt token (control) | 0.695 [0.635, 0.760] | 0.665 [0.605, 0.725] |
| Length-balanced (±2 tokens of median) | 0.695, **n = 200**: all prompts qualify, so not informative | 0.665, n = 200 |
| Position-only [prompt_len, q_end] | 0.385 (= majority) | 0.230 (= majority) |
| **Shifted token (q_end − 5 = "(E")** | **0.730** [0.670, 0.790] | **0.835** [0.780, 0.885] |
| Mean-pooled, all prompt tokens | 0.455 | 0.575 |
| Mean-pooled, question tokens | 0.445 | 0.575 |

The positional rule fails only on the shifted-token comparison, and in the direction the pre-registration said to read as
"the signal is spread across the question", not as a confound. The token 5 positions before the question end is always
"(E", the start of the last option, after options A–D are read. There the answer is *more* decodable than at
the final prompt token, especially Target B (0.835). Position alone carries nothing (prompt lengths are essentially
constant), and mean-pooling dilutes the signal, so it is localized, not uniformly spread.

## Layer profile

Both targets sit at baseline through about L15, rise through L16–19, then **jump between L19 and L21** (A 0.54 → 0.67, B 0.49 →
0.71) and plateau to L27. See `figures/1_layer_sweep.png`. Per-item agreement of the two peak probes is in
`figures/2_targetA_vs_targetB.png`, and the model's own no-CoT and with-CoT answers agree on 72% of items.

## A design confound in the items (found post hoc, disclosed)

The pre-registered distractor scheme (correct ± a subset of {1, 2, 10}) makes the **correct option identifiable from
the five option values alone**: it's the unique value from which the other four differ by allowed offsets. That holds on
**200/200** items, and "pick the median value" alone gets 134/200. With-CoT answers are 99.5% correct, so **Target B ≈ the
correct letter, and its pre-CoT decodability could reflect option-structure processing rather than computing a × b.** The
"(E" result fits either story.
Behavioral check (existing data, no new run): on the 66 items where the median rule is wrong, the no-CoT answer picks the true
answer 47 times (71%) and the median only 3 times, so the model's no-CoT behavior doesn't use the median shortcut. The
centre-of-offsets rule always equals the truth, though, so behavior can't tell it apart from real multiplication.
**Fix for any follow-up:** asymmetric distractors (e.g. all of correct + {+1, +2, +10, +11}, or distractors from other
plausible products), so that option structure can't identify the answer.

## Reading

Under the pre-registered rules both targets are supported: the answer the model gives without CoT, and the one it reaches with
CoT, are linearly decodable before any CoT token. That signal emerges abruptly around L20, not at L14. This is
consistent with D's post-hoc late-layer signal and with the capability bound: Target B decodability (0.67–0.71) ≈ no-CoT
accuracy (0.725). It's also consistent with the confound above, which a follow-up with fixed distractors has to rule out
before the result can be called "the model pre-computes the product".
