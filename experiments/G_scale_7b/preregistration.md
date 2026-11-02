# G (Qwen2.5-7B-Instruct on BBH) and H (arithmetic on Qwen2.5-1.5B-Instruct): pre-registration

**Registered:** 2026-10-11T01:04:59Z (UTC). Committed before the 7B model is downloaded or loaded, and before any H data
is generated. Inherits F's pre-registration ([`../F_layer_sweep/preregistration.md`](../F_layer_sweep/preregistration.md),
`5f12575`): probes, CV, bootstrap CIs, seeds, baselines, positional checks, peak-layer selection and its correction,
decision rules and figures. Only what's stated below changes.

**Shared change (efficiency only):** generation stops as soon as the completion contains `\nQ:` (the model starting
a new few-shot question). Only the first "answer is (X)" is ever parsed, so this can't change any answer. CoT
cap: 384 new tokens.

---

## Stage 1: G (`Qwen/Qwen2.5-7B-Instruct`)

### G1: load check
- Precision ladder: (1) bf16 on MPS; (2) if OOM, bf16 on MPS with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`;
  (3) if still OOM, **stop and report** (rule (c)). fp16 isn't a separate rung (same memory as bf16), and int8 isn't
  attempted (no reliable MPS backend).
- Verify one forward pass, the hidden-state count (= n_layers + 1) and the shape [d_model]. Record d_model and n_layers.
- Commit: `G1 load_check: Qwen 2.5 7B activation extraction verified`.

### G2: pilot
- Same as F's pilot: hyperbaton, tracking_shuffled_objects_three_objects and date_understanding; the first 30 items of each
  task's seed-0 sample of 200; BBH 3-shot raw text. Gates: G1 no-CoT ∈ [0.55, 0.85]; G2 top letter ≤ 60%; G3 with-CoT ≥ 0.85.
  Tiebreak: largest (with-CoT − no-CoT) gap, then more options.
- A constant-letter collapse (> 60%) on 7B is reported as a finding in its own right. No extra tasks are tried.
- Commit: `G2 pilot: BBH task triage on Qwen 2.5 7B`.

### Pivot rule (automatic)
**If no task passes the G pilot, G stops and H is dispatched immediately, without manual confirmation.**
If a task passes, run G3.

### G3: full F protocol on 7B (only if the pilot passes)
F's full protocol on the selected task, N = 200, layers 0 to n_layers − 1. N drops to 100 if the projected runtime is
over 6 h. Random-Gaussian baseline dimension = 7B's d_model. Hard cap 6 h for G1 + G2 + G3; at the cap, save partial
results and stop.

---

## Stage 2: H (`Qwen/Qwen2.5-1.5B-Instruct`, arithmetic), only if the G pilot fails

### Items
Generated with `random.Random(<seed>)` (variant A seed 101, B seed 102, C seed 103). Each item has the correct value plus
**4 distractors** = correct + d for 4 distinct d sampled from {−10, −2, −1, +1, +2, +10}, keeping only positive values
distinct from the correct one (resampled until there are 4). The 5 options are shuffled with the same RNG and lettered (A)–(E).
- **A:** `What is a + b?`, with a, b ~ U[10, 99] (sanity floor).
- **B:** `What is a + b - c?`, with a, b, c ~ U[10, 99], requiring a + b − c ≥ 1 (resampled otherwise).
- **C:** `What is a * b?`, with a, b ~ U[10, 20].
The pilot takes the first 30 items of each variant; the full run takes 200 items (the pilot items plus 170 more from the same stream).

### Prompts (F format, raw text, 3-shot)
Description line: `Answer the arithmetic question by choosing the correct option.` Three exemplars per variant are
generated from `random.Random(<seed> + 1000)` with the same item rules, excluded from the item stream, and given
correct letters in the order B, A, C. The CoT exemplar format is:
`Q: <question>\nOptions:\n(A) …\n…\nA: Let's think step by step.\n<worked steps>\nSo the answer is (X).`
with deterministic worked steps (A: "a + b = s."; B: "a + b = s. s - c = r."; C: "a * b = a * (b - 10) + a * 10 = p + q = r.", or
"a * 10 = r." when b = 10). The answer-only exemplars use the same items with `A: (X)`. Test prompts end exactly as in F.

### H1: pilot
No-CoT and with-CoT on 30 items of each of A, B and C. Same three gates and tiebreak as F. If no variant passes,
**stop and report** (no further variants). Commit: `H1 pilot: arithmetic difficulty calibration`.

### H2: full experiment
F's full protocol on the selected variant, N = 200, layers 0–27, all deliverables. Hard cap 2 h.
Positional check "length-balanced" uses the prompt token-length as in F. Arithmetic prompts vary little in length,
so n is expected to be large.

---

## Decision rules (same as F)
- **Target A supported** iff probe_A@peak > majority_A + 0.05 and the CI lower bound > majority_A.
- **Target B supported** iff probe_B@peak > chance + 0.10 and the CI lower bound > chance (vs majority_B also reported).
- **Positional bias ruled out** iff probe@peak beats both the position-only and the shifted-token probes by ≥ 0.10 with
  non-overlapping CIs (the report says which comparison failed, if any).
- **Inconclusive** if the uncorrected rule passes but the max-over-layers permutation p ≥ 0.05.

## Reporting
Kindle PDFs: "convert — G pilot results" (always); "convert — G results" (if G3 runs); "convert — H pilot results" (if
the pivot happens); "convert — H results" (if H2 runs). A single report back to the user at the end, or at a hard cap, or at a G1 failure.
