# E2: pre-registration (inherits E)

**Registered:** 2026-10-08T23:19:27Z (UTC), before any E2 model run (pilot or full).

**E2 inherits E's pre-registration** ([`../E_layer_sweep/preregistration.md`](../E_layer_sweep/preregistration.md),
commit `0f7353a`) unchanged, except for the prompt format and the pilot gate. Model, data sample (N = 150,
seed 0, same sha256), hypothesis, primary probe (CoT-prompt activations @ **L24** → model's **own no-CoT letter**),
probe and CV setup, bootstrap CIs, controls, permutation test and secondary analyses are as in E.

## Decision rule (E's wording, carried forward)

SUPPORTED iff (1) probe_acc@24 ≥ no_cot_acc − 0.05 **and** (2) the bootstrap 95% CI lower bound of probe_acc@24 >
max(majority-class rate of the probe's labels, 95th percentile of the permutation null).
Also reported, but not used for the verdict: the restated form (2') CI lower bound > max(0.333, majority rate).

## Change 1: prompt format (BBH standard 3-shot)

- Exemplars: the exact BBH 3-shot CoT prompt `data/logical_deduction_three_objects_cot.txt`
  (github.com/suzgunmirac/BIG-Bench-Hard, `cot-prompts/`), sha256
  `4ed65246c0c1c4b0b4053d784181892b842bce19cf4358c976f34b1a434ef034`. The task description and 3 Q/A exemplars are
  used verbatim, with the canary header line removed. Exemplar answers are B, A, C. None of the 3 exemplars appears among the 250
  dataset items (checked by exact match).
- **Raw-text completion** (as in the BBH paper), no chat template.
- **CoT prompt:** `<description>\n\n<3 CoT exemplars>\n\nQ: <input>\nA: Let's think step by step.`
  Greedy, max_new_tokens = 256. Answer = the **first** "answer is (X)" in the completion (the model may go on to
  invent further Q/A pairs; only the first answer counts).
- **No-CoT (answer-only) prompt:** the same exemplars with the reasoning removed (`A: (X)`, the BBH answer-only
  format), ending `Q: <input>\nA: (`, so the next token is the letter. Greedy, max_new_tokens = 2. Answer =
  the first character of the completion if it is A/B/C, else unparsed.
- Activation position is unchanged in definition: the **last prompt token** of each prompt, all 28 layers.
  The CoT prompt's last token is the "." of "Let's think step by step.".

## Change 2: pilot gate (run before the full N = 150)

- **Pilot items:** the first 30 items of E's sorted seed-0 sample of 150. No-CoT prompt only. No activations are
  extracted and no probe is trained in the pilot.
- **Gate:** PASS iff the pilot no-CoT accuracy (vs ground truth, unparsed counted wrong) is in **[0.55, 0.85]**.
  Otherwise STOP: write up the pilot, report back, and run no fallback task.
- **Letter-bias flag:** if any single letter is chosen on **> 60%** of the 30 pilot items, it is flagged and reported.
  A flag alone doesn't stop the run; rule 2's majority-rate floor already penalizes a constant-letter bias.
- If the gate passes, the full run uses all 150 items (pilot items included, since the pilot only measures
  task difficulty and never sees a probe).
