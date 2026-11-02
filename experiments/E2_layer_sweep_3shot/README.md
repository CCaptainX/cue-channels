# E2: E with BBH 3-shot prompts and a pilot gate. Gate FAILED; full run not executed.

## At a glance

- **Name in the write-up:** BBH 3-shot pilot (pilot failure) (Appendix A).
- **Hypothesis:** E with BBH's standard 3-shot prompts passes a no-CoT competence pilot gate (it did not).
- **Pre-registration:** `c08b6fb` · **Results:** `daa0a27` (pilot)
- **Outputs:** `pilot.json` in this directory.
- **Expected runtime:** not recorded (30-item pilot) (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_e2_layer_sweep.py --pilot
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Pre-registration: [`preregistration.md`](preregistration.md) (commit `c08b6fb`, inherits E's `8647c89`).
Pilot output: [`pilot.json`](pilot.json), including all 30 raw completions.

**Pilot outcome.** On the first 30 items of E's seed-0 sample, Qwen2.5-1.5B-Instruct prompted with BBH's exact
3-shot answer-only format (raw text, prompt ending `A: (`) reached a no-CoT accuracy of **0.267** (bootstrap 95% CI
[0.133, 0.433]). That is below chance (0.333) and far outside the pre-registered gate of [0.55, 0.85], so the
gate **FAILED** and, as pre-registered, the full N = 150 run, the probe and the layer sweep were **not** executed.
The letter-bias flag also tripped: the model answered **B on 22/30 (73%)**, A on 5 and C on 3, with no unparsed
outputs. The ground truth for these 30 items is A 11, B 6, C 13, so the B preference alone pushes accuracy below
chance. The same B preference showed up in E's zero-shot chat format (77% of parsed no-CoT answers), so it isn't an
artifact of either prompt format. Without CoT, this 1.5B model cannot do 3-object logical deduction and defaults to
a near-constant letter, so there's no pre-formed decision for the E/E2 probe to detect. Candidate fallback tasks, noted
and **not** run, pending a scope decision: `tracking_shuffled_objects_three_objects`, `date_understanding`,
`hyperbaton`. Any of them should go through the same pilot gate first.
