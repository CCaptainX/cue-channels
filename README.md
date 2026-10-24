# cue-channels: pre-CoT answer decoding on Qwen 2.5 1.5B

**Is a model's answer readable before its chain of thought starts, and does the model use it?**

Cox, Kianersi & Garriga-Alonso (2026) showed that a model's eventual answer is often linearly decodable from
activations at the last token before the chain of thought (CoT), and that steering along that direction changes the
answer, on tasks where the CoT rarely changes it. This repo runs a pre-registered replication on one of their models,
Qwen 2.5 1.5B Instruct, on generated five-option two-digit multiplication: a task where the CoT does real work (no-CoT
accuracy 0.74, with-CoT 0.985). The pre-CoT answer is linearly readable, as they found. Four pre-registered tests of
where that signal comes from and whether the model uses it came back not supported, mixed or inconclusive, including a
steering null. Post-hoc analyses fit a hypothesis, "readable but not committed", which is not established here. Along the
way, an options-only baseline caught a distractor leak that every other check had missed. Scope: one model, one task
family.

This is a scoped case study on the other side of the task-regime boundary Cox et al.'s results imply, not a
contradiction of them: their causal results come from tasks where the CoT is largely post hoc; here the CoT changes the
answer for about a quarter of items. Their tasks were not re-run through this pipeline.

The write-up is a LessWrong post (link added on publication); its source is
[`writeup/lw_post_draft.md`](writeup/lw_post_draft.md).

## Headline numbers (experiment H2)

| | No-CoT target (model's answer without CoT) | With-CoT target (final answer after CoT) |
|---|---|---|
| Peak layer, accuracy (seed 0) [95% CI] | L22, **0.705** [0.645, 0.765] | L21, **0.667** [0.596, 0.727] |
| 5-seed mean ± SD | 0.730 ± 0.019 | 0.674 ± 0.013 |
| Baseline | majority 0.280 | chance 0.200 |
| Max-over-layers permutation | p < 0.01 (registered, 100 perms); p = 0.001 (post hoc, 1,000) | same |

Options-only baseline: 1.000 on the original (leaky) distractors, 0.206 on the corrected ones (chance 0.20).
Pre-registered follow-ups J, K, M, N: none came back "supported" (see [`MAPPING.md`](MAPPING.md)).

## Navigating from the post

The post names experiments thematically ("the causal steering test") and cites commit SHAs in its appendix.
[`MAPPING.md`](MAPPING.md) maps letters to names, directories and SHAs. Every experiment directory has a README with
its hypothesis, pre-registration and results SHAs, the exact commands, and the expected runtime.

## Reproduce the headline result

Tested on macOS (Apple M4 Pro, 24 GB, MPS) with Python 3.14.5. CUDA and CPU also work through the same backend but
were not used for the committed results.

1. Clone and enter the repo: `git clone git@github.com:CCaptainX/cue-channels.git && cd cue-channels`.
2. Create the project-local venv with the pinned dependencies: `make setup` (or `make setup-exact` for the full
   transitive lock in `requirements-lock.txt`). Nothing is installed system-wide.
3. Run the fast tests: `make test` (no model download).
4. Reproduce H2: `make reproduce-h2` (equivalently `bash scripts/reproduce_h2.sh results/h2_repro`). This downloads
   `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB) on first use, runs the options-only baseline, generates the 200 items, runs the
   model with and without CoT, caches last-token activations at all 28 layers, fits the probes (5 seeds, 5-fold CV),
   runs the permutation test, and compares the result with the committed `experiments/H2_distractor_fix/metrics.json`.
   About 25 min. It writes only to `results/` (git-ignored).
5. Rebuild the post's figures from the committed metrics: `make figures` (writes `writeup/figures/`).
6. Any other experiment: follow the "Reproduce" block in its `experiments/<letter>_<name>/README.md`. J and M need
   H2's activations; pass `--h2-dir results/h2_repro` after step 4.

**Reproduction check.** On 2026-10-24, a fresh clone of this repo in a new venv built from `requirements.txt`
(Python 3.14.5, M4 Pro, model weights already in the Hugging Face cache) ran `scripts/reproduce_h2.sh` in 22 minutes
and matched the committed results exactly: behavior (0.740 / 0.985), peak layers (22 / 21), seed-0 accuracies
(0.705 / 0.667), 5-seed means (0.730 / 0.674), verdicts, and the options-only baseline (0.206; 1.000 on H).

Behavior and the options-only baseline are deterministic (greedy decoding, fixed seeds). Probe accuracies can move
slightly across hardware or library versions; `scripts/compare_h2.py` accepts differences within the seed-stability
window reported in the post (seed-to-seed SD up to 0.046) and requires the verdicts to match.

## Pre-registration discipline

Every main-body experiment's hypothesis, gates and decision rules were committed before any of its data was generated:
in `git log`, each `preregistration.md` commit precedes the commits that add its results. Where a registered rule came
back not supported, mixed or inconclusive, that verdict is the reported result; later analyses are labelled post hoc
or exploratory (for example `experiments/H2_distractor_fix/exploratory/`, commits `647b901` and `c37dc63`). No history
has been rewritten.

## Identity

All commits are authored under the project's pseudonymous GitHub identity, **CCaptainX**
(`167642035+CCaptainX@users.noreply.github.com`), the same identity that signs the post.

## Layout

```
src/cue_channels/      model backend (models.py), config, probe training (probes/train.py); eval/ holds the original
                       cue-channel task code, kept because the tests cover it
scripts/               one script per experiment step (see each experiment README); shared: f_common.py (prompts,
                       parsing), h_arith.py (multiplication items), run_full_f_protocol.py (the probe protocol),
                       run_gh.py (load checks and pilots); reproduce_h2.sh + compare_h2.py
experiments/<X>_*/     preregistration.md, pilot / metrics JSON, generations, figures, README per experiment
                       (activation pickles are regenerated, not committed)
writeup/               the post source (lw_post_draft.md), make_figures.py, md_to_docx.py, figures/
scratch/               exploration that is not part of the post (see scratch/README.md)
tests/                 fast tests (`make test`); `make test-slow` downloads weights
requirements.txt       pinned direct dependencies; requirements-lock.txt is the full environment
```

Gated models (only the abandoned Llama experiment in `scratch/`) need `HF_TOKEN` in `.env`; nothing in the post does.
