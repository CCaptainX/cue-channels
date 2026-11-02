# K: per-token × per-layer probe maps (H2 multiplication vs. I3 unit conversions)

## At a glance

- **Name in the write-up:** The per-token sweep (Section 5).
- **Hypothesis:** Multiplication and unit conversions peak at different prompt positions (≥ 0.10 margin). Not confirmed.
- **Pre-registration:** `1d8ad88` · **Results:** `7002b47`
- **Outputs:** `metrics.json` in this directory.
- **Expected runtime:** about 8 min (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_k_per_token.py
python scripts/plot_k_heatmaps.py
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Pre-registration: [`preregistration.md`](preregistration.md) (`1d8ad88`). Numbers: [`metrics.json`](metrics.json) (16 anchors × 28 layers per task,
plus H2's every-token map). Heatmaps: `figures/heatmap_H2_mult.png`, `figures/heatmap_I3_unit_conversions.png`,
`figures/heatmap_H2_mult_all_tokens.png`. Target A (no-CoT answer), seed 0, 5-fold CV (grouped by fact for unit conversions).
Runtime 8 min. Anchor tokens verified on the first 3 items of each task (`anchor_token_checks`).

## Pre-registered decision: NOT confirmed
| | H2 multiplication | I3 unit conversions |
|---|---|---|
| Best cell (layers ≥ 18) | **"A" of "A:" @ L25 = 0.765** (answer span) | **"(E" @ L20 = 0.785** (option region) |
| (a) regions differ | yes | yes |
| (b) margin over the other task's best anchor, same layer | **0.04** (needed ≥ 0.10) | **0.085** (needed ≥ 0.10) |

Each task's best cell lands where its hypothesis predicted (H2: answer span; unit conversions: at the "(E" anchor), but rule (b) fails for both,
so **task-dependent positional structure is not confirmed**. By eye, the two maps share most of their structure.

## What the maps show (layers ≥ 18)
| Anchor | H2 (L22) | Unit conversions (L22) |
|---|---|---|
| question end (before options) | 0.25 (majority 0.28) | 0.34 (majority 0.39) |
| "(A" → "(E" | 0.22 → **0.71** | 0.33 → **0.76** |
| "A" of "A:" | **0.74** | **0.70** |
| "think" | 0.45 | 0.44 |
| last token "." | **0.70** | **0.47** |

1. **Shared:** before the options the letter is undecodable (it can't be, since the letter is assigned by the options); it then builds up as each
   option is read, from about layer 18, and is high at "(E" and again at "A:" in both tasks. That fits a model that recognizes which listed value is
   its answer as it reads the options.
2. **Shared:** a dip at "think" in both tasks.
3. **The difference is the final token:** in H2 the answer comes back at the last "." (0.70), and in unit conversions it doesn't (0.47). I3's
   last-token failure is a property of that final position, not a different location for the answer.

The I3 "option-E" observation is therefore real but **not distinctive**: multiplication holds the answer at "(E" just as strongly (0.71–0.72).
H2's every-token map peaks at the "A" token 9 positions before the end (L25, 0.765).
