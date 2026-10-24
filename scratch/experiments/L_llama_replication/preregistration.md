# L: pre-registration — Llama 3.2 1B Instruct replication on times tables

**Registered:** 2026-10-15T23:22:30Z (UTC), before Llama is downloaded or run. Task = I3 times_tables (`scripts/i3_tasks.py`, `d5d1be1`): items,
prompts, gates, grouped CV and full protocol identical to I3, model swapped to `meta-llama/Llama-3.2-1B-Instruct` (raw text, no chat
template). Expected 16 layers, d_model 2048.

## Phases
- **L1 load check:** bf16 on MPS; if the gated repo is not accessible with the local Hugging Face credentials, **stop L and report**.
  Commit `L1 load check`.
- **L2 pilot:** 30 items, gates G1–G4 as I3 (G4 = 0.207, model-independent). If it fails, **stop L and report**.
- **L3 full run:** N = 200, all 16 layers, Targets A and B, 5 seeds, all H2/I3 controls, option-E token located directly.

## Hypothesis and decision rule
Proportional depth d = (layer index + 1) / n_layers. Qwen 1.5B references: H2 L22 → 0.821; I3 times tables L21 → 0.786.
**Generalizes to Llama** iff Target A's peak (5-seed mean argmax) has d ∈ [0.62, 1.00] (±0.20 around 0.82; Llama layers 9–15) **and**
seed-0 accuracy ≥ 0.55 **and** H2's rule holds (> majority + 0.05, CI lower bound > majority, permutation p < 0.05).
Otherwise: Qwen-specific (peak far from proportional expectation, or accuracy below the bar). Hard cap 3 h.

## Overlay
Layer sweeps of H2 (Qwen 1.5B, a×b 10–20), I3 times tables (Qwen 1.5B) and L (Llama 1B) on a proportional-depth x-axis.
