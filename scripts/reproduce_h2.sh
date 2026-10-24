#!/usr/bin/env bash
# Reproduce the headline result (H2, the multiplication benchmark) from a fresh clone.
#
#   bash scripts/reproduce_h2.sh [OUT_DIR]        # default OUT_DIR=results/h2_repro
#
# Steps: (1) options-only baseline, (2) full probe protocol: generate the 200 items, run the model
# with and without CoT, cache last-token activations, fit probes at all 28 layers x 5 seeds, permutation
# test, (3) compare against the committed experiments/H2_distractor_fix/metrics.json.
# Downloads Qwen/Qwen2.5-1.5B-Instruct (~3 GB) on first use. About 25 min on an Apple M4 Pro (MPS).
# Writes only to OUT_DIR; committed results are never overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/h2_repro}"
PY="${PY:-.venv/bin/python}"
mkdir -p "$OUT"
echo "[1/3] options-only baseline"
"$PY" scripts/h2_pattern_baseline.py --out "$OUT" > "$OUT/pattern_baseline.log"
echo "[2/3] full probe protocol (collection ~12 min, analysis ~9 min)"
"$PY" scripts/run_full_f_protocol.py --model qwen2.5-1.5b --suite arith --task C2 --out "$OUT" > "$OUT/protocol.log"
echo "[3/3] compare with committed results"
"$PY" scripts/compare_h2.py experiments/H2_distractor_fix "$OUT"
