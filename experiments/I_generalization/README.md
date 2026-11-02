# I: Multi-step task triage (pilot failure)

## At a glance

- **Name in the write-up:** Multi-step task triage (pilot failure) (Section 3 (multi-step task boundary), Appendix C).
- **Hypothesis:** H2's signal generalizes to other tasks the model is competent at (no candidate passed the gates).
- **Pre-registration:** `44425f0` · **Results:** `a5389fe` (pilot)
- **Outputs:** `pilot.json`, `pattern_baseline.json` in this directory.
- **Expected runtime:** about 6 min (AQuA data: see pilot.md) (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/i_pattern_baseline.py
python scripts/run_gh.py pilot --model qwen2.5-1.5b --suite i --out experiments/I_generalization
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Details: [`preregistration.md`](preregistration.md), [`pilot.md`](pilot.md).
