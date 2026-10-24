# I4: 7B on multi-step tasks (pilot failure)

## At a glance

- **Name in the LW post:** 7B on multi-step tasks (pilot failure) (Section 3 (multi-step boundary), Appendix A).
- **Hypothesis:** Qwen 2.5 7B Instruct passes the competence gates on I's multi-step tasks (it did not).
- **Pre-registration:** `a54d1a6` · **Results:** `aaa65aa` (load check), `3e99734` (pilot)
- **Outputs:** `pilot.json`, `load_check.json` in this directory.
- **Expected runtime:** about 32 min (15.5 GB model) (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_gh.py load-check --model qwen2.5-7b --out experiments/I4_scaling
python scripts/run_gh.py pilot --model qwen2.5-7b --suite i --out experiments/I4_scaling
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Details: [`preregistration.md`](preregistration.md), [`pilot.md`](pilot.md).
