# G: 7B BBH triage (pilot failure)

## At a glance

- **Name in the write-up:** 7B BBH triage (pilot failure) (Appendix A).
- **Hypothesis:** Qwen 2.5 7B Instruct passes the BBH competence gates that 1.5B failed (it did not; pre-registered pivot to H).
- **Pre-registration:** `c1f8159` · **Results:** `00dcea8` (load check), `b163ee6` (pilot)
- **Outputs:** `pilot.json`, `load_check.json` in this directory.
- **Expected runtime:** about 40 min (15.5 GB model) (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_gh.py load-check --model qwen2.5-7b --out experiments/G_scale_7b
python scripts/run_gh.py pilot --model qwen2.5-7b --suite bbh --out experiments/G_scale_7b
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Details: [`preregistration.md`](preregistration.md), [`pilot.md`](pilot.md).
