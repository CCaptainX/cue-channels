# I3: The multiplication-tables replication and the unit-conversions boundary

## At a glance

- **Name in the write-up:** The multiplication-tables replication and the unit-conversions boundary (Section 3).
- **Hypothesis:** On one-pass-retrievable tasks the no-CoT target peaks in L19–25 with accuracy ≥ 0.55.
- **Pre-registration:** `5c3e806` · **Results:** `b91b174` (pilot), `f516678` (times tables), `ba93e39` (unit conversions)
- **Outputs:** `times_tables/metrics.json`, `unit_conversions/metrics.json` in this directory.
- **Expected runtime:** about 3 min pilot + 11 min (times tables) + 22 min (unit conversions) (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/i3_pattern_baseline.py
python scripts/run_gh.py pilot --model qwen2.5-1.5b --suite i3 --out experiments/I3_memorizable_tasks
python scripts/run_full_f_protocol.py --model qwen2.5-1.5b --suite i3 --task times_tables --grouped --shifted option_e --out experiments/I3_memorizable_tasks/times_tables
python scripts/run_full_f_protocol.py --model qwen2.5-1.5b --suite i3 --task unit_conversions --grouped --shifted option_e --out experiments/I3_memorizable_tasks/unit_conversions
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

Details: [`preregistration.md`](preregistration.md), [`pilot.md`](pilot.md).
