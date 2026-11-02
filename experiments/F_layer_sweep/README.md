# F: full layer sweep, Target A vs Target B, positional checks. Stopped at the pilot gate

## At a glance

- **Name in the write-up:** Full-protocol BBH triage (pilot failure) (Appendix A).
- **Hypothesis:** At least one of three BBH tasks passes the competence gates for the full layer-sweep protocol (none did).
- **Pre-registration:** `c67a872` · **Results:** `1ac7091` (pilot)
- **Outputs:** `pilot.json` in this directory.
- **Expected runtime:** not recorded (3 tasks × 30 items) (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/run_f_pilot.py
python scripts/plot_f_pilot.py
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

- [`preregistration.md`](preregistration.md): hypotheses, gates and decision rules (commit `c67a872`).
- [`pilot.md`](pilot.md): **no candidate task passed** (hyperbaton, tracking_shuffled_objects_three_objects,
  date_understanding). With-CoT accuracy is at most 0.533 against a required ≥ 0.85, and a constant-letter "B" bias
  of 83–100% on two tasks.
- The full experiment (N = 200, 28-layer sweep, Target A/B probes, 5 seeds, Gaussian baseline, positional checks)
  was **not run**, as pre-registered. Figures 1–3 don't exist for that reason; figure 4 (letter distribution) is
  built from pilot data.
