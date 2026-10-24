# A: Load check

## At a glance

- **Name in the LW post:** Load check (Appendix A (load checks)).
- **Hypothesis:** Qwen 2.5 1.5B Instruct loads on MPS, generates a CoT, and exposes residual-stream activations.
- **Pre-registration:** none (environment check) · **Results:** `8344f88`
- **Outputs:** `summary.json` in this directory.
- **Expected runtime:** under a minute once the model is cached (Apple M4 Pro, MPS, model already downloaded).

Reproduce from the repo root (after `make setup`):

```bash
python scripts/sanity_qwen.py
```

Re-running with the default paths overwrites the committed outputs here; `git diff` then shows any change.

