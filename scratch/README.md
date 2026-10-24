# scratch/: exploration, not part of the main story

Kept for history (moved with `git mv`, so `git log --follow` still works); none of this is needed to reproduce
the results in the LessWrong post, and none of it is maintained.

| Path | What it is |
|---|---|
| `docs/scope.md` | The original project plan ("cue channels": the same wrong-answer cue delivered through different channels). The project pivoted to pre-CoT answer decoding after experiments B–E; the plan is superseded. |
| `docs/reading-notes.md` | Early reading notes. |
| `notebooks/00_setup_sanity.ipynb` | First environment sanity notebook (superseded by experiment A, `scripts/sanity_qwen.py`). |
| `scripts/run_eval_local.py`, `scripts/run_eval_sagemaker.py` | The original toy cue-channel eval (local) and an unimplemented SageMaker stub. They use `src/cue_channels/eval/`, which is kept because the tests cover it. Run from the repo root as `python scratch/scripts/run_eval_local.py`. |
| `experiments/L_llama_replication/` | Pre-registered Llama 3.2 1B replication (`07932be`). Blocked at the load check (gated model, no HF token; `34d5617`, `33f2532`); never run, not reported in the post. |
