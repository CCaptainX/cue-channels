# cue-channels

**Attention-probe analysis of CoT faithfulness: do models decide before they verbalize?**

The same wrong-answer cue is delivered through different channels (user message, few-shot
example, tool return, retrieved document). For each channel we measure:

1. **Flip rate**: how often the cue changes the answer, compared with a no-cue control.
2. **Faithfulness**: among flipped answers, how often the chain of thought admits the cue.
3. **Pre-CoT decodability**: whether a linear probe on residual-stream activations at the
   last prompt token, before any CoT token is generated, already predicts the final
   answer. If it does on unfaithful flips, the model decided before it verbalized.

See [`docs/scope.md`](docs/scope.md) for hypotheses and the full plan.

## Status

🚧 Scaffold only. The local backend works end to end on toy items. The SageMaker backend
is a stub, and the real item sets, LLM-judge monitor, and probe analysis are still TODO.

**Experiment log.** A: sanity run · B: pre-CoT probe on addition, at chance · C: Turpin few-shot bias, 0% influence ·
D: base vs Instruct, same null (distillation rejected) · E: pre-registered L24 probe on BBH logical deduction,
not supported (task at chance) · **E2 (pilot gate failed):** with BBH's standard 3-shot prompts, the 30-item no-CoT
pilot scored 0.267 (CI 0.133–0.433) against a required 0.55–0.85, with "B" chosen on 73% of items. Per the
pre-registration the full run was not executed. See `experiments/*/README.md`.

## Quickstart

```bash
make setup                 # creates .venv/ (project-local) and installs deps
cp .env.example .env       # add HF_TOKEN if you use gated models (Llama)
make test                  # fast tests, no weight download
make eval-local            # toy eval on Qwen2.5-1.5B-Instruct (downloads ~3 GB)
make eval-local MODEL=smollm2-1.7b
```

## Models

| key | HF id | notes |
|---|---|---|
| `qwen2.5-1.5b` (default) | `Qwen/Qwen2.5-1.5B-Instruct` | small, CoT-capable, TransformerLens-supported |
| `smollm2-1.7b` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | backup |
| `llama3.2-1b` | `meta-llama/Llama-3.2-1B-Instruct` | gated; needs `HF_TOKEN` |

To swap models, change `DEFAULT_MODEL` in `src/cue_channels/config.py`, set `CUE_MODEL=…`,
or pass `--model`.

## Layout

```
src/cue_channels/
  config.py          model registry + RunConfig
  models.py          ModelBackend protocol; LocalBackend (MPS/CUDA/CPU); SageMakerBackend stub
  eval/cue_tasks.py  channels, cue types, prompt builders, answer parsing
  eval/runner.py     item × channel × cue loop, records + activation cache, summary
  probes/train.py    cross-validated linear probes per layer
scripts/             run_eval_local.py, run_eval_sagemaker.py
notebooks/           00_setup_sanity.ipynb
tests/               fast tests (`-m slow` for ones that download weights)
```

## Reading list

- Turpin et al. 2023, *Language Models Don't Always Say What They Think* (arXiv:2305.04388)
- Lanham et al. 2023, *Measuring Faithfulness in Chain-of-Thought Reasoning* (arXiv:2307.13702)
- Chen et al. 2025, *Reasoning Models Don't Always Say What They Think* (arXiv:2505.05410)
- Gema et al., FACE-Eval (arXiv:2608.29464)
- Duzan & Cooper Stickland (arXiv:2608.04735)
- Alain & Bengio 2016, *Understanding intermediate layers using linear classifier probes* (arXiv:1610.01644)

Notes: [`docs/reading-notes.md`](docs/reading-notes.md).
