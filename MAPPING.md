# Experiment letters → names used in the write-up

The write-up uses thematic names in the main text and letters only in its appendices. Every experiment lives in
`experiments/<letter>_<name>/`; the SHAs below are the commits cited in the write-up's Appendix B. Older READMEs say
"Target A" / "Target B": in the write-up these are the **no-CoT target** (the model's answer without CoT) and the
**with-CoT target** (its final answer after CoT).

| Letter | Directory | Name in the write-up | Where in the write-up | Pre-registration | Results / pilot |
|---|---|---|---|---|---|
| A | [`experiments/A_sanity/`](experiments/A_sanity/) | Load check | Appendix A (load checks) | none (environment check) | `8344f88` |
| B | [`experiments/B_probe/`](experiments/B_probe/) | The initial layer-14 null | Appendix A | none (exploratory) | `d2e16f1` |
| C | [`experiments/C_turpin_baseline/`](experiments/C_turpin_baseline/) | Turpin-style cue injection | Appendix A | none (exploratory) | `ddf0053` |
| D | [`experiments/D_base_vs_instruct/`](experiments/D_base_vs_instruct/) | The externalization ("distillation") hypothesis test | Appendix A | none (exploratory) | `c01d6c7` |
| E | [`experiments/E_layer_sweep/`](experiments/E_layer_sweep/) | The letter-bias trap run | Section 4 (letter-bias trap), Appendix A | `0f7353a` | `2bc86bc` |
| E2 | [`experiments/E2_layer_sweep_3shot/`](experiments/E2_layer_sweep_3shot/) | BBH 3-shot pilot (pilot failure) | Appendix A | `c918f6e` | `f66ebcb` (pilot) |
| F | [`experiments/F_layer_sweep/`](experiments/F_layer_sweep/) | Full-protocol BBH triage (pilot failure) | Appendix A | `c67a872` | `1ac7091` (pilot) |
| G | [`experiments/G_scale_7b/`](experiments/G_scale_7b/) | 7B BBH triage (pilot failure) | Appendix A | `c1f8159` | `00dcea8` (load check), `b163ee6` (pilot) |
| H | [`experiments/H_arithmetic/`](experiments/H_arithmetic/) | The leaky first multiplication run | Section 4 (distractor-pattern correction), Appendix D | `c1f8159` | `f0c44ec` (pilot), `8cb14c3` (results) |
| H2 | [`experiments/H2_distractor_fix/`](experiments/H2_distractor_fix/) | The multiplication benchmark (headline result) | Sections 3–4, Figures 1–2 | `1ee5528` | `db4fad4` (options-only baseline), `0b9dd69` (results); exploratory: `647b901`, `c37dc63` |
| I | [`experiments/I_generalization/`](experiments/I_generalization/) | Multi-step task triage (pilot failure) | Section 3 (multi-step task boundary), Appendix C | `8a6de72` | `e094845` (pilot) |
| I3 | [`experiments/I3_memorizable_tasks/`](experiments/I3_memorizable_tasks/) | The multiplication-tables replication and the unit-conversions boundary | Section 3 | `d5d1be1` | `bbc4706` (pilot), `a13d731` (times tables), `af736fd` (unit conversions) |
| I4 | [`experiments/I4_scaling/`](experiments/I4_scaling/) | 7B on multi-step tasks (pilot failure) | Section 3 (multi-step boundary), Appendix A | `a54d1a6` | `aaa65aa` (load check), `3e99734` (pilot) |
| J | [`experiments/J_causal_steering/`](experiments/J_causal_steering/) | The causal steering test | Section 5, Section 6 (strongest steering null), Figure 3 | `d9bbb0f` (+ addendum `37676ba`) | `e9882de` |
| K | [`experiments/K_per_token_sweep/`](experiments/K_per_token_sweep/) | The per-token sweep | Section 5 | `56b863a` | `c48a077` |
| M | [`experiments/M_mid_cot_decay/`](experiments/M_mid_cot_decay/) | The mid-CoT decay test | Section 5, Figure 4 | `3742d74` | `32bef59` |
| N | [`experiments/N_shuffle_options/`](experiments/N_shuffle_options/) | The option-reversal test | Section 5, Figures 5–6 | `ecb491d` | `db62ccb` |
| L | [`scratch/experiments/L_llama_replication/`](scratch/experiments/L_llama_replication/) | Llama 3.2 1B replication | Not in the write-up: blocked at the load check (gated model), never run | `07932be` | `34d5617`, `33f2532` |

Verify any SHA with `git show --stat <sha>`; pre-registration commits precede their result commits in `git log`.
