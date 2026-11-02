# Experiment letters → names used in the write-up

The write-up uses thematic names in the main text and letters only in its appendices. Every experiment lives in
`experiments/<letter>_<name>/`; the SHAs below are the commits cited in the write-up's Appendix B. Older READMEs say
"Target A" / "Target B": in the write-up these are the **no-CoT target** (the model's answer without CoT) and the
**with-CoT target** (its final answer after CoT).

| Letter | Directory | Name in the write-up | Where in the write-up | Pre-registration | Results / pilot |
|---|---|---|---|---|---|
| A | [`experiments/A_sanity/`](experiments/A_sanity/) | Load check | Appendix A (load checks) | none (environment check) | `23ebbd7` |
| B | [`experiments/B_probe/`](experiments/B_probe/) | The initial layer-14 null | Appendix A | none (exploratory) | `d72b1c4` |
| C | [`experiments/C_turpin_baseline/`](experiments/C_turpin_baseline/) | Turpin-style cue injection | Appendix A | none (exploratory) | `a0343ab` |
| D | [`experiments/D_base_vs_instruct/`](experiments/D_base_vs_instruct/) | The externalization ("distillation") hypothesis test | Appendix A | none (exploratory) | `6f89c61` |
| E | [`experiments/E_layer_sweep/`](experiments/E_layer_sweep/) | The letter-bias trap run | Section 4 (letter-bias trap), Appendix A | `8647c89` | `bde7f3e` |
| E2 | [`experiments/E2_layer_sweep_3shot/`](experiments/E2_layer_sweep_3shot/) | BBH 3-shot pilot (pilot failure) | Appendix A | `c08b6fb` | `daa0a27` (pilot) |
| F | [`experiments/F_layer_sweep/`](experiments/F_layer_sweep/) | Full-protocol BBH triage (pilot failure) | Appendix A | `6fc116d` | `99229ed` (pilot) |
| G | [`experiments/G_scale_7b/`](experiments/G_scale_7b/) | 7B BBH triage (pilot failure) | Appendix A | `43d5977` | `f1b6026` (load check), `29020e1` (pilot) |
| H | [`experiments/H_arithmetic/`](experiments/H_arithmetic/) | The leaky first multiplication run | Section 4 (distractor-pattern correction), Appendix D | `43d5977` | `c2606e7` (pilot), `0c5d7f5` (results) |
| H2 | [`experiments/H2_distractor_fix/`](experiments/H2_distractor_fix/) | The multiplication benchmark (headline result) | Sections 3–4, Figures 1–2 | `f825941` | `ada1969` (options-only baseline), `38e5999` (results); exploratory: `3fbf46d`, `b8c9abd` |
| I | [`experiments/I_generalization/`](experiments/I_generalization/) | Multi-step task triage (pilot failure) | Section 3 (multi-step task boundary), Appendix C | `44425f0` | `a5389fe` (pilot) |
| I3 | [`experiments/I3_memorizable_tasks/`](experiments/I3_memorizable_tasks/) | The multiplication-tables replication and the unit-conversions boundary | Section 3 | `02fb86d` | `4a6498d` (pilot), `2787627` (times tables), `60ff04f` (unit conversions) |
| I4 | [`experiments/I4_scaling/`](experiments/I4_scaling/) | 7B on multi-step tasks (pilot failure) | Section 3 (multi-step boundary), Appendix A | `176b201` | `56784a0` (load check), `58c36f0` (pilot) |
| J | [`experiments/J_causal_steering/`](experiments/J_causal_steering/) | The causal steering test | Section 5, Section 6 (strongest steering null), Figure 3 | `55b6863` (+ addendum `470a541`) | `30f15cc` |
| K | [`experiments/K_per_token_sweep/`](experiments/K_per_token_sweep/) | The per-token sweep | Section 5 | `1d8ad88` | `7002b47` |
| M | [`experiments/M_mid_cot_decay/`](experiments/M_mid_cot_decay/) | The mid-CoT decay test | Section 5, Figure 4 | `77ae121` | `463b0e1` |
| N | [`experiments/N_shuffle_options/`](experiments/N_shuffle_options/) | The option-reversal test | Section 5, Figures 5–6 | `1f46016` | `2c56b83` |
| L | [`scratch/experiments/L_llama_replication/`](scratch/experiments/L_llama_replication/) | Llama 3.2 1B replication | Not in the write-up: blocked at the load check (gated model), never run | `9c0f3b0` | `4a9ebab`, `fde627d` |

Verify any SHA with `git show --stat <sha>`; pre-registration commits precede their result commits in `git log`.
