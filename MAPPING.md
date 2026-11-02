# Experiment letters → names used in the write-up

The write-up uses thematic names in the main text and letters only in its appendices. Every experiment lives in
`experiments/<letter>_<name>/`; the SHAs below are the commits cited in the write-up's Appendix B. Older READMEs say
"Target A" / "Target B": in the write-up these are the **no-CoT target** (the model's answer without CoT) and the
**with-CoT target** (its final answer after CoT).

| Letter | Directory | Name in the write-up | Where in the write-up | Pre-registration | Results / pilot |
|---|---|---|---|---|---|
| A | [`experiments/A_sanity/`](experiments/A_sanity/) | Load check | Appendix A (load checks) | none (environment check) | `43b910c` |
| B | [`experiments/B_probe/`](experiments/B_probe/) | The initial layer-14 null | Appendix A | none (exploratory) | `2f743fe` |
| C | [`experiments/C_turpin_baseline/`](experiments/C_turpin_baseline/) | Turpin-style cue injection | Appendix A | none (exploratory) | `6c7002d` |
| D | [`experiments/D_base_vs_instruct/`](experiments/D_base_vs_instruct/) | The externalization ("distillation") hypothesis test | Appendix A | none (exploratory) | `8b71125` |
| E | [`experiments/E_layer_sweep/`](experiments/E_layer_sweep/) | The letter-bias trap run | Section 4 (letter-bias trap), Appendix A | `411b554` | `b29a082` |
| E2 | [`experiments/E2_layer_sweep_3shot/`](experiments/E2_layer_sweep_3shot/) | BBH 3-shot pilot (pilot failure) | Appendix A | `80f4000` | `cdebe3a` (pilot) |
| F | [`experiments/F_layer_sweep/`](experiments/F_layer_sweep/) | Full-protocol BBH triage (pilot failure) | Appendix A | `5f12575` | `18957c2` (pilot) |
| G | [`experiments/G_scale_7b/`](experiments/G_scale_7b/) | 7B BBH triage (pilot failure) | Appendix A | `f3ce9d4` | `1e8231d` (load check), `905adfd` (pilot) |
| H | [`experiments/H_arithmetic/`](experiments/H_arithmetic/) | The leaky first multiplication run | Section 4 (distractor-pattern correction), Appendix D | `f3ce9d4` | `f7df7bd` (pilot), `533dbff` (results) |
| H2 | [`experiments/H2_distractor_fix/`](experiments/H2_distractor_fix/) | The multiplication benchmark (headline result) | Sections 3–4, Figures 1–2 | `7c41311` | `63dd6cb` (options-only baseline), `2ac5425` (results); exploratory: `8b0c762`, `013fdfe` |
| I | [`experiments/I_generalization/`](experiments/I_generalization/) | Multi-step task triage (pilot failure) | Section 3 (multi-step task boundary), Appendix C | `be4fee1` | `0eb4fcb` (pilot) |
| I3 | [`experiments/I3_memorizable_tasks/`](experiments/I3_memorizable_tasks/) | The multiplication-tables replication and the unit-conversions boundary | Section 3 | `5c3e806` | `b91b174` (pilot), `f516678` (times tables), `ba93e39` (unit conversions) |
| I4 | [`experiments/I4_scaling/`](experiments/I4_scaling/) | 7B on multi-step tasks (pilot failure) | Section 3 (multi-step boundary), Appendix A | `28c6a1b` | `6c6eb43` (load check), `6cf21db` (pilot) |
| J | [`experiments/J_causal_steering/`](experiments/J_causal_steering/) | The causal steering test | Section 5, Section 6 (strongest steering null), Figure 3 | `d310a5b` (+ addendum `c4a9e51`) | `00c70ac` |
| K | [`experiments/K_per_token_sweep/`](experiments/K_per_token_sweep/) | The per-token sweep | Section 5 | `9256836` | `06a8204` |
| M | [`experiments/M_mid_cot_decay/`](experiments/M_mid_cot_decay/) | The mid-CoT decay test | Section 5, Figure 4 | `6cb35be` | `da18d77` |
| N | [`experiments/N_shuffle_options/`](experiments/N_shuffle_options/) | The option-reversal test | Section 5, Figures 5–6 | `f9ab9ad` | `dbcf320` |
| L | [`scratch/experiments/L_llama_replication/`](scratch/experiments/L_llama_replication/) | Llama 3.2 1B replication | Not in the write-up: blocked at the load check (gated model), never run | `11f8dcb` | `ac436d3`, `c1ecda8` |

Verify any SHA with `git show --stat <sha>`; pre-registration commits precede their result commits in `git log`.
