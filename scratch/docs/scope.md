# Artifact Scope v1: P1 CoT-Faithfulness Eval Suite

**Date:** 2026-10-05 · **Status:** draft, novel axis undecided · **Sources:** `artifact_pivot_options.md` (P1), `artifact_projects_detailed.md` (P1)

---

## 1. Title + pitch

**Cues Through Tools: an open CoT-faithfulness eval suite, with a novel-axis extension**

I'll build an inspect-ai eval suite that puts the same cue into different delivery channels (user message, few-shot example, tool return, retrieved document). For each open-weight model it measures whether the chain of thought admits the cue drove the answer, and whether a CoT monitor catches it. Agents get most of their influence through tools and retrieval, but most faithfulness tests only use the user message. This suite maps the channel × cue × model space that FACE-Eval opened. One extension (Option 3 or 4) then asks *why* faithfulness varies.

---

## 2. Pre-registered hypotheses

The thresholds below are rough. They get frozen in `PREREG.md` (committed and timestamped) after the W1 pilot and before any full runs.

- **H1 (detection baseline):** On the user-message channel, a CoT monitor flags cue-influenced answers with F1 roughly ≥ 0.7 on the 7–8B instruct models.
- **H2 (channel effect):** Faithfulness (the share of cue-flipped answers whose CoT mentions the cue) is lower for tool-return and retrieved-doc cues than for user-message cues. The drop holds in ≥ 3 of 4 baseline models. Few-shot cues fall between the two.
- **H3a (if Option 3):** On unfaithful flips, a linear probe on residual-stream activations *before the first CoT token* predicts the final answer clearly above chance. On faithful flips it predicts the answer less well. That pattern would mean the decision is formed before the CoT, not in it.
- **H3b (if Option 4):** Faithfulness on non-user channels differs between base and instruct checkpoints of the same family. The direction is not pre-specified, and either direction counts as a finding.

**Falsification:** H2 fails if the channel ordering isn't consistent across models (the noise risk the shortlist flags).

---

## 3. Baseline scope (P1 as shortlisted)

**Models:** a 4-model core panel, plus a stretch model.
- Llama 3.1 8B Instruct
- Qwen 2.5 7B Instruct
- Qwen 2.5 32B Instruct
- DeepSeek-R1-Distill-Llama-8B. The shortlist targets *reasoning* models, so at least one belongs in the panel.
- Stretch: Llama 3.1 70B, 4-bit quantized, run only on the headline cells. GPT-OSS is optional if time allows.

**Harness:** inspect-ai tasks with a simple sandboxed tool environment (a mock search tool and a mock file read) for the tool and retrieval channels.

**Cue injection:**
- Channels: user message, few-shot example, tool return, retrieved doc. Memory is a stretch.
- Cue types: preference hint, authority/"expert says", and implicit (inferable from data).
- Each cue points to a wrong answer on MCQ-style items, so a flip is unambiguous.

**Metrics:**
- Flip rate (cue vs. no-cue control)
- Faithfulness: verbalization rate among flips
- Monitor F1 (LLM-judge monitor)
- Bootstrap CIs, with ≥ 3 seeds or paraphrase sets per cell

**Deliverables:**
- Public repo with reusable tasks and a README
- Blog post built around a channel × model heatmap
- Short preprint if the results are crisp

**Anchors:**
- Gema et al. (arXiv:2608.29464, FACE-Eval)
- Duzan & Cooper Stickland (arXiv:2608.04735)

---

## 4. Novel-axis options (pick one)

### Option 3: Interp extension (Anthropic-max signal)

**Question:** Is the cue-driven decision formed *in* the CoT text, or is it already in the hidden state before the CoT starts?

**Method:**
1. Use the base eval to label examples as faithful flips, unfaithful flips or no-flips.
2. Cache residual-stream activations at the last prompt token across layers.
3. Train linear probes to predict the final answer and whether the cue flipped it.
4. Compare probe accuracy across the three label groups.
5. Run activation patching (cue run → clean run) to localize where the flip enters.

**Deliverable:** a separate `interp/` directory in the repo, plus its own blog post or paper subsection.

**Anchors:**
- Wang et al. (arXiv:2609.23065): SAE concepts plus causal interventions to test whether verbalized reasoning drives answers. This is the closest prior work.
- Nguyen et al. (arXiv:2607.29062): steering vectors for faithfulness.

**Cost:** +2–3 weeks, about 50 A100-hours. Run it on the 7–8B models only.

### Option 4: Training-stage comparison (OpenAI-max signal)

**Question:** How do training stages change CoT faithfulness?

**Method:** Run the same eval on several checkpoints of one model family:
- Llama 3.1 8B: base → Instruct → R1-Distill (a reasoning-SFT stage)
- Qwen 2.5 7B: base → Instruct

**Caveats:**
- Neither family releases a separate RLHF-only checkpoint. That gives 2–3 stage points, not a clean per-stage ablation.
- Base models need completion-format prompts and handle tool turns poorly, so prompt format is confounded with training stage. Mitigation: a format-matched user-message control.
- The pitch cites recent OpenAI Model Behavior work on this. That paper isn't in the verified shortlist, so I need to find and verify it before citing it publicly.

**Anchor:** Nguyen et al. (arXiv:2607.29062) found faithfulness steering reliable only at the largest scale tested. That suggests stage and scale matter.

**Cost:** +1–2 weeks, about 30 A100-hours.

### Comparison

| | Option 3 (interp) | Option 4 (training stage) |
|---|---|---|
| Anthropic signal | **Very high** (interp + eval together) | Medium |
| OpenAI signal | Medium-high | **High** |
| Null risk | Medium. Probes usually decode something; the risk is a trivial result, so controls are needed. | Medium. Only 2–3 stage points, and the format confound may swamp the effect. |
| Null still publishable? | Yes ("decision not detectable pre-CoT") | Yes ("training stage doesn't move channel faithfulness") |
| Fits 6 weeks? | Tight. Probes are the core; patching is a stretch. | Yes |
| Personal fit | Builds on the IOI / TransformerLens reproduction | Purely eval-engineering |

**Recommendation: Option 3, scoped as probes-first with patching as a stretch goal.** It has a verified close neighbour (Wang et al.), it reuses the TransformerLens skills from the IOI reproduction, and it asks a sharper question. Option 4 is the safer pick if the Nov 15 date is fixed.

---

## 5. Milestones (6 weeks, ship Nov 15)

**W1 (Oct 5–11): scaffold**
- Repo, CI, and the vLLM serving script
- inspect-ai hello-world task
- Reproduce one FACE-Eval-style user-message vs. tool-return comparison on one model
- Draft `PREREG.md`

**W2 (Oct 12–18): first results**
- All 4 channels × 3 cue types on Llama 3.1 8B
- Monitor-judge prompt, plus a hand-label check of about 50 items
- Freeze and commit `PREREG.md`
- **Kill checkpoint**

**W3 (Oct 19–25): full panel**
- Baseline panel complete, with seeds and paraphrases
- Novel-axis kickoff. Option 3: activation-caching pipeline. Option 4: base-checkpoint prompt adapters.

**W4 (Oct 26–Nov 1): novel axis**
- Option 3: probe training, plus patching on 1 model if time allows. Option 4: full stage sweep.
- First full results and heatmaps
- **Kill checkpoint**

**W5 (Nov 2–8): write-up**
- Blog draft and figures
- README and a one-command repro script
- Clean up repo history
- Ask 1–2 people for a read-through

**W6 (Nov 9–15): launch**
- Final edits
- Repo goes public
- Public write-up and Twitter thread
- Preprint only if the results are crisp

---

## 6. Tools, models, compute

**Tools:**
- inspect-ai (primary)
- vLLM for serving
- HuggingFace transformers
- TransformerLens (Option 3 only; nnsight as the fallback for 32B+)

**Models:**
- Llama 3.1 8B / 70B (70B is stretch)
- Qwen 2.5 7B / 32B
- DeepSeek-R1-Distill-Llama-8B
- Option 4 adds the base checkpoints.

**Compute** (RunPod A100):
- Option 4: about $200–350
- Option 3: about $300–500

**Repo:** private on GitHub until W6.

---

## 7. Success + kill criteria

**Success:** By Nov 15, the public repo, blog post and reproducible eval are out. The headline finding is how faithfulness is distributed across cue channels.

**Partial success:** The eval ships but the novel axis is null. That is still publishable as "we tested this specifically and it didn't change."

**Kill / re-scope:**
- **End of W2:** If the base eval isn't running end-to-end on Llama 3.1 8B, drop to 2 channels and 2 models.
- **End of W4:** If the novel axis is clearly ship-nothing, drop it and ship the base eval only.
- **Any time:** If no channel effect replicates on *any* model, reframe as a negative-result note (per the shortlist).

---

## 8. Open questions

1. **Novel axis:** Option 3, Option 4, or neither (pure base scope)?
2. **Timeline:** Does Nov 15 hold? The base scope alone is about 50–70 h, roughly 5–6 weeks at 10–12 h/week. Option 3 probably needs either Nov 29 or a probes-only scope.
3. **Model focus:** Llama, Qwen, or mixed (current default: mixed)?
4. **API models:** Include 1–2 API models, as the shortlist suggested, or keep it fully open-weight?
