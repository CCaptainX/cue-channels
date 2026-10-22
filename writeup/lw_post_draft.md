# Readable before the chain of thought, but used? Pre-CoT answer decoding where the CoT does real work


*By CCaptainX. Draft v3 — October 21, 2026. Code and pre-registrations: [github.com/CCaptainX/cue-channels](https://github.com/CCaptainX/cue-channels) (private, access on request).*

## TL;DR

Does a model know its answer before it starts reasoning out loud? Cox et al. (2026) showed it is often decodable from pre-CoT activations, and that steering along that direction changes it. I ran a pre-registered replication on Qwen 2.5 1.5B Instruct, one of their models, on two-digit multiplication, where the chain of thought visibly does work. The pre-CoT answer is linearly readable, as they found. Four pre-registered tests of where the signal comes from and whether the model uses it came back not supported, mixed or inconclusive; even steering close to Cox et al.'s protocol, strong enough to break the model, did not move its answer. Post-hoc analyses fit a hypothesis, "readable but not committed": the signal drops out once the CoT begins and returns after the arithmetic. The options-only baseline caught a distractor leak every other check missed; I'd treat it as standard for any MCQ probing with constructed distractors. 'Readable but not causally used' is one hypothesis consistent with the data; redundant encoding, non-linearity, or use at earlier layers remain live alternatives. Scope: one model, one task family.

## 1. Introduction and motivation

A central question in CoT faithfulness is whether a model's final answer is *computed during* the chain of thought, or was already determined before verbalization, with the CoT acting as a post-hoc rationalization. Behavioral work tested this from outside (Lanham et al. 2023; Turpin et al. 2023; Chen et al. 2025), and on multi-step arithmetic Kudo et al. (2026) found that models compute the answer while generating the reasoning. Whether CoT tokens do serial work depends on the setting: filler tokens can stand in for a CoT on some synthetic tasks (Pfau, Merrill & Bowman 2024), stepwise training can internalize it into hidden states (Deng, Choi & Shieber 2024), and Jia & Mu (2026) find that CoT becomes more load-bearing as task difficulty rises. Cox, Kianersi & Garriga-Alonso (2026) asked the mechanistic version. Across five instruction-tuned Gemma 2 and Qwen 2.5 models (including Qwen 2.5 1.5B) and four binary tasks, a probe direction at the last token before the CoT predicts the final answer with >0.9 AUC on most model–task pairs, and steering along it flips answers well beyond norm-matched orthogonal controls. On those tasks, removing the CoT rarely changed the answer: the CoT was largely post-hoc.

This post replicates their probe on one of their models, Qwen 2.5 1.5B Instruct, in a format they did not test: generated five-option arithmetic. I find the same kind of pre-CoT readability; whether the model uses it is the harder question, and there my pre-registered tests did not come back cleanly.

## 2. Methodology

**Prompt format.** Each item follows a one-line instruction and three worked examples: a question, five labelled options, and an "A:" answer marker (short for "Answer:") followed by "Let's think step by step." The real item for 11 × 15:

```
Q: What is 11 * 15?
Options:
(A) 165
(B) 160
(C) 210
(D) 266
(E) 340
A: Let's think step by step.
```

Two positions recur: the "(E" token that begins option E inside the option list (the option-E-start token; nothing about the letter matters, it is simply the start of the last option before the answer marker), and the "A:" answer marker after the options. "The final prompt token" always means the closing "." of "step by step.", immediately before generation.

**Probe.** For each item I take residual-stream activations at every layer (28 blocks, hidden size 1536) at the final prompt token and fit a standardized, L2-regularized multinomial logistic regression (C = 0.1) with 5-fold cross-validation, so every item is predicted once. 95% CIs come from 2,000 bootstrap resamples of per-item correctness; each cell is repeated with 5 seeds.

**Per-position retrained probes.** For the positional and mid-CoT analyses I also retrain a probe at each token position: applying the last-token probe elsewhere asks whether the *same direction* carries the answer; a retrained probe asks whether the *information* is linearly present at all.

**Targets.** The *no-CoT target* is the model's answer when prompted to answer directly, without CoT; the *with-CoT target* is its final answer with CoT (parsed from "So the answer is (X)").

**Baselines.** Every accuracy is reported next to chance (1/K), the majority-class rate of that target's labels, and the same probe trained on Gaussian noise. Peak layers are chosen from 28, so verdicts also have to survive a max-over-layers permutation test.

**Pre-registration.** Every experiment in the main body had its hypotheses and decision rules committed to git before any data was collected; the commit log is the audit trail (Appendix B). Three confound controls are part of every pre-registration: the majority-letter baseline, an options-only baseline, and positional sanity checks (Section 4). Where a registered rule returns not-supported, mixed or inconclusive, that verdict is the result; anything further is labelled post-hoc.

## 3. Main result: the answer is readable before the CoT

**The multiplication benchmark.** Each item asks "What is a × b?" with a, b drawn uniformly from 10–20. Each distractor is another product a′ × b′ from the same range, so the options carry no structural tell. On 200 items, no-CoT accuracy is 0.740 and with-CoT accuracy 0.985.

- **No-CoT target:** peak at **layer 22, 0.705 [0.645, 0.765]** against a majority baseline of 0.280; 5-seed mean 0.730 ± 0.019; p < 0.01 under the registered 100-permutation max-over-layers null; a post-hoc 1,000-permutation rerun gives p = 0.001 (null maximum 0.380).
- **With-CoT target:** peak at **layer 21, 0.667 [0.596, 0.727]** against chance of 0.20; 5-seed mean 0.674 ± 0.013; p < 0.01 (post-hoc rerun: p = 0.001, null maximum 0.338).
- Gaussian-noise probe: 0.246 (no-CoT) and 0.236 (with-CoT).

(Figure 1: Pre-CoT probe accuracy by layer on the multiplication benchmark. Both targets sit near baseline through layer 19 and jump at layers 20–21; chance and majority-baseline lines shown, peaks annotated.)

The plateau sits close to the model's own no-CoT accuracy (0.74). The probe reads the model's answer, not the true one: on the 52 items the model gets wrong without CoT, it recovers the model's wrong letter on 61.5% of them and the correct letter on 26.9% (exploratory).

**Scope: one replication and two boundaries.**

*The multiplication-tables replication.* The same finding holds on times tables with a, b ∈ 2–12 (no-CoT 0.855, with-CoT 1.000). The no-CoT-target probe peaks at layer 21 with 0.780 [0.720, 0.835] against a majority of 0.230, with the same jump around layers 19–21.

*The unit-conversions boundary observation.* On unit conversions ("How many meters are in 9 kilometers?"; no-CoT 0.545), the probe at the final prompt token reaches only 0.465, short of its pre-registered success bar, yet at the option-E-start token it reads 0.755. The answer is present, but mostly not at the final token (Section 5).

*The multi-step task boundary.* On tasks the model needs the CoT to solve at all — carry addition, modular arithmetic, easy GSM8K items — no-CoT accuracy is near chance (0.23–0.43, even for Qwen 2.5 7B), so there is no one-pass answer to decode. These tasks are untested, not negative. See Appendix C for a target-by-target analysis of what the probe would and would not reveal on these tasks.

## 4. Validation stack

**The letter-bias trap.** An early pre-registered probe on a BIG-Bench Hard logic task reached 0.731, which looked like a result until the majority baseline came back: the model answered "B" on 77% of items, so "always B" scored 0.769, and the model was at chance on the task itself. The probe was decoding letter bias. Sosis & Gavenčiak (2025) flagged the same hazard in small Qwen1.5 models. Cox et al.'s binary tasks randomize option content; in a five-option format I report the majority-letter baseline next to every probe accuracy.

**The distractor-pattern methodology correction.** My first positive run on multiplication passed every pre-registered check and still had a flaw. Its distractors were the correct answer ± 1, ± 2 or ± 10, so the correct option is the one value the others are offset from, a rule that identifies it on 200/200 items without reading the question. A probe at the option-E-start token decoded the with-CoT answer at 0.835, suspiciously above the final token. I replaced the distractors with other products from the same range and made an options-only baseline part of every pre-registration: a classifier that sees only the option values scores 1.000 on the old distractors and 0.206 on the new ones (chance 0.20). The headline barely moved (0.695 → 0.705); the option-E-start advantage disappeared. I would treat an options-only baseline as standard for any MCQ probing with constructed distractors. A concrete worked example and the full options-only baseline numbers are in Appendix D.

(Figure 2: Options-only classifier accuracy: 1.000 on the original ±1/±2/±10 distractors, 0.206 on the corrected ones; chance 0.20.)

**Positive controls, noise and seeds.** Probes on the same activations read facts written in the prompt at 0.93–1.00; probes given only positional features score at the majority rate; the permutation null's 95th percentile is 0.335 (no-CoT) and 0.298 (with-CoT); seed-to-seed SD across all 28 layers × 2 targets is 0.005–0.046.

## 5. Is the pre-CoT signal causal? Pre-registered verdicts first

I pre-registered four tests of where the signal comes from and whether the model uses it. None fired "supported": the causal steering test came back not supported, the mid-CoT decay test returned its registered "signal was always noise" verdict, the option-reversal test came back mixed, and the per-token sweep was inconclusive under its 0.10-gap rule. Four pre-registered rules, zero cleanly supported hypotheses; the rest of each paragraph is post hoc.

**The causal steering test (registered verdict: not supported).** Following Rimsky et al. (2023), I built a steering vector at layer 22, last prompt token: the mean activation of the 18 items whose no-CoT answer is option (A) minus the mean of the other 182 (‖v‖ = 3.72 against a residual norm of ~116). On 100 held-out items I added α·v and recorded the CoT answer. The prediction failed: 0.230 "A" answers at α = 0 and 0.230 at α = +2, flat across α ∈ {−2, −1, +1, +2, +5}, whether I steered only the last prompt token or every generated token. CoTs stayed coherent and 99% correct. An exploratory addendum at α = +20 also left answers unmoved (Section 6). This is a null for one construction: an 18-item mean difference, one letter, one layer.

| Condition (100 held-out items) | share answering "A" | Δ vs α = 0 | with-CoT accuracy |
|---|---|---|---|
| α = 0 | 0.230 | — | 0.990 |
| α = +2, last prompt token (primary) | 0.230 | 0.000 | 0.990 |
| α = +5, every token | 0.230 | 0.000 | 1.000 |
| α = +20, every token (exploratory) | 0.200 | −0.030 | 0.740 |
| random direction, α = +20, every token | 0.180 | −0.050 | 0.870 |

(Figure 3: Causal steering test. The answer distribution stays flat across α; only at α = +20 on every token does accuracy fall, with no shift toward A.)

**The per-token sweep (registered verdict: not confirmed).** The rule: each task's best position had to beat that layer's accuracy at the other task's best position by ≥ 0.10. The margins were 0.04 for multiplication and 0.085 for unit conversions. *Post hoc,* both tasks share one pattern. The letter is undecodable before the options and, from about layer 18, becomes readable as each option is read, through the "A:" answer marker. Multiplication peaks at the answer marker (0.765), unit conversions at the option-E-start token (0.785). At the final prompt token multiplication still carries the answer (0.70) and unit conversions mostly do not (0.47), which explains the unit-conversions boundary.

**The mid-CoT decay test (registered verdict: signal was always noise).** On the same 100 held-out items, the registered rules applied to the last-token probe at every generated CoT token (layer 22). It reads 0.62 before the CoT and falls below the majority rate (0.32) from the first CoT token on (0.16 at token 1). By the registered rule that is "signal was always noise", and that is the result. *Post hoc,* probes retrained at each position read the letter at 0.58 before the CoT, at the majority rate while the model writes out the multiplication (0.30 at the first "=", 0.32 just before the product), 0.48 once the product is written and 0.61 at "answer is (". That is consistent with recomputation, but Section 6 lists readings it does not exclude.

(Figure 4: Mid-CoT decay test. The registered last-token probe collapses at the first CoT token; post-hoc retrained probes dip to majority during the arithmetic and rebuild after the product.)

**The option-reversal test (registered verdict: mixed).** I reversed each item's option order (slot c → 4 − c) and compared peak positions across the two orders. The registered regression of peak shift on correct-option shift gave R² = 0.29, slope 0.44: the "mixed" band. *Post hoc:* the population maps barely change under reversal and peak at the answer marker in both orders (0.745 / 0.795), and an item's peak falls at or after its own correct option on 93% of items against an 84% permutation baseline (p ≤ 0.001). A 9-point gap over that baseline suggests, but does not establish, that the signal is partly content-dependent and partly structural.

(Figure 5: Option-reversal test. Bottom: the registered shift-vs-shift regression. Top, post hoc: per-item peak position against the correct option's slot in both orders.)

(Figure 6: Option-reversal test, post hoc. Population probe accuracy by position × layer in original and reversed order; the original panel is the per-token sweep's multiplication map.)

## 6. Interpretation: a hypothesis, not a conclusion

**The hypothesis.** The post-hoc analyses are consistent with "readable but not committed": the model registers its answer while reading the options, a probe reads it at about one-pass accuracy by the end of the prompt, and once the CoT begins it drops out until the product has been worked out. Two parts of this are observations, not hypotheses: the last-token direction stops transferring at the first CoT token, and readability at layer 22 dips to the majority rate during the written arithmetic. Only "recomputation" is the hypothesis. This fits Kudo et al. (2026) and Jia & Mu (2026). Thurnherr (2026) informally places output-digit computation in direct multiplication on Qwen 2.5 7B at layers 22–24, close to the depth here.

**What it does not rule out.** (1) An epiphenomenal correlate: the probe reads a direction that tracks the answer but that the model never uses. (2) A position-conditional code: the answer is carried through the CoT in a different direction at each position, so the dip reflects re-encoding, not recomputation. (3) A trailing-context readout: once the product is written, the retrained probe may simply read the written string. Discriminating tests, to be pre-registered: probe-weight, per-letter or reading-vector (Zou et al. 2023) steering with larger contrast sets for (1); probes trained mid-CoT and tested before the CoT for (2), where transfer would weaken recomputation; patching the written product for (3).

**If probe-weight steering works.** If a probe-weight or per-letter direction later flips answers, the "not committed" half falls and the contrast with Cox et al. reduces to "same phenomenon, weaker vector". The mid-CoT readability dip and the options-only methodology would survive; the headline hypothesis would not.

**Relation to Cox et al.** This is a case study on the other side of the task-regime boundary their own results imply. On multiplication the CoT adds ~25 points of accuracy and visibly recomputes; there, a readable pre-CoT answer coexists with a steering null and, on post-hoc evidence, a CoT during whose arithmetic the answer stops being linearly readable in any direction I probed. Same model family and size; the difference is whether the CoT does computation that can change the answer. I did not run their tasks through this pipeline or this task through their full protocol: reimplementing their four tasks and steering setup through the same confound stack did not fit this post's time budget; it is the first next experiment.

**The strongest steering null.** Even close to Cox et al.'s protocol — steering at every generated token with α up to +20, a perturbation of about 64% of the residual norm (20 × 3.72 ≈ 74 against ~116) — the model's answer did not move toward the steered letter (0.200 "A" answers, against 0.230 at α = 0). Accuracy fell to 0.74; a random direction of the same norm also degraded the model (0.87) without shifting answers (0.180). On the model family and size where Cox et al. report flips, a perturbation large enough to damage the model did not steer it, so "steering was too weak" is not a defense of this construction. It remains one construction.

**For CoT monitoring.** In this arithmetic regime, "decodable before the reasoning, so the reasoning is post-hoc" would have been wrong: removing the CoT costs 25 points. Decodability alone was not evidence of pre-commitment here.

## 7. Limitations

- **One model family, one task family:** Qwen 2.5 1.5B Instruct on two multiplication item sets; other model families are untested, and the split may depend on scale.
- **One steering construction** (18 vs 182 items, one letter, one layer); probe-weight and per-letter directions untried.
- **Probe regularization:** C = 0.1 was fixed in advance. A post-hoc sweep on the multiplication benchmark gives 0.705–0.715 at layer 22 for C ∈ {0.1, 1, 10} and 0.645 at C = 0.01 (with-CoT probe 0.64–0.66); other tasks and per-position probes were not swept.
- **Multiple comparisons:** the headline cells survive a max-over-layers null (p = 0.001 with 1,000 permutations, itself the floor); the per-position and post-hoc analyses, which may also overfit to position, are not corrected.
- **Eventual vs correct answer:** with CoT the model is right 98.5% of the time, so the two cannot be separated for the with-CoT target.
- **Small held-out sets** (100–200 items; CIs about ±0.06–0.10).

## 8. Next experiments

Cox et al.'s tasks through this pipeline and this task through theirs, on the same model; pre-registered versions of the Section 6 discriminating tests on a held-out task; a task where the CoT changes some answers but not all.

## Appendix A: Preliminary and auxiliary experiments

Letters refer to `experiments/<letter>_<name>/` in the repo and to Appendix B. Load checks (A and the 7B check) only verified activation extraction.

**B — the initial layer-14 null.** A layer-14 probe on 200 two-option addition prompts scored 0.51 ± 0.06 (chance 0.50) while positive controls read prompt facts at 0.95–1.00. It was a layer-choice artifact: later sweeps put the signal at layers 18–22.

**C — Turpin-style cue injection.** Six few-shot examples all answering (A), on 40 easy questions whose answer was never (A): 0/40 answers flipped in either cue direction and no CoT mentioned the pattern (95% upper bound on influence about 9%). Small-N, exploratory.

**D — the externalization hypothesis test.** If post-training taught the instruct model to externalize computation, the base model should show more pre-CoT signal. It did not (layer 14: base 0.540, instruct 0.500, near-identical behaviour); a post-hoc sweep showed both rising to 0.58–0.63 at layers 21–27, the first hint of the late-layer signal.

**E and H — the runs behind Section 4.** E is the BIG-Bench Hard probe that decoded letter bias (0.731 vs "always B" at 0.769); H is the ±1/±2/±10 run (Appendix D).

**E2, F, G, I, I3, I4 — tasks that could not be probed.** Pilots on BIG-Bench Hard and multi-step arithmetic, at 1.5B and 7B, left no one-pass answer to decode (7B raised CoT accuracy, not no-CoT accuracy; see Appendix C). At the other extreme, capitals and element symbols were solved perfectly without CoT, leaving nothing for the CoT to add.

(Figure 7: The same question under each distractor rule. The original options cluster at ±1, ±2, ±10 around the answer; the corrected ones are other products with no structural tell.)

## Appendix B: Repository and pre-registration audit

Repository: [github.com/CCaptainX/cue-channels](https://github.com/CCaptainX/cue-channels) (private; access on request). Each experiment lives in `experiments/<letter>_<name>/` with its pre-registration, pilot, metrics and README; probe code is in `src/cue_channels/probes/`. B–D were exploratory; E onward were pre-registered.

Main-body experiments (pre-registration → results):
- Multiplication benchmark (H2): `1ee5528` → `db4fad4` → `0b9dd69` — supported, both targets.
- Multiplication-tables replication and unit-conversions boundary (I3): `d5d1be1` → `bbc4706`, `a13d731`, `af736fd` — times tables supported; unit conversions not supported at the last token.
- Causal steering test (J): `d9bbb0f` (+ exploratory addendum `37676ba`) → `e9882de` — not supported.
- Per-token sweep (K): `56b863a` → `c48a077` — not confirmed.
- Mid-CoT decay test (M): `3742d74` → `32bef59` — registered verdict "signal was always noise" (not supported); retrained-probe analysis post hoc.
- Option-reversal test (N): `ecb491d` → `db62ccb` — mixed; post-hoc evidence labelled.
- Exploratory robustness checks (regularization sweep, correct/incorrect breakdown; not pre-registered): `647b901`; 1,000-permutation null: `c37dc63`.

Appendix experiments: E `0f7353a` → `2bc86bc`; E2 `c918f6e` → `f66ebcb`; F `c67a872` → `1ac7091`; G/H `c1f8159` → `00dcea8`, `b163ee6`, `f0c44ec`, `8cb14c3`; I `8a6de72` → `e094845`; I4 `a54d1a6` → `aaa65aa`, `3e99734`.

## Appendix C: What "untested" means for multi-step tasks

On carry addition, modular arithmetic and easy GSM8K, consider what each probe target would reveal.

**The no-CoT target would decode letter bias.** Forced to answer directly, the 1.5B model answers "D" on 50%, 97% and 53% of items, at near-chance accuracy (0.27–0.33). A probe would learn that default, the letter-bias trap of Section 4, and come back "positive" near the majority-letter rate for the wrong reason.

**The with-CoT target would most likely give chance.** With CoT the model is mostly right (1.00, 0.90, 0.80), so this asks whether the correct result is encoded at the last prompt token, before any arithmetic is written. The model cannot compute it in one pass, so I expect chance; I did not run this probe.

**The honest claim.** I skipped the full probe stack because both targets look uninformative by design: a decision based on a predicted outcome, not a methodological impossibility, and someone could run it and let the data speak. The pilot numbers therefore do not show that "pre-CoT decoding fails on carry addition". The question is open.

## Appendix D: The distractor leak and the options-only baseline

**A worked example.** In the first run (H), each wrong option was the answer shifted by ±1, ±2 or ±10. For "What is 11 × 15?" the options were 165, 167, 163, 175 and 164 (Figure 7): four are small offsets of the fifth, which is the answer. "Pick the value the others are offset from" is right on 200/200 items without reading the question ("pick the median" only about 61%).

**How it showed up.** A with-CoT-answer probe scored 0.835 at the "(E" token that begins option E but 0.665 at the final prompt token. The corrected benchmark also peaks before the final token, at the answer marker (0.765 vs 0.70), as computing while reading predicts; the leak's signature was a peak inside the option list together with an options-only classifier at 1.000 (0.206 after the fix). That comparison, not timing alone, ruled the leak out.

**Fix and baseline.** H2 draws distractors from the answer's own law (other products a′ × b′, a′, b′ ∈ 10–20). The options-only baseline (best of logistic regression, random forest and two hand rules; 5,000 items; option values only) scores 1.000 on H's distractors and 0.206 on H2's (chance 0.20).

**What survived.** Had H's no-CoT probe (0.695) been pure artifact, H2 would have fallen to its 0.28 majority baseline; it held at 0.705. The leak's signature vanished: the option-E-start probe fell from 0.835 (0.17 above the final token) to 0.702 (0.035 above 0.667). 

## References

- Chen, Y. et al. (2025). *Reasoning Models Don't Always Say What They Think.* [arXiv:2505.05410](https://arxiv.org/abs/2505.05410).
- Cox, K., Kianersi, D., & Garriga-Alonso, A. (2026). *Post-Hoc Reasoning in Chain of Thought: Decoding and Steering Pre-Committed Answers.* Mechanistic Interpretability Workshop at ICML 2026. [arXiv:2603.01437](https://arxiv.org/abs/2603.01437).
- Deng, Y., Choi, Y., & Shieber, S. (2024). *From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step.* [arXiv:2405.14838](https://arxiv.org/abs/2405.14838).
- Jia, R., & Mu, D. (2026). *From Decorative to Load-Bearing: Task Difficulty Shapes the Causal Role of Chain-of-Thought.* Transactions on Machine Learning Research. [arXiv:2609.25366](https://arxiv.org/abs/2609.25366).
- Kudo, K., Aoki, Y., Kuribayashi, T., Sone, S., Taniguchi, M., Brassard, A., Sakaguchi, K., & Inui, K. (2026). *LLMs Faithfully and Iteratively Compute Answers During CoT: A Systematic Analysis With Multi-step Arithmetics.* Findings of EACL 2026. [arXiv:2412.01113](https://arxiv.org/abs/2412.01113).
- Lanham, T. et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning.* [arXiv:2307.13702](https://arxiv.org/abs/2307.13702).
- Pfau, J., Merrill, W., & Bowman, S. R. (2024). *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models.* COLM 2024. [arXiv:2404.15758](https://arxiv.org/abs/2404.15758).
- Rimsky, N. et al. (2023). *Steering Llama 2 via Contrastive Activation Addition.* [arXiv:2312.06681](https://arxiv.org/abs/2312.06681).
- Sosis, B., & Gavenčiak, T. (2025). *Measuring Beliefs of Language Models During Chain-of-Thought Reasoning.* LessWrong / AI Alignment Forum. [lesswrong.com/posts/a86uAnPykqNtmEbDH](https://www.lesswrong.com/posts/a86uAnPykqNtmEbDH/measuring-beliefs-of-language-models-during-chain-of-thought-1).
- Thurnherr, H. (2026). *Trying to use NLAs to find out how Qwen 2.5 7B does multiplication.* LessWrong, May 16, 2026. [lesswrong.com/posts/XN6CrJDiZvphGCC2e](https://lesswrong.com/posts/XN6CrJDiZvphGCC2e/trying-to-use-nlas-to-find-out-how-qwen-2-5-7b-does).
- Turpin, M. et al. (2023). *Language Models Don't Always Say What They Think.* [arXiv:2305.04388](https://arxiv.org/abs/2305.04388).
- Zou, A. et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* [arXiv:2310.01405](https://arxiv.org/abs/2310.01405).

---

*Comments and critique welcome. The points I most want pushback on: is "readable but not committed" the right reading of the steering and mid-CoT results, or could a different direction (probe weights, earlier layers) still find a causal role? And is the task-level explanation of the contrast with Cox et al. the most plausible one?*
