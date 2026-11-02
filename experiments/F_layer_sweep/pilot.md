# F pilot: no task passed; full experiment not run

Pre-registration: [`preregistration.md`](preregistration.md) (commit `6fc116d`). Raw numbers and CoT samples are in
[`pilot.json`](pilot.json); the figure is [`figures/4_no_cot_letter_distribution.png`](figures/4_no_cot_letter_distribution.png).
Qwen2.5-1.5B-Instruct, BBH 3-shot raw text, first 30 of each task's seed-0 sample. Runtime 4 min 53 s.

| Task | no-CoT acc [95% CI] | with-CoT acc [95% CI] | top no-CoT letter | G1 [0.55, 0.85] | G2 ≤ 60% | G3 ≥ 0.85 |
|---|---|---|---|---|---|---|
| hyperbaton (2-way) | 0.533 [0.367, 0.700] | 0.433 [0.267, 0.600] | **B 83%** (25/30) | fail | fail | fail |
| tracking_shuffled_objects_three_objects | 0.267 [0.133, 0.433] | 0.333 [0.167, 0.500] | **B 100%** (30/30) | fail | fail | fail |
| date_understanding (6-way) | 0.267 [0.133, 0.433] | 0.533 [0.367, 0.700] | C 47% (14/30) | fail | pass | fail |

**Outcome.** Every task fails, and every task fails G3. The highest with-CoT accuracy is 0.533, far from the 0.85
needed. On hyperbaton, 8/30 CoT answers were unparsed, but even counting all of them as correct gives 0.70, so the
outcome doesn't depend on parsing. As pre-registered, **the full experiment (N = 200, layer sweep, positional checks)
was not run**, and no other task was tried.

**What the pilot shows.**
- **Constant-letter bias without CoT is the norm for this model**, not an E-specific artifact. It answers "B" on 83% of
  hyperbaton items, 100% of tracking items, 73% of E2's logical-deduction items and 77% of E's. Ground truth has no such skew
  (hyperbaton truth: A 19, B 11).
- **This 1.5B model doesn't get CoT's benefit on BBH.** With-CoT accuracy is at or near chance on 2 of 3 tasks, and below
  no-CoT on hyperbaton. Its CoTs follow the exemplar format but reason wrongly. G3 as specified (≥ 0.85) isn't
  reachable at this scale on these tasks.
- So within the F design, BBH can't supply the "partly solvable without CoT, solvable with CoT" regime for a
  1.5B model. Of everything tried so far, only D's two-digit addition is (about 60% no-CoT, 100% with-CoT), and its
  no-CoT letter bias was small (base 43.5% / Instruct 57.5% "A").

**Data files (sha256):**
```
    0148d4ac5fca05b2f82373e5fef7208e15363b9c6079493034d77b3dab496bf5  date_understanding.json
    a0a13c392f689a1828f5f6ffb9dee3d44483794490da5d92f0e2a723787b1cfb  date_understanding_cot.txt
    b6d7f55c75f5d02f2987a4f3123802dfa093872089a56a334144aae8a7264fdc  hyperbaton.json
    19898af91c35cb26bbdf5fdb6f6fb17e138677250332e2994005625ffb50b261  hyperbaton_cot.txt
    546cc7bf10c6a01a6a4c17ae28a4966485f21f29fd5d111f37858ec78e354f6b  tracking_shuffled_objects_three_objects.json
    2ef54cb2a72052adc3484a76209cb6a6535f194d677e79bcec2e2e71cf3768af  tracking_shuffled_objects_three_objects_cot.txt
```
