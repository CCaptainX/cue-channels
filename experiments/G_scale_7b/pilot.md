# G pilot (Qwen2.5-7B-Instruct, BBH 3-shot): no task passed, so the pivot to H was triggered

Pre-registration: [`preregistration.md`](preregistration.md) (`43d5977`). Load check: [`load_check.json`](load_check.json)
(`f1b6026`; bf16 on MPS, 15.5 GB, 28 layers, d_model 3584). Raw numbers: [`pilot.json`](pilot.json). Figure:
[`figures/4_no_cot_letter_distribution.png`](figures/4_no_cot_letter_distribution.png). Same 30 items per task as F. Runtime 39 min 37 s.

| Task | no-CoT acc [95% CI] | with-CoT acc [95% CI] | top no-CoT letter | G1 [0.55, 0.85] | G2 ≤ 60% | G3 ≥ 0.85 |
|---|---|---|---|---|---|---|
| hyperbaton (2-way) | 0.400 [0.233, 0.567] | 0.600 [0.433, 0.767] | **B 97%** (29/30) | fail | fail | fail |
| tracking_shuffled_objects_three_objects | 0.233 [0.100, 0.400] | **0.933** [0.833, 1.000] | B 67% (20/30) | fail | fail | **pass** |
| date_understanding (6-way) | **0.667** [0.500, 0.833] | 0.700 [0.533, 0.867] | B 37% | **pass** | **pass** | fail |

**Outcome: no task passes all three gates.** As pre-registered, G stops here (no G3), and **H is dispatched
automatically**.

- **The gate failures don't depend on parsing.** Hyperbaton's 12 unparsed CoTs appear to be cut off at the 384-token
  cap (7B writes long adjective-order explanations), but hyperbaton fails G1 and G2 regardless. Tracking parsed fully.
  date_understanding's best case, counting its 4 unparsed CoTs as correct, is 25/30 = 0.833 < 0.85.
- **Scale buys CoT benefit, but the no-CoT letter collapse remains.** Compared with 1.5B (F), 7B's with-CoT accuracy rises
  sharply on tracking (0.333 → 0.933) and on date (0.533 → 0.700). But its no-CoT answers still collapse onto "B" on the
  2–3-option tasks (hyperbaton 97%, tracking 67%; 1.5B: 83%, 100%). That is the finding the pre-registration asked to
  report: **the constant-letter default without CoT survives a 4.7× scale-up.**
- The two tasks are each one gate short of the needed profile in opposite directions: tracking has the CoT benefit but
  no usable no-CoT signal, and date has no-CoT signal but little CoT benefit.
