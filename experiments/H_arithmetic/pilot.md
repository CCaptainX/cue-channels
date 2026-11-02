# H1 pilot: arithmetic difficulty calibration (Qwen2.5-1.5B-Instruct)

Dispatched automatically after the G pilot failed (pre-registration `43d5977`). Items and prompts come from
`scripts/h_arith.py` exactly as registered. 30 items per variant, 5 options (A)–(E), 3-shot raw text. Runtime 3 min 22 s.
Raw numbers: [`pilot.json`](pilot.json).

| Variant | no-CoT acc [95% CI] | with-CoT acc [95% CI] | top no-CoT letter | G1 [0.55, 0.85] | G2 ≤ 60% | G3 ≥ 0.85 |
|---|---|---|---|---|---|---|
| A: a + b | 0.267 [0.100, 0.433] | 1.000 [1.000, 1.000] | **D 73%** | fail | fail | pass |
| B: a + b − c | 0.233 [0.100, 0.400] | 1.000 [1.000, 1.000] | **D 93%** | fail | fail | pass |
| **C: a × b (10–20)** | **0.700** [0.533, 0.833] | **1.000** [1.000, 1.000] | B 33% | **pass** | **pass** | **pass** |

**Selected: C** (the only variant passing all three gates). H2 runs the full F protocol on it.

Notes: the "sanity floor" variant A wasn't trivial without CoT. With 5 near-neighbour options the model
collapses onto one letter ("D") without CoT, just as it collapses onto "B" on BBH. Multiplication is the exception, oddly:
no-CoT answers are spread across letters and 70% correct. A plausible explanation is that small products are
memorized lookups while sums of near-identical distractors are not, but that's a hypothesis, not tested here.
