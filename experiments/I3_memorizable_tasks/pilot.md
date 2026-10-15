# I3 pilot: memorizable-task triage on Qwen 2.5 1.5B Instruct: 2 of 5 pass

Pre-registration: [`preregistration.md`](preregistration.md) (`d5d1be1`). Raw numbers: [`pilot.json`](pilot.json), [`pattern_baseline.json`](pattern_baseline.json).
30 items per task, 5 options, 3-shot raw text. Pilot runtime about 3 min.

| Task | G1 no-CoT [95% CI] (0.55–0.85) | G3 with-CoT (≥ 0.85) | G2 top letter (≤ 60%) | G4 options-only (≤ 0.30) | Result |
|---|---|---|---|---|---|
| country_capitals | **1.000** [1.000, 1.000] | 1.000 | A 37% | 0.200 | **fail** (G1, ceiling) |
| us_state_capitals | **1.000** [1.000, 1.000] | 1.000 | E 30% | 0.206 | **fail** (G1, ceiling) |
| element_symbols | **1.000** [1.000, 1.000] | 1.000 | B 30% | 0.201 | **fail** (G1, ceiling) |
| **times_tables** | **0.833** [0.700, 0.967] | 1.000 | E/A/D 23% | 0.207 | **pass** |
| **unit_conversions** | **0.600** [0.433, 0.767] | 0.933 | D 37% | 0.199 | **pass** |

**Full runs (I3 phase 2): times_tables and unit_conversions**, N = 200, grouped CV by fact.

Notes:
- The three fact-recall tasks fail G1 at the *top*: the model answers every item correctly without CoT, so there is no room for CoT to
  matter and nothing distinguishes the no-CoT and with-CoT answers. The [0.55, 0.85] band rules them out as pre-registered.
- The two passing tasks are the arithmetic-fact ones, in the same family as H2 (memorized products and conversion factors).
