# I1 pilot: 4-candidate task triage — no task passed; I2 not run

Pre-registration: [`preregistration.md`](preregistration.md) (`44425f0`). Raw numbers: [`pilot.json`](pilot.json) (model gates) and
[`pattern_baseline.json`](pattern_baseline.json) (G4). Qwen2.5-1.5B-Instruct, 3-shot raw text, 30 items per task, 5 options. Pilot runtime ≈ 6 min.

| Task | G1 no-CoT acc [95% CI] (need 0.55–0.85) | G3 with-CoT acc (need ≥ 0.85) | G2 top no-CoT letter (need ≤ 60%) | G4 options-only best (need ≤ 0.30) | Result |
|---|---|---|---|---|---|
| add_carry | **0.333** [0.167, 0.500] | 1.000 | D 50% | 0.208 | **fail** (G1) |
| mod | **0.267** [0.100, 0.433] | 0.900 | **D 97%** | 0.204 | **fail** (G1, G2) |
| gsm8k_easy | **0.333** [0.167, 0.500] | **0.800** | D 53% | 0.235 | **fail** (G1, G3) |
| aqua_easy | **0.233** [0.100, 0.400] | **0.567** | **D 93%** | **0.354** | **fail** (G1, G2, G3, G4) |

**Outcome: 0 of 4 tasks passed, so per the pre-registration I2 was not run and this is reported back for a scope decision.**

**Findings from the triage itself**
- **Every task fails G1, and in the same direction:** without CoT the model is at or near chance (0.23–0.33 vs chance 0.20) and leans on
  "D" (50–97%), the same constant-letter default seen on BBH (E–G) and on H's addition variants. With CoT, the arithmetic tasks are solved
  (add_carry 1.000, mod 0.900).
- **Multiplication (H/H2: 0.725–0.740 no-CoT) is the outlier, not the rule.** Two-digit addition *with carry*, which is easier to compute
  than 10–20 × 10–20 multiplication, is at chance without CoT. A plausible reading is that small products are memorized facts the model can
  retrieve in one pass, while carry-addition and modular reduction are multi-step and need the CoT. That would fit the capability-bound
  story (no one-pass competence → no pre-CoT decision to decode), but it is a hypothesis from the triage, not a tested result.
- **AQuA's native distractors leak** (random forest 0.354 from option text alone), so it would have been excluded on G4 regardless.

**Disclosed pre-run deviation (G4, gsm8k_easy).** The pre-registration says GSM8K distractors are answers of other filtered items with "the same
distribution as the correct answer". The first implementation sampled uniformly over *distinct* answer values, which under-weights frequent
answers; the G4 check caught it (0.336). Before any model run, the sampler was corrected to draw other items' answers item-wise
(frequency-weighted), matching the registered rule; G4 then gave 0.235. Both numbers are reported here; gsm8k_easy fails G1/G3 either way.

**Data files (sha256).** `aqua_train.json` (47 MB) is not committed; re-download from github.com/google-deepmind/AQuA (`train.json`) and check the hash.
```
    35b70c7cb1660c34d302bb932dfd3656ce3d05e8c982c797767fec31d7d1e0ba  aqua_train.json
    17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465  gsm8k_train.jsonl
```
