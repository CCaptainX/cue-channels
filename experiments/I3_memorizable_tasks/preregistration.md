# I3: pre-registration — memorizable (one-pass-retrievable) tasks on Qwen 2.5 1.5B Instruct

**Registered:** 2026-10-15T12:58:17Z (UTC), together with the item generator `scripts/i3_tasks.py` (fact lists are fixed in that file), before
any I3 item is shown to a model. Ancestors: H2 (`f825941` → `38e5999`), I (`44425f0` → `a5389fe`). Everything not stated here is
identical to H2/I: `Qwen/Qwen2.5-1.5B-Instruct`, bf16 MPS, greedy, raw-text 3-shot (exemplar letters forced B, A, C), stop at `\nQ:`,
5 options (chance 0.20), probe and statistics as H2, 100-permutation max-over-layers test, Gaussian / majority / chance baselines.

## Hypothesis
For every task passing the gates: **Target A** (no-CoT answer) peaks (argmax of 5-seed mean, layers 0–27) in **[19, 25]** with seed-0
accuracy **≥ 0.55** and H2's rule (acc > majority + 0.05, CI lower bound > majority, permutation p < 0.05). Target B is reported with
H2's rule but is not part of the I3 hypothesis.

## Tasks (fact lists in `scripts/i3_tasks.py`)
1. **country_capitals**: 155 countries with a single, uncontested capital whose name does not contain the country's name. Excluded,
   e.g.: Bolivia, South Africa, the Netherlands, Côte d'Ivoire, Sri Lanka, Eswatini, Malaysia, Benin, Israel, Palestine, Tanzania,
   Burundi, Indonesia, Equatorial Guinea, Yemen, Kiribati, Nauru, Taiwan, Kosovo; and Mexico City, Kuwait City, Panama City, Guatemala
   City, Luxembourg, Singapore, Monaco, San Marino, Djibouti, Andorra la Vella, Vatican City, San Salvador, Bissau.
2. **us_state_capitals**: 50 states.
3. **element_symbols**: 50 common elements.
4. **times_tables**: a × b, a, b ∈ {2..12} (121 ordered facts, 66 unordered groups).
5. **unit_conversions**: "How many <small> are in n <big>?" for 10 fixed integer factors (feet→inches 12, hours→minutes 60,
   minutes→seconds 60, km→m 1000, kg→g 1000, weeks→days 7, days→hours 24, yards→feet 3, m→cm 100, gallons→quarts 4), n ∈ {2..12};
   plus °C→°F for C ∈ {−20, −15, …, 100}.

**Distractors (4, distinct, ≠ correct):** capitals / symbols: other facts' answers in the same task, uniform. times_tables: other
products a′ × b′ drawn pair-wise from {2..12}² (H2 rule). unit_conversions: other answers of the **same template** (different n or C),
uniform — so magnitude and option values give nothing away. Options shuffled uniformly.

**Items:** facts are cycled in a shuffled order until N items are reached; each repeat of a fact gets fresh distractors and a fresh
option order. Every item carries a `group` = fact id (times_tables: the unordered pair). **All probe CV in I3 is
StratifiedGroupKFold(5) by group** (seeded), so a fact never appears in both train and test. Pilot = first 30 items (distinct facts);
full run = first 200 items.

## Gates (all four required; as H2/I)
G1 no-CoT acc ∈ [0.55, 0.85]; G2 top no-CoT letter ≤ 60%; G3 with-CoT acc ≥ 0.85 (30 items); G4 options-only best ≤ 0.30 (5,000
items, seed offset 9000; per-option features: first number in the option if any, rank, sorted absolute differences, mean difference,
is-median, string length, alphabetical rank, count of uppercase letters; logistic regression, random forest, median rule).
**0 tasks pass → report I3 and continue to I4. ≥ 1 → full run on each passing task.**

## Full run
N = 200, layers 0–27, Targets A and B, 5 seeds, all H2 controls; option-E sanity token located directly ("(E" token), as in I.

## Verdict (I3 part of the joint I3+I4 scenarios — see ../I4_scaling/preregistration.md)
"I3 yes" iff at least one I3 task runs and every I3 task that runs meets the hypothesis; "I3 no" otherwise (including 0 passing).
