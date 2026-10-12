"""Experiment H items and prompts (pre-registered: experiments/G_scale_7b/preregistration.md)."""

from __future__ import annotations

import random

SEEDS = {"A": 101, "B": 102, "C": 103, "C2": 201}
DESCRIPTION = "Answer the arithmetic question by choosing the correct option."
LETTERS = "ABCDE"
DELTAS = [-10, -2, -1, 1, 2, 10]


def _operands(variant: str, rng: random.Random) -> tuple[str, int, str]:
    if variant == "A":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return f"What is {a} + {b}?", a + b, f"{a} + {b} = {a + b}."
    if variant == "B":
        while True:
            a, b, c = rng.randint(10, 99), rng.randint(10, 99), rng.randint(10, 99)
            if a + b - c >= 1:
                break
        return (f"What is {a} + {b} - {c}?", a + b - c,
                f"{a} + {b} = {a + b}. {a + b} - {c} = {a + b - c}.")
    if variant == "C":
        a, b = rng.randint(10, 20), rng.randint(10, 20)
        if b == 10:
            steps = f"{a} * 10 = {a * b}."
        else:
            p, q = a * (b - 10), a * 10
            steps = f"{a} * {b} = {a} * {b - 10} + {a} * 10 = {p} + {q} = {a * b}."
        return f"What is {a} * {b}?", a * b, steps
    raise ValueError(variant)


def _product_distractors(correct: int, rng: random.Random) -> list[int]:
    """H2: 4 distinct other products a'*b' with (a', b') ~ U{10..20}^2 (same law as the correct)."""
    opts: list[int] = []
    while len(opts) < 4:
        v = rng.randint(10, 20) * rng.randint(10, 20)
        if v != correct and v not in opts:
            opts.append(v)
    return opts


def _item(variant: str, rng: random.Random, force_letter: str | None = None) -> dict:
    question, correct, steps = _operands("C" if variant == "C2" else variant, rng)
    if variant == "C2":
        opts = _product_distractors(correct, rng)
    else:
        while True:
            ds = rng.sample(DELTAS, 4)
            opts = [correct + d for d in ds]
            if all(o > 0 for o in opts) and len(set(opts)) == 4:
                break
    values = [correct] + opts
    rng.shuffle(values)
    if force_letter is not None:  # exemplars only: move the correct option to a fixed letter
        values.remove(correct)
        values.insert(LETTERS.index(force_letter), correct)
    letter = LETTERS[values.index(correct)]
    options = "\n".join(f"({ltr}) {v}" for ltr, v in zip(LETTERS, values, strict=True))
    return {"input": f"{question}\nOptions:\n{options}", "target": letter, "letters": LETTERS,
            "steps": steps, "question": question}


def exemplars(variant: str) -> list[dict]:
    rng = random.Random(SEEDS[variant] + 1000)
    return [_item(variant, rng, force_letter=ltr) for ltr in "BAC"]


def items(variant: str, n: int) -> list[dict]:
    rng = random.Random(SEEDS[variant])
    ex_q = {e["question"] for e in exemplars(variant)}
    out = []
    while len(out) < n:
        it = _item(variant, rng)
        if it["question"] in ex_q:
            continue
        it["idx"] = len(out)
        out.append(it)
    return out


def prompts(variant: str) -> tuple[str, str]:
    """(cot_prefix, answer_only_prefix) in the F/BBH layout."""
    ex = exemplars(variant)
    cot = [f"Q: {e['input']}\nA: Let's think step by step.\n{e['steps']}\n"
           f"So the answer is ({e['target']})." for e in ex]
    ao = [f"Q: {e['input']}\nA: ({e['target']})" for e in ex]
    return DESCRIPTION + "\n\n" + "\n\n".join(cot), DESCRIPTION + "\n\n" + "\n\n".join(ao)
