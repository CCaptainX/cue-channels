"""Experiment I tasks (pre-registered in experiments/I_generalization/preregistration.md, 8a6de72)."""

from __future__ import annotations

import json
import random
import re
from functools import cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "experiments/I_generalization/data"
LETTERS = "ABCDE"
TASKS = ["add_carry", "mod", "gsm8k_easy", "aqua_easy"]
SEEDS = {"add_carry": 301, "mod": 302, "gsm8k_easy": 303, "aqua_easy": 304}
DESCRIPTIONS = {
    "add_carry": "Answer the arithmetic question by choosing the correct option.",
    "mod": "Answer the arithmetic question by choosing the correct option.",
    "gsm8k_easy": "Solve the math word problem by choosing the correct option.",
    "aqua_easy": "Solve the math word problem by choosing the correct option.",
}


def _fmt(question: str, values: list, letter_forced: str | None, rng: random.Random, correct):
    vals = list(values)
    if letter_forced is not None:
        vals.remove(correct)
        vals.insert(LETTERS.index(letter_forced), correct)
    letter = LETTERS[vals.index(correct)]
    opts = "\n".join(f"({ltr}) {v}" for ltr, v in zip(LETTERS, vals, strict=True))
    return {"input": f"{question}\nOptions:\n{opts}", "target": letter, "letters": LETTERS,
            "question": question, "option_values": [str(v) for v in vals]}


# ------------------------------------------------------------- generated tasks
def _carry_pair(rng):
    while True:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if a % 10 + b % 10 >= 10:
            return a, b


def _gen_item(task: str, rng: random.Random, force: str | None = None) -> dict:
    if task == "add_carry":
        a, b = _carry_pair(rng)
        c = a + b
        ds: list[int] = []
        while len(ds) < 4:
            x, y = _carry_pair(rng)
            v = x + y
            if v != c and v not in ds:
                ds.append(v)
        q, steps = f"What is {a} + {b}?", f"{a} + {b} = {c}."
    elif task == "mod":
        a, b, m = rng.randint(10, 99), rng.randint(10, 99), rng.choice([7, 11, 13])
        s = a + b
        c = s % m
        ds = rng.sample([r for r in range(m) if r != c], 4)
        q = f"What is ({a} + {b}) mod {m}?"
        steps = f"{a} + {b} = {s}. {s} = {s // m} × {m} + {c}, so ({a} + {b}) mod {m} = {c}."
    else:
        raise ValueError(task)
    vals = [c] + ds
    rng.shuffle(vals)
    it = _fmt(q, vals, force, rng, c)
    it["steps"] = steps
    return it


# ------------------------------------------------------------- dataset tasks
_SENT = re.compile(r"[.?!](\s|$)")


@cache
def gsm8k_pool() -> list[dict]:
    rows = []
    for line in (DATA / "gsm8k_train.jsonl").read_text().splitlines():
        d = json.loads(line)
        ans = d["answer"].split("####")[-1].strip().replace(",", "")
        if not re.fullmatch(r"\d+", ans) or not 10 <= int(ans) <= 99:
            continue
        if len(_SENT.findall(d["question"].strip())) > 2:
            continue
        sol = re.sub(r"<<[^>]*>>", "", d["answer"].split("####")[0]).strip()
        rows.append({"question": d["question"].strip(), "answer": int(ans), "solution": sol})
    return rows


@cache
def aqua_pool() -> list[dict]:
    rows = []
    for line in (DATA / "aqua_train.json").read_text().splitlines():
        d = json.loads(line)
        opts = d.get("options", [])
        if len(opts) != 5 or [o[:2] for o in opts] != [f"{ltr})" for ltr in LETTERS]:
            continue
        rat_lines = [ln for ln in d["rationale"].splitlines() if ln.strip()]
        if len(rat_lines) > 3 or len(d["question"].split()) > 60 or d["correct"] not in LETTERS:
            continue
        rows.append({"question": d["question"].strip(), "options": [o[2:].strip() for o in opts],
                     "correct": d["correct"], "rationale": rat_lines})
    return rows


def _gsm_item(row, pool, rng, force=None):
    # Distractors = answers of other filtered items drawn item-wise (frequency-weighted), so they follow
    # the same distribution as correct answers. (Pre-run fix: the first implementation sampled uniformly
    # over distinct values, which the G4 options-only check flagged at 0.336.)
    ds: list[int] = []
    while len(ds) < 4:
        v = rng.choice(pool)["answer"]
        if v != row["answer"] and v not in ds:
            ds.append(v)
    vals = [row["answer"]] + ds
    rng.shuffle(vals)
    it = _fmt(row["question"], vals, force, rng, row["answer"])
    it["steps"] = row["solution"]
    return it


def _aqua_item(row):
    opts = "\n".join(f"({ltr}) {v}" for ltr, v in zip(LETTERS, row["options"], strict=True))
    steps = "\n".join(ln for ln in row["rationale"] if "answer" not in ln.lower()) or row["rationale"][0]
    return {"input": f"{row['question']}\nOptions:\n{opts}", "target": row["correct"],
            "letters": LETTERS, "question": row["question"], "option_values": row["options"],
            "steps": steps}


# ------------------------------------------------------------- public API
def exemplars(task: str) -> list[dict]:
    rng = random.Random(SEEDS[task] + 1000)
    if task in ("add_carry", "mod"):
        return [_gen_item(task, rng, ltr) for ltr in "BAC"]
    if task == "gsm8k_easy":
        pool = gsm8k_pool()
        rows = rng.sample(pool, 3)
        return [_gsm_item(r, pool, rng, ltr) for r, ltr in zip(rows, "BAC", strict=True)]
    pool = aqua_pool()
    return [_aqua_item(r) for r in rng.sample(pool, 3)]


def items(task: str, n: int, seed_offset: int = 0) -> list[dict]:
    rng = random.Random(SEEDS[task] + seed_offset)
    ex_q = {e["question"] for e in exemplars(task)}
    out: list[dict] = []
    if task in ("add_carry", "mod"):
        while len(out) < n:
            it = _gen_item(task, rng)
            if it["question"] not in ex_q:
                out.append(it)
    else:
        pool = gsm8k_pool() if task == "gsm8k_easy" else aqua_pool()
        cand = [r for r in pool if r["question"] not in ex_q]
        rows = rng.sample(cand, min(n, len(cand)))
        out = [(_gsm_item(r, pool, rng) if task == "gsm8k_easy" else _aqua_item(r)) for r in rows]
    for i, it in enumerate(out):
        it["idx"] = i
    return out


def prompts(task: str) -> tuple[str, str]:
    ex = exemplars(task)
    cot = [f"Q: {e['input']}\nA: Let's think step by step.\n{e['steps']}\nSo the answer is ({e['target']})."
           for e in ex]
    ao = [f"Q: {e['input']}\nA: ({e['target']})" for e in ex]
    d = DESCRIPTIONS[task]
    return d + "\n\n" + "\n\n".join(cot), d + "\n\n" + "\n\n".join(ao)
