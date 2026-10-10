"""Shared prompt/data helpers for experiment F (BBH 3-shot raw-text prompts)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

F_DIR = Path("experiments/F_layer_sweep")
TASKS = ["hyperbaton", "tracking_shuffled_objects_three_objects", "date_understanding"]


def load_items(task: str, n: int = 200) -> list[dict]:
    ex = json.loads((F_DIR / f"data/{task}.json").read_text())["examples"]
    idx = sorted(random.Random(0).sample(range(len(ex)), n))
    items = []
    for i in idx:
        e = ex[i]
        letters = "".join(re.findall(r"^\(([A-Z])\)", e["input"], re.M))
        items.append({"idx": i, "input": e["input"], "target": e["target"].strip("()"),
                      "letters": letters})
    return items


def load_prompts(task: str) -> tuple[str, str]:
    """(cot_prefix, answer_only_prefix), verbatim BBH 3-shot minus the canary header."""
    txt = (F_DIR / f"data/{task}_cot.txt").read_text()
    body = txt.split("-----\n", 1)[1].strip()
    description, *blocks = re.split(r"\n\n(?=Q: )", body)
    cot = description + "\n\n" + "\n\n".join(b.strip() for b in blocks)
    ao = []
    for b in blocks:
        q, a = b.split("\nA: ", 1)
        letter = re.search(r"So the answer is \(([A-Z])\)\.", a).group(1)
        ao.append(f"{q}\nA: ({letter})")
    return cot, description + "\n\n" + "\n\n".join(ao)


def cot_prompt(prefix: str, item: dict) -> str:
    return f"{prefix}\n\nQ: {item['input']}\nA: Let's think step by step."


def no_cot_prompt(prefix: str, item: dict) -> str:
    return f"{prefix}\n\nQ: {item['input']}\nA: ("


def question_end_prefix(prefix: str, item: dict) -> str:
    """Prompt text up to and including the last character of the question input."""
    return f"{prefix}\n\nQ: {item['input']}"


def parse_no_cot(text: str, letters: str) -> str | None:
    t = text.strip()
    return t[0] if t and t[0] in letters else None


_ANS = re.compile(r"answer is:?\s*\(?([A-Z])\)?")


def parse_cot(text: str, letters: str) -> str | None:
    m = _ANS.search(text)
    return m.group(1) if m and m.group(1) in letters else None
