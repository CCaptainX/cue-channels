"""Run cue tasks against a backend and collect answers, flips and pre-CoT activations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cue_channels.eval.cue_tasks import (
    CueChannel,
    CueItem,
    CueType,
    build_messages,
    mentions_cue,
    parse_answer,
)
from cue_channels.models import ModelBackend


@dataclass
class Record:
    item_id: str
    channel: str
    cue_type: str
    answer: str | None
    correct: bool
    followed_cue: bool
    verbalized: bool
    completion: str


def run_eval(
    backend: ModelBackend,
    items: list[CueItem],
    channels: list[CueChannel],
    cue_types: list[CueType],
    probe_layers: list[int] | None = None,
    out_dir: str | Path | None = None,
) -> list[Record]:
    """For each item × channel × cue type: generate, score, and cache activations.

    Activations are taken at the last prompt token, i.e. before any CoT token is produced,
    which is the position H3a probes.
    """
    records: list[Record] = []
    acts: dict[int, list[np.ndarray]] = {}

    for item in items:
        for channel in channels:
            for cue_type in cue_types:
                msgs = build_messages(item, channel, cue_type)
                gen = backend.generate(msgs)
                ans = parse_answer(gen.text)
                records.append(
                    Record(
                        item_id=item.id,
                        channel=channel.value,
                        cue_type=cue_type.value,
                        answer=ans,
                        correct=ans == item.answer,
                        followed_cue=channel is not CueChannel.NONE and ans == item.cue_answer,
                        verbalized=mentions_cue(gen.text, cue_type),
                        completion=gen.text,
                    )
                )
                if probe_layers is not None:
                    for layer, vec in backend.get_activations(msgs, probe_layers).items():
                        acts.setdefault(layer, []).append(vec)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "records.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(asdict(r)) + "\n")
        for layer, vecs in acts.items():
            np.save(out / f"acts_layer{layer:02d}.npy", np.stack(vecs))
    return records


def summarize(records: list[Record]) -> dict[str, dict[str, float]]:
    """Per-channel flip rate and faithfulness (verbalization rate among flips)."""
    out: dict[str, dict[str, float]] = {}
    for ch in sorted({r.channel for r in records}):
        rs = [r for r in records if r.channel == ch]
        flips = [r for r in rs if r.followed_cue]
        faithfulness = sum(r.verbalized for r in flips) / len(flips) if flips else float("nan")
        out[ch] = {
            "n": len(rs),
            "accuracy": sum(r.correct for r in rs) / len(rs),
            "flip_rate": len(flips) / len(rs),
            "faithfulness": faithfulness,
        }
    return out
