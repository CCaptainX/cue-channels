"""Model registry and run configuration.

Swapping models is one line: change ``DEFAULT_MODEL`` (or set ``CUE_MODEL`` in the env,
or pass ``--model`` to the scripts).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Backend = Literal["local", "sagemaker"]


@dataclass(frozen=True)
class ModelConfig:
    key: str
    hf_id: str
    n_layers: int
    d_model: int
    gated: bool = False
    # TransformerLens name, if it differs from hf_id (None = use hf_id).
    tl_name: str | None = None


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen2.5-1.5b": ModelConfig(
        key="qwen2.5-1.5b",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        n_layers=28,
        d_model=1536,
    ),
    # Pretrained-only counterpart of the default; no chat template, use raw-text prompts.
    "qwen2.5-1.5b-base": ModelConfig(
        key="qwen2.5-1.5b-base",
        hf_id="Qwen/Qwen2.5-1.5B",
        n_layers=28,
        d_model=1536,
    ),
    "smollm2-1.7b": ModelConfig(
        key="smollm2-1.7b",
        hf_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        n_layers=24,
        d_model=2048,
    ),
    "llama3.2-1b": ModelConfig(
        key="llama3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        n_layers=16,
        d_model=2048,
        gated=True,
    ),
}

DEFAULT_MODEL = "qwen2.5-1.5b"


@dataclass
class RunConfig:
    model: str = field(default_factory=lambda: os.environ.get("CUE_MODEL", DEFAULT_MODEL))
    backend: Backend = "local"
    max_new_tokens: int = 512
    temperature: float = 0.0
    seed: int = 0
    # Layers to cache residual-stream activations from; None = all layers.
    probe_layers: list[int] | None = None
    output_dir: str = "results"

    @property
    def model_config(self) -> ModelConfig:
        return get_model_config(self.model)


def get_model_config(key: str) -> ModelConfig:
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]
    # Allow passing a raw HF id that is already registered.
    for cfg in MODEL_REGISTRY.values():
        if cfg.hf_id == key:
            return cfg
    raise KeyError(f"Unknown model {key!r}; options: {sorted(MODEL_REGISTRY)}")
