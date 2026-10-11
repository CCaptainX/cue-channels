"""Model backends.

``ModelBackend`` is the interface the eval runner depends on. ``LocalBackend`` runs HF
transformers on MPS / CUDA / CPU; ``SageMakerBackend`` is a stub for remote batch jobs.

Heavy imports (torch, transformers) are deferred to ``load()`` so that importing this
module, and running the fast test suite, never needs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from cue_channels.config import ModelConfig, RunConfig

if TYPE_CHECKING:
    import numpy as np

Messages = list[dict[str, str]]


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int


@runtime_checkable
class ModelBackend(Protocol):
    def load(self) -> None: ...

    def generate(self, messages: Messages, **kwargs: Any) -> Generation: ...

    def get_activations(
        self, messages: Messages, layers: list[int] | None = None
    ) -> dict[int, np.ndarray]:
        """Residual-stream activations at the last prompt token (just before the CoT).

        Returns ``{layer: array[d_model]}``.
        """
        ...

    def shutdown(self) -> None: ...


def detect_device() -> str:
    """Pick the best available torch device: cuda > mps > cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalBackend:
    def __init__(self, model_config: ModelConfig, run_config: RunConfig | None = None):
        self.model_config = model_config
        self.run_config = run_config or RunConfig(model=model_config.key)
        self.device: str | None = None
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = detect_device()
        # bf16 rather than fp16: Qwen2.5 activations can overflow fp16.
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.hf_id, dtype=dtype
        ).to(self.device)
        self.model.eval()

    def _encode(self, messages: Messages):
        assert self.tokenizer is not None, "call load() first"
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return enc["input_ids"].to(self.device)

    def encode_raw(self, text: str):
        """Tokenize plain text with no chat template (e.g. for completion-style prompts)."""
        assert self.tokenizer is not None, "call load() first"
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def generate(self, messages: Messages, **kwargs: Any) -> Generation:
        import torch

        input_ids = self._encode(messages)
        temperature = kwargs.pop("temperature", self.run_config.temperature)
        gen_kwargs = {
            "max_new_tokens": kwargs.pop("max_new_tokens", self.run_config.max_new_tokens),
            "do_sample": temperature > 0,
            **kwargs,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)
        completion = out[0, input_ids.shape[1] :]
        return Generation(
            text=self.tokenizer.decode(completion, skip_special_tokens=True),
            prompt_tokens=int(input_ids.shape[1]),
            completion_tokens=int(completion.shape[0]),
        )

    def get_activations(self, messages: Messages, layers: list[int] | None = None):
        import torch

        input_ids = self._encode(messages)
        with torch.no_grad():
            out = self.model(input_ids, output_hidden_states=True)
        # hidden_states[0] is the embedding output; hidden_states[i + 1] is after block i.
        hs = out.hidden_states
        layers = layers if layers is not None else list(range(len(hs) - 1))
        return {i: hs[i + 1][0, -1].float().cpu().numpy() for i in layers}

    def generate_text(
        self, prompt: str, max_new_tokens: int = 64, stop_strings: list[str] | None = None
    ) -> Generation:
        """Greedy completion of a raw-text prompt (no chat template), e.g. for base models."""
        import torch

        input_ids = self.encode_raw(prompt)
        extra = {"stop_strings": stop_strings, "tokenizer": self.tokenizer} if stop_strings else {}
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **extra,
            )
        completion = out[0, input_ids.shape[1] :]
        return Generation(
            text=self.tokenizer.decode(completion, skip_special_tokens=True),
            prompt_tokens=int(input_ids.shape[1]),
            completion_tokens=int(completion.shape[0]),
        )

    def get_activations_text(self, prompt: str, layers: list[int] | None = None):
        """Like ``get_activations`` but for a raw-text prompt: last prompt token, per layer."""
        import torch

        input_ids = self.encode_raw(prompt)
        with torch.no_grad():
            out = self.model(input_ids, output_hidden_states=True)
        hs = out.hidden_states
        layers = layers if layers is not None else list(range(len(hs) - 1))
        return {i: hs[i + 1][0, -1].float().cpu().numpy() for i in layers}

    def get_residual_stream(self, messages: Messages, layer: int):
        """Residual stream after block ``layer`` at every prompt position: [seq, d_model]."""
        import torch

        input_ids = self._encode(messages)
        with torch.no_grad():
            out = self.model(input_ids, output_hidden_states=True)
        return out.hidden_states[layer + 1][0].float().cpu().numpy()

    def shutdown(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()
        elif self.device == "mps":
            import torch

            torch.mps.empty_cache()


class SageMakerBackend:
    """Stub: run generation + activation capture as SageMaker async jobs.

    Intended pattern (not implemented yet):
      1. Upload a JSONL batch of prompts to s3://$SAGEMAKER_S3_BUCKET/inputs/<run_id>.jsonl
      2. Submit an async inference / processing job that loads the HF model on a GPU
         instance, runs generate + hidden-state capture, and writes results to
         s3://.../outputs/<run_id>/ (completions as JSONL, activations as .npy per layer)
      3. Poll the job, then download outputs and expose them through the same
         ``generate`` / ``get_activations`` interface (served from a local cache).
    """

    def __init__(self, model_config: ModelConfig, run_config: RunConfig | None = None):
        self.model_config = model_config
        self.run_config = run_config or RunConfig(model=model_config.key, backend="sagemaker")
        self._session = None

    def load(self) -> None:
        # TODO: import sagemaker/boto3, create Session, resolve SAGEMAKER_ROLE_ARN and bucket.
        # TODO: create (or reuse) a HuggingFaceModel endpoint config with async inference.
        raise NotImplementedError("SageMakerBackend.load: see class docstring for the plan")

    def generate(self, messages: Messages, **kwargs: Any) -> Generation:
        # TODO: batch requests instead of one call per prompt; submit via invoke_endpoint_async
        # and poll the S3 OutputLocation.
        raise NotImplementedError

    def get_activations(self, messages: Messages, layers: list[int] | None = None):
        # TODO: custom inference.py that returns hidden states; write .npy to S3 per layer.
        raise NotImplementedError

    def shutdown(self) -> None:
        # TODO: delete endpoint to stop billing.
        pass


def make_backend(run_config: RunConfig) -> ModelBackend:
    cfg = run_config.model_config
    if run_config.backend == "local":
        return LocalBackend(cfg, run_config)
    if run_config.backend == "sagemaker":
        return SageMakerBackend(cfg, run_config)
    raise ValueError(f"Unknown backend {run_config.backend!r}")
