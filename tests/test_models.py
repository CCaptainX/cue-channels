"""Fast tests: imports and pure-Python logic only. No model weights are downloaded."""

import numpy as np
import pytest

from cue_channels.config import DEFAULT_MODEL, MODEL_REGISTRY, RunConfig, get_model_config
from cue_channels.eval.cue_tasks import (
    CueChannel,
    CueType,
    build_messages,
    parse_answer,
    toy_items,
)
from cue_channels.models import (
    LocalBackend,
    ModelBackend,
    SageMakerBackend,
    detect_device,
    make_backend,
)


def test_default_model_registered():
    assert DEFAULT_MODEL in MODEL_REGISTRY
    assert get_model_config(DEFAULT_MODEL).hf_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert get_model_config("Qwen/Qwen2.5-1.5B-Instruct").key == DEFAULT_MODEL


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model_config("nope")


def test_model_swap_via_env(monkeypatch):
    monkeypatch.setenv("CUE_MODEL", "smollm2-1.7b")
    assert RunConfig().model_config.hf_id == "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def test_backends_satisfy_protocol():
    cfg = get_model_config(DEFAULT_MODEL)
    assert isinstance(LocalBackend(cfg), ModelBackend)
    assert isinstance(SageMakerBackend(cfg), ModelBackend)
    assert isinstance(make_backend(RunConfig(backend="sagemaker")), SageMakerBackend)


def test_detect_device():
    assert detect_device() in {"cuda", "mps", "cpu"}


@pytest.mark.parametrize("channel", list(CueChannel))
def test_build_messages_injects_cue(channel):
    item = toy_items()[0]
    msgs = build_messages(item, channel, CueType.AUTHORITY)
    text = " ".join(m["content"] for m in msgs)
    assert item.question in text
    assert ("professor" in text) == (channel is not CueChannel.NONE)


def test_parse_answer():
    assert parse_answer("blah\nAnswer: (B)") == "B"
    assert parse_answer("Answer: a ... Answer: C") == "C"
    assert parse_answer("no answer here") is None


def test_probe_on_synthetic_data():
    pytest.importorskip("sklearn")
    from cue_channels.probes.train import train_linear_probe

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    X = rng.normal(size=(200, 32))
    X[:, 0] += 3 * y  # plant a linearly decodable signal
    res = train_linear_probe(X, y)
    assert res.accuracy > 0.9


@pytest.mark.slow
def test_local_backend_generates():
    backend = LocalBackend(get_model_config(DEFAULT_MODEL))
    backend.load()
    try:
        gen = backend.generate([{"role": "user", "content": "Say hi."}], max_new_tokens=8)
        assert gen.completion_tokens > 0
        acts = backend.get_activations([{"role": "user", "content": "hi"}], layers=[0])
        assert acts[0].shape == (backend.model_config.d_model,)
    finally:
        backend.shutdown()
