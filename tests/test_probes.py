import numpy as np
import pytest

pytest.importorskip("sklearn")

from cue_channels.probes.train import train_probe  # noqa: E402


def _data(n=200, d=32, signal=3.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, d))
    X[:, 0] += signal * y
    return X, y


def test_train_probe_learns_planted_signal():
    X, y = _data()
    probe, m = train_probe(X, y)
    assert m["test_accuracy"] > 0.9
    assert m["n_train"] + m["n_test"] == len(y)
    assert probe.predict(X[:5]).shape == (5,)


def test_train_probe_near_chance_without_signal():
    X, y = _data(signal=0.0)
    _, m = train_probe(X, y)
    assert m["test_accuracy"] < 0.75


def test_grouped_split_keeps_groups_together():
    X, y = _data()
    groups = np.arange(len(y)) // 2
    _, m = train_probe(X, y, groups=groups)
    assert m["n_test"] % 2 == 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        train_probe(np.zeros((10, 4)), np.zeros(9))
