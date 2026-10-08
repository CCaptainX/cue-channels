import numpy as np
import pytest

pytest.importorskip("sklearn")

from cue_channels.probes.analysis import grouped_cv_accuracy, wilson_ci  # noqa: E402


def test_wilson_ci_brackets_estimate():
    lo, hi = wilson_ci(100, 200)
    assert lo < 0.5 < hi
    assert hi - lo == pytest.approx(0.138, abs=0.005)


def test_grouped_cv_tests_every_prompt_once():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 100)
    groups = np.repeat(np.arange(100), 2)
    X = rng.normal(size=(200, 16))
    X[:, 0] += 3 * y
    res = grouped_cv_accuracy(X, y, groups)
    assert res["n"] == 200
    assert res["accuracy"] > 0.9
    assert res["ci95"][0] < res["accuracy"] < res["ci95"][1]


def test_grouped_cv_chance_without_signal():
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 100)
    groups = np.repeat(np.arange(100), 2)
    res = grouped_cv_accuracy(rng.normal(size=(200, 16)), y, groups)
    assert res["ci95"][0] < 0.5 + 0.1
