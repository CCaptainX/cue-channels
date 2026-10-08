"""Linear probes on pre-CoT residual-stream activations.

H3a: on unfaithful flips, a probe at the last prompt token predicts the final answer
well above chance, meaning the decision exists before the CoT verbalizes anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProbeResult:
    layer: int
    accuracy: float
    accuracy_std: float
    chance: float
    n: int


def _make_pipeline(C: float = 0.1, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, C=C, random_state=seed)
    )


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    groups: np.ndarray | None = None,
    C: float = 0.1,
    seed: int = 0,
):
    """Fit a standardized logistic-regression probe on a held-out split.

    ``X``: [N, D] activations, ``y``: [N] labels. If ``groups`` is given, the split keeps
    every group entirely in train or test (e.g. both orderings of the same question).

    Returns ``(probe, metrics)``, where ``probe`` is a fitted sklearn pipeline.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError(f"expected X [N, D] and y [N], got {X.shape} and {y.shape}")

    idx = np.arange(len(y))
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))
    else:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=y
        )

    probe = _make_pipeline(C=C, seed=seed)
    probe.fit(X[train_idx], y[train_idx])
    _, counts = np.unique(y[test_idx], return_counts=True)
    metrics = {
        "train_accuracy": float(probe.score(X[train_idx], y[train_idx])),
        "test_accuracy": float(probe.score(X[test_idx], y[test_idx])),
        "test_chance": float(counts.max() / counts.sum()),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "d": int(X.shape[1]),
        "C": C,
    }
    return probe, metrics


def train_linear_probe(
    X: np.ndarray, y: np.ndarray, layer: int = -1, folds: int = 5, seed: int = 0
) -> ProbeResult:
    """Cross-validated logistic-regression probe. ``X``: [n, d_model], ``y``: [n] labels."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    _, counts = np.unique(y, return_counts=True)
    folds = max(2, min(folds, int(counts.min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(_make_pipeline(seed=seed), X, y, cv=cv)
    return ProbeResult(
        layer=layer,
        accuracy=float(scores.mean()),
        accuracy_std=float(scores.std()),
        chance=float(counts.max() / counts.sum()),
        n=int(len(y)),
    )


def probe_all_layers(acts: dict[int, np.ndarray], y: np.ndarray, **kw) -> list[ProbeResult]:
    return [train_linear_probe(X, y, layer=layer, **kw) for layer, X in sorted(acts.items())]
