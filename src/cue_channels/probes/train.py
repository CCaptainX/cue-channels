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


def train_linear_probe(
    X: np.ndarray, y: np.ndarray, layer: int = -1, folds: int = 5, seed: int = 0
) -> ProbeResult:
    """Cross-validated logistic-regression probe. ``X``: [n, d_model], ``y``: [n] labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _, counts = np.unique(y, return_counts=True)
    folds = max(2, min(folds, int(counts.min())))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv)
    return ProbeResult(
        layer=layer,
        accuracy=float(scores.mean()),
        accuracy_std=float(scores.std()),
        chance=float(counts.max() / counts.sum()),
        n=int(len(y)),
    )


def probe_all_layers(acts: dict[int, np.ndarray], y: np.ndarray, **kw) -> list[ProbeResult]:
    return [train_linear_probe(X, y, layer=layer, **kw) for layer, X in sorted(acts.items())]
