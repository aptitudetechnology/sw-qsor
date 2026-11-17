"""Chemistry-free floors (readme rule 2): what can be predicted without structure.

Each floor is a function ``(train_x, train_Y, test_x) -> test scores`` with ``Y`` a molecules x
labels 0/1 matrix. Hyperparameters are fixed here, before any floor was scored, and are not
tuned. The point of a floor is to be cheap, not to be good.

- :func:`base_rate`: each label's training frequency, the same for every test molecule.
  Per-label ROC AUC is exactly 0.5 by construction; average precision equals prevalence.
- :func:`knn_1d`: label frequency among the ``k`` training molecules nearest on one scalar
  feature (molecular weight, in sw-qsor). MW is derived from the formula, so this is "nearly
  free" rather than free: it knows a molecule's size and nothing about its arrangement.

Written for sw-qsor as a project-independent utility; the idea is sw-enose's
chemistry-free baseline (``demo_m2or_baseline.py``), reimplemented.
"""

from __future__ import annotations

import numpy as np

KNN_K = 50  # fixed before any run; not tuned


def base_rate(train_x, train_Y: np.ndarray, test_x) -> np.ndarray:
    """Training-set label frequency, broadcast to every test molecule."""
    n_test = len(test_x)
    return np.broadcast_to(np.asarray(train_Y, float).mean(0), (n_test, train_Y.shape[1])).copy()


def knn_1d(train_x, train_Y: np.ndarray, test_x, k: int = KNN_K) -> np.ndarray:
    """Mean label vector of the ``k`` training molecules nearest on a scalar feature.

    Ties in distance are broken by training order (stable sort), so results are deterministic.
    """
    train_x = np.asarray(train_x, float)
    test_x = np.asarray(test_x, float)
    Y = np.asarray(train_Y, float)
    k = min(k, len(train_x))
    order = np.argsort(train_x, kind="stable")
    xs = train_x[order]
    out = np.empty((len(test_x), Y.shape[1]))
    for i, x in enumerate(test_x):
        # the k nearest lie in a window of the sorted array around x's insertion point
        lo = max(0, np.searchsorted(xs, x) - k)
        hi = min(len(xs), lo + 2 * k + 1)
        cand = np.arange(lo, hi)
        nearest = cand[np.argsort(np.abs(xs[cand] - x), kind="stable")[:k]]
        out[i] = Y[order[nearest]].mean(0)
    return out
