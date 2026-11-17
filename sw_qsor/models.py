"""Structural models for milestone 2, specified in plan-milestone-2.md before any run.

The settings here are that plan's. Changing them is a new plan, not a tweak.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit import DataStructs
from sklearn.ensemble import RandomForestClassifier

RF_PARAMS = dict(n_estimators=1000, max_features="sqrt", min_samples_leaf=1, random_state=0,
                 n_jobs=-1)
TANIMOTO_K = 10


def fingerprint_matrix(fps: Sequence) -> np.ndarray:
    """Bit vectors as a dense (n, n_bits) uint8 array."""
    out = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, out[i])
    return out


def random_forest(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray) -> np.ndarray:
    """P(positive) from the plan's random forest. ``X`` may be fingerprints or a single column."""
    train_X = np.asarray(train_X)
    test_X = np.asarray(test_X)
    if train_X.ndim == 1:
        train_X, test_X = train_X[:, None], test_X[:, None]
    rf = RandomForestClassifier(**RF_PARAMS).fit(train_X, train_y)
    return rf.predict_proba(test_X)[:, list(rf.classes_).index(1)]


def tanimoto_knn(train_fps: Sequence, train_y: np.ndarray, test_fps: Sequence,
                 k: int = TANIMOTO_K) -> np.ndarray:
    """Mean label of the ``k`` most Tanimoto-similar training molecules (ties: training order).

    This is the neighbour-lookup reference: what a model gets from recognising analogues alone.
    """
    train_y = np.asarray(train_y, float)
    k = min(k, len(train_fps))
    out = np.empty(len(test_fps))
    for i, fp in enumerate(test_fps):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, list(train_fps)))
        out[i] = train_y[np.argsort(-s, kind="stable")[:k]].mean()
    return out


def tanimoto_neighbours(train_fps: Sequence, test_fps: Sequence, k: int = TANIMOTO_K) -> np.ndarray:
    """Indices of the ``k`` most Tanimoto-similar training molecules per test molecule, ties by
    training order: the neighbours :func:`tanimoto_knn` averages over, computed once so that
    many labels can share them. ``train_y[idx].mean(1)`` equals ``tanimoto_knn``."""
    k = min(k, len(train_fps))
    train_fps = list(train_fps)
    return np.array([np.argsort(-np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps)),
                                kind="stable")[:k] for fp in test_fps])
