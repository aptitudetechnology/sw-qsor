"""Group-held-out evaluation of multi-label scores.

Per-label metrics are computed **within each test fold** and then summarised across folds
(mean, SD). Pooling out-of-fold scores across folds before scoring is avoided, because any
predictor fitted on the training folds (even a base rate) carries a fold-dependent offset,
and pooling turns that offset into spurious ranking. For a base-rate predictor that bias
pushes pooled AUC below 0.5.

Written for sw-qsor as a project-independent utility.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MIN_POSITIVES = 1  # a fold is scored for a label if it has >=1 positive and >=1 negative


def cross_validate(predict: Callable, x, Y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold scores: for each fold, ``predict(x[train], Y[train], x[test])``."""
    x = np.asarray(x)
    Y = np.asarray(Y)
    S = np.full(Y.shape, np.nan)
    for f in np.unique(folds):
        te = folds == f
        S[te] = predict(x[~te], Y[~te], x[te])
    return S


def per_label_per_fold(Y: np.ndarray, S: np.ndarray, folds: np.ndarray,
                       labels: list[str]) -> pd.DataFrame:
    """Long table: one row per (label, fold) with positives, prevalence, ROC AUC, AP.

    AUC and AP are NaN where the fold has no positives or no negatives for that label.
    """
    rows = []
    for f in np.unique(folds):
        te = folds == f
        for j, lab in enumerate(labels):
            y, s = Y[te, j], S[te, j]
            pos = int(y.sum())
            ok = MIN_POSITIVES <= pos < len(y)
            rows.append({
                "label": lab, "fold": int(f), "n": int(te.sum()), "positives": pos,
                "prevalence": pos / len(y),
                "auc": roc_auc_score(y, s) if ok else np.nan,
                "ap": average_precision_score(y, s) if ok else np.nan,
            })
    return pd.DataFrame(rows)


def summarise_labels(long: pd.DataFrame) -> pd.DataFrame:
    """Per label: total positives, prevalence, and mean / SD across folds of AUC and AP,
    plus AP lift (mean AP / mean prevalence)."""
    g = long.groupby("label", sort=False)
    out = pd.DataFrame({
        "positives": g.positives.sum(),
        "prevalence": g.positives.sum() / g.n.sum(),
        "folds_scored": g.auc.count(),
        "auc_mean": g.auc.mean(), "auc_sd": g.auc.std(ddof=1),
        "ap_mean": g.ap.mean(), "ap_sd": g.ap.std(ddof=1),
    })
    out["ap_lift"] = out.ap_mean / out.prevalence
    return out


def pooled_micro_auc(Y: np.ndarray, S: np.ndarray) -> float:
    """ROC AUC over all (molecule, label) pairs at once.

    Shown because it is a common headline number and is **not** a structure measure: a
    base-rate predictor scores well on it simply because labels differ in frequency.
    """
    return float(roc_auc_score(np.asarray(Y).ravel(), np.asarray(S).ravel()))


def within_fold_ranks(S: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Replace each score by its percentile rank among its own fold's scores for that label
    (ties averaged). AUC within a fold is unchanged; offsets *between* folds are removed, so
    scores can be pooled across folds without the bias described in the module docstring.
    A constant-per-fold predictor becomes constant overall, i.e. AUC exactly 0.5."""
    from scipy.stats import rankdata

    R = np.empty_like(np.asarray(S, float))
    for f in np.unique(folds):
        m = folds == f
        R[m] = (rankdata(S[m], axis=0) - 0.5) / m.sum()  # all-tied -> exactly 0.5
    return R


def by_bin(Y: np.ndarray, S: np.ndarray, bins: np.ndarray, labels: list[str],
           folds: np.ndarray, min_pos: int = 10) -> pd.DataFrame:
    """Per nearest-neighbour bin: macro-mean AUC over labels with >= ``min_pos`` positives
    and >= 1 negative in that bin. Bins cut across folds, so scores are first converted to
    within-fold ranks (:func:`within_fold_ranks`) and then pooled within the bin."""
    S = within_fold_ranks(S, folds)
    rows = []
    for b in sorted(set(bins.tolist())):
        m = bins == b
        aucs = [roc_auc_score(Y[m, j], S[m, j]) for j in range(len(labels))
                if min_pos <= Y[m, j].sum() < m.sum()]
        rows.append({"bin": b, "molecules": int(m.sum()), "labels_scored": len(aucs),
                     "auc_macro": float(np.mean(aucs)) if aucs else np.nan,
                     "auc_median": float(np.median(aucs)) if aucs else np.nan})
    return pd.DataFrame(rows).set_index("bin")
