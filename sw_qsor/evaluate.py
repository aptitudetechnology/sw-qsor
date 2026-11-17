"""Group-held-out evaluation of multi-label scores.

Per-label metrics are computed **within each test fold** and then summarised across folds
(mean, SD). Pooling out-of-fold scores across folds before scoring is avoided, because any
predictor fitted on the training folds (even a base rate) carries a fold-dependent offset,
and pooling turns that offset into spurious ranking. For a base-rate predictor that bias
pushes pooled AUC below 0.5.

Written for sw-qsor as a project-independent utility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

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


def fold_aucs(y: np.ndarray, s: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """ROC AUC of one label within each fold, in fold order (NaN if a fold is unscoreable)."""
    out = []
    for f in np.unique(folds):
        m = folds == f
        ok = 0 < y[m].sum() < m.sum()
        out.append(roc_auc_score(y[m], s[m]) if ok else np.nan)
    return np.array(out)


def group_bootstrap_margin(y: np.ndarray, model: np.ndarray, floors: Sequence[np.ndarray],
                           groups: np.ndarray, mask: np.ndarray, *, n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """AUC margin of ``model`` over the best of ``floors`` on the molecules in ``mask``, with a
    95% percentile interval from resampling **groups** (not molecules) with replacement.

    Scores should already be comparable across folds (:func:`within_fold_ranks`). In each
    resample the floor is the higher of the floors' AUCs in that resample. Resamples that
    lack a positive or a negative are skipped and counted.
    """
    y, model, groups = np.asarray(y)[mask], np.asarray(model)[mask], np.asarray(groups)[mask]
    floors = [np.asarray(f)[mask] for f in floors]

    def margin(idx):
        yy = y[idx]
        return roc_auc_score(yy, model[idx]) - max(roc_auc_score(yy, f[idx]) for f in floors)

    point = margin(np.arange(len(y)))
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inverse == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws, skipped = [], 0
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        if 0 < y[idx].sum() < len(idx):
            draws.append(margin(idx))
        else:
            skipped += 1
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"margin": float(point), "lo": float(lo), "hi": float(hi), "n_boot": len(draws),
            "skipped": skipped, "molecules": int(len(y)), "positives": int(y.sum()),
            "groups": int(len(uniq))}
