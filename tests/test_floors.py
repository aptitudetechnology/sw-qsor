import numpy as np

from sw_qsor.evaluate import (
    by_bin,
    cross_validate,
    per_label_per_fold,
    pooled_micro_auc,
    summarise_labels,
)
from sw_qsor.floors import base_rate, knn_1d


def _data(seed=0, n=600):
    rng = np.random.default_rng(seed)
    x = rng.uniform(50, 300, n)
    common = (rng.random(n) < 0.4).astype(int)
    rare = (rng.random(n) < 0.03).astype(int)
    heavy = (x > 200).astype(int)  # a label MW genuinely predicts
    return x, np.column_stack([common, rare, heavy]), rng.integers(0, 5, n)


def test_base_rate_is_constant_per_label_and_auc_exactly_half():
    x, Y, folds = _data()
    S = cross_validate(base_rate, x, Y, folds)
    long = per_label_per_fold(Y, S, folds, ["common", "rare", "heavy"])
    assert np.allclose(long.auc.dropna(), 0.5)
    np.testing.assert_allclose(long.ap.dropna(), long.prevalence[long.ap.notna()])


def test_micro_auc_rewards_base_rate_for_frequency_alone():
    x, Y, folds = _data()
    S = cross_validate(base_rate, x, Y, folds)
    assert pooled_micro_auc(Y, S) > 0.6


def test_knn_1d_finds_a_size_driven_label_but_not_random_ones():
    x, Y, folds = _data()
    S = cross_validate(knn_1d, x, Y, folds)
    s = summarise_labels(per_label_per_fold(Y, S, folds, ["common", "rare", "heavy"]))
    assert s.loc["heavy", "auc_mean"] > 0.95
    assert abs(s.loc["common", "auc_mean"] - 0.5) < 0.1


def test_knn_1d_matches_brute_force():
    rng = np.random.default_rng(3)
    tx, ty = rng.uniform(0, 1, 200), rng.integers(0, 2, (200, 2))
    qx = rng.uniform(-0.1, 1.1, 30)
    got = knn_1d(tx, ty, qx, k=7)
    for i, q in enumerate(qx):
        nn = np.argsort(np.abs(tx - q), kind="stable")[:7]
        np.testing.assert_allclose(got[i], ty[nn].mean(0))


def test_unscoreable_folds_are_nan_and_bins_respect_min_pos():
    Y = np.array([[0], [0], [1], [0]])
    S = np.array([[0.1], [0.2], [0.3], [0.4]])
    long = per_label_per_fold(Y, S, np.array([0, 0, 1, 1]), ["l"])
    assert np.isnan(long.loc[long.fold == 0, "auc"]).all()
    b = by_bin(Y, S, np.array(["a", "a", "a", "a"]), ["l"], np.array([0, 0, 1, 1]), min_pos=2)
    assert b.loc["a", "labels_scored"] == 0


def test_base_rate_by_bin_is_exactly_half_despite_fold_offsets():
    # Pooling raw base-rate scores across folds gives AUC < 0.5; within-fold ranks fix it.
    from sklearn.metrics import roc_auc_score
    x, Y, folds = _data(n=2000)
    S = cross_validate(base_rate, x, Y, folds)
    assert roc_auc_score(Y[:, 0], S[:, 0]) < 0.5
    bins = np.where(x > 150, "hi", "lo")
    b = by_bin(Y, S, bins, ["common", "rare", "heavy"], folds)
    assert np.allclose(b.auc_macro, 0.5)
