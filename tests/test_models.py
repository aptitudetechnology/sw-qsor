import numpy as np
from rdkit import DataStructs

from sw_qsor.evaluate import fold_aucs, group_bootstrap_margin, within_fold_ranks
from sw_qsor.models import fingerprint_matrix, random_forest, tanimoto_knn
from sw_qsor.splits import morgan_fingerprints

SMI = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CC(=O)O", "CCC(=O)O"]


def test_fingerprint_matrix_matches_bits():
    fps = morgan_fingerprints(SMI)
    X = fingerprint_matrix(fps)
    assert X.shape == (len(SMI), 2048)
    for i, fp in enumerate(fps):
        assert set(np.nonzero(X[i])[0]) == set(fp.GetOnBits())


def test_tanimoto_knn_matches_brute_force():
    fps = morgan_fingerprints(SMI)
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    got = tanimoto_knn(fps[:6], y[:6], fps[6:], k=3)
    for i, fp in enumerate(fps[6:]):
        s = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps[:6]))
        assert got[i] == y[:6][np.argsort(-s, kind="stable")[:3]].mean()


def test_random_forest_learns_a_separable_feature_and_accepts_1d():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 400)
    y = (x > 0.5).astype(int)
    p = random_forest(x[:300], y[:300], x[300:])
    assert ((p > 0.5) == y[300:]).mean() > 0.95


def test_fold_aucs_and_bootstrap():
    rng = np.random.default_rng(1)
    n = 1000
    folds = rng.integers(0, 5, n)
    groups = rng.integers(0, 300, n)
    y = rng.integers(0, 2, n)
    good = y + rng.normal(0, 0.7, n)
    noise = rng.normal(0, 1, n)
    a = fold_aucs(y, good, folds)
    assert a.shape == (5,) and (a > 0.7).all()
    mask = np.ones(n, bool)
    R = within_fold_ranks(np.column_stack([good, noise]), folds)
    r = group_bootstrap_margin(y, R[:, 0], [R[:, 1]], groups, mask, n_boot=200)
    assert r["lo"] > 0.1 and r["lo"] < r["margin"] < r["hi"]
    r0 = group_bootstrap_margin(y, R[:, 1], [R[:, 1]], groups, mask, n_boot=200)
    assert r0["margin"] == 0 and r0["lo"] == 0 == r0["hi"]


def test_shared_neighbours_equal_tanimoto_knn():
    from sw_qsor.models import tanimoto_neighbours
    fps = morgan_fingerprints(SMI)
    Y = np.array([[1, 0], [1, 1], [1, 0], [0, 0], [0, 1], [0, 0], [1, 1], [0, 0]])
    idx = tanimoto_neighbours(fps[:6], fps[6:], k=3)
    for j in range(2):
        np.testing.assert_array_equal(Y[:6][idx, j].mean(1), tanimoto_knn(fps[:6], Y[:6, j], fps[6:], k=3))


def test_auc_columns_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score

    from sw_qsor.evaluate import auc_columns
    rng = np.random.default_rng(4)
    Y = rng.integers(0, 2, (300, 6))
    Y[:, 5] = 0
    S = np.round(rng.normal(0, 1, (300, 6)) + Y, 1)  # rounding creates ties
    got = auc_columns(Y, S)
    for j in range(5):
        assert abs(got[j] - roc_auc_score(Y[:, j], S[:, j])) < 1e-12
    assert np.isnan(got[5])


def test_weighted_lookup_matches_brute_force_and_breaks_ties():
    from sw_qsor.models import tanimoto_weighted_lookup
    fps = morgan_fingerprints(SMI)
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    got = tanimoto_weighted_lookup(fps[:6], y[:6], fps[6:], k=4)
    for i, fp in enumerate(fps[6:]):
        s = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps[:6]))
        nn_ = np.argsort(-s, kind="stable")[:4]
        assert abs(got[i] - (s[nn_] @ y[:6][nn_]) / s[nn_].sum()) < 1e-12
    multi = tanimoto_weighted_lookup(fps[:6], np.c_[y[:6], 1 - y[:6]], fps[6:], k=4)
    np.testing.assert_allclose(multi[:, 0], got)
