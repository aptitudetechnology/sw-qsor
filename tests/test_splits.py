import numpy as np
import pytest
from rdkit import DataStructs

from sw_qsor.splits import (
    assign_folds,
    cross_fold_nn_similarity,
    group_size_summary,
    morgan_fingerprints,
    nn_bin_labels,
    similarity_components,
)

# Two homologous ester series, some aromatics, and odd ones out.
SMILES = [
    "CCCCCCOC(C)=O", "CCCCCCCOC(C)=O", "CCCCCCCCOC(C)=O", "CCCCCOC(C)=O",
    "CCCCCCO", "CCCCCCCO", "CCCCCCCCO",
    "c1ccccc1C=O", "Cc1ccccc1C=O", "COc1ccccc1C=O", "Oc1ccc(C=O)cc1",
    "CC(C)=CCCC(C)=CCO", "CC1=CCC(CC1)C(C)=C", "O=C1CCCCC1", "S", "CCSC",
]


@pytest.fixture(scope="module")
def fps():
    return morgan_fingerprints(SMILES)


def _pairwise(fps):
    return np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])


@pytest.mark.parametrize("t", [0.3, 0.5, 0.7, 0.9])
def test_no_pair_at_or_above_threshold_crosses_groups(fps, t):
    g = similarity_components(fps, t)
    sim = _pairwise(fps)
    cross = g[:, None] != g[None, :]
    assert (sim[cross] < t).all()


def test_components_are_order_independent(fps):
    g = similarity_components(fps, 0.5)
    perm = np.random.default_rng(1).permutation(len(fps))
    gp = similarity_components([fps[i] for i in perm], 0.5)
    # same partition: molecules i, j together in one iff together in the other
    same = g[:, None] == g[None, :]
    same_p = np.empty_like(same)
    same_p[np.ix_(perm, perm)] = gp[:, None] == gp[None, :]
    assert (same == same_p).all()


def test_folds_keep_groups_whole_balance_and_are_order_independent(fps):
    g = similarity_components(fps, 0.5)
    f = assign_folds(g, SMILES, k=3)
    for grp in set(g):
        assert len(set(f[g == grp])) == 1
    sizes = np.bincount(f, minlength=3)
    assert sizes.max() - sizes.min() <= max(np.bincount(g))
    perm = np.random.default_rng(2).permutation(len(fps))
    gp = similarity_components([fps[i] for i in perm], 0.5)
    fp_ = assign_folds(gp, [SMILES[i] for i in perm], k=3)
    assert (fp_ == f[perm]).all()


def test_cross_fold_nn_respects_threshold_and_matches_brute_force(fps):
    t = 0.5
    g = similarity_components(fps, t)
    f = assign_folds(g, SMILES, k=3)
    nn = cross_fold_nn_similarity(fps, f)
    sim = _pairwise(fps)
    brute = np.where(f[:, None] != f[None, :], sim, -1).max(1)
    np.testing.assert_allclose(nn, brute)
    assert (nn < t).all()


def test_keys_force_records_of_one_molecule_into_one_group():
    # neutral and ionised forms: same molecule, fingerprint similarity 0.31
    smi = ["CC1=NC(C)OC1C", "CC1=[NH+]C(C)OC1C", "CCO"]
    fps_ = morgan_fingerprints(smi)
    g = similarity_components(fps_, 0.7)
    assert g[0] != g[1]
    g = similarity_components(fps_, 0.7, keys=["A", "A", "B"])
    assert g[0] == g[1] != g[2]
    with pytest.raises(ValueError):
        similarity_components(fps_, 0.7, keys=["A"])


def test_nn_bins():
    lab = nn_bin_labels(np.array([0.1, 0.4, 0.45, 0.69, 0.7, 0.95]))
    assert lab.tolist() == ["0.0-0.4", "0.4-0.5", "0.4-0.5", "0.6-0.7", ">=0.7", ">=0.7"]


def test_group_size_summary():
    s = group_size_summary([0, 0, 0, 1, 1, 2])
    assert s["groups"] == 3 and s["largest"] == 3 and s["largest_frac"] == 0.5
    assert s["singletons"] == 1 and s["size_hist"]["2-4"] == 2


def test_unparseable_smiles_is_an_error():
    with pytest.raises(ValueError, match="unparseable"):
        morgan_fingerprints(["CCO", "not a smiles"])
