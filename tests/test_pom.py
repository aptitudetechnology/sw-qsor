import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sw_qsor.pom import (  # noqa: E402
    ATOM_FDIM, BOND_FDIM, POM, TrainConfig, collate, featurize, imbalance_weights, pom_loss,
    predict, regularization, train,
)

SMI = ["CCO", "c1ccccc1O", "CC(=O)OCC", "S", "CSC", "C=CC(=O)O", "CC(C)=CCCC(C)=CCO", "N#CC"]


def test_feature_dimensions_match_openpom():
    assert ATOM_FDIM == 134 and BOND_FDIM == 6
    g = featurize("CC(=O)O")
    assert g.x.shape == (4, 134) and g.edge_attr.shape == (6, 6) and g.edge_index.shape == (2, 6)
    assert (g.x.sum(1) == 6).all()  # exactly one hot per one-hot block
    s = featurize("S")
    assert s.x.shape == (1, 134) and s.edge_index.shape == (2, 0)


def test_atom_features_known_values():
    g = featurize("C")  # methane carbon: valence 4, degree 4 (with Hs), 4 Hs, charge 0, C, SP3
    row = g.x[0]
    assert row[4] == 1  # valence 4
    assert row[8 + 4] == 1  # degree 4
    assert row[8 + 7 + 4] == 1  # 4 hydrogens
    assert row[8 + 7 + 6 + 4] == 1  # formal charge 0 is index 4 in [-1,-2,1,2,0]
    assert row[8 + 7 + 6 + 6 + 5] == 1  # atomic number 6 -> index 5
    assert row[8 + 7 + 6 + 6 + 101 + 2] == 1  # SP3


def test_forward_shapes_and_single_atoms_identical():
    torch.manual_seed(0)
    m = POM(n_tasks=7).eval()
    logits, emb = m(collate([featurize(s) for s in ["S", "O", "C", "CCO"]]))
    assert logits.shape == (4, 7) and emb.shape == (4, 256)
    # single heavy atoms: readout input is all zero, so identical outputs
    assert torch.allclose(logits[0], logits[1]) and torch.allclose(logits[1], logits[2])
    assert not torch.allclose(logits[0], logits[3])


def test_batching_does_not_change_predictions():
    torch.manual_seed(0)
    m = POM(n_tasks=3).eval()
    gs = [featurize(s) for s in SMI]
    together = m(collate(gs))[0]
    alone = torch.cat([m(collate([g]))[0] for g in gs])
    assert torch.allclose(together, alone, atol=1e-5)


def test_loss_equals_bce_on_2l_minus_1():
    l = torch.tensor([[0.3, -1.2], [2.0, 0.1]])
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    w = torch.tensor([0.5, 2.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(2 * l - 1, y, reduction="none")
    assert torch.allclose(pom_loss(l, y, w), (bce * w).sum(1).mean())


def test_imbalance_weights_favour_common_labels_as_openpom_code_does():
    w = imbalance_weights(np.array([[1, 1], [1, 0], [1, 0], [1, 0]]))
    assert w[0] > w[1] and abs(float(w[0]) - np.log(2)) < 1e-6


def test_regularization_excludes_biases():
    m = POM(n_tasks=2)
    names = [n for n, _ in m.named_parameters() if "bias" not in n]
    assert "mpnn.gnn_layer.bias" not in names and "mpnn.gru.weight_ih_l0" in names
    assert regularization(m, 1.0).item() > 0


def test_training_is_deterministic():
    gs = [featurize(s) for s in SMI * 4]
    Y = np.random.default_rng(0).integers(0, 2, (len(gs), 3))
    cfg = TrainConfig(epochs=2, batch_size=8, threads=2)
    a = predict(train(gs, Y, seed=3, cfg=cfg), gs)[0]
    b = predict(train(gs, Y, seed=3, cfg=cfg), gs)[0]
    c = predict(train(gs, Y, seed=4, cfg=cfg), gs)[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)
