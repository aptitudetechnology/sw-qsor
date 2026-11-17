import os
from pathlib import Path

import numpy as np
import pytest

from sw_qsor.receptors import (
    DESCRIPTOR_NAMES,
    agonist_profile,
    descriptors,
    load_m2or,
    load_structures,
    observed,
    receptor_rate,
)

HEADER = "species;Parameter;Mixture;Gene ID;InChI Key;Responsive;Name"


def _m2or(tmp_path, rows):
    p = tmp_path / "m2or.csv"
    p.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return p


def test_load_m2or_filters_and_shapes(tmp_path):
    rows = [
        "Homo sapiens;norm_other;mono;OR1A1;AAA;1;alpha",
        "Homo sapiens;norm_other;mono;OR1A1;BBB;0;beta",
        "Homo sapiens;norm_other;mono;OR2B6;AAA;0;alpha",
        "Mus musculus;norm_other;mono;OR9Z9;AAA;1;alpha",   # wrong species
        "Homo sapiens;ec50;mono;OR9Z9;AAA;1;alpha",         # not screening
        "Homo sapiens;norm_other;mix;OR9Z9;AAA;1;alpha",    # mixture
    ]
    m = load_m2or(_m2or(tmp_path, rows))
    assert m["receptors"].tolist() == ["OR1A1", "OR2B6"]
    assert m["odorants"].tolist() == ["AAA", "BBB"]
    assert m["names"].tolist() == ["alpha", "beta"]
    assert m["y"][0].tolist() == [1.0, 0.0]
    assert m["y"][1, 0] == 0.0 and np.isnan(m["y"][1, 1])
    R, O, Y = observed(m)
    assert len(Y) == 3 and Y.sum() == 1


def test_conflicting_pairs_are_dropped_not_guessed(tmp_path):
    rows = ["Homo sapiens;norm_other;mono;OR1A1;AAA;1;alpha",
            "Homo sapiens;norm_other;mono;OR1A1;AAA;0;alpha",
            "Homo sapiens;norm_other;mono;OR1A1;BBB;1;beta"]
    m = load_m2or(_m2or(tmp_path, rows))
    assert m["odorants"].tolist() == ["BBB"]
    kept = load_m2or(_m2or(tmp_path, rows), drop_conflicts=False)
    assert kept["y"][0].tolist() == [1.0, 1.0]  # "1" present wins when not dropping


def test_load_structures_keeps_hash_in_smiles(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("# a comment line\ninchikey,name,pubchem_cid,smiles\n"
                 "KKK,acetonitrile,6342,CC#N\nNOS,unresolved,,\n")
    s = load_structures(p)
    assert s == {"KKK": "CC#N"}


def test_descriptors_shape_and_known_values():
    d = descriptors("CCO")
    assert len(d) == len(DESCRIPTOR_NAMES) == 20
    v = dict(zip(DESCRIPTOR_NAMES, d))
    assert v["heavy_atoms"] == 3 and v["n_carbon"] == 2 and v["n_oxygen"] == 1
    assert v["h_donors"] == 1 and v["rings"] == 0 and abs(v["mol_weight"] - 46.07) < 0.01
    with pytest.raises(ValueError, match="unparseable"):
        descriptors("not a molecule")


def test_receptor_rate_uses_training_only_and_smooths():
    R = np.array([0, 0, 0, 1, 1])
    Y = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    train = np.array([True, True, True, True, False])  # last pair held out
    r = receptor_rate(R, Y, train, n_rec=3)
    prior = Y[train].mean()  # 0.75
    assert abs(r[0] - (prior * 5 + 3) / 8) < 1e-12
    assert abs(r[1] - (prior * 5 + 0) / 6) < 1e-12
    assert abs(r[2] - prior) < 1e-12  # unseen receptor falls back to the prior
    # changing a held-out label must not change the rate
    Y2 = Y.copy()
    Y2[4] = 0.0
    assert np.allclose(r, receptor_rate(R, Y2, train, n_rec=3))


def test_agonist_profile_shape_fallback_and_cosine():
    F = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]])
    R = np.array([0, 0, 1, 1])
    O = np.array([0, 1, 2, 3])
    Y = np.array([1.0, 0.0, 0.0, 0.0])
    train = np.array([True, True, True, False])
    A = agonist_profile(F, R, O, Y, train, n_rec=2)
    assert A.shape == (4, 3)
    assert np.all(np.abs(A[:, 2]) <= 1 + 1e-9)
    # receptor 0's only training agonist is molecule 0, so that pair's cosine is 1
    assert abs(A[0, 2] - 1.0) < 1e-9 and abs(A[0, 1]) < 1e-9
    # receptor 1 has no training agonist, so it falls back to the mean responsive molecule
    assert np.isfinite(A[2:]).all()


SW_ENOSE = Path(os.environ.get("SW_ENOSE_DIR", Path.home() / "sw-enose"))
M2OR = SW_ENOSE / "data/m2or/repo/M2OR_20230428.csv"


@pytest.mark.skipif(not M2OR.exists(), reason="sw-enose checkout with fetched M2OR not present")
def test_port_matches_sw_enose_on_the_real_file():
    """The port must agree with sw-enose's own loader and descriptors, exactly."""
    import subprocess
    import sys

    code = (
        "import json,sys,numpy as np;"
        "sys.path.insert(0, sys.argv[1]);"
        "from sw_enose.m2or_matrix import load_matrix, observed;"
        "from sw_enose.chem import load_structures, descriptors, NAMES;"
        "m=load_matrix(sys.argv[2]); R,O,Y=observed(m);"
        "s=load_structures(sys.argv[3]);"
        "k=sorted(s)[:25];"
        "print(json.dumps({'rec':m['receptors'].tolist(),'od':m['odorants'].tolist(),"
        "'npairs':int(len(Y)),'pos':float(Y.sum()),'names':NAMES,"
        "'desc':[descriptors(s[x]).tolist() for x in k],'keys':k}))"
    )
    out = subprocess.run([sys.executable, "-c", code, str(SW_ENOSE), str(M2OR),
                          str(SW_ENOSE / "data/m2or/odorant_structures.csv")],
                         capture_output=True, text=True, env={**os.environ,
                                                              "PYTHONDONTWRITEBYTECODE": "1"})
    assert out.returncode == 0, out.stderr[-2000:]
    import json
    ref = json.loads(out.stdout)

    m = load_m2or(M2OR)
    R, O, Y = observed(m)
    assert m["receptors"].tolist() == ref["rec"]
    assert m["odorants"].tolist() == ref["od"]
    assert len(Y) == ref["npairs"] and Y.sum() == ref["pos"]
    assert DESCRIPTOR_NAMES == ref["names"]
    s = load_structures(SW_ENOSE / "data/m2or/odorant_structures.csv")
    for key, row in zip(ref["keys"], ref["desc"]):
        np.testing.assert_allclose(descriptors(s[key]), row)
