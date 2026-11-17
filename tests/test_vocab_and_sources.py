import numpy as np
import pytest

from sw_qsor.datasets import load_arctander_labels, load_openpom_source
from sw_qsor.evaluate import joint_group_bootstrap
from sw_qsor.vocab import arctander_mapping


def test_mapping_exact_variant_unmapped_and_unknown():
    m = arctander_mapping(["floral", "musky", "lilac"], ["floral", "musk"])
    assert m == {"floral": "floral", "musky": "musk"}
    with pytest.raises(ValueError, match="not covered"):
        arctander_mapping(["glorious"], ["floral"])
    with pytest.raises(ValueError, match="variant target"):
        arctander_mapping(["musky"], ["floral"])


def test_openpom_source_loader(tmp_path):
    (tmp_path / "x.csv").write_text("IsomericSMILES,Updated_Desc,a,b\nCCO,a,1,0\n")
    df, labels = load_openpom_source(tmp_path, "x.csv")
    assert labels == ["a", "b"] and df.smiles.tolist() == ["CCO"]


def test_arctander_loader_unions_by_cid_drops_cid0_flags_unlabelled(tmp_path):
    (tmp_path / "behavior_1.csv").write_text("Stimulus,floral,woody\n1,1,0\n2,0,1\n3,1,1\n4,0,0\n")
    (tmp_path / "stimuli.csv").write_text(
        "Stimulus,ChemicalName,CAS,new_CID\n1,a,x,702\n2,b,x,702\n3,c,x,0\n4,d,x,887\n")
    (tmp_path / "molecules.csv").write_text(
        "CID,MolecularWeight,IsomericSMILES,IUPACName,name\n702,46,CCO,e,e\n887,32,CO,m,m\n")
    per, labels, counts = load_arctander_labels(tmp_path)
    assert labels == ["floral", "woody"]
    assert per.loc[702, ["floral", "woody"]].tolist() == [1, 1]
    assert bool(per.loc[702, "labelled"]) and not bool(per.loc[887, "labelled"])
    assert counts == {"stimuli": 4, "stimuli_without_structure": 1, "molecules": 2,
                      "unlabelled_molecules": 1}


def test_joint_group_bootstrap_resamples_groups():
    x = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    g = np.array([0, 0, 1, 1, 2, 2])
    r = joint_group_bootstrap(lambda i: x[i].mean(), g, n_boot=500)
    assert r["estimate"] == pytest.approx(4 / 6) and r["groups"] == 3
    assert 0 <= r["lo"] < r["estimate"] < r["hi"] <= 1
    # resampling groups, not molecules: the 2-molecule groups move together, so an interval
    # this wide (3 effective units) is expected; molecule-level resampling would be narrower
    assert r["hi"] - r["lo"] > 0.5
