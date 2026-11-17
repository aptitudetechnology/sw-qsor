"""Loader tests on tiny fixtures that mimic each upstream schema."""

import pytest

from sw_qsor.datasets import GSLF_FILE, load_gslf, load_pyrfume_molecules


def test_load_gslf(tmp_path):
    (tmp_path / GSLF_FILE).write_text(
        "nonStereoSMILES,descriptors,fruity,green\nCCO,fruity,1,0\nCCCO,fruity;green,1,1\n"
    )
    df, labels = load_gslf(tmp_path)
    assert labels == ["fruity", "green"]
    assert list(df.smiles) == ["CCO", "CCCO"]
    assert df["green"].tolist() == [0, 1] and str(df["green"].dtype) == "int8"


def test_pyrfume_standard_schema(tmp_path):
    (tmp_path / "molecules.csv").write_text(
        "CID,MolecularWeight,IsomericSMILES,IUPACName,name\n"
        "702,46.07,CCO,ethanol,ethanol\n-5,30.0,C=O,,local id\n"
    )
    m = load_pyrfume_molecules(tmp_path)
    assert m.index.tolist() == [702, -5]
    assert m.loc[702, "smiles"] == "CCO"
    assert set(m.smiles_source) == {"IsomericSMILES"}


def test_pyrfume_keller_schema(tmp_path):
    (tmp_path / "molecules.csv").write_text(
        '"","CID","OdorName","CAS","CanonicalSMILES","MolecularWeight"\n'
        '"1","702","Ethanol","64-17-5","CCO","46.07"\n'
    )
    m = load_pyrfume_molecules(tmp_path)
    assert list(m.columns) == ["name", "CAS", "smiles", "MolecularWeight", "smiles_source"]
    assert m.loc[702, "name"] == "Ethanol"
    assert m.loc[702, "smiles_source"] == "CanonicalSMILES"


def test_pyrfume_without_smiles_is_an_error(tmp_path):
    (tmp_path / "molecules.csv").write_text("CID,name\n1,x\n")
    with pytest.raises(ValueError, match="no SMILES"):
        load_pyrfume_molecules(tmp_path)
