from sw_qsor.identity import connectivity_key, inchikey, key_table, overlap_matrix

GERANIOL = "CC(C)=CCC/C(C)=C/CO"
NEROL = "CC(C)=CCC/C(C)=C\\CO"


def test_stereoisomers_share_connectivity_key_but_not_full_key():
    assert inchikey(GERANIOL) != inchikey(NEROL)
    assert connectivity_key(GERANIOL) == connectivity_key(NEROL) == connectivity_key(
        "CC(C)=CCCC(C)=CCO")


def test_protonation_states_share_connectivity_key():
    # the GS+LF duplicate pair
    assert connectivity_key("CC1=NC(C)OC1C") == connectivity_key("CC1=[NH+]C(C)OC1C")


def test_salt_matches_free_acid_only_with_parent():
    salt, acid = "CC(=O)[O-].[Na+]", "CC(=O)O"
    assert connectivity_key(salt) != connectivity_key(acid)
    assert connectivity_key(salt, parent=True) == connectivity_key(acid)


def test_bad_and_missing_smiles_give_none():
    t = key_table(["CCO", "not smiles", None])
    assert t.key.isna().tolist() == [False, True, True]


def test_overlap_matrix():
    m = overlap_matrix({"a": {1, 2, 3}, "b": {2, 3, 4, 5}})
    assert m.loc["a", "a"] == 3 and m.loc["b", "b"] == 4
    assert m.loc["a", "b"] == m.loc["b", "a"] == 2


def test_collapse_duplicates_unions_labels_and_drops_missing():
    import numpy as np

    from sw_qsor.identity import collapse_duplicates
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.int8)
    keep, m = collapse_duplicates(["A", "B", "A", None], y)
    assert keep.tolist() == [0, 1]
    assert m.tolist() == [[1, 0, 1], [0, 1, 0]]


def test_independence_classes():
    from sw_qsor.experiment import independence
    sm = {"A": ["CCCCCCOC(C)=O", "c1ccccc1C=O"],
          "B": ["CCCCCCOC(C)=O", "CCCCCCCOC(C)=O", "ClC(Cl)(Cl)Cl"]}
    c = independence(sm, "A")["classes"]["B"]
    assert len(c["inside"]) == 1 and len(c["near"]) == 1 and len(c["independent"]) == 1
