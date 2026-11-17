"""Molecule identity: what counts as "the same molecule" across datasets (readme rule 1).

The primary key is the **InChIKey connectivity block** (first 14 characters). It ignores
stereochemistry, which is necessary because GS+LF and Keller 2016 distribute SMILES without
stereo; matching on full InChIKeys would make stereo-labelled datasets look disjoint from them.
It also ignores protonation (the InChIKey's last character), which merges GS+LF's two
neutral/ionised duplicate pairs.

The cost is that stereoisomers merge: geraniol and nerol, or (+)- and (-)-carvone, which smell
different, become one molecule. :func:`key_table` reports the full key alongside, so the cost
can be counted.

The optional ``parent`` standardisation additionally strips counter-ions and neutralises
(RDKit ``ChargeParent``), so sodium acetate matches acetic acid. That is a looser claim of
identity, since a salt and its free acid need not smell alike, so it is a sensitivity check,
not the primary key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

CONNECTIVITY_LEN = 14


def _mol(smiles: str, parent: bool):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    if parent:
        m = rdMolStandardize.ChargeParent(m)
    return m


def inchikey(smiles: str, *, parent: bool = False) -> str | None:
    """Full InChIKey, or ``None`` if the SMILES does not parse."""
    if not isinstance(smiles, str):
        return None
    m = _mol(smiles, parent)
    return Chem.MolToInchiKey(m) if m is not None else None


def connectivity_key(smiles: str, *, parent: bool = False) -> str | None:
    """InChIKey connectivity block: identity ignoring stereo and protonation."""
    k = inchikey(smiles, parent=parent)
    return k[:CONNECTIVITY_LEN] if k else None


def key_table(smiles: Iterable, *, parent: bool = False) -> pd.DataFrame:
    """One row per input: ``smiles``, ``inchikey``, ``key`` (connectivity block; None if
    unparseable or missing), in input order."""
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    try:
        rows = []
        for s in smiles:
            k = inchikey(s, parent=parent)
            rows.append((s, k, k[:CONNECTIVITY_LEN] if k else None))
    finally:
        lg.setLevel(RDLogger.WARNING)
    return pd.DataFrame(rows, columns=["smiles", "inchikey", "key"])


def overlap_matrix(sets: Mapping[str, set]) -> pd.DataFrame:
    """Shared-molecule counts between every pair of sets; the diagonal is each set's size."""
    names = list(sets)
    return pd.DataFrame([[len(sets[a] & sets[b]) for b in names] for a in names],
                        index=names, columns=names)


def collapse_duplicates(keys: Sequence, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge records that share a key: keep the first record's index, OR the label rows.

    Returns ``(keep, merged)``: indices of the kept records in input order, and their label
    matrix with each duplicate's labels unioned in. Rows with a missing key are dropped.
    """
    labels = np.asarray(labels)
    first: dict = {}
    keep, merged = [], []
    for i, k in enumerate(keys):
        if not isinstance(k, str):
            continue
        if k in first:
            merged[first[k]] = merged[first[k]] | labels[i]
        else:
            first[k] = len(keep)
            keep.append(i)
            merged.append(labels[i].copy())
    return np.array(keep, dtype=int), np.array(merged).reshape(len(keep), *labels.shape[1:])
