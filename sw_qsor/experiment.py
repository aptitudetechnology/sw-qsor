"""The GS+LF experiment set-up, built once so every run uses identical molecules and folds.

Readme rules 1 and 3: duplicate records merged by connectivity key (labels unioned), groups
from similarity components at 0.70 with identity keys linked, groups assigned to 5 folds.
Both the floor (demo_dataset_survey.py) and milestone 2 build their data through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from sw_qsor.datasets import load_gslf
from sw_qsor.identity import collapse_duplicates, key_table
from sw_qsor.splits import (
    assign_folds,
    cross_fold_nn_similarity,
    morgan_fingerprints,
    nn_bin_labels,
    similarity_components,
)


@dataclass(frozen=True)
class Experiment:
    smiles: np.ndarray  # (n,) str
    keys: list[str]  # InChIKey connectivity blocks
    labels: list[str]
    Y: np.ndarray  # (n, labels) 0/1
    mw: np.ndarray  # (n,) RDKit MolWt
    fps: list  # Morgan bit vectors (sw_qsor.splits defaults)
    groups: np.ndarray
    folds: np.ndarray
    nn: np.ndarray  # nearest-training-neighbour Tanimoto under ``folds``
    nn_bins: np.ndarray  # nn_bin_labels(nn)

    def label(self, name: str) -> np.ndarray:
        return self.Y[:, self.labels.index(name)]


def gslf_experiment(archive_dir: str | Path) -> Experiment:
    """Build the GS+LF experiment from the directory holding the GS+LF csv."""
    df, labels = load_gslf(archive_dir)
    keys = key_table(df.smiles).key.tolist()
    keep, Y = collapse_duplicates(keys, df[labels].to_numpy())
    smiles = df.smiles.to_numpy()[keep]
    keys = [keys[i] for i in keep]
    mw = np.array([Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in smiles])
    fps = morgan_fingerprints(smiles)
    groups = similarity_components(fps, keys=keys)
    folds = assign_folds(groups, keys)
    nn = cross_fold_nn_similarity(fps, folds)
    return Experiment(smiles=smiles, keys=keys, labels=labels, Y=Y, mw=mw, fps=fps,
                      groups=groups, folds=folds, nn=nn, nn_bins=nn_bin_labels(nn))


def independence(smiles_by_dataset: dict, anchor: str) -> dict:
    """Classify every dataset's molecules relative to ``anchor`` under rule 3.

    All datasets are unioned and deduplicated by connectivity key (the anchor's SMILES
    represent shared keys, then datasets in the order given), grouped by similarity
    components, and each non-anchor molecule is ``inside`` the anchor, ``near`` (absent but
    sharing a group with an anchor molecule) or ``independent``.

    Returns ``{"keys": {name: set}, "group_of": {key: group}, "smiles_of": {key: smiles},
    "classes": {name: {"inside", "near", "independent": set}}}``.
    """
    import pandas as pd

    names = [anchor] + [n for n in smiles_by_dataset if n != anchor]
    tables = {n: key_table(smiles_by_dataset[n]) for n in names}
    sets = {n: set(t.key.dropna()) for n, t in tables.items()}
    smiles_of: dict = {}
    for n in names:
        for s, k in zip(tables[n].smiles, tables[n].key):
            if pd.notna(k):
                smiles_of.setdefault(k, s)
    keys = list(smiles_of)
    groups = similarity_components(morgan_fingerprints([smiles_of[k] for k in keys]))
    group_of = dict(zip(keys, groups.tolist()))
    anchor_groups = {group_of[k] for k in sets[anchor]}
    classes = {}
    for n in names[1:]:
        inside = sets[n] & sets[anchor]
        near = {k for k in sets[n] - inside if group_of[k] in anchor_groups}
        classes[n] = {"inside": inside, "near": near, "independent": sets[n] - inside - near}
    return {"keys": sets, "group_of": group_of, "smiles_of": smiles_of, "classes": classes,
            "anchor_groups": anchor_groups}
