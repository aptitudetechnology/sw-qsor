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
