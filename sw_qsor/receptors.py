"""Receptor-response data and features for milestone 5 (sw-enose's M2OR screening set).

sw-qsor does not depend on sw-enose: dependencies point from new to old. Everything here is
**ported** from sw-enose, which remains the authority, and reads its files from a path the
caller gives. Nothing is vendored, imported or copied into this repository.

Ported from (sw-enose at `b8b96d6`):
  - ``sw_enose/m2or_matrix.py``: the screening-matrix loader and ``observed``.
  - ``sw_enose/chem.py``: the 20 interpretable descriptors, capped at 20 by
    ``receptor-prediction-plan.md``.
  - ``demo_m2or_chem.py``: the smoothed receptor response rate and the agonist-profile
    features.

Data: M2OR (Lalis et al.), snapshot ``M2OR_20230428.csv``, fetched by sw-enose's own script
into its gitignored ``data/m2or/repo/``. Its licence is sw-enose's to state. Structures come
from sw-enose's committed ``data/m2or/odorant_structures.csv``, resolved from PubChem.

"Responsive" is the database's own flag, pooling assays, concentrations and thresholds. It is
a weaker label than an EC50, and that pooling is not modelled here either.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

SCREEN_PARAM, SINGLE = "norm_other", "mono"
PSEUDO_COUNTS = 5.0  # sw-enose's smoothing: 5 observations of the global rate

# Exactly sw-enose's set and order (sw_enose/chem.py). Ported, not imported.
DESCRIPTORS = {
    "mol_weight": Descriptors.MolWt,
    "logp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "h_donors": rdMolDescriptors.CalcNumHBD,
    "h_acceptors": rdMolDescriptors.CalcNumHBA,
    "rings": rdMolDescriptors.CalcNumRings,
    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "fraction_sp3": rdMolDescriptors.CalcFractionCSP3,
    "molar_refractivity": Crippen.MolMR,
    "n_carbon": lambda m: sum(a.GetSymbol() == "C" for a in m.GetAtoms()),
    "n_oxygen": lambda m: sum(a.GetSymbol() == "O" for a in m.GetAtoms()),
    "n_nitrogen": lambda m: sum(a.GetSymbol() == "N" for a in m.GetAtoms()),
    "n_sulfur": lambda m: sum(a.GetSymbol() == "S" for a in m.GetAtoms()),
    "n_halogen": lambda m: sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in m.GetAtoms()),
    "stereocentres": lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True)),
    "formal_charge": Chem.GetFormalCharge,
    "double_bonds": lambda m: sum(b.GetBondTypeAsDouble() == 2.0 for b in m.GetBonds()),
    "balaban_j": Descriptors.BalabanJ,
}
DESCRIPTOR_NAMES = list(DESCRIPTORS)
assert len(DESCRIPTORS) == 20


def load_m2or(path: str | Path, drop_conflicts: bool = True) -> dict:
    """Binary human screening matrix from sw-enose's M2OR snapshot.

    Returns ``{"receptors", "odorants", "names", "y"}`` with ``y`` (receptors x odorants):
    1 responsive, 0 not, NaN never screened. Pairs carrying both a responsive and a
    non-responsive record are dropped by default rather than guessed at.
    """
    hits: dict = defaultdict(set)
    display: dict = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if (r["species"].strip().lower() != "homo sapiens"
                    or r["Parameter"].strip() != SCREEN_PARAM
                    or r["Mixture"].strip() != SINGLE):
                continue
            rec, od = r["Gene ID"].strip(), r["InChI Key"].strip()
            if not rec or not od:
                continue
            hits[(rec, od)].add(r["Responsive"].strip())
            display.setdefault(od, r["Name"].strip())
    if drop_conflicts:
        hits = {k: v for k, v in hits.items() if len(v) == 1}
    receptors = sorted({k[0] for k in hits})
    odorants = sorted({k[1] for k in hits})
    ri = {r: i for i, r in enumerate(receptors)}
    oi = {o: j for j, o in enumerate(odorants)}
    y = np.full((len(receptors), len(odorants)), np.nan)
    for (rec, od), v in hits.items():
        y[ri[rec], oi[od]] = 1.0 if "1" in v else 0.0
    return {"receptors": np.array(receptors), "odorants": np.array(odorants),
            "names": np.array([display[o] for o in odorants]), "y": y}


def observed(m: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(receptor index, odorant index, label) for every screened pair."""
    r, o = np.where(~np.isnan(m["y"]))
    return r, o, m["y"][r, o]


def load_structures(path: str | Path) -> dict:
    """``{InChIKey: SMILES}`` from sw-enose's committed table (``#`` starts a whole-line
    comment only; it is also the SMILES triple bond)."""
    with open(path) as f:
        rows = list(csv.DictReader(ln for ln in f if not ln.startswith("#")))
    return {r["inchikey"]: r["smiles"] for r in rows if r["smiles"]}


def descriptors(smiles: str) -> np.ndarray:
    """sw-enose's 20-descriptor vector. Raises ``ValueError`` on an unparseable SMILES."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"unparseable SMILES {smiles!r}")
    return np.array([float(f(m)) for f in DESCRIPTORS.values()])


def receptor_rate(R: np.ndarray, Y: np.ndarray, train: np.ndarray, n_rec: int) -> np.ndarray:
    """Smoothed per-receptor response rate from training pairs only (sw-enose's form:
    ``PSEUDO_COUNTS`` observations of the training-fold global rate as the prior)."""
    prior = Y[train].mean()
    s = np.full(n_rec, prior * PSEUDO_COUNTS)
    c = np.full(n_rec, PSEUDO_COUNTS)
    np.add.at(s, R[train], Y[train])
    np.add.at(c, R[train], 1.0)
    return s / c


def agonist_profile(F: np.ndarray, R: np.ndarray, O: np.ndarray, Y: np.ndarray,
                    train: np.ndarray, n_rec: int) -> np.ndarray:
    """sw-enose's third model's features: for each pair, the dot product, distance and cosine
    between the molecule and the mean (standardised) feature vector of the molecules its
    receptor responded to **in the training fold**.

    ``F`` is (molecules, features). Receptors with no training agonist get the mean of all
    responsive training molecules. Returns (pairs, 3). Standardisation uses ``F`` as a whole,
    which carries no label information.
    """
    sd = F.std(0)
    Z = (F - F.mean(0)) / np.where(sd > 0, sd, 1.0)
    pos = train & (Y == 1)
    overall = Z[O[pos]].mean(0) if pos.any() else np.zeros(Z.shape[1])
    s = np.zeros((n_rec, Z.shape[1]))
    c = np.zeros(n_rec)
    np.add.at(s, R[pos], Z[O[pos]])
    np.add.at(c, R[pos], 1.0)
    prof = np.where(c[:, None] > 0, s / np.maximum(c, 1)[:, None], overall)
    z, p = Z[O], prof[R]
    dot = np.einsum("ij,ij->i", z, p)
    dist = np.linalg.norm(z - p, axis=1)
    cos = dot / np.maximum(np.linalg.norm(z, axis=1) * np.linalg.norm(p, axis=1), 1e-9)
    return np.column_stack([dot, dist, cos])
