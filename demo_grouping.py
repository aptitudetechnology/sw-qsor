"""Reproduce the grouping comparison behind readme rule 3, on GS+LF.

Usage: python demo_grouping.py [--data-dir DIR]

Structure only: no labels are read and no model is fitted, so nothing here can have been tuned
to an accuracy figure. The rejected candidates (Murcko, Murcko+Butina, Butina, carbon skeleton)
are implemented here rather than in the package because nothing else uses them.
See reports/02_grouping.md for the write-up.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from sw_qsor.datasets import load_gslf
from sw_qsor.identity import key_table
from sw_qsor.splits import (
    assign_folds,
    cross_fold_nn_similarity,
    group_size_summary,
    morgan_fingerprints,
    nn_bin_labels,
    similarity_components,
)

RDLogger.DisableLog("rdApp.*")
REPO_ROOT = Path(__file__).resolve().parent


def butina(fps, idx, cutoff):
    idx = list(idx)
    d = []
    for i in range(1, len(idx)):
        d.extend(1 - np.asarray(DataStructs.BulkTanimotoSimilarity(fps[idx[i]],
                                                                   [fps[j] for j in idx[:i]])))
    lab = np.empty(len(idx), int)
    for k, members in enumerate(Butina.ClusterData(d, len(idx), cutoff, isDistData=True)):
        lab[list(members)] = k
    return lab


def carbon_skeleton(mol):
    """Largest all-carbon fragment after deleting heteroatoms, with bond orders erased."""
    em = Chem.RWMol(mol)
    for i in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 6), reverse=True):
        em.RemoveAtom(i)
    frags = Chem.GetMolFrags(em.GetMol(), asMols=True, sanitizeFrags=False)
    if not frags:
        return "<no carbon>"
    f = Chem.RWMol(max(frags, key=lambda x: x.GetNumAtoms()))
    for b in f.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in f.GetAtoms():
        a.SetIsAromatic(False)
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    return Chem.MolToSmiles(f.GetMol())


def row(name, groups, folds, fps, murcko):
    nn = cross_fold_nn_similarity(fps, folds)
    m = np.asarray(murcko, dtype=object)
    seen = [np.mean([s in set(m[folds != k]) for s in m[(folds == k) & (m != "")]])
            for k in range(5) if ((folds == k) & (m != "")).any()]
    largest = "-" if groups is None else f"{group_size_summary(groups)['largest_frac']:.1%}"
    print(f"{name:32s} {largest:>7s} {str(np.bincount(folds).tolist()):>28s} "
          f"{np.mean(nn >= 0.999):6.1%} {np.mean(nn >= 0.7):6.1%} {np.mean(nn > 0.5):6.1%} "
          f"{np.median(nn):6.2f} {np.mean(seen):7.1%}")


def main(data_dir):
    df, _ = load_gslf(data_dir / "gslf")
    smiles = df.smiles.tolist()
    keys = key_table(smiles).key.tolist()  # rule 1: duplicate records share a group
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = morgan_fingerprints(smiles)
    n = len(fps)
    murcko = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]

    print(f"GS+LF: {n} molecules; Morgan r=2, 2048 bits; 5 folds.\n")
    print("Columns: largest group as % of molecules; fold sizes; then, over all molecules, the")
    print("similarity of the nearest molecule in another fold: % identical, % >= 0.7, % > 0.5,")
    print("median; finally % of ring scaffolds in a test fold that also occur in training.\n")
    print(f"{'grouping':32s} {'largest':>7s} {'fold sizes':>28s} {'ident':>6s} {'>=0.7':>6s} "
          f"{'>0.5':>6s} {'median':>6s} {'scaf.seen':>7s}")

    row("random molecules (reference)", None, np.random.default_rng(0).permutation(n) % 5,
        fps, murcko)
    row("Murcko", murcko, assign_folds(murcko, smiles), fps, murcko)

    acyc = [i for i, s in enumerate(murcko) if s == ""]
    a = list(murcko)
    for j, lab in zip(acyc, butina(fps, acyc, 0.7)):
        a[j] = f"acyclic-{lab}"
    row("A: Murcko, acyclics Butina d<0.7", a, assign_folds(a, smiles), fps, murcko)

    b = butina(fps, range(n), 0.7)
    row("B: Butina all, d<0.7", b, assign_folds(b, smiles), fps, murcko)

    c = [carbon_skeleton(m) for m in mols]
    row("C: carbon skeleton", c, assign_folds(c, smiles), fps, murcko)

    for t in (0.65, 0.70, 0.75):
        d = similarity_components(fps, t, keys=keys)
        row(f"D: components sim>={t:.2f}", d, assign_folds(d, smiles), fps, murcko)

    d = similarity_components(fps, keys=keys)
    folds = assign_folds(d, smiles)
    s = group_size_summary(d)
    print(f"\nChosen (rule 3): D at 0.70. {s['groups']} groups; largest {s['largest']} "
          f"({s['largest_frac']:.1%}); top 10 {s['top10']}; singletons {s['singletons']}.")
    print(f"Group-size histogram: {s['size_hist']}")
    big = Counter(d.tolist()).most_common(3)
    for g, size in big:
        members = [smiles[i] for i in np.nonzero(d == g)[0][:6]]
        print(f"  group of {size}: e.g. {', '.join(members)}")
    nn = cross_fold_nn_similarity(fps, folds)
    bins = Counter(nn_bin_labels(nn).tolist())
    print("Test molecules by nearest-training-neighbour similarity: "
          + ", ".join(f"{k}: {bins.get(k, 0)}" for k in sorted(bins)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    main(ap.parse_args().data_dir)
