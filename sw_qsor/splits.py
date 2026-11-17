"""Group-aware splits by fingerprint-similarity connected components (readme rule 3).

Molecules are linked when their Morgan fingerprints have Tanimoto similarity >= ``threshold``;
a group is a connected component of that graph (single linkage). Consequently no pair of
molecules in different groups -- hence in different folds -- is that similar. This is a
guarantee that can be checked, unlike Butina-style clustering, whose clusters depend on input
order and leave above-cutoff pairs across cluster boundaries.

Everything here is order-independent: permuting the input permutes the output and changes
nothing else. Memory is O(n) per row of similarities; no n x n matrix is held.

Written for sw-qsor as a project-independent utility; the evaluation discipline it encodes is
sw-enose's (group-held-out splits), reimplemented here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Fixed by readme rule 3. Changing them is changing the rule.
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
COMPONENT_THRESHOLD = 0.7
N_FOLDS = 5


def morgan_fingerprints(smiles: Sequence[str], *, radius: int = MORGAN_RADIUS,
                        n_bits: int = MORGAN_BITS) -> list:
    """Morgan bit-vector fingerprints. Raises ``ValueError`` naming any SMILES that won't parse."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, bad = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad.append(s)
        else:
            fps.append(gen.GetFingerprint(m))
    if bad:
        raise ValueError(f"{len(bad)} unparseable SMILES, e.g. {bad[:3]}")
    return fps


def similarity_components(fps: Sequence, threshold: float = COMPONENT_THRESHOLD,
                          *, keys: Sequence | None = None) -> np.ndarray:
    """Connected components of the graph linking pairs with Tanimoto >= ``threshold``.

    ``keys``, if given (e.g. :func:`sw_qsor.identity.connectivity_key` per row), also links
    rows with equal keys, so that two records of one molecule always share a group. This
    matters: GS+LF's neutral and ionised duplicates have fingerprint similarity of only
    0.31-0.55, and without it they could land on opposite sides of a split.

    Returns an integer group label per molecule. Labels are canonical: groups are numbered by
    the index of their first member, so the labelling is a function of the partition alone.
    """
    n = len(fps)
    rows, cols = [], []
    for i in range(1, n):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
        j = np.nonzero(s >= threshold)[0]
        rows.append(np.full(len(j), i))
        cols.append(j)
    if keys is not None:
        if len(keys) != n:
            raise ValueError("keys must have one entry per fingerprint")
        first_of: dict = {}
        for i, k in enumerate(keys):
            j = first_of.setdefault(k, i)
            if j != i:
                rows.append(np.array([i]))
                cols.append(np.array([j]))
    r = np.concatenate(rows) if rows else np.empty(0, int)
    c = np.concatenate(cols) if cols else np.empty(0, int)
    adj = coo_matrix((np.ones(len(r), bool), (r, c)), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    first = {}
    return np.array([first.setdefault(x, len(first)) for x in lab])


def assign_folds(groups: Sequence, keys: Sequence[str], k: int = N_FOLDS) -> np.ndarray:
    """Assign whole groups to ``k`` folds, balancing molecule counts.

    Greedy: groups in decreasing size, each to the currently smallest fold (lowest index on
    ties). Equal-sized groups are ordered by the smallest ``key`` among their members, so with
    canonical SMILES or InChIKeys as ``keys`` the result does not depend on input order.
    """
    groups = np.asarray(groups)
    size = Counter(groups.tolist())
    tiebreak: dict = {}
    for g, key in zip(groups.tolist(), keys):
        tiebreak[g] = min(tiebreak.get(g, key), key)
    load = [0] * k
    fold_of = {}
    for g in sorted(size, key=lambda g: (-size[g], tiebreak[g])):
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += size[g]
    return np.array([fold_of[g] for g in groups.tolist()])


def cross_fold_nn_similarity(fps: Sequence, folds: Sequence[int]) -> np.ndarray:
    """For each molecule, its highest Tanimoto similarity to any molecule in another fold.

    This is how close the nearest training molecule is when that molecule is in the test fold.
    """
    folds = np.asarray(folds)
    out = np.empty(len(fps))
    for i, fp in enumerate(fps):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps))
        other = folds != folds[i]
        out[i] = s[other].max() if other.any() else np.nan
    return out


# Bins for reporting results by distance to the training set (rule 3).
NN_BINS = (0.0, 0.4, 0.5, 0.6, COMPONENT_THRESHOLD)


def nn_bin_labels(nn: np.ndarray, edges: Sequence[float] = NN_BINS) -> np.ndarray:
    """Label each molecule with its nearest-training-neighbour similarity bin, e.g. '0.5-0.6'."""
    names = [f"{a:.1f}-{b:.1f}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]:.1f}"]
    idx = np.clip(np.searchsorted(edges, nn, side="right") - 1, 0, len(names) - 1)
    return np.array(names, dtype=object)[idx]


def group_size_summary(groups: Sequence) -> dict:
    """Group-size distribution: counts, the largest groups, and the largest as a fraction."""
    sizes = np.array(sorted(Counter(np.asarray(groups).tolist()).values(), reverse=True))
    n = int(sizes.sum())
    return {
        "molecules": n,
        "groups": len(sizes),
        "largest": int(sizes[0]),
        "largest_frac": float(sizes[0] / n),
        "top10": sizes[:10].tolist(),
        "singletons": int((sizes == 1).sum()),
        "size_hist": {b: int(((sizes >= lo) & (sizes <= hi)).sum())
                      for b, lo, hi in [("1", 1, 1), ("2-4", 2, 4), ("5-19", 5, 19),
                                        ("20-99", 20, 99), ("100+", 100, 10**9)]},
    }
