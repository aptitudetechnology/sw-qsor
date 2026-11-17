"""Milestone 1 survey: what is in the data, and how independent the datasets are.

Usage: python demo_dataset_survey.py [--data-dir DIR]

Step 3 of next-session.md: overlap between datasets by InChIKey connectivity block
(sw_qsor.identity), and how much of each dataset could serve as a test set independent of
GS+LF under readme rule 3. Structure only; no labels are read.

Step 4: the chemistry-free floor on GS+LF's 138 descriptors (rule 2), under rule-3 folds,
reported per label (mean and SD across folds) and by nearest-training-neighbour similarity.
Floors are base rate and molecular-weight k-NN (sw_qsor.floors). The per-molecule label count
is not knowable without the labels themselves, so it is not a floor and is not computed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sw_qsor.datasets import load_gslf, load_pyrfume_molecules
from sw_qsor.evaluate import (
    by_bin,
    cross_validate,
    per_label_per_fold,
    pooled_micro_auc,
    summarise_labels,
)
from sw_qsor.experiment import gslf_experiment
from sw_qsor.floors import KNN_K, base_rate, knn_1d
from sw_qsor.identity import key_table, overlap_matrix
from sw_qsor.splits import (
    COMPONENT_THRESHOLD,
    N_FOLDS,
    morgan_fingerprints,
    similarity_components,
)

REPO_ROOT = Path(__file__).resolve().parent
PYRFUME = ["arctander_1960", "mayhew_2022", "keller_2016", "dravnieks_1985", "abraham_2012"]
SHORT = {"gslf": "GS+LF", "arctander_1960": "Arctander", "mayhew_2022": "Mayhew",
         "keller_2016": "Keller", "dravnieks_1985": "Dravnieks", "abraham_2012": "Abraham"}


def load_smiles(data_dir):
    out = {"GS+LF": load_gslf(data_dir / "gslf")[0].smiles.tolist()}
    for a in PYRFUME:
        out[SHORT[a]] = load_pyrfume_molecules(data_dir / a).smiles.tolist()
    return out


def identity_section(smiles):
    print("== Molecules per dataset (rule 1: distinct molecules, not rows)\n")
    tables, rows = {}, []
    for name, smi in smiles.items():
        t = key_table(smi)
        tables[name] = t
        ok = t.dropna(subset=["key"])
        rows.append({
            "dataset": name, "rows": len(t), "no structure": int(t.key.isna().sum()),
            "distinct full InChIKey": ok.inchikey.nunique(),
            "distinct molecules (connectivity)": ok.key.nunique(),
            "merged by dropping stereo/protonation": ok.inchikey.nunique() - ok.key.nunique(),
        })
    print(pd.DataFrame(rows).set_index("dataset").to_string(), "\n")
    return tables


def overlap_section(tables, smiles):
    sets = {n: set(t.key.dropna()) for n, t in tables.items()}
    m = overlap_matrix(sets)
    print("== Shared molecules (connectivity key, structures as given)\n")
    print(m.to_string(), "\n")
    print("Row dataset's share contained in column dataset (%):\n")
    pct = m.div(np.diag(m), axis=0) * 100
    print(pct.round(0).astype(int).to_string(), "\n")

    parent = {n: set(key_table(s, parent=True).key.dropna()) for n, s in smiles.items()}
    pm = overlap_matrix(parent)
    print("Sensitivity: matches with GS+LF after stripping salts/neutralising (parent), vs as given:")
    for n in sets:
        if n != "GS+LF":
            print(f"  {n:10s} {m.loc[n, 'GS+LF']:5d} -> {pm.loc[n, 'GS+LF']:5d}")
    print()
    return sets


def independence_section(tables, sets, anchor="GS+LF"):
    """Union all datasets, dedupe by key, group by rule 3, and classify each molecule of each
    dataset relative to the anchor."""
    reps = {}
    for name, t in tables.items():  # anchor first, so its SMILES represent shared keys
        for s, k in zip(t.smiles, t.key):
            if pd.notna(k):
                reps.setdefault(k, s)
    keys = list(reps)
    fps = morgan_fingerprints([reps[k] for k in keys])
    groups = similarity_components(fps)
    group_of = dict(zip(keys, groups))
    anchor_groups = {group_of[k] for k in sets[anchor]}

    print(f"== Independence from {anchor} under rule 3 (union of all datasets, {len(keys)} "
          f"distinct molecules, components at Tanimoto >= {COMPONENT_THRESHOLD})\n")
    rows = []
    for name, ks in sets.items():
        if name == anchor:
            continue
        inside = ks & sets[anchor]
        near = {k for k in ks - inside if group_of[k] in anchor_groups}
        indep = ks - inside - near
        rows.append({"dataset": name, "molecules": len(ks), f"in {anchor}": len(inside),
                     f"not in {anchor}, same group": len(near),
                     "independent": len(indep),
                     "independent %": round(100 * len(indep) / len(ks)),
                     "independent groups": len({group_of[k] for k in indep})})
    print(pd.DataFrame(rows).set_index("dataset").to_string(), "\n")

    others = set().union(*(ks for n, ks in sets.items() if n != anchor))
    union_indep = {k for k in others if group_of[k] not in anchor_groups}
    print(f"All non-{anchor} datasets together: {len(union_indep)} distinct molecules "
          f"independent of {anchor}.\n")


def _q(x):
    q = np.nanpercentile(x, [10, 25, 50, 75, 90])
    return " / ".join(f"{v:.3f}" for v in q)


def floors_section(data_dir):
    ex = gslf_experiment(data_dir / "gslf")
    labels, Y, mw, folds, nn_bins = ex.labels, ex.Y, ex.mw, ex.folds, ex.nn_bins
    n = len(Y)

    print(f"== Chemistry-free floor: GS+LF, {n} molecules (duplicate records merged), "
          f"{len(labels)} labels, {N_FOLDS} rule-3 folds {np.bincount(folds).tolist()}\n")
    per_mol = Y.sum(1)
    print(f"Labels per molecule: median {np.median(per_mol):.0f}, mean {per_mol.mean():.2f}, "
          f"max {per_mol.max()}, molecules with none: {(per_mol == 0).sum()}")
    pos = Y.sum(0)
    print(f"Positives per label: median {np.median(pos):.0f}; labels with >=50: "
          f"{(pos >= 50).sum()}, >=100: {(pos >= 100).sum()}, >=500: {(pos >= 500).sum()}; "
          f"min {pos.min()}, max {pos.max()} ({labels[int(pos.argmax())]})\n")

    results = {}
    for name, fn in [("base rate", base_rate), (f"MW k-NN (k={KNN_K})", knn_1d)]:
        S = cross_validate(fn, mw, Y, folds)
        long = per_label_per_fold(Y, S, folds, labels)
        summ = summarise_labels(long)
        results[name] = (S, long, summ)
        print(f"-- {name}")
        print(f"   per-label AUC, mean across folds; 10/25/50/75/90th pct over labels: "
              f"{_q(summ.auc_mean)}")
        print(f"   macro-mean AUC {summ.auc_mean.mean():.3f}; per-label fold SD, median "
              f"{summ.auc_sd.median():.3f}")
        print(f"   per-label AP lift over prevalence, pct as above: {_q(summ.ap_lift)}")
        print(f"   pooled micro AUC over all (molecule, label) pairs: "
              f"{pooled_micro_auc(Y, S):.3f}   <- not a structure measure")
        print()

    _, _, knn = results[f"MW k-NN (k={KNN_K})"]
    knn = knn.assign(z=(knn.auc_mean - 0.5) / (knn.auc_sd / np.sqrt(knn.folds_scored)))
    print("MW k-NN by label frequency (labels with >=50 positives are the ones the plan's "
          "decision rule counts):")
    bands = pd.cut(knn.positives, [0, 49, 99, 249, 499, 10**6],
                   labels=["<50", "50-99", "100-249", "250-499", "500+"])
    t = knn.groupby(bands, observed=True).agg(
        labels=("auc_mean", "size"), auc_median=("auc_mean", "median"),
        auc_max=("auc_mean", "max"), fold_sd_median=("auc_sd", "median"),
        ap_lift_median=("ap_lift", "median"))
    print(t.round(3).to_string(), "\n")

    print("Highest MW k-NN floors (mean AUC across folds, fold SD):")
    top = knn.sort_values("auc_mean", ascending=False).head(12)
    print(top[["positives", "auc_mean", "auc_sd", "ap_mean", "prevalence", "ap_lift"]]
          .round(3).to_string(), "\n")
    print("Lowest:")
    print(knn.sort_values("auc_mean").head(6)[["positives", "auc_mean", "auc_sd"]]
          .round(3).to_string(), "\n")
    print(f"Labels whose MW floor is clearly above 0.5 (mean - 2 SE > 0.5): "
          f"{int((knn.z > 2).sum())} of {len(knn)}; clearly below: {int((knn.z < -2).sum())}\n")

    print("By nearest-training-neighbour similarity (rule 3), macro AUC over labels with "
          ">=10 positives in the bin:")
    for name, (S, _, _) in results.items():
        print(f"-- {name}")
        print(by_bin(Y, S, nn_bins, labels, folds).round(3).to_string())
    print()
    return results, Y, labels, folds


def main(data_dir, floor_csv=None):
    smiles = load_smiles(data_dir)
    tables = identity_section(smiles)
    sets = overlap_section(tables, smiles)
    independence_section(tables, sets)
    results, *_ = floors_section(data_dir)
    if floor_csv:
        _, _, br = results["base rate"]
        _, _, knn = results[f"MW k-NN (k={KNN_K})"]
        t = knn[["positives", "prevalence", "folds_scored", "auc_mean", "auc_sd", "ap_mean",
                 "ap_sd", "ap_lift"]].add_prefix("mw_knn_").rename(columns={
                     "mw_knn_positives": "positives", "mw_knn_prevalence": "prevalence",
                     "mw_knn_folds_scored": "folds_scored"})
        t.insert(3, "base_rate_ap", br.ap_mean)
        t.sort_values("positives", ascending=False).round(4).to_csv(floor_csv)
        print(f"Per-label floor table written to {floor_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--floor-csv", type=Path, help="write the per-label floor table here")
    args = ap.parse_args()
    main(args.data_dir, args.floor_csv)
