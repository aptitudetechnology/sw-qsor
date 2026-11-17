"""Milestone 3, Phase C: train on all of GS+LF, test on Arctander molecules independent of it.

Usage: python demo_milestone3_arctander.py [--data-dir DIR]

Per plan-milestone-3.md. Phase A decided this is reported as an EXTERNAL TEST, with the
caveat that Leffingwell (half of GS+LF) tracks Arctander more closely than GoodScents does.
Test set: Arctander's labelled molecules that are independent of GS+LF under rule 3 (no
GS+LF molecule in their similarity group); labels: the 25 fixed in A3. Models and settings
are Phase B's, fitted once on all 4,981 GS+LF molecules. Uncertainty: 2,000 bootstraps of
Arctander's rule-3 groups, seed 0; as in Phase B the floor in each resample is the higher
of F-MW and RF-MW. Writes reports/05_phase_c_per_label.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors

from sw_qsor.datasets import load_gslf, load_pyrfume_molecules
from sw_qsor.evaluate import auc_columns, joint_group_bootstrap
from sw_qsor.experiment import arctander_by_key, gslf_experiment, independence
from sw_qsor.floors import knn_1d
from sw_qsor.models import fingerprint_matrix, random_forest, tanimoto_neighbours
from sw_qsor.splits import morgan_fingerprints, nn_bin_labels

REPO_ROOT = Path(__file__).resolve().parent
PYRFUME = [("Arctander", "arctander_1960"), ("Mayhew", "mayhew_2022"),
           ("Keller", "keller_2016"), ("Dravnieks", "dravnieks_1985"),
           ("Abraham", "abraham_2012")]


def main(data_dir, reports):
    ex = gslf_experiment(data_dir / "gslf")
    _, gslf_labels = load_gslf(data_dir / "gslf")
    c_tab = pd.read_csv(reports / "05_phase_c_labels.csv", index_col=0)
    labels = c_tab.index[c_tab.phase_c].tolist()
    targets = c_tab.index.tolist()

    sm = {"GS+LF": ex.smiles.tolist()}
    for name, d in PYRFUME:
        sm[name] = load_pyrfume_molecules(data_dir / d).smiles.tolist()
    ind = independence(sm, "GS+LF")
    A = arctander_by_key(data_dir / "arctander_1960", gslf_labels, targets)
    test_keys = sorted(ind["classes"]["Arctander"]["independent"] & set(A.index))
    assert len(test_keys) == 805, len(test_keys)  # Phase A, A3
    test_smiles = [ind["smiles_of"][k] for k in test_keys]
    groups = np.array([ind["group_of"][k] for k in test_keys])
    Yt = A.loc[test_keys, labels].to_numpy().astype(int)

    print(f"Phase C: train on GS+LF ({len(ex.Y)} molecules), test on {len(test_keys)} "
          f"labelled Arctander molecules independent of GS+LF ({len(set(groups))} groups), "
          f"{len(labels)} labels.")
    print("Framing (Phase A): EXTERNAL TEST; caveat: Leffingwell tracks Arctander closest.\n")

    fps_t = morgan_fingerprints(test_smiles)
    mw_t = np.array([Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in test_smiles])
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fp, ex.fps)) for fp in fps_t])
    bins = pd.Series(nn_bin_labels(nn)).value_counts().sort_index()
    print("Test molecules' similarity to their nearest GS+LF molecule: median "
          f"{np.median(nn):.2f}; by bin " + ", ".join(f"{k}: {v}" for k, v in bins.items()))
    print("(GS+LF's own test folds in Phase B: 455 / 635 / 1536 / 2355 in the same bins)\n")

    Ytr = np.column_stack([ex.label(lab) for lab in labels]).astype(int)
    S = {"F-MW": knn_1d(ex.mw, Ytr, mw_t)}
    S["kNN-Tanimoto"] = Ytr[tanimoto_neighbours(ex.fps, fps_t)].mean(1)
    X_tr, X_te = fingerprint_matrix(ex.fps), fingerprint_matrix(fps_t)
    S["RF-MW"] = np.column_stack([random_forest(ex.mw, Ytr[:, j], mw_t)
                                  for j in range(len(labels))])
    S["RF-Morgan"] = np.column_stack([random_forest(X_tr, Ytr[:, j], X_te)
                                      for j in range(len(labels))])

    auc = {m: auc_columns(Yt, s) for m, s in S.items()}
    floor = np.maximum(auc["F-MW"], auc["RF-MW"])
    tab = pd.DataFrame({"arctander_positives": Yt.sum(0), "gslf_positives": Ytr.sum(0),
                        **{f"{m}_auc": a for m, a in auc.items()}, "floor_auc": floor,
                        "margin": auc["RF-Morgan"] - floor,
                        "margin_vs_knn": auc["RF-Morgan"] - auc["kNN-Tanimoto"]},
                       index=pd.Index(labels, name="label"))
    p_b = pd.read_csv(reports / "05_m3_per_label.csv", index_col=0)
    tab["phase_b_RF-Morgan_auc"] = p_b.loc[labels, "RF-Morgan_auc"]
    tab["phase_b_verdict"] = p_b.loc[labels, "verdict"]
    tab.round(4).to_csv(reports / "05_phase_c_per_label.csv")
    show = tab[["arctander_positives", "F-MW_auc", "RF-MW_auc", "kNN-Tanimoto_auc",
                "RF-Morgan_auc", "margin", "margin_vs_knn", "phase_b_RF-Morgan_auc",
                "phase_b_verdict"]]
    print(show.sort_values("arctander_positives", ascending=False).round(3).to_string())
    print("\nMacro-mean AUC: " + ", ".join(f"{m} {a.mean():.3f}" for m, a in auc.items()))
    print(f"Labels with RF-Morgan above floor: {int((tab.margin > 0).sum())}/{len(tab)}; "
          f"above kNN-Tanimoto: {int((tab.margin_vs_knn > 0).sum())}/{len(tab)}\n")

    def margin_stat(comparators):
        def stat(idx):
            yy = Yt[idx]
            m = auc_columns(yy, S["RF-Morgan"][idx])
            c = np.max([auc_columns(yy, S[k][idx]) for k in comparators], axis=0)
            return float(np.nanmean(m - c))
        return stat

    over_floor = joint_group_bootstrap(margin_stat(["F-MW", "RF-MW"]), groups)
    over_knn = joint_group_bootstrap(margin_stat(["kNN-Tanimoto"]), groups)
    print("Pooled (macro-mean over the 25 labels; 2,000 group bootstraps, seed 0):")
    print(f"  over floor:        {over_floor['estimate']:+.4f} [{over_floor['lo']:+.4f}, "
          f"{over_floor['hi']:+.4f}] -> "
          f"{'TRANSFERS' if over_floor['lo'] > 0 else 'transfer NOT shown'}")
    print(f"  over kNN-Tanimoto: {over_knn['estimate']:+.4f} [{over_knn['lo']:+.4f}, "
          f"{over_knn['hi']:+.4f}] -> "
          f"{'BEYOND SIMILARITY on a new source' if over_knn['lo'] > 0 else 'beyond similarity NOT shown'}")

    print("\nSecondary (decides nothing): pooled margins by nearest-GS+LF-neighbour bin")
    for b in sorted(set(nn_bin_labels(nn))):
        m = nn_bin_labels(nn) == b
        if m.sum() < 30:
            print(f"  {b}: {m.sum()} molecules, too few")
            continue
        yy = Yt[m]
        ok = (yy.sum(0) >= 5) & (yy.sum(0) < m.sum())
        mm = auc_columns(yy[:, ok], S["RF-Morgan"][m][:, ok])
        fl = np.maximum(auc_columns(yy[:, ok], S["F-MW"][m][:, ok]),
                        auc_columns(yy[:, ok], S["RF-MW"][m][:, ok]))
        kn = auc_columns(yy[:, ok], S["kNN-Tanimoto"][m][:, ok])
        print(f"  {b}: {m.sum():4d} molecules, {ok.sum():2d} labels with >= 5 positives; "
              f"RF-Morgan {np.mean(mm):.3f}, over floor {np.mean(mm - fl):+.3f}, over kNN "
              f"{np.mean(mm - kn):+.3f}")
    print(f"\nPer-label table: {reports / '05_phase_c_per_label.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = ap.parse_args()
    main(args.data_dir, REPO_ROOT / "reports")
