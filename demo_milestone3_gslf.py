"""Milestone 3, Phase B: all 112 GS+LF labels, milestone 2's protocol, one run.

Usage: python demo_milestone3_gslf.py [--data-dir DIR]

Per plan-milestone-3.md. One forest per label (not multi-output), so floral reproduces
milestone 2. Out-of-fold scores are cached per label under DATA/m3_scores/ so that a crash
resumes without refitting; resuming continues the same run with the same settings.

Order: fit everything, then check 1 (shuffled-label macro AUC in 0.48-0.52) and check 2
(floral reproduces milestone 2), and only then compute and print the measurements.
Writes reports/05_m3_per_label.csv.

Exact-equivalence speed-ups, both tested: kNN-Tanimoto neighbours computed once per fold
for all labels (models.tanimoto_neighbours); pooled bootstrap AUCs computed for all labels
at once (evaluate.auc_columns).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sw_qsor.evaluate import auc_columns, fold_aucs, joint_group_bootstrap, within_fold_ranks
from sw_qsor.experiment import gslf_experiment
from sw_qsor.floors import knn_1d
from sw_qsor.models import fingerprint_matrix, random_forest, tanimoto_neighbours

REPO_ROOT = Path(__file__).resolve().parent
MODELS = ["F-MW", "RF-MW", "kNN-Tanimoto", "RF-Morgan"]
M2_FLORAL = {  # plan-milestone-2.md results, per-fold AUC to 3 dp
    "F-MW": [0.678, 0.693, 0.694, 0.667, 0.673],
    "RF-MW": [0.725, 0.739, 0.732, 0.721, 0.730],
    "kNN-Tanimoto": [0.808, 0.789, 0.823, 0.804, 0.808],
    "RF-Morgan": [0.834, 0.809, 0.834, 0.820, 0.835],
}
BANDS = [(50, 99), (100, 249), (250, 499), (500, 10**6)]


def oof_rf(X, y, folds, shuffle=False):
    rng = np.random.default_rng(0) if shuffle else None
    s = np.full(len(y), np.nan)
    for f in np.unique(folds):
        te = folds == f
        ytr = y[~te] if rng is None else rng.permutation(y[~te])
        s[te] = random_forest(X[~te], ytr, X[te])
    return s


def fit_all(ex, labels, cache):
    cache.mkdir(parents=True, exist_ok=True)
    folds = ex.folds
    X = fingerprint_matrix(ex.fps)
    Yall = np.column_stack([ex.label(lab) for lab in labels]).astype(int)

    # label-independent pieces, computed once
    f_mw = np.full(Yall.shape, np.nan)
    knn = np.full(Yall.shape, np.nan)
    for f in np.unique(folds):
        te = folds == f
        f_mw[te] = knn_1d(ex.mw[~te], Yall[~te], ex.mw[te])
        idx = tanimoto_neighbours([ex.fps[i] for i in np.nonzero(~te)[0]],
                                  [ex.fps[i] for i in np.nonzero(te)[0]])
        knn[te] = Yall[~te][idx].mean(1)

    t0 = time.time()
    for j, lab in enumerate(labels):
        path = cache / f"{lab}.npz"
        if path.exists():
            continue
        y = Yall[:, j]
        np.savez(path, rf_morgan=oof_rf(X, y, folds), rf_mw=oof_rf(ex.mw, y, folds),
                 shuffled=oof_rf(X, y, folds, shuffle=True))
        done = j + 1
        el = time.time() - t0
        print(f"  [{done:3d}/{len(labels)}] {lab:16s} {el / 60:5.1f} min", flush=True)

    S = {m: np.full(Yall.shape, np.nan) for m in MODELS + ["shuffled"]}
    S["F-MW"], S["kNN-Tanimoto"] = f_mw, knn
    for j, lab in enumerate(labels):
        z = np.load(cache / f"{lab}.npz")
        S["RF-Morgan"][:, j], S["RF-MW"][:, j], S["shuffled"][:, j] = (
            z["rf_morgan"], z["rf_mw"], z["shuffled"])
    return Yall, S


def pooled(Y, R, groups, mask, labels_ok, comparator):
    """Joint group bootstrap of the macro-mean margin RF-Morgan minus comparator over
    ``labels_ok`` on ``mask``. ``comparator`` is a list of models; the max is taken."""
    Ym, gm = Y[mask][:, labels_ok], groups[mask]
    Rm = {k: v[mask][:, labels_ok] for k, v in R.items()}

    def stat(idx):
        yy = Ym[idx]
        model = auc_columns(yy, Rm["RF-Morgan"][idx])
        comp = np.max([auc_columns(yy, Rm[c][idx]) for c in comparator], axis=0)
        return float(np.nanmean(model - comp))

    return joint_group_bootstrap(stat, gm)


def main(data_dir, reports):
    ex = gslf_experiment(data_dir / "gslf")
    floor_tab = pd.read_csv(reports / "04_floor_per_label.csv", index_col=0)
    labels = floor_tab.index[floor_tab.positives >= 50].tolist()
    labels = ["floral"] + [lab for lab in labels if lab != "floral"]  # fail fast on check 2
    print(f"Phase B: {len(labels)} labels, {len(ex.Y)} molecules, folds "
          f"{np.bincount(ex.folds).tolist()}. Fitting (3 forests x 5 folds per label) ...")
    Y, S = fit_all(ex, labels, data_dir / "m3_scores")
    folds, groups = ex.folds, ex.groups
    A = {m: np.array([fold_aucs(Y[:, j], S[m][:, j], folds) for j in range(len(labels))])
         for m in S}  # (labels, folds)

    print("\n== Checks")
    shuf = A["shuffled"].mean(1)
    ok1 = 0.48 <= shuf.mean() <= 0.52
    print(f"1. Shuffled-label RF-Morgan macro-mean AUC {shuf.mean():.4f} "
          f"(labels range {shuf.min():.3f}-{shuf.max():.3f}): {'pass' if ok1 else 'FAIL'}")
    fl = labels.index("floral")
    ok2 = all(np.allclose(np.round(A[m][fl], 3), M2_FLORAL[m], atol=1e-9) for m in MODELS)
    for m in MODELS:
        print(f"   floral {m:13s} {np.round(A[m][fl], 3).tolist()} vs M2 {M2_FLORAL[m]}")
    print(f"2. Floral reproduces milestone 2: {'pass' if ok2 else 'FAIL'}")
    if not (ok1 and ok2):
        print("\nA CHECK FAILED. Per the plan: stop and find out why; no result is reported.")
        return

    floor = np.maximum(A["F-MW"], A["RF-MW"])
    margin = A["RF-Morgan"] - floor
    mm = margin.mean(1)
    npos = (margin > 0).sum(1)
    verdict = np.where((mm >= 0.05) & (npos == 5), "pass",
                       np.where((mm < 0.02) | ((margin <= 0).sum(1) >= 2), "fail",
                                "inconclusive"))
    vs_knn = (A["RF-Morgan"] - A["kNN-Tanimoto"])
    far = ex.nn < 0.5
    far_pos = Y[far].sum(0)
    tab = pd.DataFrame({
        "positives": Y.sum(0), "far_positives": far_pos,
        **{f"{m}_auc": A[m].mean(1) for m in MODELS},
        "floor_auc": floor.mean(1), "margin": mm, "margin_sd": margin.std(1, ddof=1),
        "folds_positive": npos, "verdict": verdict,
        "margin_vs_knn": vs_knn.mean(1), "folds_better_than_knn": (vs_knn > 0).sum(1),
        "shuffled_auc": shuf,
    }, index=pd.Index(labels, name="label")).sort_values("positives", ascending=False)
    tab.round(4).to_csv(reports / "05_m3_per_label.csv")

    print("\n== Q1, per label (pass: mean margin >= 0.05 and 5/5 folds > 0; "
          "fail: < 0.02 or >= 2 folds <= 0)")
    counts = tab.verdict.value_counts()
    print(f"pass {counts.get('pass', 0)}, fail {counts.get('fail', 0)}, inconclusive "
          f"{counts.get('inconclusive', 0)} of {len(tab)}")
    for lo, hi in BANDS:
        b = tab[(tab.positives >= lo) & (tab.positives <= hi)]
        vc = b.verdict.value_counts()
        print(f"  positives {lo}-{hi if hi < 10**6 else '+'}: {len(b):3d} labels; pass "
              f"{vc.get('pass', 0):3d}, fail {vc.get('fail', 0):3d}, inconclusive "
              f"{vc.get('inconclusive', 0):3d}; median margin {b.margin.median():+.3f}, "
              f"median floor {b.floor_auc.median():.3f}, median RF-Morgan "
              f"{b['RF-Morgan_auc'].median():.3f}")
    npass = counts.get("pass", 0)
    reading = "BROAD" if npass >= 56 else "NARROW" if npass >= 10 else "ABSENT"
    print(f"Reading: {reading} ({npass} pass; broad >= 56, narrow 10-55, absent < 10; "
          f"plan.md predicted narrow)")
    print(f"Margin over floor across labels: median {tab.margin.median():+.3f}, "
          f"range {tab.margin.min():+.3f} to {tab.margin.max():+.3f}")
    print("Failed:", ", ".join(tab.index[tab.verdict == "fail"]) or "none")
    print("Inconclusive:", ", ".join(tab.index[tab.verdict == "inconclusive"]) or "none")

    R = {m: within_fold_ranks(S[m], folds) for m in MODELS}
    ok = np.nonzero(far_pos >= 10)[0]
    print(f"\n== Pooled over the {len(ok)} labels with >= 10 positives below 0.5 similarity "
          f"({far.sum()} molecules, {int(far_pos[ok].sum())} positives; 2,000 joint group "
          f"bootstraps, seed 0)")
    q1 = pooled(Y, R, groups, far, ok, ["F-MW", "RF-MW"])
    print(f"Q1 pooled: margin over floor {q1['estimate']:+.4f} [{q1['lo']:+.4f}, "
          f"{q1['hi']:+.4f}] -> "
          f"{'structure signal PERSISTS far from training' if q1['lo'] > 0 else 'NOT shown to persist'}")
    q2 = pooled(Y, R, groups, far, ok, ["kNN-Tanimoto"])
    q2r = ("LEARNING BEYOND SIMILARITY" if q2["lo"] > 0 else
           "WORSE THAN LOOKUP" if q2["hi"] < 0 else "NO EVIDENCE BEYOND SIMILARITY")
    print(f"Q2: margin over kNN-Tanimoto {q2['estimate']:+.4f} [{q2['lo']:+.4f}, "
          f"{q2['hi']:+.4f}], width {q2['hi'] - q2['lo']:.4f} -> {q2r}")

    print("\nSecondary (decides nothing): margin by bin, same labels and bootstrap")
    bins = {"0.0-0.4": ex.nn < 0.4, "0.4-0.5": (ex.nn >= 0.4) & (ex.nn < 0.5),
            "0.5-0.6": (ex.nn >= 0.5) & (ex.nn < 0.6), "0.6-0.7": ex.nn >= 0.6}
    for name, m in bins.items():
        a = pooled(Y, R, groups, m, ok, ["F-MW", "RF-MW"])
        b = pooled(Y, R, groups, m, ok, ["kNN-Tanimoto"])
        print(f"  {name}: {m.sum():5d} mols; over floor {a['estimate']:+.4f} [{a['lo']:+.4f}, "
              f"{a['hi']:+.4f}]; over kNN {b['estimate']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]")
    print(f"Labels better than kNN-Tanimoto overall (mean of folds): "
          f"{int((tab.margin_vs_knn > 0).sum())}, worse {int((tab.margin_vs_knn < 0).sum())}; "
          f"better in 5/5 folds {int((tab.folds_better_than_knn == 5).sum())}, worse in 5/5 "
          f"{int((tab.folds_better_than_knn == 0).sum())}")
    print(f"\nPer-label table: {reports / '05_m3_per_label.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = ap.parse_args()
    main(args.data_dir, REPO_ROOT / "reports")
