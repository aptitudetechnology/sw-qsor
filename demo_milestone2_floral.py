"""Milestone 2: predict GS+LF `floral` from structure, exactly as plan-milestone-2.md fixes it.

Usage: python demo_milestone2_floral.py [--data-dir DIR]

Prints every number under the plan's "Measurements" and the verdict against its bar. Three
points the plan leaves open are resolved here and printed with the results:
  1. In the distance bins, the floor is the higher of F-MW and RF-MW *within each bootstrap
     resample*, the conservative reading of "the higher of F-MW and RF-MW".
  2. The shuffled-label check uses one generator, seed 0, permuting each fold's training
     labels in fold order.
  3. F-MW is recomputed and checked against the recorded floor (0.681) before anything else.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from sw_qsor.evaluate import fold_aucs, group_bootstrap_margin, within_fold_ranks
from sw_qsor.experiment import gslf_experiment
from sw_qsor.floors import knn_1d
from sw_qsor.models import fingerprint_matrix, random_forest, tanimoto_knn

REPO_ROOT = Path(__file__).resolve().parent
LABEL = "floral"
RECORDED_FLOOR = 0.6810  # reports/04_floor_per_label.csv, mw_knn_auc_mean for floral


def out_of_fold(fit_predict, X, y, folds, shuffle_rng=None):
    s = np.full(len(y), np.nan)
    for f in np.unique(folds):
        te = folds == f
        ytr = y[~te] if shuffle_rng is None else shuffle_rng.permutation(y[~te])
        s[te] = fit_predict(X[~te], ytr, X[te])
    return s


def main(data_dir):
    ex = gslf_experiment(data_dir / "gslf")
    y = ex.label(LABEL).astype(int)
    folds, groups = ex.folds, ex.groups
    print(f"Milestone 2: '{LABEL}' on GS+LF, {len(y)} molecules, {y.sum()} positives, "
          f"folds {np.bincount(folds).tolist()}, positives per fold "
          f"{[int(y[folds == f].sum()) for f in range(5)]}\n")

    X = fingerprint_matrix(ex.fps)
    fps = np.empty(len(ex.fps), dtype=object)
    fps[:] = ex.fps

    scores = {}
    scores["F-MW"] = out_of_fold(lambda a, b, c: knn_1d(a, b[:, None], c)[:, 0], ex.mw, y, folds)
    f_mw = fold_aucs(y, scores["F-MW"], folds)
    print(f"Check 3: F-MW recomputed mean AUC {f_mw.mean():.4f}, recorded {RECORDED_FLOOR}")
    assert abs(f_mw.mean() - RECORDED_FLOOR) < 5e-4, "floor does not reproduce; stop"

    print("Pipeline check: RF-Morgan on labels shuffled within training folds ...")
    shuf = out_of_fold(random_forest, X, y, folds, shuffle_rng=np.random.default_rng(0))
    shuf_auc = fold_aucs(y, shuf, folds)
    print(f"  per-fold AUC {np.round(shuf_auc, 3).tolist()}, mean {shuf_auc.mean():.3f}")
    if not 0.45 <= shuf_auc.mean() <= 0.55:
        print("\nPIPELINE CHECK FAILED: shuffled-label AUC outside 0.45-0.55. "
              "Per the plan, stop and find the leak; no result is reported.")
        return
    print("  passed (0.45-0.55)\n")

    scores["RF-MW"] = out_of_fold(random_forest, ex.mw, y, folds)
    scores["kNN-Tanimoto"] = out_of_fold(tanimoto_knn, fps, y, folds)
    scores["RF-Morgan"] = out_of_fold(random_forest, X, y, folds)

    print(f"{'model':14s} {'per-fold AUC':38s} {'mean':>6s} {'SD':>6s} {'mean AP':>8s}")
    aucs = {}
    for name, s in scores.items():
        a = fold_aucs(y, s, folds)
        aucs[name] = a
        ap = np.mean([average_precision_score(y[folds == f], s[folds == f]) for f in range(5)])
        print(f"{name:14s} {str(np.round(a, 3).tolist()):38s} {a.mean():6.3f} "
              f"{a.std(ddof=1):6.3f} {ap:8.3f}")
    print(f"(prevalence {y.mean():.3f}; shuffled RF-Morgan mean AUC {shuf_auc.mean():.3f})\n")

    floor = np.maximum(aucs["F-MW"], aucs["RF-MW"])
    margin = aucs["RF-Morgan"] - floor
    print(f"Floor per fold (max of F-MW, RF-MW): {np.round(floor, 3).tolist()}")
    print(f"Paired margin per fold:              {np.round(margin, 3).tolist()}")
    print(f"Mean paired margin {margin.mean():.3f}; folds with margin > 0: "
          f"{int((margin > 0).sum())}/5\n")
    vs_knn = aucs["RF-Morgan"] - aucs["kNN-Tanimoto"]
    print(f"RF-Morgan minus kNN-Tanimoto per fold: {np.round(vs_knn, 3).tolist()}, "
          f"mean {vs_knn.mean():.3f}\n")

    R = {k: within_fold_ranks(v[:, None], folds)[:, 0] for k, v in scores.items()}
    floors = [R["F-MW"], R["RF-MW"]]
    print("By nearest-training-neighbour similarity (within-fold ranks; 2,000 group "
          "bootstraps, seed 0; floor = max of F-MW, RF-MW per resample):")
    print(f"{'bin':10s} {'mols':>5s} {'pos':>4s} {'groups':>6s}  {'margin over floor':28s}  "
          f"{'margin over kNN-Tanimoto':28s}")
    bins = {"< 0.5": ex.nn < 0.5, "0.0-0.4": ex.nn < 0.4, "0.4-0.5": (ex.nn >= 0.4) & (ex.nn < 0.5),
            "0.5-0.6": (ex.nn >= 0.5) & (ex.nn < 0.6), "0.6-0.7": ex.nn >= 0.6}
    primary = None
    for name, m in bins.items():
        r = group_bootstrap_margin(y, R["RF-Morgan"], floors, groups, m)
        k = group_bootstrap_margin(y, R["RF-Morgan"], [R["kNN-Tanimoto"]], groups, m)
        if name == "< 0.5":
            primary, primary_knn = r, k
        print(f"{name:10s} {r['molecules']:5d} {r['positives']:4d} {r['groups']:6d}  "
              f"{r['margin']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{'':9s}  "
              f"{k['margin']:+.3f} [{k['lo']:+.3f}, {k['hi']:+.3f}]")
    print(f"(primary bin '< 0.5': {primary['n_boot']} resamples used, "
          f"{primary['skipped']} skipped)\n")

    success = margin.mean() >= 0.05 and (margin > 0).all() and primary["lo"] > 0
    failure = margin.mean() < 0.02 or (margin <= 0).sum() >= 2
    verdict = "SUCCESS" if success else "FAILURE" if failure else "INCONCLUSIVE"
    print("Against the bar (plan-milestone-2.md):")
    print(f"  mean paired margin >= 0.05:            {margin.mean():.3f}  "
          f"{'yes' if margin.mean() >= 0.05 else 'no'}")
    print(f"  margin > 0 in all 5 folds:             {int((margin > 0).sum())}/5  "
          f"{'yes' if (margin > 0).all() else 'no'}")
    print(f"  < 0.5 bin interval entirely above 0:   [{primary['lo']:+.3f}, "
          f"{primary['hi']:+.3f}]  {'yes' if primary['lo'] > 0 else 'no'}")
    print(f"VERDICT: {verdict}")
    beats_knn = vs_knn.mean() > 0 and primary_knn["lo"] > 0
    print(f"Neighbour-lookup reading: RF-Morgan {'beats' if beats_knn else 'does not clearly beat'}"
          f" kNN-Tanimoto (overall mean {vs_knn.mean():+.3f}; < 0.5 bin "
          f"[{primary_knn['lo']:+.3f}, {primary_knn['hi']:+.3f}])")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    main(ap.parse_args().data_dir)
