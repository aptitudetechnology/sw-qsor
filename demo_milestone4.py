"""Milestone 4: does the published odour model (OpenPOM) beat neighbour lookup far out?

Usage: python demo_milestone4.py [--data-dir DIR] [--sw-enose-dir DIR]

Runs plan-milestone-4.md (committed 70874b4) exactly, in its order:
  checks C1, C2, C3 -> stop if any fails;
  D1, D1b (structure only; D1 needs --sw-enose-dir);
  the POM ensembles on rule-3 folds -> Q1, Q2, S1, S2.
Each trained model's test-fold probabilities are cached under DATA/m4_scores/, so a crash
resumes without retraining; resuming continues the same run.

One point the plan leaves open, resolved here and printed: C1's iterative stratified split is
seeded (numpy's global generator, seed 0, which scikit-multilearn uses for tie-breaking; its
random_state argument is rejected by current scikit-learn). OpenPOM's notebook did not seed
its split.

--smoke exercises the whole pipeline with every label matrix row-permuted, 1 epoch, 1 model
per fold and 50 bootstraps, into a separate cache. Its numbers are meaningless by
construction; it exists to find bugs before the real run, without seeing any result.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from scipy.stats import spearmanr

from sw_qsor.datasets import load_gslf
from sw_qsor.evaluate import (
    auc_columns,
    fold_aucs,
    joint_group_bootstrap,
    pooled_margin,
    within_fold_ranks,
)
from sw_qsor.experiment import gslf_experiment
from sw_qsor.floors import knn_1d
from sw_qsor.identity import connectivity_key
from sw_qsor.models import (
    fingerprint_matrix,
    random_forest,
    tanimoto_neighbours,
    tanimoto_weighted_lookup,
)
from sw_qsor.pom import TrainConfig, featurize, predict, train
from sw_qsor.splits import morgan_fingerprints, nn_bin_labels

REPO_ROOT = Path(__file__).resolve().parent
PUBLISHED = 0.8872
N_MODELS = 10
CFG = TrainConfig()  # 62 epochs, batch 128, lr 1e-3 halving every 640 steps, wd 1e-5, 12 threads
N_BOOT = 2000
SMOKE = False
CHEMICAL = ("alcoholic aldehydic camphoreous coumarinic ethereal ketonic lactonic phenolic "
            "sulfurous terpenic anisic vanilla cinnamon clove").split()
QUALITY = ("aromatic bitter clean cooling dry fresh juicy natural powdery pungent ripe sharp "
           "sour sweet warm").split()
M2_FLORAL = {"kNN-Tanimoto": [0.808, 0.789, 0.823, 0.804, 0.808],
             "RF-Morgan": [0.834, 0.809, 0.834, 0.820, 0.835]}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cached(path: Path, fn):
    if path.exists():
        return np.load(path)
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def macro_fold_auc(Y, P, folds, cols=None):
    """Mean over folds of the macro-mean (over scoreable labels) per-label AUC."""
    cols = np.arange(Y.shape[1]) if cols is None else cols
    per = [np.nanmean(auc_columns(Y[folds == f][:, cols], P[folds == f][:, cols]))
           for f in np.unique(folds)]
    return float(np.mean(per)), per


# ------------------------------------------------------------------- checks

def check_c1(data_dir, cache):
    from skmultilearn.model_selection import IterativeStratification

    df, labels = load_gslf(data_dir / "gslf")  # the raw 4,983-row file, as OpenPOM used it
    Y = df[labels].to_numpy().astype(int)
    graphs = [featurize(s) for s in df.smiles]
    if SMOKE:
        Y = Y[np.random.default_rng(1).permutation(len(Y))]
    np.random.seed(0)  # scikit-multilearn breaks ties with numpy's global generator
    folds = list(IterativeStratification(n_splits=5, order=2).split(np.zeros((len(Y), 1)), Y))
    scores = []
    for f, (tr, te) in enumerate(folds):
        def fit():
            m = train([graphs[i] for i in tr], Y[tr], seed=0, cfg=CFG)
            return predict(m, [graphs[i] for i in te])[0]
        P = cached(cache / f"c1_fold{f}.npy", fit)
        scores.append(float(np.nanmean(auc_columns(Y[te], P))))
        log(f"C1 fold {f}: {len(tr)} train / {len(te)} test, macro AUC {scores[-1]:.4f}")
    mean = float(np.mean(scores))
    ok = mean >= 0.855
    print(f"C1. Reimplementation under OpenPOM's own protocol (iterative stratified 5-fold, "
          f"numpy seed 0, one model per fold): mean {mean:.4f} (notebook single models "
          f"0.862-0.880, mean about 0.870; pass >= 0.855): {'pass' if ok else 'FAIL'}")
    return ok, mean, scores


def check_c2(ex, graphs, cache):
    rng = np.random.default_rng(0)
    P = np.full(ex.Y.shape, np.nan)
    for f in range(5):
        te, tr = ex.folds == f, ex.folds != f
        Yp = ex.Y[tr][rng.permutation(tr.sum())]

        def fit():
            m = train([graphs[i] for i in np.nonzero(tr)[0]], Yp, seed=0, cfg=CFG)
            return predict(m, [graphs[i] for i in np.nonzero(te)[0]])[0]
        P[te] = cached(cache / f"c2_fold{f}.npy", fit)
        log(f"C2 fold {f} done")
    mean, per = macro_fold_auc(ex.Y, P, ex.folds)
    ok = 0.48 <= mean <= 0.52
    print(f"C2. Permuted-label POM on rule-3 folds: macro AUC {mean:.4f} (folds "
          f"{np.round(per, 3).tolist()}; pass 0.48-0.52): {'pass' if ok else 'FAIL'}")
    return ok, mean


def check_c3(ex, data_dir):
    fl = ex.labels.index("floral")
    y = ex.Y[:, fl].astype(int)
    knn = np.full(len(y), np.nan)
    for f in range(5):
        te = ex.folds == f
        idx = tanimoto_neighbours([ex.fps[i] for i in np.nonzero(~te)[0]],
                                  [ex.fps[i] for i in np.nonzero(te)[0]])
        knn[te] = y[~te][idx].mean(1)
    m3 = data_dir / "m3_scores" / "floral.npz"
    if m3.exists():
        rf = np.load(m3)["rf_morgan"]
    else:
        X = fingerprint_matrix(ex.fps)
        rf = np.full(len(y), np.nan)
        for f in range(5):
            te = ex.folds == f
            rf[te] = random_forest(X[~te], y[~te], X[te])
    got = {"kNN-Tanimoto": np.round(fold_aucs(y, knn, ex.folds), 3).tolist(),
           "RF-Morgan": np.round(fold_aucs(y, rf, ex.folds), 3).tolist()}
    ok = all(np.allclose(got[k], M2_FLORAL[k]) for k in M2_FLORAL)
    for k in M2_FLORAL:
        print(f"   floral {k:13s} {got[k]} vs M2 {M2_FLORAL[k]}")
    print(f"C3. Floral continuity with milestone 2: {'pass' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- D1, D1b

def d1(ex, sw_enose_dir, reports):
    if sw_enose_dir is None:
        print("D1. Skipped: no --sw-enose-dir given.")
        return
    d = Path(sw_enose_dir)
    # skip only whole-line comments: '#' is also the SMILES triple bond
    import io
    text = (d / "data/m2or/odorant_structures.csv").read_text().splitlines()
    st = pd.read_csv(io.StringIO("\n".join(ln for ln in text if not ln.startswith("#"))))
    mols = [("M2OR", r.inchikey, r.smiles) for r in st.itertuples() if isinstance(r.smiles, str)]
    mols += [("breath", "hydrogen sulfide", "S"), ("breath", "methyl mercaptan", "CS"),
             ("breath", "dimethyl sulfide", "CSC")]
    gk = {k: i for i, k in enumerate(ex.keys)}
    rows = []
    for src, name, smi in mols:
        fp = morgan_fingerprints([smi])[0]
        k = connectivity_key(smi)
        heavy = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
        inside = k in gk
        nn = max(DataStructs.BulkTanimotoSimilarity(fp, ex.fps))
        tiny = fp.GetNumOnBits() <= 5
        rows.append({"source": src, "id": name, "smiles": smi, "in_gslf": inside,
                     "gslf_labels": ";".join(np.array(ex.labels)[ex.Y[gk[k]] == 1]) if inside else "",
                     "bits": fp.GetNumOnBits(), "heavy_atoms": heavy, "nn_gslf": round(nn, 3),
                     "bin": ("in GS+LF" if inside else "too small for fingerprints" if tiny
                             else nn_bin_labels(np.array([nn]))[0])})
    t = pd.DataFrame(rows)
    t.to_csv(reports / "06_sw_enose_coverage.csv", index=False)
    m = t[t.source == "M2OR"]
    print(f"D1. sw-enose's {len(m)} M2OR molecules: {int(m.in_gslf.sum())} inside GS+LF; "
          f"the rest by bin: {m[~m.in_gslf].bin.value_counts().to_dict()}")
    print(t[t.source == "breath"][["id", "in_gslf", "bits", "heavy_atoms", "bin",
                                   "gslf_labels"]].to_string(index=False))


def d1b(ex):
    bits = np.array([fp.GetNumOnBits() for fp in ex.fps])
    heavy = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in ex.smiles])
    tiny = bits <= 5
    print(f"D1b. GS+LF molecules with <= 5 fingerprint bits: {int(tiny.sum())}; of these, "
          f"{len(set(ex.groups[tiny]))} distinct rule-3 groups, and {int((tiny & (ex.nn >= 0.5)).sum())} "
          f"have a cross-fold neighbour at similarity >= 0.5.")
    print(f"     Single heavy atom (identical POM readout): {int((heavy == 1).sum())}: "
          f"{', '.join(ex.smiles[heavy == 1])}")
    print(f"     Tiny molecules: {', '.join(ex.smiles[tiny][:40])}"
          f"{' ...' if tiny.sum() > 40 else ''}")


# ------------------------------------------------------------------- main run

def pom_ensemble(ex, graphs, cache):
    P = np.zeros(ex.Y.shape)
    t0 = time.time()
    done = 0
    for f in range(5):
        te, tr = ex.folds == f, ex.folds != f
        acc = np.zeros((te.sum(), ex.Y.shape[1]))
        for s in range(N_MODELS):
            def fit():
                m = train([graphs[i] for i in np.nonzero(tr)[0]], ex.Y[tr], seed=s, cfg=CFG)
                return predict(m, [graphs[i] for i in np.nonzero(te)[0]])[0]
            acc += cached(cache / f"pom_fold{f}_seed{s}.npy", fit)
            done += 1
            log(f"POM fold {f} seed {s} ({done}/50, {(time.time() - t0) / 60:.0f} min)")
        P[te] = acc / N_MODELS
    return P


def references(ex, data_dir, labels112):
    """F-MW, RF-MW (cached from milestone 3, else refitted), Lookup-10, Lookup-W."""
    Y = ex.Y
    L10 = np.full(Y.shape, np.nan)
    LW = np.full(Y.shape, np.nan)
    FMW = np.full(Y.shape, np.nan)
    for f in range(5):
        te = ex.folds == f
        tr_fps = [ex.fps[i] for i in np.nonzero(~te)[0]]
        te_fps = [ex.fps[i] for i in np.nonzero(te)[0]]
        L10[te] = Y[~te][tanimoto_neighbours(tr_fps, te_fps)].mean(1)
        LW[te] = tanimoto_weighted_lookup(tr_fps, Y[~te], te_fps)
        FMW[te] = knn_1d(ex.mw[~te], Y[~te], ex.mw[te])
    RFMW = np.full(Y.shape, np.nan)
    RFM = np.full(Y.shape, np.nan)
    for lab in labels112:
        j = ex.labels.index(lab)
        z = data_dir / "m3_scores" / f"{lab}.npz"
        if z.exists():
            d = np.load(z)
            RFMW[:, j], RFM[:, j] = d["rf_mw"], d["rf_morgan"]
        else:
            for f in range(5):
                te = ex.folds == f
                RFMW[te, j] = random_forest(ex.mw[~te], Y[~te, j], ex.mw[te])
    return {"Lookup-10": L10, "Lookup-W": LW, "F-MW": FMW, "RF-MW": RFMW, "RF-Morgan": RFM}


def main(data_dir, sw_enose_dir, reports):
    cache = data_dir / ("m4_smoke" if SMOKE else "m4_scores")
    ex = gslf_experiment(data_dir / "gslf")
    if SMOKE:
        import dataclasses
        ex = dataclasses.replace(ex, Y=ex.Y[np.random.default_rng(1).permutation(len(ex.Y))])
        print("SMOKE RUN: all labels row-permuted; every number below is meaningless.")
    graphs = [featurize(s) for s in ex.smiles]
    floor_tab = pd.read_csv(reports / "04_floor_per_label.csv", index_col=0)
    labels112 = floor_tab.index[floor_tab.positives >= 50].tolist()
    i112 = np.array([ex.labels.index(lab) for lab in labels112])
    log(f"Milestone 4: {len(ex.Y)} molecules, {len(ex.labels)} labels, folds "
        f"{np.bincount(ex.folds).tolist()}; {CFG.threads} threads, {CFG.epochs} epochs")

    print("\n== Checks (before anything is read)")
    ok1, c1_mean, _ = check_c1(data_dir, cache)
    if not ok1 and not SMOKE:
        print("C1 FAILED: the reimplementation is not the published model. Stop, per the plan.")
        return
    ok3 = check_c3(ex, data_dir)
    if not ok3 and not SMOKE:
        print("C3 FAILED. Stop, per the plan.")
        return
    ok2, _ = check_c2(ex, graphs, cache)
    if not ok2 and not SMOKE:
        print("C2 FAILED: something leaks. Stop, per the plan.")
        return

    print("\n== D1 / D1b (structure only)")
    d1(ex, sw_enose_dir, reports)
    d1b(ex)

    print("\n== Training the POM ensembles (10 models x 5 rule-3 folds)")
    P = pom_ensemble(ex, graphs, cache)
    S = references(ex, data_dir, labels112)
    S["POM"] = P
    Y, folds, groups = ex.Y, ex.folds, ex.groups

    print("\n== Q2: the published number under rule 3")
    q2_138, per138 = macro_fold_auc(Y, P, folds)
    q2_112, _ = macro_fold_auc(Y, P, folds, i112)
    print(f"POM ensemble, rule-3 folds: macro AUC {q2_138:.4f} over 138 labels (folds "
          f"{np.round(per138, 3).tolist()}), {q2_112:.4f} over the 112; published "
          f"{PUBLISHED} on stratified folds; C1 single models here {c1_mean:.4f}")
    for k in ["Lookup-10", "Lookup-W", "RF-Morgan", "F-MW", "RF-MW"]:
        print(f"   for comparison, {k:9s} over the 112: {macro_fold_auc(Y, S[k], folds, i112)[0]:.4f}")

    R = {k: within_fold_ranks(np.nan_to_num(v, nan=0.0), folds) for k, v in S.items()}
    far = ex.nn < 0.5
    far_pos = Y[far][:, i112].sum(0)
    ok = i112[far_pos >= 10]
    print("\nBy bin (macro AUC, within-fold ranks, labels with >= 10 positives in the bin, "
          "among the 112):")
    for b in sorted(set(ex.nn_bins)):
        m = ex.nn_bins == b
        cols = i112[(Y[m][:, i112].sum(0) >= 10)]
        a = {k: np.nanmean(auc_columns(Y[m][:, cols], R[k][m][:, cols]))
             for k in ["POM", "RF-Morgan", "Lookup-10", "Lookup-W"]}
        print(f"   {b}: {m.sum():5d} mols, {len(cols):3d} labels; " +
              ", ".join(f"{k} {v:.3f}" for k, v in a.items()))

    print(f"\n== Pooled below 0.5 similarity: {len(ok)} labels, {far.sum()} molecules; 2,000 "
          f"joint group bootstraps, seed 0")
    def fmt(r):
        return f"{r['estimate']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"

    # Amendment (plan, before the run): one comparator per margin; no per-resample max.
    q1 = {c: pooled_margin(Y, R, groups, far, ok, "POM", [c], n_boot=N_BOOT)
          for c in ["Lookup-10", "Lookup-W"]}
    above = all(r["lo"] > 0 for r in q1.values())
    below = all(r["hi"] < 0 for r in q1.values())
    q1r = ("BEYOND SIMILARITY" if above else "WORSE THAN LOOKUP" if below
           else "NO EVIDENCE BEYOND SIMILARITY")
    for c, r in q1.items():
        print(f"Q1: POM over {c:9s}: {fmt(r)}, width {r['hi'] - r['lo']:.4f}")
    print(f"Q1 verdict (both above 0 / both below 0 / otherwise): {q1r}")
    s1 = {c: pooled_margin(Y, R, groups, far, ok, "POM", [c], n_boot=N_BOOT)
          for c in ["F-MW", "RF-MW"]}
    for c, r in s1.items():
        print(f"S1: POM over {c:9s}: {fmt(r)}")
    print(f"S1 verdict: {'persists' if all(r['lo'] > 0 for r in s1.values()) else 'not shown to persist'}")
    for c in ["Lookup-10", "Lookup-W"]:
        r = pooled_margin(Y, R, groups, far, ok, "RF-Morgan", [c], n_boot=N_BOOT)
        print(f"   reference (decides nothing): RF-Morgan over {c:9s}: {fmt(r)}")

    pos_tot = Y.sum(0)
    Ym, gm = Y[far], groups[far]
    Rm = {k: R[k][far] for k in ["POM", "Lookup-10", "Lookup-W"]}

    def s2_stat(cols, comparator):
        def stat(idx):
            yy = Ym[idx][:, cols]
            mg = (auc_columns(yy, Rm["POM"][idx][:, cols])
                  - auc_columns(yy, Rm[comparator][idx][:, cols]))
            keep = ~np.isnan(mg)
            return spearmanr(mg[keep], pos_tot[cols][keep])[0] if keep.sum() >= 3 else np.nan
        return stat

    print("\nS2: Spearman of the distant-bin margin with total positives (same resamples as Q1); "
          "decided on Lookup-W, Lookup-10 alongside")
    cat = {"all": ok,
           "chemical": np.array([j for j in ok if ex.labels[j] in CHEMICAL]),
           "quality": np.array([j for j in ok if ex.labels[j] in QUALITY]),
           "source": np.array([j for j in ok if ex.labels[j] not in CHEMICAL + QUALITY])}
    for name, cols in cat.items():
        if len(cols) < 3:
            print(f"   {name:8s}: {len(cols)} labels, too few")
            continue
        rw = joint_group_bootstrap(s2_stat(cols, "Lookup-W"), gm, n_boot=N_BOOT)
        r10 = joint_group_bootstrap(s2_stat(cols, "Lookup-10"), gm, n_boot=N_BOOT)
        print(f"   {name:8s}: {len(cols):3d} labels, rho vs Lookup-W {rw['estimate']:+.3f} "
              f"[{rw['lo']:+.3f}, {rw['hi']:+.3f}]; vs Lookup-10 {r10['estimate']:+.3f} "
              f"[{r10['lo']:+.3f}, {r10['hi']:+.3f}]")

    A = {k: np.array([fold_aucs(Y[:, j], S[k][:, j], folds).mean() for j in i112])
         for k in ["POM", "RF-Morgan", "Lookup-10", "Lookup-W", "F-MW", "RF-MW"]}
    far_m = {k: auc_columns(Y[far][:, i112], R[k][far][:, i112])
             for k in ["POM", "Lookup-10", "Lookup-W"]}
    tab = pd.DataFrame({"positives": pos_tot[i112], "far_positives": far_pos,
                        **{f"{k}_auc": v for k, v in A.items()},
                        "far_margin_vs_lookup10": far_m["POM"] - far_m["Lookup-10"],
                        "far_margin_vs_lookupW": far_m["POM"] - far_m["Lookup-W"],
                        "category": ["chemical" if lab in CHEMICAL else "quality"
                                     if lab in QUALITY else "source" for lab in labels112]},
                       index=pd.Index(labels112, name="label"))
    tab.sort_values("positives", ascending=False).round(4).to_csv(reports / "06_m4_per_label.csv")
    log(f"Per-label table: {reports / '06_m4_per_label.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--sw-enose-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        SMOKE, N_MODELS, N_BOOT = True, 1, 50
        CFG = TrainConfig(epochs=1)
    out = REPO_ROOT / "reports"
    if args.smoke:  # never write meaningless smoke tables where real results go
        out = args.data_dir / "m4_smoke" / "reports"
        out.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(REPO_ROOT / "reports" / "04_floor_per_label.csv", out)
    main(args.data_dir, args.sw_enose_dir, out)
