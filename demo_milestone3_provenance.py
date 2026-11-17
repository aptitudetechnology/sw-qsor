"""Milestone 3, Phase A: Arctander vocabulary, label provenance, Phase C's label set.

Usage: python demo_milestone3_provenance.py [--data-dir DIR]

Labels and structures only; no model is fitted (plan-milestone-3.md, Phase A). Writes
reports/05_arctander_vocab.csv and reports/05_phase_c_labels.csv.

Points the plan left open, resolved here before any model touches Arctander:
  - Arctander molecules with no label at all are treated as missing, not as negatives:
    their text descriptions show the labels were not extracted (load_arctander_labels).
  - A2 uses the 64 mapped labels among the 112, the set Phase C draws from.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from sw_qsor.datasets import (
    load_arctander_labels,
    load_gslf,
    load_openpom_source,
    load_pyrfume_molecules,
)
from sw_qsor.evaluate import joint_group_bootstrap
from sw_qsor.experiment import arctander_by_key, independence
from sw_qsor.identity import collapse_duplicates, key_table
from sw_qsor.splits import morgan_fingerprints, similarity_components
from sw_qsor.vocab import arctander_mapping

REPO_ROOT = Path(__file__).resolve().parent
MIN_POS_A2 = 20
MIN_POS_C = 20
KAPPA_THRESHOLD = 0.10


def by_key(smiles, Y):
    """Connectivity key per row, then union of label rows sharing a key."""
    keys = key_table(smiles).key.tolist()
    keep, merged = collapse_duplicates(keys, np.asarray(Y))
    return pd.DataFrame(merged, index=[keys[i] for i in keep])


def main(data_dir, reports):
    gslf, gslf_labels = load_gslf(data_dir / "gslf")
    floor = pd.read_csv(reports / "04_floor_per_label.csv", index_col=0)
    the_112 = set(floor.index[floor.positives >= 50])

    # ---- A1: vocabulary
    arc, arc_labels, counts = load_arctander_labels(data_dir / "arctander_1960")
    mapping = arctander_mapping(arc_labels, gslf_labels)
    vocab = pd.DataFrame([{"arctander": a, "gslf": g, "kind": "exact" if a == g else "variant",
                           "in_112": g in the_112} for a, g in mapping.items()])
    vocab.to_csv(reports / "05_arctander_vocab.csv", index=False)
    mapped = vocab[vocab.in_112]
    print("== A1. Vocabulary (fixed by name in the plan)")
    print(f"Arctander labels {len(arc_labels)}; mapped {len(vocab)} "
          f"({(vocab.kind == 'exact').sum()} exact, {(vocab.kind == 'variant').sum()} variant); "
          f"among the 112: {len(mapped)} (plan: 64)")
    print(f"Arctander stimuli {counts['stimuli']}; without a resolved structure (CID 0) "
          f"{counts['stimuli_without_structure']}; molecules {counts['molecules']}, of which "
          f"unlabelled {counts['unlabelled_molecules']} (treated as missing)")
    arc_lab = arc[arc.labelled]
    # Many-to-one targets (Arctander has both 'vanilla' and 'vanillin' -> 'vanilla') are
    # unioned: the GS+LF label is present if Arctander gives either.
    targets = list(dict.fromkeys(mapped.gslf))
    merged = np.column_stack([arc_lab[mapped.arctander[mapped.gslf == t]].max(axis=1)
                              for t in targets])
    A = by_key(arc_lab.smiles, merged)
    A.columns = targets
    assert A.equals(arctander_by_key(data_dir / "arctander_1960", gslf_labels, targets))
    many = mapped.gslf[mapped.gslf.duplicated(keep=False)]
    print(f"Distinct GS+LF targets among the 112: {len(targets)}; many-to-one: "
          f"{', '.join(f'{t} <- ' + ' + '.join(mapped.arctander[mapped.gslf == t]) for t in many.unique())}")
    print(f"Labelled Arctander molecules by connectivity key: {len(A)}\n")

    # ---- A2: provenance
    gs, gs_labels = load_openpom_source(data_dir / "openpom_goodscents", "curated_goodcents.csv")
    lf, lf_labels = load_openpom_source(data_dir / "openpom_leffingwell",
                                        "curated_leffingwell.csv")
    labs = [g for g in targets if g in gs_labels and g in lf_labels]
    G = by_key(gs.smiles, gs[labs].to_numpy())
    G.columns = labs
    L = by_key(lf.smiles, lf[labs].to_numpy())
    L.columns = labs
    common = sorted(set(G.index) & set(L.index) & set(A.index))
    print("== A2. Does GoodScents track Arctander more closely than Leffingwell does?")
    print(f"Molecules by key: GoodScents {len(G)}, Leffingwell {len(L)}, labelled Arctander "
          f"{len(A)}; in all three: {len(common)}")
    if len(labs) < len(targets):
        print(f"  ({len(targets) - len(labs)} mapped label(s) missing from a source vocabulary: "
              f"{sorted(set(targets) - set(labs))})")
    g, lv, a = G.loc[common, labs], L.loc[common, labs], A.loc[common, labs]
    used = [x for x in labs if a[x].sum() >= MIN_POS_A2]
    print(f"Labels with >= {MIN_POS_A2} Arctander positives on these molecules: {len(used)}")
    gv, lvv, av = g[used].to_numpy(), lv[used].to_numpy(), a[used].to_numpy()

    def kappas(idx, src):
        out = []
        for j in range(len(used)):
            y1, y2 = src[idx, j], av[idx, j]
            out.append(cohen_kappa_score(y1, y2) if len(set(y1)) > 1 and len(set(y2)) > 1
                       else np.nan)
        return np.array(out)

    k_gs, k_lf = kappas(slice(None), gv), kappas(slice(None), lvv)
    per = pd.DataFrame({"arc_pos": av.sum(0), "gs_pos": gv.sum(0), "lf_pos": lvv.sum(0),
                        "kappa_gs": k_gs, "kappa_lf": k_lf, "diff": k_gs - k_lf}, index=used)
    print(per.sort_values("diff", ascending=False).round(3).to_string())
    # rule-3 groups over the shared molecules; every one is in GoodScents, so use its SMILES
    gs_reps = pd.Series(gs.smiles.values, index=key_table(gs.smiles).key)
    gs_reps = gs_reps[~gs_reps.index.duplicated()]
    groups = similarity_components(morgan_fingerprints(gs_reps[common].tolist()), keys=common)

    def stat(idx):
        return float(np.nanmean(kappas(idx, gv) - kappas(idx, lvv)))

    r = joint_group_bootstrap(stat, groups)
    print(f"\nMacro-mean kappa: GoodScents vs Arctander {np.nanmean(k_gs):.3f}, Leffingwell vs "
          f"Arctander {np.nanmean(k_lf):.3f}")
    print(f"Difference {r['estimate']:+.3f}, 95% group-bootstrap interval [{r['lo']:+.3f}, "
          f"{r['hi']:+.3f}] ({r['groups']} groups, {r['n_boot']} resamples, "
          f"{r['nan_draws']} NaN)")
    derived = r["estimate"] >= KAPPA_THRESHOLD and r["lo"] > 0
    print("DECISION: " + ("Phase C is a test of CROSS-SOURCE CONSISTENCY, not independent "
                          "validation." if derived else
                          "Phase C is reported as an EXTERNAL TEST (with the plan's caveat)."))
    print(f"  (rule: difference >= {KAPPA_THRESHOLD} and interval excluding 0)\n")

    # ---- A3: Phase C label set
    sm = {"GS+LF": gslf.smiles.tolist()}
    for name, a_dir in [("Arctander", "arctander_1960"), ("Mayhew", "mayhew_2022"),
                        ("Keller", "keller_2016"), ("Dravnieks", "dravnieks_1985"),
                        ("Abraham", "abraham_2012")]:
        sm[name] = load_pyrfume_molecules(data_dir / a_dir).smiles.tolist()
    ind = independence(sm, "GS+LF")
    indep = ind["classes"]["Arctander"]["independent"]
    labelled_indep = sorted(indep & set(A.index))
    print("== A3. Phase C label set")
    print(f"Arctander molecules independent of GS+LF: {len(indep)} (report 03: 901); "
          f"of those labelled: {len(labelled_indep)}")
    pos = A.loc[labelled_indep].sum().sort_values(ascending=False)
    c = pd.DataFrame({"positives_in_independent_labelled": pos,
                      "phase_c": pos >= MIN_POS_C})
    c.index.name = "gslf_label"
    c.to_csv(reports / "05_phase_c_labels.csv")
    print(f"Mapped labels with >= {MIN_POS_C} positives there: {int(c.phase_c.sum())} of "
          f"{len(c)}: {', '.join(c.index[c.phase_c])}")
    print(f"\nWritten: {reports / '05_arctander_vocab.csv'}, "
          f"{reports / '05_phase_c_labels.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = ap.parse_args()
    main(args.data_dir, REPO_ROOT / "reports")
