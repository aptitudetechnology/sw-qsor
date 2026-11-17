# 03 — Overlap between datasets, and what can serve as an independent test set

Written 17 Nov 2026. Structure only; no labels were used to compute anything here, except
Mayhew's label *sources* in the last section. Reproduce with `python demo_dataset_survey.py`.

## Answer

**Only Arctander can provide a sizeable odour-descriptor test set independent of GS+LF: 901
molecules in 838 rule-3 groups.** Keller 2016 is 98% inside GS+LF and Dravnieks 92%, so
neither can validate a model trained on GS+LF; what is left of them (8 and 10 molecules) is
too small to measure anything. Mayhew contributes 1,109 independent molecules, but its label
is a different task: whether a molecule has an odour at all, not what it smells like.

## Identity

Molecules are matched on the InChIKey connectivity block (`sw_qsor.identity`). It ignores
stereochemistry, which is necessary because GS+LF and Keller distribute SMILES without it, and
protonation. Distinct molecules per dataset (rule 1):

| Dataset | Rows | No structure | Distinct full InChIKeys | Distinct molecules | Merged |
|---|---|---|---|---|---|
| GS+LF | 4,983 | 0 | 4,983 | **4,981** | 2 (neutral/ionised duplicates) |
| Arctander | 2,751 | 0 | 2,751 | **2,718** | 33 (stereoisomers) |
| Mayhew | 1,865 | 0 | 1,865 | **1,856** | 9 |
| Keller | 481 | 2 | 474 | **474** | 0 (but 5 duplicate rows: 1 repeated CID, 4 stereo pairs whose stereo was dropped upstream) |
| Dravnieks | 138 | 0 | 138 | **138** | 0 |
| Abraham | 268 | 0 | 268 | **266** | 2 |

The survey's counts (4,983, 2,751, 1,865, 481) were row counts. The differences are small,
except for what they say about stereo: 33 Arctander molecules are distinguished only by
stereochemistry, and a stereo-free key cannot tell them apart.

## Shared molecules

| | GS+LF | Arctander | Mayhew | Keller | Dravnieks | Abraham |
|---|---|---|---|---|---|---|
| **GS+LF** | 4,981 | 1,477 | 656 | 464 | 127 | 168 |
| **Arctander** | 1,477 | 2,718 | 468 | 326 | 101 | 141 |
| **Mayhew** | 656 | 468 | 1,856 | 248 | 91 | 254 |
| **Keller** | 464 | 326 | 248 | 474 | 69 | 86 |
| **Dravnieks** | 127 | 101 | 91 | 69 | 138 | 30 |
| **Abraham** | 168 | 141 | 254 | 86 | 30 | 266 |

Share of the row dataset contained in the column dataset:

| | GS+LF | Arctander | Mayhew | Keller | Dravnieks | Abraham |
|---|---|---|---|---|---|---|
| **Arctander** | 54% | — | 17% | 12% | 4% | 5% |
| **Mayhew** | 35% | 25% | — | 13% | 5% | 14% |
| **Keller** | **98%** | 69% | 52% | — | 15% | 18% |
| **Dravnieks** | **92%** | 73% | 66% | 50% | — | 22% |
| **Abraham** | 63% | 53% | **95%** | 32% | 11% | — |

Stripping salts and neutralising (RDKit `ChargeParent`) adds at most 12 matches with GS+LF
(Mayhew 656 → 668). The result does not depend on how salts are handled.

## Independence under rule 3

A molecule absent from GS+LF is not automatically independent of it. Under rule 3, a test
molecule must not have a training molecule at Tanimoto ≥ 0.70, so a molecule in the same group
as a GS+LF molecule could never be held out from a model trained on GS+LF. Grouping the union of
all six datasets (7,389 distinct molecules) by rule 3:

| Dataset | Molecules | In GS+LF | Not in GS+LF but same group | **Independent** | Groups |
|---|---|---|---|---|---|
| Arctander | 2,718 | 1,477 | 340 | **901 (33%)** | 838 |
| Mayhew | 1,856 | 656 | 91 | **1,109 (60%)** | 1,044 |
| Keller | 474 | 464 | 2 | **8 (2%)** | 8 |
| Dravnieks | 138 | 127 | 1 | **10 (7%)** | 10 |
| Abraham | 266 | 168 | 13 | **85 (32%)** | 75 |

Across all five, 1,985 distinct molecules are independent of GS+LF.

## What follows

- **Keller and Dravnieks are inside GS+LF.** Their value is as *different measurements of the
  same molecules*: panel intensity and pleasantness ratings (Keller) and graded applicability
  (Dravnieks), against GS+LF's binary descriptors. They can test whether GS+LF labels agree
  with panel data. They cannot test generalisation to new molecules.
- **Arctander's 901 is the external test set for descriptors**, subject to two checks not yet
  made. (1) Its vocabulary differs from GS+LF's 138 labels and must be mapped, and that
  mapping must be fixed before any model is scored on it. (2) Label provenance: GoodScents
  descriptions often draw on Arctander's text, so even on disjoint molecules the *labelling*
  may not be independent. A molecule-level check is impossible on disjoint molecules; the
  check is whether, on the 1,477 shared molecules, the two sources agree more than separately
  produced labels would.
- **Mayhew answers a different question** (odorous or not: 1,615 "Odor", 309 "Odorless" across
  1,924 label rows). Its independent molecules lean odourless: 260 of 1,224 label rows not in
  GS+LF are "Odorless", against 49 of 700 inside it. That is expected, since GS+LF only
  contains molecules someone described as smelling of something.
- **Abraham is largely inside Mayhew** (95%) and is one of Mayhew's label sources (126 rows),
  so the two are not independent of each other either.
- **The cost of the stereo-free key.** It is unavoidable for matching against GS+LF, but it
  means enantiomers that smell different (the carvones, for instance) count as one molecule. Any
  task that depends on stereo, if one is ever attempted, needs Arctander or Mayhew alone,
  matched on full InChIKeys.

## A correction this step made to step 2

GS+LF's two duplicate pairs have fingerprint similarity of only 0.31 and 0.55, so the
similarity components alone could have split a molecule across train and test. Fixed:
`similarity_components(..., keys=...)` links equal identity keys. See the correction note in
`reports/02_grouping.md`.
