# Plan: milestone 2, one label from structure

Written 17 Nov 2026, in the session that measured the floor and **before any structural model
has been run**. Everything below is fixed now. If the run disappoints, the response is a new
plan with a new date, not an edit to this one.

## The question

Can a model given molecular structure predict GS+LF's **`floral`** descriptor better than molecular
weight alone, on molecules held out by rule 3, and does any of that advantage survive on
molecules with no close analogue in training?

## Why `floral`

Chosen from label structure and floor alone (`reports/04_floor_per_label.csv` and the counts
below), without seeing any structural model's performance:

| | floral | runner-up: woody | fruity |
|---|---|---|---|
| Positives / groups carrying them | 1,100 / 797 | 678 / 615 | 1,901 / 1,183 |
| Largest group's share of positives | 3.1% | 1.0% | 9.5% (the ester group) |
| Positives with no training neighbour ≥ 0.5 | 172 | 216 | 215 |
| Fewest positives in any fold | 196 | 111 | 353 |
| MW floor, AUC (fold SD) | 0.681 (0.012) | 0.731 (0.023) | 0.664 (0.020) |
| In Arctander's vocabulary | yes | yes | yes |

Floral has the most precise floor of any well-populated label: the lowest fold SD among labels
with ≥ 250 positives. Its positives are spread across many groups, and it has enough
low-similarity positives for the hard bin to be measurable. It also exists in Arctander, so an
external test is possible later. Woody's floor is higher and noisier. Fruity is dominated by
one ester family.

## Data and split (no choices left)

- GS+LF, 4,981 molecules after merging duplicate records (`collapse_duplicates`).
- Rule-3 groups and folds exactly as `demo_dataset_survey.py` builds them:
  `similarity_components(fps, keys=keys)` at 0.70, then `assign_folds(groups, keys)`, 5 folds.
- One label: `floral`. Other labels are not modelled, looked at, or reported in milestone 2.

## Models, fixed now

| Name | Role | Specification |
|---|---|---|
| **RF-Morgan** | the model | `RandomForestClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=1, random_state=0, n_jobs=-1)` on Morgan radius 2, 2048-bit vectors (`morgan_fingerprints`) |
| F-MW | floor (recorded) | `knn_1d` on MolWt, k = 50: AUC 0.681 |
| RF-MW | learner-matched floor | the same forest, same seed, on MolWt as its only feature |
| kNN-Tanimoto | reference: neighbour lookup | mean label of the 10 training molecules with highest Tanimoto similarity, ties by training order |
| RF-Morgan, shuffled | pipeline check | RF-Morgan trained on `floral` permuted within the training folds (seed 0) |

**The floor for each fold is the higher of F-MW and RF-MW.** That way the margin measures
structure, not a better learner.

No hyperparameter search, no feature selection, no class weighting, no second seed, and no
other fingerprint. One run.

## Measurements

1. Per-fold ROC AUC and average precision for every model above (`per_label_per_fold`).
2. **Paired margin** per fold: RF-Morgan AUC minus that fold's floor. Report all five and the
   mean.
3. **Margin by distance bin**: within-fold ranked scores (`by_bin` / `within_fold_ranks`).
   Primary bin is **NN similarity < 0.5** (0.0–0.4 and 0.4–0.5 merged: about 1,090 molecules,
   172 positives). Uncertainty comes from 2,000 bootstrap resamples **of groups**, not
   molecules, seed 0, giving a 95% percentile interval for the margin.
4. RF-Morgan minus kNN-Tanimoto, per fold and by bin. Reported, but it does not gate the
   outcome.

## The bar

**Pipeline check first.** If RF-Morgan shuffled scores mean AUC outside 0.45–0.55, something
leaks. In that case stop, find it, and report that instead of any result.

**Success**, all three required:
- mean paired margin over the floor ≥ **0.05** AUC;
- margin > 0 in **all 5** folds;
- in the < 0.5 similarity bin, the group-bootstrap 95% interval of the margin lies **entirely
  above 0**.

**Failure**, either one suffices:
- mean paired margin < **0.02**; or
- margin ≤ 0 in **two or more** folds.

**Inconclusive**: anything else, including passing overall but failing in the < 0.5 bin. It is
recorded as inconclusive, not rounded up.

Why 0.05: it is four times the floor's fold SD for floral (0.012), so a margin that size is not
fold noise. It is also large enough to matter when the floor is 0.68. Why the bin condition:
75% of molecules have a training neighbour at 0.5–0.7. Without that condition, a model that
only recognises close analogues could pass.

### How the kNN-Tanimoto comparison is read

- RF-Morgan passes the bar but does **not** beat kNN-Tanimoto: the result is written up as
  "structure helps, but only as far as neighbour lookup does". That is a weaker claim.
- RF-Morgan beats both overall but not kNN-Tanimoto in the < 0.5 bin: same reading.

## What follows each outcome

- **Success.** Milestone 3 extends the identical protocol to all 112 labels with ≥ 50
  positives, each against its own floor in `reports/04_floor_per_label.csv`. In parallel it fixes
  the Arctander vocabulary mapping and runs the label-provenance check
  (`reports/03_overlap.md`), both before any external score is seen.
- **Failure.** Recorded in this file's results section with the same care as a success.
  Whether a different representation helps is a new question with its own plan. Re-running on
  these folds with changed settings until something passes is not allowed.
- **Inconclusive.** Recorded as such. A follow-up needs a new plan that says in advance what
  would resolve it.

## Not in milestone 2

- Other labels, multi-label models, neural networks, pretrained embeddings.
- Arctander, Keller, Dravnieks or Mayhew as test sets.
- Anything generative (readme: prediction, not design).

## Deliverable

`demo_milestone2_floral.py`, a caller of the package. Anything reusable (the Tanimoto kNN,
the group bootstrap, the paired margin) goes into `sw_qsor/`, with tests. It prints every number
under "Measurements" and the verdict against the bar. A results section is appended below,
whatever the verdict.
