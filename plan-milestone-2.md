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

## Results (17 Nov 2026)

Run once with `python demo_milestone2_floral.py`, from commit `49c0f9f`'s plan, unmodified.
It took 74 seconds. The settings were not changed and nothing was re-run.

**Verdict: SUCCESS against the bar. Under the plan's reading rule this is "structure helps,
but only as far as neighbour lookup does":** RF-Morgan beats kNN-Tanimoto overall but not in
the < 0.5 similarity bin.

### Checks

- F-MW recomputed: 0.6810, the same as the recorded floor.
- Pipeline check: RF-Morgan trained on shuffled labels scored per-fold AUC 0.517 / 0.494 /
  0.499 / 0.459 / 0.535, mean **0.501**. That is inside 0.45–0.55, so no leak.

### Per-fold results (1,100 positives; 196–236 per fold)

| Model | Fold AUCs | Mean | SD | Mean AP |
|---|---|---|---|---|
| F-MW (recorded floor) | 0.678, 0.693, 0.694, 0.667, 0.673 | 0.681 | 0.012 | 0.344 |
| RF-MW (learner-matched floor) | 0.725, 0.739, 0.732, 0.721, 0.730 | **0.730** | 0.007 | 0.393 |
| kNN-Tanimoto (neighbour lookup) | 0.808, 0.789, 0.823, 0.804, 0.808 | 0.806 | 0.012 | 0.532 |
| **RF-Morgan** | 0.834, 0.809, 0.834, 0.820, 0.835 | **0.826** | 0.012 | 0.588 |

Prevalence is 0.221. RF-MW was the higher floor in every fold, so the bar was 0.73, not
0.68. Including the learner-matched floor was worth it: the same forest given only molecular
weight gains 0.05 over the 50-NN floor.

Paired margin over the floor: **+0.109, +0.070, +0.103, +0.098, +0.105; mean +0.097, positive
in 5/5.** RF-Morgan minus kNN-Tanimoto: +0.026, +0.020, +0.012, +0.016, +0.027; mean +0.020,
positive in 5/5.

### By distance to training

Within-fold ranked scores; 95% intervals from 2,000 group bootstraps, seed 0. In each resample
the floor is the higher of F-MW and RF-MW. The plan did not specify this, and it was chosen as
the conservative reading before the run.

| Nearest-neighbour similarity | Molecules | Positives | Margin over floor | Margin over kNN-Tanimoto |
|---|---|---|---|---|
| **< 0.5 (primary)** | 1,090 | 172 | **+0.083 [+0.032, +0.132]** | +0.004 [−0.025, +0.034] |
| 0.0–0.4 | 455 | 49 | +0.078 [−0.017, +0.160] | +0.038 [−0.025, +0.105] |
| 0.4–0.5 | 635 | 123 | +0.077 [+0.017, +0.137] | −0.006 [−0.036, +0.024] |
| 0.5–0.6 | 1,536 | 294 | +0.092 [+0.054, +0.129] | +0.021 [+0.000, +0.044] |
| 0.6–0.7 | 2,355 | 634 | +0.104 [+0.076, +0.130] | +0.023 [+0.010, +0.038] |

### Against the bar

| Condition | Required | Observed | Met |
|---|---|---|---|
| Mean paired margin | ≥ 0.05 | 0.097 | yes |
| Folds with margin > 0 | 5/5 | 5/5 | yes |
| < 0.5 bin, 95% interval | entirely above 0 | [+0.032, +0.132] | yes |

### What it means

- **Structure carries information about `floral` that size does not.** RF-Morgan exceeds the
  better of two MW floors by about 0.1 AUC in every fold. The margin persists, at +0.08, on
  molecules with no training neighbour at 0.5 or above.
- **Nearly all of that is available to neighbour lookup.** Averaging the labels of the ten
  most similar training molecules reaches 0.806 against the forest's 0.826. The forest's extra
  0.02 is consistent across folds but comes from the 0.5–0.7 bins. Below 0.5 the two cannot be
  told apart (+0.004, interval spanning 0). So on molecules unlike anything in training, this
  experiment shows no structure–odour learning beyond similarity.
- **The margin over the floor shrinks with distance** (+0.104 at 0.6–0.7, +0.078 below 0.4), as
  expected if part of it is analogue recognition. The 0.0–0.4 bin has only 49 positives, and its
  interval includes 0.
- **One label, one model, one run.** Nothing here says how other labels behave. Arctander has
  not been touched, so nothing here is an external test either.

### Next, per the plan's "What follows each outcome"

Milestone 3: the identical protocol for all 112 labels with ≥ 50 positives, each against its own
floor. Before any external score is seen, it also fixes the Arctander vocabulary mapping and
runs the label-provenance check. The neighbour-lookup reading above makes kNN-Tanimoto worth
keeping as a standing reference in milestone 3. That is a decision for milestone 3's own plan,
written before it runs.
