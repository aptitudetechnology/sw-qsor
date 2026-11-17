# Plan: milestone 5, does a learned odour representation help sw-enose's receptors?

**Fixed 18 Nov 2026, before the run.** Changes after this commit are a new plan, not edits to
this one. Nothing below has been run. Counts of
sw-enose's receptors, molecules, groups and folds were recomputed here from its data rather
than taken from its notes, and are marked "measured here". sw-enose's own *model results* are
cited from its notes and reports and have not been re-run.

This milestone exists to deliver something to sw-enose, so it starts from sw-enose's
constraints rather than sw-qsor's interests.

## What sw-enose needs, and what it does not

sw-enose is blocked by missing receptor binding data. It closed three routes to predicting it,
each against a bar fixed in advance, and later a fourth: per-receptor neighbour lookup on
fingerprints, which also failed (−0.034 below its floor, 0 of 5 folds).

**Its constraint is labels, not features** (confirmed here from
`sw-enose/data/m2or/repo/M2OR_20230428.csv`): of 1,020 receptors, **719 have no known agonist
at all**, and only **59 have ≥ 5 positives**. No representation helps a target with no
positives. So milestone 5 cannot supply sw-enose's missing data, and does not try.

What it can do is test one question cleanly, on the only part of the data where any
representation could matter:

> **Does sw-qsor's learned odour representation predict receptor responses better than
> sw-enose's 20 interpretable descriptors, on the 59 receptors that have enough positives to
> learn from?**

Milestone 4 is the reason to ask. The POM graph network beat similarity lookup on molecules
unlike its training set (+0.081 [+0.069, +0.093] below 0.5 similarity), which no
fingerprint model in milestones 2–3 managed. It is the first representation here shown to carry
transferable structure–percept information.

## The 59 receptors, measured here

| | |
|---|---|
| Receptors with ≥ 5 positives | **59** (of 1,020) |
| Screened pairs among them | 2,730 |
| Positives among them | 517 (18.9%) |
| Molecules screened per receptor | median 45 (range 8–111) |
| Positives per receptor | median 7 (range 5–34) |
| Receptors with ≥ 5 positives **and** ≥ 5 negatives | 57 |
| Receptors with ≥ 10 positives | 14 |

With a median of 7 positives spread over 5 folds, **per-receptor per-fold AUC is not
measurable**. Scoring is therefore pooled over the 59 receptors' pairs within each fold, as
sw-enose's own M2OR demos do, and never pooled across folds (`evaluate.within_fold_ranks`).

## Molecules and split

The 180 M2OR odorants, all with committed structures. Measured here:

- Rule-3 groups at Tanimoto ≥ 0.70: **115 groups, largest 10 molecules (6%)**, giving 5 folds
  of exactly **36 molecules** each. This replaces the Murcko split sw-enose used, whose folds
  were 7 / 11 / 42 / 3 / 117 (`sw-enose/reports/sw-qsor-audit-m2or-splits.md`).
- Pairs per fold for the 59 receptors: 522 / 325 / 481 / 665 / 737, with **69 / 123 / 126 /
  112 / 87 positives**. Every fold is scoreable.
- **No molecule among the 180 is bondless**, so POM's hard limit (`plan-milestone-4.md`, D1b)
  does not bite here. Three have ≤ 5 fingerprint bits.

## The leakage that must be controlled, and how

**163 of the 180 molecules are inside GS+LF** (measured; about 32 of each 36-molecule fold). A
POM model trained on all of GS+LF has therefore seen most test molecules *with their percept
labels*. That is not receptor-label leakage, and the embedding never sees M2OR at all. But it
would make the comparison against descriptors unfair in a specific way: descriptors are
computed from structure with no fitting, while the embedding would have been fitted knowing the
test molecules' percepts.

So two embedding variants, and the **primary one is the clean one**:

| Name | How the POM model is trained | Role |
|---|---|---|
| **E-held** | Per receptor fold, on GS+LF **minus** every molecule sharing a rule-3 group with a test-fold molecule. 5 models, seed 0. | **primary** |
| E-all | Once on all of GS+LF, seed 0. Matches what a practitioner would download. | secondary, reported |

**Groups are computed over the union** of GS+LF and the 180 M2OR molecules, deduplicated by
connectivity key (4,998 distinct molecules, measured here), so a GS+LF molecule similar to a
test molecule is excluded too, not merely an identical one. Measured here, E-held drops little
training data: **275 / 80 / 61 / 92 / 78 GS+LF molecules for folds 0–4** (1.2–5.5%), leaving
4,706–4,920 to train on. POM settings are milestone 4's, unchanged (`sw_qsor/pom.py`, 62
epochs, 12 threads, deterministic).

**Cost**, from milestone 4's measured 8 min 50 s per POM model: 5 E-held models plus 1 E-all,
about **55 minutes**, plus minutes for the gradient-boosting fits and the permuted-label check.
Well under an hour, so nothing here needs trimming.

## Representations compared

All share one learner and one receptor term, so only the molecular representation differs.

| Name | Molecular features | Dimensions |
|---|---|---|
| **R0** (floor) | none | — |
| **D** | sw-enose's 20 descriptors (`sw_enose/chem.py`, copied, not imported) | 20 |
| **E-held**, **E-all** | POM embedding | 256 |
| **E-20** | first 20 principal components of the POM embedding, fitted on training molecules only | 20 |

- **Receptor term:** the receptor's smoothed training-fold response rate, exactly as
  `sw-enose/demo_m2or_chem.py` computes it (prior × 5 pseudo-counts). R0 is that term alone,
  which is sw-enose's established chemistry-free floor.
- **Learner:** `HistGradientBoostingClassifier(random_state=0)`, sw-enose's best-performing
  choice, with no tuning.
- **Agonist-profile variant:** sw-enose's third model added, per receptor, the mean feature
  vector of the molecules it responded to in the training fold, plus dot product, distance and
  cosine against the test molecule. It was the natural form of the hypothesis there, so it is
  run for **D** and for **E-held** alike. Reported separately, not merged into the primary
  comparison.
- **E-20 exists to separate information from dimensionality:** 256 features over ~144 training
  molecules can overfit, so a 20-dimensional embedding matches D exactly on width.

## Checks, before anything is read

1. **C1, floor reproduction.** R0's per-fold AUCs must be reported next to sw-enose's own
   figure (per-fold mean 0.791 on its random-molecule split, recomputed in the audit). Rule-3
   folds will differ; a floor below 0.70 or above 0.88 means something is wrong with the port
   of the receptor term, and the run stops.
2. **C2, permuted labels.** The full pipeline with M2OR responses permuted across pairs
   (seed 0) must give per-fold AUC in 0.45–0.55 for **D** and **E-held**.
3. **C3, embedding sanity.** E-held embeddings must differ between every pair of the 180
   molecules (none is bondless, so none should collapse), and the 5 fold models must each load
   and embed all 180.

## The bar

**Primary.** Paired margin per fold of **E-held** over **D**, both on the same folds.

- **Success:** mean paired margin ≥ **0.03** AUC, positive in **all 5** folds, **and** E-held
  beats R0 by ≥ 0.05 on average. The last clause matters: a representation that beats
  descriptors while neither beats the floor has shown nothing about molecules.
- **Failure:** mean paired margin < **0.01**, or negative in **2 or more** folds.
- **Inconclusive:** anything else.

**Secondary, reported, deciding nothing:** D, E-all, E-20 and the agonist-profile variants
against R0; E-all against E-held (how much the in-sample embedding flatters itself); and the
margin restricted to the 14 receptors with ≥ 10 positives.

Why 0.03: sw-enose's own receptor bar was +0.05 over its floor, set for a model against no
model. This compares two representations, where the shared receptor term dominates both, so a
smaller margin is still meaningful. It is fixed here and not negotiable afterwards.

## Expectation, stated before the run

**Failure is the more likely outcome, and the test is still worth running.** With a median of 7
positives per receptor, nothing the features do can overcome the label constraint, and five
routes to receptor prediction have already failed on this data. The reason to run it anyway is
that all five used representations that milestone 4 showed to be weak far from training, and
this is the one that was not.

If it fails, the finding is sharper than another null: *even a representation that demonstrably
carries transferable structure–percept information does not predict receptor responses here*,
which closes the feature side of the question and leaves only the label side.

## Division of labour, and what sw-enose receives

Dependencies point from new to old: **sw-enose does not import sw-qsor.** So sw-qsor runs this
and hands over artefacts:

1. `reports/07_pom_embedding.csv.gz` — the 256-d E-all embedding for the 180 M2OR molecules,
   keyed by InChIKey, with the POM commit, settings and date in a header comment. sw-enose can
   use it without running PyTorch.
2. A results note for sw-enose stating the verdict, the floor under rule-3 folds, and the
   caveats below.
3. `sw_qsor/pom.py` and the rule-3 grouping, already there to copy.

M2OR data is **read** from a sw-enose checkout given as `--sw-enose-dir` and never vendored or
copied into sw-qsor, as milestone 4's D1 did. Its licence is sw-enose's to state.

## Caveats to carry into the results, whatever they are

- **One dataset, one learner, one seed per variant.** A null result is about this setting.
- **The embedding is trained on percepts, and percepts are not receptor responses.** If the
  biological path from receptor activation to percept is many-to-one, a percept-optimal
  representation may discard exactly the receptor-level detail needed here. That is a reason a
  null result would not be surprising, and is not a defect of the test.
- **E-held removes percept-label leakage for the test molecules, not the fact that GS+LF and
  M2OR overlap.** 163 of 180 molecules are in GS+LF; E-held only ensures no model saw the ones
  it is scored on.
- **The 59 receptors are not representative.** They are the best-measured 6%, so a positive
  result would not extend to the 719 with no agonist.

## Not in milestone 5

- The other 961 receptors, EC50 regression, QM/MM, or anything sw-enose has already closed.
- Changes to POM's settings, other architectures, or fine-tuning POM on receptor data.
- The compilation check from `note-2-from-sw-enose.md` §2, still open and unscheduled
  (`next-session.md`); milestone 5 uses no Pyrfume numeric archive.
- Anything generative.

## Deliverables

- `sw_qsor/receptors.py`: M2OR loading by path, the receptor-rate term, the descriptor set
  (copied from sw-enose with its origin noted), and the agonist-profile features — with tests.
- `demo_milestone5.py`: checks, then the comparison, then the verdict. POM models cached per
  fold so a crash resumes.
- `reports/07_m5_per_fold.csv`, `reports/07_pom_embedding.csv.gz`, and a results section
  appended here whatever the outcome.

## Decided at review (18 Nov 2026): all five as recommended

1. **The bar stays 0.03, with the "also beat the floor by 0.05" clause.** sw-enose's own 0.05
   was a model against no model; this compares two representations sharing a receptor term
   that dominates both. The second clause stops a representation "winning" while neither it
   nor descriptors beat the chemistry-free floor.
2. **E-held is primary.** The 5 extra trainings (about 45 minutes) buy the only comparison
   that is fair to descriptors, which involve no fitting at all. E-all is reported alongside.
3. **E-20 is kept**, with PCA fitted on training molecules only, so no test fold leaks into
   the projection.
4. **The agonist-profile variants are included**, reported separately and not deciding the
   verdict, so the comparison is against sw-enose's best model rather than a weaker form of
   its hypothesis.
5. **Scoring is pooled over the 59 receptors' pairs within each fold.** A median of 7
   positives per receptor gives about 1.4 per fold, so per-receptor AUC is not measurable.
   **Stated limitation:** a pooled score weights heavily screened receptors more, so the
   result is dominated by the best-measured receptors among an already best-measured 6%.
