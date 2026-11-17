# Next session: review and fix the milestone 4 plan, then run it

Rewritten 17 Nov 2026. The earlier milestone 1 checklist is in git history (`03d4c88` and
before). Every result it led to is recorded in `plan.md`, `plan-milestone-2.md` and
`plan-milestone-3.md`.

## Where things stand

- **Milestone 1** (`plan.md`): GS+LF holds 4,981 molecules and 3,434 rule-3 groups. Overlap and
  licences are counted. The chemistry-free floor: molecular weight alone gives macro AUC 0.664.
- **Milestone 2** (`plan-milestone-2.md`): floral passes the bar. A forest on Morgan bits beats
  the size floor, but not neighbour lookup below 0.5 similarity.
- **Milestone 3** (`plan-milestone-3.md`):
  - 71 of 112 labels pass (broad);
  - pooled below 0.5 similarity, no evidence of learning beyond neighbour lookup
    (+0.006 [−0.008, +0.020]);
  - the signal transfers to Arctander's independent molecules.
- **`note-from-sw-enose.md`** (read it): sw-enose's receptor route failed because receptors
  have a median of zero positives, not because it had too few molecules. It also observed that
  the margin over lookup shrinks as labels gain positives. Three corrections apply to that
  observation, all descriptive and deciding nothing:
  - the forest is flat across label sizes; it is lookup that improves;
  - k = 10 lookup ties heavily on rare labels (44% of rare-label positives score exactly 0);
  - the observation used overall margins, not distant-bin margins.
  All three are built into the milestone 4 draft.
- **sw-enose** received an audit note, `sw-enose/reports/sw-qsor-audit-m2or-splits.md`. Its
  verdict stands; its scaffold folds and its documented floor needed correcting.
- **`note-2-from-sw-enose.md`** (read it) accepts the corrections and withdraws the first note's
  milestone 4 question; Lookup-W is the agreed response. sw-enose has corrected its documented
  floor (`3124ca3`). The note also reports:
  - neighbour lookup on receptors fails too (−0.034, 0 of 5 folds);
  - `abraham_2012` re-reports Nagata's thresholds exactly;
  - 163 of its 180 molecules are inside GS+LF;
  - only 59 receptors have ≥ 5 positives.
- **sw-enose also holds a reply** to the first note, `sw-enose/reports/reply-from-sw-qsor-to-note.md`.
  It is partial: a safety classifier stopped the write, and the file now points to
  `plan-milestone-4.md` for the remaining correction. It is uncommitted in sw-enose.

`note-3-from-sw-enose.md` — review of milestone 4 while it runs. Nothing is at risk in the run
itself. Two things to act on when reading results: score the 10 ensemble members individually
(free, already cached) to separate ensembling from architecture; and verify the single-atom
mechanism by comparing embeddings rather than outputs, because C1 does not exercise the no-edge
path. Also: all three single-atom molecules in GS+LF are labelled `odorless`, which makes the
hydrogen sulfide result a correct generalisation rather than a representational gap, and a
sharper claim.

## 1. Review `plan-milestone-4.md` (DRAFT)

The question: does OpenPOM's published network beat neighbour lookup on unfamiliar molecules,
and how much of its published 0.8872 survives rule 3? It is reimplemented in plain PyTorch on
the CPU with OpenPOM's own settings, which are read from its files at the pinned commit and
quoted in the plan. The draft ends with five points open for review. Since then it has gained:
- **D1:** sw-enose coverage;
- **D1b:** tiny molecules (≤ 5 fingerprint bits) reported as a limit of fingerprints, scoped
  so the claim does not cover every structure-based model;
- milestone 5's honest scope in "Looking ahead".

## 2. Before committing the plan

- Install the CPU `torch` wheel into `.venv` as an optional `pom` extra. Check first that a
  Python 3.14 wheel exists.
- Write `sw_qsor/pom.py` and its tests: feature dimensions 134 / 6, determinism, and Lookup-W
  against brute force.
- **Measure cost on row-permuted labels only**, so that no result is seen: time per epoch for one
  model on one rule-3 fold. Write the estimate into the plan. If 60 trainings would exceed
  about 24 hours, reduce the ensemble size, and only then.
- Then commit the plan, and only then run it.

## 3. Run milestone 4

Checks in order: C1 (the reimplementation reproduces OpenPOM's single-model ~0.87 under its own
split), C2 (permuted labels give 0.48–0.52), C3 (floral continuity). Then D1/D1b (structure
only, before any result is read), then Q1, Q2, S1 and S2.
Results go in the plan file, whatever they are.

## Then: milestone 5 is for sw-enose

Decided 17 Nov 2026. Milestone 5 plans what sw-qsor delivers to sw-enose, building on milestone
4's outputs:
- D1's coverage table, and D1b's tiny-molecule finding, which sw-enose will cite;
- a POM embedding from a model trained on all of GS+LF after milestone 4;
- the reusable code (`pom.py`, Lookup-W, rule-3 grouping).
Write its plan with sw-enose's needs as the starting point, not sw-qsor's. **Its honest scope**,
from note 2: the constraint is labels, not features. 719 of 1,020 receptors have no agonist, and
only 59 have ≥ 5 positives. So the question is whether a learned embedding beats sw-enose's 20
descriptors on those 59, not whether it supplies missing receptor data.

## Open, not yet scheduled: the compilation check

From `note-2-from-sw-enose.md`, section 2. Two sources can share molecules legitimately, so
InChIKey overlap cannot reveal that one re-reports the other. Exact agreement of values on
shared molecules can. Two places it could go, still to decide:
- **A label-only addendum to milestone 3 Phase A, marked as after the fact:** on molecules
  shared by GoodScents, Leffingwell and Arctander, how often a source's full label set equals
  Arctander's mapped set. This bears on Phase C's framing as an external test.
- **Milestone 5:** exact numeric agreement across Pyrfume's numeric archives (for example
  `abraham_2012` against Nagata), if milestone 5 uses them.

## Standing practice (unchanged)

- Fix the bar before the run and run once. A changed setting is a new plan.
- Commit each phase's results before the next phase starts.
- Data is fetched by script, pinned, with checksums. Nothing is vendored.
- sw-enose takes settings from sw-qsor (Morgan r2/2048, k = 10, a forest of 1,000 trees), so
  keep the two aligned if either changes.
