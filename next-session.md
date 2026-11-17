# Next session: finish milestone 1

Written 17 Nov 2026. Milestone 1's literature half is done (`reports/01_dataset_survey.md`);
what is left is local and needs no external research. Ordered by dependency, not importance.

## 0. Package skeleton first (~20 min)

Settled below: sw-qsor is built as an installable library from the start. So before any data,
create `pyproject.toml` and an empty `sw_qsor/` package, and put the test directory in place.
Everything after this goes inside the package, with demos as callers.

## 1. Fetch and cache the data (~20 min)

`scripts/fetch_gslf.py` and `scripts/fetch_pyrfume.py`, following sw-enose's convention: data
goes into a gitignored `data/`, never vendored, with the retrieval date and licence recorded in
the loader's docstring.

- GS+LF: `curated_GS_LF_merged_4983.csv` from `BioMachineLearning/openpom` (MIT).
- Pyrfume: `pip install pyrfume`, or raw files from `pyrfume/pyrfume-data`. Start with
  `arctander_1960`, `mayhew_2022`, `keller_2016`, `dravnieks_1985`, `abraham_2012`.
- **Check for a `LICENSE` file beside each archive before using it.** Arctander is CC
  BY-NonCommercial; most others state nothing, which is "not stated", not MIT.

## Progress, 17 Nov 2026

Steps 0 and 1 done: `sw_qsor/` package (`fetch`, `sources`, `datasets`), `scripts/fetch_gslf.py`,
`scripts/fetch_pyrfume.py`, 10 offline tests. Sources pinned to openpom `5434c4b` and
pyrfume-data `8054ea9`. Licences observed on fetch: Arctander CC BY-NC 4.0 (own LICENSE); the
other four Pyrfume archives and GS+LF "not stated". Molecule counts recounted from the fetched
files match the survey, with one exception that bears on step 3:

- **Keller 2016's 481 is rows, not molecules.** 480 distinct CIDs (cinnamyl alcohol appears
  twice), 2 rows with no SMILES (CIDs 8038, 1549778), and its SMILES are `CanonicalSMILES`
  with no stereochemistry, so geraniol/nerol, geranyl/neryl acetate, D-/DL-camphor and
  L-/DL-phenylalanine collapse: 474 distinct structures.
- GS+LF ships non-stereo SMILES only. **The overlap in step 3 must therefore match on the
  InChIKey connectivity block (first 14 characters), not the full key**, or stereo-labelled
  archives (Arctander, Mayhew, ...) will appear falsely disjoint from it. Report both, and say
  how many molecules each dataset loses to stereo collapse.

Step 2 done: rule 3 amended to similarity-component groups at Tanimoto 0.70, with results
binned by nearest-training-neighbour similarity. Candidates A–C were all rejected, and the
plan's 10% size test turned out not to tell them apart; reasons in `reports/02_grouping.md`.
`sw_qsor/splits.py` implements it; `demo_grouping.py` reproduces the comparison.
**For step 4: the floor must be scored under these folds *and* these bins.**

Step 3 done (`reports/03_overlap.md`, `sw_qsor/identity.py`, first section of
`demo_dataset_survey.py`). **Only Arctander gives a sizeable descriptor test set independent of
GS+LF under rule 3: 901 molecules, 838 groups.** Keller (98% inside GS+LF) and Dravnieks
(92%) are re-measurements of GS+LF molecules, not test sets. Mayhew's 1,109 independent
molecules answer a different question (odorous or not). Abraham is 95% inside Mayhew and is one
of its label sources. Two open checks before Arctander can be used as a test set: a fixed
vocabulary mapping to the GS+LF labels, and whether GoodScents labels derive from Arctander's text
(test the agreement on the 1,477 shared molecules). Step 3 also found that GS+LF's duplicate records could be split
across folds; fixed in `splits.py` (identity keys), and report 02 re-run.
**For step 4: GS+LF has 4,981 molecules, not 4,983; merge the two duplicate pairs' labels
before scoring the floor.**

Step 4 done: floors in `sw_qsor/floors.py` + `sw_qsor/evaluate.py`, second half of
`demo_dataset_survey.py`, results appended to `plan.md`, per-label table in
`reports/04_floor_per_label.csv`. **Molecular weight alone: macro AUC 0.664, above chance on
125 of 138 labels; base rate scores 0.775 on pooled micro AUC.** Milestone 1 clears the decision
rule. Step 4 also found that pooling out-of-fold scores within distance bins biased the base
rate to 0.43; fixed by within-fold ranks (`evaluate.within_fold_ranks`).

Step 5 done: `plan-milestone-2.md` fixes the label (`floral`), the model (one random forest
on Morgan bits), two floors (MW k-NN and the same forest on MW), a neighbour-lookup reference,
a shuffled-label pipeline check, and the bar (mean paired margin ≥ 0.05 over 5/5 folds, plus a
group-bootstrap interval above 0 in the < 0.5 similarity bin). **Milestone 1 is complete. The
next session runs milestone 2 exactly as written, with no changes to that file before the
run.**

## 2. Amend rule 3 before any split is written (~45 min)

The survey found 2,236 of 4,983 GS+LF molecules (45%) are acyclic and collapse into one empty
Murcko scaffold, so a Murcko split would hold out nearly half the data at once. Decide the
grouping now, in writing, before seeing any accuracy:

- Candidate A: Murcko scaffold, with acyclics subdivided by Butina clustering on Morgan
  fingerprints at a fixed Tanimoto cutoff.
- Candidate B: cluster **everything** by fingerprint similarity and ignore Murcko entirely.
- Candidate C: group by carbon skeleton after removing functional groups.

Pick one, say why, and record the group-size distribution it produces. The test of a good
grouping is that no group holds more than about 10% of the molecules. Write the choice into
`readme.md` rule 3 so it is fixed for everything that follows.

## 3. Overlap between datasets, by InChIKey (~30 min)

Compute, not estimate: GS+LF against Arctander (2,751), Mayhew (1,865), Keller (481),
Dravnieks (138). Report a matrix of shared molecules.

**This decides what can ever serve as an independent test set.** Two datasets sharing most of
their molecules are one dataset, and using them as a train/test pair would be self-deception.
If Arctander turns out to be largely inside GS+LF, it is not a validation set.

## 4. The chemistry-free floor (~1 hour)

The last piece of milestone 1, and the one everything afterwards is measured against. For each
prediction task the data supports, measure what can be predicted with **no structural
information at all**:

- Per-label frequency (predict the base rate of "fruity" for every molecule).
- Per-molecule label count, if that is knowable without structure.
- Nearest neighbour on a non-structural feature such as molecular weight alone.

Scored under the grouping chosen in step 2, held out by group. Report ROC AUC and average
precision per label, and the distribution across the 138 labels. Expect it to vary widely:
common descriptors will have a high floor, rare ones a low one.

Deliverable: `demo_dataset_survey.py`, printing the counts from step 3 and the floors from
step 4, and a results section appended to `plan.md` whether or not it is encouraging.

## 5. Decide on milestone 2 (~15 min)

With the floor known, write `plan-milestone-2.md`: one well-populated label, predicted from
structure, group-held-out, with the bar fixed before the run. Do not start modelling in the
same session that measures the floor — the gap between them is what keeps the bar honest.

## Not in this session

- `01_datasets_full.md` (benchmarks, splits, baselines in published work). Worth running now
  that there is a short list, but it is a research task, not a coding one.
- Any model. Milestone 2 only begins when milestone 1 is written up.

## A practical note on running the research

`reports/01_dataset_survey.md` was produced in-session with ordinary web search and file
retrieval rather than by a deep-research tool, and that worked better than expected: every
molecule count in it was obtained by downloading the file and counting, which no research tool
would have done. Prefer that route for anything where the answer is a number in a file.

Two literature searches in the sibling project were stopped by a safety classifier, for reasons
that were never visible and did not obviously match the material. Nothing in this project's
subject matter resembles what was blocked, and today's survey ran cleanly, so this is recorded
as a known quirk rather than an expected obstacle. If a prompt does get stopped, the fallback
is the same as sw-enose's: run it externally and audit the result on return.

## Packaging and code sharing, settled

Three questions that look like one.

**Is sw-qsor a library?** Yes, and build for that from the first commit, because it costs
nothing now and is awkward to retrofit. The end state is `pip install sw-qsor` then
`predict_odour(smiles)`. Practically, from step 1:

- a real `sw_qsor/` package, installable, with a `pyproject.toml`;
- no imports from the repository root, no paths relative to the current directory, no reading
  of files the caller did not name;
- demos are callers of the package, never the place where logic lives;
- anything a demo needs twice belongs in the package.

**Does sw-enose depend on sw-qsor?** No. Dependencies point from new to old. sw-enose has 166
tests and a body of settled results; sw-qsor has no code and may yet find its data too thin.
Coupling the proven project to the unproven one buys nothing and risks the stable thing.

**Where does shared code live?** Not in a third library, not yet. The genuinely reusable asset
is not the cheminformatics — descriptors and scaffolds are thin wrappers over RDKit — but the
**evaluation discipline**: chemistry-free floors, group-aware splits, bars fixed before the
run, paired margins over matched controls. That is not specific to olfaction and is not, as far
as we know, available as a library. It is worth extracting eventually.

But extract after writing it twice, not once. sw-enose has one implementation, spread across
`demo_m2or_baseline.py` and the wind-tunnel demos; sw-qsor has none. Designing an API from a
single example usually produces the wrong abstraction.

**So: copy, but write the copy as though it were already that library.** Clean signatures, no
assumption about which project is calling, no global state, no project-specific paths. Note the
origin in each docstring. After milestone 1 there will be two real implementations and the
right shape will be evident; extraction then is mechanical rather than speculative.
