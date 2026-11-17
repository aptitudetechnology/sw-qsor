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
