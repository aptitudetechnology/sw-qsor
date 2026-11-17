# Next session: finish milestone 1

Written 17 Nov 2026. Milestone 1's literature half is done (`reports/01_dataset_survey.md`);
what is left is local and needs no external research. Ordered by dependency, not importance.

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

## Open question worth settling early

Does sw-qsor reuse sw-enose code, or stay independent? `sw_enose/chem.py` (descriptors,
scaffolds) and the pattern in `demo_m2or_baseline.py` (floors, pre-registered splits) are
directly applicable. Copying keeps the projects independent; a shared package couples their
release cycles. Recommendation: copy, and note the origin in the docstring. These are small
files and independence is worth more than reuse here.
