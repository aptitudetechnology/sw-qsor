# Plan: milestone 1, what is actually in the public data

Written 17 Nov 2026, before any dataset has been downloaded. The point of writing it now is that
the counting rules below cannot then be relaxed once a number is disappointing.

## Why this comes first

The binding constraint on structure–odour prediction is molecular diversity, not row count. In
sw-enose a dataset advertising 24,735 measurements turned out to carry 180 distinct molecules in
37 scaffold groups, 92 of them sharing a single acyclic group — and the modelling failed for
exactly that reason. Had the counting been done first, the result would have been predictable
and the work unnecessary.

So before any model: establish what is in each candidate dataset, how much of it is independent,
and what can be predicted from it without using structure at all.

## Candidate datasets

Named from memory and **all UNVERIFIED** until fetched; the first job is to confirm each exists,
is public, and says what it is believed to say.

| Dataset | Believed to contain | Status |
|---|---|---|
| DREAM Olfaction Prediction Challenge (Keller & Vosshall) | a few hundred molecules rated by many subjects on perceptual scales, with per-subject data | UNVERIFIED |
| GoodScents / Leffingwell flavour and fragrance descriptors | low thousands of molecules with descriptor labels ("floral", "woody"); believed to be the basis of recent deep-learning work | UNVERIFIED |
| Dravnieks, *Atlas of Odor Character Profiles* | ~150 molecules profiled on ~146 descriptors by large panels | UNVERIFIED |
| Nagata triangle-odour-bag thresholds | ~220 compounds with measured detection thresholds in ppm; already read and partly transcribed in sw-enose | partly verified |
| Pyrfume | a collection that aggregates many of the above | UNVERIFIED |

Licences must be checked before anything is committed. sw-enose's convention applies: data is
fetched by script into a gitignored directory, never vendored, unless it is a small table that
is transcribed with its provenance.

## What to measure, for each dataset

Fixed now, and reported whether or not it flatters the dataset:

1. **Distinct molecules** with both a structure and at least one label. Not rows, not
   molecule–label pairs, not ratings.
2. **Distinct Bemis-Murcko scaffolds**, and the size of the largest group. Acyclic molecules
   collapse into one group, which can be most of a fragrance set.
3. **Label structure**: how many labels, how often each occurs, and how imbalanced. A label
   carried by three molecules cannot be learned, however many ratings it has.
4. **Overlap between datasets**, by InChIKey. Two sources sharing most of their molecules are
   one dataset for splitting purposes, and treating them as a train/test pair would be
   self-deception.
5. **The chemistry-free floor** for each prediction task the dataset supports: what accuracy
   comes from label frequency alone, under a scaffold-held-out split.

## Decision rule for proceeding

A dataset is worth modelling only if, after the counting:

- it has **at least 1,000 distinct molecules** across **at least 100 scaffold groups**, and
- at least one label is carried by at least 50 molecules, and
- the chemistry-free floor leaves visible room, judged against the fold-to-fold spread rather
  than against zero.

If no dataset clears this, the finding is that public structure–odour data is too thin for the
task, which is reported as the milestone's result. That is a real outcome and the project says
so rather than modelling something too small and calling it a baseline.

## Before the counting: a literature survey

Two prompts, run in that order:

- `research_prompts/01_datasets_mvp.md` asks one question only -- how many **distinct
  molecules** each dataset holds, and whether it can be downloaded. That is what the decision
  rule above turns on, and it is short enough for a research tool to answer properly.
- `research_prompts/01_datasets_full.md` adds the rest: published benchmarks, how each split
  its data, whether any non-structural baseline was reported, and code availability. Worth
  running only on the datasets the MVP shows are large enough to care about.

Findings from either are leads to verify, not facts, and are audited the way sw-enose audits
such reports.

## Deliverables

- `scripts/fetch_*.py` per dataset, with licence and retrieval date recorded.
- `sw_qsor/datasets.py`: loaders that return molecules, structures, labels and provenance.
- `demo_dataset_survey.py`: prints the five counts above for every dataset, plus the
  chemistry-free floors.
- A results section appended to this file, written whether the answer is encouraging or not.

## What milestone 2 would be, if milestone 1 clears

Predict one well-populated perceptual label from structure, scaffold-held-out, against its
chemistry-free floor, with the bar written down first. Not before.

## Expected outcome

Stated in advance so the result reads honestly either way: the Nagata thresholds and Dravnieks
are almost certainly too small on their own. The fragrance descriptor sets are the only
candidates likely to clear 1,000 molecules, and their labels are expected to be severely
imbalanced, with a long tail of descriptors used a handful of times. The most likely useful
outcome is that a small number of common descriptors are learnable and the rest are not — and
knowing which is which is worth having.
