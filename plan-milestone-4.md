# Plan: milestone 4, does the published odour model beat neighbour lookup on unfamiliar molecules?

**Fixed 17 Nov 2026, before the run.** Changes after this commit are a new plan, not edits to
this one. Nothing below has been run on real labels. The facts it
relies on come from OpenPOM's own files at the commit already pinned for GS+LF (`5434c4b`):
the README, `examples/ensemble_cv.ipynb` with its saved outputs,
`examples/trials/1_trials_params.json`, `openpom/feat/graph_featurizer.py` and
`openpom/utils/loss.py`. They also draw on this project's committed results.

## Why this question

Milestone 3 found that a random forest on Morgan bits beats molecular weight on 71 of 112
labels. But on molecules below 0.5 similarity to training it does no better than averaging the
labels of the ten most similar training molecules (Q2: +0.006 [−0.008, +0.020]). That is one
model. The published state of the art for this dataset is OpenPOM's message-passing network,
the "principal odour map". It reports **0.8872** mean AUC by 5-fold cross-validation on the
same GS+LF file. That split stratifies labels but not structure, so under it near-analogues
cross folds freely.

Milestone 4 asks two things:
- How much of that published number survives rule 3?
- Does the published model beat neighbour lookup on unfamiliar molecules, where the forest
  could not?

## Questions

- **Q1 (primary): beyond similarity, for the published model.** Pooled over labels with ≥ 10
  positives below 0.5 similarity, does the POM ensemble beat the best lookup reference there?
- **Q2 (primary): the published number under rule 3.** The POM ensemble's macro-mean AUC over
  all 138 labels on rule-3 folds, against the published 0.8872 on stratified folds. This is
  descriptive: it states the gap, and no threshold decides anything.
- **S1 (secondary): persistence.** The POM ensemble's pooled margin over the floor below 0.5,
  as in milestone 3's Q1.
- **S2 (secondary): the sw-enose note's question,** with the corrections recorded after it.
  Within the distant bins, does the margin over lookup fall as labels gain positives? Tested
  per label category (below).

## Models

### The POM ensemble, reimplemented

OpenPOM's code depends on DeepChem and DGL, which are not expected to support Python 3.14. The
architecture is reimplemented in plain PyTorch, on the CPU, from the settings below. Every
setting is OpenPOM's own; none is tuned here.

- **Featuriser** (`graph_featurizer.py`): atom one-hots for total valence (0–6), degree (0–5),
  number of hydrogens (0–4), formal charge (−1, −2, 1, 2, 0) and atomic number (0–99), each with
  an unknown slot, plus hybridisation (SP to SP3D2, plus unknown). That gives 134 atom features.
  There are 6 bond features (bond type and ring membership, as in the file). Input is non-stereo
  SMILES, as published.
- **Network** (`trials/1_trials_params.json`, `ensemble_cv.ipynb`):
  - message passing: an edge network (NNConv) with a GRU, 5 steps, node width 100, edge hidden
    width 75, edge out width 100, residual, sum aggregation, no self-loops;
  - readout: Set2Set, 3 steps, 2 layers;
  - head: feed-forward layers [392, 392], embedding width 256, ReLU, dropout 0.12;
  - one sigmoid output per label, all 138 labels trained jointly.
- **Loss** (`utils/loss.py`, through DeepChem's `_StandardLoss`). Ported as the code runs, not
  as its docstrings describe:
  - The raw logit l is fed into a two-class softmax over [1 − l, l], which equals binary
    cross-entropy on 2l − 1.
  - Each label is weighted by log(1 + count / max count) on the training fold. So **common
    labels weigh more**, the opposite of what the docstring says.
  - The loss is summed over labels and averaged over the batch.
  - The model's regularisation term, 1e-5 × (L1 + L2 norms of the non-bias parameters), is
    added to every batch.
- **Training** (`ensemble_cv.ipynb`): Adam (betas 0.9 / 0.999, eps 1e-8, no built-in weight
  decay), initial learning rate 0.001, halved every 640 steps (staircase, stepped per batch),
  batch 128, **62 epochs**, no early stopping and no validation set. Each epoch is shuffled;
  the last batch is padded to full size by cycling its own samples, as DeepChem does, which
  makes 32 batches per epoch, so 640 steps = 20 epochs.
- **Ensemble:** 10 models per fold with seeds 0–9, predictions averaged.
- **Determinism:** CPU only, `torch.use_deterministic_algorithms(True)`, a fixed thread
  count, and seeds set per model.

### References and floor (from milestone 3, unchanged)

- **Floor:** the higher of F-MW and RF-MW, per label and per fold or resample.
  *(Superseded for S1 by the Amendment below: each floor is tested separately.)*
- **Lookup-10:** kNN-Tanimoto, k = 10, as in milestone 3.
- **Lookup-W (new, fixed here):** similarity-weighted lookup over the 50 most similar training
  molecules, score = Σ simᵢ·yᵢ / Σ simᵢ. It is added because k = 10 ties heavily on rare labels:
  44% of positives in the rarest quarter of labels score exactly 0. That handicap flatters
  any model compared against it. k = 50 is chosen here, not tuned.
- **The lookup bar is the higher of Lookup-10 and Lookup-W in each resample**, the same
  convention as the floor.
  *(Superseded by the Amendment below: Q1 tests each lookup separately; S2 uses Lookup-W.)*
- **RF-Morgan** (milestone 3) is reported alongside for comparison, but decides nothing.

## Data and split

GS+LF through `gslf_experiment`, the same 5 rule-3 folds as milestones 2 and 3. Scored on the
112 labels with ≥ 50 positives, and Q2 also on all 138. Q1 uses the labels with ≥ 10
positives below 0.5 similarity (91 in milestone 3).

## Checks, run before anything is read

1. **Reimplementation (C1).** Under OpenPOM's own protocol, the reimplemented single model must
   reproduce the published single-model scores. The protocol is the iterative stratified 5-fold
   split (scikit-multilearn, order 2) with one model per fold. The notebook's 50 single models
   score 0.862–0.880 test AUC, mean about 0.870. **Pass if the 5-fold mean is ≥ 0.855.** If it
   fails, stop: the model being tested would not be the published one.
2. **Pipeline (C2).** One model per rule-3 fold, trained with the label matrix's rows permuted
   (seed 0). Must give macro AUC 0.48–0.52.
3. **Continuity (C3).** Floral's RF-Morgan and Lookup-10 per-fold AUCs must reproduce milestone 2
   to three decimals.

## The bar

**Q1:** pooled macro-mean margin of the POM ensemble over the lookup bar, on molecules below 0.5
similarity. Scores are ranked within folds; 2,000 joint group bootstraps, seed 0.
- **Beyond similarity:** the interval lies entirely above 0.
- **No evidence beyond similarity:** the interval includes 0. Its width is reported.
- **Worse than lookup:** the interval lies entirely below 0.

**Q2:** reported as the POM ensemble's macro AUC on rule-3 folds (138 labels and 112), next to
0.8872. The four bins are reported too.

**S1:** as milestone 3's pooled Q1, for the POM ensemble.

**S2:** for each label with ≥ 10 distant positives, the POM ensemble's distant-bin margin over
the lookup bar. Spearman correlation of that margin with the label's total positives, with a
95% group-bootstrap interval (the same resamples as Q1). It is reported overall and within each
category below. The sw-enose note predicts a negative correlation. A negative correlation that
appears only among quality words would point to vagueness rather than data quantity.

Label categories, fixed here by name only, never by results:
- **Chemical** (named after a compound class or a specific compound): alcoholic, aldehydic,
  camphoreous, coumarinic, ethereal, ketonic, lactonic, phenolic, sulfurous, terpenic, anisic,
  vanilla, cinnamon, clove.
- **Quality** (describes an overall character, not a source): aromatic, bitter, clean, cooling,
  dry, fresh, juicy, natural, powdery, pungent, ripe, sharp, sour, sweet, warm.
- **Source:** every other label, named after a material, food, plant or object (fruity, floral,
  rose, woody, and so on).

## D1 (descriptive, decides nothing): do these results apply to sw-enose's molecules?

Milestone 4's distance bins say how far structure-based prediction reaches. D1 places
sw-enose's molecules on that scale, so it is clear which bins apply to them. It uses structure
only, no labels, and it is computed before Q1 is read.

- **Molecules.**
  - The 180 M2OR odorants, from `sw-enose/data/m2or/odorant_structures.csv`.
  - The three breath gases in `sw-enose/data/breath_vsc/thresholds_ppb.csv`: hydrogen sulfide
    `S`, methyl mercaptan `CS`, dimethyl sulfide `CSC`.
  - Both are read from a sw-enose checkout whose path the caller gives (`--sw-enose-dir`). D1
    is skipped, and says so, if the path is not given. sw-qsor does not import or depend on
    sw-enose.
- **For each molecule:**
  - whether it is in GS+LF by connectivity key (and if so, its GS+LF descriptors);
  - otherwise its Tanimoto similarity to the nearest GS+LF molecule;
  - the milestone 4 bin it would fall in.
- **Summary for the 180:** the share inside GS+LF and the distribution across the four bins.
- **Degenerate molecules, flagged, not interpreted.** A Morgan fingerprint with ≤ 5 bits set
  makes Tanimoto similarity meaningless. Hydrogen sulfide has 1 bit; methyl mercaptan 3;
  dimethyl sulfide 4. These are listed as "too small for fingerprint similarity", not put in a
  bin. Already known from a structure-only check while drafting: methyl mercaptan and dimethyl
  sulfide are **in** GS+LF (labelled sulfurous, cabbage, garlic, onion and others); hydrogen
  sulfide is not.
- **Already measured in sw-enose** (`note-2-from-sw-enose.md`): 163 of the 180 M2OR molecules
  (91%) are inside GS+LF. D1 recomputes this rather than citing it.
- **Output:** `reports/06_sw_enose_coverage.csv`, and a short paragraph in the results.

### D1b: tiny molecules are a limit of fingerprints, reported as a finding

Raised by sw-enose as worth stating in its own right. Across GS+LF and sw-enose's molecules,
count the molecules whose Morgan fingerprint has ≤ 5 bits set, and list them. For those in
GS+LF, report:
- the rule-3 groups they fall in;
- how many have a cross-fold nearest neighbour at similarity ≥ 0.5 purely because both
  fingerprints are nearly empty.

**Scope of the claim, corrected while porting the model (17 Nov 2026).** The first draft said
a graph network still represents a single atom. That is true of graph networks in general, but
**not of OpenPOM's**. Its readout sums messages arriving over bonds, so a molecule with one
heavy atom gets an all-zero readout, and every such molecule gets the same prediction. This
was confirmed with OpenPOM's own released weights (`examples/example_model.pt`) loaded into the
port: hydrogen sulfide (`S`) and water (`O`) receive identical predictions, with "odorless"
top at 0.92. So:
- **Below about five bits**, fingerprint methods cannot tell molecules apart.
- **At one heavy atom**, the published graph network cannot either.
- **Molecular weight is the only model in this plan that distinguishes them.**

D1b therefore also reports how many GS+LF molecules have a single heavy atom, and which they
are.

## Expectation, stated before the run

Milestone 3's forest showed no advantage over lookup below 0.5, so the expected Q1 outcome is
*no evidence beyond similarity*, perhaps with a small positive point estimate. For Q2, the
expected rule-3 macro AUC is clearly below 0.8872, because the published split lets analogues
cross. How far below is the point of measuring it.

## Cost (measured 17 Nov 2026, before commit)

10 models × 5 folds × 62 epochs on rule-3 folds, plus 5 single models for C1 and 5 for C2: **60
trainings**. Measured on rule-3 fold 0 with **row-permuted labels**, so that no result was seen:
**8.6 s per epoch at 12 threads**, against 17 s at 24, where hyperthreads compete for the same
cores. That is about **9 minutes per model and about 9 hours in all**, run sequentially. This is
under the 24-hour limit, so **the ensemble stays at 10**. The thread count is fixed at 12 for
every training, because results are deterministic only for a fixed count.

## Port verified before commit

- `sw_qsor/pom.py` loads OpenPOM's released checkpoint (`examples/example_model.pt`, SHA-256
  matching its LFS pointer) with **all 44 parameter tensors matching in name and shape**.
- With those weights, its predictions are recognisable without consulting any labels:
  vanillin → sweet, vanilla; ethyl butyrate → fruity; dimethyl sulfide → vegetable,
  sulfurous, onion; geraniol → floral, rose.
- Unit tests cover feature dimensions 134 / 6, known atom features, batch invariance, the loss
  identity (BCE on 2l − 1), label weights, regularisation, determinism under a fixed seed, and
  Lookup-W against brute force.

## Looking ahead: milestone 5 is for sw-enose

Decided 17 Nov 2026: **milestone 5 focuses on what sw-qsor can deliver to sw-enose.** Milestone
4 is kept to sw-qsor's own question, but it is built so that its outputs can feed milestone 5
without re-running:
- D1 says which of sw-enose's molecules lie within reach of structure-based prediction.
- After the milestone 4 run, and only after it, one POM model trained on **all** of GS+LF with
  the same settings can supply a 256-dimensional embedding per molecule. Producing it, and
  testing it in sw-enose, is milestone 5's to plan; nothing here commits to it.
- **Scope, stated now** (from `note-2-from-sw-enose.md`): sw-enose's constraint is labels, not
  features. 719 of 1,020 receptors have no known agonist, and no representation helps them.
  Only **59 receptors have ≥ 5 positives**. So milestone 5's honest question is: *does a learned
  embedding beat sw-enose's 20 descriptors on those 59 receptors?* That is a clean test of
  representation quality. It cannot supply sw-enose's missing receptor data.
- `sw_qsor/pom.py` and Lookup-W are written as project-independent code, so sw-enose can take
  them as it took the rule-3 grouping.

## Not in milestone 4

- The GPU (a later speed-up, checked against CPU results as in sw-enose's `gpu-to-do.md`).
- Any tuning of the POM settings, other architectures, pretrained weights, or OpenPOM's
  released checkpoints. Those were trained on all of GS+LF, so on any GS+LF test fold they would
  leak by construction.
- Arctander. A Phase-C-style external test could follow, but only under a plan of its own.
- **The compilation check** raised in `note-2-from-sw-enose.md`, section 2: whether one source
  re-reports another's values exactly, which molecule overlap cannot detect. It is about data
  provenance, not models, so it is kept separate (see `next-session.md`).

## Deliverables

- `sw_qsor/pom.py` (featuriser, network, loss, training loop) and `sw_qsor/models.py` additions
  (Lookup-W), with tests: feature dimensions 134 / 6; Lookup-W against brute force; determinism,
  meaning two runs with the same seed give identical weights.
- `pyproject.toml` gains an optional `pom` extra (`torch`, CPU wheel; `scikit-multilearn` for C1).
- `demo_milestone4.py`: the checks first, then every measurement and verdict. It caches each
  trained model's out-of-fold scores so that a crash resumes without retraining.
- Per-label results in `reports/06_m4_per_label.csv`, D1 in `reports/06_sw_enose_coverage.csv`,
  and a results section appended here whatever the outcome.

## Amendment, 17 Nov 2026, before the run (no real result seen)

**Why.** `demo_milestone4.py --smoke` runs the whole pipeline with every label row-permuted,
so there is no signal, and every margin should then sit at zero. The committed bars did not:

| Pooled margin below 0.5, permuted labels | Estimate [95% interval] |
|---|---|
| Q1: POM over the per-resample **max** of Lookup-10 and Lookup-W | −0.035 [−0.058, −0.025] |
| S1: POM over the per-resample **max** of F-MW and RF-MW | −0.044 [−0.075, −0.041] |

The higher of two noisy AUCs is biased upward, so a margin over it is biased downward. That
is about 0.03–0.04 here, against effects of the size milestone 3 measured (Q2: +0.006). Under
the committed rule, Q1 would have read "worse than lookup" on data with no signal at all.
Milestone 3's floor comparisons used the same max, so they were conservative. Its Q2 lookup
comparison had a single comparator and was unbiased.

**Changes**, each a single comparator, so each margin is unbiased when there is no signal:
- **Q1:** pooled margins of the POM ensemble over Lookup-10 and over Lookup-W, each with its
  own 2,000 joint group bootstraps (seed 0, same groups):
  - *Beyond similarity:* both intervals entirely above 0.
  - *Worse than lookup:* both entirely below 0.
  - *No evidence beyond similarity:* otherwise.
- **S1:** margins over F-MW and over RF-MW separately. *Persists* only if both intervals are
  entirely above 0.
- **S2:** the per-label distant-bin margin is over **Lookup-W** alone, the tie-free lookup S2
  exists to use. The margin over Lookup-10 is reported alongside and decides nothing.
- **The reference row** (RF-Morgan over lookup) and **the per-label table** report margins over
  each comparator separately.

Nothing else changes. The smoke run's numbers are meaningless as results and are recorded only
as the reason for this amendment. Its output went to `data/m4_smoke/`, never `reports/`.

## Amendment 2, 17 Nov 2026, before the run (no real result seen): reporting only

Prompted by `note-3-from-sw-enose.md`. These additions decide nothing, and no bar changes.
- **Single models alongside the ensemble.** Q1 compares a 10-model averaged ensemble, and
  averaging alone raises AUC. Every member's predictions are cached, so for each of the 10
  seeds, the single-model pooled distant margin over Lookup-10 and over Lookup-W (point
  estimates) and its macro AUC are reported. Mean, minimum and maximum are given next to the
  ensemble. This separates the architecture from ensembling.
- **D1b is extended from single heavy atoms to all bondless entries.** Checked before the run
  with OpenPOM's released weights:
  - Bit-identical embeddings for `S` and `O`, `[Cl-]` and `[Na+]`, the salt `[Cl-].[Na+]` and
    `O`, and `S` and `[He]`.
  - Different embeddings for `CS` and `CO`.
  - Atom features do reach the node states (`S` and `O` differ after message passing). But the
    readout sums only messages arriving from neighbours, and DGL zero-fills nodes that receive
    none.

  So the collapse covers every molecule in which no heavy atom has a bond, salts of
  single-atom ions included. That is a property of OpenPOM's architecture, not of the port.
  GS+LF has 11 such entries: 10 labelled `odorless` and one, ammonium sulfide, labelled
  `sulfurous`. D1b lists them with their labels.

## Decided at review (17 Nov 2026): all five as drafted

1. **Q2 is co-primary.** It is descriptive and decides nothing.
2. **The C1 threshold is 0.855.**
3. **Lookup-W uses k = 50**, weighted by similarity.
4. **The S2 category lists stand as above.**
5. **Ensemble size 10,** to be reduced only if the cost measurement exceeds about 24 hours.
