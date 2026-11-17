# Plan: milestone 3, all well-populated labels, and whether anything beats similarity

**Fixed 17 Nov 2026, before anything below has been run.** Changes after this commit are a new
plan with a new date, not edits to this one. The facts it
relies on are label counts and vocabulary, not model results: positives per label and per
distance bin, Arctander's column names, and the layout of OpenPOM's separate GoodScents and
Leffingwell files.

## What milestone 2 left open

`floral` beat the better molecular-weight floor by 0.097 AUC, including on molecules far from
training. But below 0.5 similarity the forest was no better than averaging the labels of the
ten most similar training molecules (+0.004 [−0.025, +0.034]). Milestone 3 asks whether that
holds across labels, and whether it holds on a different dataset.

## Questions

- **Q1, breadth.** For which of the 112 labels with ≥ 50 positives does structure beat the
  floor? Across labels, does the margin persist on molecules far from training?
- **Q2, beyond similarity (co-primary).** Pooled across labels, does RF-Morgan beat
  kNN-Tanimoto on molecules with no training neighbour at 0.5 or above? One label could not
  answer this: floral had 172 distant positives. Across labels there are 4,079.
- **Q3, external.** Trained on GS+LF, how do the models do on Arctander molecules independent of
  GS+LF? Q3 runs only after Phase A has settled what such a test would mean.

## Order, one commit per phase

**A (labels only, no models) → B (GS+LF run) → C (Arctander run).** Each phase's results are
committed before the next phase starts. Phase A uses no structural model and cannot be tuned
to one.

---

## Phase A: Arctander vocabulary and label provenance

### A1. Vocabulary mapping, fixed here by name alone

The mapping is fixed by label *name*, not by how labels co-occur on shared molecules. Fitting
it to the data would make the provenance check circular.

- **Exact names:** the 59 Arctander labels that appear verbatim among GS+LF's 138 labels.
- **Spelling variants** (8): `ambre`→`amber`, `camphoraceous`→`camphoreous`,
  `caramelic`→`caramellic`, `leather`→`leathery`, `minty`→`mint`, `musky`→`musk`,
  `sulfuraceous`→`sulfurous`, `vanillin`→`vanilla`.
- **Unmapped** (10): `acid`, `coco` (ambiguous between cocoa and coconut), `lilac`, `mimosa`,
  `mossy`, `narcissus`, `orange-blossom`, `pepper`, `rooty`, `tarry`. No semantic mappings
  are made, such as `lilac`→`floral`.

That gives 64 mapped labels among the 112. `orange-blossom`→`orangeflower` is excluded
because orangeflower has 33 GS+LF positives, below the 50 needed to be among the 112.

Arctander molecules: `behavior_1.csv` is indexed by stimulus (3,102), mapped to CIDs through
`stimuli.csv` and to structures through `molecules.csv` (2,751). Stimuli without a CID are
dropped. Stimuli sharing a connectivity key have their labels unioned, as was done for GS+LF.
All of these counts are reported.

### A2. Is GoodScents labelling derived from Arctander?

GoodScents entries often quote Arctander, so labels on disjoint molecules may still not be
independent. The reference point is Leffingwell, the other source in GS+LF.

- Fetch OpenPOM `curated_goodcents.csv` and `curated_leffingwell.csv` from the pinned commit
  `5434c4b`. They use the same 138-label vocabulary as the merged file.
- Molecules present in GoodScents, Leffingwell and Arctander, by connectivity key: count and
  report.
- For each mapped label with ≥ 20 positives in Arctander on those molecules: Cohen's κ
  between GoodScents and Arctander, and between Leffingwell and Arctander.
- Statistic: the macro-mean of κ(GS, Arc) − κ(LF, Arc), with a 95% interval from 2,000 group
  bootstraps (rule-3 groups, seed 0).

**Decision, fixed now:**
- If the mean difference is ≥ 0.10 and its interval excludes 0, **Phase C is reported as a
  test of cross-source consistency, not as independent validation.** Phase C runs either way;
  only its framing changes.
- Otherwise Phase C is reported as an external test, carrying the caveat below.

**Limit of this check:** it detects derivation specific to GoodScents. If Leffingwell also
drew on Arctander, both κs rise together and the check misses it. Absolute κ values are
reported so that a reader can judge.

### A3. Phase C's label set, counted in Phase A

For each of the 64 mapped labels: positives among Arctander's 901 independent molecules (see
`reports/03_overlap.md`). **Phase C scores the labels with ≥ 20 positives there.** The list is
committed with Phase A.

---

## Phase B: GS+LF, all 112 labels

The protocol is milestone 2's, per label, with nothing changed:

- **Data and folds:** `gslf_experiment`, the same folds.
- **Models:** RF-Morgan, RF-MW, F-MW (k = 50) and kNN-Tanimoto (k = 10), plus RF-Morgan on
  shuffled labels, all exactly as in `sw_qsor/models.py` and `sw_qsor/floors.py`.
- **Separate per label:** one forest per label, not a multi-output forest, so that floral's
  numbers reproduce milestone 2's.
- **Floor:** per label per fold, the higher of F-MW and RF-MW.
- **Cost:** about 1,700 forest fits, an estimated 70–90 minutes. It is run once.

### Checks, before anything is read

1. **Pipeline:** macro-mean shuffled-label AUC across the 112 labels must lie in 0.48–0.52.
   If not, stop and find the leak.
2. **Reproduction:** floral's per-fold AUCs for all four models must equal milestone 2's to
   three decimal places. If not, stop and find out why.

### Q1, per label

Milestone 2's criteria, minus the distant-bin condition, which moves to the pooled test:

- **Pass:** mean paired margin ≥ 0.05, and the margin is > 0 in all 5 folds.
- **Fail:** mean paired margin < 0.02, or the margin is ≤ 0 in 2 or more folds.
- **Inconclusive:** anything else.

**False passes.** With no true margin, all five folds come out positive about 1 time in 32.
Adding the 0.05 mean margin makes a pass rarer still, so chance passes among 112 labels should
number about 3 at most. Results are reported by frequency band, since fold SD runs from about
0.10 for rare labels to 0.02 for common ones.

**How Q1 is read, stated in advance:**
- *Broad:* ≥ 56 labels (half) pass.
- *Narrow:* 10–55 pass.
- *Absent:* fewer than 10 pass.

`plan.md` predicted *narrow*: a small number of common descriptors learnable, the rest not.

### Q1, pooled distant test

Over the **91 labels with ≥ 10 positives in the < 0.5 bin**: the macro-mean margin of RF-Morgan
over the floor. For each label, the floor is the higher of F-MW and RF-MW in each resample.
Scores are ranked within folds. The 95% interval comes from 2,000 **joint** group bootstraps:
one resample of groups is applied to every label at once, seed 0.

- **Structure signal persists far from training:** the interval lies entirely above 0.
- **Otherwise:** it is not shown to persist.

### Q2, beyond similarity

Same 91 labels, same joint bootstrap: the macro-mean margin of RF-Morgan over kNN-Tanimoto in
the < 0.5 bin.

- **Learning beyond similarity:** the interval lies entirely above 0.
- **No evidence beyond similarity:** the interval includes 0. The interval's width is
  reported, to show how large an advantage could have gone undetected.
- **Worse than lookup:** the interval lies entirely below 0.

Secondary, reported but not deciding anything: the same margin in each of the four bins, and
the number of labels individually better or worse than kNN-Tanimoto overall.

---

## Phase C: Arctander, external

Runs after Phases A and B are committed. Arctander is not touched by any model before then.

- **Train** on all 4,981 GS+LF molecules, with no folds. The models and settings are the same
  as in Phase B.
- **Test** on Arctander's 901 independent molecules (838 groups), using the labels fixed in A3.
  None of these molecules is in GS+LF, and none shares a rule-3 group with a GS+LF molecule.
- **Per label:** AUC for RF-Morgan, RF-MW, F-MW and kNN-Tanimoto. The floor is the higher of
  F-MW and RF-MW.
- **Pooled:** the macro-mean margin of RF-Morgan over the floor, and over kNN-Tanimoto, with
  2,000 group bootstraps over Arctander's groups, seed 0.

**Reading:**
- *Transfers:* the pooled margin over the floor has its interval entirely above 0.
- *Beyond similarity on a new source:* the pooled margin over kNN-Tanimoto has its interval
  entirely above 0.
- Either can fail on its own. Both outcomes are framed according to A2's decision.

**Caveat whatever A2 finds.** Independence here means no GS+LF molecule at Tanimoto ≥ 0.70.
These molecules are new to GS+LF but not necessarily unfamiliar in structure, so the
nearest-GS+LF-neighbour distribution of the 901 is reported with the results.

---

## Not in milestone 3

- Any change to the models, features, k, thresholds or seeds; hyperparameter search; neural
  networks; pretrained embeddings; multi-output models.
- Labels with < 50 GS+LF positives.
- Keller, Dravnieks and Mayhew.
- Anything generative.

## Deliverables

- `sw_qsor/sources.py` gains the two OpenPOM per-source files. `sw_qsor/datasets.py` gains
  loaders for them and for Arctander labels by molecule. The vocabulary mapping is committed as
  a data file with the plan. κ and the joint group bootstrap go in `sw_qsor/evaluate.py`, with
  tests.
- `demo_milestone3_provenance.py` (Phase A), `demo_milestone3_gslf.py` (Phase B) and
  `demo_milestone3_arctander.py` (Phase C) are callers of the package. Each prints its checks,
  its measurements and its verdict.
- Phase B's per-label table goes in `reports/05_m3_per_label.csv`.
- A results section is appended here per phase, whatever the outcome.

## Decided at review (17 Nov 2026), all four as drafted

1. **Q2 is co-primary.** The alternative was to keep kNN-Tanimoto as a reported reference only.
2. **`vanillin`→`vanilla` is kept.** It is the one mapping that is more than spelling, because
   vanillin is a molecule, not a descriptor. Dropping it would have left 63 mapped labels.
3. **The A2 threshold is a κ difference ≥ 0.10.** It is arbitrary, and it is fixed here before
   anything is computed.
4. **The Q1 bands are half / 10 labels.** Their only purpose is to make the stated expectation
   checkable.

## Phase A results (17 Nov 2026)

Run with `python demo_milestone3_provenance.py`, using labels and structures only. No model
was fitted. Outputs: `reports/05_arctander_vocab.csv` and `reports/05_phase_c_labels.csv`.

### A1. Vocabulary

- The fixed mapping covers 67 of Arctander's 77 labels (59 exact, 8 variants), 64 of them
  among the 112, as planned. **But those 64 map onto 63 GS+LF labels:** Arctander has both
  `vanilla` and `vanillin`, so the approved `vanillin`→`vanilla` mapping merges two columns.
  Merging is the only coherent reading of an approved many-to-one mapping, so they are
  unioned: a molecule is `vanilla` if Arctander gives either label.
- **Error in the review note:** "dropping it leaves 63" was wrong. Dropping `vanillin` would
  also have left 63.
- Arctander: 3,102 stimuli. 278 have CID 0 (structure never resolved) and are dropped,
  leaving 2,751 molecules. **233 molecules carry no label at all and are treated as missing,
  not negative.** Their text descriptions show the labels went unextracted. One entry reads
  "simultaneously non-floral and heavy-sweet-floral" and has no `floral` label. This is
  decided here, before any model sees Arctander. It also says Arctander's labels are an
  imperfect reading of its own text.
- That leaves 2,488 labelled Arctander molecules by connectivity key.

### A2. Provenance

- **Plan error:** Leffingwell's file does not use the same 138 labels; it lacks `cortex`
  (137). That label is not mapped, so A2 is unaffected.
- Molecules in GoodScents (3,918), Leffingwell (3,388) and labelled Arctander together:
  **1,058**. Labels with ≥ 20 Arctander positives among them: **39**.

| | Macro-mean κ over 39 labels |
|---|---|
| GoodScents vs Arctander | 0.421 |
| Leffingwell vs Arctander | 0.559 |
| *GoodScents vs Leffingwell (not in the plan; context only, decides nothing)* | *0.485* |

**κ(GS, Arc) − κ(LF, Arc) = −0.138, 95% group-bootstrap interval [−0.163, −0.112]** (705
groups, 2,000 resamples). Only 3 of the 39 labels favour GoodScents: hyacinth, amber and
orange, all with 21–22 Arctander positives.

**Decision, under the fixed rule: Phase C is reported as an external test**, with the plan's
caveat.

What the rule could not see, as the plan anticipated: the asymmetry runs the *other* way.
Leffingwell, half of GS+LF, is the source that tracks Arctander most closely of the three
pairs. That fits Leffingwell drawing partly on Arctander, but does not establish it: three
sources agreeing at 0.42–0.56 could also reflect shared perfumery convention. Phase C's
results should be read with that possibility in mind. Separately, agreement between sources of
only 0.42–0.56 caps what any model trained on one source can score on another.

### A3. Phase C's label set

- Arctander molecules independent of GS+LF: **901**, as in report 03. Of those, **805** are
  labelled.
- **25 mapped labels have ≥ 20 positives** among the 805, and these are Phase C's labels:
  floral, green, herbal, woody, fruity, balsamic, rose, oily, camphoreous, spicy, ethereal,
  musty, mint, earthy, musk, amber, medicinal, fatty, citrus, winey, anisic, nutty, leafy,
  animal, caramellic.

Phase B can proceed. Phase C waits for Phase B's results to be committed.

## Phase B results (17 Nov 2026)

Run once with `python demo_milestone3_gslf.py`, plan unmodified. Fitting took about 40
minutes on 24 threads: 112 labels × 3 forests × 5 folds. Per-label table:
`reports/05_m3_per_label.csv`.

### Checks

1. **Pipeline:** shuffled-label macro-mean AUC **0.4852**, labels ranging 0.342–0.587.
   Inside 0.48–0.52: **pass**. One caveat: the check is weaker than intended. Milestone 2's
   scheme, a generator seeded 0 per run, was followed per label, so every label's shuffled
   training set uses the *same* permutation of molecules. The 112 shuffled runs are therefore
   correlated, not independent, which likely explains a mean 0.015 below 0.5.
2. **Reproduction:** floral's per-fold AUCs equal milestone 2's to three decimals for all
   four models: **pass**.

### Q1, per label: 71 pass, 22 fail, 19 inconclusive. Reading: **BROAD**

| GS+LF positives | Labels | Pass | Fail | Inconclusive | Median floor | Median RF-Morgan | Median margin |
|---|---|---|---|---|---|---|---|
| 50–99 | 50 | 31 | 11 | 8 | 0.665 | 0.812 | +0.118 |
| 100–249 | 39 | 23 | 9 | 7 | 0.723 | 0.834 | +0.113 |
| 250–499 | 15 | 12 | 1 | 2 | 0.745 | 0.828 | +0.110 |
| 500+ | 8 | 5 | 1 | 2 | 0.682 | 0.797 | +0.089 |

**`plan.md`'s prediction was wrong.** It expected a few common descriptors to be learnable and
the rest not. Instead 71 of 112 labels pass, and rare labels pass about as often as common
ones. Across labels the median margin over the floor is +0.111 (range −0.071 to +0.242).
Macro-mean AUC: RF-Morgan 0.802, floor 0.701, kNN-Tanimoto 0.759.

- **Failed (22):** fresh, ethereal, musty, pungent, creamy, dry, fermented, metallic, banana,
  alcoholic, sharp, warm, rummy, ripe, natural, aromatic, grassy, solvent, orris, tea, fruit
  skin, juicy.
- **Inconclusive (19):** sweet, herbal, tropical, earthy, winey, berry, pear, savory, animal,
  cognac, tobacco, musk, pine, hay, lily, medicinal, geranium, cortex, soapy.

Descriptive observations, not tests: the plan did not ask for these.
- **Failure is not explained by a high floor.** Failed and passing labels have the same median
  floor (0.701 vs 0.700). Some failures are labels molecular weight already predicts
  (alcoholic 0.864, ethereal 0.843, solvent, pungent). Many others are words for an overall
  quality rather than a chemical character: fresh, dry, warm, natural, aromatic, sharp,
  juicy, ripe.
- **The largest margins are on labels named after chemical classes or specific materials:**
  coumarinic +0.242, lactonic +0.241, chocolate, muguet, cinnamon, and odorless (+0.200). A
  label like "lactonic" comes close to a structural definition, so part of the broad result
  is structure recognising structure.
- **For some labels RF-Morgan scores *below* RF-MW** (sharp −0.071, pungent −0.056): the
  forest on fingerprint bits does not recover the size signal that molecular weight gives
  directly.

### Pooled over 91 labels with ≥ 10 positives below 0.5 similarity

1,090 molecules and 3,970 positives; 2,000 joint group bootstraps, seed 0.

| | Margin | 95% interval | Reading |
|---|---|---|---|
| **Q1:** RF-Morgan over floor | **+0.040** | [+0.013, +0.050] | **Structure signal persists far from training** |
| **Q2:** RF-Morgan over kNN-Tanimoto | **+0.006** | [−0.008, +0.020] | **No evidence beyond similarity** |

**Q2's interval is 0.028 wide**, so any pooled advantage over neighbour lookup on distant
molecules larger than about +0.02 AUC is excluded.

**The Q1 interval is lopsided** (estimate near its upper end). The likely reason: in each
resample the floor is the higher of two noisy AUCs, and that maximum is biased upward, which
shifts resampled margins down. This makes the interval conservative, the direction the plan
intended. Q2 has a single comparator, and its interval is centred.

Secondary, deciding nothing, by bin:

| Similarity bin | Molecules | Over floor | Over kNN-Tanimoto |
|---|---|---|---|
| 0.0–0.4 | 455 | **+0.000** [−0.043, +0.019] | +0.009 [−0.019, +0.035] |
| 0.4–0.5 | 635 | +0.046 [+0.014, +0.058] | +0.004 [−0.012, +0.019] |
| 0.5–0.6 | 1,536 | +0.098 [+0.079, +0.105] | +0.030 [+0.021, +0.038] |
| 0.6–0.7 | 2,355 | +0.097 [+0.081, +0.104] | +0.037 [+0.028, +0.046] |

Overall (all molecules), 101 labels score above kNN-Tanimoto and 11 below; 56 are above in
all 5 folds and none is below in all 5.

### What it means

- **Structure predicts most odour descriptors better than size does,** and broadly: 71 of
  112 labels pass the same bar floral passed. The prediction of a narrow result was wrong.
- **The forest beats neighbour lookup only where neighbours exist.** At 0.5–0.7 similarity it
  is ahead by 0.03–0.04 AUC, consistently. Below 0.5 the pooled advantage is +0.006 and any
  advantage above about 0.02 is ruled out. Milestone 2's single-label finding holds across
  labels.
- **On the most unfamiliar molecules (similarity < 0.4), even the margin over molecular weight
  disappears** (+0.000). The pooled "persists" verdict rests on the 0.4–0.5 bin. This is a
  secondary result and decides nothing, but it is the plainest statement in this run of where
  the limit lies.
- Phase C now tests whether this transfers to Arctander's 805 labelled independent molecules.
  Those are new to GS+LF, but mostly not this unfamiliar.

## Phase C results (17 Nov 2026)

Run once with `python demo_milestone3_arctander.py`, plan unmodified, in 93 seconds. Models
were fitted on all 4,981 GS+LF molecules and tested on **805 labelled Arctander molecules
independent of GS+LF** (745 rule-3 groups), over the 25 labels fixed in A3. Per-label table:
`reports/05_phase_c_per_label.csv`. Framing per Phase A: **an external test, with the caveat
that Leffingwell, half of GS+LF, tracks Arctander more closely than GoodScents does.**

The plan specified no shuffled-label check for Phase C, and none was run.

### How unfamiliar the test molecules are

Similarity of each test molecule to its nearest GS+LF molecule: median 0.56.

| Bin | 0.0–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 |
|---|---|---|---|---|
| Arctander test molecules | 93 (12%) | 139 (17%) | 273 (34%) | 300 (37%) |
| GS+LF test folds, Phase B | 9% | 13% | 31% | 47% |

Slightly more distant than GS+LF's own folds, but most are still within 0.5–0.7 of a training
molecule, as the plan's caveat expected.

### Results

| Model | Macro-mean AUC over 25 labels |
|---|---|
| F-MW | 0.610 |
| RF-MW | 0.621 |
| kNN-Tanimoto | 0.707 |
| **RF-Morgan** | **0.745** |

RF-Morgan is above the floor on 24 of 25 labels (the exception is leafy, −0.022) and above
kNN-Tanimoto on 17.

| Pooled (2,000 group bootstraps, seed 0) | Margin | 95% interval | Reading |
|---|---|---|---|
| RF-Morgan over floor | **+0.110** | [+0.084, +0.126] | **Transfers** |
| RF-Morgan over kNN-Tanimoto | **+0.038** | [+0.024, +0.052] | **Beyond similarity on a new source** |

Secondary, deciding nothing; point estimates only, over labels with ≥ 5 positives in the bin:

| Bin | Molecules | Labels | RF-Morgan | Over floor | Over kNN-Tanimoto |
|---|---|---|---|---|---|
| 0.0–0.4 | 93 | 13 | 0.674 | +0.064 | **+0.001** |
| 0.4–0.5 | 139 | 16 | 0.666 | +0.059 | +0.033 |
| 0.5–0.6 | 273 | 24 | 0.748 | +0.082 | +0.044 |
| 0.6–0.7 | 300 | 25 | 0.754 | +0.123 | +0.038 |

### What it means

- **The structure signal transfers to a different source.** Trained on GS+LF, the forest beats
  the size floor on Arctander by 0.11 AUC pooled, on 24 of 25 labels. The floors are lower on
  Arctander (about 0.61 against 0.70), so the margin is larger than in Phase B, though absolute
  AUCs are lower (0.745 against 0.80). Sources agreeing only at κ 0.42–0.56 (Phase A) would
  produce exactly that.
- **"Beyond similarity" here is not a contradiction of Phase B's Q2, and should not be read
  as one.** Phase C's criterion pools *all* test molecules. Most sit at 0.5–0.7 similarity,
  where Phase B also found the forest ahead of lookup by 0.03–0.04. Phase C shows that
  advantage carries over to another source's labels. On the 93 most distant molecules (< 0.4)
  the advantage over lookup is again +0.001, though there are too few molecules for that
  figure to decide anything.
- **The caveat stands.** If Leffingwell drew on Arctander, part of this transfer is a shared
  description tradition rather than independent perception.

## Milestone 3, summary

1. **Breadth (Q1):** structure beats molecular weight on 71 of 112 GS+LF labels under rule 3,
   with rare labels doing about as well as common ones. `plan.md`'s narrow prediction was
   wrong.
2. **Beyond similarity (Q2):** on GS+LF molecules below 0.5 similarity, pooled over 91 labels,
   the forest is no better than averaging the labels of the 10 most similar training molecules
   (+0.006 [−0.008, +0.020]). Its advantage over lookup, 0.03–0.04, exists where close
   neighbours exist.
3. **External (Q3):** both findings carry over to Arctander. The forest beats the floor by 0.11
   and lookup by 0.04, the latter again coming from molecules with neighbours at 0.5–0.7.
4. **Where the limit is:** below 0.4 similarity the forest neither beats lookup (both datasets)
   nor, on GS+LF, even molecular weight. A single forest on 2048 Morgan bits has not been shown
   to learn anything about molecules unlike its training set.

What would move this is a different representation or model tested on the distant bins under
the same protocol. That is a new question for a new plan, per milestone 2's rule against re-running
with changed settings.
