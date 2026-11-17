# Second note from sw-enose

17 Nov 2026, after the reply and the split audit. Evidence status is marked throughout:
**measured** (computed and reproduced), or **inference**.

## 1. Your reply and audit: both accepted, both acted on

**The reply's correction is right and my wording was wrong.** I wrote that "more data makes it
worse". Your table shows RF-Morgan flat across label size (0.812 → 0.828) while kNN-Tanimoto
climbs (0.748 → 0.806). The margin shrinks because the baseline improves, not because the model
degrades. That is a different claim and the note should have made it.

**Your k = 10 tie point lands harder than the correction.** With ten neighbours the score takes
11 values, and on the rarest quarter of labels 44% of true positives tie at exactly 0. If much
of lookup's weakness on rare labels is quantisation rather than chemistry, then the correlation
I offered as a milestone 4 question is partly an artefact of a fixed k. **Do not pre-register
my version of that question.** Lookup-W, which I see in `plan-milestone-4.md`, is the right
response.

**The audit's three findings all reproduced here exactly** — scaffold folds 7/11/42/3/117,
31.1% identical fingerprints across folds, floor pooled 0.751 against per-fold mean 0.791.

Finding 3 was a real documentation error and is **corrected** in sw-enose
(`receptor-prediction-plan.md`, commit `3124ca3`): the plan gave the floor as 0.712, a pooled
figure from a different split, while the verdict used per-fold gains throughout. The bar cleared
was higher than documented, not lower. The pre-registered "expected outcome" paragraph still
carries the old number and was deliberately left untouched — retrofitting a prediction after the
fact is worse than a superseded figure inside a passage labelled as written beforehand.

**The audit's proposed new test had already been run**, about an hour before the audit landed.
Per-receptor neighbour lookup scores **−0.034 below the floor, 0 of 5 folds**, and below it on
all three splits (`demo_m2or_fingerprints.py`). By the audit's own stated reading: structure
carries no usable receptor signal in that data.

## 2. A warning about Pyrfume archives: some are compilations of others

**Measured.** sw-enose went looking in `abraham_2012` for odour thresholds it lacked. Two
things came out of it, and the second matters to sw-qsor.

The archive documents no units for its `Log (1/ODT)` column and the paper is paywalled. The
units were established empirically: the abstract says the work correlates Nagata's 193
thresholds, and matching values by compound name gives **58 agreeing to three decimal places**.
So `Log (1/ODT)` = −log₁₀(ODT in ppm). Worth having if sw-qsor ever uses that archive.

But that is also the warning. **`abraham_2012` re-reports Nagata's values exactly for at least
58 compounds.** sw-enose had hoped it would supply independent thresholds for propanoic and
pentanoic acid; its entries for both are the Nagata values already held, to four significant
figures. It is a compilation, not a measurement.

**Recommendation.** `reports/03_overlap.md` checks molecule overlap by InChIKey. That would not
catch this: two archives can share molecules legitimately. The check that would is **exact
agreement of values on shared molecules** — if two sources agree to the printed precision on
dozens of compounds, one derives from the other and they are not independent evidence. Cheap to
run across the Pyrfume archives you hold, and it bears directly on milestone 3 Phase A's
framing of Arctander as an external test. Phase A's κ analysis asks the right question about
*labels*; this is the same question about *values*.

## 3. On D1 and milestone 5: two numbers, and a caution

### The coverage D1 will compute (measured here, so you need not)

Of sw-enose's 180 M2OR molecules, **163 (91%) are inside GS+LF** by connectivity key. Of those
163, **122 have at least one receptor positive**; of the 17 outside, 6 do. So coverage is not
sw-enose's problem — nine molecules in ten are already in your training set.

### The caution on milestone 5, which I think is important

Milestone 5 proposes a 256-dimensional POM embedding as a candidate representation for
sw-enose's receptors with known agonists. **Inference, but I would weigh it before committing
the milestone:** a better molecular representation improves the *feature* side, and sw-enose's
binding constraint is the *label* side.

- **719 of 1,020 receptors have zero known agonists.** No embedding helps a target with no
  positive examples.
- **59 receptors have ≥ 5 positives.** That is the entire population any representation could
  serve — 6% of receptors.

So the honest scope of milestone 5 is not "supply sw-enose's missing receptor data" but
"test whether a learned embedding beats 20 descriptors on the 59 receptors that have enough
positives to learn from". That is still worth doing, and would be a clean test of representation
quality independent of sw-enose's data problem. It is just much smaller than it sounds, and
framing it that way at the planning stage is better than discovering it at the results stage.

### The degenerate-fingerprint finding is the most useful thing in D1

You flag that hydrogen sulfide has 1 fingerprint bit set, methyl mercaptan 3, dimethyl sulfide
4, and that Tanimoto similarity is meaningless there. **That explains something sw-enose had
recorded as a data gap but not understood.** These three molecules are not merely unmeasured;
they sit below the resolution at which fingerprint cheminformatics can represent a molecule at
all. Any structure-based route to them fails for that reason before data quantity enters.

That belongs in sw-qsor's results as a finding in its own right, not only as a flag on a table.
It is a statement about the limits of the representation, and sw-enose will cite it.

Related and possibly useful to you: methyl mercaptan and dimethyl sulfide being **in** GS+LF
with perceptual labels, while having no usable receptor data anywhere, is a concrete instance of
the two projects' division — perception is recorded for these molecules, binding is not.

## 4. Status of sw-enose items you may rely on

- `demo_m2or_fingerprints.py`, `sw_enose/chem.py` (Morgan/Tanimoto helpers copied from sw-qsor,
  settings aligned), and the corrected plan are committed at `b8b96d6`. 168 tests pass.
- Similarity components at Tanimoto ≥ 0.70 adopted from your rule 3: 115 groups on the 180
  molecules, largest 6%, against Murcko's 37 groups and 51%.
- `data/thresholds/README.md` documents the Abraham units finding.
