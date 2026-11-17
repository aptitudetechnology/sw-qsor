# Note from sw-enose: what the sibling project found, and what it implies here

Written 17 Nov 2026 for whoever picks up sw-qsor. Assumes no knowledge of sw-enose.

Everything below is one of three things, and each claim says which: **measured** in sw-enose
against a pre-registered bar, **measured here** on sw-qsor's own committed results but *not*
pre-registered, or **inference**.

## The one-paragraph context

[sw-enose](https://github.com/aptitudetechnology/sw-enose) is this project's sibling: a
biomimetic nose that goes air → mucus → olfactory receptor activation → neural stages. It is
blocked by missing receptor binding data, and it tested three routes to predicting that data —
fitting related compounds' EC50s, physics-based QM/MM, and statistical models over 24,735
published receptor–odorant screening pairs. All three failed against bars fixed in advance.

## 1. The exchange that prompted this note

sw-qsor's milestones 2 and 3 found that **kNN-Tanimoto on Morgan fingerprints is the model
hardest to beat** (measured). sw-enose's statistical route had excluded fingerprints entirely,
reasoning that ~144 training molecules could not support 2,048 bits — which is true of a
forest but not of neighbour lookup, which fits no parameters at all. So sw-enose had closed a
route without running the strongest baseline.

It was run (measured, `sw-enose/demo_m2or_fingerprints.py`). Both kNN-Tanimoto (−0.034) and
RF-Morgan (−0.107) land **below** the chemistry-free floor on all three splits, 0 of 5 folds
positive. The closure held, now on the strongest baseline rather than the most convenient.

**Transfer: always include the parameter-free structural baseline.** sw-qsor already does; this
is a record of what happens when a project does not.

## 2. The correction that matters more

sw-enose had recorded that its failure was caused by *180 distinct molecules in 37 scaffold
groups* — too small a sample. Two measurements show that was the wrong diagnosis:

- **Molecular coverage was not the problem.** Median nearest-neighbour similarity to another
  fold is **0.62** in sw-enose's 180 molecules against **0.56** in sw-qsor's GS+LF folds. Its
  molecules are *closer* together than the dataset where structural models work.
- **Positives per target was the problem.** Median positives per receptor: **0**. 719 of 1,020
  receptors (70%) have no known agonist at all. In a 4/5 training split a receptor has on
  average **0.77 positive examples**.

The task was never one structure–property relationship learned from 180 molecules. It was
**1,020 separate relationships learned from under one positive example each**. No feature set
or model repairs a training set whose median target has zero positives.

**Transfer: count positives per target, not molecules per dataset.** sw-qsor's milestone 1
already does this (112 labels with ≥ 50 positives), which is why it is not in the same trouble.
Keep doing it, and keep it in any dataset documentation — the headline molecule count hides it.

## 3. An observation on sw-qsor's own results (measured here, NOT pre-registered)

sw-enose's result suggests an obvious question: is sw-qsor's "structure does not beat neighbour
lookup far out" also a data-quantity limit that more positives would fix? Its per-label table
(`reports/05_m3_per_label.csv`) answers it directly, because label size varies 50 to 1,901.

**The answer is no, and it goes the other way.** The margin over kNN-Tanimoto *shrinks* as
labels get more positives:

| Positives per label | Labels | Median margin over kNN | Median margin over floor |
|---|---|---|---|
| 50–75 | 28 | **+0.064** | +0.124 |
| 75–110 | 28 | +0.049 | +0.118 |
| 110–199 | 28 | +0.034 | +0.094 |
| 199–1,901 | 28 | **+0.026** | +0.100 |

Spearman correlation of margin-over-kNN with positives: **−0.469 (p < 0.001)**; with
log₁₀ positives, −0.451 (Pearson, p < 0.001). The margin over the *size floor* shows no such
trend (−0.172, p = 0.071), so this is specific to the comparison with neighbour lookup.

**Inference, and the reason to pre-register it before believing it:** a plausible mechanism is
that with few positives, kNN has few labelled neighbours to draw on and the forest's ability to
pool across the whole dataset helps; with many positives, kNN has plenty and catches up. If
that is right, **"structure does not beat similarity" is not a limitation more data removes —
more data makes it worse**, and that is a materially different claim from the one milestone 3
currently makes.

**Confound to rule out first.** Common labels may simply be vaguer (`fresh`, `sweet`,
`natural`, `warm` are among milestone 3's failures and are all high-count), and vagueness could
depress both methods and compress the gap between them. Label frequency and label vagueness
are not separated by anything measured so far. A pre-registered test would need to control for
it — for example by comparing margin against positives *within* the chemically-specific labels
(`lactonic`, `coumarinic`, `cinnamon`) separately from the quality words.

This observation was computed after the fact on committed results. Under this project's own
rules it decides nothing. It is offered as the most promising pre-registered question available.

## 4. What this says about the two projects together

- sw-enose: structure → **receptor response** fails, because targets have ~0 positives each.
- sw-qsor: structure → **percept** works against a size floor, broadly (71 of 112 labels), but
  does not beat neighbour lookup on unfamiliar chemistry.

Taken together (inference): the published structure–olfaction relationship is learnable where
targets are well populated and not otherwise, and where it is learnable it has so far bought
nothing a similarity lookup does not already provide on genuinely new molecules. That is a
bound worth stating plainly, and neither project could state it alone.

## 5. Practical notes

- sw-enose's settings were copied *from* sw-qsor (Morgan radius 2, 2,048 bits, k = 10, forest
  of 1,000 trees, sqrt features) so the two projects' numbers are directly comparable. Keep
  them aligned if either changes.
- sw-enose's similarity-component grouping at Tanimoto ≥ 0.70 was adopted from sw-qsor's
  amended rule 3, and was markedly better there too: 115 groups with the largest at 6%, against
  Murcko's 37 groups with the largest at 51%.
- Two literature searches in sw-enose were stopped by a safety classifier without visible
  reason. Nothing in sw-qsor's subject matter resembles them, and its dataset survey ran
  cleanly. Recorded as a quirk, not an expected obstacle.
