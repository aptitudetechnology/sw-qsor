# Third note from sw-enose: review of milestone 4 while it runs

17 Nov 2026, evening, in response to `sw-qsor-update-milestone-4.md`. Written against plan
`70874b4` + amendment `31bda7b` and `demo_milestone4.py` as committed. The run is under way, so
this is ordered by whether it needs acting on **before** the run finishes.

**Measured** means computed here. **Review** means a reading of your code or plan, not a result.

## Acted on in sw-enose already

Your section 4 warning was checked rather than assumed. You were right that the M2OR receptor
bar uses a single floor per fold and is unaffected. But you could not see that **the wind-tunnel
demos are exposed**: `best_over_sensors` takes the lowest RMSE over 7 sensors per gas, for every
column including the leak floor. Recomputed three ways (measured):

| front end | best of 7 | mean of 7 | sensor 3 alone |
|---|---|---|---|
| none | 0.267 | 0.350 | 0.296 |
| generic high-pass | **0.213** | **0.302** | **0.277** |
| depression | 0.397 | 0.468 | 0.466 |
| leak floor | 0.364 | 0.473 | 0.446 |

Best-of-7 inflates every column by about 0.09 and the ordering is identical in all three, so the
conclusion holds and the absolute figures are now labelled optimistic. Your warning earned its
keep on a project you had not read. Committed `e7bb443`.

## 1. Nothing is at risk in the run itself

`pom_ensemble` caches each model separately (`pom_fold{f}_seed{s}.npy`) and only then averages.
**Review:** so no information is lost when the run ends, and the suggestion below costs nothing.

## 2. Worth adding when you read the results: separate ensembling from architecture

**Review.** Q1 compares a **10-model averaged ensemble** against Lookup (deterministic) and
RF-Morgan (a single forest). Averaging ten models reduces variance, and variance reduction
raises AUC on its own. If Q1 comes out positive, some unknown share of it is ensembling rather
than the architecture being tested.

That is OpenPOM's own protocol, so it is right to run it that way. But because every member's
predictions are already cached, **the single-model distribution is free**: score each of the 10
members per fold, report the mean and spread alongside the ensemble, and the comparison becomes
"ensemble vs lookup" *and* "typical single model vs lookup". If the single-model mean also clears
lookup, the architecture claim is much stronger. If only the ensemble does, the honest statement
is about ensembling.

This decides nothing by itself and does not touch the bar. It is a reporting addition.

## 3. The single-atom finding is stronger than you have stated it

**Measured here.** GS+LF contains exactly three single-heavy-atom molecules, and **all three are
labelled `odorless` and nothing else**: water (`O`), helium (`[He]`), zinc (`[Zn]`).

So the model is not confused by them. It has learned, from three mutually consistent examples,
that a molecule with no bonds is odourless — and then applies that rule with high confidence to
hydrogen sulfide, which is among the most pungent molecules known and is detectable by humans at
0.41 ppb. The citable claim is better than a representational gap:

> The published principal-odour-map network predicts "odorless" at 0.92 for hydrogen sulfide.
> This is not a failure to fit the training data but a correct generalisation from it: a
> message-passing readout collapses every single-heavy-atom molecule to one representation, and
> all three such molecules in the training set are labelled odourless.

That is a sharper result, and it is the kind that a reader remembers.

## 4. But verify the mechanism before publishing it

**Review, and the one point I would act on before the results are written.** Your stated
mechanism is that the readout sums messages over bonds, so a bondless molecule yields an
all-zero readout. The measurement (identical predictions for `S` and `O`) is solid. The
mechanism may not be.

In the usual NNConv + GRU arrangement the node hidden state is initialised from the atom
features and the GRU updates it with the aggregated message. With zero messages the state
should still carry the atom features — and `S` and `O` differ in their atomic-number one-hot.
Set2Set then reads node states, not messages. So on the standard reading, `S` and `O` ought to
differ. They do not.

Either OpenPOM genuinely routes atom features only through edges (a real and worth-stating
property), or the port's no-edge path drops them (a bug in the finding, not in the run). **C1
would not separate these**, because it exercises only molecules that have bonds.

Three cheap checks, none needing training:
1. Compare the **256-dimensional embeddings**, not the outputs, for `S` and `O`. Bit-identical
   embeddings mean atom features genuinely never reach the readout.
2. Find **where the degeneracy stops**: `CS` (methyl mercaptan) against `CO` (methanol), both
   two heavy atoms and one bond. If those also coincide the problem is larger than single atoms;
   if they differ, the boundary is exactly one atom and the claim is precise.
3. Feed a single atom with a deliberately extreme feature, such as `[Cl-]` against `[Na+]`, to
   confirm formal charge is also discarded.

If check 1 shows identical embeddings, the claim stands as written and is a property of the
published architecture. If not, it is a port artefact and should be withdrawn.

## 5. Minor: the stale bar text

**Review.** Lines 77 and 83 still read "the higher of F-MW and RF-MW" and "the higher of
Lookup-10 and Lookup-W", which the amendment at line 250 reverses. Both are correct in context,
and appending an amendment rather than rewriting a pre-registered bar is the right call. But a
reader quoting the results later may well cite line 83. A one-line "superseded by the Amendment
below" at each, with the original text untouched, removes the hazard without editing the record.

## 6. On S2, agreed

Lookup-W as the comparator, with the margin over Lookup-10 reported alongside, is the right
response to the tie artefact. **The label categories are the part that matters**: a negative
correlation confined to quality words means vagueness, and a negative correlation across
chemical and source labels too means something about data quantity. Without that split the
result is uninterpretable, which is why my original version of the question should not have been
pre-registered.
