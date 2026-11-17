# 02 — How molecules are grouped for held-out splits

Written 17 Nov 2026, before any model or floor was run. The comparison uses structure only:
no labels were read, so the choice cannot have been tuned to a result. Reproduce with
`python demo_grouping.py`.

## Answer

**Groups are connected components of the graph linking molecules whose Morgan fingerprints
(radius 2, 2048 bits) have Tanimoto similarity ≥ 0.70.** Whole groups go to 5 folds, balanced
by molecule count. Records sharing an InChIKey connectivity block (`sw_qsor.identity`) are
forced into one group. On GS+LF this gives 3,434 groups, the largest holding 235 molecules (4.7%),
and by construction **no test molecule has a training molecule at similarity ≥ 0.70**. Every
result is also reported binned by each test molecule's similarity to its nearest training
molecule, because the guarantee stops at 0.70 and most test molecules sit just below it.

## Why the plan's test was not enough

`next-session.md` proposed one test: no group above about 10% of molecules. It is necessary
but does not separate the candidates, because group size and leakage trade against each
other. Plain Murcko leaks least precisely *because* it lumps 45% of molecules into one group.
The measure that matters is fold-level: for each molecule, how similar is the nearest molecule
in another fold, which is its nearest training neighbour when it is in the test fold.

## Comparison (GS+LF, 4,983 molecules, 5 folds)

"Nearest in another fold" columns are over all molecules. "Scaffold seen" is the share of
ring-bearing test molecules whose Murcko scaffold also appears in training.

| Grouping | Largest group | Folds balanced | Identical FP across folds | ≥ 0.7 | > 0.5 | Median | Scaffold seen |
|---|---|---|---|---|---|---|---|
| Random molecules (reference) | — | yes | 9.7% | 36.4% | 77.4% | 0.64 | 84.5% |
| Murcko scaffold | 44.9% | **no** (2,236 vs 662) | 0.6% | 3.4% | 23.7% | 0.41 | 0% |
| A: Murcko, acyclics by Butina d<0.7 | 15.3% | yes | 0.8% | 12.8% | 50.3% | 0.51 | 0% |
| B: Butina on everything, d<0.7 | 18.0% | yes | 0.0% | 15.1% | 56.9% | 0.54 | 82.3% |
| C: carbon skeleton | 6.6% | yes | 9.7% | 28.5% | 65.9% | 0.58 | 81.6% |
| D: components, sim ≥ 0.65 | 12.3% | yes | 0.0% | 0.0% | 69.4% | 0.56 | 83.6% |
| **D: components, sim ≥ 0.70** | **4.7%** | yes | **0.0%** | **0.0%** | 74.4% | 0.59 | 84.3% |
| D: components, sim ≥ 0.75 | 2.2% | yes | 0.0% | 16.2% | 75.8% | 0.61 | 85.0% |

Butina distance cutoffs 0.6 and 0.8 were also tried for A and B. 0.6 gives slightly smaller
largest groups (B: 9.6%) and slightly more leakage; 0.8 gives a largest group of 30–37%.

## Reasons

- **C is out.** Grouping by carbon skeleton splits homologous series by chain length, and
  hexyl and heptyl esters have *identical* radius-2 fingerprints. It leaks identical
  fingerprints across folds as often as a random split does (9.7%).
- **A and B fail the size test** at the cutoffs that limit leakage, and even where they pass
  it they leave 13–20% of molecules with a ≥ 0.7 neighbour in training. Butina also depends on
  input order. A's one real advantage is that ring scaffolds never cross folds. Keeping that
  while also merging on similarity would mean merging on shared scaffold *or* similarity, and
  the benzene scaffold alone (762 molecules, 15%) breaks the size limit.
- **D gives a guarantee that can be checked**: a stated threshold that no train–test pair
  crosses. It does not depend on input order, it passes the size test with room to spare, and
  its large groups are chemically what a split should keep together: the 235-molecule group
  is ethyl/propyl esters, the next two are fatty acids (63) and aliphatic aldehydes (62).
- **Why 0.70 and not another threshold.** Components percolate just below it: the largest
  group is 1,507 molecules (30%) at 0.60, 612 (12%) at 0.65, 371 at 0.68, 235 at 0.70, and 165
  at 0.72. 0.70 is the lowest round threshold that passes the size test. Going lower is not
  possible; going higher gives up the guarantee for pairs just above 0.70.

## What D does not do, stated plainly

- **Moderate analogues still cross.** 74.4% of molecules have a neighbour above 0.5 in another
  fold, close to random's 77%. Nearest-training-neighbour similarity of test molecules under
  D: 442 below 0.4, 629 at 0.4–0.5, 1,551 at 0.5–0.6, 2,361 at 0.6–0.7. GS+LF is dense with
  analogues, and no balanced split makes most of its test molecules far from training.
- **Ring scaffolds are shared** across folds about as often as under a random split (84%).
- **The guarantee is in one fingerprint's terms.** Two molecules that are close by another
  measure can still be split apart.

Hence the reporting requirement in rule 3: results binned by nearest-training-neighbour
similarity (< 0.4, 0.4–0.5, 0.5–0.6, 0.6–0.7), with the chemistry-free floor computed under the
same split and bins. A model that beats the floor only in the 0.6–0.7 bin has learned to
recognise neighbours, not structure–odour rules.

## Group-size distribution under D at 0.70 (GS+LF)

| Group size | Groups |
|---|---|
| 1 | 3,010 |
| 2–4 | 345 |
| 5–19 | 68 |
| 20–99 | 10 |
| 100+ | 1 (235) |

Top ten: 235, 63, 62, 49, 47, 40, 39, 38, 28, 27.

## Scope

Groups are computed over **every molecule in the experiment at once**. For a cross-dataset
test (train on GS+LF, test on Arctander, say), that means the union, deduplicated first, so
that the guarantee holds across datasets as well as within them.

## Correction, same day (step 3)

The first version of this report grouped GS+LF's 4,983 rows as they come. Step 3 found that
two molecules appear twice, once neutral and once ionised, and that each pair's fingerprint
similarity is only 0.31 and 0.55. So components alone could have put one record of a molecule
in training and the other in test. `similarity_components` now takes identity keys and links
equal keys. The figures above are re-run with that fix. The largest group (235) and both
guarantees (0.0% identical, 0.0% ≥ 0.7 across folds) are unchanged; the other figures moved
by under half a percentage point.
