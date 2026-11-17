# sw-qsor

**Quantitative structure–odour relationships: predicting what a molecule smells like, and
testing such predictions honestly.**

Nothing is built yet. This file states what the project is and, more importantly, how it will
evaluate itself, written before any code exists so the rules cannot be adjusted to fit a result.

## What it is

Given a molecule's structure, predict its perceptual properties: what it smells like, how
strongly, and at what concentration a person can detect it. Software only, built on published
data.

Two halves, and the second is the point:

1. **Models.** Structure in, percept out.
2. **Benchmarks.** Evaluation harnesses that make it hard to fool yourself. Accuracy figures in
   this area depend entirely on how the data was split, and the splits reported are often not
   the hard ones.

## Scope: prediction, not design

The project predicts the perceptual properties of molecules that already exist and have been
characterised. It does not generate novel molecules to meet a target odour profile.

That line is drawn deliberately rather than by accident of what has been built so far.
Predicting what a known compound smells like is a question about published measurements.
Generating new compounds with specified properties is molecular design, which carries
obligations this project is not set up to meet — and the two get conflated easily once a model
that scores structures exists, because running it backwards is a small technical step and a
large change in what the project is.

If generation ever becomes interesting here, it starts with a fresh decision and its own
document, not by extending a scoring model.

[sw-enose](https://github.com/aptitudetechnology/sw-enose) is a biomimetic nose: air to mucus by
Henry's law, mucus to receptor activation by measured EC50 values, then the fly's own processing
stages (convergence, divisive normalisation, sparse expansion). Testing whether those biological
stages earn their place is what that project exists to do.

sw-qsor skips all of it and maps structure directly to percept. That is a different claim about
how olfaction should be modelled, so it gets a different repository rather than quietly becoming
a second architecture inside the first.

It also starts from something sw-enose established the hard way. Three independent routes to
predicting *receptor* responses were tested there and all failed: fitting the measured EC50s of
related compounds, physics-based QM/MM binding energies, and statistical models over 24,735
published receptor–odorant pairs. Receptor-level prediction is closed on current public data.
Perceptual prediction is a different target with more data behind it, and appears tractable —
which is the bet this project makes.

## The rules, fixed now

These come from mistakes made in sw-enose, each of which produced a convincing wrong answer
before it was caught.

1. **Count distinct molecules, never rows.** A dataset of "24,735 pairs" turned out to rest on
   180 molecules in 37 scaffold groups. The molecule count is the sample size for anything
   structural; the row count is decoration.
2. **Establish the floor before building the model.** Measure what can be predicted *without*
   structure — from label frequency, from dataset quirks, from whatever is free. In sw-enose a
   chemistry-free baseline reached AUC 0.75, so a model scoring 0.80 would have looked like a
   success while having learned nothing about molecules. Every result is reported as the margin
   over its matched floor.
3. **Hold out scaffolds, not just molecules.** A random split puts close analogues on both
   sides, and the model wins by recognising a neighbour. Scaffold-held-out is the honest test.
4. **Fix the bar before the run,** in writing, including what counts as failure.
5. **Negative results are results.** They are recorded with the same care as positive ones, in
   the same files. Most of what sw-enose established is negative, and it is the more useful half.
6. **Provenance on every number.** Measured or synthetic, with a source. Refuse rather than
   guess; a model that invents a value is worse than one that declines.

## Status

Nothing implemented. First milestone in `plan.md`: survey the public structure–percept datasets
and count what is actually in them.

## Licence

MIT.
