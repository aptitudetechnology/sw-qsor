# Prompt 1: public structure–odour datasets and how they have been evaluated

Paste everything below the line into a deep-research tool. Use the strongest model you have.

---

## Role

You are a careful analyst surveying public datasets and published benchmarks in olfactory
cheminformatics. The aim is to establish what data exists, how large it actually is, and how
honestly it has been evaluated. Report what you find with sources; say plainly when something
is not found. Do not argue for a conclusion.

## Context

We are building an open-source, software-only project that predicts the perceptual properties
of a molecule (what it smells like, its detection threshold) from its structure, and that
benchmarks such predictions. We need to know, before writing any model, whether public data is
large and diverse enough to support the task.

One constraint governs everything we ask for. **The sample size for structure-based learning is
the number of distinct molecules, not the number of rows.** In a related project a dataset
advertised as 24,735 measurements turned out to hold 180 distinct molecules in 37 Bemis-Murcko
scaffold groups, 92 of them sharing a single acyclic group, and modelling failed for exactly
that reason. So wherever a size is reported, we need the molecule count, and ideally the
scaffold count, not the record count.

Datasets we believe exist, named from memory and possibly wrong. Correct us where we are wrong,
and add any we have missed:

- The DREAM Olfaction Prediction Challenge data (associated with Keller and Vosshall)
- The GoodScents and Leffingwell flavour and fragrance descriptor collections
- Dravnieks, *Atlas of Odor Character Profiles*
- Nagata's odour thresholds by the triangle odour bag method
- Pyrfume, which we believe aggregates several of the above
- Arctander's perfume and flavour materials

"Not found in these searches" means a gap in the search, not evidence that something does not
exist.

## Find

1. **Every public dataset that links molecular structure to human olfactory perception.** For
   each: its name, who produced it, where it is hosted (a stable URL), its licence and any
   usage restriction, and whether it can be downloaded without registration.

2. **The real size of each**, in this order of preference: distinct molecules with both a
   structure and at least one perceptual label; distinct molecules overall; then record count.
   State which you are reporting. If the molecule count is not stated anywhere, say so rather
   than inferring it from a record count.

3. **What form the labels take** in each: free-text descriptors, fixed descriptor vocabularies,
   intensity or pleasantness ratings, detection thresholds in air, or something else. How many
   distinct labels, and whether per-subject responses are available or only group means.

4. **Whether structures are included** as SMILES, InChI or InChIKey, or whether only names and
   CAS numbers are given, which would force a separate structure lookup.

5. **Overlap between the datasets.** Which are derived from or aggregate which others, and any
   published statement about how much molecular overlap exists between them. Two sources that
   share most of their molecules cannot be used as independent training and test sets.

6. **Published models and benchmarks on these data.** For each significant piece of work: the
   reference, the dataset used, the number of distinct molecules, the reported performance and
   on what metric, and critically **how the data was split** — random split, held-out molecules,
   scaffold split, or held-out subjects. State explicitly whether a scaffold or
   structure-aware split was used, because a random split over a small molecular set measures
   something much easier than generalisation to a new molecule.

7. **Whether any baseline or chance level is reported** in those papers: what a model scores
   using no structural information at all (label frequency, mean rating, nearest neighbour on
   a non-structural feature). If papers report no such baseline, say so, as that is itself
   informative.

8. **Code and reproducibility**: which of these works published code, under what licence, and
   whether the published results have been independently reproduced.

9. **Any dataset of odour detection thresholds in air** with more than about 200 compounds,
   with its measurement method, since threshold prediction is a separate target from quality
   prediction.

## Rules

1. Rank sources by strength of evidence: the dataset's own documentation and primary papers
   first, then reviews, then everything else. **Anything carrying a conclusion needs a DOI, a
   PMID or a stable URL, and you must have opened it.** Say whether you read the full text,
   only an abstract, or only a landing page.
2. Copy titles, authors, year and venue from the record, never from memory. Quote the sentence
   or table cell that gives each number, **as printed**; do not convert or round.
3. Keep stated figures apart from your own inferences, and label which is which. If a molecule
   count is your inference from a record count, mark it clearly as such.
4. Where a licence is unclear, say "not stated" rather than assuming it is open.
5. **Do not stretch.** If a dataset cannot be found or accessed, write "not found in these
   searches" and move on.

## Output

1. **Direct answer** (120 words): which datasets, if any, hold at least 1,000 distinct
   molecules with perceptual labels and permissive licences; and whether published work on them
   has used structure-aware splits.
2. **Dataset table:** name | producer | URL | licence | distinct molecules (stated or inferred,
   say which) | scaffold count if stated | label type | number of labels | structures included?
   | per-subject data?
3. **Overlap notes:** which datasets derive from or duplicate which others, with sources.
4. **Benchmark table:** reference | dataset | distinct molecules | task | metric and score |
   split type | non-structural baseline reported? | code available?
5. **Threshold datasets:** name | compounds | method | units | source.
6. **Gaps:** what could not be established, and what would settle it.
7. **Verification table:** identifier | title | first author | year | full text, abstract or
   landing page only.
