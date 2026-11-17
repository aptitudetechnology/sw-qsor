# Dataset survey (milestone 1, literature part)

Answers `research_prompts/01_datasets_mvp.md`. Run on 17 Nov 2026 by Claude inside the project
session, using web search plus direct retrieval of the data files and the GitHub API, not by a
deep-research tool. **Every molecule count below marked "counted" was obtained by downloading
the file and counting it here**, not read from a paper. Counts marked "reported" come from a
source and were not independently checked.

## Direct answer

Yes. The merged **GoodScents + Leffingwell** descriptor set, distributed as
`curated_GS_LF_merged_4983.csv` in the MIT-licensed OpenPOM repository, holds **4,983 distinct
molecules in 556 Bemis-Murcko scaffold groups, with 112 of its 138 odour descriptors carried by
50 or more molecules**. That clears every condition of the decision rule in `plan.md` by a wide
margin, and all 4,983 SMILES parse with RDKit. **Pyrfume** is the other major resource: 60
archives aggregating most of the published datasets, with structures already resolved to CIDs
and SMILES. Two cautions follow below: the scaffold distribution is lopsided, and Pyrfume's
MIT repository licence does not apply uniformly to the data inside it.

## Dataset table

"Counted" = downloaded and counted in this session. Molecule counts are distinct molecules.

| Dataset | Where | Licence | Molecules | Labels | Structures |
|---|---|---|---|---|---|
| **GS+LF merged** (OpenPOM `curated_GS_LF_merged_4983.csv`) | github.com/BioMachineLearning/openpom | MIT (repo) | **4,983 counted**, 556 scaffolds counted | 138 binary odour descriptors; 112 on >=50 molecules | SMILES, all parse |
| GoodScents (via Pyrfume) | pyrfume-data/goodscents | repo MIT, no archive LICENSE | 4,565 counted | odour descriptors | CID + SMILES |
| Leffingwell (via Pyrfume) | pyrfume-data/leffingwell | repo MIT, no archive LICENSE | 3,522 counted | free-text descriptors | CID + SMILES |
| Arctander 1960 (via Pyrfume) | pyrfume-data/arctander_1960 | **CC BY-NC 4.0** (own LICENSE file) | 2,751 counted | descriptors | CID + SMILES |
| Mayhew 2022 | pyrfume-data/mayhew_2022 | repo MIT, no archive LICENSE | 1,865 counted | perceptual | in archive |
| Snitz 2013 | pyrfume-data/snitz_2013 | repo MIT, no archive LICENSE | 1,323 counted | similarity ratings | CID + SMILES |
| Sigma-Aldrich 2014 | pyrfume-data/sigma_2014 | repo MIT, no archive LICENSE | 867 counted | descriptors | CID + SMILES |
| Keller & Vosshall 2016 (DREAM) | pyrfume-data/keller_2016 | repo MIT, no archive LICENSE | 481 counted | 21 scales, **per-subject** (49 subjects) | CID + SMILES |
| Abraham 2012 | pyrfume-data/abraham_2012 | repo MIT, no archive LICENSE | 268 counted | **detection thresholds**, log(1/ODT), with Abraham solvation descriptors | CID + SMILES |
| Dravnieks 1985 | pyrfume-data/dravnieks_1985 | repo MIT, no archive LICENSE | 138 counted | ~146 character descriptors | CID + SMILES |
| Bushdid 2014 | pyrfume-data/bushdid_2014 | repo MIT, no archive LICENSE | 128 counted | discrimination | CID + SMILES |
| Nagata thresholds | not a Pyrfume archive under that name | Japan Ministry of the Environment PDF | ~220 reported, 10 transcribed in sw-enose | detection thresholds, ppm | names only |

The full Pyrfume archive is 60 directories, including `abraham_2012 ahmed_2018 arctander_1960
aromadb arshamian_2022 bolding_2018 burton_2022 bushdid_2014 chae_2019 dravnieks_1985 flavordb
flavornet foodb goodscents haddad_2008 ifra_2019 keller_2012 keller_2016 leffingwell
mainland_2015 mayhew_2022 nakayama_2022 prestwick ravia_2020 scott_2014 sharma_2021a/b
sigma_2014 snitz_2013/2019 soh_2013 superscent weiss_2012 yu_2015` and others. Archives follow a
common layout (`molecules.csv` with CID, MolecularWeight, IsomericSMILES, IUPACName, name;
plus `behavior*.csv`, `stimuli.csv`, `manifest.toml`).

## Two cautions, both load-bearing

**1. The MIT licence on `pyrfume/pyrfume-data` does not cover every archive.** Arctander 1960
ships its own `LICENSE`: Creative Commons Attribution-**NonCommercial** 4.0. Of the eight other
archives checked, none carries its own LICENSE file, which leaves the repository licence as the
only stated term but is not an explicit grant for the data. Before using any archive, check for
a `LICENSE` beside it, and treat the absence of one as "not stated" rather than as MIT.
Unresolved here: GoodScents and Leffingwell are commercial databases in origin, and no
archive-level licence states the terms under which their contents are redistributed.

**2. Scaffold splitting will not work naively on GS+LF.** Of its 4,983 molecules, **2,236
(45%) are acyclic and therefore collapse into a single empty Bemis-Murcko scaffold**. Only 69
scaffold groups hold five or more molecules. A scaffold-held-out split would put nearly half
the dataset in one group. This is a milder form of sw-enose's failure case (92 of 180 molecules
in one group), but it means `plan.md`'s rule 3 needs a different grouping for acyclics —
fingerprint clustering, or carbon-skeleton grouping — rather than Murcko scaffolds alone.

## Label balance in GS+LF (counted)

138 descriptors, mean 4.9 per molecule. Most common: fruity 1,902, green 1,446, sweet 1,429,
floral 1,100, herbal 756. Rarest: chamomile 31, bergamot 32, orangeflower 33. **112 of 138
descriptors are carried by at least 50 molecules**, so label scarcity is less of a problem than
expected — the imbalance is in frequency, not in a long tail of unusable labels.

Known bias, reported not verified: the set consists of perfumery materials and is therefore
skewed away from malodorous compounds.

## Against `plan.md`'s decision rule

| Condition | Required | GS+LF | Verdict |
|---|---|---|---|
| Distinct molecules | >= 1,000 | 4,983 | PASS |
| Scaffold groups | >= 100 | 556 | PASS (but see caution 2) |
| A label on >= 50 molecules | >= 1 | 112 labels | PASS |

**Milestone 1's literature question is answered: the data exists.** What remains of milestone 1
is the chemistry-free floor, which has to be computed locally, and the overlap check between
GS+LF, Arctander and Mayhew by InChIKey.

## Not established

- Whether the 4,983 GS+LF molecules overlap the 2,751 Arctander or 1,865 Mayhew molecules, and
  by how much. Must be computed by InChIKey before any of them is treated as an independent
  test set.
- The redistribution terms for GoodScents and Leffingwell content specifically.
- Whether published work on GS+LF used scaffold-aware splits, and whether any reported a
  non-structural baseline. That is `01_datasets_full.md`, now worth running against a short
  list rather than an open field.
- The Nagata table was not re-fetched here; sw-enose holds 10 transcribed sulfur values and the
  source URL.

## Sources

Retrieved 17 Nov 2026. GitHub API and raw file retrieval for all counts.

- OpenPOM: https://github.com/BioMachineLearning/openpom (MIT), dataset file
  `openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv`
- Pyrfume data: https://github.com/pyrfume/pyrfume-data (MIT), package
  https://github.com/pyrfume/pyrfume
- Pyrfume paper: Scientific Data (2024), https://www.nature.com/articles/s41597-024-04051-z —
  **not read**, the publisher redirected to a login page. A preprint exists at
  https://www.biorxiv.org/content/10.1101/2022.09.08.507170v1.full, also not read.
- Lee et al., "A principal odor map unifies diverse tasks in olfactory perception",
  PMID 37651511 — **abstract and secondary descriptions only**, full text not read.
- Keller & Vosshall DREAM challenge: Synapse syn2811262 and dream-olfaction.github.io,
  reported in search results, **not opened**.
