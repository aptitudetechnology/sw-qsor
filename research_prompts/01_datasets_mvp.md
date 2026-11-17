# Prompt 1 (MVP): how big are the public structure–odour datasets?

The short version, to run first. `01_datasets_full.md` asks the same question plus benchmarks,
splits, baselines and reproducibility; run that only once this one has identified which
datasets are worth the effort.

Paste everything below the line into a deep-research tool.

---

Find out how many **distinct molecules** are in each public dataset that links molecular
structure to human smell perception, and whether each can be freely downloaded.

Only the molecule count matters, not the number of rows or ratings. A dataset with 50,000
ratings of 300 molecules is a 300-molecule dataset for our purposes, and we have been misled by
exactly that before. Where a source gives only a record count, say so rather than inferring a
molecule count from it.

Check at least these, and add any others you find. The names are from memory and may be wrong,
so correct them:

- DREAM Olfaction Prediction Challenge (Keller and Vosshall)
- GoodScents
- Leffingwell
- Dravnieks, *Atlas of Odor Character Profiles*
- Pyrfume
- Nagata odour thresholds (triangle odour bag method)
- Arctander, perfume and flavour materials

For each, report:

1. **Distinct molecules** with a structure and at least one perceptual label. Say whether the
   figure is stated by the source or inferred by you.
2. **Where to download it** — a working URL.
3. **Licence**, or "not stated". Do not assume open access.
4. **What the labels are**: odour descriptors, ratings, detection thresholds, or something else.
5. **Whether structures are included** (SMILES, InChI or InChIKey) or only names and CAS
   numbers.

Rules: open every source you cite and give its URL or DOI. Quote figures as printed. If a
dataset cannot be found or accessed, write "not found" and move on rather than estimating.

Output one table with those five columns plus the dataset name, then three sentences saying
which datasets hold at least 1,000 distinct molecules under a permissive licence.
