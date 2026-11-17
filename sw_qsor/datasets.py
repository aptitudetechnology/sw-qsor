"""Loaders for fetched datasets.

Every loader takes the directory to read from; none looks anywhere it was not told to. Fetch
first with ``scripts/fetch_gslf.py`` / ``scripts/fetch_pyrfume.py``.

Licences, as observed on retrieval (2026-11-17); ``provenance.json`` in each archive directory
is the authoritative record:

- GS+LF (OpenPOM ``curated_GS_LF_merged_4983.csv``): repository MIT, no file-level licence;
  GoodScents and Leffingwell are commercial in origin.
- Pyrfume ``arctander_1960``: CC BY-NC 4.0, from the archive's own LICENSE file.
- Pyrfume ``mayhew_2022``, ``keller_2016``, ``dravnieks_1985``, ``abraham_2012``: no archive
  LICENSE; repository MIT. Recorded as "not stated".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GSLF_FILE = "curated_GS_LF_merged_4983.csv"
_GSLF_NON_LABEL = ("nonStereoSMILES", "descriptors")


def load_gslf(archive_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the merged GoodScents + Leffingwell descriptor set.

    Source: github.com/BioMachineLearning/openpom at commit 5434c4b, retrieved 2026-11-17.
    Licence: repository MIT; no licence on the file itself (see module docstring).

    Returns ``(frame, labels)``: ``frame`` has a ``smiles`` column (non-stereo SMILES, as
    distributed), the original ``descriptors`` string, and one 0/1 integer column per label;
    ``labels`` lists the 138 label columns in file order.
    """
    df = pd.read_csv(Path(archive_dir) / GSLF_FILE)
    labels = [c for c in df.columns if c not in _GSLF_NON_LABEL]
    df = df.rename(columns={"nonStereoSMILES": "smiles"})
    df[labels] = df[labels].astype("int8")
    return df, labels


def load_pyrfume_molecules(archive_dir: str | Path) -> pd.DataFrame:
    """Load a Pyrfume archive's ``molecules.csv``.

    Source: github.com/pyrfume/pyrfume-data at commit 8054ea9, retrieved 2026-11-17.
    Licence: archive-specific; see module docstring and the archive's ``provenance.json``.

    Index ``CID``: negative values are Pyrfume-local identifiers for molecules with no PubChem
    entry, not PubChem CIDs. Guaranteed columns: ``smiles``, ``name``, ``MolecularWeight``,
    and ``smiles_source``, naming the upstream column ``smiles`` came from.

    Archives do not share one schema. Most have ``IsomericSMILES`` and ``name``;
    ``keller_2016`` has only ``CanonicalSMILES`` (no stereochemistry) and ``OdorName``.
    Isomeric SMILES is preferred where present. Other upstream columns are kept as they are.
    """
    df = pd.read_csv(Path(archive_dir) / "molecules.csv")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    for col in ("IsomericSMILES", "CanonicalSMILES"):
        if col in df.columns:
            df = df.rename(columns={col: "smiles"})
            df["smiles_source"] = col
            break
    else:
        raise ValueError(f"{archive_dir}: molecules.csv has no SMILES column")
    if "name" not in df.columns and "OdorName" in df.columns:
        df = df.rename(columns={"OdorName": "name"})
    return df.set_index("CID")


def load_openpom_source(archive_dir: str | Path, filename: str) -> tuple[pd.DataFrame, list[str]]:
    """Load one of OpenPOM's per-source files (``curated_goodcents.csv``,
    ``curated_leffingwell.csv``) from the pinned commit, as :func:`load_gslf` does.

    These carry ``IsomericSMILES`` (so stereoisomers are separate rows) and a descriptor
    string column. Leffingwell's vocabulary is GS+LF's minus ``cortex`` (137 labels).
    Returns ``(frame, labels)`` with a ``smiles`` column and 0/1 int8 label columns.
    """
    df = pd.read_csv(Path(archive_dir) / filename)
    desc = [c for c in df.columns if c.startswith("Updated_Desc")]
    labels = [c for c in df.columns if c not in ("IsomericSMILES", *desc)]
    df = df.rename(columns={"IsomericSMILES": "smiles"})
    df[labels] = df[labels].astype("int8")
    return df, labels


def load_arctander_labels(archive_dir: str | Path) -> tuple[pd.DataFrame, list[str], dict]:
    """Arctander 1960 binary labels per molecule (Pyrfume ``behavior_1.csv``).

    Licence CC BY-NC 4.0 (archive LICENSE file). Stimuli are mapped to CIDs via
    ``stimuli.csv`` and to structures via ``molecules.csv``. Stimuli with CID 0 (structure
    never resolved) are dropped. Stimuli sharing a CID have their labels unioned. Molecules
    with **no label at all** are kept but flagged ``labelled=False``: their text descriptions
    show they are unextracted rather than odourless, so they are not negatives.

    Returns ``(frame indexed by CID with smiles, labelled, label columns; labels; counts)``.
    """
    d = Path(archive_dir)
    b = pd.read_csv(d / "behavior_1.csv")
    s = pd.read_csv(d / "stimuli.csv")[["Stimulus", "new_CID"]]
    m = load_pyrfume_molecules(d)
    labels = [c for c in b.columns if c != "Stimulus"]
    j = b.merge(s, on="Stimulus", how="left")
    counts = {"stimuli": len(b), "stimuli_without_structure": int((~j.new_CID.isin(m.index)).sum())}
    j = j[j.new_CID.isin(m.index)]
    per = j.groupby("new_CID")[labels].max().astype("int8")
    per.index.name = "CID"
    per.insert(0, "labelled", per[labels].sum(axis=1) > 0)
    per.insert(0, "smiles", m.loc[per.index, "smiles"])
    counts.update(molecules=len(per), unlabelled_molecules=int((~per.labelled).sum()))
    return per, labels, counts
