"""Pinned upstream sources for the datasets sw-qsor uses.

Each source is pinned to a commit, so a re-fetch gets the same bytes, or fails the checksum
check visibly. This module only *describes* sources; ``scripts/fetch_*.py`` downloads them,
and :mod:`sw_qsor.datasets` loads them from a directory the caller names.

Commits were the heads of their default branches when first pinned on 2026-11-17.
"""

from __future__ import annotations

from dataclasses import dataclass

RAW = "https://raw.githubusercontent.com"


@dataclass(frozen=True)
class Source:
    name: str  # local directory name under the data root
    repo: str  # "owner/repo"
    commit: str
    subdir: str  # path inside the repo
    files: tuple[str, ...]  # files to fetch from subdir
    repo_licence: str  # what the repository as a whole states
    note: str = ""

    def url(self, filename: str) -> str:
        sub = f"{self.subdir}/" if self.subdir else ""
        return f"{RAW}/{self.repo}/{self.commit}/{sub}{filename}"

    @property
    def label(self) -> str:
        return f"github.com/{self.repo}/{self.subdir}"


OPENPOM_COMMIT = "5434c4b7badebca8e7d180aa34d2f787402121ee"
PYRFUME_DATA_COMMIT = "8054ea98ed675005ec10e67359902f500e4911b0"

GSLF = Source(
    name="gslf",
    repo="BioMachineLearning/openpom",
    commit=OPENPOM_COMMIT,
    subdir="openpom/data/curated_datasets",
    files=("curated_GS_LF_merged_4983.csv",),
    repo_licence="MIT",
    note=(
        "Repository is MIT; no licence file accompanies the dataset itself. The underlying "
        "GoodScents and Leffingwell databases are commercial in origin and no stated term "
        "covers their redistribution."
    ),
)

# Processed files only. Raw spreadsheets, upstream processing code, and Mayhew's large
# supplementary files (S1/S3/physics, ~53 MB) are not needed for milestone 1.
_PYRFUME_FILES = {
    "arctander_1960": ("molecules.csv", "stimuli.csv", "behavior_1.csv", "behavior_1_sparse.csv",
                       "behavior_2.csv"),
    "mayhew_2022": ("molecules.csv", "stimuli.csv", "behavior_1.csv", "behavior_2.csv"),
    "keller_2016": ("molecules.csv", "stimuli.csv", "subjects.csv", "behavior.csv"),
    "dravnieks_1985": ("molecules.csv", "stimuli.csv", "behavior_1.csv", "behavior_2.csv"),
    "abraham_2012": ("molecules.csv", "stimuli.csv", "behavior.csv"),
}

PYRFUME = {
    name: Source(
        name=name,
        repo="pyrfume/pyrfume-data",
        commit=PYRFUME_DATA_COMMIT,
        subdir=name,
        files=files + ("manifest.toml",),
        repo_licence="MIT",
    )
    for name, files in _PYRFUME_FILES.items()
}


def fetch(source: Source, data_root, *, log=print):
    """Download ``source`` into ``data_root/source.name`` and write its ``provenance.json``.

    Also tries ``LICENSE`` beside the files: if present it is downloaded and its first line
    recorded; if absent the licence is recorded as "not stated", whatever the repository says.
    """
    from pathlib import Path

    from sw_qsor import fetch as f

    dest = Path(data_root) / source.name
    records = []
    for name in source.files:
        log(f"  {source.name}/{name}")
        records.append(f.download(source.url(name), dest / name))

    lic = f.try_download(source.url("LICENSE"), dest / "LICENSE")
    if lic is None:
        licence = "not stated"
        note = f"No LICENSE file in the archive; repository licence is {source.repo_licence}."
    else:
        records.append(lic)
        first = next(
            (ln.strip() for ln in (dest / "LICENSE").read_text(errors="replace").splitlines()
             if ln.strip()),
            "",
        )
        licence = f"archive LICENSE file: {first}"
        note = f"Repository licence ({source.repo_licence}) is overridden by the archive's own."
    if source.note:
        note = f"{note} {source.note}"

    rec = f.ArchiveRecord(
        name=source.name, source=source.label, commit=source.commit, retrieved=f.today(),
        licence=licence, licence_note=note, files=records,
    )
    f.write_provenance(dest, rec)
    log(f"  -> {dest}  licence: {licence}")
    return rec
