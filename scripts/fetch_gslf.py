"""Fetch the GS+LF merged descriptor set (OpenPOM) into data/gslf/, and its two per-source
files into data/openpom_goodscents/ and data/openpom_leffingwell/.

Usage: python scripts/fetch_gslf.py [--data-dir DIR]

Default DIR is ``data/`` at the repository root (gitignored). Writes provenance.json with
checksums, the pinned commit, the retrieval date and the observed licence.
"""

import argparse
from pathlib import Path

from sw_qsor.sources import GOODSCENTS, GSLF, LEFFINGWELL, fetch

REPO_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = ap.parse_args()
    for src in (GSLF, GOODSCENTS, LEFFINGWELL):
        fetch(src, args.data_dir)
