"""Fetch selected Pyrfume archives into data/<archive>/.

Usage: python scripts/fetch_pyrfume.py [--data-dir DIR] [ARCHIVE ...]

With no ARCHIVE names, fetches all archives pinned in sw_qsor.sources.PYRFUME. Each archive is
checked for its own LICENSE file; absence is recorded as "not stated", not as the repository's
MIT licence.
"""

import argparse
from pathlib import Path

from sw_qsor.sources import PYRFUME, fetch

REPO_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("archives", nargs="*", metavar="ARCHIVE", help=", ".join(sorted(PYRFUME)))
    args = ap.parse_args()
    unknown = sorted(set(args.archives) - set(PYRFUME))
    if unknown:
        ap.error(f"unknown archive(s): {', '.join(unknown)}")
    for name in args.archives or sorted(PYRFUME):
        fetch(PYRFUME[name], args.data_dir)
