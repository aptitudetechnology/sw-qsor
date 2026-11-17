"""Download files with a provenance record.

Every fetched file is recorded with its source URL, the upstream commit it was pinned to, its
SHA-256, the retrieval date, and the licence status of the archive it came from. Licence status
is observed, not assumed: an archive with no ``LICENSE`` file of its own is recorded as
``"not stated"``, never as inheriting its repository's licence.

Written for sw-qsor, in the shape of a project-independent utility: no global state, no paths
other than those passed in.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROVENANCE_FILENAME = "provenance.json"


@dataclass(frozen=True)
class FileRecord:
    """One downloaded file."""

    path: str  # relative to the archive directory
    url: str
    sha256: str
    bytes: int


@dataclass
class ArchiveRecord:
    """One upstream archive: a directory of files fetched from a single pinned source."""

    name: str
    source: str  # human-readable, e.g. "github.com/pyrfume/pyrfume-data/arctander_1960"
    commit: str
    retrieved: str  # ISO date
    licence: str  # e.g. "CC BY-NC 4.0 (archive LICENSE file)", or "not stated"
    licence_note: str = ""
    files: list[FileRecord] = field(default_factory=list)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, *, timeout: float = 120.0) -> FileRecord:
    """Download ``url`` to ``dest`` atomically and return its record.

    Raises ``FileNotFoundError`` on HTTP 404, so callers can distinguish "absent" from
    "failed".
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code == 404:
            raise FileNotFoundError(url) from e
        raise
    except urllib.error.URLError as e:  # file:// URLs report absence this way
        tmp.unlink(missing_ok=True)
        if isinstance(e.reason, FileNotFoundError):
            raise FileNotFoundError(url) from e
        raise
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    return FileRecord(path=dest.name, url=url, sha256=sha256_of(dest), bytes=dest.stat().st_size)


def try_download(url: str, dest: Path, **kw) -> FileRecord | None:
    """As :func:`download`, but return ``None`` if the URL does not exist (HTTP 404)."""
    try:
        return download(url, dest, **kw)
    except FileNotFoundError:
        return None


def today() -> str:
    return _dt.date.today().isoformat()


def write_provenance(archive_dir: Path, record: ArchiveRecord) -> Path:
    """Write ``provenance.json`` into ``archive_dir``."""
    path = Path(archive_dir) / PROVENANCE_FILENAME
    path.write_text(json.dumps(asdict(record), indent=2) + "\n")
    return path


def read_provenance(archive_dir: Path) -> ArchiveRecord:
    """Read ``provenance.json`` from ``archive_dir``."""
    d = json.loads((Path(archive_dir) / PROVENANCE_FILENAME).read_text())
    d["files"] = [FileRecord(**f) for f in d.get("files", [])]
    return ArchiveRecord(**d)


def verify(archive_dir: Path) -> list[str]:
    """Check every recorded file against its SHA-256. Returns a list of problems (empty = ok)."""
    archive_dir = Path(archive_dir)
    rec = read_provenance(archive_dir)
    problems = []
    for f in rec.files:
        p = archive_dir / f.path
        if not p.exists():
            problems.append(f"missing: {p}")
        elif sha256_of(p) != f.sha256:
            problems.append(f"checksum mismatch: {p}")
    return problems
