"""Offline tests for download + provenance, using file:// URLs in place of GitHub."""

import json

import pytest

from sw_qsor import fetch as f
from sw_qsor import sources
from sw_qsor.sources import Source


def test_download_records_checksum_and_size(tmp_path):
    src = tmp_path / "src.txt"
    src.write_bytes(b"hello\n")
    rec = f.download(src.as_uri(), tmp_path / "out" / "x.txt")
    assert (tmp_path / "out" / "x.txt").read_bytes() == b"hello\n"
    assert rec.bytes == 6
    assert rec.sha256 == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    assert not list((tmp_path / "out").glob("*.part"))


def test_missing_url_is_file_not_found_and_leaves_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        f.download((tmp_path / "absent").as_uri(), tmp_path / "out" / "x")
    assert f.try_download((tmp_path / "absent").as_uri(), tmp_path / "out" / "x") is None
    assert not list((tmp_path / "out").iterdir())


def _fake_repo(tmp_path, monkeypatch, with_licence):
    """Lay out tmp/owner/repo/<commit>/sub/ and point sources.RAW at it."""
    d = tmp_path / "remote" / "o" / "r" / "c0ffee" / "sub"
    d.mkdir(parents=True)
    (d / "a.csv").write_text("x\n1\n")
    if with_licence:
        (d / "LICENSE").write_text("\nSome Licence 1.0\nbody\n")
    monkeypatch.setattr(sources, "RAW", (tmp_path / "remote").as_uri())
    return Source(name="arch", repo="o/r", commit="c0ffee", subdir="sub", files=("a.csv",),
                  repo_licence="MIT")


def test_absent_archive_licence_is_not_stated_not_repo_licence(tmp_path, monkeypatch):
    src = _fake_repo(tmp_path, monkeypatch, with_licence=False)
    rec = sources.fetch(src, tmp_path / "data", log=lambda *_: None)
    assert rec.licence == "not stated"
    assert "MIT" in rec.licence_note
    assert [r.path for r in rec.files] == ["a.csv"]


def test_archive_licence_file_is_recorded(tmp_path, monkeypatch):
    src = _fake_repo(tmp_path, monkeypatch, with_licence=True)
    rec = sources.fetch(src, tmp_path / "data", log=lambda *_: None)
    assert rec.licence == "archive LICENSE file: Some Licence 1.0"
    assert {r.path for r in rec.files} == {"a.csv", "LICENSE"}


def test_provenance_roundtrip_and_verify_detects_tampering(tmp_path, monkeypatch):
    src = _fake_repo(tmp_path, monkeypatch, with_licence=False)
    arch = tmp_path / "data" / "arch"
    rec = sources.fetch(src, tmp_path / "data", log=lambda *_: None)
    assert f.read_provenance(arch) == rec
    assert json.loads((arch / "provenance.json").read_text())["commit"] == "c0ffee"
    assert f.verify(arch) == []
    (arch / "a.csv").write_text("tampered\n")
    assert f.verify(arch) == [f"checksum mismatch: {arch / 'a.csv'}"]
    (arch / "a.csv").unlink()
    assert f.verify(arch) == [f"missing: {arch / 'a.csv'}"]


def test_pinned_sources_are_pinned_to_full_commits():
    for s in [sources.GSLF, *sources.PYRFUME.values()]:
        assert len(s.commit) == 40 and s.commit in s.url("x")
        assert "manifest.toml" in s.files or s is sources.GSLF
