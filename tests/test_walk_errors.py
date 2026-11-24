"""Pre-release test 2026-11-19, finding F7: a folder that cannot be listed (no permission, a path too long for the system)
was skipped silently, so the scan said "no findings" and exited 0 even with --fail-on-incomplete. It is now an error note:
the scan exits 3 and the Action warns."""
import os

from limnex_preflight import cli
from limnex_preflight.scanner import scan_paths


def _tree(tmp_path):
    root = tmp_path / "models"
    (root / "locked").mkdir(parents=True)
    (root / "locked" / "config.json").write_text('{"out": "/home/alice/run"}', encoding="utf-8")
    (root / "open.json").write_text("{}", encoding="utf-8")
    return root


def _refuse(monkeypatch, name):
    real = os.scandir

    def scandir(path="."):
        if os.path.basename(os.fspath(path)) == name:
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real(path)
    monkeypatch.setattr(os, "scandir", scandir)


def test_a_folder_that_cannot_be_listed_is_an_error(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    _refuse(monkeypatch, "locked")
    res = scan_paths([str(root)])
    errors = [n for n in res.notes if n.kind == "error"]
    assert errors and "locked" in errors[0].file and "could not be listed" in errors[0].message, res.notes
    assert any(s.path.endswith("open.json") for s in res.files)        # the rest is still scanned


def test_the_cli_exits_3_for_it(tmp_path, monkeypatch, capsys):
    root = _tree(tmp_path)
    _refuse(monkeypatch, "locked")
    assert cli.main([str(root)]) == 3
    assert cli.main([str(root), "--fail-on-incomplete"]) == 3


def test_a_readable_tree_has_no_error(tmp_path):
    res = scan_paths([str(_tree(tmp_path))])
    assert not [n for n in res.notes if n.kind == "error"]
