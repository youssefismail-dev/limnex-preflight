"""Command-line edge cases from the pre-release test of 2026-11-19 (F10, F12, F13, F19)."""
import io
import json
import sys

import pytest

from limnex_preflight import cli


def _dir(tmp_path):
    d = tmp_path / "m"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"lr": 0.1}), encoding="utf-8")
    return d


# ---- F10: a path that does not exist is a usage error, whatever --no-error-exit says --------------------------------
def test_a_missing_path_is_a_usage_error(tmp_path, capsys):
    assert cli.main([str(tmp_path / "does-not-exist")]) == 2
    assert cli.main([str(tmp_path / "does-not-exist"), "--no-error-exit"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_an_existing_path_is_still_scanned(tmp_path):
    assert cli.main([str(_dir(tmp_path))]) == 0


# ---- F12: the ignore file's encoding --------------------------------------------------------------------------------
def test_an_ignore_file_with_a_utf8_byte_order_mark_is_read(tmp_path, capsys):
    ign = tmp_path / "bom.ign"
    ign.write_bytes(b"\xef\xbb\xbf# accepted findings\n")
    assert cli.main([str(_dir(tmp_path)), "--ignore-file", str(ign)]) == 0


def test_a_utf16_ignore_file_gives_a_plain_error_not_a_crash(tmp_path, capsys):
    ign = tmp_path / "u16.ign"
    ign.write_bytes("# accepted\n0123456789abcdef  # x\n".encode("utf-16"))
    assert cli.main([str(_dir(tmp_path)), "--ignore-file", str(ign)]) == 2
    err = capsys.readouterr().err
    assert "UTF-8" in err and "internal error" not in err and "Traceback" not in err


# ---- F13: reports and output ------------------------------------------------------------------------------------------
def test_two_reports_on_stdout_are_refused(tmp_path, capsys):
    assert cli.main([str(_dir(tmp_path)), "--json", "-", "--sarif", "-"]) == 2
    assert cli.main([str(_dir(tmp_path)), "--format", "json", "--sarif", "-"]) == 2


def test_a_closed_pipe_ends_quietly_with_the_scans_own_exit_code(tmp_path, monkeypatch, capsys):
    class Closed(io.StringIO):
        def write(self, s):
            raise BrokenPipeError(32, "Broken pipe")
    monkeypatch.setattr(sys, "stdout", Closed())
    assert cli.main([str(_dir(tmp_path))]) == 0
    err = capsys.readouterr().err
    assert "internal error" not in err and "Traceback" not in err


def test_the_summary_counts_only_files_that_were_scanned(tmp_path, capsys):
    d = _dir(tmp_path)
    (d / "blob.xyz").write_bytes(bytes(range(256)) * 8)          # an unknown binary format: not scanned
    cli.main([str(d)])
    out = capsys.readouterr().out
    assert "scanned 1 file(s)" in out and "1 not scanned" in out, out


# ---- F19: --max-depth must be at least 1 ------------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["0", "-1"])
def test_max_depth_below_one_is_refused(tmp_path, value):
    with pytest.raises(SystemExit) as e:
        cli.main([str(_dir(tmp_path)), "--max-depth", value])
    assert e.value.code == 2
