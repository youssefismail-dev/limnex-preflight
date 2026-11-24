import json
import os
import sys

import pytest

import action_fixtures as af
from limnex_preflight import __version__
from limnex_preflight.cli import main
from limnex_preflight.context import ScanOptions
from limnex_preflight.report import to_sarif
from limnex_preflight.scanner import scan_paths
from sarif_check import check_sarif


def sarif_of(*paths, base=None):
    return json.loads(to_sarif(scan_paths([str(p) for p in paths], ScanOptions()), __version__, base_dir=base))


def test_every_fixture_produces_structurally_valid_sarif(built):
    formats, unsafe = built
    doc = sarif_of(os.path.dirname(formats[0].path), os.path.dirname(unsafe[0].path))
    check_sarif(doc)
    run = doc["runs"][0]
    assert len(run["results"]) > 300
    levels = {r["level"] for r in run["results"]}
    assert levels == {"error", "warning", "note"}


def test_one_rule_per_finding_type_and_severity_maps_to_level(built):
    formats, unsafe = built
    doc = sarif_of(os.path.dirname(unsafe[0].path))
    run = doc["runs"][0]
    ids = [r["id"] for r in run["tool"]["driver"]["rules"]]
    assert len(ids) == len(set(ids))
    for res in run["results"]:
        sev = res["properties"]["severity"]
        assert res["level"] == {"critical": "error", "high": "error", "medium": "warning", "low": "note", "info": "note"}[sev]
    rule = next(r for r in run["tool"]["driver"]["rules"] if r["id"] == "unsafe-pickle-exec")
    assert rule["defaultConfiguration"]["level"] == "error" and float(rule["properties"]["security-severity"]) >= 9


def test_text_findings_carry_a_line_binary_findings_carry_a_byte_offset(tmp_path):
    (tmp_path / "c.txt").write_text("a\nb\ncontact fx.person@gmail.com\n", encoding="utf-8")
    (tmp_path / "evil.pkl").write_bytes(af.fx.pickle_proto4_system())
    (tmp_path / "bundle.zip").write_bytes(b"")
    import zipfile

    with zipfile.ZipFile(tmp_path / "bundle.zip", "w") as z:
        z.writestr("m.pkl", af.fx.pickle_proto4_system())
    doc = sarif_of(tmp_path, base=str(tmp_path))
    check_sarif(doc)
    by_uri = {}
    for r in doc["runs"][0]["results"]:
        by_uri.setdefault(r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"], []).append(r["locations"][0]["physicalLocation"]["region"])
    assert by_uri["c.txt"][0]["startLine"] == 3 and "byteOffset" not in by_uri["c.txt"][0]
    assert by_uri["evil.pkl"][0]["byteOffset"] == 16 and by_uri["evil.pkl"][0]["byteLength"] == 1
    # offsets inside an archive member are relative to the member, so they are NOT presented as offsets in the zip file
    assert "byteOffset" not in by_uri["bundle.zip"][0]


def test_uris_are_relative_forward_slash_and_percent_encoded(tmp_path):
    d = tmp_path / "sub dir" / "café"
    d.mkdir(parents=True)
    (d / "a b#c.json").write_text(json.dumps({"p": "/home/fx_zed_user/x"}), encoding="utf-8")
    doc = sarif_of(d, base=str(tmp_path))
    check_sarif(doc)
    uri = doc["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    assert uri == "sub%20dir/caf%C3%A9/a%20b%23c.json"


@pytest.mark.skipif(sys.platform == "win32", reason="newlines are not allowed in Windows file names")
def test_newline_in_a_file_name_is_percent_encoded_not_raw(tmp_path):
    (tmp_path / "x\ny.json").write_text(json.dumps({"p": "/home/fx_zed_user/x"}), encoding="utf-8")
    doc = sarif_of(tmp_path, base=str(tmp_path))
    check_sarif(doc)
    assert doc["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "x%0Ay.json"


def test_cli_format_sarif_to_stdout_and_to_file(tmp_path, capsys, monkeypatch):
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"p": "/home/fx_zed_user/x"}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["--format", "sarif", "c.json"]) == 1
    doc = json.loads(capsys.readouterr().out)  # stdout is the SARIF document only
    check_sarif(doc)
    assert main(["--format", "sarif", "--output", "out.sarif", "c.json"]) == 1
    printed = capsys.readouterr().out
    assert "username-in-path" in printed  # human report still printed
    check_sarif(json.loads((tmp_path / "out.sarif").read_text(encoding="utf-8")))


@pytest.mark.skipif(not os.environ.get("LIMNEX_SARIF_SCHEMA"), reason="set LIMNEX_SARIF_SCHEMA=/path/to/sarif-schema-2.1.0.json to also validate against the official OASIS schema")
def test_validates_against_the_official_schema_when_available(built):
    jsonschema = pytest.importorskip("jsonschema")
    with open(os.environ["LIMNEX_SARIF_SCHEMA"], encoding="utf-8") as fh:
        schema = json.load(fh)
    formats, unsafe = built
    jsonschema.validate(sarif_of(os.path.dirname(formats[0].path), os.path.dirname(unsafe[0].path)), schema)
