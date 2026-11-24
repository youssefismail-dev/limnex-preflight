"""SECURITY: SARIF goes to GitHub's Security tab, the summary and annotations are visible to everyone with read access.
Nothing but masked values may appear in any of them - not even when the raw secret is a KEY, a tensor name or a member name."""
import json
import os
import struct
import zipfile

import pytest

import action_fixtures as af
import fixtures as fx
from ci_helpers import run_action
from limnex_preflight import __version__
from limnex_preflight.context import ScanOptions
from limnex_preflight.report import to_human, to_json, to_sarif
from limnex_preflight.sanitize import scrub_display, scrub_text
from limnex_preflight.scanner import scan_paths


def every_planted_value():
    vals = set(af.secrets_that_must_never_appear())
    vals |= {p.value for p in fx.PLANTED if len(p.value) >= 6 and p.rule != "private-key-pem"}
    vals |= {p.text for p in fx.PLANTED if len(p.text) >= 12 and "\n" not in p.text}
    return sorted(vals)


def assert_no_raw(blob, what, extra=()):
    for needle in list(every_planted_value()) + list(extra):
        assert needle not in blob, f"unmasked value leaked into {what}: {needle[:4]}..."


def test_every_fixture_secret_is_masked_in_sarif_summary_annotations_and_log(built, tmp_path, monkeypatch):
    formats, unsafe = built
    folder = os.path.dirname([f for f in formats if f.kind == "planted"][0].path)
    ws = tmp_path / "ws"
    ws.mkdir()
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path=folder, fail_on="none", sarif_file="all.sarif")
    sarif = (ws / "all.sarif").read_text(encoding="utf-8")
    doc = json.loads(sarif)
    assert len(doc["runs"][0]["results"]) > 100, "the test must actually have findings to check"
    annotations = "\n".join(l for l in out.splitlines() if l.startswith("::error") or l.startswith("::warning") or l.startswith("::notice"))
    assert annotations, "annotations were expected"
    for what, blob in (("SARIF", sarif), ("job summary", summary), ("annotations", annotations), ("log", out)):
        assert_no_raw(blob, what)
    # and the library reports used by the CLI
    res = scan_paths([folder], ScanOptions())
    for what, blob in (("JSON", to_json(res, __version__, None)), ("human", to_human(res, __version__, verbose=True)), ("SARIF", to_sarif(res, __version__))):
        assert_no_raw(blob, what)


def test_masked_forms_are_what_is_reported(built, tmp_path, monkeypatch):
    formats, _ = built
    f = next(x for x in formats if x.format == "json" and x.kind == "planted")
    ws = tmp_path / "ws"
    ws.mkdir()
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path=f.path, fail_on="none", sarif_file="s.sarif")
    sarif = (ws / "s.sarif").read_text(encoding="utf-8")
    assert "AKIA...LE" in sarif and "AKIA...LE" in summary and "AKIA...LE" in out
    assert "hf_FA...45" in sarif


def build_key_borne_leaks(d):
    """The raw secret is a KEY / a name, not a value: it can reach a report through `location` or `file`."""
    d.mkdir(parents=True)
    aws, hf = af.AWS, fx.BY_LABEL["hf"].text
    gh = fx.BY_LABEL["github"].text
    (d / "keys.json").write_text(json.dumps({aws: 1, "nested": {hf: "/home/fx_keyborne_user/x"}}), encoding="utf-8")
    with zipfile.ZipFile(d / "bundle.zip", "w") as z:
        z.writestr(f"{aws}.json", json.dumps({"p": "/home/fx_member_user/x"}))
    header = json.dumps({gh: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "__metadata__": {"owner": "/home/fx_tensor_user/x"}}).encode()
    (d / "t.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")
    return [aws, hf, gh, "fx_keyborne_user", "fx_member_user", "fx_tensor_user"]


def test_secrets_used_as_keys_member_names_or_tensor_names_never_reach_a_report(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    raw = build_key_borne_leaks(ws / "m")
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path="m", fail_on="none", sarif_file="k.sarif")
    sarif = (ws / "k.sarif").read_text(encoding="utf-8")
    assert json.loads(sarif)["runs"][0]["results"], "findings expected"
    rules = {r["ruleId"] for r in json.loads(sarif)["runs"][0]["results"]}
    assert {"aws-access-key-id", "huggingface-token", "github-token"} <= rules
    res = scan_paths([str(ws / "m")], ScanOptions())
    blobs = {"SARIF": sarif, "summary": summary, "log": out, "JSON": to_json(res, __version__, None),
             "human": to_human(res, __version__, verbose=True)}
    for what, blob in blobs.items():
        for needle in raw:
            assert needle not in blob, f"{needle[:6]}... leaked into {what}"


def test_scrubbing_is_idempotent_and_leaves_ordinary_locations_alone():
    for loc in ("$.config.layers[0].name", "pickle offset 1234 (BINUNICODE)", "cell 3 (code) output 1 (stream stderr), line 2", "line 17"):
        assert scrub_text(loc) == loc
    dirty = '$["' + af.AWS + '"]'
    once = scrub_text(dirty)
    assert af.AWS not in once and scrub_text(once) == once
    assert scrub_display("models/a.zip!/" + af.AWS + ".bin") .startswith("models/a.zip!/")
    assert af.AWS not in scrub_display("models/a.zip!/" + af.AWS + ".bin")
    assert scrub_display("models/" + "home-dir-name/a.zip") == "models/home-dir-name/a.zip"  # the user's own path is kept


def test_a_parser_error_that_quotes_file_content_is_masked_in_every_report(tmp_path):
    """Review finding 2026-10-24: a structural finding's message and value carry text from the file (here a parser
    error that quotes the content). A damaged data file: a protocol-0 string whose closing quote is missing."""
    token = next(p.value for p in fx.PLANTED if p.rule == "huggingface-token")
    bad = tmp_path / "damaged.pkl"
    bad.write_bytes(b"S'" + token.encode() + b"\n.")
    res = scan_paths([str(bad)], ScanOptions())
    assert any(f.rule_id == "pickle-incomplete-analysis" for f in res.findings)
    for what, blob in (("json", to_json(res, __version__, None)), ("human", to_human(res, __version__, verbose=True)),
                       ("sarif", to_sarif(res, __version__))):
        assert token not in blob, f"raw token leaked into the {what} report"
