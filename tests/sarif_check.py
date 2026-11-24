"""Offline structural check of the SARIF 2.1.0 subset this tool emits (no schema download, no extra dependency).

The official OASIS schema is NOT vendored: it is governed by OASIS IPR terms rather than an open-source licence. If you
have a local copy, set LIMNEX_SARIF_SCHEMA=/path/to/sarif-schema-2.1.0.json and tests/test_sarif.py validates against it too.
"""

import re

LEVELS = {"error", "warning", "note", "none"}
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def check_sarif(doc: dict) -> None:
    assert doc.get("version") == "2.1.0", "version"
    assert isinstance(doc.get("$schema"), str) and "sarif" in doc["$schema"]
    assert isinstance(doc.get("runs"), list) and len(doc["runs"]) == 1
    run = doc["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] and driver["version"] and driver["informationUri"]
    rules = driver["rules"]
    ids = [r["id"] for r in rules]
    assert len(ids) == len(set(ids)), "rule ids must be unique (one rule per finding type)"
    for r in rules:
        assert r["shortDescription"]["text"] and r["help"]["text"]
        assert r["defaultConfiguration"]["level"] in LEVELS
        assert float(r["properties"]["security-severity"]) >= 0
    for res in run["results"]:
        assert res["ruleId"] in ids and rules[res["ruleIndex"]]["id"] == res["ruleId"], "ruleIndex must point at ruleId"
        assert res["level"] in LEVELS
        assert isinstance(res["message"]["text"], str) and res["message"]["text"]
        assert not _CTRL.search(res["message"]["text"]), "control characters in message"
        assert res["partialFingerprints"] and all(res["partialFingerprints"].values())
        loc = res["locations"][0]["physicalLocation"]
        uri = loc["artifactLocation"]["uri"]
        assert uri and "\\" not in uri and not _CTRL.search(uri) and " " not in uri, f"uri not a clean URI: {uri!r}"
        assert not re.match(r"^[A-Za-z]:", uri), "drive letters must not appear in the URI"
        if loc["artifactLocation"].get("uriBaseId"):
            assert not uri.startswith(("/", "file:")), "relative URIs only when uriBaseId is set"
        region = loc["region"]
        assert isinstance(region["startLine"], int) and region["startLine"] >= 1
        if "byteOffset" in region:
            assert isinstance(region["byteOffset"], int) and region["byteOffset"] >= 0 and region["byteLength"] >= 1
        assert res["properties"]["maskedValue"] is not None
    inv = run["invocations"][0]
    assert isinstance(inv["executionSuccessful"], bool)
    for n in inv.get("toolExecutionNotifications", []):
        assert n["level"] in LEVELS and n["message"]["text"]
