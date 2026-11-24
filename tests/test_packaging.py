"""Packaging (pre-release test 2026-11-19, F16 and F20): the licence as an SPDX expression with its file (the table form and
the licence classifier are deprecated, unsupported by setuptools from 2027-02-18), project links for PyPI, and a source
distribution without a test suite that cannot run outside the repository."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tomllib = pytest.importorskip("tomllib")


def _project():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def test_the_licence_is_an_spdx_expression_with_its_file():
    doc = _project()
    assert doc["project"]["license"] == "Apache-2.0"
    assert doc["project"]["license-files"] == ["LICENSE"]
    assert not [c for c in doc["project"]["classifiers"] if c.startswith("License ::")]
    req = doc["build-system"]["requires"][0]
    assert req.startswith("setuptools>=") and int(req.split(">=")[1].split(".")[0]) >= 77


def test_pypi_shows_source_issues_and_changelog_links():
    urls = _project()["project"]["urls"]
    assert {"Source", "Issues", "Changelog"} <= set(urls)


def test_the_source_distribution_leaves_out_tests_and_evaluation():
    text = open(os.path.join(ROOT, "MANIFEST.in"), encoding="utf-8").read().split()
    assert "prune" in text and "tests" in text and "evaluation" in text
