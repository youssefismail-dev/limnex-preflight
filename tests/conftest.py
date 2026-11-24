import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import fixtures as fx  # noqa: E402


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixtures")
    formats, unsafe = fx.build_all(str(out))
    return formats, unsafe


@pytest.fixture()
def scan():
    from limnex_preflight.context import ScanOptions
    from limnex_preflight.scanner import scan_paths

    def _scan(*paths, **opts):
        return scan_paths([str(p) for p in paths], ScanOptions(**opts))

    return _scan
