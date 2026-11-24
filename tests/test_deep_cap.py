"""--deep reads raw payload bytes up to a per-payload cap (4 GiB). Past the cap it must SAY so, never stop silently.

Found by the held-out round of 2026-11-05 (stress files): a 6.15 GB safetensors file scanned with --deep read exactly the same
4.3 GB as a 4.7 GB one and the report was clean and complete. The cap is lowered here (to one read window) so no giant file
is needed.
"""
import json
import struct

from limnex_preflight.formats import binary

MB = 1024 * 1024


def _safetensors(path, payload_bytes):
    header = json.dumps({"w": {"dtype": "U8", "shape": [payload_bytes], "data_offsets": [0, payload_bytes]}}).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)) + header)
        fh.write(b"\0" * payload_bytes)


def _problems(res):
    return [n for n in res.notes if n.kind in ("skipped", "degraded", "error")]


def test_deep_scan_that_stops_at_its_cap_says_so(tmp_path, scan, monkeypatch):
    monkeypatch.setattr(binary.scan_stream_deep, "__defaults__", (binary._WINDOW,))
    p = tmp_path / "big.safetensors"
    _safetensors(p, 3 * binary._WINDOW)
    assert _problems(scan(p, deep=True)), "payload past the cap was silently not scanned"


def test_deep_scan_within_its_cap_has_no_note(tmp_path, scan, monkeypatch):
    monkeypatch.setattr(binary.scan_stream_deep, "__defaults__", (binary._WINDOW,))
    p = tmp_path / "small.safetensors"
    _safetensors(p, binary._WINDOW // 2)
    assert _problems(scan(p, deep=True)) == []
