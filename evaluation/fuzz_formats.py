"""Mutation fuzzing of every format parser, starting from the BENIGN clean-twin fixtures only.

Each seed is a clean file written by `evaluation/fixtures.py` (`clean_items()`, no planted values, no unsafe pickle).
Random damage is applied (bit flips, overwritten bytes, large integers over length fields, inserted / deleted /
duplicated chunks, truncation; for zip-based formats also damage to one member inside an otherwise valid archive), and
every damaged file is scanned with the default options. A case FAILS when:

  * crash      an exception escapes `scan_paths` (the scanner promises a note, never an exception);
  * bug        a "scan failed (...)" note names an exception type that means a programming error rather than a
               malformed input (TypeError, AttributeError, KeyError, IndexError, NameError, RecursionError, ...);
  * hang       one scan takes longer than --timeout seconds (the worker is killed by faulthandler and restarted);
  * memory     one scan allocates more than --max-mb MB at its peak (tracemalloc), from a seed of a few KB;
  * clean-cut  a TRUNCATED binary file comes back with no finding and no error/degraded note, i.e. reported as clean
               and complete (rule: a damaged file is "incomplete", never "clean"). Text formats are exempt: a shorter
               text file is still a valid text file.

Usage:
  python evaluation/fuzz_formats.py --cases 300 --seed 20261024 --out evaluation/results/fuzz.md
  python evaluation/fuzz_formats.py --cases 20 --quick      # what tests/test_fuzz_smoke.py runs
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import platform
import random
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

BUG_TYPES = ("TypeError", "AttributeError", "KeyError", "IndexError", "NameError", "UnboundLocalError",
             "RecursionError", "AssertionError", "ZeroDivisionError", "MemoryError", "SystemError", "NotImplementedError")
TEXT_FORMATS = {"ipynb", "json", "yaml", "txt", "cfg"}
# Cuts the scanner cannot see by design, because those bytes are never parsed. They are reported in their own table,
# not as failures, and each one is a documented limit in the README ("What it cannot find"):
#   catboost      the CTR data after the declared model core is not parsed;
#   torch-legacy  the raw tensor bytes after the five pickles of a pre-1.6 PyTorch file are skipped (weights are skipped).
ZIP_FORMATS = {"torch-zip", "npz", "keras-v3", "zip-nested"}
MUTATIONS = ("flip", "bytes", "bigint", "insert", "delete", "dup", "truncate", "member")


# --------------------------------------------------------------------------------------
# Seeds and mutations
# --------------------------------------------------------------------------------------
def build_seeds(outdir: str, seed: int = 0) -> Dict[str, Tuple[str, bytes]]:
    """format name -> (extension, bytes of its clean twin). Formats whose writer library is missing are left out.

    Some writers fill fake tensor bytes with os.urandom; it is replaced by a seeded generator here, so the same --seed
    gives byte-identical files and the same report. Each format has its own stream, and the time and "made by" system
    of every zip entry are fixed (normalize_zip), so a seed that differs on another system changes only its own row
    (pre-release test 3, R6).
    """
    sys.path.insert(0, HERE)
    import fixtures as fx

    real_urandom = os.urandom
    seeds = {}
    try:
        os.makedirs(outdir, exist_ok=True)
        for fmt in fx.FORMATS:
            if not fx.have(fmt.needs):
                continue
            rnd = random.Random(f"{seed}:{fmt.name}")
            os.urandom = lambda n, r=rnd: bytes(r.getrandbits(8) for _ in range(n))
            path = os.path.join(outdir, f"seed_{fmt.name}{fmt.ext}")
            fmt.writer(path, fx.clean_items())
            with open(path, "rb") as fh:
                data = normalize_zip(fh.read())
            with open(path, "wb") as fh:
                fh.write(data)
            seeds[fmt.name] = (fmt.ext, data)
    finally:
        os.urandom = real_urandom
    return seeds


def normalize_zip(data: bytes) -> bytes:
    """A zip written by a library (NumPy's .npz) carries the current local time and the writer's system in every entry.
    Set both to fixed values in the local headers and the central directory; nothing else changes. Not a zip: unchanged."""
    if data[:4] != b"PK\x03\x04":
        return data
    try:
        infos = zipfile.ZipFile(io.BytesIO(data)).infolist()
    except zipfile.BadZipFile:
        return data
    b = bytearray(data)
    dos_time, dos_date = 0, (0 << 9) | (1 << 5) | 1          # 1980-01-01 00:00:00
    for info in infos:
        o = info.header_offset
        if b[o:o + 4] == b"PK\x03\x04":
            struct.pack_into("<HH", b, o + 10, dos_time, dos_date)
    end = data.rfind(b"PK\x05\x06")
    if end < 0:
        return bytes(b)
    count, _size, cd = struct.unpack_from("<HII", data, end + 10)
    if cd == 0xFFFFFFFF:                                     # zip64: the central directory offset is in the zip64 record
        loc = data.rfind(b"PK\x06\x07", 0, end)
        rec = struct.unpack_from("<Q", data, loc + 8)[0] if loc >= 0 else None
        if rec is None or data[rec:rec + 4] != b"PK\x06\x06":
            return bytes(b)
        count, cd = struct.unpack_from("<QQ", data, rec + 32)[0], struct.unpack_from("<Q", data, rec + 48)[0]
        b[rec + 13] = 3                                       # the zip64 record's "version made by" system as well
    o = cd
    for _ in range(count):
        if b[o:o + 4] != b"PK\x01\x02":
            break
        b[o + 5] = 3                                          # "version made by": system 3 (Unix), its version byte kept
        struct.pack_into("<HH", b, o + 12, dos_time, dos_date)
        n, m, k = struct.unpack_from("<HHH", b, o + 28)
        o += 46 + n + m + k
    return bytes(b)


def seed_digests(seeds: Dict[str, Tuple[str, bytes]]) -> Dict[str, dict]:
    return {fmt: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for fmt, (_ext, data) in sorted(seeds.items())}


def environment() -> str:
    """The system and the library versions the seeds were written with (they decide the bytes of library-written seeds).
    zlib too: joblib's zlib seed is compressed by the zlib build Python links (zlib-ng on some systems), and two builds
    write different bytes at the same level (pre-release test 4, S10: that seed differed on Linux with the same joblib)."""
    import zlib
    zlib_build = f"zlib {getattr(zlib, 'ZLIB_RUNTIME_VERSION', zlib.ZLIB_VERSION)}"
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return f"{platform.system()}, Python {platform.python_version()}, {zlib_build}"
    libs = []
    for lib in ("h5py", "onnx", "numpy", "joblib"):
        try:
            libs.append(f"{lib} {version(lib)}")
        except PackageNotFoundError:
            libs.append(f"{lib} not installed")
    return f"{platform.system()}, Python {platform.python_version()}, " + ", ".join(libs + [zlib_build])


def _damage(rnd: random.Random, data: bytes, how: str) -> bytes:
    b = bytearray(data)
    n = len(b)
    if n == 0:
        return bytes(b)
    if how == "flip":
        for _ in range(rnd.randint(1, 8)):
            i = rnd.randrange(n)
            b[i] ^= 1 << rnd.randrange(8)
    elif how == "bytes":
        for _ in range(rnd.randint(1, 8)):
            b[rnd.randrange(n)] = rnd.choice((0x00, 0xFF, 0x7F, 0x80, rnd.randrange(256)))
    elif how == "bigint":
        width = rnd.choice((2, 4, 8))
        value = rnd.choice((2 ** (8 * width) - 1, 2 ** (8 * width - 1) - 1, 2 ** (8 * width - 1), 2 ** 31, 0)) % 2 ** (8 * width)
        i = rnd.randrange(max(1, n - width))
        b[i:i + width] = value.to_bytes(width, rnd.choice(("little", "big")))
    elif how == "insert":
        i = rnd.randrange(n + 1)
        b[i:i] = bytes(rnd.randrange(256) for _ in range(rnd.randint(1, 64)))
    elif how == "delete":
        i = rnd.randrange(n)
        del b[i:i + rnd.randint(1, max(1, min(64, n - i)))]
    elif how == "dup":
        i = rnd.randrange(n)
        chunk = b[i:i + rnd.randint(1, 256)]
        j = rnd.randrange(n + 1)
        b[j:j] = chunk
    elif how == "truncate":
        del b[rnd.randrange(1, n) if n > 1 else 0:]
    return bytes(b)


def mutate(rnd: random.Random, fmt: str, data: bytes) -> Tuple[str, bytes]:
    how = rnd.choice(MUTATIONS if fmt in ZIP_FORMATS else MUTATIONS[:-1])
    if how != "member":
        return how, _damage(rnd, data, how)
    # damage one member of a valid archive: the container parses, the content is broken
    src = zipfile.ZipFile(io.BytesIO(data))
    names = [i.filename for i in src.infolist() if not i.is_dir()]
    victim = rnd.choice(names)
    inner = rnd.choice([m for m in MUTATIONS[:-1]])
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            payload = src.read(info.filename)
            if info.filename == victim:
                payload = _damage(rnd, payload, inner)
            dst.writestr(info, payload)
    return f"member/{inner}", out.getvalue()


# --------------------------------------------------------------------------------------
# Worker: scans cases one by one under a watchdog and reports one JSON line per case
# --------------------------------------------------------------------------------------
def worker(case_dir: str, start: int, timeout: float, max_mb: float) -> int:
    import faulthandler
    import tracemalloc

    sys.path.insert(0, REPO)
    from limnex_preflight.context import ScanOptions
    from limnex_preflight.scanner import scan_paths

    cases = json.load(open(os.path.join(case_dir, "cases.json")))
    tracemalloc.start()
    for i in range(start, len(cases)):
        c = cases[i]
        print(json.dumps({"start": i}), flush=True)
        faulthandler.dump_traceback_later(timeout, exit=True)
        tracemalloc.reset_peak()
        t = time.perf_counter()
        rec = {"i": i}
        try:
            res = scan_paths([os.path.join(case_dir, c["file"])], ScanOptions())
            rec["findings"] = len(res.findings)
            rec["notes"] = [[n.kind, n.message[:200]] for n in res.notes]
        except BaseException as exc:  # noqa: BLE001 -- exactly what the fuzzer is looking for
            rec["crash"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        faulthandler.cancel_dump_traceback_later()
        rec["seconds"] = round(time.perf_counter() - t, 3)
        rec["peak_mb"] = round(tracemalloc.get_traced_memory()[1] / 2 ** 20, 1)
        print(json.dumps(rec), flush=True)
    return 0


def unparsed_from(fmt: str, data: bytes) -> Optional[int]:
    """Offset from which a seed's bytes are never parsed (see the comment at TEXT_FORMATS), or None."""
    if fmt == "catboost" and data[:4] == b"CBM1":
        return 8 + int.from_bytes(data[4:8], "little")
    if fmt == "torch-legacy":
        import pickletools

        f = io.BytesIO(data)
        for _ in range(5):  # magic number, protocol version, sys info, the object, the storage keys
            for opcode, _arg, _pos in pickletools.genops(f):  # decodes only; the seed is a benign fixture
                if opcode.name == "STOP":
                    break
        return f.tell()
    return None


def protobuf_field_ends(data: bytes) -> List[int]:
    """Offsets where a top-level protobuf field ends (ONNX). Protobuf has no length header and no end marker, so a file
    cut exactly there is itself a complete, valid file (onnx.checker accepts it): the cut cannot be seen by any reader.
    Found by this fuzzer on 2026-11-20 (an ONNX model cut between two metadata entries); a known limit, not a bug."""
    ends, i = [], 0

    def varint(j: int) -> Tuple[int, int]:
        shift = value = 0
        while True:
            if j >= len(data) or shift > 63:
                raise ValueError("not a protobuf varint")
            c = data[j]
            j += 1
            value |= (c & 0x7F) << shift
            shift += 7
            if not c & 0x80:
                return value, j
    try:
        while i < len(data):
            key, i = varint(i)
            wire = key & 7
            if wire == 0:
                _, i = varint(i)
            elif wire == 2:
                n, i = varint(i)
                i += n
            elif wire == 1:
                i += 8
            elif wire == 5:
                i += 4
            else:
                break
            if i > len(data):
                break
            ends.append(i)
    except ValueError:
        pass
    return ends


def judge(case: dict, rec: dict, max_mb: float) -> List[str]:
    problems = []
    if "crash" in rec:
        problems.append(f"crash: {rec['crash']}")
        return problems
    for kind, msg in rec["notes"]:
        if kind == "error" and msg.startswith("scan failed (") and msg[len("scan failed ("):].startswith(BUG_TYPES):
            problems.append(f"bug: {msg}")
    if rec["peak_mb"] > max_mb:
        problems.append(f"memory: {rec['peak_mb']} MB")
    if case["mutation"] == "truncate" and case["format"] not in TEXT_FORMATS:
        if rec["findings"] == 0 and not any(k in ("error", "degraded") for k, _ in rec["notes"]):
            if case.get("unparsed_from") is not None and case["bytes"] >= case["unparsed_from"]:
                problems.append("known-limit: cut inside bytes that are never parsed")
            elif case["bytes"] in case.get("complete_at", ()):
                problems.append("known-limit: cut between two top-level protobuf fields (the rest is a complete, valid file)")
            else:
                problems.append("clean-cut: truncated file reported as clean and complete")
    return problems


def run(cases_per_format: int, seed: int, timeout: float, max_mb: float, workdir: str) -> dict:
    seeds = build_seeds(os.path.join(workdir, "seeds"), seed)
    case_dir = os.path.join(workdir, "cases")
    os.makedirs(case_dir, exist_ok=True)
    cases = []
    for fmt in sorted(seeds):
        ext, data = seeds[fmt]
        rnd = random.Random(f"{seed}:{fmt}")      # one stream per format: a seed that differs elsewhere moves only its row
        limit = unparsed_from(fmt, data)
        complete_at = protobuf_field_ends(data) if fmt == "onnx" else []
        for k in range(cases_per_format):
            how, damaged = mutate(rnd, fmt, data)
            name = f"{fmt}_{k:04d}{ext}"
            with open(os.path.join(case_dir, name), "wb") as fh:
                fh.write(damaged)
            cases.append({"format": fmt, "mutation": how, "file": name, "bytes": len(damaged), "unparsed_from": limit,
                          "complete_at": complete_at})
    with open(os.path.join(case_dir, "cases.json"), "w") as fh:
        json.dump(cases, fh)

    records: Dict[int, dict] = {}
    start = 0
    while start < len(cases):
        cmd = [sys.executable, "-I", os.path.abspath(__file__), "--worker", case_dir, "--start", str(start),
               "--timeout", str(timeout), "--max-mb", str(max_mb)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        last_started = start
        for line in proc.stdout.splitlines():
            msg = json.loads(line)
            if "start" in msg:
                last_started = msg["start"]
            else:
                records[msg["i"]] = msg
        if proc.returncode == 0:
            break
        # the worker died on case `last_started` (the watchdog fired, or the interpreter itself crashed): record, go on
        timed_out = "Timeout (" in proc.stderr
        first_line = (proc.stderr.strip().splitlines() or ["worker died"])[0][:200]
        records[last_started] = {"i": last_started, ("hang" if timed_out else "crash"): first_line, "notes": [],
                                 "findings": 0, "seconds": timeout if timed_out else 0, "peak_mb": 0}
        start = last_started + 1

    failures, known = [], []
    by_format = collections.defaultdict(collections.Counter)
    for i, case in enumerate(cases):
        rec = records.get(i, {"i": i, "crash": "no record", "notes": [], "findings": 0, "seconds": 0, "peak_mb": 0})
        problems = [f"hang: over {timeout}s"] if "hang" in rec else judge(case, rec, max_mb)
        outcome = "not clean" if rec.get("findings") or any(k in ("error", "degraded") for k, _ in rec.get("notes", [])) else "clean"
        by_format[case["format"]]["cases"] += 1
        by_format[case["format"]][outcome] += 1
        limits = [p for p in problems if p.startswith("known-limit")]
        problems = [p for p in problems if not p.startswith("known-limit")]
        if limits:
            by_format[case["format"]]["known limit"] += 1
            known.append({**case, "problems": limits})
        if problems:
            by_format[case["format"]]["FAIL"] += 1
            failures.append({**case, "problems": problems})
    slowest = max(records.values(), key=lambda r: r.get("seconds", 0))
    peak = max(records.values(), key=lambda r: r.get("peak_mb", 0))
    return {"seed": seed, "cases": len(cases), "formats": {k: dict(v) for k, v in sorted(by_format.items())},
            "failures": failures, "known_limits": known, "slowest_s": slowest.get("seconds"), "peak_mb": peak.get("peak_mb"),
            "skipped_formats": sorted(_all_formats() - set(seeds)), "timeout": timeout, "max_mb": max_mb,
            "seeds": seed_digests(seeds), "environment": environment()}


def _all_formats() -> set:
    sys.path.insert(0, HERE)
    import fixtures as fx

    return {f.name for f in fx.FORMATS}


def render(rep: dict) -> str:
    lines = ["# Mutation fuzzing of the format parsers (generated by evaluation/fuzz_formats.py)", "",
             f"Seed {rep['seed']}, {rep['cases']} damaged files built from the benign clean-twin fixtures. Limits: "
             f"{rep['timeout']} s and {rep['max_mb']} MB per file. Slowest file {rep['slowest_s']} s, highest peak "
             f"{rep['peak_mb']} MB.", "",
             f"Made on: {rep.get('environment', 'not recorded')}. Each format has its own random stream, so on another system",
             "only the rows whose seed differs (SHA-256 in the last table) can change; timings and peaks always differ.", "",
             "| format | cases | reported not clean | reported clean | known limit | **failures** |", "|---|---|---|---|---|---|"]
    for fmt, c in rep["formats"].items():
        lines.append(f"| {fmt} | {c.get('cases', 0)} | {c.get('not clean', 0)} | {c.get('clean', 0)} | "
                     f"{c.get('known limit', 0)} | **{c.get('FAIL', 0)}** |")
    lines += ["", "\"Reported clean\" is not an error by itself: most damage to text, and some damage to binary files, leaves",
              "a file that is still valid. Only a truncated binary file must never come back clean (rule `clean-cut`).",
              "\"Known limit\" is a cut inside bytes the scanner never parses by design (CatBoost extra data after the model",
              "core; the raw tensor bytes of a legacy PyTorch file), or a cut exactly between two top-level fields of an ONNX",
              "file, which leaves a complete, valid file. Those cuts are not detected, and the README says so.", ""]
    if rep["skipped_formats"]:
        lines += [f"Not fuzzed here (the seed writer needs a library that is not installed): {', '.join(rep['skipped_formats'])}.", ""]
    if rep["failures"]:
        lines += ["## Failures", "", "| file | mutation | problems |", "|---|---|---|"]
        for f in rep["failures"]:
            lines.append(f"| {f['file']} | {f['mutation']} | {'; '.join(f['problems'])} |")
    else:
        lines.append("**No failures.**")
    if rep.get("known_limits"):
        lines += ["", "## Known limits (not failures)", "", "| file | cut at byte | why |", "|---|---|---|"]
        for f in rep["known_limits"]:
            lines.append(f"| {f['file']} | {f['bytes']} | {'; '.join(f['problems'])} |")
    if rep.get("seeds"):
        lines += ["", "## Seeds", "", "| format | bytes | SHA-256 |", "|---|---|---|"]
        for fmt, d in rep["seeds"].items():
            lines.append(f"| {fmt} | {d['bytes']} | `{d['sha256']}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300, help="damaged files per format")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--timeout", type=float, default=20.0, help="seconds per file before it counts as a hang")
    ap.add_argument("--max-mb", type=float, default=256.0, help="peak allocation per file before it counts as a memory failure")
    ap.add_argument("--out", help="write the Markdown report here")
    ap.add_argument("--json", help="write the full JSON report here")
    ap.add_argument("--quick", action="store_true", help="print a one-line summary only")
    ap.add_argument("--worker", metavar="CASE_DIR", help=argparse.SUPPRESS)
    ap.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker, args.start, args.timeout, args.max_mb)
    with tempfile.TemporaryDirectory(prefix="preflight-fuzz-") as work:
        rep = run(args.cases, args.seed, args.timeout, args.max_mb, work)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render(rep))
    print(f"{rep['cases']} cases, {len(rep['failures'])} failures, slowest {rep['slowest_s']} s, peak {rep['peak_mb']} MB")
    if not args.quick:
        for f in rep["failures"][:30]:
            print(f"  {f['file']} [{f['mutation']}]: {'; '.join(f['problems'])}")
    return 1 if rep["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
