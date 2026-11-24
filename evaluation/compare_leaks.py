"""Four leak scanners on the SAME files: limnex-preflight, gitleaks, TruffleHog, detect-secrets.

  python evaluation/compare_leaks.py --private DRAW.json --work DIR --results RESULTS.json \
       --python PY --gitleaks BIN --trufflehog BIN --detect-secrets BIN [--only community-repo-101,...]

Re-downloads the main repositories of the private draw (same pinned commits, same files, same size caps; stress files
skipped), scans each with the four tools, keeps ONLY normalised metadata (tool, rule, file, line), then deletes the files.
Safety:
  * gitleaks / TruffleHog / detect-secrets run in an EMPTY NETWORK NAMESPACE (os.unshare(CLONE_NEWNET)): they cannot contact
    any service, whatever their flags. TruffleHog additionally gets --no-verification --no-update.
  * gitleaks runs with --redact (the report holds REDACTED); detect-secrets reports hashes only, and the hash is dropped;
    TruffleHog's JSON (which contains the raw secret) is parsed from a pipe and ONLY detector name, file and line are kept.
  * nothing downloaded is executed; every Python process that reads it runs with -I.
Real repository ids stay in the private files (refused inside the repository).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import heldout_v2 as H  # noqa: E402

DEADLINE = 1500  # seconds per tool per repository


def norm_file(path: str, work: str, known):
    if not os.path.isabs(path):
        path = os.path.join(work, path)
    return H.relpath_of(path, work, known) or path.replace(work + os.sep, "")


def run_preflight(py, work, scratch, known):
    out = os.path.join(scratch, "lim.json")
    if os.path.exists(out):
        os.remove(out)
    code, wall, rss, text, _ = H.timed([py, "-I", "-m", "limnex_preflight", "--format", "json", "-o", out, "--fail-on", "none", "--no-color",
                                        "--no-ignore", work], scratch, deadline=DEADLINE)
    rep = json.load(open(out)) if os.path.exists(out) else None
    fs = []
    for f in (rep or {}).get("findings", []):
        fs.append({"tool": "preflight", "rule": f["rule_id"], "severity": f["severity"], "category": f["category"],
                   "file": norm_file(f["file"], work, known), "line": f.get("line"), "location": f["location"], "masked": f["value_masked"]})
    return {"wall_s": round(wall, 3), "rss_mb": round(rss, 1), "failed": rep is None, "findings": fs,
            "scanned": len((rep or {}).get("files", []))}


def run_gitleaks(binp, work, scratch, known):
    out = os.path.join(scratch, "gl.json")
    if os.path.exists(out):
        os.remove(out)
    code, wall, rss, text, _ = H.timed([binp, "detect", "--no-git", "--redact", "--source", work, "--report-format", "json", "--report-path", out,
                                        "--exit-code", "0", "--no-banner", "--log-level", "error"], scratch, no_network=True, deadline=DEADLINE)
    fs = []
    if os.path.exists(out):
        for f in json.load(open(out)) or []:
            fs.append({"tool": "gitleaks", "rule": f["RuleID"], "file": norm_file(f["File"], work, known), "line": f.get("StartLine"),
                       "masked": f.get("Match") if f.get("Match") == "REDACTED" else "REDACTED"})
    return {"wall_s": round(wall, 3), "rss_mb": round(rss, 1), "exit": code, "findings": fs, "failed": not os.path.exists(out)}


def run_detect_secrets(binp, work, scratch, known):
    code, wall, rss, text, _ = H.timed([binp, "scan", "--all-files", "."], work, no_network=True, deadline=DEADLINE)
    fs, failed = [], False
    try:
        d = json.loads(text)
        for fn, items in d["results"].items():
            for it in items:
                fs.append({"tool": "detect-secrets", "rule": it["type"], "file": norm_file(fn, work, known), "line": it.get("line_number")})  # hash dropped
    except ValueError:
        failed = True
    return {"wall_s": round(wall, 3), "rss_mb": round(rss, 1), "exit": code, "findings": fs, "failed": failed}


def run_trufflehog(binp, work, scratch, known):
    """Streams TruffleHog's JSON lines (they contain the raw secret) through a pipe; keeps detector, file, line ONLY."""
    t0 = time.perf_counter()
    cmd = [binp, "filesystem", work, "--no-verification", "--json", "--no-update", "--no-color"]
    p = subprocess.Popen(cmd, cwd=scratch, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                         env={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8", "HOME": scratch}, preexec_fn=H._no_network)
    fs, peak, last = [], 0, 0.0
    want = [c.encode() for c in cmd]
    for line in p.stdout:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if "DetectorName" in d:
            fd = (d.get("SourceMetadata") or {}).get("Data", {}).get("Filesystem", {})
            fs.append({"tool": "trufflehog", "rule": d["DetectorName"], "file": norm_file(fd.get("file", ""), work, known), "line": fd.get("line"),
                       "verified": bool(d.get("Verified"))})
        d = None  # drop the raw secret immediately
        now = time.perf_counter()
        if now - last > 0.2:
            last = now
            peak = max(peak, H._proc_int(f"/proc/{p.pid}/status", b"VmHWM"))
        if now - t0 > DEADLINE:
            p.kill()
            break
    p.wait()
    wall = time.perf_counter() - t0
    return {"wall_s": round(wall, 3), "rss_mb": round(peak / 1024.0, 1), "exit": p.returncode, "findings": fs, "failed": False,
            "any_verified": any(f["verified"] for f in fs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--gitleaks", required=True)
    ap.add_argument("--trufflehog", required=True)
    ap.add_argument("--detect-secrets", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for p in (a.private, a.work, a.results):
        H.refuse_inside_repo(p)
    draw = json.load(open(a.private))
    done = json.load(open(a.results)) if os.path.exists(a.results) else {}
    only = set(a.only.split(",")) if a.only else None
    scratch = tempfile.mkdtemp(prefix="cmp_", dir=a.work)
    for rec in draw["repos"]:
        if rec["stratum"].startswith("stress") or rec["pseudo"] in done or (only and rec["pseudo"] not in only):
            continue
        work = os.path.join(a.work, rec["pseudo"])
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        errs = H.fetch_repo(rec, work)
        known = [os.path.relpath(os.path.join(dp, fn), work) for dp, _, fns in os.walk(work) for fn in fns]
        ent = {"pseudo": rec["pseudo"], "stratum": rec["stratum"], "download_errors": errs, "n_files": len(known),
               "bytes": sum(os.path.getsize(os.path.join(work, k)) for k in known)}
        ent["preflight"] = run_preflight(a.python, work, scratch, known)
        ent["gitleaks"] = run_gitleaks(a.gitleaks, work, scratch, known)
        ent["detect-secrets"] = run_detect_secrets(getattr(a, "detect_secrets"), work, scratch, known)
        ent["trufflehog"] = run_trufflehog(a.trufflehog, work, scratch, known)
        shutil.rmtree(work, ignore_errors=True)
        done[rec["pseudo"]] = ent
        json.dump(done, open(a.results, "w"), indent=1)
        print(rec["pseudo"], rec["stratum"], f"{ent['bytes'] / 1e6:.0f} MB |", " ".join(f"{t}:{len(ent[t]['findings'])}/{ent[t]['wall_s']}s" for t in ("preflight", "gitleaks", "trufflehog", "detect-secrets")), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
