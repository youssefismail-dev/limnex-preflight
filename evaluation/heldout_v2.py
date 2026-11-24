"""Held-out set v2: draw (API metadata only) and run (download -> scan with three tools -> delete).

The rule is in HELDOUT_V2.md and was committed before any download. Real repository ids are written ONLY to the private
files named on the command line (this script refuses to write them inside the repository). Nothing downloaded is imported,
unpickled or executed; every scanner runs as a separate process whose working directory is an empty scratch directory.

  python evaluation/heldout_v2.py draw --private DRAW.json --cache-dir DIR
  python evaluation/heldout_v2.py run  --private DRAW.json --work DIR --results RESULTS.json \
        --picklescan BIN --modelscan BIN [--only community-repo-101,...]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

BASE = "https://huggingface.co"
UA = {"User-Agent": "limnex-preflight-eval/0.1"}
SEED = 20261105
CUTOFF = "2025-11-05T00:00:00"
MIN_DOWNLOADS = 100
GB = 10 ** 9
FORMAT_CAP_TOTAL = 1 * GB
FORMAT_CAP_FILE = 1 * GB
FORMAT_MAX_FILES = 6
TEXT_EXT = (".json", ".yaml", ".yml", ".txt", ".md", ".cfg", ".ini", ".toml", ".ipynb")
TEXT_MAX = 5 * 1024 * 1024
TEXT_MAX_FILES = 30
STRESS_MIN, STRESS_MAX = 3 * GB, 8 * GB
LISTING_MAX, ELIGIBLE_MAX = 5000, 250  # deviation D1
PER_STRATUM = 10

NO_DOWNLOAD_FLOOR = {"joblib"}  # deviation D2 (HELDOUT_V2.md): the sklearn tag has only 1 recent repository with >= 100 downloads
PICKLE = ("pytorch_model*.bin", "*.pt", "*.pth", "*.ckpt", "*.pkl", "*.pickle")
STRATA = [  # (name, listing filter, file patterns)
    ("pickle", "pytorch", PICKLE),
    ("safetensors", "safetensors", ("*.safetensors",)),
    ("gguf", "gguf", ("*.gguf",)),
    ("onnx", "onnx", ("*.onnx",)),
    ("keras", "keras", ("*.h5", "*.hdf5", "*.keras")),
    ("joblib", "sklearn", ("*.joblib", "*.pkl", "*.pickle")),
]


# --------------------------------------------------------------------------------------------- HTTP
def http_json(url: str):
    last = None
    for i in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return json.load(r), r.headers.get("Link")
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (400, 401, 403, 404):
                raise
        except Exception as exc:  # noqa: BLE001 - network noise
            last = exc
        time.sleep(min(60, 2 ** i))
    raise last  # type: ignore[misc]


def listing(flt: str):
    url = (f"{BASE}/api/models?filter={urllib.parse.quote(flt)}&sort=downloads&direction=-1&limit=100"
           "&expand[]=downloads&expand[]=createdAt&expand[]=gated&expand[]=private&expand[]=disabled&expand[]=siblings")
    items, eligible = [], 0
    while url and len(items) < LISTING_MAX and eligible < ELIGIBLE_MAX:
        page, link = http_json(url)
        items += page
        eligible += sum(1 for x in page if meta_ok(x))
        m = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
        url = m.group(1) if m else None
    return items


def meta_ok(x, min_downloads: int = MIN_DOWNLOADS) -> bool:
    return (x.get("createdAt", "") >= CUTOFF and x.get("downloads", 0) >= min_downloads and x.get("gated") is False
            and not x.get("private") and not x.get("disabled"))


def matches(name: str, pats) -> bool:
    base = name.rsplit("/", 1)[-1].lower()
    return any(fnmatch.fnmatch(base, p) for p in pats)


def flagged_by_hub(info) -> bool:
    if (info.get("securityRepoStatus") or {}).get("filesWithIssues"):
        return True
    for s in info.get("siblings", []):
        st = s.get("securityFileStatus")
        if st is not None and not (isinstance(st, dict) and st.get("status") in (None, "safe")):
            return True
    return False


def plan_files(info, pats):
    """(format files, text files) per section 4 of HELDOUT_V2.md, or None if the repository is ineligible."""
    files = [(s["rfilename"], s.get("size") or (s.get("lfs") or {}).get("size") or 0) for s in info["siblings"]]
    fmt = sorted(((p, n) for p, n in files if matches(p, pats)), key=lambda t: (t[1], t[0]))
    if not fmt or fmt[0][1] > FORMAT_CAP_FILE:
        return None
    chosen, total = [], 0
    for p, n in fmt:
        if len(chosen) >= FORMAT_MAX_FILES or n > FORMAT_CAP_FILE or total + n > FORMAT_CAP_TOTAL:
            break
        chosen.append((p, n))
        total += n
    text = sorted((p, n) for p, n in files if p.lower().endswith(TEXT_EXT) and n <= TEXT_MAX)
    always = [t for t in text if t[0] in ("README.md", "config.json")]
    rest = [t for t in text if t not in always]
    return chosen, (always + rest)[:TEXT_MAX_FILES]


def load_exclusions(mapping_path: str):
    ids = set()
    for name in ("hf_models.txt", "hf_models_heldout.txt"):
        for line in open(os.path.join(HERE, name), encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("community-repo-"):
                ids.add(line.lower())
    if mapping_path:
        def walk(o):
            if isinstance(o, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", o):
                ids.add(o.lower())
            elif isinstance(o, dict):
                for k, v in o.items():
                    walk(k)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(json.load(open(mapping_path, encoding="utf-8")).get("repos", {}))
    return ids


def refuse_inside_repo(path: str):
    if os.path.realpath(path).startswith(REPO + os.sep):
        raise SystemExit(f"refusing to write {path!r}: inside the repository")


# --------------------------------------------------------------------------------------------- draw
def cmd_draw(args) -> int:
    refuse_inside_repo(args.private)
    refuse_inside_repo(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)
    excl = load_exclusions(args.exclusion_mapping)
    used, repos, stats = set(), [], {}
    pools = {}
    for name, flt, pats in STRATA:
        cache = os.path.join(args.cache_dir, f"listing_{flt}.json")
        if os.path.exists(cache):
            items = json.load(open(cache))
        else:
            items = listing(flt)
            json.dump(items, open(cache, "w"))
        pools[name] = items
        floor = 0 if name in NO_DOWNLOAD_FLOOR else MIN_DOWNLOADS
        cands = [x for x in items if meta_ok(x, floor) and x["id"].lower() not in excl
                 and any(matches(s["rfilename"], pats) for s in x.get("siblings", []))]
        n_excluded = sum(1 for x in items if meta_ok(x, floor) and x["id"].lower() in excl)
        cands.sort(key=lambda x: x["id"])
        random.Random(SEED).shuffle(cands)
        st = Counter(listed=len(items), metadata_eligible=sum(1 for x in items if meta_ok(x, floor)), excluded_by_set=n_excluded,
                     with_stratum_file=len(cands))
        accepted = []
        for c in cands:
            if len(accepted) >= PER_STRATUM:
                break
            if c["id"].lower() in used:
                st["already_in_earlier_stratum"] += 1
                continue
            info, _ = http_json(f"{BASE}/api/models/{c['id']}?blobs=true&securityStatus=true")
            if flagged_by_hub(info):
                st["rejected_hub_flagged"] += 1
                continue
            pl = plan_files(info, pats)
            if pl is None:
                st["rejected_size_rule"] += 1
                continue
            used.add(c["id"].lower())
            accepted.append({"id": c["id"], "stratum": name, "sha": info["sha"], "format_files": pl[0], "text_files": pl[1]})
        st["accepted"] = len(accepted)
        stats[name] = dict(st)
        repos += accepted
    # stress files: 2 safetensors + 1 gguf, one file of 3-8 GB each
    stress = []
    for name, want in (("safetensors", 2), ("gguf", 1)):
        pats = dict((n, p) for n, _, p in STRATA)[name]
        cands = [x for x in pools[name] if meta_ok(x) and x["id"].lower() not in excl and x["id"].lower() not in used
                 and any(matches(s["rfilename"], pats) for s in x.get("siblings", []))]
        cands.sort(key=lambda x: x["id"])
        random.Random(SEED).shuffle(cands)
        got, rej = 0, Counter()
        for c in cands:
            if got >= want:
                break
            info, _ = http_json(f"{BASE}/api/models/{c['id']}?blobs=true&securityStatus=true")
            if flagged_by_hub(info):
                rej["hub_flagged"] += 1
                continue
            big = sorted((s.get("size") or (s.get("lfs") or {}).get("size") or 0, s["rfilename"]) for s in info["siblings"]
                         if matches(s["rfilename"], pats) and STRESS_MIN <= (s.get("size") or (s.get("lfs") or {}).get("size") or 0) <= STRESS_MAX)
            if not big:
                rej["no_3_to_8_gb_file"] += 1
                continue
            used.add(c["id"].lower())
            stress.append({"id": c["id"], "stratum": "stress-" + name, "sha": info["sha"], "format_files": [(big[0][1], big[0][0])], "text_files": []})
            got += 1
        stats["stress-" + name] = {"candidates": len(cands), "accepted": got, **dict(rej)}
    for i, r in enumerate(repos):
        r["pseudo"] = f"community-repo-{101 + i}"
    for i, r in enumerate(stress):
        r["pseudo"] = f"community-repo-{161 + i}"
    json.dump({"seed": SEED, "cutoff": CUTOFF, "stats": stats, "repos": repos + stress}, open(args.private, "w"), indent=1)
    print(json.dumps(stats, indent=1))
    print("repos:", len(repos), "stress:", len(stress),
          "planned GB:", round(sum(x[1] for r in repos + stress for x in r["format_files"]) / GB, 2))
    return 0


# --------------------------------------------------------------------------------------------- download
def download(url: str, dest: str, size: int, parts: int = 1):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as fh:
        if size:
            fh.truncate(size)
    segs = [(0, size - 1)] if parts <= 1 or size < 64 * 1024 * 1024 else \
        [(i * (size // parts), (size - 1) if i == parts - 1 else (i + 1) * (size // parts) - 1) for i in range(parts)]

    def seg(a_b):
        a, b = a_b
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{b}"} if size else UA)
                with urllib.request.urlopen(req, timeout=120) as r, open(dest, "r+b") as out:
                    pos = a
                    while True:
                        chunk = r.read(4 << 20)
                        if not chunk:
                            break
                        out.seek(pos)
                        out.write(chunk)
                        pos += len(chunk)
                if size and pos != b + 1:
                    raise OSError(f"short read {pos - a} of {b + 1 - a}")
                return
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=len(segs)) as ex:
        list(ex.map(seg, segs))
    got = os.path.getsize(dest)
    if size and got != size:
        raise OSError(f"size mismatch: {got} != {size}")


def fetch_repo(rec, work):
    errs = []

    def one(item):
        path, size = item
        url = f"{BASE}/{rec['id']}/resolve/{rec['sha']}/{urllib.parse.quote(path)}"
        try:
            download(url, os.path.join(work, path), size, parts=4 if size >= 64 * 1024 * 1024 else 1)
            return None
        except Exception as exc:  # noqa: BLE001
            return (path, f"{type(exc).__name__}")

    items = [tuple(x) for x in rec["format_files"]] + [tuple(x) for x in rec["text_files"]]
    with ThreadPoolExecutor(max_workers=3) as ex:
        for r in ex.map(one, items):
            if r:
                errs.append(r)
    return errs


# --------------------------------------------------------------------------------------------- run tools
def _proc_int(path: str, key: str) -> int:
    try:
        with open(path, "rb") as fh:
            for line in fh:
                if line.startswith(key):
                    return int(line.split(b":")[1].split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _no_network():
    os.unshare(os.CLONE_NEWNET)  # the child gets an empty network namespace: it cannot reach any service


def timed(cmd, cwd, no_network=False, deadline=None):
    """Run a command; return (exit, wall seconds, peak RSS MB, output text, bytes read via read(2)).

    Peak RSS is the child's own VmHWM polled from /proc while it runs (wait4()'s ru_maxrss includes the parent's memory at
    fork, which is why it is NOT used). Readings are only accepted once /proc/<pid>/cmdline shows the exec'd command, so the
    pre-exec copy of this process cannot leak in. rchar (/proc/<pid>/io) counts bytes read through read(2): it is the evidence
    for 'streamed, not loaded'."""
    t0 = time.perf_counter()
    want = [c.encode() for c in cmd]
    peak_kb, rchar = 0, 0
    with tempfile.TemporaryFile() as tmp:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=tmp, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8", "HOME": cwd},
                             preexec_fn=_no_network if no_network else None)
        while p.poll() is None:
            if deadline and time.perf_counter() - t0 > deadline:
                p.kill()
                p.wait()
                break
            try:
                with open(f"/proc/{p.pid}/cmdline", "rb") as fh:
                    argv = fh.read().split(b"\0")[:-1]
            except OSError:
                argv = []
            if len(argv) >= len(want) and argv[len(argv) - len(want) + 1:] == want[1:]:  # shebang scripts change argv[0]
                peak_kb = max(peak_kb, _proc_int(f"/proc/{p.pid}/status", b"VmHWM"))
                rchar = max(rchar, _proc_int(f"/proc/{p.pid}/io", b"rchar"))
            time.sleep(0.002)
        wall = time.perf_counter() - t0
        tmp.seek(0)
        text = tmp.read().decode("utf-8", "replace")
    return p.returncode, wall, peak_kb / 1024.0, text, rchar


def relpath_of(f: str, work: str, known):
    f = f.replace(work + os.sep, "")
    best = ""
    for k in known:
        if (f == k or f.startswith(k + ":") or f.startswith(k + "!/")) and len(k) > len(best):
            best = k
    return best or None


def run_limnex(py, work, scratch, known, extra=()):
    out = os.path.join(scratch, "lim.json")
    if os.path.exists(out):
        os.remove(out)
    code, wall, rss, text, rchar = timed([py, "-I", "-m", "limnex_preflight", "--format", "json", "-o", out, "--fail-on", "none",
                                          "--no-color", *extra, work], scratch)
    rep = json.load(open(out)) if os.path.exists(out) else None
    res = {"exit": code, "wall_s": round(wall, 3), "rss_mb": round(rss, 1), "read_mb": round(rchar / 1e6, 1), "failed": rep is None}
    if rep:
        res["summary"] = rep["summary"]
        res["files"] = [{"file": relpath_of(f["path"], work, known) or f["path"].replace(work + os.sep, ""), "kind": f["kind"], "size": f["size"]}
                        for f in rep["files"]]
        res["findings"] = [{**{k: f.get(k) for k in ("rule_id", "category", "severity", "location", "value_masked", "fingerprint", "occurrences")},
                            "file": relpath_of(f["file"], work, known) or f["file"].replace(work + os.sep, ""), "member": f["file"].replace(work + os.sep, "")}
                           for f in rep["findings"]]
        res["notes"] = [{"kind": n["kind"], "file": relpath_of(n["file"], work, known) or n["file"].replace(work + os.sep, ""),
                         "message": n["message"]} for n in rep["notes"]]
    else:
        res["output_tail"] = text[-600:]
    return res


def run_picklescan(binp, work, scratch, known):
    code, wall, rss, text, rchar = timed([binp, "-p", work], scratch)
    flagged, errors = {}, set()
    for line in text.splitlines():
        m = re.match(r"^(.*?): dangerous import '([^']*)' FOUND$", line)
        if m:
            f = relpath_of(m.group(1), work, known)
            flagged.setdefault(f or m.group(1), []).append(m.group(2))
            continue
        m = re.match(r"^(?:ERROR: parsing pickle in |WARNING: could not parse )(.*?)(?::|\s+as pickle).*$", line)
        if m:
            errors.add(relpath_of(m.group(1), work, known) or m.group(1))
    sc = re.search(r"Scanned files: (\d+)", text)
    inf = re.search(r"Infected files: (\d+)", text)
    return {"exit": code, "wall_s": round(wall, 3), "rss_mb": round(rss, 1), "flagged": flagged, "errors": sorted(errors),
            "scanned_files": int(sc.group(1)) if sc else None, "infected_files": int(inf.group(1)) if inf else None}


def run_modelscan(binp, work, scratch, known):
    out = os.path.join(scratch, "ms.json")
    if os.path.exists(out):
        os.remove(out)
    code, wall, rss, text, rchar = timed([binp, "-p", work, "-r", "json", "-o", out], scratch)
    res = {"exit": code, "wall_s": round(wall, 3), "rss_mb": round(rss, 1)}
    if os.path.exists(out):
        d = json.load(open(out))
        flagged = {}
        for i in d.get("issues", []):
            f = relpath_of(os.path.join(work, i["source"]), work, known) or i["source"]
            flagged.setdefault(f, []).append(f"{i['module']}.{i['operator']}:{i['severity']}")
        res.update({"flagged": flagged, "scanned_files": d["summary"]["scanned"]["total_scanned"],
                    "skipped_files": d["summary"].get("skipped", {}).get("total_skipped"),
                    "skipped_reasons": sorted({str(x.get("category", x.get("description", ""))) for x in d["summary"].get("skipped", {}).get("skipped_files", [])}),
                    "errors": [{"category": e.get("category"), "file": relpath_of(os.path.join(work, e.get("source", "")), work, known)}
                               for e in d.get("errors", [])]})
    else:
        res["output_tail"] = text[-400:]
    return res


def cmd_run(args) -> int:
    refuse_inside_repo(args.private)
    refuse_inside_repo(args.results)
    refuse_inside_repo(args.work)
    draw = json.load(open(args.private))
    done = json.load(open(args.results)) if os.path.exists(args.results) else {}
    only = set(args.only.split(",")) if args.only else None
    scratch = tempfile.mkdtemp(prefix="v2scratch_", dir=args.work)
    for rec in draw["repos"]:
        if rec["pseudo"] in done or (only and rec["pseudo"] not in only):
            continue
        work = os.path.join(args.work, rec["pseudo"])
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        t0 = time.time()
        errs = fetch_repo(rec, work)
        dl_s = time.time() - t0
        known = []
        for dp, _, fns in os.walk(work):
            for fn in fns:
                known.append(os.path.relpath(os.path.join(dp, fn), work))
        total = sum(os.path.getsize(os.path.join(work, k)) for k in known)
        entry = {"pseudo": rec["pseudo"], "stratum": rec["stratum"], "download_errors": errs, "download_s": round(dl_s, 1),
                 "bytes": total, "n_files": len(known),
                 "expected_format_bytes": sum(x[1] for x in rec["format_files"])}
        entry["limnex"] = run_limnex(args.python, work, scratch, known)
        entry["picklescan"] = run_picklescan(args.picklescan, work, scratch, known)
        entry["modelscan"] = run_modelscan(args.modelscan, work, scratch, known)
        entry["per_file"] = {}
        for f in entry["limnex"].get("files", []):  # file-by-file verdicts of the two other tools on pickle-bearing files
            if f["kind"] in ("pickle", "zip", "zlib"):
                one = os.path.join(work, f["file"])
                entry["per_file"][f["file"]] = {"picklescan": run_picklescan(args.picklescan, one, scratch, [f["file"]]),
                                                "modelscan": run_modelscan(args.modelscan, one, scratch, [f["file"]])}
        shutil.rmtree(work, ignore_errors=True)
        done[rec["pseudo"]] = entry
        json.dump(done, open(args.results, "w"), indent=1)
        print(f"{rec['pseudo']} {rec['stratum']}: {len(known)} files {total / 1e6:.0f} MB, dl {dl_s:.0f}s, "
              f"limnex {entry['limnex'].get('summary', {}).get('findings', '?')} findings {entry['limnex']['wall_s']}s "
              f"{entry['limnex']['rss_mb']} MB", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


def cmd_deep(args) -> int:
    """Task 4: the stress files again, scanned with --deep (every payload byte is read): time, peak RSS, bytes read."""
    refuse_inside_repo(args.private)
    refuse_inside_repo(args.out)
    refuse_inside_repo(args.work)
    draw = json.load(open(args.private))
    scratch = tempfile.mkdtemp(prefix="v2deep_", dir=args.work)
    rows = []
    for rec in draw["repos"]:
        if not rec["stratum"].startswith("stress"):
            continue
        work = os.path.join(args.work, rec["pseudo"])
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        fetch_repo(rec, work)
        known = [os.path.relpath(os.path.join(dp, fn), work) for dp, _, fns in os.walk(work) for fn in fns]
        size = sum(os.path.getsize(os.path.join(work, k)) for k in known)
        res = run_limnex(args.python, work, scratch, known, extra=("--deep",))
        shutil.rmtree(work, ignore_errors=True)
        rows.append({"repo": rec["pseudo"], "gb": round(size / 1e9, 2), "seconds": res["wall_s"], "mb_per_s": round(size / 1e6 / res["wall_s"], 1),
                     "rss_mb": res["rss_mb"], "read_mb": res["read_mb"], "findings": len(res.get("findings", []))})
        print(rows[-1], flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--private", required=True)
    d.add_argument("--cache-dir", required=True)
    d.add_argument("--exclusion-mapping", default="")
    d.set_defaults(fn=cmd_draw)
    r = sub.add_parser("run")
    r.add_argument("--private", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--picklescan", required=True)
    r.add_argument("--modelscan", required=True)
    r.add_argument("--only", default="")
    r.set_defaults(fn=cmd_run)
    k = sub.add_parser("deep")
    k.add_argument("--private", required=True)
    k.add_argument("--work", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--python", default=sys.executable)
    k.set_defaults(fn=cmd_deep)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
