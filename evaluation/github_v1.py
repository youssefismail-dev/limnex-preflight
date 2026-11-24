"""Public-GitHub round 1: draw (API metadata only) and run (shallow clone -> scan -> delete). Rule: GITHUB_V1.md.

  python evaluation/github_v1.py draw --private DRAW.json
  python evaluation/github_v1.py run  --private DRAW.json --work DIR --results RESULTS.json [--python PY]

Real repository names are written only to the private files named on the command line (refused inside this repository).
Nothing cloned is imported, executed or unpickled; no git hook or LFS filter runs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import heldout_v2 as H  # noqa: E402  (timing / scanner helpers)

SEED = 20261105
QUERIES = ['topic:machine-learning language:"Jupyter Notebook"', "topic:deep-learning language:Python", "topic:pytorch"]
QUAL = " stars:20..3000 pushed:>2025-11-05 archived:false fork:false size:<150000"
MODEL = (".pt", ".pth", ".pkl", ".pickle", ".joblib", ".h5", ".hdf5", ".keras", ".onnx", ".safetensors", ".ckpt", ".npz", ".bin")
TEXT = (".ipynb", ".yaml", ".yml", ".json", ".cfg", ".ini", ".toml", ".txt", ".md")
MB = 10 ** 6


def gh(path: str):
    for i in range(4):
        r = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            return json.loads(r.stdout)
        time.sleep(2 ** i)
    raise RuntimeError(f"gh api failed: {r.stderr[:120]}")


def refuse_inside_repo(path: str):
    if os.path.realpath(path).startswith(REPO + os.sep):
        raise SystemExit(f"refusing {path!r}: inside the repository")


def cmd_draw(a) -> int:
    refuse_inside_repo(a.private)
    cands = {}
    for q in QUERIES:
        for page in (1, 2, 3):
            res = gh("search/repositories?q=" + urllib.parse.quote(q + QUAL) + f"&sort=stars&order=desc&per_page=100&page={page}")
            for it in res["items"]:
                cands[it["full_name"]] = it["default_branch"]
            time.sleep(2.5)  # search API: 30 requests/minute
    names = sorted(cands)
    random.Random(SEED).shuffle(names)
    stats = {"pool": len(names), "no_model_or_text": 0, "truncated": 0, "lfs_pointer": 0, "too_big": 0, "api_error": 0}
    accepted = []
    for full in names:
        if len(accepted) >= 10:
            break
        try:
            tree = gh(f"repos/{full}/git/trees/{urllib.parse.quote(cands[full])}?recursive=1")
        except Exception:  # noqa: BLE001
            stats["api_error"] += 1
            continue
        if tree.get("truncated"):
            stats["truncated"] += 1
            continue
        blobs = [(t["path"], t.get("size", 0)) for t in tree["tree"] if t["type"] == "blob"]
        models = [b for b in blobs if b[0].lower().endswith(MODEL)]
        texts = [b for b in blobs if b[0].lower().endswith(TEXT)]
        if not models or len(texts) < 3:
            stats["no_model_or_text"] += 1
            continue
        if any(s <= 200 for _, s in models):
            stats["lfs_pointer"] += 1
            continue
        if sum(s for _, s in models + texts) > 300 * MB or any(s > 150 * MB for _, s in models + texts):
            stats["too_big"] += 1
            continue
        accepted.append({"name": full, "branch": cands[full], "n_model": len(models), "n_text": len(texts),
                         "bytes": sum(s for _, s in models + texts), "pseudo": f"github-repo-{101 + len(accepted)}"})
    stats["accepted"] = len(accepted)
    json.dump({"seed": SEED, "stats": stats, "repos": accepted}, open(a.private, "w"), indent=1)
    print(json.dumps(stats), "planned MB:", round(sum(r["bytes"] for r in accepted) / MB))
    return 0


def cmd_run(a) -> int:
    for p in (a.private, a.results, a.work):
        refuse_inside_repo(p)
    draw = json.load(open(a.private))
    done = json.load(open(a.results)) if os.path.exists(a.results) else {}
    scratch = tempfile.mkdtemp(prefix="ghscratch_", dir=a.work)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": scratch, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1",
           "GIT_CONFIG_NOSYSTEM": "1", "LANG": "C.UTF-8"}
    for k in ("HTTPS_PROXY", "https_proxy", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "GIT_SSL_CAINFO", "CURL_CA_BUNDLE", "NO_PROXY", "no_proxy"):
        if k in os.environ:
            env[k] = os.environ[k]
    for rec in draw["repos"]:
        if rec["pseudo"] in done:
            continue
        work = os.path.join(a.work, rec["pseudo"])
        shutil.rmtree(work, ignore_errors=True)
        r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "core.symlinks=false", "-c", "filter.lfs.smudge=",
                            "-c", "filter.lfs.process=", "-c", "filter.lfs.required=false", "clone", "--quiet", "--depth", "1",
                            "--single-branch", "--no-recurse-submodules", f"https://github.com/{rec['name']}.git", work],
                           capture_output=True, text=True, env=env, timeout=900)
        if r.returncode != 0:
            done[rec["pseudo"]] = {"pseudo": rec["pseudo"], "clone_failed": True}
            print(rec["pseudo"], "clone failed", flush=True)
            continue
        shutil.rmtree(os.path.join(work, ".git"), ignore_errors=True)
        known = [os.path.relpath(os.path.join(dp, fn), work) for dp, _, fns in os.walk(work) for fn in fns]
        out = os.path.join(scratch, "lim.json")
        if os.path.exists(out):
            os.remove(out)
        code, wall, rss, text, rchar = H.timed([a.python, "-I", "-m", "limnex_preflight", "--format", "json", "-o", out,
                                                "--fail-on", "none", "--no-color", "--no-ignore", work], scratch)
        rep = json.load(open(out)) if os.path.exists(out) else None
        total = sum(os.path.getsize(os.path.join(work, k)) for k in known)
        shutil.rmtree(work, ignore_errors=True)
        ent = {"pseudo": rec["pseudo"], "bytes": total, "n_files_on_disk": len(known), "wall_s": round(wall, 3), "rss_mb": round(rss, 1),
               "failed": rep is None}
        if rep:
            ent["summary"] = rep["summary"]
            ent["files"] = [{"file": f["path"].replace(work + os.sep, ""), "kind": f["kind"], "size": f["size"]} for f in rep["files"]]
            ent["findings"] = [{**{k: f.get(k) for k in ("rule_id", "category", "severity", "location", "value_masked", "occurrences")},
                                "file": f["file"].replace(work + os.sep, "")} for f in rep["findings"]]
            ent["notes"] = [{"kind": n["kind"], "file": n["file"].replace(work + os.sep, ""), "message": n["message"]} for n in rep["notes"]]
        done[rec["pseudo"]] = ent
        json.dump(done, open(a.results, "w"), indent=1)
        print(rec["pseudo"], f"{len(known)} files {total / MB:.0f} MB, {len((rep or {}).get('findings', []))} findings, {wall:.1f}s {rss:.0f} MB", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--private", required=True)
    d.set_defaults(fn=cmd_draw)
    r = sub.add_parser("run")
    r.add_argument("--private", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--python", default=sys.executable)
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
