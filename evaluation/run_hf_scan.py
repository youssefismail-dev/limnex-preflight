"""Scan the downloaded Hugging Face set, one repo at a time, and write a MASKED findings report.

--dump-raw FILE writes the unmasked surrounding context of every finding for the human reviewer. That file may
contain real third-party data (that is the point of reviewing), so it is refused inside this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from limnex_preflight import context as ctxmod  # noqa: E402
from limnex_preflight.context import ScanOptions  # noqa: E402
from limnex_preflight.scanner import scan_paths  # noqa: E402

RAW = []  # (repo, display, location, rule, raw value, context)


def install_raw_capture(repo_name_ref):
    orig = ctxmod.ScanContext._add_hit

    def patched(self, h, text, location, line_numbers, line_offset, newlines=None):
        s, e = h.start, h.end
        ctx = text[max(0, s - 60): e + 60].replace("\n", "\\n")
        RAW.append({"repo": repo_name_ref[0], "file": self.display, "location": location, "rule": h.rule.id,
                    "severity": h.severity.label, "value": h.value[:200], "context": ctx[:300]})
        return orig(self, h, text, location, line_numbers, line_offset, newlines)

    ctxmod.ScanContext._add_hit = patched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "hf_findings.json"))
    ap.add_argument("--dump-raw")
    args = ap.parse_args()
    if args.dump_raw and os.path.realpath(args.dump_raw).startswith(REPO + os.sep):
        print("refusing to write raw third-party context inside the repository", file=sys.stderr)
        return 2
    manifest = json.load(open(args.manifest))
    repo_ref = [""]
    if args.dump_raw:
        install_raw_capture(repo_ref)
    report = {}
    for repo in sorted(manifest):
        d = os.path.join(args.cache, repo.replace("/", "__"))
        if not os.path.isdir(d):
            continue
        repo_ref[0] = repo
        t = time.time()
        res = scan_paths([d], ScanOptions())
        secs = time.time() - t
        kinds = Counter(f.kind for f in res.files)
        strip = d + os.sep
        report[repo] = {
            "sha": manifest[repo].get("sha"),
            "files_scanned": len(res.files),
            "kinds": dict(kinds),
            "seconds": round(secs, 2),
            "findings": [
                {**f.to_dict(), "file": f.file.replace(strip, "")} for f in res.findings
            ],
            "notes": [{"kind": n.kind, "file": n.file.replace(strip, ""), "message": n.message} for n in res.notes if n.kind in ("error",)],
            "skipped": [{"file": n.file.replace(strip, ""), "message": n.message} for n in res.notes if n.kind == "skipped"],
        }
        print(f"{repo}: {len(res.files)} files, {len(res.findings)} findings, {secs:.1f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    if args.dump_raw:
        with open(args.dump_raw, "w") as fh:
            json.dump(RAW, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
