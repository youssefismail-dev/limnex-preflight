"""Build the PUBLIC evaluation files from PRIVATE real-id inputs.

Why: the study found real personal data (user names, host names, paths) in community repositories. The published
results must show how many leaks of which kind the tool found, not point at individuals before they could be told
privately. Every repository that has a ``leak`` verdict anywhere (any result set) is replaced, EVERYWHERE, by a random
pseudonym ``community-repo-NNN``:

  * the model lists (``hf_models.txt``, ``hf_models_heldout.txt``),
  * the manifests (the file list, commit SHA and download count of a pseudonymised repository are dropped),
  * every result file (repository keys and every file path, including archive member names that echo the file name).

Otherwise (list minus results) would reveal the leaking repositories. The real ids exist only in the private mapping
file, which is refused if it would be written inside this repository.

Usage:
  python evaluation/pseudonymize_results.py --private-dir PRIVATE --mapping PRIVATE/mapping.json

PRIVATE must contain: hf_models.txt, hf_models_heldout.txt, hf_manifest_tuning.json, hf_manifest_heldout.json and the five
result files hf_{v0_tuning,v0_heldout,heldout_first_look,final_tuning,final_heldout}.json produced by run_hf_scan.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hf_review  # noqa: E402

PSEUDO_REPO = re.compile(r"^community-repo-\d+$")
PSEUDO_FILE = re.compile(r"^file-\d+(\.[a-z0-9]{1,12})?$")
RESULT_TAGS = [t for t, _ in hf_review.SETS]
MANIFESTS = {"tuning": "hf_manifest_tuning.json", "heldout": "hf_manifest_heldout.json"}
LISTS = {"hf_models.txt": "tuning", "hf_models_heldout.txt": "heldout"}


class Mapper:
    """repo -> pseudonym, and (repo, relative path) -> file-N.ext. Stable across re-runs through the private mapping."""

    def __init__(self, stored: dict):
        self.repo_to_pseudo: Dict[str, str] = {v["repo"]: k for k, v in stored.get("repos", {}).items()}
        self.files: Dict[str, Dict[str, str]] = stored.get("files", {})  # pseudonym -> {real path: fake name}
        self.rnd = random.SystemRandom()

    def repo(self, real: str) -> str:
        if real not in self.repo_to_pseudo:
            taken = {int(p.rsplit("-", 1)[1]) for p in self.repo_to_pseudo.values()}
            n = self.rnd.choice([i for i in range(100, 1000) if i not in taken])
            self.repo_to_pseudo[real] = f"community-repo-{n}"
        return self.repo_to_pseudo[real]

    def file(self, pseudo: str, path: str) -> str:
        table = self.files.setdefault(pseudo, {})
        if path not in table:
            ext = os.path.splitext(path)[1].lower()
            ext = ext if re.fullmatch(r"\.[a-z0-9]{1,12}", ext) else ""
            table[path] = "file-%d%s" % (len(table) + 1, ext)
        return table[path]

    def scrub_chain(self, pseudo: str, display: str) -> str:
        """'dir/model.bin!/model/data.pkl' -> 'file-3.bin!/file-3/data.pkl' (the member folder echoes the file stem)."""
        top, sep, rest = display.partition("!/")
        fake = self.file(pseudo, top)
        if rest:
            stem = os.path.splitext(os.path.basename(top))[0]
            fake_stem = os.path.splitext(fake)[0]
            if stem:
                rest = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(stem) + r"(?![A-Za-z0-9_])", fake_stem, rest)
        return fake + sep + rest

    def dump(self) -> dict:
        return {
            "repos": {p: {"repo": r} for r, p in self.repo_to_pseudo.items()},
            "files": self.files,
        }


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private-dir", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    if os.path.realpath(args.mapping).startswith(os.path.realpath(REPO) + os.sep):
        print("refusing to write the de-anonymisation mapping inside the repository", file=sys.stderr)
        return 2
    if os.path.realpath(args.private_dir).startswith(os.path.realpath(REPO) + os.sep):
        print("the private directory must be outside the repository", file=sys.stderr)
        return 2
    stored = load_json(args.mapping) if os.path.exists(args.mapping) else {}
    if stored and "repos" not in stored:  # first-generation mapping: {pseudonym: real repo}
        stored = {"repos": {p: {"repo": r} for p, r in stored.items()}, "files": {}}
    m = Mapper(stored)

    results = {t: load_json(os.path.join(args.private_dir, f"hf_{t}.json")) for t in RESULT_TAGS}
    leak_repos = sorted({repo for d in results.values() for repo, e in d.items() for f in e["findings"]
                         if hf_review.verdict(f) == "leak"} | set(m.repo_to_pseudo))
    pseudo = {r: m.repo(r) for r in leak_repos}

    os.makedirs(args.results, exist_ok=True)
    # ---- result files
    for t, d in results.items():
        out = {}
        for repo, e in d.items():
            if repo in pseudo:
                p = pseudo[repo]
                e = json.loads(json.dumps(e))
                for f in e["findings"]:
                    f["file"] = m.scrub_chain(p, f["file"])
                    if hf_review.verdict(f) == "leak":
                        f["value_masked"] = f["value_masked"][:2] + "…"  # even the masked form says less about a person
                for key in ("notes", "skipped"):
                    for n in e.get(key, []):
                        n["file"] = m.scrub_chain(p, n["file"])
                e.pop("sha", None)
                repo = p
            out[repo] = e
        with open(os.path.join(args.results, f"hf_{t}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    # ---- manifests
    for name, fn in MANIFESTS.items():
        man = load_json(os.path.join(args.private_dir, fn))
        out = {}
        for repo, e in man.items():
            if repo in pseudo:
                p = pseudo[repo]
                # no file list at all: even pseudonymous names keep each repository's file count and extension mix,
                # which can be matched against the Hub's file listings (pre-release test 4, S9). Hashes, sizes, the
                # commit and the download count went earlier.
                e = {"sha": None, "downloads": None, "files": [], "pseudonymised": True}
                repo = p
            out[repo] = e
        with open(os.path.join(args.results, fn), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    # ---- aggregate totals (computed from the private manifests, so pseudonymised repos still count)
    tot = {"repos": 0, "files_downloaded": 0, "bytes_downloaded": 0, "partial_files": 0, "hub_bytes_of_partial_files": 0}
    for fn in MANIFESTS.values():
        for e in load_json(os.path.join(args.private_dir, fn)).values():
            tot["repos"] += 1
            for f in e.get("files", []):
                if "bytes_downloaded" in f:
                    tot["files_downloaded"] += 1
                    tot["bytes_downloaded"] += f["bytes_downloaded"]
                    if f.get("partial"):
                        tot["partial_files"] += 1
                        tot["hub_bytes_of_partial_files"] += f.get("size_on_hub", 0)
    tot["scan_seconds_final_code"] = round(sum(e["seconds"] for t in ("final_tuning", "final_heldout") for e in results[t].values()), 1)
    with open(os.path.join(args.results, "hf_totals.json"), "w", encoding="utf-8") as fh:
        json.dump(tot, fh, indent=1)
    # ---- model lists (comments kept, ids replaced in place)
    for fn in LISTS:
        lines: List[str] = []
        with open(os.path.join(args.private_dir, fn), encoding="utf-8") as fh:
            for line in fh:
                key = line.strip()
                lines.append(pseudo[key] + "\n" if key in pseudo else line)
        with open(os.path.join(HERE, fn), "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    with open(args.mapping, "w", encoding="utf-8") as fh:
        json.dump(m.dump(), fh, indent=1)
    print(f"{len(pseudo)} leaking repositories pseudonymised; real ids only in {args.mapping}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
