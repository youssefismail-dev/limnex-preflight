"""Fail if any real identifier of a pseudonymised repository appears in a file of this repository.

Needs the PRIVATE mapping written by pseudonymize_results.py (kept outside the repo). Checks, case-insensitively, every file
under the repository root except .git:  the real repo id, the repo name, the owner (unless the owner also owns a repo that is
listed publicly), and every real file name that is not also a file name in a public repo (e.g. README.md is fine).

Usage: python evaluation/check_no_real_ids.py --mapping PRIVATE/mapping.json   (exit 1 on any hit; prints WHERE, never the id)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


# File names that tool chains write into almost every repository say nothing about whose repository it is.
GENERIC_NAMES = frozenset({
    "training_args.bin", "all_results.json", "eval_results.json", "train_results.json", "model.ipynb", "model.pickle",
    "history.pickle", "tokenizer.joblib", "pyproject.toml", "requirements.txt", "entry_points.txt", "top_level.txt",
    "dependency_links.txt", "vendor.txt", "sysconfig.cfg", "catboost_training.json", "generator.ckpt", "model-checkpoint.ipynb",
    "readme.md", "config.json", "license", "license.md", "pytorch_model.bin", "model.safetensors",
})


def forbidden_tokens(mapping: dict, public_ids: set, public_basenames: set) -> dict:
    """token (lower-case) -> kind. Kinds are reported, the tokens themselves never are."""
    public_owners = {i.split("/")[0].lower() for i in public_ids if "/" in i}
    out = {}
    for pseudo, v in mapping.get("repos", {}).items():
        real = v["repo"]
        owner, _, name = real.partition("/")
        out[real.lower()] = f"{pseudo}: repo id"
        if len(name) >= 5:
            out[name.lower()] = f"{pseudo}: repo name"
        if len(owner) >= 5 and owner.lower() not in public_owners:
            out[owner.lower()] = f"{pseudo}: owner"
        for real_path in mapping.get("files", {}).get(pseudo, {}):
            base = os.path.basename(real_path.split("!/")[0]).lower()  # directory names alone (site-packages ...) are not identifying
            if base not in public_basenames and base not in GENERIC_NAMES and len(base) >= 8:
                out[base] = f"{pseudo}: file name"
    return out


def public_sets(results_dir: str):
    ids, basenames = set(), set()
    for fn in ("hf_manifest_tuning.json", "hf_manifest_heldout.json"):
        with open(os.path.join(results_dir, fn)) as fh:
            for repo, e in json.load(fh).items():
                if not repo.startswith("community-repo-"):
                    ids.add(repo)
                    for f in e.get("files", []):
                        basenames.add(os.path.basename(f["path"]).lower())
    return ids, basenames


def scan_tree(tokens: dict, root: str):
    hits = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for fn in files:
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            hay_name = rel.lower()
            try:
                with open(path, "rb") as fh:
                    data = fh.read().decode("utf-8", "ignore").lower()
            except OSError:
                continue
            for tok, kind in tokens.items():
                if tok in data or tok in hay_name:
                    hits.append((rel, kind))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--root", default=REPO)
    args = ap.parse_args()
    with open(args.mapping) as fh:
        mapping = json.load(fh)
    ids, bases = public_sets(os.path.join(HERE, "results"))
    tokens = forbidden_tokens(mapping, ids, bases)
    hits = scan_tree(tokens, args.root)
    for rel, kind in sorted(set(hits)):
        print(f"REAL IDENTIFIER in {rel} ({kind})")
    print(f"checked {len(tokens)} forbidden tokens; {len(set(hits))} hit(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
