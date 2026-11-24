"""Download the evaluation set from Hugging Face. Network is only used here, never by the scanner.

Rules (so the run is reproducible and bounded):
  * text-like files (.json .yaml .yml .txt .md .cfg .ini .toml .ipynb) up to 5 MB are downloaded in full
  * weight files (.bin .pt .pth .ckpt .pkl .pickle .joblib .npz .npy .h5 .keras .onnx .cbm) up to 60 MB, at most
    8 per repo and 250 MB per repo
  * .safetensors: only the header (8-byte length + JSON) via an HTTP Range request
  * .gguf: only the first 16 MB via an HTTP Range request
  * everything else is not downloaded
Nothing downloaded is ever imported, unpickled or executed. The manifest records repo commit SHA, sizes and SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request

TEXT = (".json", ".yaml", ".yml", ".txt", ".md", ".cfg", ".ini", ".toml", ".ipynb")
WEIGHTS = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib", ".npz", ".npy", ".h5", ".keras", ".onnx", ".cbm")
TEXT_MAX = 5 * 1024 * 1024
WEIGHT_MAX = 60 * 1024 * 1024
GGUF_HEAD = 16 * 1024 * 1024
PER_REPO_FILES = 8
PER_REPO_BYTES = 250 * 1024 * 1024
UA = {"User-Agent": "limnex-preflight-eval/0.1"}


def api(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch(url: str, dest: str, rng: str | None = None) -> bytes | None:
    headers = dict(UA)
    if rng:
        headers["Range"] = rng
    req = urllib.request.Request(url, headers=headers)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def plan(tree):
    """Choose files to download. Returns [(path, size, mode)] mode in full|safetensors|gguf."""
    files = [(t["path"], t.get("size") or 0) for t in tree if t["type"] == "file"]
    chosen, weights, budget = [], 0, 0
    for path, size in sorted(files):
        low = path.lower()
        if low.endswith(TEXT) and size <= TEXT_MAX:
            chosen.append((path, size, "full"))
    for path, size in sorted(files, key=lambda x: x[1]):
        low = path.lower()
        if low.endswith(WEIGHTS) and size <= WEIGHT_MAX and weights < PER_REPO_FILES and budget + size <= PER_REPO_BYTES:
            chosen.append((path, size, "full"))
            weights += 1
            budget += size
    st = [p for p in sorted(files) if p[0].lower().endswith(".safetensors")][:4]
    chosen += [(p, s, "safetensors") for p, s in st]
    gg = [p for p in sorted(files, key=lambda x: x[1]) if p[0].lower().endswith(".gguf")][:2]
    chosen += [(p, s, "gguf") for p, s in gg]
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "hf_models.txt"))
    ap.add_argument("--cache", required=True, help="directory for downloads (untrusted data; keep outside the repo)")
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "results", "hf_manifest.json"))
    args = ap.parse_args()
    repos = [l.strip() for l in open(args.models) if l.strip() and not l.startswith("#")]
    manifest = {}
    for repo in repos:
        if repo.startswith("community-repo-"):
            print(f"{repo}: pseudonymised in the public list (real id kept privately); skipped", flush=True)
            continue
        d = os.path.join(args.cache, repo.replace("/", "__"))
        try:
            info = api(f"https://huggingface.co/api/models/{repo}")
            tree = api(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true")
        except Exception as exc:
            print(f"!! {repo}: {type(exc).__name__}: {exc}", file=sys.stderr)
            manifest[repo] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        entry = {"sha": info.get("sha"), "downloads": info.get("downloads"), "files": []}
        for path, size, mode in plan(tree):
            url = f"https://huggingface.co/{repo}/resolve/{info['sha']}/{urllib.parse.quote(path)}"
            dest = os.path.join(d, path)
            try:
                if mode == "full":
                    digest, n = fetch(url, dest)
                    partial = False
                elif mode == "gguf":
                    digest, n = fetch(url, dest, f"bytes=0-{GGUF_HEAD - 1}")
                    partial = n < size
                else:  # safetensors header
                    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-7"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        n_hdr = struct.unpack("<Q", r.read(8))[0]
                    if n_hdr > 100 * 1024 * 1024:
                        raise ValueError("implausible safetensors header")
                    digest, n = fetch(url, dest, f"bytes=0-{8 + n_hdr - 1}")
                    partial = True
                entry["files"].append({"path": path, "size_on_hub": size, "bytes_downloaded": n, "partial": partial, "sha256": digest, "mode": mode})
            except Exception as exc:
                entry["files"].append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
        manifest[repo] = entry
        print(f"{repo}: {len(entry['files'])} files", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
