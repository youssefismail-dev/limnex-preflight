# Public GitHub ML repositories, round 1: sampling rule (written and committed BEFORE any clone)

Status when this file was committed: **nothing from any sampled repository has been cloned or downloaded.** Only the GitHub
search API was called once to check that it is reachable (3 repository names were printed and are not used by this rule).
Same discipline as `HELDOUT_V2.md`: deviations are listed at the bottom, in the commit that makes them.

Purpose: how does the scanner behave on the model files, notebooks and configs that people commit to GitHub, where real credentials
are far more likely than on Hugging Face model pages? **Counts only; benign-by-assumption; no value is read, printed, stored or
tried.**

## 1. Fixed parameters

| | |
|---|---|
| seed | `20261105`; candidates sorted by `full_name` then shuffled with `random.Random(seed)` |
| search (3 fixed queries, 3 pages of 100 each, `sort=stars`) | `topic:machine-learning language:"Jupyter Notebook"`; `topic:deep-learning language:Python`; `topic:pytorch` — each with `stars:20..3000 pushed:>2025-11-05 archived:false fork:false size:<150000` (size in KB: at most ~150 MB) |
| eligibility, from the repository's default-branch tree (`git/trees?recursive=1`, no clone) | at least one **model file** (`*.pt *.pth *.pkl *.pickle *.joblib *.h5 *.hdf5 *.keras *.onnx *.safetensors *.ckpt *.npz *.bin`) AND at least 3 **text-like files** (`*.ipynb *.yaml *.yml *.json *.cfg *.ini *.toml *.txt *.md`); the tree is not truncated; no git-LFS pointer files among the model files (LFS content is not fetched) |
| exclusions | any repository whose owner is one already used in the Hugging Face sets (none expected; checked by name only) |
| sample | the first 10 eligible in shuffled order |
| size cap | the files to be scanned (model + text-like files) must total at most **300 MB**; no single file above 150 MB |

## 2. What is downloaded and how

* `git clone --depth 1 --single-branch --no-recurse-submodules`, with `core.hooksPath=/dev/null`, `core.symlinks=false`,
  `GIT_LFS_SKIP_SMUDGE=1`, LFS filters disabled, `GIT_TERMINAL_PROMPT=0`. No hook, filter or submodule is ever run or fetched; nothing
  that was cloned is imported, executed, installed or unpickled. The scratch directory is outside the repository.
* The `.git` directory is deleted before scanning. The scanner runs as `python -I -m limnex_preflight --no-ignore --fail-on none --format json`
  (no ignore file of the third-party repository is honoured), with an empty working directory.
* One repository at a time: clone -> scan -> delete.

## 3. Measurements (fixed in advance)

Files scanned; errors / degraded / not scanned; findings per rule and severity; every MEDIUM+ finding labelled `true` / `intended` /
`false` / `unclear` **from rule, file type, location class and the masked value only** (`unclear` counts as false); Wilson 95%
intervals; repository-level rate. A finding that looks like a live secret is reported as "possible live secret, pseudonym X"
and nothing else. Results are published only as pseudonyms `github-repo-NNN` (numbered in draw order), file types, location
classes and counts: no repository name, file name, commit, hash or exact size. The real list stays in a private scratch file.

## 4. Limits up front

* A sample of 10, chosen by stars (so: popular enough to be "normal"), with committed model files (so: a minority of ML repositories).
* Not a measurement of detection. Whether a flagged token is live is unknown, and is not tested.
* GitHub search results change daily; the draw cannot be repeated exactly later.

## Deviations

**D1 (2026-11-05): THE ROUND WAS NOT EXECUTED. No repository was drawn, cloned or scanned, and there are no results.**
The environment the round ran in could reach only one GitHub repository. The search stage of `github_v1.py draw` ran (9
search calls returning only public names and branch names; nothing was saved and nothing was used), but every request for a
repository's tree returned HTTP 403. Fetching 10 arbitrary third-party repositories needed access the environment did not
have, so the round stopped.
Before that limit was noticed, three calls outside it had been made, all read-only and none reading repository contents: two
uses of the search API (a connectivity probe that printed 3 names, and the draw above) and one `git ls-remote` on a public
sample repository. `evaluation/github_v1.py` is written but **untested beyond the search stage** (the clone and scan path
has never run). To execute the round, run `python evaluation/github_v1.py draw ...` and `run ...` on a machine with normal
GitHub access.
