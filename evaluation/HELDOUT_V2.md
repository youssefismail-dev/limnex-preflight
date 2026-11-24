# Held-out set v2: sampling rule (written and committed BEFORE any model file was downloaded)

Status when this file was committed: **no file of any sampled repository has been downloaded.** The only network use so far
was the connectivity check and reading the public model *listing* (names, tags, dates, counts) to learn which API filters
exist. Nothing below was tuned on a result. Any deviation from this text that happens later is recorded in the
"Deviations" section at the bottom, in the commit that makes it, never by editing the rule above it.

Purpose: one more **benign-only, held-out** measurement of the scanner as it stood on 2026-11-05, on ~60 ordinary, recent Hugging Face model
repositories that were not used in README sections 2-5. It says nothing about malicious files.

## 1. Fixed parameters

| | |
|---|---|
| seed | `20261105` (the date of this round; chosen before the first listing call; never changed) |
| RNG | Python `random.Random(seed)`; candidates are sorted by repository id BEFORE shuffling, so the order does not depend on API paging |
| scanner under test | `limnex_preflight` at the commit named in the report (no scanner change is allowed in this round unless a test proves a bug; the scan results are NOT re-run after such a fix, both numbers are reported) |
| listing | Hugging Face Hub API, `GET /api/models?filter=<tag>&sort=downloads&direction=-1&limit=100`, paged until 400 listings were read per stratum |
| "recent" | `createdAt` >= **2025-11-05T00:00:00Z** (the last 12 months) |
| "used" | `downloads` (the API's 30-day count) >= **100** |
| repository filters | not `gated`, not `private`, not `disabled`; id NOT in the exclusion set (section 2); no license filter |
| safety filter | repository status `securityRepoStatus.filesWithIssues` empty AND no file with a `securityFileStatus` of anything other than `safe` or absent. A repository that Hugging Face flags as unsafe OR caution is **never downloaded** and is replaced by the next one in the shuffled list. |

## 2. Exclusion set

Every repository id that appears in `evaluation/hf_models.txt` or `evaluation/hf_models_heldout.txt` (README sections 2-5), plus the
seven repositories whose ids exist only in the private mapping (they are pseudonymised in those lists). The check is by exact
id, case-insensitive. The same author may appear again with a different repository (it is a repository-level, not an
author-level exclusion). Excluded ids found during the draw are counted, not listed.

## 3. Strata, 10 repositories each (60), plus 3 stress files

A repository is assigned to the FIRST stratum (in the order below) whose listing it is drawn from; a repository drawn once
is never drawn again.

| # | stratum | listing `filter=` | repository must contain at least one file named |
|---|---|---|---|
| 1 | pickle | `pytorch` | `pytorch_model*.bin`, `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, `*.pickle` |
| 2 | safetensors | `safetensors` | `*.safetensors` |
| 3 | gguf | `gguf` | `*.gguf` |
| 4 | onnx | `onnx` | `*.onnx` |
| 5 | keras | `keras` | `*.h5`, `*.hdf5`, `*.keras` |
| 6 | joblib/sklearn | `sklearn` | `*.joblib`, `*.pkl`, `*.pickle` |

Draw procedure per stratum: read the listing (section 1), keep the metadata-eligible repositories whose file list (from the
listing, `expand[]=siblings`) contains a stratum file, sort by id, shuffle with the seeded RNG, then walk the shuffled list:
for each repository fetch `GET /api/models/<id>?blobs=true&securityStatus=true`, apply the size rule (section 4) and the
safety filter, and accept it until 10 are accepted. If fewer than 10 can be found within the 400 listings, the stratum is
reported short; it is **not** widened.

**Stress stratum (3 files, task 4):** drawn the same way, separately, from the `safetensors` and `gguf` listings, among
repositories that satisfy every filter above and contain one `*.safetensors` or `*.gguf` file of **3 GB to 8 GB**: seeded
draw, 2 safetensors + 1 gguf. Only that single file is downloaded (it is streamed to disk and deleted after the scan). They
are not part of the 60 and are reported separately, but they are in the set in the sense of task 4: the largest files.

## 4. What is downloaded (size cap declared in advance)

* **format files** of the repository's stratum (the patterns above; for the pickle stratum all five pickle patterns): sorted by
  (size, path); take them in that order until 6 files or **1.0 GB in total**, never one file above **1.0 GB**. A repository
  whose smallest format file is above 1.0 GB is ineligible and the walk continues. (So large single-file models are
  under-represented; the stress stratum exists to cover them.)
* **text files** `*.json *.yaml *.yml *.txt *.md *.cfg *.ini *.toml *.ipynb` up to 5 MB each, at most 30 per repository
  (sorted by path). `README.md` and `config.json` are always included when present.
* Nothing else is downloaded. Files are **downloaded in full** (no header-only or sparse stand-ins), because truncated
  safetensors/GGUF files are now correctly reported as incomplete.
* Per repository: download -> scan with all three tools (section 5) -> delete. The cache holds one repository at a time
  (the stress files: one file at a time). Nothing downloaded is imported, unpickled or executed; every Python process that
  reads downloaded data runs with `-I`.

## 5. Measurements and analysis (fixed in advance)

1. **Scanner run:** `limnex-preflight --format json --fail-on none <dir>`, one process per repository, wall time and peak
   RSS from `wait4()` of that process. Also `--fail-on-incomplete`'s notion of incomplete (error / degraded / "not scanned")
   counted per file format.
2. **Report:** files scanned; error, degraded and not-scanned counts per format; findings per rule and severity; seconds per
   GB and peak RSS (maximum over repositories, and the ratio to the largest single file).
3. **Labels:** every finding of severity MEDIUM or higher is labelled by hand as `true` (a real disclosure of the kind the rule
   means), `intended` (a real value that the owner published on purpose, e.g. a contact address in a README) or `false`
   (not what the rule describes). The labeller sees rule, severity, file *type*, location and the **masked** value, and
   nothing else; raw values are not read, printed or stored. When the masked evidence is not enough, the label is
   `unclear` and counted against the scanner (as false) in the headline figure and reported separately.
   Anything that looks like a live secret is recorded as "possible live secret, pseudonym X" with no value.
4. **Rates:** false-positive rate = false / (all MEDIUM+ findings), and the share of repositories with at least one false MEDIUM+
   finding; both with **Wilson 95% intervals**. Zero events are reported with their upper bound, not as "0%".
5. **Comparison:** `picklescan` (latest from PyPI) and ProtectAI `modelscan` (latest from PyPI) on the same files before
   deletion. Their verdict per file is "issues reported / none / error or not supported". Agreement table for the unsafe-code
   verdict; every disagreement is explained from the finding's rule and masked/structural context.
6. **Names:** committed results use pseudonyms `community-repo-101` ... (numbered in draw order, which is not alphabetical)
   and `file-N.ext`; no real repository id, file name, commit, hash or exact size is committed. The real draw is kept in a
   private scratch file outside the repository.

## 6. Limits stated up front

* Benign only. The safety filter removes anything Hugging Face flags, so this set cannot contain a malicious pickle that
  Hugging Face already detects; it says nothing about real attacks.
* Popular-recent repositories (sorted by downloads) are not "all of Hugging Face".
* The API listing changes daily, so the draw cannot be repeated exactly later; the seed and the code make it repeatable
  against a stored listing, which is kept privately.
* Size cap: models above 1.0 GB per file are under-represented in the 60 (and covered by the 3 stress files).
* One labeller (the author of the scanner's evaluation, an AI assistant), working from masked evidence.

## Deviations

**D1 (before any download, found by a listing-only dry run).** Section 1 says "paged until 400 listings were read per stratum".
The dry run read the first 400 listings of each filter, sorted by downloads, and counted the metadata-eligible ones
(recent, >= 100 downloads, not gated/private/disabled): pickle 19, safetensors 121, gguf 322, onnx 53, keras 116, sklearn 4.
Pickle and sklearn cannot reach 10 repositories after the file, safety, size and exclusion filters from such pools. The
pool depth was a flaw of the rule, found from pool *counts* only (no model file, no scan result, no finding was seen).
Replacement for that one sentence: **"paged until 5000 listings were read or 250 metadata-eligible repositories were found,
whichever comes first"**. Everything else (sort, filters, seed, shuffle, size rule) is unchanged. The probe listings are
kept in the private scratch area.

**D2 (before any download, found from pool counts only).** The first draw filled five strata with 10 repositories each and the
3 stress files, but stratum 6 (joblib/sklearn) found **0**: the whole `sklearn` tag holds 1,201 listings, and only **1** of
them was created in the last 12 months with >= 100 downloads in 30 days (most scikit-learn repositories have 0 downloads in
that window). The rule says a short stratum is reported short and not widened; widening is the better scientific choice here
only because a format with no sample at all cannot be evaluated, so it is done openly, for this stratum only:
**stratum 6 has no download floor (`downloads >= 0`); all other filters, the 12-month recency cut-off, the exclusion set, the
safety filter, the size rule, the seed and the shuffle are unchanged.** The listing is exhausted, so every recent
repository of the tag that has a `*.joblib/*.pkl/*.pickle` file is a candidate (586 of them). These repositories are less
used than the others, which is a limit of this stratum and is stated in the report. Strata 1-5 and the stress draw are
re-run unchanged by the same code (the listings are cached).

**D3 (after the first full pass, before any number was reported).** The first pass measured peak memory with `wait4()`'s
`ru_maxrss`. On Linux that value includes the PARENT's resident memory at fork, so every tool showed the driver's own heap as a
floor (61.7, 69.7, 75.7 ... 126.0 MB, rising with the driver). The numbers were discarded. The driver now polls the child's own
`VmHWM` and bytes read (`/proc/<pid>/status`, `/proc/<pid>/io`) and the whole set was downloaded and scanned a SECOND time
(same pinned commits). The second pass produced identical finding fingerprints for all 63 repositories. All reported
numbers come from the second pass.

**D4 (second pass).** ProtectAI `modelscan` had been installed without its `h5py` and `tensorflow` extras in the first
pass, so it could not read `.keras` files (56 DEPENDENCY errors). The extras were installed and the second pass used them.
The second pass also ran `picklescan` and `modelscan` on every pickle-bearing file separately, to get per-file verdicts.

**D5 (labelling).** Two MEDIUM review findings in one pickle were first labelled `unclear` from the masked value alone. To
look further, the first 400 bytes of that pickle were disassembled and only the *opcode names and class/attribute names* were
read (no values): the pickle applies `getattr(keras.src.models.functional.Functional, '_unpickle_model')` to an in-memory
archive, the standard Keras 3 pickle loader. Both findings were relabelled `false`. This is a deviation from "masked evidence
only"; it involved no personal data and no credential, and it is reported as such.

**D6 (two scanner bugs found and fixed, each with a failing test first).** See the report: `hostname-field` matched the
PyTorch exporter's `pkg.torch.onnx.original_node_name`, and `--deep` stopped silently at its 4 GiB cap. The labelled results
above were produced by the scanner BEFORE both fixes (the 2026-11-05 version); the effect of each fix on the same real files is reported
separately.
