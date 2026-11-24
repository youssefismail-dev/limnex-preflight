# Changelog

## 0.1.0 (2026-11-24)

The first public release of Limnex Preflight (developed privately as `limnex-modelscan`).

* **Scanner** (`limnex-preflight PATH...`): a static scan of ML model files, notebooks and configs for secrets,
  personal data and unsafe pickles. It never loads or executes a file, has no dependencies and makes no network calls.
  It reads about 20 formats (PyTorch, pickle, joblib, NumPy, safetensors, ONNX, Keras/HDF5, CatBoost, GGUF, notebooks,
  JSON/YAML/text, and zip/tar/gz/bz2/xz archives, recursively). Reports show masked values only. Files it cannot read are
  listed as "Not scanned" and never pass silently.
* **Reports:** human-readable, JSON and SARIF 2.1.0. Exit codes 0 / 1 / 2 / 3. Findings are accepted one by one through
  fingerprints in `.preflightignore`.
* **GitHub Action** (`uses: youssefismail-dev/limnex-preflight@v0.1.0`): job summary, annotations, optional SARIF upload,
  and a changed-files-only mode. On pull requests the ignore file is read from the base commit. The default `fail-on` is
  `medium`, and the action's PyPI installs are pinned to exact versions.
* **pre-commit hook** `limnex-preflight`.
* **Evaluation**, with its limits stated in the README: planted fixtures (a regression suite), 80 public Hugging Face
  repositories (tuning and held-out), a comparison with picklescan and ProtectAI `modelscan` on the author's own cases,
  and mutation fuzzing of every parser (`evaluation/results/fuzz.md`). It has not been tested on real-world malicious
  model files.
* **Dogfooding on the author's own ML projects** before release (about 14 GB, 1,366 files, including 57 PyTorch, 12
  joblib and 3 CatBoost models: no unsafe-code false alarm). It found the author's Kaggle user name inside a published
  model, and four problems in the scanner, all fixed: non-ASCII UTF-8 text files with an unlisted extension were
  reported "not scanned"; `.py`/`.sh` files are now always text; URL user-info (`http://a.com@b.com`, scheme-less
  `user@host/path`) is no longer reported as an e-mail address; a file name containing `api_key` (a manifest of
  hashes) or a call taking a string literal (`re.compile(r"...")`) is no longer a credential.
* **Second pre-release review (2026-11-11)**, three fixes found with harmless test files:
  * the bytes after a joblib `NumpyArrayWrapper`, skipped as array data, are also read the way a plain pickle loader
    reads them (as instructions);
  * a PyTorch-style archive that Python's zip reader cannot read (a changed CRC, an unknown "version needed", a local
    name differing from the central one), but that PyTorch may still load, is a MEDIUM finding,
    `unscannable-model-archive`, so the default gate fails instead of only warning;
  * a later pickle in protocol 0 or 1 in the same file is analysed, not only carved as trailing data;
  * the other readers: new MEDIUM rules `onnx-external-data-path` (ONNX external data outside the model's folder) and
    `unsafe-yaml-python-tag` (YAML tags that Python's unsafe YAML loaders turn into objects); the bytes after the last
    safetensors tensor are scanned; HDF5 parts that used to be skipped silently are reported as reduced coverage.
* **Independent pre-release test (2026-11-19)**, a report-only review of the release tree. Everything it found is fixed,
  each with a test:
  * a damaged Keras/HDF5 file could make `h5py` loop forever on Linux, so a scan or a CI job hung: `h5py` now runs in a
    child process that is stopped after 10 s without progress, and the file is scanned as raw strings; the fuzz run now
    covers Keras/HDF5 and ONNX (6,300 files);
  * on a pull request whose base commit is missing (the default shallow checkout) the Action used the pull request's own
    ignore file, so a pull request could silence its own finding: it now fetches the base commit, and otherwise applies no
    ignore entry;
  * a whole protocol-0/1 pickle is analysed as a pickle whatever its name (`.bin`, `.p`, `model.data`, no extension), since
    a loader does not look at the name;
  * a compressed tar (`model.tar.gz` and friends) is read member by member, so one larger than 512 MB is still scanned;
  * a folder that cannot be listed is an error, not silently skipped; a path that does not exist is a usage error (2);
  * the pre-commit hook covers every file type the scanner handles (it skipped `.env`, `.p`, `.py` and more);
  * every Python call in the Action runs with `-I`, the install step included;
  * new: `--exclude` (and the Action's `exclude` input), with nothing left out by default; symbolic links are followed
    where it is safe (`--follow-symlinks` for links out of the scanned folder; never out of the Action's workspace);
  * smaller: a UTF-8 byte-order mark or a non-UTF-8 ignore file, two reports on stdout, a closed pipe, the "scanned N
    files" count, `--max-depth` below 1, warnings for Action inputs that do nothing on their own, the release checking the
    tag against `__version__`, licence metadata and a source distribution without the test suite;
  * the evaluation publishes only totals for the real-model scan (no per-repository results, which could be matched to
    the named lists even under pseudonyms), so no finding appears next to a person's or a team's name; the lists still
    name the scanned public repositories.
* **Second pre-release test (2026-11-20)**, a re-check of those fixes and a full retest. Everything it found is fixed or,
  where noted, documented:
  * the Action never writes its SARIF report through a symbolic link (a pull request could commit one at that path);
  * on pull requests the base commit's ignore file is read like the CLI reads it (a byte-order mark failed every run);
  * `exclude` also applies with `changed-only`; what is excluded is listed as "Left out on purpose" and, being a choice,
    does not count as incomplete; a link to a place that is scanned anyway does not either;
  * inside the scanned folder, a file reached through symbolic links is scanned once (a Hugging Face cache blob, which
    has a hash name, under the link's name; see the third test for the final rule), and a link into an excluded place is
    reported as excluded;
  * one file count everywhere (JSON `files_scanned` / `files_not_scanned`, the new Action output `files-not-scanned`, the
    job summary); "nothing was checked" when no file was scanned;
  * yes/no Action inputs accept only `true` / `false`; `-o -` counts as stdout; an unwritable report path, and a broken
    symbolic link named on the command line, are usage errors (2);
  * the pre-commit hook also passes upper-case extensions, `.env.*` and files without an extension;
  * the text seeds of the fuzz run are written with the same bytes on every system, and the run was repeated with every
    optional library loadable;
  * documented, not changed: each tar member up to 512 MB is read whole (planned for 0.2), and an ONNX file cut exactly
    between two top-level fields is a complete, valid file that no reader can tell from the original.
* **Third pre-release test (2026-11-20)**, which found two blockers, both through a symbolic link committed to the scanned
  repository. Everything it found is fixed:
  * a link's name never decides how a file is read: the format is judged by the file's own name and content, and a link's
    name is used only for a file without an extension (`notes.txt -> model.joblib` had made the scanner read the joblib
    as text, so its pickle was never analysed); compressed pickles (zlib, bz2, joblib's lzma) are recognised by their
    content under any name, which also covers a renamed file without any link;
  * the Action never reads the ignore file through a symbolic link (in the checkout or, on a pull request, in the base
    commit), and an invalid ignore entry is reported by its line number only, so no file content reaches the log;
  * private IPv4 addresses next to words such as "server" or "driver", or in `user@address`, are reported;
  * named pipes, sockets and devices are listed as not scanned instead of hanging the scan;
  * the Action's job summary says "nothing was checked" when no file was scanned; `changed-only` treats links as a full
    scan does; yes/no inputs with surrounding spaces are refused (the upload step would have read them as false);
  * in the public tree, any file that is meant to stay private makes the test suite fail;
  * the fuzz seeds no longer carry the time or the writing system, each format has its own random stream, and
    `fuzz.md` lists every seed's SHA-256;
  * the evaluation of 2026-11-05 is published as totals only (its per-repository rows stay private), and the README
    states the tar memory cost (about twice a member), advises `--no-ignore` for someone else's files, and no longer ties
    a benign finding to one repository.
* **Fourth pre-release test (2026-11-20)**, which found one blocker and showed that three fixes of the third test were
  incomplete. Everything it found is fixed:
  * a joblib file written with pickle protocol 0 or 1 is analysed under any name and inside any of joblib's wrappers
    (zlib, gzip, bz2, xz, lzma): the zlib and lzma streams are recognised by a trial decompression whatever they hold,
    joblib's NumPy arrays after a protocol-0/1 wrapper are skipped the way joblib reads them, and content inside a joblib
    wrapper that does not parse as a pickle is a MEDIUM finding, not a note (such files had passed with exit 0, also
    through `model.joblib -> notes.txt`);
  * `changed-only` lists the full scan's walk and keeps what concerns a changed file, so links, `exclude` and
    fingerprints are decided exactly as in a full scan (a changed link out of the scanned folder had been scanned);
  * no link is followed into `.git`, `.hg` or `.svn`; a hash-named file reached through several links is read under the
    link name that says most about it (`model.onnx` before `a_copy.txt`);
  * on `pull_request_target`, `changed-only` compares with the pull request's head when it is checked out, and scans
    everything otherwise (it compared the base with itself and scanned nothing);
  * file names that are not UTF-8 are shown as `\xNN` in every report (they crashed SARIF, `--json FILE` and the job
    summary); a report never overwrites a path to scan or a file that is not a report of this tool (`--sarif model.pkl
    other/` had replaced the pickle);
  * a cut zlib stream and a cut `.npy` header are errors (exit 3), like a cut gzip stream; "not scanned" is one number
    everywhere, the length of the "Not scanned" list;
  * the published study of 2026-11-05 gives no count that can be subtracted into a label for keys, the manifests keep no
    file list for a pseudonymised repository, and no benign finding is tied to one repository any more; `fuzz.md`
    records the zlib build; `[test]` installs PyYAML, so the workflow check runs there too.
* **Fifth pre-release test (2026-11-20): passed with notes**, no blocker. Fixed from its notes:
  * NumPy string arrays inside pickles and joblib files are read, element by element (a classifier's class labels
    holding an e-mail address or a home path came back clean);
  * an LZ4 frame (joblib's `compress="lz4"`, which needs a dependency to read) is a MEDIUM finding under any name instead
    of being read as text; a pickle named `.onnx` or `.safetensors` is analysed as a pickle;
  * a private-key header alone (a string in code, a sentence) is no longer CRITICAL: a key body must follow, and the
    masked value says "[body redacted]" once;
  * on a pull request, a problem in the pull request's own ignore file (which is not applied there) no longer stops the
    run before the scan: everything is scanned and reported, then the run exits 2; the Action pins pip's build
    environment with `PIP_BUILD_CONSTRAINT` too (pip 26.2+);
  * the report-overwrite guard accepts only a JSON or SARIF report inside a scanned folder and knows a text report by its
    exact summary line; a string-array `.npy` cut inside its data is an error; an annotation keeps a backslash that is
    part of a Linux file name;
  * the 2026-11-05 study gives "4 false or unclear" as one number; the README's tar memory note (up to about three times
    a member) and test count are corrected.
* **User review before launch (2026-11-21):** an ML engineer's review, free in how it tested (7/10). Fixed from it:
  * credentials set through environment variables in code (`os.environ['WANDB_API_KEY'] = ...`, `setdefault`,
    `putenv`), `wandb.login(key=...)` and the W&B entry in `~/.netrc` are found (every rule missed them);
  * a Git LFS pointer is a MEDIUM finding (`git-lfs-pointer`): `actions/checkout` fetches LFS files only with
    `lfs: true`, which the examples now set, so a CI check could read only pointers and pass;
  * files found but not one read ("nothing was checked") is exit 3 in the CLI; a Hugging Face cache snapshot is read
    without `--follow-symlinks`;
  * a compressed tar named `.tar` is read without a damage note; an archive member name that leaves the extraction
    folder is a MEDIUM finding (`archive-path-traversal`);
  * `--list-rules` shows each rule's general description; the README opens with a quick start.
* **The same review, retested on release candidate 8 (2026-11-21, 8/10).** It confirmed the fixes above and found two
  problems that release candidate 8 itself had introduced, plus two older ones. All four are fixed, each with a test:
  * a Hugging Face cache built by a newer `huggingface_hub` keeps Xet-backed files in a cache-wide store
    (`<cache>/blobs/<2 hex>/<hash>`), and the repository's `blobs/<etag>` entry links onward to it. Only links into the
    repository's own `blobs/` were followed, so most of the repository's files were skipped and the scan
    passed. Links into the cache-wide store are followed too;
  * with `h5py` installed, a `.keras` model's weights member was read whole and piped to the HDF5 child, so memory grew
    to about twice the model's size (1.55 GB for a 763 MB model). An uncompressed member (Keras's default) is now opened
    in place (a 554 MB-weights model: 1,133 MB -> 79 MB). A compressed one is piped in 1 MB pieces, but the HDF5 reader
    still holds the whole member, so it costs about the weights' size in memory (planned for 0.2);
  * the machine name in a Windows Trainer run folder (`out\runs\<date>_<host>`, also written `\\` in JSON) is found,
    like the `/` form;
  * the limit of 200 matches of one rule per text was silent (a CSV with 400 e-mail addresses showed 200): passing it
    is now said under "Reduced coverage".
* **A follow-up to the cache fix above (2026-11-22):** in the shared-blob layout a cache repository
  holds two links to each file (`snapshots/<rev>/<name>` and `blobs/<etag>`), both leading out of the folder, and only
  links to places inside the scanned folder were read once. Scanning a whole `models--org--name` folder read every file
  once per link and reported each finding two or more times, one of them under a hash name. Followed links that lead out
  of the folder are now read once per file too, under the snapshot's name; this also covers two `--follow-symlinks`
  links to one file, which was planned for 0.2.
* **The same review's second retest, on release candidate 11 (2026-11-23, 8/10).** All five items above passed (the cache
  with the shared blob store, one read per file, `.keras` memory, Windows run folders, the 200-match note) and nothing it had
  confirmed before had broken. Fixed from it, each with a test:
  * the Action passed (exit 0, with a warning) when a full scan found files but could read none of them, where the CLI
    exits 3. It now exits 2 ("nothing was checked"). An empty folder still only warns, and with `changed-only` a change
    that touches no scannable file still passes;
  * huggingface_hub's own cache metadata (the shared blob store's `<hash>.refs` lists and a repository's
    `trees/<commit>.json` file lists) is left out on purpose and listed as such: it only repeated the names of files that
    are scanned through the snapshots, as LOW noise.

## Planned for 0.2

* From the user review: fewer review findings on standard fastai learners (12 of 16 MEDIUM findings on real fastai
  repositories were `pickle-getattr-review` false alarms; the allow-list needs to know the owning module); TensorBoard
  event files; pickle findings located by field name (`TrainingArguments.output_dir`) as well as by offset; notebook
  annotations on the line of the raw `.ipynb`; a shorter README with the evaluation moved to `docs/`. From its retest:
  a Keras Lambda layer in an `.h5` file found from the raw `model_config` when `h5py` is not installed; repeated
  findings grouped per file and rule; overlapping command-line paths (`x.pkl` and `./x.pkl`) scanned once. From its second
  retest: a compressed `.keras` weights member read in pieces by the HDF5 reader too (it is held whole today).
* Faster on very long single tokens (a 50 MB run of word characters takes about a minute).
* An allow-list review tier for pickles: any import outside known-safe ML modules is reported for review, so an
  attacker gains nothing by picking a callable the deny-list does not know. Measured on the held-out sets before it ships.
* GGUF chat templates: judge the template code, not only secrets and personal data in it.
* HDF5: scan large string datasets in chunks, and fall back to raw strings for parts h5py cannot read.
* ONNX: read the `training_info` graphs.
* Read the members of a PyTorch-style archive that Python's zip reader rejects, the tolerant way PyTorch does.

Known limits are listed under "What it cannot find" in the README.
