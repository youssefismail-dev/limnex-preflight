# Four leak scanners on the same 60 repositories (2026-11-05)

**Status: complete.** The protocol section below was committed (in the private development history) before the first scan; the results follow it; deviations are listed at the end.

**This public version gives totals only.** The per-repository rows (which finding is in which repository, under a
pseudonym) and the per-finding labels are kept in the private development repository: with 60 repositories drawn by
a published rule, one such row could point at one repository's finding.

## Protocol

* **Files:** the 60 main repositories of the private draw of `HELDOUT_V2.md` (same pinned commits, same files, same size caps; the 3
  stress files are skipped). Downloaded in full, scanned, deleted, one repository at a time. Real ids stay private; this public
  version gives totals only.
* **Tools and options:** `limnex-preflight` (this branch, `--no-ignore`, default options); `gitleaks detect --no-git --redact`; `trufflehog
  filesystem --no-verification --no-update --json`; `detect-secrets scan --all-files .` (default plugins). gitleaks and
  detect-secrets are built/installed from their public releases (gitleaks v8.30.1, trufflehog v3.99.8, detect-secrets 1.5.0);
  default rules, no custom config.
* **No service is ever contacted.** gitleaks, TruffleHog and detect-secrets run inside an **empty network namespace**
  (`unshare(CLONE_NEWNET)` in the child before exec), so they could not reach any service even if a flag asked them to. TruffleHog also
  gets `--no-verification` and `--no-update`.
* **No secret is printed or stored.** gitleaks runs with `--redact`; detect-secrets reports only hashes (dropped); TruffleHog's JSON
  (it contains the raw secret) is read from a pipe and only detector name, file and line are kept.
* **One finding = one tool hit** `(tool, rule, file, line)`. **One event** merges hits of different tools on the same file whose lines differ
  by at most 1; a Preflight hit without a line number (a key path, a tensor header, a pickle offset) merges with other tools' hits only if
  the same file has exactly one event from another tool (otherwise it stays separate); the rule is stated again in the results.
* **Labels:** `true` (a real disclosure of the kind the tool means), `intended` (a real value the owner published on purpose, e.g. a contact
  address in a README or licence), `false` (not what the rule describes), `unclear` (counts as false). Labels are assigned per event
  from rule names, file type, location and masked value only (Preflight masks; gitleaks and TruffleHog give none). Events that are
  identical in (file, rule set, tool set) are labelled once as a cluster and counted per event.
* **Two rows in the final table:** *keys and passwords only* (events with at least one hit from a secret/credential/entropy rule of any
  tool; the fair fight for all four) and *everything* (adds personal data: e-mail, user names, host names, IPs, notebook paths, which
  only Preflight looks for).
* **"Real leaks the others missed" for tool T** = events labelled `true` that T found and none of the other three did.
* **Possible live secrets** are disclosed privately to the owner and never published.
* **Limits stated up front:** benign, Hub-filtered sample (no malicious file); 60 repositories; one labeller working from rule names and
  masked evidence; tools' default rules only; the other tools' detector sets differ by design.

## Results

60 repositories, 528 files, 20.5 GB downloaded in full again and deleted after the scans; 0 download errors; 0 failed tool runs; **no tool
reported a verified secret and none could have: the three third-party tools ran in an empty network namespace.** The Preflight in this
comparison is the branch head after the two fixes of the earlier round (so its LOW hostname noise on exported ONNX files is gone).
Preflight's two unsafe-pickle findings (category `unsafe-code`, not leaks) are left out of this comparison: that is a scope decision I took
when I saw them in the cluster list (deviation D1). Everything else any tool reported is in.

How to read the tables: a *finding* here is a merged **event** (hits of different tools on the same place count once; raw hits are shown
too). `true` / `intended` / `false` / `unclear` are the labels of the protocol (`unclear` counts as false for precision). **"Real leaks the
others missed"** is the number of `true` events that this tool reported and none of the other three did.

### Keys and passwords only (the fair fight: all four tools look for these)

| tool | events labelled false or unclear |
|---|---|
| limnex-preflight | 4 |
| gitleaks | 3 |
| TruffleHog | 0 |
| detect-secrets | 134 |

A benign sample of 60 repositories is not built to measure how well a tool finds keys, so this table shows false alarms
only, and this public version gives no other count for keys and passwords. detect-secrets' entropy plugins reported the
most (mostly hex strings in model JSON files); TruffleHog was the quietest. Redaction (gitleaks `--redact`; detect-secrets stores only hashes) hides the evidence of most
single-tool hits, so they are counted as unclear rather than guessed.

### Personal data (only Preflight looks for it)

| tool | findings (merged events) | true | intended | false | unclear |
|---|---|---|---|---|---|
| limnex-preflight | 19 | 4 | 11 | 4 | 0 |

**All 4 true events were found by Preflight alone:** a hosted-notebook directory in a training-script text file (platform
path, low sensitivity); a user name inside an exporter path in ONNX metadata (one value, 120 occurrences, in
`metadata_props`); and two third-party e-mail addresses inside a text corpus. The first is a plain-text line the others
could in principle read; they have no rule for it. The second sits in the plain-text bytes of a binary ONNX file;
Preflight parses the container and reports where in the graph it is. The other three tools are not built to find
e-mail addresses, user names, host names, IPs or notebook paths, so this row is extra coverage, not a fair contest: it
shows what the coverage adds on real files (4 true and 11 intended items in 60 repositories) and what it costs (4 false).

### Cost (indicative only: the three batches ran in parallel, so these seconds are not benchmark numbers)

| tool | total seconds over 60 repositories | peak RSS, max (MB) |
|---|---|---|
| limnex-preflight | 93.7 | 215.0 |
| gitleaks | 3498.0 | 57.4 |
| TruffleHog | 842.0 | 164.7 |
| detect-secrets | 319.0 | 354.5 |

gitleaks and TruffleHog read model weights end to end as if they were text (a 0.9 GB GGUF repository took gitleaks about 3 minutes);
Preflight reads headers and metadata and skips raw tensor bytes unless `--deep`.

## Fairness notes

* **Recall on keys is not measured.** A benign sample is not built for it; recall on harder cases (split keys, base64,
  UTF-16, secrets inside archives), where the tools differ, is not measured here either.
* **Different designs.** detect-secrets' entropy plugins are noisy by construction; TruffleHog ran with
  `--no-verification`, so its main strength (checking whether a key is live) was switched off on purpose. gitleaks and TruffleHog could be
  tuned with custom rules and allow-lists; I used the defaults of every tool.
* **Labelling is one assistant working from masked evidence, and redaction cuts both ways:** it makes gitleaks'/detect-secrets' single-tool
  hits mostly `unclear` (precision understated), and it equally prevents me from calling any of them a leak the others missed (recall of
  the others understated). Preflight's own masked values gave more evidence for its events. That is an asymmetry of the method, not a
  finding.
* **Preflight authored this evaluation's tooling** (the same author wrote the scanner and the harness). I removed what I could (the
  event-merge rule and the label rules were fixed before the table was computed; the protocol was committed first) but the bias is real.
* **Benign, Hub-filtered sample** (no malicious file, no flagged repository): the number of true leaks is small by construction.
* 60 repositories, 10 per format; the personal-data row rests on 4 true events.

## Deviations

* **D1 (scope, decided after seeing the cluster list, before the table was computed).** Preflight's two `unsafe-code` findings (review tier on a
  Keras 3 pickle) are excluded: they are not leaks and none of the other three tools has such a rule.
* **D2 (tools).** No release binaries could be downloaded for gitleaks and TruffleHog (the environment could reach only one
  GitHub repository), so both were **built from source taken from the Go module proxy**: gitleaks v8.30.1 and TruffleHog v3.99.8, with the
  Go 1.26.10 toolchain that the modules require. detect-secrets 1.5.0 from PyPI. A build from the module proxy is the same source as the release tag.
* **D3 (parallel runs).** The 60 repositories were scanned in three parallel batches to save time; all four tools of one repository ran
  one after the other inside a batch. The findings do not depend on it; the timings do.
* **D4 (label rules written after the cluster list, before the table).** The labels were assigned per cluster by rules written after I
  had seen the 30 cluster descriptions (rule names, file types, location classes, Preflight's masked values) and before any table was
  computed. The reasons are in the appendix.
