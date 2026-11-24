"""Stdlib-only fixtures for the GitHub Action: used by pytest AND by .github/workflows/selftest.yml.

    python tests/action_fixtures.py clean   DIR      # a folder that must pass
    python tests/action_fixtures.py bad     DIR      # planted fake secret + booby-trapped pickle: must fail with exit 1
    python tests/action_fixtures.py changed DIR      # a git repo with 3 commits; prints c1=<sha> c2=<sha> c3=<sha>
    python tests/action_fixtures.py check-sarif FILE [--rules a,b] [--min-results N] [--max-results N]
                                                     # structure + "no unmasked secret anywhere" check (exit 1 on failure)

All secret-shaped values come from evaluation/fixtures.py, which assembles them from fragments at run time (nothing
secret-shaped is stored in this repository). The pickle only CONTAINS a call to os.system; nothing ever loads it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import fixtures as fx  # noqa: E402  (stdlib-only at import time)

AWS = fx.BY_LABEL["aws_key"].text
HF = fx.BY_LABEL["hf"].text


def write(path: str, data, mode: str = "w") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode + ("b" if isinstance(data, bytes) else ""), **({} if isinstance(data, bytes) else {"encoding": "utf-8"})) as fh:
        fh.write(data)


def make_clean(d: str) -> None:
    write(os.path.join(d, "config.json"), json.dumps({"model_type": "bert", "hidden_size": 768, "bos_token": "<s>"}, indent=1))
    write(os.path.join(d, "README.txt"), "A small clean model folder.\nContact: user@example.com\nSee /home/ubuntu/work\n")
    write(os.path.join(d, "weights", "benign.pkl"), fx.benign_pickle_bytes())


def make_bad(d: str) -> None:
    write(os.path.join(d, "config.json"), json.dumps({"model_type": "bert", "note": "key " + AWS}, indent=1))
    write(os.path.join(d, "weights", "evil.pkl"), fx.pickle_proto4_system())
    write(os.path.join(d, "weights", "benign.pkl"), fx.benign_pickle_bytes())


def git(d: str, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    return subprocess.run(["git", "-C", d, *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env).stdout.decode().strip()


def make_changed_repo(d: str) -> dict:
    """c1: legacy_bad.json (planted secret) + ok.json + a file with a space and unicode in its name.
    c2: only ok.json changes.  c3: only legacy_bad.json changes."""
    os.makedirs(d, exist_ok=True)
    git(d, "init", "-q", "-b", "main")
    write(os.path.join(d, "legacy_bad.json"), json.dumps({"note": "key " + AWS}, indent=1))
    write(os.path.join(d, "ok.json"), json.dumps({"version": 1}, indent=1))
    write(os.path.join(d, "sub dir", "café notes.txt"), "nothing here\n")
    git(d, "add", "-A")
    git(d, "commit", "-q", "-m", "c1")
    c1 = git(d, "rev-parse", "HEAD")
    write(os.path.join(d, "ok.json"), json.dumps({"version": 2}, indent=1))
    git(d, "commit", "-q", "-am", "c2")
    c2 = git(d, "rev-parse", "HEAD")
    write(os.path.join(d, "legacy_bad.json"), json.dumps({"note": "key " + AWS, "extra": 1}, indent=1))
    git(d, "commit", "-q", "-am", "c3")
    c3 = git(d, "rev-parse", "HEAD")
    return {"c1": c1, "c2": c2, "c3": c3}


def secrets_that_must_never_appear() -> list:
    out = [p.value for p in fx.PLANTED if len(p.value) >= 6 and p.rule != "private-key-pem"]
    out += [p.text for p in fx.PLANTED if len(p.text) >= 12 and "\n" not in p.text]
    out += ["FAKEfakeFAKEfake"]  # the body of the planted PEM key
    return sorted(set(out))


def check_sarif_file(path: str, rules=None, min_results=None, max_results=None) -> None:
    from sarif_check import check_sarif  # tests/sarif_check.py

    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    for needle in secrets_that_must_never_appear():
        if needle in raw:
            raise SystemExit("UNMASKED planted value found in the SARIF file")
    doc = json.loads(raw)
    check_sarif(doc)
    results = doc["runs"][0]["results"]
    ids = {r["ruleId"] for r in results}
    if rules and not set(rules) <= ids:
        raise SystemExit(f"expected rules {sorted(set(rules) - ids)} missing; got {sorted(ids)}")
    if min_results is not None and len(results) < min_results:
        raise SystemExit(f"expected at least {min_results} results, got {len(results)}")
    if max_results is not None and len(results) > max_results:
        raise SystemExit(f"expected at most {max_results} results, got {len(results)}")
    print(f"SARIF ok: {len(results)} result(s), rules {sorted(ids)}")


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "clean" and len(argv) == 3:
        make_clean(argv[2])
        return 0
    if cmd == "bad" and len(argv) == 3:
        make_bad(argv[2])
        return 0
    if cmd == "changed" and len(argv) == 3:
        for k, v in make_changed_repo(argv[2]).items():
            print(f"{k}={v}")
        return 0
    if cmd == "check-sarif" and len(argv) >= 3:
        rules, mn, mx = None, None, None
        rest = argv[3:]
        while rest:
            opt = rest.pop(0)
            if opt == "--rules":
                rules = [x for x in rest.pop(0).split(",") if x]
            elif opt == "--min-results":
                mn = int(rest.pop(0))
            elif opt == "--max-results":
                mx = int(rest.pop(0))
        check_sarif_file(argv[2], rules, mn, mx)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main(sys.argv))
