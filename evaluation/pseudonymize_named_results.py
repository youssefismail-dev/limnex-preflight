"""Give every still-named repository a pseudonym in the RESULT files only (pre-release test 2026-11-19, finding F4).

The study names the 73 scanned public repositories that had no personal-data leak, so anyone can re-run it. Their per-
repository results, however, put findings next to a person's or a team's name: for example a vendor's build e-mail found
inside a CatBoost model reads like that person's leak. The choice made (option a): keep the lists and manifests
named (the run stays reproducible), and show findings only under pseudonyms. So in the five result files every repository
not already pseudonymised becomes ``scanned-repo-NNN`` (random, from the operating system, so nobody can recompute it),
its file names become ``file-N.ext`` (as for the 7 leaking repositories) and its commit SHA is dropped. The
findings themselves are unchanged, so every published total stays the same. The mapping is written only outside the
repository.

Usage:
  python evaluation/pseudonymize_named_results.py --mapping PRIVATE/results_mapping.json
  python evaluation/hf_review.py          # then regenerate results/hf_review.md from the pseudonymised files
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pseudonymize_results as P  # noqa: E402

PREFIX = "scanned-repo-"
ANY_PSEUDO = re.compile(r"^(community-repo|scanned-repo)-\d+$")


class NamedMapper(P.Mapper):
    def repo(self, real: str) -> str:
        if real not in self.repo_to_pseudo:
            taken = {int(p.rsplit("-", 1)[1]) for p in self.repo_to_pseudo.values()}
            n = self.rnd.choice([i for i in range(100, 1000) if i not in taken])
            self.repo_to_pseudo[real] = f"{PREFIX}{n}"
        return self.repo_to_pseudo[real]


def pseudonymise(data: dict, m: NamedMapper) -> dict:
    out = {}
    for repo, e in data.items():
        if ANY_PSEUDO.match(repo):
            out[repo] = e
            continue
        p = m.repo(repo)
        e = json.loads(json.dumps(e))
        for f in e.get("findings", []):
            f["file"] = m.scrub_chain(p, f["file"])
        for key in ("notes", "skipped"):
            for n in e.get(key, []):
                n["file"] = m.scrub_chain(p, n["file"])
        e.pop("sha", None)
        out[p] = e
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    if os.path.realpath(args.mapping).startswith(os.path.realpath(REPO) + os.sep):
        print("refusing to write the mapping inside the repository", file=sys.stderr)
        return 2
    stored = P.load_json(args.mapping) if os.path.exists(args.mapping) else {}
    m = NamedMapper(stored)
    for tag in P.RESULT_TAGS:
        path = os.path.join(args.results, f"hf_{tag}.json")
        data = P.load_json(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(pseudonymise(data, m), fh, indent=1, sort_keys=True)
    with open(args.mapping, "w", encoding="utf-8") as fh:
        json.dump(m.dump(), fh, indent=1, sort_keys=True)
    print(f"{len(m.repo_to_pseudo)} repositories pseudonymised in the result files; mapping written outside the repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
