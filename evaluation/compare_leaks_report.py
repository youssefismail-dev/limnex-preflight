"""Merge the four tools' findings into events, cluster them for labelling, and compute the comparison tables.

  python evaluation/compare_leaks_report.py --results A.json B.json C.json --show-clusters      # for the labeller (private output)
  python evaluation/compare_leaks_report.py --results ... --out-json OUT.json --out-md OUT.md    # needs LABELS to be complete

Protocol: evaluation/COMPARE_LEAKS_2026-11-05.md. Event = hits of any tools on the same repo+file whose lines differ by <= 1; a hit
without a line number joins the file's single other event, otherwise it is an event of its own. Output uses pseudonyms and file
TYPES only; no file names, no values.

Its inputs (the per-repository results and the labels, results/compare_leaks_labels.json) and its per-repository output stay
in the private development repository; the public report gives totals only (decided after pre-release test 3).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

TOOLS = ("preflight", "gitleaks", "trufflehog", "detect-secrets")
NAMES = {"preflight": "limnex-preflight", "gitleaks": "gitleaks", "trufflehog": "TruffleHog", "detect-secrets": "detect-secrets"}

# cluster key -> (label, reason). Key = "pseudonym | ext | tool:rules;... | location class". Filled by the labeller (see LABEL_SOURCE).
LABELS: dict = {}
LABEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "compare_leaks_labels.json")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def ext_of(f):
    e = os.path.splitext(f)[1].lower()
    return e if e else "(none)"


def loc_class(loc):
    import re
    return re.sub(r"\d+", "N", loc or "")[:70]


def load(paths):
    R = {}
    for p in paths:
        R.update(json.load(open(p)))
    return R


def build_events(R):
    events = []
    for pseudo, r in sorted(R.items()):
        by_file = defaultdict(list)
        for t in TOOLS:
            for f in r[t]["findings"]:
                if f.get("category") == "unsafe-code":
                    continue  # Preflight's unsafe-pickle findings are not leaks; this is a leak-scanner comparison
                by_file[f["file"]].append(f)
        for fn, hits in by_file.items():
            lined = sorted((h for h in hits if h.get("line")), key=lambda h: h["line"])
            unlined = [h for h in hits if not h.get("line")]
            groups, cur = [], []
            for h in lined:
                if cur and h["line"] - cur[-1]["line"] > 1:
                    groups.append(cur)
                    cur = []
                cur.append(h)
            if cur:
                groups.append(cur)
            if unlined:
                if len(groups) == 1 and not any(h["tool"] == "preflight" for h in groups[0]) :
                    groups[0] += unlined  # joins the file's single other-tool event
                else:
                    groups.append(unlined)
            for g in groups:
                events.append({"repo": pseudo, "file": fn, "ext": ext_of(fn), "hits": g})
    for e in events:
        e["tools"] = sorted({h["tool"] for h in e["hits"]})
        e["rules"] = {t: sorted({h["rule"] for h in e["hits"] if h["tool"] == t}) for t in e["tools"]}
        pf = [h for h in e["hits"] if h["tool"] == "preflight"]
        e["pf_cat"] = sorted({h["category"] for h in pf})
        e["pf_loc"] = sorted({loc_class(h["location"]) for h in pf})
        e["pf_masked"] = sorted({h["masked"] for h in pf})
        e["pf_sev"] = sorted({h["severity"] for h in pf})
        # keys/passwords class: any hit by another tool, or a Preflight secret-category hit
        e["klass"] = "keys" if (any(h["tool"] != "preflight" for h in e["hits"]) or "secret" in e["pf_cat"]) else "personal"
        e["ckey"] = " | ".join([e["repo"], e["ext"], ";".join(f"{t}:{','.join(e['rules'][t])}" for t in e["tools"]), "/".join(e["pf_loc"])])
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--show-clusters", action="store_true")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    a = ap.parse_args()
    for p in a.results:
        if os.path.realpath(p).startswith(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep):
            raise SystemExit("private results must stay outside the repository")
    R = load(a.results)
    events = build_events(R)
    clusters = defaultdict(list)
    for e in events:
        clusters[e["ckey"]].append(e)
    if a.show_clusters:
        for k, es in sorted(clusters.items()):
            e = es[0]
            print(f"{len(es):4d} x {k}  [{e['klass']}] sev={e['pf_sev']} masked={e['pf_masked'][:2]}")
        print(len(events), "events,", len(clusters), "clusters")
        return 0
    labels = json.load(open(LABEL_FILE)) if os.path.exists(LABEL_FILE) else {}
    missing = [k for k in clusters if k not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} unlabelled clusters, e.g. {missing[:2]}")
    stale = [k for k in labels if k not in clusters]
    if stale:
        raise SystemExit(f"{len(stale)} labels without a cluster, e.g. {stale[:2]}")
    for e in events:
        e["label"] = labels[e["ckey"]]["label"]
    out = {"events": len(events), "clusters": len(clusters), "repos": len(R)}
    rows = {}
    for scope, pred in (("keys and passwords only", lambda e: e["klass"] == "keys"), ("everything", lambda e: True)):
        scope_events = [e for e in events if pred(e)]
        true_all = [e for e in scope_events if e["label"] == "true"]
        table = {}
        for t in TOOLS:
            mine = [e for e in scope_events if t in e["tools"]]
            c = Counter(e["label"] for e in mine)
            hits = sum(1 for e in mine for h in e["hits"] if h["tool"] == t)
            excl = [e for e in mine if e["label"] == "true" and e["tools"] == [t]]
            excl_unclear = [e for e in mine if e["label"] == "unclear" and e["tools"] == [t]]
            found_true = c["true"]
            table[t] = {"hits": hits, "events": len(mine), "true": c["true"], "intended": c["intended"], "false": c["false"], "unclear": c["unclear"],
                        "true_only_this_tool": len(excl), "unclear_only_this_tool": len(excl_unclear), "share_of_all_true": [found_true, len(true_all)],
                        "precision_true": [c["true"], len(mine)], "precision_wilson95": wilson(c["true"], len(mine))}
        rows[scope] = {"table": table, "events": len(scope_events), "true_events": len(true_all)}
    out["rows"] = rows
    # leaks found by one tool only
    only = defaultdict(list)
    for e in events:
        if e["label"] == "true" and len(e["tools"]) == 1:
            only[e["tools"][0]].append({"repo": e["repo"], "ext": e["ext"], "klass": e["klass"], "rules": e["rules"][e["tools"][0]],
                                        "location": e["pf_loc"], "severity": e["pf_sev"]})
    out["true_found_by_one_tool"] = dict(only)
    out["tool_runtime"] = {t: {"seconds_total": round(sum(r[t]["wall_s"] for r in R.values()), 1), "peak_rss_mb_max": max(r[t]["rss_mb"] for r in R.values()),
                               "failed_runs": sum(1 for r in R.values() if r[t].get("failed"))} for t in TOOLS}
    out["trufflehog_any_verified"] = any(r["trufflehog"].get("any_verified") for r in R.values())
    out["download_errors"] = sum(len(r["download_errors"]) for r in R.values())
    out["clusters_table"] = [{"cluster": k, "events": len(v), "label": labels[k]["label"], "why": labels[k]["why"], "klass": v[0]["klass"], "tools": v[0]["tools"]}
                             for k, v in sorted(clusters.items())]
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k not in ("clusters_table",)}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
