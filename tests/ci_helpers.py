"""Shared helpers for the GitHub-Action tests (real temporary git repositories, fake GitHub environments)."""

import io
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

import action_fixtures as af

HAVE_GIT = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not HAVE_GIT, reason="git not installed")
P = "LIMNEX_PREFLIGHT_"
ZERO = "0" * 40


def commit_all(repo, msg="c"):
    af.git(repo, "add", "-A")
    af.git(repo, "commit", "-q", "-m", msg, "--allow-empty")
    return af.git(repo, "rev-parse", "HEAD")


def new_repo(path):
    os.makedirs(path, exist_ok=True)
    af.git(path, "init", "-q", "-b", "main")
    return str(path)


def write_event(tmp_path, payload):
    p = tmp_path / "event.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def parse_github_output(path):
    """Parse $GITHUB_OUTPUT the way the runner does: ``name=value`` or ``name<<DELIM`` ... ``DELIM``."""
    out = {}
    if not os.path.exists(path):
        return out
    lines = open(path, encoding="utf-8").read().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        m = re.match(r"^([^<=]+)<<(.+)$", line)
        if m:
            name, delim = m.groups()
            i += 1
            buf = []
            while i < len(lines) and lines[i] != delim:
                buf.append(lines[i])
                i += 1
            out[name] = "\n".join(buf)
        else:
            name, _, value = line.partition("=")
            out[name] = value
        i += 1
    return out


def run_action(tmp_path, monkeypatch, cwd, env_extra=None, **inputs):
    """Run limnex_preflight.ci.action.main in-process with a fake GitHub environment. Returns (code, stdout, outputs, summary)."""
    from limnex_preflight.ci import action

    out_file = tmp_path / "gh_output.txt"
    sum_file = tmp_path / "gh_summary.md"
    for f in (out_file, sum_file):
        if f.exists():
            f.unlink()
    env = {"GITHUB_OUTPUT": str(out_file), "GITHUB_STEP_SUMMARY": str(sum_file), "GITHUB_WORKSPACE": str(cwd)}
    env.update(env_extra or {})
    for k, v in inputs.items():
        env[P + k.upper()] = v
    monkeypatch.chdir(cwd)
    buf = io.StringIO()
    code = action.main(env, buf)
    summary = sum_file.read_text(encoding="utf-8") if sum_file.exists() else ""
    return code, buf.getvalue(), parse_github_output(str(out_file)), summary


def outside_guard(stdout):
    """stdout lines that are NOT inside a ::stop-commands:: ... ::token:: block (these are the ones GitHub would interpret)."""
    lines = stdout.split("\n")
    out, token = [], None
    for line in lines:
        if token is None:
            m = re.match(r"^::stop-commands::(\S+)$", line)
            if m:
                token = m.group(1)
                continue
            out.append(line)
        elif line == f"::{token}::":
            token = None
    return out
