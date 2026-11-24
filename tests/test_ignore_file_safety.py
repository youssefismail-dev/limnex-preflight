"""The ignore file is never a way to read the runner's files (pre-release test 3, R2).

A pull request controls what is on disk in the checkout. It could commit `.preflightignore` as a symbolic link to any file
the runner can read; the Action followed it, and the "invalid entry" error repeated the file's first line word for word in
a public log annotation (the tester printed a planted value, and with the process environment file, the whole
environment). Now: an invalid line is reported by its number only; in the Action an ignore file that is a link (or sits in
a linked folder) stops the run, also in the base commit of a pull request; and an ignore file must be a regular file
everywhere, because a named pipe would block the read."""
import os
import subprocess
import types

import pytest

from ci_helpers import commit_all, needs_git, new_repo, run_action, write_event
from limnex_preflight import cli, ignore
from limnex_preflight.ignore import IgnoreFileError, load_ignore_file, parse_ignore_text
import action_fixtures as af

PLANTED = "FAKE_RUNNER_SECRET=fx-not-a-real-value-123456"     # stands for a line of a file on the runner


def test_an_invalid_line_is_reported_by_its_number_only():
    with pytest.raises(IgnoreFileError) as e:
        parse_ignore_text("# accepted findings\n" + PLANTED + "\n", "x/.preflightignore")
    msg = str(e.value)
    assert "x/.preflightignore:2: invalid entry" in msg
    assert "FAKE_RUNNER_SECRET" not in msg and "fx-not-a-real" not in msg


def test_the_action_refuses_an_ignore_file_that_is_a_link_simulated(tmp_path, monkeypatch):
    """Without a real link (Windows cannot create one without Developer Mode); the real one is tested below."""
    ws = tmp_path / "ws"
    ws.mkdir()
    p = ws / ".preflightignore"
    p.write_text("", encoding="utf-8")
    real_islink = os.path.islink
    monkeypatch.setattr(ignore.os.path, "islink", lambda q: os.path.abspath(q) == str(p) or real_islink(q))
    with pytest.raises(IgnoreFileError, match="never read through a link"):
        load_ignore_file(str(p), confine_to=str(ws))
    load_ignore_file(str(p))                       # the CLI on your own machine still reads it


def test_an_ignore_file_that_is_not_a_regular_file_is_refused_simulated(tmp_path, monkeypatch):
    p = tmp_path / ".preflightignore"
    p.write_text("", encoding="utf-8")
    monkeypatch.setattr(ignore, "stat", types.SimpleNamespace(S_ISREG=lambda mode: False))
    with pytest.raises(IgnoreFileError, match="not a regular file"):
        load_ignore_file(str(p))


def _link(target, link):
    try:
        os.symlink(str(target), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here (Windows without Developer Mode)")


def _outside_file(tmp_path):
    secret = tmp_path / "runner" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text(PLANTED + "\n", encoding="utf-8")
    return secret


@needs_git
@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_a_committed_link_as_ignore_file_stops_the_action_and_prints_nothing_of_the_target(tmp_path, monkeypatch, event):
    secret = _outside_file(tmp_path)
    repo = new_repo(tmp_path / "r")
    (tmp_path / "r" / "model.json").write_text("{}", encoding="utf-8")
    base = commit_all(repo, "base")
    af.git(repo, "checkout", "-q", "-b", "feature")
    _link(secret, tmp_path / "r" / ".preflightignore")
    head = commit_all(repo, "link the ignore file to a runner file")
    payload = ({"pull_request": {"base": {"sha": base}, "head": {"sha": head}}} if event == "pull_request"
               else {"before": base, "after": head})
    env = {"GITHUB_EVENT_NAME": event, "GITHUB_SHA": head, "GITHUB_EVENT_PATH": write_event(tmp_path, payload)}
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, repo, env_extra=env, path=".")
    assert code == 2 and "never read through a link" in out
    for text in (out, summary, "\n".join(f"{k}={v}" for k, v in outputs.items())):
        assert "FAKE_RUNNER_SECRET" not in text and "fx-not-a-real" not in text


@needs_git
def test_a_base_commit_whose_ignore_file_is_a_link_is_refused(tmp_path, monkeypatch):
    """Made with git's own plumbing, so it runs where the file system cannot hold a link."""
    repo = new_repo(tmp_path / "r")
    (tmp_path / "r" / "model.json").write_text("{}", encoding="utf-8")
    commit_all(repo, "first")
    sha = subprocess.run(["git", "-C", repo, "hash-object", "-w", "--stdin"], input=b"../runner/secret.txt",
                         capture_output=True, check=True).stdout.decode().strip()
    af.git(repo, "update-index", "--add", "--cacheinfo", f"120000,{sha},.preflightignore")
    af.git(repo, "commit", "-q", "-m", "ignore file as a link")
    base = af.git(repo, "rev-parse", "HEAD").strip()
    af.git(repo, "checkout", "-q", "-b", "feature")
    af.git(repo, "rm", "-q", "--cached", ".preflightignore")
    if os.path.lexists(os.path.join(repo, ".preflightignore")):
        os.remove(os.path.join(repo, ".preflightignore"))
    head = commit_all(repo, "drop it")
    env = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_SHA": head,
           "GITHUB_EVENT_PATH": write_event(tmp_path, {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}})}
    code, out, _, _ = run_action(tmp_path, monkeypatch, repo, env_extra=env, path=".")
    assert code == 2 and "is a symbolic link" in out


def test_a_named_pipe_as_ignore_file_is_refused_not_waited_on(tmp_path, monkeypatch):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no named pipes on this system")
    d = tmp_path / "d"
    d.mkdir()
    (d / "ok.json").write_text("{}", encoding="utf-8")
    os.mkfifo(str(d / ".preflightignore"))
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(d)]) == 2                 # it used to wait forever for a writer
