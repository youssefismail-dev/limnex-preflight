"""SECURITY (review finding 2026-10-24): a pull request must not be able to silence its own finding by adding the finding's
fingerprint to .preflightignore. On pull requests the Action reads the ignore files from the BASE commit."""
import json
import os

import fixtures as fx
from ci_helpers import commit_all, needs_git, new_repo, run_action, write_event
from limnex_preflight.context import ScanOptions
from limnex_preflight.scanner import scan_paths
import action_fixtures as af

pytestmark = needs_git


def _repo_with_a_finding(tmp_path, monkeypatch):
    repo = new_repo(tmp_path / "r")
    key = next(p.value for p in fx.PLANTED if p.rule == "aws-access-key-id")
    with open(os.path.join(repo, "config.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"aws_access_key_id": key}, fh)
    base = commit_all(repo, "base")
    fps = {f.fingerprint for f in scan_paths([repo], ScanOptions()).findings}
    assert fps, "the fixture must produce a finding"
    return repo, base, fps


def _ignore(repo, fps):
    with open(os.path.join(repo, ".preflightignore"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(f"{fp}  # accepted\n" for fp in sorted(fps)))


def _pr_env(tmp_path, base, head):
    return {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_SHA": head,
            "GITHUB_EVENT_PATH": write_event(tmp_path, {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}})}


def test_a_pull_request_cannot_silence_its_own_finding(tmp_path, monkeypatch):
    repo, base, fps = _repo_with_a_finding(tmp_path, monkeypatch)
    af.git(repo, "checkout", "-q", "-b", "feature")
    _ignore(repo, fps)                                       # the pull request adds the fingerprint of its own finding
    head = commit_all(repo, "silence it")
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, repo, env_extra=_pr_env(tmp_path, base, head),
                                             path=".", fail_on="high")
    assert code == 1, "the finding must still fail the pull request"
    assert "NOT applied" in out and "NOT applied" in summary


def test_an_entry_already_in_the_base_branch_is_honoured(tmp_path, monkeypatch):
    repo, _, fps = _repo_with_a_finding(tmp_path, monkeypatch)
    _ignore(repo, fps)                                       # accepted on the base branch, reviewed there
    base = commit_all(repo, "accept")
    af.git(repo, "checkout", "-q", "-b", "feature")
    head = commit_all(repo, "unrelated")
    code, out, _, _ = run_action(tmp_path, monkeypatch, repo, env_extra=_pr_env(tmp_path, base, head), path=".", fail_on="high")
    assert code == 0 and "NOT applied" not in out


def test_a_push_uses_the_checkouts_own_ignore_file(tmp_path, monkeypatch):
    repo, base, fps = _repo_with_a_finding(tmp_path, monkeypatch)
    _ignore(repo, fps)
    head = commit_all(repo, "accept")
    env = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": head,
           "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": base, "after": head})}
    code, _, _, _ = run_action(tmp_path, monkeypatch, repo, env_extra=env, path=".", fail_on="high")
    assert code == 0


def test_a_missing_base_commit_applies_no_ignore_entry(tmp_path, monkeypatch):
    """Pre-release test 2026-11-19 (F2): this used to fall back to the pull request's own ignore file (exit 0), so with
    actions/checkout's default shallow clone a pull request could silence its own finding. Now it fails closed."""
    repo, _, fps = _repo_with_a_finding(tmp_path, monkeypatch)
    _ignore(repo, fps)
    head = commit_all(repo, "silence it")
    code, out, _, _ = run_action(tmp_path, monkeypatch, repo, env_extra=_pr_env(tmp_path, "f" * 40, head), path=".",
                                 fail_on="high")
    assert code == 1                                         # the finding is not silenced ...
    assert "::warning" in out and "NO ignore entries were applied" in out and "fetch-depth: 0" in out  # ... and it says why


def test_a_shallow_checkout_fetches_the_base_commit(tmp_path, monkeypatch):
    """The default actions/checkout clone has only the merge commit: the base commit is fetched, so an entry the base
    branch accepted is honoured and one the pull request adds is not."""
    origin, _, fps = _repo_with_a_finding(tmp_path, monkeypatch)
    _ignore(origin, fps)
    base = commit_all(origin, "accept on the base branch")
    af.git(origin, "checkout", "-q", "-b", "feature")
    with open(os.path.join(origin, "notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("unrelated\n")
    head = commit_all(origin, "unrelated change")
    shallow = str(tmp_path / "shallow")
    af.git(str(tmp_path), "clone", "-q", "--depth=1", "--branch=feature", "file://" + origin.replace(os.sep, "/"), shallow)
    assert af.git(shallow, "rev-parse", "--is-shallow-repository").strip() == "true"
    code, out, _, _ = run_action(tmp_path, monkeypatch, shallow, env_extra=_pr_env(tmp_path, base, head), path=".",
                                 fail_on="high")
    assert code == 0 and "ignore entries read from the base commit" in out and "NO ignore entries" not in out, out


def test_a_pull_request_cannot_silence_its_own_finding_in_a_shallow_checkout(tmp_path, monkeypatch):
    """The pre-release tester's exact case (F2): depth-1 checkout, the pull request adds its own finding's fingerprint."""
    origin, base, fps = _repo_with_a_finding(tmp_path, monkeypatch)
    af.git(origin, "checkout", "-q", "-b", "feature")
    _ignore(origin, fps)
    head = commit_all(origin, "silence it")
    shallow = str(tmp_path / "shallow")
    af.git(str(tmp_path), "clone", "-q", "--depth=1", "--branch=feature", "file://" + origin.replace(os.sep, "/"), shallow)
    code, out, _, _ = run_action(tmp_path, monkeypatch, shallow, env_extra=_pr_env(tmp_path, base, head), path=".",
                                 fail_on="high")
    assert code == 1 and "NOT applied" in out, out


# ---- encodings of the base commit's ignore file (pre-release test 2, N3) -------------------------------------------------
def _pr_with_base_ignore(tmp_path, monkeypatch, data: bytes):
    repo, _, fps = _repo_with_a_finding(tmp_path, monkeypatch)
    with open(os.path.join(repo, ".preflightignore"), "wb") as fh:
        fh.write(data.replace(b"FPS", "".join(f"{fp}  # accepted\n" for fp in sorted(fps)).encode("ascii")))
    base = commit_all(repo, "accept")
    af.git(repo, "checkout", "-q", "-b", "feature")
    head = commit_all(repo, "unrelated")
    return run_action(tmp_path, monkeypatch, repo, env_extra=_pr_env(tmp_path, base, head), path=".", fail_on="high")


def test_a_byte_order_mark_in_the_base_ignore_file_is_accepted(tmp_path, monkeypatch):
    code, out, _, _ = _pr_with_base_ignore(tmp_path, monkeypatch, b"\xef\xbb\xbfFPS")
    assert code == 0, out                                   # the accepted finding stays accepted, as with the CLI


def test_crlf_line_ends_in_the_base_ignore_file_are_accepted(tmp_path, monkeypatch):
    code, out, _, _ = _pr_with_base_ignore(tmp_path, monkeypatch, b"# accepted findings\r\nFPS")
    assert code == 0, out


def test_a_utf16_base_ignore_file_is_a_clear_usage_error(tmp_path, monkeypatch):
    repo, _, fps = _repo_with_a_finding(tmp_path, monkeypatch)
    with open(os.path.join(repo, ".preflightignore"), "wb") as fh:
        fh.write("".join(f"{fp}\n" for fp in sorted(fps)).encode("utf-16"))
    base = commit_all(repo, "accept")
    af.git(repo, "checkout", "-q", "-b", "feature")
    _ignore(repo, fps)                                       # the head's own file is valid UTF-8: only the base one is not
    head = commit_all(repo, "re-saved as UTF-8")
    code, out, _, _ = run_action(tmp_path, monkeypatch, repo, env_extra=_pr_env(tmp_path, base, head), path=".")
    assert code == 2 and "not UTF-8 text" in out and "(base " in out and "Traceback" not in out
