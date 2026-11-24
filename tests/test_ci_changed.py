"""changed-only: every edge case from the specification, against real temporary git repositories."""
import os
import pathlib
import subprocess
import sys

import pytest

import action_fixtures as af
from ci_helpers import ZERO, commit_all, needs_git, new_repo, write_event
from limnex_preflight.ci.changed import changed_files

pytestmark = needs_git


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    r = new_repo(tmp_path / "r")
    monkeypatch.chdir(r)
    return r


def put(repo, rel, text="x\n"):
    p = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def push_env(tmp_path, before, sha):
    return {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": sha, "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": before, "after": sha})}


def names(ch):
    assert ch.files is not None, ch.note
    return sorted(f.replace(os.sep, "/") for f in ch.files)


def test_push_range_is_before_to_sha(repo, tmp_path):
    put(repo, "a.json"), put(repo, "b.json")
    c1 = commit_all(repo)
    put(repo, "a.json", "changed\n")
    put(repo, "c.json")
    c2 = commit_all(repo)
    ch = changed_files(push_env(tmp_path, c1, c2), ".")
    assert names(ch) == ["a.json", "c.json"]


def test_pull_request_uses_the_merge_base_not_the_branch_tip(repo, tmp_path):
    put(repo, "base.json")
    base0 = commit_all(repo)
    af.git(repo, "checkout", "-q", "-b", "feature")
    put(repo, "feature.json")
    head = commit_all(repo)
    af.git(repo, "checkout", "-q", "main")
    put(repo, "main_only.json")  # base branch moved on after the branch point
    base_tip = commit_all(repo)
    af.git(repo, "checkout", "-q", "feature")
    env = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_SHA": head,
           "GITHUB_EVENT_PATH": write_event(tmp_path, {"pull_request": {"base": {"sha": base_tip}, "head": {"sha": head}}})}
    ch = changed_files(env, ".")
    assert names(ch) == ["feature.json"], "files only changed on the base branch must not be reported"


def _target_pr(repo, tmp_path):
    """A pull request whose branch adds feature.json while the base branch moved on; GITHUB_SHA is the base branch's tip,
    as GitHub sets it on pull_request_target."""
    put(repo, "base.json")
    commit_all(repo)
    af.git(repo, "checkout", "-q", "-b", "feature")
    put(repo, "feature.json")
    head = commit_all(repo)
    af.git(repo, "checkout", "-q", "main")
    put(repo, "main_only.json")
    base_tip = commit_all(repo)
    env = {"GITHUB_EVENT_NAME": "pull_request_target", "GITHUB_SHA": base_tip,
           "GITHUB_EVENT_PATH": write_event(tmp_path, {"pull_request": {"base": {"sha": base_tip}, "head": {"sha": head}}})}
    return env, head


def test_pull_request_target_with_the_pull_requests_head_checked_out(repo, tmp_path):
    """Pre-release test 4, S2: the head was GITHUB_SHA, the base branch's tip on this event: base was compared with base,
    0 files were scanned, exit 0."""
    env, head = _target_pr(repo, tmp_path)
    af.git(repo, "checkout", "-q", head)
    assert names(changed_files(env, ".")) == ["feature.json"]


def test_pull_request_target_with_the_default_checkout_scans_everything(repo, tmp_path):
    env, _ = _target_pr(repo, tmp_path)          # the checkout stays on the base branch, GitHub's default for this event
    ch = changed_files(env, ".")
    assert ch.files is None and "pull_request_target" in ch.note and "head" in ch.note


def test_first_push_of_a_branch_falls_back_and_says_so(repo, tmp_path):
    put(repo, "a.json")
    c1 = commit_all(repo)
    ch = changed_files(push_env(tmp_path, ZERO, c1), ".")
    assert ch.files is None and "first push" in ch.note and "all zeros" in ch.note


def test_shallow_checkout_falls_back_and_says_so(tmp_path, monkeypatch):
    src = new_repo(tmp_path / "src")
    put(src, "a.json")
    c1 = commit_all(src)
    put(src, "a.json", "2\n")
    c2 = commit_all(src)
    put(src, "a.json", "3\n")
    c3 = commit_all(src)
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", pathlib.Path(src).resolve().as_uri(), str(shallow)], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    monkeypatch.chdir(shallow)
    ch = changed_files(push_env(tmp_path, c1, c3), ".")
    assert ch.files is None
    assert "not in this checkout" in ch.note and "shallow" in ch.note
    pr_env = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_SHA": c3,
              "GITHUB_EVENT_PATH": write_event(tmp_path, {"pull_request": {"base": {"sha": c1}}})}
    ch = changed_files(pr_env, ".")
    assert ch.files is None and "shallow" in ch.note


def test_deleted_files_are_skipped_and_renames_report_the_new_name(repo, tmp_path):
    put(repo, "gone.json"), put(repo, "old_name.json", "content that is long enough to be detected as a rename\n" * 5)
    c1 = commit_all(repo)
    os.remove(os.path.join(repo, "gone.json"))
    os.rename(os.path.join(repo, "old_name.json"), os.path.join(repo, "new_name.json"))
    c2 = commit_all(repo)
    ch = changed_files(push_env(tmp_path, c1, c2), ".")
    assert names(ch) == ["new_name.json"]


def test_file_deleted_later_in_the_working_tree_is_skipped(repo, tmp_path):
    put(repo, "a.json")
    c1 = commit_all(repo)
    put(repo, "b.json")
    c2 = commit_all(repo)
    os.remove(os.path.join(repo, "b.json"))  # in the diff, but gone from disk
    assert names(changed_files(push_env(tmp_path, c1, c2), ".")) == []


@pytest.mark.parametrize("name", ["with space.json", "café 中文.json", "quote'and\"double.json", "semi;colon&amp.json", "tab\there.json"])
def test_paths_with_spaces_unicode_and_quotes(repo, tmp_path, name):
    if sys.platform == "win32" and any(ch in name for ch in "\"\t"):
        pytest.skip("not a valid Windows file name")
    put(repo, "keep.json")
    c1 = commit_all(repo)
    put(repo, name)
    c2 = commit_all(repo)
    ch = changed_files(push_env(tmp_path, c1, c2), ".")
    assert names(ch) == [name]


@pytest.mark.skipif(sys.platform == "win32", reason="newlines are not allowed in Windows file names")
def test_path_with_a_newline_survives_git_diff_z(repo, tmp_path):
    put(repo, "keep.json")
    c1 = commit_all(repo)
    put(repo, "line1\nline2.json")
    c2 = commit_all(repo)
    assert names(changed_files(push_env(tmp_path, c1, c2), ".")) == ["line1\nline2.json"]


def test_only_files_under_the_scan_path_are_returned(repo, tmp_path):
    put(repo, "keep.json")
    c1 = commit_all(repo)
    put(repo, "models/a.json"), put(repo, "docs/b.json"), put(repo, "models-extra/c.json")
    c2 = commit_all(repo)
    assert names(changed_files(push_env(tmp_path, c1, c2), "models")) == ["models/a.json"]
    assert names(changed_files(push_env(tmp_path, c1, c2), "models/a.json")) == ["models/a.json"]


def test_scan_path_in_a_subdirectory_is_a_valid_checkout_root_lookup(repo, tmp_path):
    put(repo, "keep.json")
    c1 = commit_all(repo)
    put(repo, "deep/er/a.json")
    c2 = commit_all(repo)
    assert names(changed_files(push_env(tmp_path, c1, c2), "deep/er")) == ["deep/er/a.json"]


def test_a_changed_file_inside_a_normally_skipped_directory_is_still_returned(repo, tmp_path):
    # a directory walk skips VCS internals only; .ipynb_checkpoints & co are published, so they are scanned
    put(repo, "keep.json")
    c1 = commit_all(repo)
    put(repo, ".ipynb_checkpoints/x-checkpoint.json")
    c2 = commit_all(repo)
    assert names(changed_files(push_env(tmp_path, c1, c2), ".")) == [".ipynb_checkpoints/x-checkpoint.json"]


def test_other_events_fall_back(repo, tmp_path):
    put(repo, "a.json")
    commit_all(repo)
    ch = changed_files({"GITHUB_EVENT_NAME": "schedule"}, ".")
    assert ch.files is None and "push, pull_request and pull_request_target" in ch.note


def test_not_a_git_checkout_falls_back(tmp_path, monkeypatch):
    d = tmp_path / "plain"
    d.mkdir()
    monkeypatch.chdir(d)
    ch = changed_files({"GITHUB_EVENT_NAME": "push"}, ".")
    assert ch.files is None and "not a git checkout" in ch.note


@pytest.mark.parametrize("hostile", ["--output=/tmp/pwned", "-p", "--exec=touch x", "a b", "$(touch x)", "x;y", ""])
def test_hostile_revisions_are_refused_never_passed_to_git(repo, tmp_path, hostile):
    put(repo, "a.json")
    commit_all(repo)
    ch = changed_files({"GITHUB_EVENT_NAME": "push"}, ".", base=hostile or "-", head="")
    assert ch.files is None and ("refusing" in ch.note or "not a git" in ch.note or "explicit" in ch.note or "changed-only" in ch.note)
    assert not os.path.exists("/tmp/pwned") and not os.path.exists(os.path.join(repo, "x"))


def test_explicit_base_and_head_override_the_event(repo, tmp_path):
    put(repo, "a.json")
    c1 = commit_all(repo)
    put(repo, "b.json")
    c2 = commit_all(repo)
    put(repo, "c.json")
    commit_all(repo)
    ch = changed_files({"GITHUB_EVENT_NAME": "schedule"}, ".", base=c1, head=c2)
    assert names(ch) == ["b.json"]
    ch = changed_files({}, ".", base="HEAD~1")
    assert names(ch) == ["c.json"]


def test_git_diff_is_run_without_external_diff_drivers(repo, tmp_path):
    # a repository can configure diff.external; changed-only must never run it
    put(repo, "a.json")
    c1 = commit_all(repo)
    af.git(repo, "config", "diff.external", "touch pwned-by-external-diff")
    put(repo, "a.json", "2\n")
    c2 = commit_all(repo)
    assert names(changed_files(push_env(tmp_path, c1, c2), ".")) == ["a.json"]
    assert not os.path.exists(os.path.join(repo, "pwned-by-external-diff"))


def test_a_workspace_reached_through_a_symlink_still_matches_gits_physical_paths(tmp_path, monkeypatch):
    """macOS temp dirs live behind /var -> /private/var; git reports the physical path, the workspace may use the link."""
    real = new_repo(tmp_path / "real")
    link = tmp_path / "link"
    try:
        os.symlink(real, str(link), target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here")
    put(real, "keep.json")
    c1 = commit_all(real)
    put(real, "models/a.json")
    c2 = commit_all(real)
    monkeypatch.chdir(str(link))
    ch = changed_files(push_env(tmp_path, c1, c2), "models")
    assert names(ch) == ["models/a.json"]
    # the scan path is given THROUGH the link while the working directory is somewhere else: abspath() would not match git's paths
    monkeypatch.chdir(str(tmp_path))
    ch = changed_files(push_env(tmp_path, c1, c2), str(link / "models"))
    assert names(ch) == ["real/models/a.json"]
