"""The published evaluation files must be consistent and must not let anyone recover pseudonymised repositories.

(list - results) reveals a hidden repository, so the model lists, the manifests and ALL result sets must cover exactly the
same repositories, and every repository with a personal-data leak must appear only as a pseudonym everywhere.

Pre-release test 2026-11-19 (F4, option a): the lists and manifests name the 73 public repositories without a
leak (so the run can be repeated), and no finding may sit next to a person's or a team's name. Leaking repositories are
``community-repo-NNN`` everywhere; the others are ``scanned-repo-NNN`` in the per-repository result files.

Pre-release test 2 (2026-11-20, N1): even under pseudonyms, a per-repository result (its file count and file types) could
be matched to the named manifests. So the per-repository result files are kept in the private development repository
(`export-ignore` in .gitattributes) and only totals are published (hf_review.md, hf_totals.json). The tests that read
those files skip in the public tree; the others check that nothing per-repository is published.
"""
import json
import os
import re

import pytest

import check_no_real_ids as checker
import hf_review

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "evaluation")
RESULTS = os.path.join(EVAL, "results")
PSEUDO_REPO = re.compile(r"^community-repo-\d+$")
ANY_PSEUDO = re.compile(r"^(community-repo|scanned-repo)-\d+$")
PSEUDO_FILE = re.compile(r"^file-\d+(\.[a-z0-9]{1,12})?$")
PUBLIC_REPO = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")

SETS = {
    "tuning": {"list": "hf_models.txt", "manifest": "hf_manifest_tuning.json", "results": ["v0_tuning", "final_tuning"]},
    "heldout": {"list": "hf_models_heldout.txt", "manifest": "hf_manifest_heldout.json",
                "results": ["v0_heldout", "heldout_first_look", "final_heldout"]},
}


PER_REPO = [f"hf_{tag}.json" for spec in SETS.values() for tag in spec["results"]]
HAVE_PER_REPO = all(os.path.exists(os.path.join(RESULTS, fn)) for fn in PER_REPO)
needs_per_repo = pytest.mark.skipif(not HAVE_PER_REPO, reason="per-repository results are kept in the private "
                                    "development repository only (pre-release test 2, N1)")


def export_ignored():
    """Paths marked export-ignore in .gitattributes (left out of the public tree)."""
    out = set()
    with open(os.path.join(ROOT, ".gitattributes"), encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2 and not parts[0].startswith("#") and "export-ignore" in parts[1:]:
                out.add(parts[0])
    return out


def read_list(fn):
    with open(os.path.join(EVAL, fn)) as fh:
        return [l.strip() for l in fh if l.strip() and not l.startswith("#")]


def read_json(fn):
    with open(os.path.join(RESULTS, fn)) as fh:
        return json.load(fh)


@pytest.mark.parametrize("name", sorted(SETS))
def test_list_and_manifest_name_exactly_the_same_repositories(name):
    spec = SETS[name]
    listed = read_list(spec["list"])
    assert len(listed) == len(set(listed)), "duplicate repository in list"
    assert set(read_json(spec["manifest"])) == set(listed)


@needs_per_repo
@pytest.mark.parametrize("name", sorted(SETS))
def test_every_private_result_set_covers_exactly_the_listed_repositories(name):
    spec = SETS[name]
    listed = read_list(spec["list"])
    first = None
    for tag in spec["results"]:
        keys = set(read_json(f"hf_{tag}.json"))
        assert len(keys) == len(listed), tag
        assert all(ANY_PSEUDO.match(k) for k in keys), (tag, sorted(k for k in keys if not ANY_PSEUDO.match(k)))
        assert {k for k in keys if PSEUDO_REPO.match(k)} == {r for r in listed if PSEUDO_REPO.match(r)}, tag
        first = keys if first is None else first
        assert keys == first, f"{tag}: one pseudonym per repository across the group's result sets"


def _named():
    named = {r for spec in SETS.values() for r in read_list(spec["list"]) if not PSEUDO_REPO.match(r)}
    assert len(named) >= 70, "the lists must still name the scanned public repositories (reproducibility)"
    return named


def test_the_review_table_names_no_listed_repository():
    review = open(os.path.join(RESULTS, "hf_review.md"), encoding="utf-8").read()
    assert not [r for r in _named() if r in review]


@needs_per_repo
def test_no_private_result_file_names_a_listed_repository():
    named = _named()
    for spec in SETS.values():
        for tag in spec["results"]:
            assert not named & set(read_json(f"hf_{tag}.json")), tag


def test_every_repository_id_is_a_public_id_or_a_pseudonym():
    for spec in SETS.values():
        for repo in read_list(spec["list"]):
            assert PSEUDO_REPO.match(repo) or PUBLIC_REPO.match(repo), repo


@needs_per_repo
def test_no_repository_with_a_personal_data_finding_is_named():
    leaking = set()
    for spec in SETS.values():
        for tag in spec["results"]:
            for repo, e in read_json(f"hf_{tag}.json").items():
                if any(hf_review.verdict(f) == "leak" for f in e["findings"]):
                    leaking.add(repo)
    assert leaking, "the study found leaks; none found means the test lost its data"
    assert all(PSEUDO_REPO.match(r) for r in leaking), sorted(r for r in leaking if not PSEUDO_REPO.match(r))


def test_pseudonymised_repositories_expose_no_file_list_hash_size_or_commit_in_the_manifests():
    pseudo = set()
    for spec in SETS.values():
        pseudo |= {r for r in read_list(spec["list"]) if PSEUDO_REPO.match(r)}
        man = read_json(spec["manifest"])
        for repo in (r for r in man if PSEUDO_REPO.match(r)):
            e = man[repo]
            assert e["sha"] is None and e["downloads"] is None
            # no file list: a file count and an extension mix can be matched against the Hub (pre-release test 4, S9)
            assert e["files"] == [], repo
    assert len(pseudo) >= 7


@needs_per_repo
def test_private_result_files_use_pseudonymous_file_names_and_cut_values():
    for spec in SETS.values():
        for tag in spec["results"]:
            for repo, e in read_json(f"hf_{tag}.json").items():
                if not ANY_PSEUDO.match(repo):
                    continue
                assert not e.get("sha")
                for f in e["findings"]:
                    assert PSEUDO_FILE.match(f["file"].split("!/")[0]), f["file"]
                    if hf_review.verdict(f) == "leak":
                        assert len(f["value_masked"]) <= 3, "personal values are cut to two characters in the public files"
                for n in e.get("notes", []) + e.get("skipped", []):
                    assert PSEUDO_FILE.match(n["file"].split("!/")[0]), n["file"]


# ---- totals only (pre-release test 2, N1) ---------------------------------------------------------------------------
def test_per_repository_result_files_are_never_published():
    # in the public tree these files are absent; in the private repository they exist and must be export-ignored
    ignored = export_ignored()
    for fn in PER_REPO:
        assert f"evaluation/results/{fn}" in ignored, f"evaluation/results/{fn} must be export-ignore in .gitattributes"


# RELEASE_CHECKLIST.md is itself export-ignore: it exists in the private development repository and never in the public tree
PRIVATE_TREE = os.path.exists(os.path.join(ROOT, "RELEASE_CHECKLIST.md"))


def private_files_present(root, ignored):
    """The export-ignore paths that exist under ``root`` (each must be absent from a public tree)."""
    return sorted(p for p in ignored if os.path.exists(os.path.join(root, *p.split("/"))))


def test_no_private_file_is_in_the_public_tree():
    """Pre-release test 3, R5: the README said this file's tests fail if a per-repository result is published again, but
    they only checked .gitattributes, so a tree with such a file put back still passed. In the public tree (no release
    checklist), every export-ignore path must be absent."""
    if PRIVATE_TREE:
        pytest.skip("the private development repository keeps these files on purpose")
    assert private_files_present(ROOT, export_ignored()) == []


def test_the_public_tree_check_can_fail(tmp_path):
    (tmp_path / "evaluation" / "results").mkdir(parents=True)
    (tmp_path / "evaluation" / "results" / "hf_v0_tuning.json").write_text("{}", encoding="utf-8")
    assert private_files_present(str(tmp_path), export_ignored()) == ["evaluation/results/hf_v0_tuning.json"]


def test_the_published_review_shows_no_repository_file_or_value():
    text = open(os.path.join(RESULTS, "hf_review.md"), encoding="utf-8").read()
    assert "community-repo-" not in text and "scanned-repo-" not in text
    assert not re.search(r"\bfile-\d+", text), "no (pseudonymous) file names in the published review"
    assert chr(96) not in text, "no masked values (in backticks) in the published review"


def _published_result_json():
    ignored = export_ignored()
    for fn in sorted(os.listdir(RESULTS)):
        # the manifests are the named lists themselves, kept for reproducibility (F4, option a)
        if fn.endswith(".json") and f"evaluation/results/{fn}" not in ignored and not fn.startswith("hf_manifest_"):
            yield fn, read_json(fn)


def test_no_published_result_file_is_keyed_by_repository():
    for fn, data in _published_result_json():
        if isinstance(data, dict):
            keyed = [k for k in data if ANY_PSEUDO.match(k) or PUBLIC_REPO.match(k)]
            assert not keyed, (fn, len(keyed))


def test_the_matching_attack_finds_nothing_to_match_in_published_files():
    """The pre-release test's attack matched each per-repository entry (file count, file kinds) to exactly one named
    manifest entry. With totals only there is nothing to match; this fails if per-repository entries come back."""
    entries = 0
    for _, data in _published_result_json():
        if isinstance(data, dict):
            entries += sum(1 for v in data.values() if isinstance(v, dict) and ("files_scanned" in v or "kinds" in v))
    assert entries == 0


@needs_per_repo
def test_the_published_review_is_regenerated_from_the_private_results(tmp_path):
    import shutil
    import subprocess
    import sys
    for fn in PER_REPO:
        shutil.copy(os.path.join(RESULTS, fn), tmp_path / fn)
    subprocess.run([sys.executable, os.path.join(EVAL, "hf_review.py"), "--results", str(tmp_path)], check=True,
                   capture_output=True)
    with open(os.path.join(RESULTS, "hf_review.md"), encoding="utf-8") as fh:
        assert (tmp_path / "hf_review.md").read_text(encoding="utf-8") == fh.read()


# ---- the checker itself (synthetic data: it must be able to fail) -------------------------------------------------
def test_checker_detects_planted_real_identifiers(tmp_path):
    mapping = {"repos": {"community-repo-999": {"repo": "someowner99/secret-model-name"}},
               "files": {"community-repo-999": {"private_notes_final.ipynb": "file-1.ipynb"}}}
    tokens = checker.forbidden_tokens(mapping, public_ids={"other/public"}, public_basenames={"readme.md"})
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "a.md").write_text("see community-repo-999 and file-1.ipynb")
    assert checker.scan_tree(tokens, str(clean)) == []
    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "b.md").write_text("oops: SomeOwner99/Secret-Model-Name")
    (dirty / "c.json").write_text('{"f": "private_notes_final.ipynb"}')
    (dirty / "d.txt").write_text("the owner someowner99 only")
    hit_files = {rel for rel, _ in checker.scan_tree(tokens, str(dirty))}
    assert hit_files == {"b.md", "c.json", "d.txt"}


def test_checker_ignores_a_shared_owner_and_generic_file_names():
    mapping = {"repos": {"community-repo-1": {"repo": "bigorg/leaky-model-x"}}, "files": {"community-repo-1": {"README.md": "file-1.md", "training_args.bin": "file-2.bin"}}}
    tokens = checker.forbidden_tokens(mapping, public_ids={"bigorg/another"}, public_basenames=set())
    assert "bigorg" not in tokens and "training_args.bin" not in tokens and "readme.md" not in tokens
    assert "bigorg/leaky-model-x" in tokens


@pytest.mark.skipif(not os.environ.get("LIMNEX_PRIVATE_MAPPING"), reason="needs the private mapping (kept outside the repository)")
def test_working_tree_contains_no_real_identifier_of_a_pseudonymised_repo():
    with open(os.environ["LIMNEX_PRIVATE_MAPPING"]) as fh:
        mapping = json.load(fh)
    ids, bases = checker.public_sets(RESULTS)
    tokens = checker.forbidden_tokens(mapping, ids, bases)
    assert tokens
    hits = checker.scan_tree(tokens, ROOT)
    assert not hits, sorted(set(hits))
