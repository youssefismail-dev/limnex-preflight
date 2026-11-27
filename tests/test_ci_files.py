"""The workflow / action / hook files themselves: pinned, injection-safe, consistent with pyproject and with the code."""
import glob
import os
import re

import pytest

yaml = pytest.importorskip("yaml")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return yaml.safe_load(fh), open(os.path.join(ROOT, rel), encoding="utf-8").read()


def workflow_files():
    return sorted(glob.glob(os.path.join(ROOT, ".github", "workflows", "*.yml")))


def action_files():
    return [os.path.join(ROOT, "action.yml")] + workflow_files()


def all_steps(doc):
    if "runs" in doc:  # composite action
        return list(doc["runs"]["steps"])
    return [s for job in doc["jobs"].values() for s in job.get("steps", [])]


def test_there_are_workflows():
    names = {os.path.basename(p) for p in workflow_files()}
    assert {"tests.yml", "selftest.yml"} <= names


@pytest.mark.parametrize("path", action_files())
def test_third_party_actions_are_pinned_to_a_full_commit_sha_with_the_version_in_a_comment(path):
    doc, text = load(os.path.relpath(path, ROOT))
    for step in all_steps(doc):
        uses = step.get("uses")
        if not uses or uses.startswith("./"):
            continue
        assert PIN.match(uses), f"{os.path.basename(path)}: {uses} is not pinned to a 40-hex commit SHA"
        line = next(l for l in text.splitlines() if uses in l)
        assert re.search(r"#\s*v\d+\.\d+\.\d+", line), f"missing '# vX.Y.Z' comment: {line.strip()}"


@pytest.mark.parametrize("path", action_files())
def test_no_expression_is_ever_interpolated_into_a_run_script(path):
    doc, _ = load(os.path.relpath(path, ROOT))
    for step in all_steps(doc):
        run = step.get("run")
        if run:
            assert "${{" not in run, f"{os.path.basename(path)}: expression inside run: {run[:80]!r} (pass it through env:)"


@pytest.mark.parametrize("path", workflow_files())
def test_workflows_declare_minimal_permissions(path):
    doc, _ = load(os.path.relpath(path, ROOT))
    perms = doc.get("permissions")
    assert perms is not None, "top-level permissions: missing (the default token is too broad)"
    assert perms.get("contents") == "read"
    assert all(v in ("read", "none") for v in perms.values()), perms


@pytest.mark.parametrize("path", workflow_files())
def test_checkouts_do_not_persist_credentials(path):
    doc, _ = load(os.path.relpath(path, ROOT))
    for step in all_steps(doc):
        if str(step.get("uses", "")).startswith("actions/checkout@"):
            assert step.get("with", {}).get("persist-credentials") is False


def test_action_interface_matches_the_specification():
    doc, _ = load("action.yml")
    inp = doc["inputs"]
    assert {"path", "fail-on", "changed-only", "sarif-file", "deep", "fail-on-incomplete"} <= set(inp)
    assert inp["path"]["default"] == "." and inp["fail-on"]["default"] == "medium"
    assert inp["changed-only"]["default"] == "false" and inp["sarif-file"]["default"] == ""
    assert inp["deep"]["default"] == "false" and inp["fail-on-incomplete"]["default"] == "false"
    assert inp["upload-sarif"]["default"] == "false", "SARIF upload must be off by default"
    out = doc["outputs"]
    for name in ("findings-critical", "findings-high", "findings-medium", "findings-low", "findings-info", "findings-total",
                 "highest-severity", "sarif-path", "exit-code", "files-scanned"):
        assert name in out
    assert doc["runs"]["using"] == "composite"


def test_every_action_input_reaches_the_scanner_through_the_environment():
    from limnex_preflight.branding import ACTION_ENV_PREFIX

    doc, _ = load("action.yml")
    scan = next(s for s in doc["runs"]["steps"] if s.get("id") == "scan")
    env = scan["env"]
    for name in doc["inputs"]:
        if name in ("upload-sarif", "sarif-category", "python-version"):
            continue
        key = ACTION_ENV_PREFIX + name.upper().replace("-", "_")
        assert key in env, f"input {name} is not passed as {key}"
        assert env[key] == "${{ inputs.%s }}" % name
    code = open(os.path.join(ROOT, "limnex_preflight", "ci", "action.py"), encoding="utf-8").read()
    for key in env:
        if key.startswith(ACTION_ENV_PREFIX):
            short = key[len(ACTION_ENV_PREFIX):]
            assert f'"{short}"' in code, f"{key} is set by action.yml but never read by the code"


def test_the_scanner_runs_in_isolated_mode():
    _, text = load("action.yml")
    assert "python" in text and re.search(r'"\$VENV_PYTHON" -I ', text), "the scanner must run with python -I"


def test_every_python_call_in_the_action_is_isolated():
    """Pre-release test 2026-11-19, finding F6: the install step ran `python -m venv` and `python -m pip` in the workspace
    without -I, and `-m` puts the current folder first on sys.path, so a file in the pull request could shadow a module."""
    _, text = load("action.yml")
    flags = re.findall(r'"\$(?:BASE_PYTHON|vpy|VENV_PYTHON)"([^\n]*?) -m ', text)    # the options before -m
    assert len(flags) >= 4, flags                       # venv, two pip installs, the scanner
    assert all("-I" in f.split() for f in flags), f"every python -m call in action.yml must use -I: {flags}"


def test_upload_step_is_conditional_and_off_by_default():
    doc, _ = load("action.yml")
    up = next(s for s in doc["runs"]["steps"] if "upload-sarif" in str(s.get("uses", "")))
    assert "inputs.upload-sarif == 'true'" in up["if"] and "always()" in up["if"]


def dispatch_and_push(value):
    """The two lists of a `fromJSON(github.event_name == 'workflow_dispatch' && '[...]' || '[...]')` matrix entry:
    (the full list run on workflow_dispatch, the reduced list run on push / pull_request)."""
    import json
    if isinstance(value, list):
        return [str(v) for v in value], [str(v) for v in value]
    lists = re.findall(r"'(\[[^']*\])'", value)
    assert "workflow_dispatch" in value and len(lists) == 2, value
    return [str(v) for v in json.loads(lists[0])], [str(v) for v in json.loads(lists[1])]


def test_test_matrix_matches_the_python_versions_pyproject_claims():
    import sys

    tomllib = None
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            pass
    text = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    classifiers = set(re.findall(r'"Programming Language :: Python :: (3\.\d+)"', text))
    requires = re.search(r'requires-python\s*=\s*">=(3\.\d+)"', text).group(1)
    assert classifiers, "pyproject.toml must list the supported versions as classifiers"
    assert min(classifiers, key=lambda v: int(v.split(".")[1])) == requires
    doc, _ = load(".github/workflows/tests.yml")
    matrix = doc["jobs"]["pytest"]["strategy"]["matrix"]
    full_py, quick_py = dispatch_and_push(matrix["python"])
    full_os, quick_os = dispatch_and_push(matrix["os"])
    assert set(full_py) == classifiers                       # the full matrix (workflow_dispatch) covers every version
    assert set(full_os) == {"ubuntu-24.04", "windows-latest", "macos-latest"}   # ubuntu-latest (26.04) has no Python 3.9
    if isinstance(matrix["os"], list):                       # public repository: the full matrix on every event
        assert isinstance(matrix["python"], list)
    else:                                                    # private repository: ubuntu only on push / pull_request
        assert quick_os == ["ubuntu-24.04"] and set(quick_py) <= set(full_py)
        assert min(full_py, key=lambda v: int(v.split(".")[1])) in quick_py and max(full_py, key=lambda v: int(v.split(".")[1])) in quick_py
    assert doc["jobs"]["pytest"]["strategy"]["fail-fast"] is False


def test_selftest_asserts_failures_instead_of_merely_allowing_them():
    doc, text = load(".github/workflows/selftest.yml")
    uses_local = [s for job in doc["jobs"].values() for s in job["steps"] if s.get("uses") == "./"]
    assert len(uses_local) >= 6
    for step in uses_local:
        if step.get("continue-on-error"):
            sid = step["id"]
            assert f"steps.{sid}.outcome" in text, f"{sid} is allowed to fail but its outcome is never asserted"


def test_precommit_hook_is_consistent_with_the_action():
    hook, _ = load(".pre-commit-hooks.yaml")
    args = hook[0]["args"]
    assert args[args.index("--fail-on") + 1] == "medium", "same default as the action's fail-on"
    assert hook[0]["entry"] == "limnex-preflight"


def test_precommit_hook_covers_every_file_type_the_scanner_knows():
    """Pre-release test 2026-11-19, finding F9: the hook's `files:` pattern was narrower than the scanner (.env, .p, .dill,
    .py, .sh, .log, .conf, .bz2, .xz, .tgz, .whl were skipped), so a committed .env with a token passed the hook while the
    CLI and the Action caught it. The pattern must match every extension and file name the scanner handles."""
    from limnex_preflight import scanner
    hook, _ = load(".pre-commit-hooks.yaml")
    pattern = re.compile(hook[0]["files"])
    known = set(scanner.PICKLE_EXTS) | set(scanner.TEXT_EXTS) | set(scanner.EXPECTED_KINDS) | {".safetensors", ".onnx"}
    missed = sorted(ext for ext in known if not pattern.search("models/sub/file" + ext))
    assert not missed, f"the pre-commit hook skips: {missed}"
    for name in scanner.TEXT_NAMES:
        assert pattern.search(name) and pattern.search("config/" + name), name
    assert not pattern.search("photo.png") and not pattern.search("notes.envelope")


def test_precommit_hook_passes_any_case_env_variants_and_files_without_an_extension():
    """Pre-release test 2 (N12): Model.PKL, .env.local and an extension-less pickle ("weights", which the CLI finds by
    its content) were never passed to the hook."""
    hook, _ = load(".pre-commit-hooks.yaml")
    pattern = re.compile(hook[0]["files"])
    for name in ("Model.PKL", "sub/WEIGHTS.BIN", ".env.local", "app/.env.production", "weights", "models/checkpoint",
                 "Dockerfile"):
        assert pattern.search(name), name
    for name in ("photo.png", "data.csv", "notes.envelope", "lib/x.so", "README.rst"):
        assert not pattern.search(name), name


def test_example_workflows_exist_and_use_the_released_action():
    from limnex_preflight import __version__
    from limnex_preflight.branding import REPO_URL

    released = REPO_URL.replace("https://github.com/", "") + "@v" + __version__    # the tag of this release
    for rel in ("examples/github-action-minimal.yml", "examples/github-action-full.yml"):
        doc, text = load(rel)
        assert released in text and "OWNER/REPO" not in text
        perms = doc["jobs"][next(iter(doc["jobs"]))].get("permissions") or doc.get("permissions")
        assert perms and perms.get("contents") == "read"
    assert not os.path.exists(os.path.join(ROOT, "examples", "github-action.yml")), "replaced by the two examples"
    _, full = load("examples/github-action-full.yml")
    assert "security-events: write" in full and "changed-only" in full and "upload-sarif" in full and "fetch-depth: 0" in full
    cfg, _ = load("examples/pre-commit-config.yaml")
    assert cfg["repos"][0]["hooks"][0]["id"] == "limnex-preflight"


def test_action_dependencies_are_pinned_to_exact_versions():
    """Supply-chain pinning: every package the action installs from PyPI has one exact version per Python range."""
    _, action = load("action.yml")
    assert 'PIP_CONSTRAINT="${ACTION_PATH}/action-constraints.txt"' in action
    # pip 26.2 applies PIP_CONSTRAINT to the build environment no more (pre-release test 5, T3)
    assert 'PIP_BUILD_CONSTRAINT="${ACTION_PATH}/action-constraints.txt"' in action
    lines = [l.split("#", 1)[0].strip() for l in open(os.path.join(ROOT, "action-constraints.txt"), encoding="utf-8")]
    pins = [l for l in lines if l]
    assert pins
    names = set()
    for pin in pins:
        assert re.fullmatch(r'[A-Za-z0-9_.-]+==\d+(\.\d+)+( ; python_version (<|>=|==) "3\.\d+")?', pin), pin
        names.add(pin.split("==")[0])
    assert names == {"setuptools", "h5py", "numpy"}


def test_release_workflow_publishes_only_from_the_public_repository_with_a_scoped_token():
    from limnex_preflight.branding import REPO_URL

    doc, _ = load(".github/workflows/release.yml")
    on = doc.get("on", doc.get(True))  # YAML 1.1 reads the bare key `on` as True
    assert on == {"release": {"types": ["published"]}}
    build, publish = doc["jobs"]["build"], doc["jobs"]["publish"]
    assert build["if"] == "github.repository == '%s'" % REPO_URL.replace("https://github.com/", "")
    assert "permissions" not in build, "the build job keeps the workflow's contents: read"
    assert publish["permissions"] == {"id-token": "write"} and publish["needs"] == "build"
    assert publish["environment"]["name"] == "pypi"
    assert not any(str(s.get("uses", "")).startswith("actions/checkout@") for s in publish["steps"]), \
        "the job that holds the upload token never checks out code"


def test_the_release_checks_every_version_against_the_tag():
    """Pre-release test 2026-11-19, finding F22: the tag was compared with pyproject.toml only, not with __version__."""
    _, text = load(".github/workflows/release.yml")
    assert "__version__" in text and 'test "$version" = "$code_version"' in text
    assert "--version | tee" in text and "TAG#v" in text


def test_a_release_is_built_only_from_a_commit_on_main():
    """F22 (optional part): a tag on a side branch must not reach PyPI."""
    doc, _ = load(".github/workflows/release.yml")
    steps = doc["jobs"]["build"]["steps"]
    assert steps[0]["uses"].startswith("actions/checkout@") and steps[0]["with"]["fetch-depth"] == 0
    assert steps[1]["run"].strip() == 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main'


def test_the_release_build_tool_and_its_dependencies_are_pinned():
    """F22 (last part): build==1.6.1 pulled packaging and pyproject_hooks at whatever version was newest that day."""
    import re
    import shlex

    _, text = load(".github/workflows/release.yml")
    line = next(l.strip() for l in text.splitlines() if "pip install" in l and '"build==' in l)
    args = shlex.split(line)
    assert "--no-deps" in args
    reqs = [a for a in args if re.match(r"^[A-Za-z0-9_.-]+[=<>!~]", a)]
    assert {r.split("==")[0].lower().replace("-", "_") for r in reqs} == {"build", "packaging", "pyproject_hooks"}
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9.]*", r) for r in reqs), reqs
