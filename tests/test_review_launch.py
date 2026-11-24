"""Fixes from the user review before launch (2026-11-21).

1. Credentials set through environment variables in Python code (os.environ['KEY'] = '...', setdefault, putenv),
   wandb.login(key=...) and the ~/.netrc entry W&B writes were missed by every rule. All values here are fake."""
import pytest

from limnex_preflight.detectors import ENGINE

WANDB = "3f9a1c7e5b2d8f4a6c0e9b1d7f3a5c8e2b4d6f0a"          # 40 hex, fake
KAGGLE = "9c4e1a7f3b5d2e8c6a0f4b9d1e7c3a5f"                  # 32 hex, fake
PASSWORD = "Xk9-Passw0rd-4471"


def _rules(text):
    return {h.rule.id for h in ENGINE.scan(text)}


@pytest.mark.parametrize("text,rule", [
    (f"os.environ['WANDB_API_KEY'] = '{WANDB}'", "wandb-api-key"),
    (f'os.environ["WANDB_API_KEY"]="{WANDB}"', "wandb-api-key"),
    (f'os.environ.setdefault("WANDB_API_KEY", "{WANDB}")', "wandb-api-key"),
    (f"%env WANDB_API_KEY={WANDB}", "wandb-api-key"),
    (f'wandb.login(key="{WANDB}")', "wandb-api-key"),
    (f"wandb.login(relogin=True, key='{WANDB}')", "wandb-api-key"),
    (f"machine api.wandb.ai\n  login user\n  password {WANDB}\n", "wandb-api-key"),
    (f"os.environ['KAGGLE_KEY'] = '{KAGGLE}'", "kaggle-api-key"),
    (f'os.putenv("KAGGLE_KEY", "{KAGGLE}")', "kaggle-api-key"),
    (f"os.environ['DB_PASSWORD'] = '{PASSWORD}'", "generic-credential"),
    (f'os.environ.setdefault("API_SECRET", "{PASSWORD}")', "generic-credential"),
    (f'"cells": [{{"source": ["os.environ[\'WANDB_API_KEY\'] = \'{WANDB}\'"]}}]', "wandb-api-key"),
])
def test_credentials_set_through_environment_variables(text, rule):
    assert rule in _rules(text), text


@pytest.mark.parametrize("text", [
    "os.environ['WANDB_API_KEY'] = userdata.get('wandb')",
    "os.environ['DB_PASSWORD'] = os.getenv('PW')",
    'api_key = os.getenv("OPENAI_API_KEY", None)',
    'os.environ.setdefault("API_SECRET", "")',
    "if os.environ['DB_PASSWORD'] == expected:",
    "wandb.login(key=WANDB_KEY)",
    'ent["api_secret"] = make_secret(getattr(args, "secret_name"), work)',
])
def test_references_in_code_are_not_credentials(text):
    assert not _rules(text) & {"wandb-api-key", "kaggle-api-key", "generic-credential"}, text


# ---- 2. a Git LFS pointer is not the file -------------------------------------------------------------------------------
LFS = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "ab" * 32 + "\nsize 438000000\n"


@pytest.mark.parametrize("name", ["pytorch_model.bin", "training_args.bin", "model.safetensors", "model.onnx", "weights"])
def test_a_git_lfs_pointer_is_a_medium_finding_under_any_name(tmp_path, name):
    from limnex_preflight import cli
    from limnex_preflight.scanner import scan_paths
    p = tmp_path / name
    p.write_bytes(LFS.encode())
    res = scan_paths([str(p)])
    assert [s.kind for s in res.files] == ["lfs-pointer"]
    assert [(f.rule_id, f.severity.label) for f in res.findings] == [("git-lfs-pointer", "medium")]
    assert cli.main([str(p)]) == 1                    # it was "no findings", exit 0, even with --fail-on-incomplete


def test_text_that_mentions_git_lfs_is_not_a_pointer(tmp_path):
    from limnex_preflight.scanner import scan_paths
    p = tmp_path / "README.md"
    p.write_text("Weights are stored with Git LFS.\n" + LFS, encoding="utf-8")
    assert not scan_paths([str(p)]).findings


def test_the_examples_check_out_lfs_files():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("github-action-minimal.yml", "github-action-full.yml"):
        assert "lfs: true" in open(os.path.join(root, "examples", name), encoding="utf-8").read(), name


# ---- 3. "nothing was checked" is not a pass; a Hugging Face cache snapshot is read ----------------------------------------
def test_files_found_but_none_read_exit_3(tmp_path, capsys):
    from limnex_preflight import cli
    d = tmp_path / "m"
    d.mkdir()
    (d / "weights.xyz").write_bytes(bytes(range(256)) * 64)            # an unsupported type, the only file
    assert cli.main([str(d)]) == 3
    assert "nothing was checked" in capsys.readouterr().out
    assert cli.main([str(d), "--no-error-exit"]) == 0
    (tmp_path / "empty").mkdir()
    assert cli.main([str(tmp_path / "empty")]) == 0                   # nothing there at all is not a scan problem


def test_a_cache_snapshot_follows_its_own_blobs_only(tmp_path):
    """The decision alone (no real link needed): snapshot -> its own blobs/ yes; anything else no."""
    import os
    from limnex_preflight.scanner import symlink_decision
    repo = tmp_path / "hub" / "models--org--name"
    for d in ("blobs", "snapshots/abc", "refs"):
        (repo / d).mkdir(parents=True)
    (repo / "blobs" / "4f2e").write_text("{}", encoding="utf-8")
    other = tmp_path / "hub" / "models--other--x"
    (other / "blobs").mkdir(parents=True)
    (other / "snapshots").mkdir()
    (other / "blobs" / "9a").write_text("{}", encoding="utf-8")
    snap = os.path.realpath(str(repo / "snapshots" / "abc"))

    def decide(target):
        return symlink_decision(os.path.realpath(str(target)), True, explicit=False, scan_root=snap, follow_all=False,
                                confine_to=None)[0]
    assert decide(repo / "blobs" / "4f2e") is True
    assert decide(other / "blobs" / "9a") is False                    # another repository's blob
    assert decide(repo / "refs") is False


# ---- 4. archives: a compressed .tar; member names that leave the folder --------------------------------------------------
def _tar(path, member, mode="w", data=b"x=1\n"):
    import io
    import tarfile
    with tarfile.open(str(path), mode) as t:
        ti = tarfile.TarInfo(member)
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))


@pytest.mark.parametrize("mode", ["w:gz", "w:bz2", "w:xz"])
def test_a_compressed_tar_named_tar_is_read_without_a_damage_note(tmp_path, mode):
    from limnex_preflight.scanner import scan_paths
    p = tmp_path / "model.tar"
    _tar(p, "model_config.yaml", mode, b"log_dir: /home/alice/x\n")
    res = scan_paths([str(p)])
    assert {f.rule_id for f in res.findings} == {"username-in-path"}
    assert not [n for n in res.notes if n.kind in ("degraded", "error")]


@pytest.mark.parametrize("member", ["../../../../tmp/test.txt", "/etc/cron.d/x", "a/../../b.txt", "C:/Users/x.txt"])
def test_an_archive_member_that_leaves_the_folder_is_flagged(tmp_path, member):
    import zipfile
    from limnex_preflight.scanner import scan_paths
    t = tmp_path / "model.nemo"
    _tar(t, member)
    z = tmp_path / "model.keras"
    with zipfile.ZipFile(str(z), "w") as zf:
        zf.writestr(member, "x")
    for p in (t, z):
        assert "archive-path-traversal" in {f.rule_id for f in scan_paths([str(p)]).findings}, (p.name, member)


def test_ordinary_member_names_are_not_flagged(tmp_path):
    from limnex_preflight.scanner import scan_paths
    p = tmp_path / "model.nemo"
    _tar(p, "./model_weights.ckpt..bak/a..b.txt")
    assert "archive-path-traversal" not in {f.rule_id for f in scan_paths([str(p)]).findings}
