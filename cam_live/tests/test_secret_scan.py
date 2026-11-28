"""tools/secret_scan.py (G2-integration G2I-1 / G2I-2): every commit of the range and compressed / binary blobs."""
import gzip
import importlib.util
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "..", "tools", "secret_scan.py")
# a fake token of the Discord bot-token shape, built at run time (no token-shaped literal in the repository)
FAKE = ".".join(["MTIzNDU2Nzg5MDEyMzQ1Njc4OQ", "GabcdE", "abcdefghijklmnopqrstuvwxyz0123456789ab"])


def _load():
    spec = importlib.util.spec_from_file_location("secret_scan_under_test", SCAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    (r / "README").write_text("base\n")
    _git(r, "add", "README")
    _git(r, "commit", "-qm", "base")
    monkeypatch.chdir(r)
    return r


def _commit(repo, name, data, msg):
    p = repo / name
    if data is None:
        p.unlink()
        _git(repo, "rm", "-q", name)
    else:
        p.write_bytes(data)
        _git(repo, "add", name)
    _git(repo, "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD")


def test_a_credential_added_and_removed_inside_the_range_is_found(repo, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "SECRET_FILES", ())
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "x.py", f"TOKEN = '{FAKE}'\n".encode(), "add")
    head = _commit(repo, "x.py", None, "remove")
    assert _git(repo, "diff", f"{base}..{head}") == ""          # the net diff is empty
    assert mod.main([base, head]) == 3


def test_a_credential_inside_a_gzip_file_is_found(repo, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "SECRET_FILES", ())
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, "evidence.json.gz", gzip.compress(f'{{"t": "{FAKE}"}}'.encode()), "gz")
    assert mod.main([base, head]) == 3


def test_a_real_credential_is_found_in_any_commit_and_never_printed(repo, monkeypatch, tmp_path, capsys):
    mod = _load()
    secret = "real-secret-value-" + "9" * 12
    f = tmp_path / "secret"
    f.write_text(secret + "\n")
    monkeypatch.setattr(mod, "SECRET_FILES", (str(f),))
    monkeypatch.setattr(mod, "ACCOUNT_FILE", str(tmp_path / "none.json"))
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "notes.txt", f"oops {secret}\n".encode(), "leak")
    head = _commit(repo, "notes.txt", b"clean\n", "fix")
    assert mod.main([base, head]) == 4
    assert secret not in capsys.readouterr().out


def test_clean_history_and_allow_listed_fixtures_pass(repo, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "SECRET_FILES", ())
    monkeypatch.setattr(mod, "allowed_paths", lambda: {"fixture.py"})
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "fixture.py", f"T = '{FAKE}'\n".encode(), "allowed fixture")
    head = _commit(repo, "ok.txt", b"nothing here\n", "clean")
    assert mod.main([base, head]) == 0


def test_a_credential_inside_a_zip_archive_is_found(repo, monkeypatch):
    import io
    import zipfile
    mod = _load()
    monkeypatch.setattr(mod, "SECRET_FILES", ())
    base = _git(repo, "rev-parse", "HEAD")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("inner/notes.txt", f"token {FAKE}\n")
    head = _commit(repo, "bundle.zip", buf.getvalue(), "zip")
    assert mod.main([base, head]) == 3


def test_identical_content_in_an_allow_listed_file_does_not_hide_another_path(repo, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "SECRET_FILES", ())
    monkeypatch.setattr(mod, "allowed_paths", lambda: {"fixture.py"})
    base = _git(repo, "rev-parse", "HEAD")
    data = f"T = '{FAKE}'\n".encode()
    _commit(repo, "fixture.py", data, "allowed fixture")
    head = _commit(repo, "leak.py", data, "same content elsewhere")
    assert mod.main([base, head]) == 3
