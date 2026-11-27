"""Chromium under the dedicated browser uid with a profile only that uid can read (run with ``-m privileged``).

Needs root and the ``cambrowser`` service user (created by the launcher). Playwright's bundled Chromium lives under
/root, which other uids cannot traverse, so the test copies it to a world-readable temporary directory — exactly what
an operator does once for a separated deployment (``BrowserConfig.executable_path``)."""
import glob
import os
import pwd
import shutil
import stat
import subprocess
import tempfile

import pytest

from cam_live.contracts import Modality, TaskSpec
from cam_live.harness import SERVICE_USERS
from cam_live.harness.browser import BrowserConfig, BrowserExecutor
from cam_live.harness.env import ComputerEnvironment

from eb_support import Driver, GrantPolicy

pytestmark = [pytest.mark.privileged, pytest.mark.browser]


def _uids_of_processes_using(path):
    out = {}
    for status in glob.glob("/proc/[0-9]*/status"):
        pid = status.split("/")[2]
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
            if path in cmd and "chrome" in cmd:
                uid_line = [x for x in open(status).read().split("\n") if x.startswith("Uid:")][0]
                out[int(pid)] = [int(x) for x in uid_line.split()[1:]]
        except (OSError, IndexError):
            continue
    return out


def test_chromium_runs_under_the_browser_uid_with_a_private_profile(sites):
    if os.geteuid() != 0:
        pytest.skip("needs root")
    try:
        user = pwd.getpwnam(SERVICE_USERS["browser"])
        learner = pwd.getpwnam(SERVICE_USERS["learner"])
    except KeyError:
        pytest.skip("service users do not exist")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        bundled = p.chromium.executable_path
    shared = tempfile.mkdtemp(prefix="ebchr", dir="/tmp")
    run_dir = tempfile.mkdtemp(prefix="ebp", dir="/tmp")
    try:
        subprocess.run(["cp", "-a", "--reflink=auto", os.path.dirname(bundled), shared], check=True)
        subprocess.run(["chmod", "-R", "a+rX", shared], check=True)
        exe = os.path.join(shared, os.path.basename(os.path.dirname(bundled)), "chrome")
        s = sites.urls["structured"]

        def start():
            env = ComputerEnvironment(run_dir, GrantPolicy())
            env.register(BrowserExecutor(run_dir, config=BrowserConfig(executable_path=exe)))
            return Driver(env, env.policy, TaskSpec("obj_uid", "exercise uid separation", mode="staging"))

        d = start()
        try:
            sep = d.env.get_capabilities()["executors"][Modality.STRUCTURED_BROWSER]["uid_separation"]
            assert sep == {"enabled": True, "user": user.pw_name, "reason": ""}
            d.ok("browser.navigate", url=s + "/login")
            d.ok("browser.fill", role="textbox", name="Handle", text="uid-check")
            assert "Signed in as uid-check." in d.ok("browser.click", role="button", name="Sign in").observations[0].text
            profile = os.path.join(run_dir, "browser", "profiles", "default")
            procs = _uids_of_processes_using(profile)
            assert len(procs) >= 2                                   # browser process and its children
            assert all(set(uids) == {user.pw_uid} for uids in procs.values()), procs    # real/effective/saved/fs uid
            st = os.stat(profile)
            assert st.st_uid == user.pw_uid and stat.S_IMODE(st.st_mode) == 0o700
            cookie_files = [f for f in glob.glob(profile + "/**/Cookies", recursive=True)]
            assert cookie_files and all(os.stat(f).st_uid == user.pw_uid for f in cookie_files)
            # the learner's uid can read neither the profile nor the carried-over session cookies
            for target in (profile, cookie_files[0], profile + ".storage.json"):
                r = subprocess.run(["setpriv", f"--reuid={learner.pw_uid}", f"--regid={learner.pw_gid}", "--clear-groups",
                                    "cat", target], capture_output=True)
                assert r.returncode != 0 and b"Permission denied" in r.stderr, target
            got = d.ok("browser.download", url=s + "/files/report.csv")
            assert open(got.artifacts["download"]).read() == sites.structured.csv_text()
        finally:
            d.env.close()
        assert _uids_of_processes_using(os.path.join(run_dir, "browser")) == {}       # nothing left running
        d = start()
        try:
            assert "Signed in as uid-check." in d.ok("browser.navigate", url=s + "/account").observations[0].text
        finally:
            d.env.close()
    finally:
        shutil.rmtree(shared, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


def test_fallback_reason_is_reported_when_chromium_is_not_reachable_by_the_browser_uid(tmp_path):
    if os.geteuid() != 0:
        pytest.skip("needs root")
    try:
        pwd.getpwnam(SERVICE_USERS["browser"])
    except KeyError:
        pytest.skip("service users do not exist")
    import cam_live.harness.browser as B
    shared, B.SHARED_BROWSERS_DIR = B.SHARED_BROWSERS_DIR, str(tmp_path / "no-shared-install")   # none installed
    run = tempfile.mkdtemp(prefix="ebf", dir="/tmp")
    os.chmod(run, 0o755)
    try:
        ex = BrowserExecutor(run)                                    # bundled Chromium under /root: not reachable
        try:
            sep = ex.capabilities()["uid_separation"]
            if sep["enabled"]:
                pytest.skip("the bundled Chromium is reachable by the browser uid on this machine")
            assert "not executable by" in sep["reason"] and "--install-shared" in sep["reason"]
        finally:
            ex.close()
    finally:
        B.SHARED_BROWSERS_DIR = shared
        shutil.rmtree(run, ignore_errors=True)
    # a run directory the browser uid cannot reach (pytest's own temporary directories are private) is reported too
    ex = BrowserExecutor(str(tmp_path / "private"))
    try:
        sep = ex.capabilities()["uid_separation"]
        if not sep["enabled"]:
            assert "not reachable" in sep["reason"] or "not executable" in sep["reason"], sep
    finally:
        ex.close()
