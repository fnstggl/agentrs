"""G2-harness R3-M1: root's writes into the run directory (which the learner's group may write) never follow a
symlink the learner planted -- also when the learner watches the directory with inotify and races the writes.

The writes are the supervisor's (``runtime.logs.atomic_write_text`` / ``atomic_write_json``: ``preflight.json``
then ``learner.go``, the order of ``runtime/cli.py cmd_supervise``). On 85101a2 the learner redirected the second
write into a root-owned file in 5 of 5 races (temporary name ``.<name>.<pid>.<thread>.tmp`` opened without O_EXCL /
O_NOFOLLOW). Privileged (root, the real service users): run with ``-m privileged``."""
import grp
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest

pytestmark = pytest.mark.privileged
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

RACER = textwrap.dedent(r"""
    import ctypes, os, select, struct, sys, time
    d, victim = sys.argv[1], sys.argv[2]
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init()
    libc.inotify_add_watch(fd, d.encode(), 0x100)          # IN_CREATE
    print("ready", flush=True)
    end = time.time() + 20
    while time.time() < end:
        if not select.select([fd], [], [], 0.5)[0]:
            continue
        buf = os.read(fd, 4096)
        i = 0
        while i + 16 <= len(buf):
            wd, mask, cookie, ln = struct.unpack_from("iIII", buf, i)
            name = buf[i + 16:i + 16 + ln].split(b"\0")[0].decode()
            i += 16 + ln
            if name.startswith(".preflight.json.") and name.endswith(".tmp"):
                rest = name[len(".preflight.json."):-len(".tmp")]
                n = 0
                for cand in (".learner.go." + rest + ".tmp", ".learner.go.tmp", "learner.go"):
                    try:
                        os.symlink(victim, os.path.join(d, cand))
                        n += 1
                    except OSError:
                        pass
                print("planted", n, flush=True)
                sys.exit(0)
    print("timeout", flush=True)
""")

WRITER = textwrap.dedent(r"""
    import os, sys
    sys.path.insert(0, sys.argv[1])
    from cam_live.runtime.logs import atomic_write_json, atomic_write_text
    atomic_write_json(os.path.join(sys.argv[2], "preflight.json"), {"live": {"ready": True}, "checks": list(range(200))})
    try:
        atomic_write_text(os.path.join(sys.argv[2], "learner.go"), "go")
        print("written")
    except OSError as e:
        print("refused", type(e).__name__)
""")


def _needs_privileged(request):
    if "privileged" not in (request.config.getoption("-m") or "") or os.geteuid() != 0:
        pytest.skip("run with -m privileged as root")
    try:
        return pwd.getpwnam("camlearn")
    except KeyError:
        pytest.skip("the service users do not exist")


@pytest.fixture
def layout(request):
    pw = _needs_privileged(request)
    run = tempfile.mkdtemp(prefix="hx4run", dir="/tmp")
    os.chown(run, 0, pw.pw_gid)
    os.chmod(run, 0o1771)                       # the run directory as the launcher lays it out
    vdir = tempfile.mkdtemp(prefix="hx4vic", dir="/tmp")
    os.chmod(vdir, 0o755)
    victim = os.path.join(vdir, "precious")
    yield run, victim, pw
    shutil.rmtree(run, ignore_errors=True)
    shutil.rmtree(vdir, ignore_errors=True)


def test_the_learner_cannot_redirect_roots_run_directory_writes_through_a_race(layout):
    run, victim, pw = layout
    camlive = grp.getgrnam("camlive").gr_gid
    for k in range(5):
        with open(victim, "w") as f:
            f.write("precious root-owned content\n")
        os.chmod(victim, 0o600)
        for nm in os.listdir(run):
            p = os.path.join(run, nm)
            if os.path.islink(p) or os.path.isfile(p):
                os.unlink(p)
        racer = subprocess.Popen([sys.executable, "-I", "-c", RACER, run, victim], user=pw.pw_uid, group=pw.pw_gid,
                                 extra_groups=[camlive], cwd="/", stdout=subprocess.PIPE, text=True,
                                 env={"PATH": "/usr/bin:/bin"})
        assert racer.stdout.readline().strip() == "ready"
        w = subprocess.run([sys.executable, "-I", "-c", WRITER, SRC, run], capture_output=True, text=True, timeout=60)
        racer.wait(timeout=40)
        assert open(victim).read() == "precious root-owned content\n", (k, w.stdout, w.stderr[-300:])
        assert os.stat(victim).st_uid == 0 and oct(os.stat(victim).st_mode & 0o777) == "0o600"


def test_a_planted_target_or_directory_is_refused_not_followed(layout):
    run, victim, pw = layout
    from cam_live.harness import safefs
    from cam_live.runtime.logs import atomic_write_text, read_json
    with open(victim, "w") as f:
        f.write('{"secret": "precious"}')
    camlive = grp.getgrnam("camlive").gr_gid
    plant = "import os, sys\nos.symlink(sys.argv[1], sys.argv[2])\n"
    for target in ("learner.go", "control.json"):
        subprocess.run([sys.executable, "-I", "-c", plant, victim, os.path.join(run, target)], user=pw.pw_uid,
                       group=pw.pw_gid, extra_groups=[camlive], check=True, cwd="/")
    with pytest.raises(OSError):
        atomic_write_text(os.path.join(run, "learner.go"), "go")
    assert open(victim).read() == '{"secret": "precious"}'
    assert read_json(os.path.join(run, "control.json"), default="refused") == "refused"   # not read through the link
    # a directory the learner planted as a symlink is neither created through nor written into
    subprocess.run([sys.executable, "-I", "-c", plant, os.path.dirname(victim), os.path.join(run, "sandbox_host")],
                   user=pw.pw_uid, group=pw.pw_gid, extra_groups=[camlive], check=True, cwd="/")
    with pytest.raises(OSError):
        safefs.ensure_dir(os.path.join(run, "sandbox_host"), 0o700, 0, 0)
    with pytest.raises(OSError):
        safefs.write_atomic(os.path.join(run, "sandbox_host", "job.json"), b"x")
    assert sorted(os.listdir(os.path.dirname(victim))) == ["precious"]
    # an ordinary write into the run directory still works and lands as a regular root file
    atomic_write_text(os.path.join(run, "preflight.json"), "{}")
    st = os.lstat(os.path.join(run, "preflight.json"))
    assert st.st_uid == 0 and not os.path.islink(os.path.join(run, "preflight.json"))
