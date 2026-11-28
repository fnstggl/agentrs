"""G2-harness round 4 (R4-C1, R4-M1, R4-m1, R4-m2): commands the operator runs as root over a run directory the
learner's group can write. Privileged (root, real service users): run with ``-m privileged``. Each test fails on
a5ab772.

R4-C1  export-checkpoint never takes a path from the learner's state.json, never copies a symlink or special file,
       never merges into an existing tree, verifies the MANIFEST
R4-M1  plan-report never writes through a planted symlink
R4-m1  a FIFO in place of state.json does not block root's reads
R4-m2  credentials revoke does not unlink through a symlink the verifier uid planted"""
import grp
import hashlib
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

pytestmark = pytest.mark.privileged
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _skip_unless_privileged(request):
    if "privileged" not in (request.config.getoption("-m") or "") or os.geteuid() != 0:
        pytest.skip("run with -m privileged as root")
    try:
        return pwd.getpwnam("camlearn"), pwd.getpwnam("camverify")
    except KeyError:
        pytest.skip("the service users do not exist")


def _as(user, code, *args):
    pw = pwd.getpwnam(user)
    return subprocess.run([sys.executable, "-I", "-c", code, *map(str, args)], user=pw.pw_uid, group=pw.pw_gid,
                          extra_groups=[grp.getgrnam("camlive").gr_gid], cwd="/", capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"}, timeout=60)


def _cli(*args, home):
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": SRC, "CAM_LIVE_HOME": home, "HOME": "/root"}
    return subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", *args], capture_output=True, text=True,
                          env=env, timeout=120, cwd="/")


@pytest.fixture
def world(request):
    learner, verifier = _skip_unless_privileged(request)
    base = tempfile.mkdtemp(prefix="hx5", dir="/tmp")
    os.chmod(base, 0o755)
    run = os.path.join(base, "run")
    os.mkdir(run)
    os.chown(run, 0, learner.pw_gid)
    os.chmod(run, 0o1771)                                   # the run directory as the launcher lays it out
    os.mkdir(os.path.join(run, "checkpoints"))
    os.chown(os.path.join(run, "checkpoints"), learner.pw_uid, learner.pw_gid)
    home = os.path.join(base, "home")
    os.mkdir(home, 0o700)
    secret_dir = os.path.join(base, "rootonly")
    os.mkdir(secret_dir, 0o700)
    yield {"base": base, "run": run, "home": home, "rootonly": secret_dir, "learner": learner, "verifier": verifier}
    shutil.rmtree(base, ignore_errors=True)


PLANT_STATE = r"""
import json, os, sys
run, target, own = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(own, exist_ok=True)
open(os.path.join(own, "authorized_keys"), "w").write("ssh-ed25519 AAAA learner@evil\n")
json.dump({"committed": {"ckpt": {"id": "x/../../../../../../.." + target, "path": own}}},
          open(os.path.join(run, "state.json"), "w"))
"""


def test_export_checkpoint_never_copies_learner_chosen_files_anywhere(world):
    run, home, rootonly = world["run"], world["home"], world["rootonly"]
    own = os.path.join(run, "checkpoints", "mine")            # a directory of the learner's own
    cp = _as("camlearn", PLANT_STATE, run, rootonly, own)
    assert cp.returncode == 0, cp.stderr
    r = _cli("export-checkpoint", "--run-dir", run, "--json", home=home)
    assert "authorized_keys" not in os.listdir(rootonly), (r.stdout, r.stderr)
    assert r.returncode != 0 and ("no checkpoint" in r.stderr or "not a checkpoint name" in r.stderr), r.stderr
    # a real checkpoint directory of the run is exported -- by name, into the operator's destination, verified
    ck = os.path.join(run, "checkpoints", "ckpt_00000003")
    _as("camlearn", "import os, sys\nos.makedirs(sys.argv[1] + '/weights')\n"
                    "open(sys.argv[1] + '/weights/a.bin', 'wb').write(b'w' * 100)\n", ck)
    data = b"w" * 100
    manifest = {"files": {"weights/a.bin": {"bytes": 100, "sha256": hashlib.sha256(data).hexdigest()}},
                "format": None}
    _as("camlearn", "import json, sys\njson.dump(json.loads(sys.argv[2]), open(sys.argv[1] + '/agent.json', 'w'))\n",
        ck, json.dumps(manifest))
    out = os.path.join(world["base"], "export1")
    r = _cli("export-checkpoint", "--run-dir", run, "--checkpoint", "ckpt_00000003", "--out", out, "--json", home=home)
    assert r.returncode == 0, r.stderr
    assert open(os.path.join(out, "weights", "a.bin"), "rb").read() == data
    assert "authorized_keys" not in os.listdir(rootonly)
    # never merged into an existing (non-empty) tree
    open(os.path.join(rootonly, "keep.txt"), "w").write("operator file")
    r = _cli("export-checkpoint", "--run-dir", run, "--checkpoint", "ckpt_00000003", "--out", rootonly, home=home)
    assert r.returncode != 0 and "never merged" in r.stderr, r.stderr
    assert sorted(os.listdir(rootonly)) == ["keep.txt"]
    # never into the run directory
    r = _cli("export-checkpoint", "--run-dir", run, "--checkpoint", "ckpt_00000003", "--out",
             os.path.join(run, "exports", "x"), home=home)
    assert r.returncode != 0 and "outside the run directory" in r.stderr, r.stderr
    # a symlink inside the checkpoint refuses the export (nothing followed, nothing copied)
    _as("camlearn", "import os, sys\nos.symlink('/etc/shadow', sys.argv[1] + '/weights/shadow')\n", ck)
    out2 = os.path.join(world["base"], "export2")
    r = _cli("export-checkpoint", "--run-dir", run, "--checkpoint", "ckpt_00000003", "--out", out2, home=home)
    assert r.returncode != 0 and "symlink" in r.stderr and not os.path.exists(out2), r.stderr
    # a checkpoint whose files do not match its MANIFEST is refused
    os.unlink(os.path.join(ck, "weights", "shadow"))
    _as("camlearn", "import json, sys\njson.dump(json.loads(sys.argv[2]), open(sys.argv[1] + '/MANIFEST.json', 'w'))\n",
        ck, json.dumps({"files": {"weights/a.bin": {"bytes": 100, "sha256": "0" * 64}}, "format": 1}))
    out3 = os.path.join(world["base"], "export3")
    r = _cli("export-checkpoint", "--run-dir", run, "--checkpoint", "ckpt_00000003", "--out", out3, home=home)
    assert r.returncode != 0 and "MANIFEST" in r.stderr and not os.path.exists(out3), r.stderr


def test_plan_report_never_writes_through_a_planted_symlink(world):
    sys.path.insert(0, SRC)
    from cam_live.harness.auth import launcher as L
    base, home, rootonly = world["base"], world["home"], world["rootonly"]
    run = os.path.join(base, "laidout")
    L.prepare_run_dir(run, True, dict(L.DEFAULT_ROLES), None, None)      # the launcher's layout of a separated run
    with open(os.path.join(run, "state.json"), "w") as f:     # a minimal runtime state, so the report is built
        json.dump({"status": "stopped", "task": {"objective_id": "obj-hx5", "mode": "staging"}, "counters": {}}, f)
    victim = os.path.join(rootonly, "precious")
    open(victim, "w").write("precious root file\n")
    cp = _as("camlearn", "import os, sys\nos.symlink(sys.argv[1], sys.argv[2])\n", victim,
             os.path.join(run, "plan_report.md.tmp"))
    assert cp.returncode == 0, cp.stderr
    r = _cli("plan-report", "--run-dir", run, home=home)
    assert r.returncode == 0, r.stderr                       # the report is written (not refused) ...
    assert open(victim).read() == "precious root file\n"     # ... but not through the planted link
    rep = os.path.join(run, "plan_report.md")
    assert not os.path.islink(rep) and os.lstat(rep).st_uid == 0 and open(rep).read().startswith("# Plan report")
    # a planted symlink as the report itself: refused, not followed
    os.unlink(rep)
    _as("camlearn", "import os, sys\nos.symlink(sys.argv[1], sys.argv[2])\n", victim, rep)
    r = _cli("plan-report", "--run-dir", run, home=home)
    assert open(victim).read() == "precious root file\n" and r.returncode != 0


def test_a_fifo_in_place_of_state_json_does_not_block_root(world):
    run, home = world["run"], world["home"]
    cp = _as("camlearn", "import os, sys\nos.mkfifo(sys.argv[1])\n", os.path.join(run, "state.json"))
    assert cp.returncode == 0, cp.stderr
    sys.path.insert(0, SRC)
    from cam_live.runtime.logs import read_json
    t0 = time.time()
    assert read_json(os.path.join(run, "state.json"), default="refused") == "refused"
    assert time.time() - t0 < 5.0
    t0 = time.time()
    r = _cli("status", "--run-dir", run, "--json", home=home)
    assert time.time() - t0 < 60.0, r.stderr


def test_credentials_revoke_does_not_unlink_through_a_planted_symlink(world):
    run = world["run"]
    v = world["verifier"]
    vdir = os.path.join(run, "verifier")
    os.mkdir(vdir, 0o700)
    os.chown(vdir, v.pw_uid, v.pw_gid)
    keep = os.path.join(world["rootonly"], "discord_bot_token")
    open(keep, "w").write("root keeps this")
    cp = _as("camverify", "import os, sys\nos.symlink(sys.argv[1], sys.argv[2])\n", world["rootonly"],
             os.path.join(vdir, "secrets"))
    assert cp.returncode == 0, cp.stderr
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": SRC}
    subprocess.run([sys.executable, "-m", "cam_live.harness.credentials", "revoke", "discord_bot_token", "--run-dir",
                    run], capture_output=True, text=True, env=env, timeout=60, cwd="/")
    assert os.path.exists(keep) and open(keep).read() == "root keeps this"
