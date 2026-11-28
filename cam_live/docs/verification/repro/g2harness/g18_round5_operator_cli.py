"""G2-harness probe g18 (round 5; root; launcher layout, real service users, real ``cam-live`` CLI of the code under
test): the R4-C1 / R4-M1 fixes under attack by the learner, which owns ``<run>/checkpoints`` and may create names in
the run directory (mode 1771, group = learner gid).

  X1  ``ckpt_<digits>`` spoofed as a SYMLINK to a root-only directory (named in the learner's state.json): nothing of
      the root directory may reach the export.
  X2  a HARD LINK to a root-owned file inside a real checkpoint (``fs.protected_hardlinks`` decides whether the
      learner can create it) and a symlink inside a checkpoint (the export must refuse).
  X3  race: the learner flips ``ckpt_8`` between a real directory and a symlink to the root-only directory while root
      exports it repeatedly.
  X4  a learner-written MANIFEST.json whose file entries point OUTSIDE the checkpoint (an absolute path / ``..``):
      does root read and hash files the learner names, and what does the operator see?
  X5  the default destination (``<state root>/exports``): outside the run directory, and not writable by the
      learner; a legitimate learner checkpoint with a matching MANIFEST is exported there.
  X6  ``cam-live plan-report`` default output ``<run>/plan_report.md``: (a) the learner cannot replace or remove it
      once written; (b) a CAM note carrying terminal control sequences (ESC / OSC) -- text that reaches CAM's notes
      from untrusted pages -- appears raw in the Markdown the operator reads (``cat``) or not.
Everything below /tmp, removed at exit; CAM_LIVE_HOME points at a root-only scratch state root.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g18_round5_operator_cli.py
"""
import grp
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile

from _g2 import ROOT, SRC, Report  # noqa: I001

R = Report("g18_round5_operator_cli")
if os.geteuid() != 0:
    R.note("not root: skipped")
    R.done()

from cam_live.harness.auth import launcher as L  # noqa: E402

CAMDIR = os.path.dirname(ROOT)
HOME = tempfile.mkdtemp(prefix="g2hhome", dir="/tmp")            # root 0700 state root
os.chmod(HOME, 0o700)


def run_as(user, code, *args, timeout=60):
    pw = pwd.getpwnam(user)
    return subprocess.run([sys.executable, "-I", "-c", code, *map(str, args)], user=pw.pw_uid, group=pw.pw_gid,
                          extra_groups=[grp.getgrnam("camlive").gr_gid], cwd="/", env={"PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True, timeout=timeout)


def cli(*args):
    env = dict(os.environ, PYTHONPATH=SRC, CAM_LIVE_HOME=HOME)
    return subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", *args], cwd=CAMDIR, env=env,
                          capture_output=True, text=True, timeout=600)


def fresh_run():
    d = tempfile.mkdtemp(prefix="g2hx", dir="/tmp")
    os.chmod(d, 0o755)
    L.prepare_run_dir(d, True, dict(L.DEFAULT_ROLES), None, None)
    cp = run_as("camlearn", "import os, sys\nos.makedirs(os.path.join(sys.argv[1], 'checkpoints'), exist_ok=True)\n", d)
    assert cp.returncode == 0, cp.stderr
    return d


def all_files(top):
    out = []
    for dp, dns, fns in os.walk(top):
        out += [os.path.join(dp, f) for f in fns]
    return out


SECRET_DIR = tempfile.mkdtemp(prefix="g2hsec", dir="/tmp")
os.chmod(SECRET_DIR, 0o700)
SECRET = os.path.join(SECRET_DIR, "root_secret.txt")
open(SECRET, "w").write("ROOT-ONLY-SECRET-g2h\n")
os.chmod(SECRET, 0o600)
runs = []
try:
    # X1 -------------------------------------------------------------------------------------------------------
    d = fresh_run(); runs.append(d)
    code = ("import json, os, sys\nd, target = sys.argv[1], sys.argv[2]\n"
            "os.symlink(target, os.path.join(d, 'checkpoints', 'ckpt_7'))\n"
            "json.dump({'status': 'stopped', 'committed': {'ckpt': {'id': 'ckpt_7', 'path': os.path.join(d, 'checkpoints', 'ckpt_7')}}},"
            " open(os.path.join(d, 'state.json'), 'w'))\n")
    cp = run_as("camlearn", code, d, SECRET_DIR)
    out = cli("export-checkpoint", "--run-dir", d, "--out", os.path.join(HOME, "x1"))
    leaked = [f for f in all_files(HOME) if "ROOT-ONLY-SECRET" in open(f, errors="replace").read()]
    R.check(not leaked, "a ckpt_<digits> symlink to a root-only directory exports nothing of it",
            f"exit {out.returncode}: {(out.stdout + out.stderr).strip()[-160:]!r}; secret in exports: {leaked}")
    # X2 -------------------------------------------------------------------------------------------------------
    d = fresh_run(); runs.append(d)
    code = ("import os, sys\nd, secret = sys.argv[1], sys.argv[2]\nc = os.path.join(d, 'checkpoints', 'ckpt_1')\n"
            "os.makedirs(c)\nopen(os.path.join(c, 'state.json'), 'w').write('{}')\n"
            "try:\n    os.link(secret, os.path.join(c, 'hard'))\n    print('hardlink created')\n"
            "except OSError as e:\n    print('hardlink refused', type(e).__name__)\n")
    cp = run_as("camlearn", code, d, SECRET)
    R.note(f"learner hard-links a root-owned 0600 file into its checkpoint: {cp.stdout.strip()} "
           f"(fs.protected_hardlinks={open('/proc/sys/fs/protected_hardlinks').read().strip()})")
    cp = run_as("camlearn", "import os, sys\nos.symlink(sys.argv[2], os.path.join(sys.argv[1], 'checkpoints', 'ckpt_1', 'link'))\n",
                d, SECRET)
    out = cli("export-checkpoint", "--run-dir", d, "--out", os.path.join(HOME, "x2"))
    leaked = [f for f in all_files(HOME) if "ROOT-ONLY-SECRET" in open(f, errors="replace").read()]
    R.check(out.returncode != 0 and not leaked, "a symlink inside a checkpoint refuses the export (nothing followed)",
            f"exit {out.returncode}: {(out.stdout + out.stderr).strip()[-160:]!r}; secret in exports: {leaked}")
    # X3 -------------------------------------------------------------------------------------------------------
    d = fresh_run(); runs.append(d)
    flip = ("import os, sys, time, shutil\nd, target = sys.argv[1], sys.argv[2]\nc = os.path.join(d, 'checkpoints')\n"
            "real, link, name = os.path.join(c, '.r'), os.path.join(c, '.l'), os.path.join(c, 'ckpt_8')\n"
            "end = time.time() + 25\n"
            "while time.time() < end:\n"
            "    try:\n"
            "        os.makedirs(real, exist_ok=True); open(os.path.join(real, 'state.json'), 'w').write('{}')\n"
            "        if os.path.islink(name): os.unlink(name)\n"
            "        if os.path.isdir(name): shutil.rmtree(name)\n"
            "        os.rename(real, name)\n"
            "        os.symlink(target, link); shutil.rmtree(name); os.rename(link, name)\n"
            "    except OSError:\n        pass\n")
    pw = pwd.getpwnam("camlearn")
    racer = subprocess.Popen([sys.executable, "-I", "-c", flip, d, SECRET_DIR], user=pw.pw_uid, group=pw.pw_gid,
                             extra_groups=[grp.getgrnam("camlive").gr_gid], cwd="/", env={"PATH": "/usr/bin:/bin"})
    results = []
    for i in range(15):
        out = cli("export-checkpoint", "--run-dir", d, "--checkpoint", "ckpt_8", "--out", os.path.join(HOME, f"x3_{i}"))
        results.append(out.returncode)
    racer.kill()
    leaked = [f for f in all_files(HOME) if "ROOT-ONLY-SECRET" in open(f, errors="replace").read()]
    R.check(not leaked, "15 exports while the learner flips ckpt_8 between a directory and a symlink copy nothing "
                        "of the root-only directory", f"exit codes {results}; secret in exports: {leaked}")
    # X4 -------------------------------------------------------------------------------------------------------
    d = fresh_run(); runs.append(d)
    code = ("import json, os, sys\nd, secret = sys.argv[1], sys.argv[2]\nc = os.path.join(d, 'checkpoints', 'ckpt_2')\n"
            "os.makedirs(c)\nopen(os.path.join(c, 'state.json'), 'w').write('{}')\n"
            "json.dump({'format': 1, 'checkpoint_id': 'ckpt_2', 'seq': 2, 'files': {secret: {'bytes': 1, 'sha256': '0' * 64},"
            " '../../../../../../etc/hostname': {'bytes': 1, 'sha256': '0' * 64}}}, open(os.path.join(c, 'MANIFEST.json'), 'w'))\n")
    cp = run_as("camlearn", code, d, SECRET)
    out = cli("export-checkpoint", "--run-dir", d, "--out", os.path.join(HOME, "x4"))
    msg = (out.stdout + out.stderr).strip()
    R.check(str(os.path.getsize(SECRET)) not in msg.split("!=")[0] if "!=" in msg else True,
            "a learner MANIFEST naming files outside the checkpoint does not make root stat / hash them",
            f"exit {out.returncode}: {msg[-260:]!r}")
    # X5 -------------------------------------------------------------------------------------------------------
    d = fresh_run(); runs.append(d)
    code = ("import hashlib, json, os, sys\nd = sys.argv[1]\nc = os.path.join(d, 'checkpoints', 'ckpt_3')\nos.makedirs(c)\n"
            "data = b'weights of CAM'\nopen(os.path.join(c, 'blob.bin'), 'wb').write(data)\n"
            "json.dump({'format': 1, 'checkpoint_id': 'ckpt_3', 'seq': 3, 'files': {'blob.bin': {'bytes': len(data), "
            "'sha256': hashlib.sha256(data).hexdigest()}}}, open(os.path.join(c, 'MANIFEST.json'), 'w'))\n"
            "json.dump({'status': 'stopped', 'committed': {'ckpt': {'id': 'ckpt_3', 'path': c}}}, open(os.path.join(d, 'state.json'), 'w'))\n")
    cp = run_as("camlearn", code, d)
    out = cli("export-checkpoint", "--run-dir", d)
    msg = (out.stdout + out.stderr).strip()
    exp_root = os.path.join(HOME, "exports")
    st = os.stat(exp_root) if os.path.isdir(exp_root) else None
    w = run_as("camlearn", "import os, sys\nprint(os.access(sys.argv[1], os.W_OK))\n", exp_root).stdout.strip()
    R.check(out.returncode == 0 and os.path.realpath(exp_root).startswith(HOME) and w != "True",
            "the default export lands outside the run directory, in a directory the learner cannot write",
            f"exit {out.returncode}: {msg[-200:]!r}; exports dir mode {oct(st.st_mode & 0o777) if st else None}; "
            f"learner can write it: {w}")
    # X6 -------------------------------------------------------------------------------------------------------
    sys.path.insert(0, os.path.join(os.path.dirname(SRC), "tests", "runtime"))
    from cam_live.runtime import plan_report as PR  # noqa: E402
    import test_int3b_ops as T  # noqa: E402
    tmp = tempfile.mkdtemp(prefix="g2hpr", dir="/tmp")

    class _P:                      # pytest's tmp_path stand-in
        def __init__(self, p): self.p = p
        def __truediv__(self, x): return os.path.join(self.p, x)
    w6 = T._note_run(_P(tmp))
    notes = os.path.join(w6["run"], "exec", "workspace", "notes.jsonl")
    rows = [json.loads(x) for x in open(notes)]
    rows[0]["text"] = "Plan step\x1b]0;G2H-TITLE\x07 hidden\x1b[8m invisible part\x1b[0m \x1b[2J"
    open(notes, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    md = PR.render_markdown(PR.build_report(w6["run"]))
    R.check("\x1b" not in md, "terminal control sequences in a CAM note are neutralised in the Markdown report",
            f"ESC bytes in the report: {md.count(chr(27))}")
    shutil.rmtree(tmp, ignore_errors=True)
    d = fresh_run(); runs.append(d)
    run_as("camlearn", "import json, os, sys\njson.dump({'status': 'running'}, open(os.path.join(sys.argv[1], 'state.json'), 'w'))\n", d)
    out = cli("plan-report", "--run-dir", d)
    rp = os.path.join(d, "plan_report.md")
    att = run_as("camlearn", "import os, sys\np = sys.argv[1]\nres = {}\n"
                 "for k, fn in (('replace', lambda: (open(p + '.x', 'w').write('fake'), os.replace(p + '.x', p))),"
                 " ('unlink', lambda: os.unlink(p)), ('write', lambda: open(p, 'a').write('x'))):\n"
                 "    try:\n        fn(); res[k] = 'ok'\n    except Exception as e:\n        res[k] = type(e).__name__\n"
                 "print(res)\n", rp)
    R.check(out.returncode == 0 and "ok" not in att.stdout, "the learner cannot replace, remove or extend the report "
                                                            "root wrote into the run directory",
            f"plan-report exit {out.returncode}; owner uid {os.stat(rp).st_uid if os.path.exists(rp) else None}; "
            f"learner: {att.stdout.strip()}")
finally:
    for x in runs + [HOME, SECRET_DIR]:
        shutil.rmtree(x, ignore_errors=True)
R.done()
