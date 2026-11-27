"""A small real service for launcher tests: optional RPC socket, reports who it runs as, can spawn a child, can
crash on demand. Started by the launcher as ``python svc_dummy.py --run-dir R [--socket S] [...]``."""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time


class Target:
    def __init__(self, a):
        self.a = a

    def whoami(self):
        with open("/proc/self/status") as f:
            status = f.read()
        nnp = [line.split()[1] for line in status.splitlines() if line.startswith("NoNewPrivs")][0]
        caps = [line.split()[1] for line in status.splitlines() if line.startswith("CapEff")][0]
        return {"uid": os.getuid(), "gid": os.getgid(), "groups": sorted(os.getgroups()), "pid": os.getpid(),
                "env_keys": sorted(os.environ), "no_new_privs": nnp, "cap_eff": caps, "home": os.environ.get("HOME", ""),
                "umask": os.umask(0o077)}

    def can_read(self, path):
        try:
            with open(path, "rb") as f:
                f.read(1)
            return {"ok": True}
        except OSError as e:
            return {"ok": False, "errno": e.errno}

    def crash(self):
        threading.Timer(0.05, lambda: os._exit(7)).start()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--socket", default="")
    ap.add_argument("--spawn-child", action="store_true")
    ap.add_argument("--detached-child", action="store_true")
    ap.add_argument("--exit-after", type=float, default=0.0)
    ap.add_argument("--exit-code", type=int, default=3)
    ap.add_argument("--marker", default="")
    a = ap.parse_args()
    if a.marker:
        with open(a.marker, "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "uid": os.getuid(), "t": time.time()}) + "\n")
    if a.spawn_child:
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    if a.detached_child:        # a grandchild in its own session whose parent exits at once (an orphan)
        subprocess.Popen([sys.executable, "-c",
                          "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'], "
                          "start_new_session=True)"]).wait()
    server = None
    if a.socket:
        from cam_live.harness import rpc
        server = rpc.RpcServer(a.socket, Target(a), ["whoami", "can_read", "crash"], mode=0o660, name="dummy").start()
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    if a.exit_after:
        done.wait(a.exit_after)
        if not done.is_set():
            os._exit(a.exit_code)
    done.wait()
    if server is not None:
        server.stop()


if __name__ == "__main__":
    main()
