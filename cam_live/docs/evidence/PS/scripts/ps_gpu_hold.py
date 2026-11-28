"""PS: run several full-size jobs inside ONE hold of the shared GPU lock (the children get --no-lock; the parent holds
``cam_live.nn.gpu.gpu_lock`` for all of them, as ``eval.oe_batch`` does), stopping before the hold limit.

    python ps_gpu_hold.py --hold-s 840 -- "CMD1" "CMD2" ...
"""
import argparse
import os
import shlex
import subprocess
import sys
import time


def main() -> int:
    argv = sys.argv[1:]
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-s", type=float, default=840.0)
    ap.add_argument("--min-free-gb", type=float, default=12.0)
    a = ap.parse_args(argv[:cut])
    cmds = argv[cut + 1:]
    from cam_live.nn.gpu import gpu_lock
    with gpu_lock(min_free_gb=a.min_free_gb, timeout_s=4 * 3600.0, poll_s=1.0):
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] lock held", flush=True)
        for c in cmds:
            left = a.hold_s - (time.time() - t0) - 20.0
            if left < 60:
                print(f"skipped (hold limit): {c}", flush=True)
                continue
            print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] run ({left:.0f} s left): {c}", flush=True)
            try:
                r = subprocess.run(shlex.split(c), timeout=left)
                print(f"exit {r.returncode}", flush=True)
            except subprocess.TimeoutExpired:
                print("stopped at the hold limit", flush=True)
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] released after {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
