"""Storage retention for superlearner runs (a full disk killed every run on 2026-11-18; never again).

Removes only redundant or reproducible checkpoint copies; never results, logs, configs, or the checkpoints that
results depend on:
  * ``ckpt/m<N>/`` milestone checkpoints (their evaluation results live in state.json; the trainer never reads them);
  * in finished runs whose final checkpoint is verified on GitHub Releases: ``ckpt/latest`` (same as final), the
    optimizer / RNG copies in ``ckpt/initial`` (initial weights are kept), and ``ckpt/best`` when its weights are
    byte-identical to final's (2026-11-20: 13 GB of such duplicates held the queue's disk gate shut);
  * leftovers of interrupted atomic saves (``ckpt/.*.tmp``, ``ckpt/.*.old``) older than one hour;
  * checkpoints of smoke-test runs and of discarded runs (``_smoke*``, ``_invalid*``) - their logs/state are kept;
  * pytest temporary directories idle for 6 hours.
Always kept: ``final``, ``best``, ``initial/weights.pt``, ``life_*`` (transfer initialisations), ``latest`` of
running runs, every state.json / log / iters.jsonl.

    python scripts/sl_retention.py [--apply] [--loop 900]
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import time
from pathlib import Path

ROOTS = [Path("/workspace/agentrs/runs/sl"), Path("/workspace/agentrs/runs/sl_mp")]


def size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def final_backed_up(run: Path) -> bool:
    try:
        st = json.loads((run / "gh_backup_status.json").read_text())
    except (OSError, ValueError):
        return False
    return any(u.get("kind") == "final" and u.get("verified") for u in st.get("uploads", []))


def _active(run: Path, now: float) -> bool:
    """A run (e.g. a smoke test) still in progress: not finished and written to within the last hour."""
    try:
        if json.loads((run / "state.json").read_text()).get("phase") == "done":
            return False
    except (OSError, ValueError):
        pass
    try:
        newest = max(f.stat().st_mtime for f in run.rglob("*") if f.is_file())
    except ValueError:
        return False
    return now - newest < 3600


def plan() -> list:
    out = []
    now = time.time()
    for root in ROOTS:
        if not root.exists():
            continue
        for run in sorted(p for p in root.iterdir() if p.is_dir()):
            if run.name.startswith(("_smoke", "_invalid")):
                for ck in run.rglob("ckpt"):
                    if ck.is_dir() and not _active(ck.parent, now):
                        out.append((ck, "checkpoints of a smoke-test / discarded run"))
                continue
            if run.name.startswith("_"):
                continue
            ck = run / "ckpt"
            if not ck.is_dir():
                continue
            try:
                phase = json.loads((run / "state.json").read_text()).get("phase")
            except (OSError, ValueError):
                phase = None
            for d in ck.iterdir():
                if d.is_dir() and d.name.startswith("m") and d.name[1:].isdigit():
                    out.append((d, "milestone checkpoint (evaluation recorded in state.json)"))
                elif d.name.startswith(".") and (d.name.endswith(".tmp") or d.name.endswith(".old")) and now - d.stat().st_mtime > 3600:
                    out.append((d, "stale partial save"))
            if phase == "done" and final_backed_up(run) and (ck / "final" / "DONE").exists():
                if (ck / "latest").exists():
                    out.append((ck / "latest", "duplicate of final (final verified on GitHub)"))
                bw, fw = ck / "best" / "weights.pt", ck / "final" / "weights.pt"
                if bw.exists() and fw.exists() and filecmp.cmp(bw, fw, shallow=False):
                    # nothing reads ckpt/best after a run ends (the trainer only writes it); identical weights = duplicate
                    out.append((ck / "best", "best identical to final (final verified on GitHub)"))
                for f in ("optimizer.pt", "rng.pt"):
                    if (ck / "initial" / f).exists():
                        out.append((ck / "initial" / f, "initial optimizer/RNG copy (run finished, backed up)"))
    for p in Path("/tmp").glob("pytest-of-*"):
        # only idle ones: a running test session's tmp_path was deleted mid-test (2026-11-19)
        try:
            idle = now - max([p.stat().st_mtime] + [f.stat().st_mtime for f in p.rglob("*")])
        except OSError:
            continue
        if idle > 6 * 3600:
            out.append((p, "idle pytest temporary directories"))
    return out


def apply(items) -> int:
    freed = 0
    for p, _ in items:
        try:
            n = size(p)
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            freed += n
        except OSError:
            pass
    return freed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    while True:
        items = plan()
        tot = sum(size(p) for p, _ in items)
        by = {}
        for p, why in items:
            by[why] = by.get(why, 0) + size(p)
        free = shutil.disk_usage("/workspace").free / 2**30
        print(f"[retention {time.strftime('%Y-%m-%d %H:%M:%S')}] disk free {free:.1f} GB; {len(items)} items, "
              f"{tot / 2**30:.2f} GB removable: " + "; ".join(f"{k} {v / 2**30:.2f} GB" for k, v in by.items()), flush=True)
        if a.apply:
            print(f"[retention] freed {apply(items) / 2**30:.2f} GB", flush=True)
        if not a.loop:
            return 0
        time.sleep(a.loop)


if __name__ == "__main__":
    raise SystemExit(main())
