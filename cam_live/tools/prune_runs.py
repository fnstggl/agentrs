#!/usr/bin/env python
"""Checkpoint and artifact retention for CAM Live run trees (dry run by default).

Run data (`runs/`) is never in git. Over a lifetime, every runtime checkpoint and every learner promotion writes the
trainable weights *and* the optimizer state (Adam: 2x the weights); evaluation runs create one runtime per lifetime,
so a 150m-core evaluation run of four lifetimes held 17 GB. On 2026-11-27 this filled the instance's 300 GB disk and
stopped all work. This tool applies the retention policy below to finished run trees.

Policy (never touches anything but checkpoint directories):

* Records are always kept: every file outside a checkpoint directory (``*.jsonl`` logs, ``experience.sqlite`` and
  its blobs, ledgers, results, state, journal, evidence).
* A checkpoint directory is a directory named ``ckpt_*`` or ``export_*`` that contains ``weights/``, ``optim/``,
  ``*.safetensors`` or a ``MANIFEST*``/``meta*.json`` file.
* Protected checkpoints (``ckpt_pre``, ``ckpt_post``, ``export_*``, and any checkpoint named in a ``LATEST`` /
  ``PROMOTED`` pointer file) keep their weights; their optimizer state is removed with ``--drop-protected-optim``.
* In every group of sibling checkpoints (same parent directory) the newest ``--keep`` (default 1) are kept with their
  weights; older ones are deleted. Optimizer state is kept only in the newest of each group (``--keep-optim 1``,
  default) — set ``--keep-optim 0`` for finished runs that will not be resumed.
* ``--frozen-glob`` (e.g. ``'A_*' 'B_*'``): runs of conditions without persistent learning keep only ``export_*``.
* ``--drop-group`` (e.g. ``'*/lt[0-9][0-9]/checkpoints'``): sibling groups whose directory matches are removed
  entirely — for finished runs whose learning state is kept elsewhere (per-lifetime runtime checkpoints of an
  evaluation run whose final weights are in ``ckpt_post`` / ``export_post`` or the newest learner promotion).

    python cam_live/tools/prune_runs.py runs/cam_live/eval              # report what would be removed
    python cam_live/tools/prune_runs.py runs/cam_live/eval --apply --keep-optim 0 --drop-protected-optim
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
from typing import Dict, List, Tuple

PROTECTED_NAMES = ("ckpt_pre", "ckpt_post")


def dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def is_checkpoint(path: str) -> bool:
    name = os.path.basename(path)
    if not (name.startswith("ckpt_") or name.startswith("export_")):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return any(e in ("weights", "optim") or e.endswith(".safetensors") or e.startswith("MANIFEST")
               or (e.startswith("meta") and e.endswith(".json")) for e in entries)


def pointers(parent: str) -> List[str]:
    """Checkpoint names referenced by pointer files in ``parent`` (LATEST, PROMOTED, *.json with checkpoint ids)."""
    out: List[str] = []
    for fn in ("LATEST", "PROMOTED", "latest", "promoted"):
        p = os.path.join(parent, fn)
        if os.path.isfile(p):
            try:
                txt = open(p).read().strip()
            except OSError:
                continue
            out += [os.path.basename(t.strip().rstrip("/")) for t in txt.replace(",", "\n").splitlines() if t.strip()]
    return out


def plan(root: str, keep: int, keep_optim: int, drop_protected_optim: bool,
         frozen_globs: List[str], drop_groups: List[str] = ()) -> List[Tuple[str, str, int]]:
    """[(action, path, bytes)] with action 'rm' (whole checkpoint) or 'rm-optim' (its optimizer state)."""
    actions: List[Tuple[str, str, int]] = []
    groups: Dict[str, List[str]] = {}
    for dirpath, dirnames, _files in os.walk(root):
        found = [d for d in dirnames if is_checkpoint(os.path.join(dirpath, d))]
        for d in found:
            groups.setdefault(dirpath, []).append(os.path.join(dirpath, d))
        dirnames[:] = [d for d in dirnames if d not in found and not os.path.islink(os.path.join(dirpath, d))]
    for parent, ckpts in groups.items():
        rel_parts = os.path.relpath(parent, root).split(os.sep)
        frozen = any(fnmatch.fnmatch(part, g) for g in frozen_globs for part in rel_parts)
        if any(fnmatch.fnmatch(parent, g) for g in drop_groups):
            actions += [("rm", c, dir_size(c)) for c in ckpts]
            continue
        refs = set(pointers(parent))
        protected = [c for c in ckpts if os.path.basename(c) in PROTECTED_NAMES
                     or os.path.basename(c).startswith("export_") or os.path.basename(c) in refs]
        # newest first by the sequence number in the name (ckpt_00000003, ckpt_000005_<hash>); modification time
        # only breaks ties (removing an optimizer directory changes a checkpoint's mtime)
        others = sorted([c for c in ckpts if c not in protected], key=lambda c: (os.path.basename(c),
                                                                                 os.path.getmtime(c)), reverse=True)
        for c in protected:
            name = os.path.basename(c)
            if frozen and not name.startswith("export_"):
                actions.append(("rm", c, dir_size(c)))
            elif drop_protected_optim and os.path.isdir(os.path.join(c, "optim")):
                actions.append(("rm-optim", os.path.join(c, "optim"), dir_size(os.path.join(c, "optim"))))
        for i, c in enumerate(others):
            if frozen or i >= keep:
                actions.append(("rm", c, dir_size(c)))
            elif i >= keep_optim and os.path.isdir(os.path.join(c, "optim")):
                actions.append(("rm-optim", os.path.join(c, "optim"), dir_size(os.path.join(c, "optim"))))
    return actions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--keep", type=int, default=1, help="checkpoints kept per sibling group (newest first)")
    ap.add_argument("--keep-optim", type=int, default=1, help="of those, how many keep their optimizer state")
    ap.add_argument("--drop-protected-optim", action="store_true", help="remove optimizer state of ckpt_pre/post too")
    ap.add_argument("--frozen-glob", nargs="*", default=[], help="run-directory globs of frozen (non-learning) conditions")
    ap.add_argument("--drop-group", nargs="*", default=[], help="globs of checkpoint-group directories to remove entirely")
    ap.add_argument("--apply", action="store_true", help="delete (default: report only)")
    ap.add_argument("--report", default="", help="write the plan / result as JSON here")
    a = ap.parse_args(argv)
    if a.keep < 1:
        ap.error("--keep must be >= 1 (the newest checkpoint of a group is never removed by this tool)")
    all_actions: List[Tuple[str, str, int]] = []
    for r in a.roots:
        all_actions += plan(os.path.abspath(r), a.keep, a.keep_optim, a.drop_protected_optim, a.frozen_glob,
                            a.drop_group)
    total = sum(b for _a, _p, b in all_actions)
    print(f"{'APPLY' if a.apply else 'DRY RUN'}: {len(all_actions)} removals, {total / 2**30:.1f} GiB")
    done = []
    for act, path, size in all_actions:
        if a.apply:
            shutil.rmtree(path, ignore_errors=True)
        done.append({"action": act, "path": path, "bytes": size})
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        with open(a.report, "w") as f:
            json.dump({"applied": a.apply, "bytes": total, "removals": done,
                       "policy": {"keep": a.keep, "keep_optim": a.keep_optim,
                                  "drop_protected_optim": a.drop_protected_optim, "frozen_glob": a.frozen_glob,
                                  "drop_group": a.drop_group}},
                      f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
