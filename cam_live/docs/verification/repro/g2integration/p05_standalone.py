"""G2-integration probe p05 — is CAM Live standalone at run time? (IN-09)

Imports every module of the cam_live package (pkgutil walk) in one process and reports
  * any loaded module of CAM Unified / Crucible / AgentRS (top-level names superlearner, crucible, agentrs, cam7,
    cam_unified) or any module file under those projects' source trees;
  * sys.path entries that point into another project;
  * where third-party packages are loaded from (the venv is layered over another project's venv);
  * configuration defaults that point into another project's directories;
  * imports of those projects in tools/ and tests/ (the checker scans src/ only).

Usage (from /workspace/camlive):
    PYTHONPATH=<copy>/cam_live/src .venv/bin/python -I cam_live/docs/verification/repro/g2integration/p05_standalone.py
(-I is not usable with PYTHONPATH; run without -I.)
"""
from __future__ import annotations

import ast
import collections
import glob
import importlib
import os
import pkgutil
import sys
import traceback

FORBIDDEN = ("superlearner", "crucible", "agentrs", "cam7", "cam_unified")
OTHER_TREES = ("/workspace/agentrs/src", "/workspace/agentrs/superlearner", "/workspace/agentrs/crucible",
               "/workspace/camlive/superlearner", "/workspace/camlive/crucible", "/workspace/camlive/src")
broken = 0


def check(name, ok, detail=""):
    global broken
    broken += 0 if ok else 1
    print(f"[{'HOLDS ' if ok else 'BROKEN'}] {name}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    import cam_live
    pkg_dir = os.path.dirname(cam_live.__file__)
    print(f"cam_live loaded from {pkg_dir}")
    failed = {}
    skip = ("cam_live.harness.staging_discord.gateway",)   # nothing skipped by default; kept for reruns
    for m in pkgutil.walk_packages([pkg_dir], "cam_live."):
        if m.name in skip:
            continue
        try:
            importlib.import_module(m.name)
        except BaseException as e:  # noqa: BLE001
            failed[m.name] = f"{type(e).__name__}: {str(e)[:120]}"
    n = sum(1 for k in sys.modules if k == "cam_live" or k.startswith("cam_live."))
    print(f"imported {n} cam_live modules; import failures: {len(failed)} {dict(list(failed.items())[:5])}")
    bad = sorted(k for k in sys.modules if k.split(".")[0] in FORBIDDEN)
    check("1 no module of CAM Unified / Crucible / AgentRS is loaded by importing all of cam_live", not bad, bad[:10])
    files = sorted({getattr(mod, "__file__", "") or "" for mod in list(sys.modules.values())})
    other = [f for f in files if any(f.startswith(t) for t in OTHER_TREES)]
    check("2 no loaded module file lies in another project's source tree", not other, other[:10])
    paths = [p for p in sys.path if p.startswith("/workspace/agentrs")]
    print(f"[NOTE  ] sys.path entries inside /workspace/agentrs: {paths}")
    by_root = collections.Counter()
    for f in files:
        if "/site-packages/" in f:
            by_root[f.split("/site-packages/")[0] + "/site-packages"] += 1
    print(f"[NOTE  ] third-party modules loaded per site-packages directory: {dict(by_root)}")
    check("3 the run-time Python environment does not live inside another project's directory",
          not any(r.startswith("/workspace/agentrs") for r in by_root),
          "torch, transformers, ... are loaded from /workspace/agentrs/.venv (camlive/.venv is layered over it by "
          "_shared_base.pth)" if any(r.startswith("/workspace/agentrs") for r in by_root) else "")
    from cam_live import config
    check("4 no configuration default points into another project's directory",
          not config.GPU_LOCK_PATH.startswith("/workspace/agentrs"),
          f"config.GPU_LOCK_PATH = {config.GPU_LOCK_PATH} (CAM Unified's run directory)")

    root = os.path.abspath(os.path.join(pkg_dir, "..", ".."))
    hits = []
    for f in sorted(glob.glob(os.path.join(root, "tools", "**", "*.py"), recursive=True)
                    + glob.glob(os.path.join(root, "tests", "**", "*.py"), recursive=True)):
        try:
            tree = ast.parse(open(f).read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            for nm in names:
                if nm.split(".")[0] in FORBIDDEN:
                    hits.append(f"{os.path.relpath(f, root)}:{node.lineno}: {nm}")
    check("5 tools/ and tests/ import none of those projects", not hits, hits[:10])
    sh = []
    for f in glob.glob(os.path.join(root, "tools", "*.sh")):
        for i, line in enumerate(open(f), 1):
            if "/workspace/agentrs" in line or ".config/agentrs" in line:
                sh.append(f"{os.path.relpath(f, root)}:{i}")
    print(f"[NOTE  ] shell tools that refer to another project's paths: {sh}")
    print(f"== p05_standalone: {broken} broken of 5 checks")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
