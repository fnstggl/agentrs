"""Rebuild local run records from the latest GitHub Releases checkpoint bundle of every run.

Used after local run directories were lost. For each run name with a published ``ckpt-<run>-...`` release,
download and verify its newest bundle and lay it out under ``<runs>/<group>/<run>``:

  ckpt/<checkpoint>/ and ckpt/latest/    the checkpoint (weights, optimizer, config, state at save time)
  state.json, train.log, iters.jsonl ... the run-level files captured at upload time (newest record)
  RESTORED_FROM.json                     provenance: release tag, upload time, checkpoint name

Replay buffers are not part of CAM bundles, so a restored run is a complete *record* (state, evaluations,
logs, weights, optimizer) but resuming its training would start from an empty replay buffer.
Existing non-empty run directories are never overwritten.

    PYTHONPATH=src:/workspace/agentrs/src python scripts/restore_runs_from_github.py --runs /workspace/agentrs/runs
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

GROUPS = (("c21_", "cam2_v21"), ("c22_", "cam2_v22"), ("cam2_", "cam2"), ("hy_", "cam_hybrid"))


def group_of(run: str) -> str:
    for prefix, g in GROUPS:
        if run.startswith(prefix):
            return f"{g}/{run}"
    return run  # cam, cam_pfix, overnight, ...


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/workspace/agentrs/runs")
    ap.add_argument("--staging", default="/workspace/restore_staging")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    from agentrs import gh_backup as G

    gh = G.GitHub(G.DEFAULT_REPO)
    rels = [r for r in gh.releases() if not r["draft"] and r["tag_name"].startswith(G.TAG_PREFIX)
            and any(x["name"] == "MANIFEST.json" for x in r["assets"])]
    latest: dict = {}
    for r in rels:
        run = G.release_run(r)
        if not run or (a.only and run not in a.only.split(",")):
            continue
        latest.setdefault(run, []).append(r)
    for run in latest:  # newest first; older releases are fallbacks when a newer bundle is incomplete
        latest[run].sort(key=lambda r: r.get("published_at") or "", reverse=True)
    report = {}
    for run, cands in sorted(latest.items()):
        dest = Path(a.runs) / group_of(run)
        if dest.exists() and any(dest.iterdir()):
            report[run] = "exists: skipped"
            continue
        errors = []
        for r in cands:
            stage = Path(a.staging) / run
            t0 = time.time()
            try:
                res = G.restore(stage, tag=r["tag_name"], run_name=run)
                b = Path(res["dir"])
                info = json.loads((b / "BUNDLE_INFO.json").read_text())
                ck = b / "payload" / "ckpt" / info["checkpoint"]
                (dest / "ckpt").mkdir(parents=True, exist_ok=True)
                for name in {info["checkpoint"], "latest"}:
                    shutil.copytree(ck, dest / "ckpt" / name, dirs_exist_ok=True)
                src = b / "payload" / "run"
                if src.exists():
                    for p in src.iterdir():
                        if p.is_file():
                            shutil.copy2(p, dest / p.name)
                        elif p.is_dir():
                            shutil.copytree(p, dest / p.name, dirs_exist_ok=True)
                if not (dest / "state.json").exists() and (ck / "state.json").exists():
                    shutil.copy2(ck / "state.json", dest / "state.json")
                (dest / "RESTORED_FROM.json").write_text(json.dumps(
                    {"tag": r["tag_name"], "published_at": r["published_at"], "checkpoint": info["checkpoint"],
                     "newer_releases_unusable": errors, "restored_at": time.time(),
                     "note": "local run directory lost (disk full / cleanup, 2026-11-18); record restored from the newest "
                             "usable GitHub Releases bundle; replay buffer not included"}, indent=1))
                report[run] = f"restored {r['tag_name']} ({time.time() - t0:.0f}s)" + (f" after {len(errors)} unusable newer release(s)" if errors else "")
                break
            except Exception as e:  # noqa: BLE001 - try the next older release
                errors.append({"tag": r["tag_name"], "error": f"{type(e).__name__}: {e}"})
                if dest.exists() and not any(dest.iterdir()):
                    dest.rmdir()
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        else:
            report[run] = f"FAILED: no usable release ({errors[:2]})"
        print(run, report[run], flush=True)
    print(json.dumps(report, indent=1))
    return 0 if not any(v.startswith("FAILED") for v in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
