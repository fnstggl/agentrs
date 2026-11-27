"""Benchmark of the visual grounders on controlled computer-use tasks (workstream EC, spec 12A.3).

    python -m cam_live.harness.grounding.bench [--grounders a,b,...] [--per-family N] [--out FILE]

Every available grounder operates the same canvas-drawn pages (:mod:`bench_pages`) in headless Chromium **through
the visual executor**: screenshot, proposal, validation, authorization check, click, re-observation, verification.
Measured per grounder:

* ``grounding_accuracy``     — the proposed point lies inside the target's rectangle;
* ``e2e_success``            — the click really put the page into the expected state (read from the page) and the
                               executor's own verification agreed;
* ``incorrect_actions``      — a different control was activated; ``no_effect`` — nothing was hit;
                               ``no_proposal`` — the grounder returned no usable point (nothing was clicked);
* ``recovery``               — every failed operation is reproduced and retried once with the fallback grounder
                               on the screen as the failure left it;
* latency, GPU memory, and compute seconds per successful operation.

A model that needs more than 4 GB of GPU memory is run while holding the shared GPU lock, for at most
``--lock-budget-s`` seconds; what does not fit is reported as not run.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ... import contracts as c
from ...config import GPU_LOCK_PATH, PROJECT_ROOT
from .. import visual as V
from . import REGISTRY, available_grounders, load_grounder
from . import bench_pages as P

DEFAULT_OUT = os.path.join(PROJECT_ROOT, "docs", "evidence", "EC", "grounding_bench.json")
FALLBACK = {"goclick-base": "mai-ui-2b", "mai-ui-2b": "goclick-base", "vocaela-2-500m": "goclick-base",
            "reference-template": "goclick-base"}


class PageScreen:
    """Playwright page as a ``ScreenProvider`` (the benchmark does not depend on the browser executor)."""

    def __init__(self, page: Any):
        self.page = page

    def screenshot(self) -> bytes:
        return self.page.screenshot(type="png", full_page=False)

    def size(self) -> Tuple[int, int]:
        v = self.page.viewport_size
        return int(v["width"]), int(v["height"])

    def click(self, x: float, y: float) -> None:
        self.page.mouse.click(x, y)

    def double_click(self, x: float, y: float) -> None:
        self.page.mouse.dblclick(x, y)

    def drag(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.page.mouse.move(x1, y1)
        self.page.mouse.down()
        self.page.mouse.move(x2, y2, steps=8)
        self.page.mouse.up()

    def scroll(self, dx: float, dy: float) -> None:
        self.page.mouse.wheel(dx, dy)

    def type(self, text: str) -> None:
        self.page.keyboard.type(text)

    def key(self, combo: str) -> None:
        self.page.keyboard.press(combo)


class _AllowLocal:
    """Authorization stand-in for this offline benchmark: the pages are local documents with no network access.
    It allows and counts every primitive so the report can show that each click was checked."""

    def __init__(self) -> None:
        self.calls: Dict[str, int] = {}

    def __call__(self, op: c.PrimitiveOp, action: c.ActionSpec) -> c.AuthorizationDecision:
        self.calls[op.op] = self.calls.get(op.op, 0) + 1
        return c.AuthorizationDecision(verdict=c.AuthVerdict.ALLOW)


@contextlib.contextmanager
def _gpu_lock(needed: bool, timeout_s: float) -> Iterator[bool]:
    if not needed:
        yield True
        return
    os.makedirs(os.path.dirname(GPU_LOCK_PATH), exist_ok=True)
    f = open(GPU_LOCK_PATH, "a+")
    got, end = False, time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
                break
            except BlockingIOError:
                if time.monotonic() >= end:
                    break
                time.sleep(5)
        yield got
    finally:
        if got:
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _verify(task: P.Task, stage: int) -> c.VerificationCondition:
    x, y = task.progress_probe(stage)
    return c.VerificationCondition(description=f"progress box {stage + 1} is green", kind="element_state",
                                   spec={"color_at": [x, y], "rgb": list(P.GREEN), "tolerance": 60})


def _open(context: Any, task: P.Task, stage: int) -> Any:
    """A fresh page showing ``task`` at ``stage`` (earlier stages are completed by clicking their true targets)."""
    page = context.new_page()
    page.set_content(task.html, wait_until="load")
    for _ in range(stage):
        x, y, w, h = page.evaluate("window.__targets()[window.__target()]")
        page.mouse.click(x + w / 2, y + h / 2)
    return page


def _attempt(page: Any, task: P.Task, stage: int, name: str, grounder: Any, policy: _AllowLocal, run_dir: str) -> Dict[str, Any]:
    """One grounded click through the visual executor; everything recorded is read back from the page."""
    target_id = page.evaluate("window.__target()")
    x, y, w, h = page.evaluate("window.__targets()")[target_id]
    before = page.evaluate("({stage: window.__state.stage, hits: window.__state.hits.length, wrong: window.__state.wrong})")
    ex = V.VisualExecutor(PageScreen(page), grounders={name: grounder},
                          cfg=V.VisualConfig(grounder_order=(name,), max_ground_attempts=1, manage_gpu_lock=False,
                                             save_images=False))
    ctx = c.ExecContext(clock=c.WallClock(), check=policy, obs_sink=lambda o: None, run_dir=run_dir,
                        cancel=threading.Event())
    action = c.ActionSpec(action_id=c.new_id("act"), operation="visual.click_target", modality=c.Modality.VISUAL,
                          arguments={"target": task.targets[stage]}, verify=_verify(task, stage))
    t0 = time.monotonic()
    res = ex.execute(action, ctx)
    wall = time.monotonic() - t0
    state = page.evaluate("window.__state")
    ground_op = next((o for o in res.ops if o.op == "visual.ground"), None)
    clicked = any(o.op == "visual.click" and o.status == c.ResultStatus.OK for o in res.ops)
    point = None
    if ground_op is not None and "x" in ground_op.arguments:
        point = [ground_op.arguments["x"], ground_op.arguments["y"]]
    new_hits = state["hits"][before["hits"]:]
    advanced = state["stage"] == before["stage"] + 1
    if advanced and new_hits == [target_id]:
        outcome = "success"
    elif not clicked:
        outcome = "no_proposal"
    elif new_hits and new_hits[-1] not in ("none", target_id):
        outcome = "incorrect_action"
    else:
        outcome = "no_effect"
    return {"task": task.task_id, "family": task.family, "stage": stage, "target": task.targets[stage],
            "bbox": [x, y, w, h], "point": point, "outcome": outcome,
            "grounding_correct": bool(point and x <= point[0] < x + w and y <= point[1] < y + h),
            "executor_status": res.status, "executor_verified": res.verified, "failure_class": res.failure_class,
            "page_hits": new_hits, "model_seconds": float(res.cost.compute_s), "action_seconds": wall,
            "ops": [o.op for o in res.ops], "detail": res.failure_detail[:160]}


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def _summary(rows: List[Dict[str, Any]], tasks: List[P.Task]) -> Dict[str, Any]:
    n = len(rows)
    ok = [r for r in rows if r["outcome"] == "success"]
    model_s = [r["model_seconds"] for r in rows if r["point"] is not None or r["model_seconds"] > 0]
    fam: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        f = fam.setdefault(r["family"], {"targets": 0, "grounding_correct": 0, "e2e_success": 0, "incorrect_actions": 0})
        f["targets"] += 1
        f["grounding_correct"] += r["grounding_correct"]
        f["e2e_success"] += r["outcome"] == "success"
        f["incorrect_actions"] += r["outcome"] == "incorrect_action"
    for f in fam.values():
        f["grounding_accuracy"] = f["grounding_correct"] / f["targets"]
        f["e2e_success_rate"] = f["e2e_success"] / f["targets"]
    flows = [t for t in tasks if t.stages > 1]
    flow_ok = sum(all(r["outcome"] == "success" for r in rows if r["task"] == t.task_id)
                  and sum(r["task"] == t.task_id for r in rows) == t.stages for t in flows)
    total_model = sum(r["model_seconds"] for r in rows)
    return {"targets": n, "grounding_accuracy": sum(r["grounding_correct"] for r in rows) / n if n else 0.0,
            "e2e_success": len(ok), "e2e_success_rate": len(ok) / n if n else 0.0,
            "incorrect_actions": sum(r["outcome"] == "incorrect_action" for r in rows),
            "no_effect": sum(r["outcome"] == "no_effect" for r in rows),
            "no_proposal": sum(r["outcome"] == "no_proposal" for r in rows),
            "executor_verification_agrees": sum((r["executor_verified"] is True) == (r["outcome"] == "success") for r in rows),
            "workflows": len(flows), "workflows_completed_without_help": flow_ok,
            "latency_s": {"model_mean": statistics.fmean(model_s) if model_s else 0.0, "model_median": _pct(model_s, 0.5),
                          "model_p90": _pct(model_s, 0.9),
                          "action_mean": statistics.fmean(r["action_seconds"] for r in rows) if rows else 0.0},
            "compute_s_total": total_model,
            "compute_s_per_successful_operation": total_model / len(ok) if ok else None,
            "by_family": fam}


def run(names: List[str], per_family: int = 10, out: str = DEFAULT_OUT, lock_budget_s: float = 780.0,
        lock_wait_s: float = 900.0, recovery: bool = True, families: Optional[List[str]] = None) -> Dict[str, Any]:
    import torch
    from playwright.sync_api import sync_playwright
    tasks = P.make_tasks(per_family, families)
    avail = available_grounders()
    run_dir = os.path.join(os.path.dirname(out), "bench_run")
    os.makedirs(run_dir, exist_ok=True)
    report: Dict[str, Any] = {
        "what": "visual grounders on controlled canvas tasks, operated through cam_live.harness.visual.VisualExecutor",
        "command": "python -m cam_live.harness.grounding.bench " + " ".join(sys.argv[1:]),
        "started_unix": time.time(), "viewport": [P.W, P.H], "per_family": per_family,
        "families": sorted({t.family for t in tasks}), "tasks": len(tasks), "targets": sum(t.stages for t in tasks),
        "host": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "torch": torch.__version__, "load_average": list(os.getloadavg()),
                 "note": "the GPU and CPUs are shared with other jobs; latencies include that contention"},
        "grounders": {}, "not_run": {}}
    try:
        report["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                          cwd=PROJECT_ROOT).stdout.strip()
    except OSError:
        report["commit"] = ""
    rows_by: Dict[str, List[Dict[str, Any]]] = {}
    rec_rows: Dict[str, List[Dict[str, Any]]] = {}
    policy = _AllowLocal()

    def make(name: str) -> Any:
        if name == "reference-template":
            return load_grounder(name)
        return load_grounder(name, accept_noncommercial=True) if REGISTRY[name]["noncommercial"] else load_grounder(name)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        context = browser.new_context(viewport={"width": P.W, "height": P.H}, device_scale_factor=1)
        try:
            # ---- phase 1: every grounder on every target ---------------------------------------------------
            for name in names:
                if name not in REGISTRY:
                    report["not_run"][name] = "unknown grounder"
                    continue
                if not avail[name]["available"]:
                    report["not_run"][name] = avail[name]["unavailable_reason"]
                    continue
                heavy = bool(REGISTRY[name]["needs_gpu_lock"])
                with _gpu_lock(heavy, lock_wait_s) as got:
                    if not got:
                        report["not_run"][name] = "the shared GPU lock was not free in time"
                        continue
                    t_lock = time.monotonic()
                    g = make(name)
                    rows: List[Dict[str, Any]] = []
                    stopped = ""
                    try:
                        for task in tasks:
                            for stage in range(task.stages):
                                if heavy and time.monotonic() - t_lock > lock_budget_s:
                                    stopped = f"stopped after {lock_budget_s:.0f}s under the shared GPU lock"
                                    break
                                page = _open(context, task, stage)
                                try:
                                    rows.append(_attempt(page, task, stage, name, g, policy, run_dir))
                                finally:
                                    page.close()
                            if stopped:
                                break
                        info = g.info()
                    except Exception as e:  # noqa: BLE001 - a grounder that cannot run is reported, not hidden
                        report["not_run"][name] = f"{type(e).__name__}: {e}"[:400]
                        continue
                    finally:
                        g.close()
                    rows_by[name] = rows
                    entry = _summary(rows, tasks)
                    entry.update({"license": REGISTRY[name]["license"], "revision": REGISTRY[name]["revision"],
                                  "repo": REGISTRY[name]["repo"], "role": REGISTRY[name]["role"],
                                  "baseline": REGISTRY[name]["baseline"], "isolated_process": bool(info.get("isolated")),
                                  "device": info.get("device"), "params": info.get("params"),
                                  "load_s": info.get("load_s"), "gpu_mem_mb_after_load": info.get("gpu_mem_mb"),
                                  "gpu_mem_mb_peak": info.get("gpu_peak_mb"), "held_shared_gpu_lock": heavy,
                                  "lock_held_s": time.monotonic() - t_lock if heavy else 0.0, "partial": stopped})
                    report["grounders"][name] = entry
                    print(f"[bench] {name}: grounding {entry['grounding_accuracy']:.3f}  e2e {entry['e2e_success_rate']:.3f}  "
                          f"incorrect {entry['incorrect_actions']}  model {entry['latency_s']['model_mean']:.2f}s", flush=True)
                    _write(report, rows_by, out)
            # ---- phase 2: reproduce each failure and retry once with the fallback grounder -----------------
            if recovery:
                for fb in sorted({FALLBACK[n] for n in rows_by}):
                    users = [n for n in rows_by if FALLBACK[n] == fb and any(r["outcome"] != "success" for r in rows_by[n])]
                    if not users:
                        continue
                    if not avail.get(fb, {}).get("available"):
                        for n in users:
                            report["grounders"][n]["recovery"] = {"fallback": fb, "not_run": avail.get(fb, {}).get(
                                "unavailable_reason", "unknown grounder")}
                        continue
                    heavy = bool(REGISTRY[fb]["needs_gpu_lock"])
                    with _gpu_lock(heavy, lock_wait_s) as got:
                        if not got:
                            for n in users:
                                report["grounders"][n]["recovery"] = {"fallback": fb, "not_run": "shared GPU lock busy"}
                            continue
                        t_lock = time.monotonic()
                        g = make(fb)
                        try:
                            task_by = {t.task_id: t for t in tasks}
                            for n in users:
                                failed = [r for r in rows_by[n] if r["outcome"] != "success"]
                                rec: List[Dict[str, Any]] = []
                                for r in failed:
                                    if heavy and time.monotonic() - t_lock > lock_budget_s:
                                        break
                                    task = task_by[r["task"]]
                                    page = _open(context, task, r["stage"])
                                    try:
                                        if r["point"] is not None and r["outcome"] != "no_proposal":
                                            page.mouse.click(*r["point"])       # the failed click, as it happened
                                        a = _attempt(page, task, r["stage"], fb, g, policy, run_dir)
                                    finally:
                                        page.close()
                                    a["after"] = r["outcome"]
                                    rec.append(a)
                                done = sum(a["outcome"] == "success" for a in rec)
                                by_kind: Dict[str, List[int]] = {}
                                for a in rec:
                                    k = by_kind.setdefault(a["after"], [0, 0])
                                    k[0] += 1
                                    k[1] += a["outcome"] == "success"
                                report["grounders"][n]["recovery"] = {
                                    "fallback": fb, "failures": len(failed), "retried": len(rec), "recovered": done,
                                    "recovery_rate": done / len(rec) if rec else None,
                                    "by_failure_kind": {k: {"retried": v[0], "recovered": v[1]} for k, v in by_kind.items()},
                                    "new_incorrect_actions": sum(a["outcome"] == "incorrect_action" for a in rec),
                                    "compute_s": sum(a["model_seconds"] for a in rec),
                                    "success_rate_after_one_recovery": (report["grounders"][n]["e2e_success"] + done)
                                    / report["grounders"][n]["targets"]}
                                rec_rows[n + " -> recovery with " + fb] = rec
                        finally:
                            g.close()
                    _write(report, {**rows_by, **rec_rows}, out)
            for n in rows_by:
                if n in report["grounders"] and "recovery" not in report["grounders"][n]:
                    report["grounders"][n]["recovery"] = {"fallback": FALLBACK.get(n), "failures": 0, "retried": 0,
                                                          "recovered": 0, "recovery_rate": None}
        finally:
            context.close()
            browser.close()
    report["authorization_checks"] = dict(policy.calls)
    report["finished_unix"] = time.time()
    report["host"]["load_average_end"] = list(os.getloadavg())
    _write(report, {**rows_by, **rec_rows}, out)
    return report


def _write(report: Dict[str, Any], rows_by: Dict[str, List[Dict[str, Any]]], out: str) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    os.replace(tmp, out)
    with open(os.path.splitext(out)[0] + "_rows.json.tmp", "w") as f:
        json.dump(rows_by, f)
    os.replace(os.path.splitext(out)[0] + "_rows.json.tmp", os.path.splitext(out)[0] + "_rows.json")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grounders", default="reference-template,goclick-base,vocaela-2-500m,mai-ui-2b")
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--families", default="")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--lock-budget-s", type=float, default=780.0)
    ap.add_argument("--lock-wait-s", type=float, default=900.0)
    ap.add_argument("--no-recovery", action="store_true")
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    report = run([n for n in args.grounders.split(",") if n], args.per_family, args.out, args.lock_budget_s,
                 args.lock_wait_s, not args.no_recovery, [f for f in args.families.split(",") if f] or None)
    for name, e in report["grounders"].items():
        rec = e.get("recovery", {})
        print(f"{name:20s} grounding={e['grounding_accuracy']:.3f} e2e={e['e2e_success_rate']:.3f} "
              f"incorrect={e['incorrect_actions']:3d} no_effect={e['no_effect']:3d} no_proposal={e['no_proposal']:3d} "
              f"model_s={e['latency_s']['model_mean']:.3f} gpu_peak_mb={e['gpu_mem_mb_peak']} "
              f"recovered={rec.get('recovered')}/{rec.get('retried')}")
    for name, why in report["not_run"].items():
        print(f"{name:20s} NOT RUN: {why}")
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
