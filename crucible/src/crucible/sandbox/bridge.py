"""Bridge between sandboxed programs and the kernel transaction.

A program may call back into the world (`crucible_sdk.call`). Those calls run *inside the current command's
transaction* as the program's principal, so they are authorised, recorded as events and rolled back together with
the command if it fails. The program's own output is non-deterministic (timing, CPU), so the bridge records:

    {"result": ..., "ok": ..., "error": ..., "stdout": ..., "cpu_ms": ..., "rpcs": [{"tool", "args"}, ...]}

During regeneration the recorded outcome is used and the recorded RPC sequence is re-issued through the kernel in
the same order (which re-creates the same events), so the regenerated event log is identical.

`cpu_ms` is measured by the parent outside the sandboxed process (D-20, `crucible.sandbox.metering`) and is never
negative; nothing the program reports about its own work is used. Under ``cost_model = "deterministic"`` it is the
parent-measured work in nominal milliseconds (work units x LINE_MS, remembered per computation so forks and reruns of
a world bill identical computations identically); otherwise it is the measured CPU time in milliseconds.
"""
from __future__ import annotations

from typing import Any, Callable

from ..core.errors import CrucibleError
from ..core.kernel import Ctx
from .runner import Worker

LINE_MS = 0.001   # nominal CPU milliseconds per work unit in the deterministic cost model (trusted code: per line)
_LN = [0]
_TOOL = [None]
_OWNER = [None]
_GC = [None]
_TOOL_NAME = "crucible-trusted-work"


def _lines_on():
    """Count executed Python lines with a sys.monitoring tool id we own (never a reserved id: 0 debugger,
    1 coverage, 2 profiler, 5 optimizer - torch and others use those)."""
    import sys
    mon = sys.monitoring
    tid = _TOOL[0]
    if tid is None or mon.get_tool(tid) != _TOOL_NAME:
        tid = next((i for i in (3, 4) if mon.get_tool(i) in (None, _TOOL_NAME)), None)
        if tid is None:
            raise RuntimeError("no free sys.monitoring tool id for deterministic work accounting")
        if mon.get_tool(tid) is None:
            mon.use_tool_id(tid, _TOOL_NAME)
        import threading

        def on_line(code, line, _get=threading.get_ident):
            # events are process-global: count only the measuring thread (background threads such as tqdm's
            # monitor or a web server would otherwise make the count nondeterministic)
            if _get() == _OWNER[0]:
                _LN[0] += 1
        mon.register_callback(tid, mon.events.LINE, on_line)
        _TOOL[0] = tid
    import gc
    import threading
    _OWNER[0] = threading.get_ident()
    _LN[0] = 0
    # 5.0: gc callbacks installed by other code (e.g. hypothesis' gc timer) do not run inside the measured region:
    # a collection landing there would add their lines to the count, which then depended on which libraries had
    # been used earlier in the process. 4.0 processes have no gc callbacks, so their counts are unchanged.
    _GC[0] = list(gc.callbacks)
    gc.callbacks.clear()
    mon.set_events(tid, mon.events.LINE)


def _lines_off() -> int:
    import sys
    sys.monitoring.set_events(_TOOL[0], 0)
    import gc                     # after counting stopped: the counted lines are exactly 4.0's
    if _GC[0]:
        gc.callbacks.extend(cb for cb in _GC[0] if cb not in gc.callbacks)
    _GC[0] = None
    return _LN[0]


def _rpc_fn(ctx: Ctx, principal: str, log: list, replies: list | None = None) -> Callable[[str, dict], tuple[bool, Any]]:
    def rpc(tool: str, args: dict) -> tuple[bool, Any]:
        log.append({"tool": tool, "args": args})
        try:
            ok, val = True, ctx.call(tool, args, as_principal=principal, brokered=True)
        except CrucibleError as e:
            ok, val = False, e.as_dict()
        if replies is not None:
            replies.append((tool, args, ok, val))
        return ok, val
    return rpc


def run_in_ctx(ctx: Ctx, kind: str, worker_fn: Callable[[], Worker], job: dict, principal: str,
               timeout: float | None = None, close_after: bool = False, world_access: bool = True,
               post: Callable[[Any], Any] | None = None, rpc_fn: Callable[[str, dict], tuple[bool, Any]] | None = None,
               rpc_results: list | None = None) -> dict:
    """Run `job` on the worker returned by `worker_fn()` with world access as `principal`; record the outcome.

    `rpc_fn` replaces the world broker (its calls are not world calls and are not recorded for regeneration, e.g.
    replayed replies during an ablation); `rpc_results`, if given, collects (tool, args, ok, reply) of every world
    call the program made (also during regeneration, where the recorded calls are re-issued)."""
    slot, rec = ctx.reserve()
    if rec is not None:
        for r in rec.get("rpcs", []):
            try:
                ok, val = True, ctx.call(r["tool"], r["args"], as_principal=principal, brokered=True)
            except CrucibleError as e:
                ok, val = False, e.as_dict()
            if rpc_results is not None:
                rpc_results.append((r["tool"], r["args"], ok, val))
        return ctx.fill(slot, kind, rec)
    log: list = []
    w = worker_fn()
    try:
        broker = rpc_fn if rpc_fn is not None else (_rpc_fn(ctx, principal, log, rpc_results) if world_access else None)
        res = w.run(job, rpc=broker, timeout=timeout)
    finally:
        if close_after:
            w.close()
    out = recorded_outcome(res, log, post)
    return ctx.fill(slot, kind, out)


def billed_cpu_ms(res: dict) -> float:
    """The parent-measured, non-negative CPU a sandbox job is billed (nominal ms in the deterministic model)."""
    if res.get("work_lines") is not None:      # deterministic cost model: 1 work unit = LINE_MS of nominal CPU
        cpu = round(max(0, int(res["work_lines"])) * LINE_MS, 4)
    else:
        cpu = float(res.get("cpu_ms") or 0.0)
    return cpu if cpu == cpu and cpu > 0 else 0.0


def recorded_outcome(res: dict, log: list, post: Callable[[Any], Any] | None = None) -> dict:
    result = res.get("result")
    if post is not None and res.get("ok"):
        result = post(result)      # e.g. score the output and record only the score
    out = {"ok": bool(res.get("ok")), "result": result, "error": res.get("error"),
           "stdout": (res.get("stdout") or "")[:4000], "cpu_ms": billed_cpu_ms(res), "rpcs": log}
    killed = (res.get("meter") or {}).get("killed")
    if killed:
        out["killed"] = killed
    return out


def run_trusted_in_ctx(ctx: Ctx, kind: str, fn: Callable[..., Any], args: Any, caller: Any, principal: str,
                       post: Callable[[Any], Any] | None = None) -> dict:
    """Same contract for trusted in-process code (built-in incumbents): measured CPU is recorded."""
    import time
    slot, rec = ctx.reserve()
    if rec is not None:
        return ctx.fill(slot, kind, rec)
    deterministic = ctx.world.spec.params.get("cost_model") == "deterministic"
    t0 = time.process_time()
    if deterministic:
        _lines_on()
    try:
        result = fn(args, caller)
        out = {"ok": True, "result": result, "error": None}
    except Exception as e:  # noqa: BLE001 - trusted code can still have bugs on odd inputs
        out = {"ok": False, "result": None, "error": {"type": type(e).__name__, "message": str(e)[:500]}}
    finally:
        lines = _lines_off() if deterministic else None
    out["cpu_ms"] = round(lines * LINE_MS, 4) if deterministic else round((time.process_time() - t0) * 1000.0, 4)
    out["stdout"] = ""
    out["rpcs"] = []
    if post is not None and out["ok"]:
        out["result"] = post(out["result"])
    return ctx.fill(slot, kind, out)
