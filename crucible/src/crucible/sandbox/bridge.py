"""Bridge between sandboxed programs and the kernel transaction.

A program may call back into the world (`crucible_sdk.call`). Those calls run *inside the current command's
transaction* as the program's principal, so they are authorised, recorded as events and rolled back together with
the command if it fails. The program's own output is non-deterministic (timing, CPU), so the bridge records:

    {"result": ..., "ok": ..., "error": ..., "stdout": ..., "cpu_ms": ..., "rpcs": [{"tool", "args"}, ...]}

During regeneration the recorded outcome is used and the recorded RPC sequence is re-issued through the kernel in
the same order (which re-creates the same events), so the regenerated event log is identical.
"""
from __future__ import annotations

from typing import Any, Callable

from ..core.errors import CrucibleError
from ..core.kernel import Ctx
from .runner import Worker


def _rpc_fn(ctx: Ctx, principal: str, log: list) -> Callable[[str, dict], tuple[bool, Any]]:
    def rpc(tool: str, args: dict) -> tuple[bool, Any]:
        log.append({"tool": tool, "args": args})
        try:
            return True, ctx.call(tool, args, as_principal=principal)
        except CrucibleError as e:
            return False, e.as_dict()
    return rpc


def run_in_ctx(ctx: Ctx, kind: str, worker_fn: Callable[[], Worker], job: dict, principal: str,
               timeout: float | None = None, close_after: bool = False, world_access: bool = True) -> dict:
    """Run `job` on the worker returned by `worker_fn()` with world access as `principal`; record the outcome."""
    slot, rec = ctx.reserve()
    if rec is not None:
        for r in rec.get("rpcs", []):
            try:
                ctx.call(r["tool"], r["args"], as_principal=principal)
            except CrucibleError:
                pass
        return ctx.fill(slot, kind, rec)
    log: list = []
    w = worker_fn()
    try:
        res = w.run(job, rpc=_rpc_fn(ctx, principal, log) if world_access else None, timeout=timeout)
    finally:
        if close_after:
            w.close()
    out = {"ok": bool(res.get("ok")), "result": res.get("result"), "error": res.get("error"),
           "stdout": (res.get("stdout") or "")[:4000], "cpu_ms": float(res.get("cpu_ms") or 0.0),
           "rpcs": log}
    return ctx.fill(slot, kind, out)


def run_trusted_in_ctx(ctx: Ctx, kind: str, fn: Callable[..., Any], args: Any, caller: Any, principal: str) -> dict:
    """Same contract for trusted in-process code (built-in incumbents): measured CPU is recorded."""
    import time
    slot, rec = ctx.reserve()
    if rec is not None:
        return ctx.fill(slot, kind, rec)
    t0 = time.process_time()
    try:
        result = fn(args, caller)
        out = {"ok": True, "result": result, "error": None}
    except Exception as e:  # noqa: BLE001 - trusted code can still have bugs on odd inputs
        out = {"ok": False, "result": None, "error": {"type": type(e).__name__, "message": str(e)[:500]}}
    out["cpu_ms"] = round((time.process_time() - t0) * 1000.0, 4)
    out["stdout"] = ""
    out["rpcs"] = []
    return ctx.fill(slot, kind, out)
