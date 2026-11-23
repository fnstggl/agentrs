"""Error taxonomy. Every rejected action carries one of these codes so that learners can condition on *why*
an action failed (a schema error is a different signal from insufficient funds)."""
from __future__ import annotations


class CrucibleError(Exception):
    code = "error"

    def __init__(self, message: str = "", **details):
        super().__init__(message)
        self.message = message
        self.persist = details.pop("persist", None) or []   # events that must survive the rollback (e.g. a loss)
        self.details = details

    def as_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.details:
            d["details"] = self.details
        return d


class InvalidAction(CrucibleError):
    code = "invalid_action"


class UnknownTool(CrucibleError):
    code = "unknown_tool"


class Unauthenticated(CrucibleError):
    code = "unauthenticated"


class Forbidden(CrucibleError):
    code = "forbidden"


class NotFound(CrucibleError):
    code = "not_found"


class Conflict(CrucibleError):
    code = "conflict"


class InsufficientFunds(CrucibleError):
    code = "insufficient_funds"


class CapacityExceeded(CrucibleError):
    code = "capacity_exceeded"


class RateLimited(CrucibleError):
    code = "rate_limited"


class ServiceUnavailable(CrucibleError):
    code = "service_unavailable"


class Timeout(CrucibleError):
    code = "timeout"


class SandboxError(CrucibleError):
    code = "sandbox_error"


class BudgetExhausted(CrucibleError):
    code = "budget_exhausted"


def contain(ctx, fn, *args, **kw):
    """Run `fn(*args, **kw)` for ONE record of a batch process (one order of a settlement sweep, one company of the
    daily company run) inside a savepoint. If it raises anything - an in-world refusal or a bug triggered by a
    crafted record - its partial effects (state, emitted events, sequence numbers) are rolled back and the
    exception is returned, so that one bad record cannot stop the process for every other participant. Returns
    None on success. External outcomes already recorded stay recorded (they happened, and regeneration must see
    them in the same order). Callers record the failure in an event of their own."""
    state, world = ctx.state, ctx.world
    mark, n_ev, seq0 = state.savepoint(), len(ctx.events), world.seq
    try:
        fn(*args, **kw)
        return None
    except Exception as e:  # noqa: BLE001 - containment is the point; the caller records what happened
        state.rollback_to(mark)
        del ctx.events[n_ev:]
        world.seq = seq0
        return e


class InvariantViolation(Exception):
    """A kernel invariant was broken. This is a simulator bug, never an agent-facing outcome."""


class ProtocolError(Exception):
    """Misuse of the kernel API by research code (not by an in-world agent)."""
