"""Error taxonomy. Every rejected action carries one of these codes so that learners can condition on *why*
an action failed (a schema error is a different signal from insufficient funds)."""
from __future__ import annotations


class CrucibleError(Exception):
    code = "error"

    def __init__(self, message: str = "", **details):
        super().__init__(message)
        self.message = message
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


class InvariantViolation(Exception):
    """A kernel invariant was broken. This is a simulator bug, never an agent-facing outcome."""


class ProtocolError(Exception):
    """Misuse of the kernel API by research code (not by an in-world agent)."""
