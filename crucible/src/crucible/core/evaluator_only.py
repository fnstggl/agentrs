"""Evaluator-only tools, fields and metrics (Clarification 2, decision D-19), enforced by the kernel.

Crucible keeps two sides apart: participants act and observe consequences; the evaluator verifies outcomes. Some
4.0 and 5.0 modules computed evaluator quantities *inside* participant tools - goal progress (`goals.check_now`), the
hidden grader's verified quality (`use.report`, `use.probe`), the related-party verdicts of the integrity analysis
(`independent` flags in `caps.*` and `pkg.*`), the mechanism's creator-defined objective quantity (`score` in
`rig.*`), integrity-netted metrics readable through predicates (`independent_revenue`, `mrr_cents`, ... in self-goals,
charter conditions, commitment and contract deliverables). This module is the one registry of those quantities and
the policy that keeps them on the evaluator side.

Policy (a world-spec parameter, so replay and regeneration see the same rules):

    spec.params["evaluator_access"] = "evaluator"     evaluator data are hidden from participants (observation-only)
    spec.params["evaluator_access"] = "participants"  the 4.0 contract (participant tools may return evaluator data)

A raw `World.create` without the parameter keeps the 4.0 contract, so every 4.0 world family, test and recorded log
is unchanged. Every learner-facing constructor (`crucible.experience.make`, the experience adapters, the Gymnasium
and PettingZoo wrappers, `scale.rollout.WorldEnv`, the HTTP/MCP servers, the AGENTRS adapters) defaults to
``reward_mode="observation_only"`` and builds its world with ``"evaluator"``; only an explicitly selected
``legacy_shaped_reward`` keeps the 4.0 contract (`crucible.experience.modes.evaluator_access`).

What "hidden" means, all enforced in the kernel (`World.execute`, `World.query`, `Ctx.call`, `World.available_tools`,
`World.describe`, `World.observe`, `World.participant_event`):

* an evaluator-only **tool** is not listed to participants, not described by `world.describe`, and refused
  (``forbidden``) when a participant (or a nested call on a participant's behalf) calls it;
* evaluator-only **result fields** of a tool are removed from what a participant receives (top-level and nested
  calls originating from a participant, `exp` measures);
* evaluator-only **event payload fields** are removed from the events a participant receives
  (`World.participant_event`; the adapters and the API server deliver events through it);
* evaluator-only **summary fields** are removed from a participant's module summaries (`World.observe`);
* evaluator-only **metrics** cannot appear in a participant-authored predicate (self-goals, charter conditions,
  commitment milestones, contract deliverables): `goals.metrics.check_predicate_scope` and the self-goal scope check
  refuse them, so no institution or agreement can be turned into a probe of the evaluator.

Modules mark their own quantities with `endpoint(..., evaluator_only=True)` or the `register_*` functions below; the
tables here hold the quantities of modules that predate the flag. Legitimate world observations - revenue, costs,
balances, customers, requests, readings, analytics a participant could compute - are never listed here.
"""
from __future__ import annotations

from typing import Any, Iterable

PARAM = "evaluator_access"
EVALUATOR = "evaluator"          # hidden from participants (the observation-only default of learner-facing worlds)
PARTICIPANTS = "participants"    # the 4.0 contract
POLICIES = (EVALUATOR, PARTICIPANTS)

# ---------------------------------------------------------------------------------------------- the registry
# tool -> why it is evaluator-only
TOOLS: dict[str, str] = {
    "goals.check_now": "evaluates a predicate over authoritative world state on demand: a goal-progress oracle",
    "use.probe": "an acceptance lab scored by the hidden usefulness grader (hidden requests and their truth)",
    "use.report": "the usefulness ledger: hidden-grader verified quality and evaluator-netted independent revenue, "
                  "requests and customers (a service operator's own view is cloud.status/cloud.metrics and its bank "
                  "account)",
}

_CAPS_VERDICTS = ("independent_trials", "independent_trial_ok_rate", "independent_adopters", "independent")
_SCI_VERDICTS = ("related_to_author", "independent", "independent_consistent")
# tool -> keys removed (at any depth) from its result
RESULT_FIELDS: dict[str, tuple[str, ...]] = {
    "caps.search": _CAPS_VERDICTS, "caps.declaration": _CAPS_VERDICTS, "caps.declarations": _CAPS_VERDICTS,
    "caps.verify": ("independent",),
    "rig.read": ("score",), "rig.history": ("score",),
    # the science world (domains/science.py, module `sci`): a replication report's verdict "consistent/refuted" is a
    # statistical test on the reporter's own readings (a world observation), but `related_to_author` and
    # `independent` are the evaluator's relatedness analysis (`goals.metrics.is_related`) - a participant filing
    # replications from accounts it controls would otherwise learn which of them the evaluator links to it
    "sci.replicate": _SCI_VERDICTS, "sci.paper": _SCI_VERDICTS, "sci.papers": _SCI_VERDICTS,
    "sci.replications": _SCI_VERDICTS,
}
# event type -> payload keys removed (at any depth)
EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "caps.tried": ("independent",), "caps.adopted": ("independent",), "caps.verification": ("independent",),
    "pkg.executed": ("independent",), "pkg.review": ("independent",),
    "rig.report": ("score",),
    "use.probe": ("mean",),
    "sci.replication": ("related_to_author", "independent"),
}
# module name -> keys removed (at any depth) from that module's participant summary (`Module.observe`)
SUMMARY_FIELDS: dict[str, tuple[str, ...]] = {
    "rig": ("score",),
}
# metrics whose value embeds evaluator analysis (related-party / circular-flow netting, the hidden grader, the
# mechanism's objective quantity, evaluator baselines): never readable through a participant-authored predicate
METRICS: dict[str, str] = {
    "independent_revenue": "related-party and circular-flow netting over the whole payment graph",
    "independent_members": "relatedness verdicts over the membership graph",
    "service_requests_ok": "requests from callers the evaluator judges unrelated to the operator",
    "service_availability": "availability measured on callers the evaluator judges unrelated to the operator",
    "quality_margin_vs_incumbents": "hidden-grader quality",
    "revenue_from": "independent (netted) revenue from one payer",
    "capability_adopters": "independent adopters (relatedness verdicts)",
    "search_best_rank": "rank over results owned by the principal or related parties (relatedness verdicts)",
    "search_visibility": "derived from search_best_rank",
    "mrr_cents": "qualifying MRR: integrity flags, related-party and circular-flow analysis",
    "arr_cents": "12 x mrr_cents",
    "active_subscriptions": "qualifying subscriptions (integrity flags)",
    "independent_customers": "relatedness verdicts over the customers",
    "churn_rate": "computed over independent customers",
    "profit_cents": "independent revenue net of related flows",
    "one_time_revenue_cents": "revenue from unflagged payers (integrity flags)",
    "use_verified_quality": "hidden-grader quality",
    "use_independent_requests": "relatedness verdicts",
    "use_customer_value": "hidden-grader quality delivered",
    "use_independent_revenue": "relatedness verdicts",
    "use_retained_customers": "computed over independent customers",
    "rig_score": "the mechanism's creator-defined objective quantity",
    "lab_mean_delay": "exact mean delay of the hidden traffic process",
    "lab_dropped_fraction": "exact dropped fraction of the hidden traffic process",
    "lab_improvement_vs_baseline": "evaluator counterfactual baseline",
}


def register_tool(name: str, why: str) -> None:
    TOOLS[str(name)] = str(why)


def register_result_fields(tool: str, fields: Iterable[str]) -> None:
    RESULT_FIELDS[str(tool)] = tuple(dict.fromkeys(list(RESULT_FIELDS.get(tool, ())) + [str(f) for f in fields]))


def register_event_fields(event_type: str, fields: Iterable[str]) -> None:
    EVENT_FIELDS[str(event_type)] = tuple(dict.fromkeys(list(EVENT_FIELDS.get(event_type, ())) +
                                                        [str(f) for f in fields]))


def register_summary_fields(module: str, fields: Iterable[str]) -> None:
    SUMMARY_FIELDS[str(module)] = tuple(dict.fromkeys(list(SUMMARY_FIELDS.get(module, ())) + [str(f) for f in fields]))


def register_metric(name: str, why: str) -> None:
    METRICS[str(name)] = str(why)


# ---------------------------------------------------------------------------------------------- policy
def policy(world) -> str:
    """The world's evaluator-access policy: a runtime override (an adapter adopting an existing world) or the spec
    parameter; the 4.0 contract when neither is set."""
    p = world.runtime.get(PARAM) if hasattr(world, "runtime") else None
    if p is None:
        p = world.spec.params.get(PARAM, PARTICIPANTS)
    return p if p in POLICIES else EVALUATOR


def hidden(world) -> bool:
    """True when evaluator-only quantities are kept from participants in this world."""
    return policy(world) == EVALUATOR


def set_policy(spec_or_world, value: str) -> None:
    """Set the policy on a `WorldSpec` (before `World.create`: recorded in the spec, replays identically) or, for an
    existing `World`, as a runtime override of this process (documented: a regenerated copy reads its spec)."""
    if value not in POLICIES:
        raise ValueError(f"evaluator_access must be one of {POLICIES}, got {value!r}")
    if hasattr(spec_or_world, "runtime") and hasattr(spec_or_world, "state"):
        spec_or_world.runtime[PARAM] = value
    else:
        spec_or_world.params[PARAM] = value


def is_evaluator_tool(world, name: str, endpoint=None) -> bool:
    if name in TOOLS:
        return True
    return bool(getattr(endpoint, "evaluator_only", False))


def participant_tool_allowed(world, principal: str, name: str, endpoint=None) -> bool:
    from .kernel import SYSTEM
    return principal == SYSTEM or not hidden(world) or not is_evaluator_tool(world, name, endpoint)


# ---------------------------------------------------------------------------------------------- filters
def _strip(x: Any, keys: frozenset, depth: int = 0) -> Any:
    if depth > 24:
        return x
    if isinstance(x, dict):
        return {k: _strip(v, keys, depth + 1) for k, v in x.items() if k not in keys}
    if isinstance(x, list):
        return [_strip(v, keys, depth + 1) for v in x]
    return x


def participant_result(world, tool: str, result: Any) -> Any:
    """`result` of `tool` as a participant may receive it (a copy without evaluator-only fields when hidden)."""
    keys = RESULT_FIELDS.get(tool)
    if not keys or not hidden(world) or result is None:
        return result
    return _strip(result, frozenset(keys))


def participant_payload(world, event_type: str, payload: Any) -> Any:
    keys = EVENT_FIELDS.get(event_type)
    if not keys or not hidden(world) or not isinstance(payload, dict):
        return payload
    return _strip(payload, frozenset(keys))


def participant_summary(world, module: str, summary: Any) -> Any:
    keys = SUMMARY_FIELDS.get(module)
    if not keys or not hidden(world) or summary is None:
        return summary
    return _strip(summary, frozenset(keys))


def evaluator_metrics() -> dict[str, str]:
    """Every evaluator-only metric: this registry plus the lists other modules keep of their own evaluator
    measurements (`goals.metrics.EVALUATOR_ONLY_METRICS`, the commerce workstream's `commerce.observables.EVALUATOR_ONLY`
    and the recurring-revenue metrics behind it). One union, so a module that marks a metric on its side is never
    readable through a participant-authored predicate in an observation-only world because this table missed it."""
    out = dict(METRICS)
    for modname, attr, why in (("crucible.goals.metrics", "EVALUATOR_ONLY_METRICS", "an evaluator measurement "
                                "(goals.metrics.EVALUATOR_ONLY_METRICS)"),
                               ("crucible.commerce.observables", "EVALUATOR_ONLY", "computed with hidden evaluator "
                                "analysis (commerce.observables.EVALUATOR_ONLY)")):
        try:
            import importlib
            names = getattr(importlib.import_module(modname), attr, ())
        except ImportError:
            continue
        for n in names:
            out.setdefault(str(n), why)
    return out


def metric_errors(world, predicate_leaves: Iterable[dict]) -> list[str]:
    """Errors for the evaluator-only metrics among a participant-authored predicate's leaves (none when the world
    keeps the 4.0 contract)."""
    if not hidden(world):
        return []
    out = []
    ev = evaluator_metrics()
    for lf in predicate_leaves:
        m = lf.get("metric") if isinstance(lf, dict) else None
        if m in ev:
            out.append(f"metric {m!r} is evaluator-only ({ev[m]}); participants cannot read it")
    return out


def describe() -> dict:
    """The registry, for documentation and the leakage probes."""
    return {"param": PARAM, "policies": list(POLICIES), "tools": dict(sorted(TOOLS.items())),
            "result_fields": {k: list(v) for k, v in sorted(RESULT_FIELDS.items())},
            "event_fields": {k: list(v) for k, v in sorted(EVENT_FIELDS.items())},
            "summary_fields": {k: list(v) for k, v in sorted(SUMMARY_FIELDS.items())},
            "metrics": dict(sorted(evaluator_metrics().items()))}
