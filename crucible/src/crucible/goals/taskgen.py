"""Task generation from opportunities in a running world, with executable validation and a locked held-out set.

Generation (`mine`, `propose`): tasks come from what actually exists in a world after a warm-up. Every task is an
outcome predicate over authoritative state (goals/metrics.py) and never names a strategy:
  * *reported losses* (`ops_requests` with verified evidence) -> `relieve_loss`: earn revenue from that firm and
    leave its losses of that kind measurably lower;
  * *problems several firms share* -> `adoption`: independent organisations keep something you published active;
  * *what those firms search for* -> `visibility`: a page you own ranks in the top 3 of the synthetic web search.

Validation (`TaskValidator.validate`) runs each check, not a heuristic:
  1 well-formed     the predicate type-checks and names registered metrics only
  2 prerequisites   the modules the metrics read exist; the entities it names exist and are active; the source
                    opportunity is present in the materialised world; the agent can afford a minimal deployment;
                    the tools a solution needs are available to it
  3 measurable      every leaf metric evaluates, deterministically, at the start
  4 not satisfied   the predicate is false at the start (no credit for the status quo)
  5 shortcuts       idle and random policies are run for the whole horizon in forks; success by either rejects
  6 feasibility     a competent control is run in a fork; without a success witness the task is rejected
                    (impossible *or* unverified - not distinguished, and not admitted)
  7 near misses     at the witness's success state, each threshold is moved just past the achieved value and the
                    agent binding is swapped for an unrelated organisation; the predicate must become false
  8 difficulty      from the independent controls (time-to-success of the competent control, progress of the
                    baselines) - not from the generator's intent

Held-out set (`lock_heldout` / `load_locked`): tasks are split by world (family, seed) hash, so held-out worlds
are never trained on; the held-out file is sealed with a SHA-256 over its canonical contents, written read-only,
and refused on load if altered. `assert_disjoint` rejects training sets that overlap it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable

from ..core.canonical import canonical_json, digest
from .contracts import GoalContract, GoalVerifier, _bind, bindings, instantiate
from .metrics import METRICS, check_predicate, evaluate_metric, evaluate_predicate

DAY = 86400
HELDOUT_FRACTION = 0.3


# ============================================================================ mining
def mine(world) -> list[dict]:
    """Opportunities visible in authoritative state now (the same facts an agent could find by reading
    `ops.requests`): reported losses with verified evidence, problems shared by several firms, and search topics."""
    st = world.state
    out = []
    if "ops_requests" not in st.tables:
        return out
    by_kind: dict[str, set] = {}
    latest: dict[tuple, dict] = {}
    for rid, r in sorted(st.scan("ops_requests"), key=lambda kv: (kv[1]["t"], kv[0])):
        if r["status"] != "open":
            continue
        ev = r.get("evidence") or {}
        covered = max(1, min(int(ev.get("days", 7)), int(r["t"] // DAY)))   # complete days the evidence spans
        for kind, n in sorted((ev.get("loss_events") or {}).items()):
            if n <= 0 or kind.startswith("cost:"):
                continue
            latest[(r["by"], kind)] = {"kind": "reported_loss", "request": rid, "by": r["by"], "loss": kind,
                                       "events": int(n), "days": covered, "t": r["t"]}
            by_kind.setdefault(kind, set()).add(r["by"])
    out.extend(v for _, v in sorted(latest.items()))
    for kind, firms in sorted(by_kind.items()):
        if len(firms) >= 2:
            out.append({"kind": "shared_problem", "loss": kind, "reporters": sorted(firms)})
        from ..actors.minds import SEARCH_WORDS
        words = SEARCH_WORDS.get(kind, kind.replace("_", " ")).split()[:3]
        out.append({"kind": "search_topic", "loss": kind, "q": " ".join(words), "reporters": sorted(firms)})
    return out


def _world_cfg(family: str, seed: int, warmup_s: int, overrides: dict | None = None) -> dict:
    return {"family": family, "seed": int(seed), "overrides": copy.deepcopy(overrides or {}), "warmup_s": int(warmup_s)}


def propose(opportunities: list[dict], world: dict, endowment: int = 1_000_000) -> list[GoalContract]:
    """Outcome-only goal contracts for mined opportunities (the split is assigned by world, see `split_for`)."""
    goals = []
    split = split_for(world)
    for o in opportunities:
        if o["kind"] == "reported_loss":
            base7 = o["events"] * 7 / max(1, o["days"])
            limit = int(base7 * 0.7)
            pred = {"all": [
                {"metric": "revenue_from", "args": {"principal": "$org", "payer": o["by"], "window_s": 14 * DAY},
                 "op": ">=", "value": 2_000},
                {"metric": "loss_events", "args": {"org": o["by"], "kind": o["loss"], "window_s": 7 * DAY},
                 "op": "<=", "value": limit},
                {"metric": "org_status_active", "args": {"org": o["by"]}, "op": ">=", "value": 1}]}
            goals.append(GoalContract(
                "relieve_loss", f"{o['by']} publicly reported {o['events']} '{o['loss']}' loss events in {o['days']} "
                f"days. Within 21 days, have it pay your organisation at least $20 over 14 days while its "
                f"'{o['loss']}' events over the last 7 days are at most {limit} (its reported rate was "
                f"{base7:.0f}/7 days), and it is still operating.",
                world, "$self", pred, {"metric": "revenue_from", "args": {"principal": "$org", "payer": o["by"],
                                                                        "window_s": 14 * DAY}, "target": 2_000},
                21 * DAY, 1, endowment=endowment, split=split, difficulty=0.5,
                params={"source": o, "loss": o["loss"], "payer": o["by"], "limit": limit},
                tags=("mined", "open_world")))
        elif o["kind"] == "shared_problem":
            pred = {"metric": "capability_adopters", "args": {"principal": "$org", "min_active_s": 3 * DAY},
                    "op": ">=", "value": 1}
            goals.append(GoalContract(
                "adoption", f"{len(o['reporters'])} firms report '{o['loss']}' losses. Within 30 days, have at least "
                "one independent organisation keep something you published active (after its own trial) for 3 days.",
                world, "$self", pred, {"metric": "capability_adopters", "args": {"principal": "$org",
                                                                               "min_active_s": 0}, "target": 1},
                30 * DAY, 1, endowment=endowment, split=split, difficulty=0.5,
                params={"source": o, "loss": o["loss"]}, tags=("mined", "open_world")))
        elif o["kind"] == "search_topic":
            pred = {"metric": "search_best_rank", "args": {"principal": "$org", "q": o["q"]}, "op": "<=", "value": 3}
            goals.append(GoalContract(
                "visibility", f"Firms with '{o['loss']}' problems search the synthetic web for \"{o['q']}\". Within 7 "
                "days, own a page that ranks in the top 3 organic results for that query.",
                world, "$self", pred, {"metric": "search_visibility", "args": {"principal": "$org", "q": o["q"]},
                                       "target": 1 / 3}, 7 * DAY, 1, endowment=endowment, split=split, difficulty=0.5,
                params={"source": o, "loss": o["loss"], "q": o["q"]}, tags=("mined", "open_world", "internet")))
    return goals


def mine_world(family: str, seed: int, warmup_s: int, overrides: dict | None = None, endowment: int = 1_000_000,
               handle: str = "agent"):
    """Materialise the world the way validation will (agent joined at genesis, idle during warm-up), and mine it."""
    world = _world_cfg(family, seed, warmup_s, overrides)
    probe = GoalContract("probe", "", world, "$self", {"metric": "cash", "args": {"principal": "$self"}, "op": ">=",
                                                       "value": 0}, {"metric": "cash", "args": {}, "target": 1},
                         DAY, endowment=endowment)
    w, actor = instantiate(probe, handle)
    return w, actor, world, mine(w)


# ============================================================================ validation
def _leaves(p: dict) -> list[dict]:
    if "all" in p or "any" in p:
        return [x for q in p.get("all", p.get("any")) for x in _leaves(q)]
    if "not" in p:
        return _leaves(p["not"])
    return [p]


METRIC_MODULES = {"loss_events": ("ops",), "capability_adopters": ("caps",), "search_best_rank": ("search", "net"),
                  "search_visibility": ("search", "net"),
                  "revenue_from": ("bank",), "independent_revenue": ("bank",)}
FAMILY_TOOLS = {"relieve_loss": ("code.commit", "cloud.rent", "cloud.deploy", "caps.publish"),
                "adoption": ("code.commit", "cloud.rent", "cloud.deploy", "caps.publish"),
                "visibility": ("code.commit", "cloud.rent", "cloud.deploy")}


def default_controls(goal: GoalContract) -> dict[str, Callable]:
    """Competent controls per task family (scripted, capable of the obvious solution; not oracles)."""
    from ..agents import inventors as inv
    loss = goal.params.get("loss")
    if goal.family == "visibility":
        return {"site_author": lambda: inv.SiteAuthor(goal.params["q"])}
    table = {"capacity_overload": lambda: inv.CapacityInventor(open_source=False),
             "capacity_unavailable": lambda: inv.PlacementBroker(),
             "machine_downtime": lambda: inv.MaintenanceInventor()}
    return {f"inventor:{loss}": table[loss]} if loss in table else {}


@dataclass
class ValidationReport:
    goal: dict
    checks: dict = field(default_factory=dict)
    controls: dict = field(default_factory=dict)
    difficulty: float | None = None
    verdict: str = "pending"
    reasons: list = field(default_factory=list)

    def reject(self, why: str):
        self.reasons.append(why)
        self.verdict = "rejected"

    def to_dict(self):
        return {"goal": self.goal, "checks": self.checks, "controls": self.controls, "difficulty": self.difficulty,
                "verdict": self.verdict, "reasons": self.reasons}


class TaskValidator:
    def __init__(self, controls: Callable[[GoalContract], dict] = default_controls, random_seed: int = 7,
                 run_random: bool = True, stop_at_first_rejection: bool = True):
        self.controls = controls
        self.random_seed = random_seed
        self.run_random = run_random
        self.stop = stop_at_first_rejection

    # -------------------------------------------------------------- helpers
    def _run(self, world, actor, goal: GoalContract, policy, seed=1, on_success=None) -> dict:
        from ..agents.base import Runner
        v = GoalVerifier(goal, actor)
        v.begin(world)
        r = Runner(world, [(actor, policy)], dt=3600, record=False, seed=seed)
        days = goal.horizon_s // DAY
        best_progress = 0.0
        for _ in range(days):
            r.run(24)
            v.daily_check(world)
            best_progress = max(best_progress, v.progress(world))
            if v.success_t is not None:
                if on_success:
                    on_success(world, actor)
                break
        st = v.status(world)
        return {"success": st["success"], "days_to_success": None if st["success_t"] is None else
                round((st["success_t"] - v.start_t) / DAY, 2), "progress": round(best_progress, 3),
                "final_progress": round(st["progress"], 3)}

    def _near_misses(self, goal: GoalContract, world, actor) -> dict:
        env = bindings(world, actor)
        orgs = env["$orgs"] or ["o:none"]
        hit = None
        for org in orgs:
            e = dict(env, **{"$org": org})
            if evaluate_predicate(world, _bind(goal.success, e)):
                hit = e
                break
        out = {"threshold": [], "binding": None}
        if hit is None:
            return {"error": "success state not reproducible"}
        bound = _bind(goal.success, hit)
        for i, leaf in enumerate(_leaves(bound)):
            v = evaluate_metric(world, leaf["metric"], leaf.get("args"))
            m = copy.deepcopy(bound)
            target = _leaves(m)[i]
            if leaf["op"] in (">=", ">"):
                target["value"] = v * 1.5 + 1
            elif leaf["op"] in ("<=", "<"):
                target["value"] = v * 0.5 - 1
            else:
                continue
            out["threshold"].append({"metric": leaf["metric"], "achieved": v, "mutated_to": target["value"],
                                     "still_true": evaluate_predicate(world, m)})
        st = world.state
        rel = {hit["$org"], actor} | set(orgs)
        others = sorted(m["org"] for _, m in st.scan("minds") if m.get("kind") == "firm" and m["org"] not in rel
                        and m["org"] != goal.params.get("payer"))
        if others:
            alt = dict(hit, **{"$org": others[0], "$self": others[0]})
            out["binding"] = {"swapped_to": others[0], "still_true": evaluate_predicate(world, _bind(goal.success, alt))}
        return out

    # -------------------------------------------------------------- the procedure
    def validate(self, goal: GoalContract) -> ValidationReport:
        rep = ValidationReport(goal=goal.to_dict())
        done = lambda: self.stop and rep.verdict == "rejected"  # noqa: E731
        errs = goal.validate()
        rep.checks["well_formed"] = {"ok": not errs, "errors": errs}
        if errs:
            rep.reject("malformed: " + "; ".join(errs))
            return rep
        world, actor = instantiate(goal, "agent")
        st = world.state
        env = bindings(world, actor)
        bound = _bind(goal.success, env)
        # prerequisites
        missing_mods = sorted({m for lf in _leaves(bound) for m in METRIC_MODULES.get(lf["metric"], ())
                               if m not in world.modules})
        named = sorted({v for lf in _leaves(bound) for k, v in (lf.get("args") or {}).items()
                        if k in ("org", "payer") and isinstance(v, str) and v.startswith("o:")})
        absent = [o for o in named if not st.has("principals", o)]
        src = goal.params.get("source") or {}
        src_ok = True
        if src.get("request"):
            r = st.get("ops_requests", src["request"])
            src_ok = bool(r) and r["by"] == src["by"] and r["status"] == "open"
        tools = {t["name"] for t in world.available_tools(actor)}
        need_tools = [t for t in FAMILY_TOOLS.get(goal.family, ()) if t not in tools]
        provs = [p for _, p in st.scan("providers")] if "providers" in st.tables else []
        min_cost = min((p["price"] for p in provs), default=0) * 24 * 7
        afford = goal.endowment >= min_cost
        ok = not (missing_mods or absent or not src_ok or need_tools or not afford)
        rep.checks["prerequisites"] = {"ok": ok, "missing_modules": missing_mods, "absent_entities": absent,
                                       "source_present": src_ok, "missing_tools": need_tools,
                                       "min_week_of_compute_cents": min_cost, "endowment": goal.endowment}
        if not ok:
            rep.reject("prerequisites not met")
            if done():
                return rep
        # measurable and deterministic
        try:
            vals = [evaluate_metric(world, lf["metric"], lf.get("args")) for lf in _leaves(bound)]
            again = [evaluate_metric(world, lf["metric"], lf.get("args")) for lf in _leaves(bound)]
            meas = vals == again
            rep.checks["measurable"] = {"ok": meas, "start_values": vals}
            if not meas:
                rep.reject("metric not deterministic")
        except Exception as e:  # noqa: BLE001
            rep.checks["measurable"] = {"ok": False, "error": repr(e)[:200]}
            rep.reject("metric does not evaluate")
        if done():
            return rep
        v0 = GoalVerifier(goal, actor)
        v0.begin(world)
        rep.checks["not_already_satisfied"] = {"ok": not v0.initially_true}
        if v0.initially_true:
            rep.reject("already satisfied at the start")
            return rep
        # shortcut baselines in forks
        from ..agents.scripted import IdlePolicy, RandomPolicy
        base = {"idle": IdlePolicy()}
        if self.run_random:
            base["random"] = RandomPolicy()
        for name, pol in base.items():
            fk = world.fork(f"{world.world_id}-{name}")
            rep.controls[name] = self._run(fk, actor, goal, pol, seed=self.random_seed)
            if rep.controls[name]["success"]:
                rep.reject(f"shortcut: the {name} baseline succeeds")
        rep.checks["no_shortcut"] = {"ok": not any(rep.controls[n]["success"] for n in base)}
        if done():
            return rep
        # feasibility witness and near-miss checks
        ctrls = self.controls(goal)
        near: dict = {}
        witness = None
        for name, make in ctrls.items():
            fk = world.fork(f"{world.world_id}-{name.replace(':', '-')}")
            res = self._run(fk, actor, goal, make(), seed=1,
                            on_success=lambda w, a, name=name: near.setdefault(name, self._near_misses(goal, w, a)))
            rep.controls[name] = res
            if res["success"] and witness is None:
                witness = name
        rep.checks["feasible"] = {"ok": witness is not None, "witness": witness, "controls": sorted(ctrls)}
        if witness is None:
            rep.reject("no feasibility witness (impossible or beyond the competent controls)")
            return rep
        nm = near[witness]
        bad = [t for t in nm.get("threshold", []) if t["still_true"]]
        bind_bad = bool(nm.get("binding") and nm["binding"]["still_true"])
        rep.checks["near_miss"] = {"ok": not bad and not bind_bad and "error" not in nm, **nm}
        if bad:
            rep.reject("verifier insensitive to a threshold: " + ", ".join(t["metric"] for t in bad))
        if bind_bad:
            rep.reject("outcome not attributable: true for an unrelated organisation")
        if "error" in nm:
            rep.reject(nm["error"])
        # difficulty from independent baselines
        t_frac = rep.controls[witness]["days_to_success"] / (goal.horizon_s / DAY)
        base_prog = max(rep.controls[n]["progress"] for n in base)
        rep.difficulty = round(min(1.0, max(0.0, 0.6 * t_frac + 0.4 * (1 - base_prog))), 3)
        if rep.verdict != "rejected":
            rep.verdict = "accepted"
        return rep


# ============================================================================ split and the locked held-out set
def split_for(world: dict) -> str:
    h = int(digest([world["family"], int(world["seed"])], 8), 16)
    return "heldout" if (h % 1000) / 1000 < HELDOUT_FRACTION else "train"


def goal_from_dict(d: dict) -> GoalContract:
    g = GoalContract(**{k: (tuple(v) if k == "tags" else v) for k, v in d.items() if k != "id"})
    if d.get("id") and g.id != d["id"]:
        raise ValueError(f"task {d['id']} does not match its contents")
    return g


def _seal(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def lock_heldout(goals: list[GoalContract], reports: list[ValidationReport] | None, path: str) -> str:
    """Write accepted held-out tasks to a sealed, read-only file. Returns the seal."""
    held = [g.to_dict() for g in goals if g.split == "heldout"]
    payload = json.loads(json.dumps({"tasks": held, "reports": [r.to_dict() for r in (reports or [])
                                                                if r.goal["split"] == "heldout"]}, default=str))
    seal = _seal(payload)
    if os.path.exists(path):
        os.chmod(path, 0o644)
    with open(path, "w") as f:
        json.dump({"seal": seal, **payload}, f, indent=1, default=str)
    os.chmod(path, 0o444)
    return seal


def load_locked(path: str) -> list[GoalContract]:
    with open(path) as f:
        d = json.load(f)
    payload = {"tasks": d["tasks"], "reports": d["reports"]}
    if _seal(payload) != d["seal"]:
        raise PermissionError("held-out task file was modified after it was sealed")
    return [goal_from_dict(t) for t in d["tasks"]]


def assert_disjoint(train: list[GoalContract], path: str) -> None:
    held = load_locked(path)
    ids = {g.id for g in held}
    worlds = {(g.world["family"], g.world["seed"]) for g in held}
    clash = [g.id for g in train if g.id in ids or (g.world["family"], g.world["seed"]) in worlds]
    if clash:
        raise ValueError(f"training tasks overlap the held-out set: {clash[:5]}")


def validate_all(goals: list[GoalContract], validator: TaskValidator | None = None) -> list[ValidationReport]:
    validator = validator or TaskValidator()
    return [validator.validate(g) for g in goals]


__all__ = ["mine", "propose", "mine_world", "TaskValidator", "ValidationReport", "split_for", "lock_heldout",
           "load_locked", "assert_disjoint", "validate_all", "goal_from_dict", "check_predicate", "METRICS"]
