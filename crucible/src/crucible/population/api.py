"""The population's public API beyond activation (ONE_WORLD_CONTRACTS section 1): interests in the shared space,
money, decision recording with provenance, learning from experience, and the evaluator-side provenance check.

Decision provenance. `record_decisions` is the only way a subsystem records individual decisions: one
`columns.batch` whose `extra.decision` names the decision model (id, version, fidelity-register id) and the deciding
actor and tool. The first use of a model in a world registers it (`pop.decision_model`, table
`pop_decision_models`). `check_provenance(world)` (evaluator side) then verifies every decision batch against the
transaction that produced it, which the kernel log records independently of the batch: a population decision must
come from the population's own activation process (`pop.tick`) and concern only individuals activated in that
transaction (the `pop.session` cohort), or from a tool command of the individual itself (a participant's own
decision, model `participant`). A batch from module code, a system call, genesis or any other process is counted as
forced, whatever its `extra` claims.
"""
from __future__ import annotations

import numpy as np

from ..core.errors import InvalidAction, NotFound, ProtocolError
from ..core.kernel import SYSTEM, Ctx
from ..substrate import artifacts, columns, ids
from . import semantics
from .schema import TABLE

PARTICIPANT_MODEL = "participant"
DECISION_KINDS_PREFIX = ("impress.", "consume.")


# ============================================================================================ interests
def interest_dim(world) -> int:
    return int(columns.col(world, TABLE, "interests").shape[1])


def interest_coords(world, idx) -> np.ndarray:
    """(n, K) float32 unit coordinates of individuals' interests in the topic subspace of the space's topic block."""
    ix = np.asarray(idx, np.int64)
    return columns.col(world, TABLE, "interests")[ix].astype(np.float32)


def interests(world, idx) -> np.ndarray:
    """(n, semantic.space.DIM) float32 unit interest vectors in the shared space (contract section 1, D-16); zero
    outside the space's topic block (population.semantics)."""
    return semantics.lift(interest_coords(world, idx))


def to_coords(world, vecs: np.ndarray) -> np.ndarray:
    """Space vectors (n, DIM) -> the population's interest coordinates of their topic blocks (n, K); (n, K) input is
    returned as is."""
    v = np.asarray(vecs, np.float32)
    k = interest_dim(world)
    if v.ndim == 1:
        v = v[None, :]
    if v.shape[1] == semantics.DIM and k != semantics.DIM:
        return semantics.project(v, k)
    if v.shape[1] != k:
        raise InvalidAction(f"item vectors must have {semantics.DIM} (space) or {k} (interest coordinates) columns")
    return v


# ============================================================================================ decisions
def _model_key(model_id: str, model_version: str) -> str:
    return f"{model_id}@{model_version}"


def ensure_model(ctx: Ctx, model_id: str, model_version: str, fidelity_id: str, *, kind: str = "") -> str:
    """Register a decision model in this world once (`pop.decision_model`)."""
    if not isinstance(model_id, str) or not model_id or not isinstance(model_version, str) or not model_version:
        raise ProtocolError("decision provenance needs a model_id and a model_version")
    if not isinstance(fidelity_id, str) or not fidelity_id:
        raise ProtocolError("decision provenance needs a fidelity_id (FIDELITY_REGISTER row)")
    if "pop" not in ctx.world.modules:
        raise ProtocolError("record_decisions needs the population module 'pop'")
    key = _model_key(model_id, model_version)
    rec = ctx.state.get("pop_decision_models", key)
    if rec is None:
        ctx.emit("pop.decision_model", {"key": key, "model_id": model_id, "model_version": model_version,
                                        "fidelity_id": fidelity_id, "kind": kind}, vis="internal")
    elif rec["fidelity_id"] != fidelity_id:
        raise ProtocolError(f"decision model {key} is registered with fidelity id {rec['fidelity_id']}")
    return key


def record_decisions(ctx: Ctx, kind: str, idx, values: dict, *, model_id: str, model_version: str, fidelity_id: str,
                     mode: str = "set", table: str | None = None, extra: dict | None = None) -> str:
    """One recorded cohort decision with decision provenance (contract section 1). `table` defaults to the
    population table; other column tables (relation tables of the social modules) carry the same provenance."""
    ensure_model(ctx, model_id, model_version, fidelity_id, kind=kind)
    ex = dict(extra or {})
    if "decision" in ex:
        raise InvalidAction("extra.decision is reserved for decision provenance")
    ex["decision"] = {"model_id": model_id, "model_version": model_version, "fidelity_id": fidelity_id,
                      "by": ctx.actor, "tool": ctx.tool}
    ref = columns.batch(ctx, table or TABLE, kind, idx, values, mode=mode, extra=ex)
    active = ctx.world.runtime.get("pop.tick.decisions")
    if active is not None:
        active.append(ref)                 # listed by the population's session batch (provenance)
    return ref


def participant_decision(ctx: Ctx) -> dict:
    """The provenance of a decision an individual takes itself through a tool (`feed.watch`, `content.open`)."""
    i = ids.index(ctx.actor)
    if i is None:
        raise InvalidAction("only population individuals decide as participants")
    return {"model_id": PARTICIPANT_MODEL, "model_version": "1", "fidelity_id": "participant-own-decision"}


# ============================================================================================ money
def pay(ctx: Ctx, i: int, to_account: str, cents: int, *, memo: str) -> str:
    """Individual `i` pays `cents` from its own bank account (materialised, accrued income deposited) to
    `to_account`; returns the bank transaction id. Refused when it cannot afford it."""
    from .core import money_deposit
    cents = int(cents)
    if cents <= 0:
        raise InvalidAction("a payment must be positive")
    bank = ctx.world.modules.get("bank")
    if bank is None:
        raise ProtocolError("payments need the bank module")
    acct = money_deposit(ctx, int(i))
    if bank.balance(ctx.state, acct) < cents:
        raise InvalidAction(f"individual {int(i)} cannot afford {cents} cents")
    if not ctx.state.has("accounts", to_account):
        raise NotFound(f"no account {to_account}")
    return bank.move(ctx, acct, to_account, cents, "purchase", str(memo)[:200])


# ============================================================================================ experience
EXPERIENCE_MODEL = ("pop.experience", "1", "pop-experience-drift")


def experience(ctx: Ctx, idx, activity: str, utility, *, item_vecs=None, satisfaction=None) -> str | None:
    """Update individuals from realised outcomes of `activity`: interests drift towards (utility > 0) or away from
    (utility < 0) the consumed items' representations, knowledge accumulates in their direction; with
    `satisfaction` (a delayed outcome outside a session, e.g. using a product bought days ago) satisfaction and the
    activity's habit also learn from it. Inside a session the population's tick already learns satisfaction and
    habits from the utility the activity returns, so handlers pass only `item_vecs` there. One recorded batch."""
    from . import core
    w = ctx.world
    ix = np.asarray(idx, np.int64).reshape(-1)
    if ix.size == 0:
        return None
    u = np.asarray(utility, np.float64).reshape(-1)
    if u.shape[0] != ix.size or not np.all(np.isfinite(u)):
        raise InvalidAction("experience needs one finite utility per individual")
    vals: dict[str, np.ndarray] = {}
    if item_vecs is not None:
        coords = to_coords(w, item_vecs)
        if coords.shape[0] != ix.size:
            raise InvalidAction("experience needs one item vector per individual")
        c = columns.col(w, TABLE, "interests")[ix].astype(np.float32)
        prefs = columns.col(w, TABLE, "prefs")[ix].astype(np.float32)
        nrm = np.linalg.norm(coords, axis=1, keepdims=True)
        has = nrm[:, 0] > 1e-6
        d = np.where(has[:, None], coords / np.maximum(nrm, 1e-6), 0.0).astype(np.float32)
        eta = (0.02 + 0.06 * prefs[:, 1]).astype(np.float32)            # novelty seekers drift faster
        pos = np.clip(u, 0, 1).astype(np.float32) * eta
        neg = np.clip(-u, 0, 1).astype(np.float32) * eta * 0.25
        c2 = c + pos[:, None] * (d - c) - neg[:, None] * d
        c2 = c2 / np.maximum(np.linalg.norm(c2, axis=1, keepdims=True), 1e-6)
        vals["interests"] = np.where(has[:, None], c2, c).astype(np.float16)
        k = columns.col(w, TABLE, "knowledge")[ix].astype(np.float32)
        vals["knowledge"] = np.clip(k + (0.02 * (u > 0) * has)[:, None].astype(np.float32) * np.abs(d), 0, 1
                                    ).astype(np.float16)
    if satisfaction is not None:
        s_new = np.asarray(satisfaction, np.float64).reshape(-1)
        if s_new.shape[0] != ix.size:
            raise InvalidAction("experience needs one satisfaction per individual")
        sat = columns.col(w, TABLE, "satisfaction")[ix].astype(np.float64)
        loyal = columns.col(w, TABLE, "prefs")[ix, 5].astype(np.float64)
        beta = 0.25 - 0.12 * loyal
        vals["satisfaction"] = ((1 - beta) * sat + beta * np.clip(s_new, -2, 2)).astype(np.float32)
        reg = core._registry(w)
        if activity in reg["spec"]:
            slot = reg["spec"][activity]["slot"]
            hab = columns.col(w, TABLE, "habit")[ix].astype(np.float32)
            if slot < hab.shape[1]:
                hab[:, slot] = np.clip(hab[:, slot] + 0.2 * (u - sat), -4, 4).astype(np.float32)
                vals["habit"] = hab
    if not vals:
        return None
    return record_decisions(ctx, f"pop.experience.{activity}"[:64], ix, vals, model_id=EXPERIENCE_MODEL[0],
                            model_version=EXPERIENCE_MODEL[1], fidelity_id=EXPERIENCE_MODEL[2],
                            extra={"activity": activity, "n": int(ix.size)})


# ============================================================================================ provenance (evaluator)
def check_provenance(world, events=None, *, prefixes=DECISION_KINDS_PREFIX) -> dict:
    """Evaluator side: classify every decision batch on the population table whose kind starts with one of
    `prefixes` (default: audience impressions and consumption).

    * organic population decision: the batch's REF is listed by a `pop.session` batch (the population's tick lists
      every decision batch recorded while its activities ran, `extra.decisions`), its model is registered, and every
      row concerns an individual activated in that session;
    * participant decision: the batch event belongs to a tool command (kernel `cmd` number) of a population
      individual, carries the participant model, and every row is that individual;
    * everything else is forced (module code, system calls, genesis, other processes), whatever it claims."""
    evs = list(world.log if events is None else events)
    ds = [e if isinstance(e, dict) else e.to_dict() for e in evs]
    models = world.state.tables.get("pop_decision_models", {})
    listed: dict[str, np.ndarray] = {}
    cmd_actor: dict[int, tuple[str, bool]] = {}
    for d in ds:
        if d["type"] == "cols.batch" and d["payload"]["table"] == TABLE and d["payload"]["kind"] == "pop.session":
            refs = (d["payload"].get("extra") or {}).get("decisions") or []
            if refs:
                cohort = artifacts.get_array_blob(world, d["payload"]["ref"])["idx"].astype(np.int64)
                for r in refs:
                    listed[r] = cohort
        elif d["type"] == "cmd" and d["payload"].get("n") is not None:
            cmd_actor[int(d["payload"]["n"])] = (d["actor"], bool(d["payload"].get("system")))
    out = {"batches": 0, "rows": 0, "organic_population_rows": 0, "participant_rows": 0, "forced_rows": 0,
           "forced_batches": 0, "missing_provenance_batches": 0, "unregistered_model_batches": 0,
           "outside_cohort_rows": 0, "by_model": {}}
    for d in ds:
        if d["type"] != "cols.batch":
            continue
        p = d["payload"]
        if p["table"] != TABLE or not p["kind"].startswith(tuple(prefixes)):
            continue
        v = artifacts.get_array_blob(world, p["ref"])["idx"].astype(np.int64)
        n = int(v.size)
        out["batches"] += 1
        out["rows"] += n
        prov = (p.get("extra") or {}).get("decision")
        if not isinstance(prov, dict):
            out["missing_provenance_batches"] += 1
            out["forced_rows"] += n
            out["forced_batches"] += 1
            continue
        mk = _model_key(str(prov.get("model_id", "")), str(prov.get("model_version", "")))
        out["by_model"][mk] = out["by_model"].get(mk, 0) + n
        if mk not in models:
            out["unregistered_model_batches"] += 1
            out["forced_rows"] += n
            out["forced_batches"] += 1
            continue
        if p["ref"] in listed and prov.get("model_id") != PARTICIPANT_MODEL:
            inside = np.isin(v, listed[p["ref"]])
            out["organic_population_rows"] += int(inside.sum())
            bad = int((~inside).sum())
            out["outside_cohort_rows"] += bad
            out["forced_rows"] += bad
            out["forced_batches"] += int(bad > 0)
            continue
        ca = cmd_actor.get(d["cmd"]) if d.get("cmd") is not None else None
        i = ids.index(ca[0]) if ca is not None else None
        if ca is not None and not ca[1] and i is not None and prov.get("model_id") == PARTICIPANT_MODEL and \
                np.all(v == i):
            out["participant_rows"] += n
            continue
        out["forced_rows"] += n
        out["forced_batches"] += 1
    return out
