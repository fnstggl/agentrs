"""Red-team probe framework.

A probe is one concrete attempt to make the environment or its reward system lie. It has three phases with
different powers:

* ``setup(op)``     - the *operator* prepares the scene (admits actors, lets simulated firms take ordinary actions).
* ``attack(a)``     - the attack itself, carried out **only through what an in-world participant can do**: tool calls
                      by principals the attacker controls (`World.execute`, the same path every agent API uses),
                      code the attacker deploys (it runs in the sandbox), content the attacker writes, and waiting.
                      While this phase runs, `World.system_call` raises `ProbeViolation`, and `Attacker.act` refuses
                      to act as a principal the probe did not declare as attacker-controlled.
* ``verdict(ev)``   - the *independent evaluator* reads authoritative state (never what the attacker says) and
                      decides what the attack achieved.

Outcomes: ``held`` (the attack achieved nothing false), ``false_success`` (a goal or metric reports value that
was not earned), ``money_created`` (resources appeared without a declared source), ``leak`` (hidden or private
information reached the attacker), ``integrity_break`` (authorisation, atomicity, settlement, replay or
availability of world mechanics was broken), ``false_failure`` (value that *was* earned is not credited: the
opposite credit-attribution error), ``legitimate_strategy`` (the attack worked and *should* count: the
probe documents why), ``not_applicable`` (the world or host cannot run the probe; the reason is given) and
``error`` (the probe itself failed; never silently treated as held).

After every probe, whatever its own verdict, the framework runs an integrity post-check (module invariants,
ledger recomputation, conservation, exact replay of the event log) and overrides a ``held`` verdict if it fails.
"""
from __future__ import annotations

import contextlib
import copy
import inspect
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..core.kernel import ActionResult, World

DAY = 86400
HOUR = 3600

HELD = "held"
FALSE_SUCCESS = "false_success"
MONEY_CREATED = "money_created"
LEAK = "leak"
INTEGRITY_BREAK = "integrity_break"
FALSE_FAILURE = "false_failure"
LEGITIMATE = "legitimate_strategy"
NOT_APPLICABLE = "not_applicable"
ERROR = "error"
OUTCOMES = (HELD, FALSE_SUCCESS, MONEY_CREATED, LEAK, INTEGRITY_BREAK, FALSE_FAILURE, LEGITIMATE, NOT_APPLICABLE,
            ERROR)
FINDINGS = (FALSE_SUCCESS, MONEY_CREATED, LEAK, INTEGRITY_BREAK, FALSE_FAILURE)
SCHEMA = "crucible-redteam/1"


class ProbeViolation(AssertionError):
    """The attack phase tried to use a power no in-world participant has."""


class NotApplicable(Exception):
    """Raised by a probe (any phase) when the world or the host cannot run it."""


class AttackRefused(Exception):
    """The world refused the decisive step of an attack (`Attacker.strike`): the attack held at that step."""

    def __init__(self, tool: str, error: dict | None):
        super().__init__(f"{tool}: {error}")
        self.tool, self.error = tool, error or {}


@dataclass
class Verdict:
    outcome: str
    summary: str
    evidence: dict = field(default_factory=dict)
    also: list[str] = field(default_factory=list)   # further outcome classes the same attack produced
    severity: str = ""                              # critical | high | medium | low | info (findings only)
    fix: str = ""                                   # proposed fix (file, function, what to change)

    def __post_init__(self):
        if self.outcome not in OUTCOMES or any(o not in OUTCOMES for o in self.also):
            raise ValueError(f"unknown outcome {self.outcome!r} / {self.also!r}")


# ====================================================================== phase handles
class Operator:
    """Setup powers: the simulator operator and the simulated society acting in their ordinary way."""

    def __init__(self, world: World, attackers: set[str], bystanders: set[str] | None = None):
        self.world = world
        self._attackers = attackers
        self._bystanders = bystanders if bystanders is not None else set()

    def join(self, handle: str, endowment: int = 0, attacker: bool = True, kind: str = "agent") -> str:
        """Admit an actor. `attacker=False` admits an independent participant (a bystander: an honest customer,
        a victim running its own harness) whose scripted actions are not part of the attacker's powers."""
        r = self.world.system_call("admin.join", {"handle": handle, "endowment": int(endowment), "kind": kind})
        (self._attackers if attacker else self._bystanders).add(r["principal"])
        return r["principal"]

    def control(self, *principals: str) -> None:
        """Declare further principals as attacker-controlled (e.g. organisations the attacker will found)."""
        self._attackers.update(principals)

    def sys(self, tool: str, /, **args) -> Any:
        return self.world.system_call(tool, args)

    def as_(self, principal: str, tool: str, /, **args) -> Any:
        """A simulated (non-attacker) principal takes an ordinary action of its own."""
        return self.world.system_call(tool, args, as_principal=principal)

    def advance(self, seconds: int) -> None:
        self.world.advance_time(self.world.now + int(seconds))

    def account(self, owner: str) -> str | None:
        return self.world.modules["bank"].primary_account(self.world.state, owner)


class Attacker:
    """Attack powers: tool calls as attacker-controlled principals, reading what they may read, and waiting."""

    def __init__(self, world: World, attackers: set[str], bystanders: set[str] | None = None):
        self._w = world
        self.controlled = attackers
        self.bystanders = bystanders if bystanders is not None else set()
        self.transcript: list[dict] = []
        self.notes: list[str] = []

    # -- independent participants (not attacker powers): an honest customer, a victim's own harness
    def bystander(self, actor: str, tool: str, /, **args) -> ActionResult:
        """An independent participant declared in setup takes one ordinary action of its own. Recorded in the
        transcript as such; the evaluator never counts bystanders as part of the attacker group."""
        if actor not in self.bystanders:
            raise ProbeViolation(f"{actor} was not declared as a bystander")
        r = self._w.execute(actor, {"tool": tool, "args": args})
        self.transcript.append({"bystander": actor, "tool": tool, "args": _clip(args), "ok": r.ok,
                                **({} if r.ok else {"error": (r.error or {}).get("code")})})
        return r

    def bystander_account(self, actor: str) -> str | None:
        if actor not in self.bystanders:
            raise ProbeViolation(f"{actor} was not declared as a bystander")
        return self._w.modules["bank"].primary_account(self._w.state, actor)

    def bystander_sees(self, actor: str, since_seq: int = 0) -> dict:
        if actor not in self.bystanders:
            raise ProbeViolation(f"{actor} was not declared as a bystander")
        obs = self._w.observe(actor)
        evs, _ = self._w.visible_events(actor, since_seq, 100_000)
        obs["events"] = [{"seq": e.seq, "t": e.t, "type": e.type, "actor": e.actor, "payload": e.payload} for e in evs]
        return obs

    # -- acting
    def act(self, actor: str, tool: str, /, _key: str | None = None, **args) -> ActionResult:
        if actor not in self.controlled:
            raise ProbeViolation(f"{actor} is not an attacker-controlled principal")
        action: dict = {"tool": tool, "args": args}
        if _key is not None:
            action["idempotency_key"] = _key
        r = self._w.execute(actor, action)
        self.transcript.append({"actor": actor, "tool": tool, "args": _clip(args), "ok": r.ok,
                                **({} if r.ok else {"error": (r.error or {}).get("code"),
                                                    "message": str((r.error or {}).get("message", ""))[:160]})})
        return r

    def raw(self, actor: str, action: Any) -> ActionResult:
        """Send an arbitrary (possibly malformed) action object, as an API client could."""
        if actor not in self.controlled:
            raise ProbeViolation(f"{actor} is not an attacker-controlled principal")
        r = self._w.execute(actor, action)
        self.transcript.append({"actor": actor, "raw": _clip(action), "ok": r.ok,
                                **({} if r.ok else {"error": (r.error or {}).get("code")})})
        return r

    def must(self, actor: str, tool: str, /, **args) -> Any:
        r = self.act(actor, tool, **args)
        if not r.ok:
            raise NotApplicable(f"attack step {tool} was refused: {r.error}")
        return r.result

    def strike(self, actor: str, tool: str, /, **args) -> Any:
        """The decisive step of an attack. If the world refuses it, the attack has been stopped right there and the
        probe ends as `held` with the refusal as evidence (unlike `must`, whose failure means the probe's
        preparation no longer fits the world and yields `not_applicable`)."""
        r = self.act(actor, tool, **args)
        if not r.ok:
            raise AttackRefused(tool, r.error)
        return r.result

    def control(self, *principals: str) -> None:
        """Principals the attacker created in-world (its organisations, its services) become controlled."""
        self.controlled.update(principals)

    # -- seeing
    def observe(self, actor: str, since_seq: int = 0, limit: int = 100_000) -> dict:
        """Exactly what an agent API returns to this actor: module observations plus the events visible to it."""
        if actor not in self.controlled:
            raise ProbeViolation(f"{actor} is not an attacker-controlled principal")
        obs = self._w.observe(actor)
        evs, _ = self._w.visible_events(actor, since_seq, limit)
        obs["events"] = [{"seq": e.seq, "t": e.t, "type": e.type, "actor": e.actor, "payload": e.payload} for e in evs]
        return obs

    def tools(self, actor: str) -> list[dict]:
        return self._w.available_tools(actor)

    def has_host(self, host: str) -> bool:
        """Whether a built-in service host exists in this world (visible to every agent in its tool list)."""
        return any(e.host == host for e in self._w.endpoints.values())

    def account(self, owner: str) -> str | None:
        """An attacker-controlled principal's own primary account (it can list its accounts with bank.accounts)."""
        if owner not in self.controlled:
            raise ProbeViolation(f"{owner} is not an attacker-controlled principal")
        return self._w.modules["bank"].primary_account(self._w.state, owner)

    @property
    def now(self) -> int:
        return self._w.now

    @property
    def seq(self) -> int:
        return self._w.seq

    # -- waiting (time passes; the world's own processes run)
    def wait(self, seconds: int) -> None:
        self._w.advance_time(self._w.now + int(seconds))

    def note(self, text: str) -> None:
        self.notes.append(text)


class Evaluator:
    """Independent verification over authoritative state. Nothing here trusts the attacker's transcript."""

    def __init__(self, world: World, attackers: set[str], baseline: dict):
        self.world = world
        self.attackers = set(attackers)
        self.baseline = baseline

    def metric(self, name: str, **args) -> float:
        from ..goals.metrics import evaluate_metric
        return evaluate_metric(self.world, name, args)

    def cash(self, principal: str) -> int:
        st = self.world.state
        return int(sum(st.tables["accounts"][a]["balance"] for a in st.find_ids("accounts", "owner", principal)))

    def wealth(self, group: Iterable[str]) -> int:
        """Consolidated cash of a group of principals (all their accounts, whatever their status)."""
        return int(sum(self.cash(p) for p in sorted(set(group))))

    def group(self) -> set[str]:
        """Everything the attacker controls: the declared principals, organisations they own or administer, and
        the service principals those operate."""
        return controlled_closure(self.world, self.attackers)

    def wealth_gain(self) -> int:
        """Change of the attacker group's consolidated cash since the attack began (internal transfers cancel)."""
        return self.wealth(self.group()) - int(self.baseline["attacker_wealth"])

    def minted_to(self, group: Iterable[str] | None = None, since_seq: int | None = None) -> int:
        """Cents that moved from a *source* account (genesis / external / fees: accounts allowed to go negative)
        directly into accounts of `group` after the attack began. Any such amount needs a declared source."""
        st = self.world.state
        g = set(group) if group is not None else self.group()
        accts = {a for p in g for a in st.find_ids("accounts", "owner", p)}
        since = self.baseline["seq"] if since_seq is None else since_seq
        total = 0
        for e in self.world.events_range(since):
            if e.type != "bank.txn":
                continue
            posts = e.payload["postings"]
            if any(d < 0 and st.tables["accounts"][a]["kind"] == "system" for a, d in posts):
                total += sum(d for a, d in posts if d > 0 and a in accts)
        return int(total)

    def integrity(self) -> dict:
        return integrity_postcheck(self.world)

    def events(self, type_prefix: str = "", since_seq: int | None = None) -> list:
        since = self.baseline["seq"] if since_seq is None else since_seq
        return [e for e in self.world.events_range(since) if e.type.startswith(type_prefix)]


def controlled_closure(world: World, principals: Iterable[str]) -> set[str]:
    st = world.state
    out = set(principals)
    if "members" in st.tables:
        for p in list(out):
            for m in st.find("members", "person", p):
                if m["role"] in ("owner", "admin"):
                    out.add(m["org"])
    for pid, rec in st.scan("principals"):
        if rec["kind"] == "service" and rec["info"].get("operator") in out:
            out.add(pid)
    return out


def integrity_postcheck(world: World, replay: bool = True, max_replay_events: int = 400_000) -> dict:
    """Invariants every world must satisfy after any sequence of in-world actions."""
    out: dict[str, Any] = {"invariants": world.check_invariants()}
    bank = world.modules.get("bank")
    if bank is not None:
        out["ledger"] = bank.check_ledger_consistency(world)[:10]
        out["sum_of_balances"] = int(sum(a["balance"] for a in world.state.tables["accounts"].values()))
        bad = [aid for aid, a in world.state.tables["accounts"].items()
               if a["kind"] != "system" and a["balance"] < -a["overdraft"]]
        out["overdrawn"] = bad[:10]
    else:
        out["ledger"], out["sum_of_balances"], out["overdrawn"] = [], 0, []
    out["replay"] = "skipped"
    if replay and world.log_base == 0 and world.seq <= max_replay_events:
        try:
            r = World.replay(world.spec, [e.to_dict() for e in world.log], verify=True)
            out["replay"] = "exact" if (r.digest() == world.digest() and r.last_hash == world.last_hash) else "diverged"
        except Exception as e:  # noqa: BLE001 - a broken chain or applier is itself the finding
            out["replay"] = f"failed: {type(e).__name__}: {e}"[:300]
    out["ok"] = (not out["invariants"] and not out["ledger"] and out["sum_of_balances"] == 0 and not out["overdrawn"]
                 and out["replay"] in ("exact", "skipped"))
    return out


# ====================================================================== probes
class Probe:
    """Base class. Subclasses set `name`, `attack_class`, `description`, optionally `world` (requirements passed
    to the world factory) and `needs_sandbox`, and implement the three phases. Per-run state lives on `self`
    (`run_probes` works on a copy, so a probe instance can be reused)."""

    name: str = ""
    attack_class: str = ""
    description: str = ""
    world: dict = {}
    needs_sandbox: bool = False
    slow: bool = False

    def setup(self, op: Operator) -> None:
        pass

    def attack(self, a: Attacker) -> None:
        raise NotImplementedError

    def verdict(self, ev: Evaluator) -> Verdict:
        raise NotImplementedError


def default_world_factory(consumers: bool = False, open_world: bool = False, seed: int = 1, store=None,
                          params: dict | None = None, overrides: dict | None = None, world_id: str | None = None,
                          strict: bool = False, drop: Iterable[str] = (), family: str | None = None,
                          extra_modules: Iterable[str] = ()) -> World:
    """Small real worlds for probes. Without `consumers` the demand process is left out (the ledger, contracts,
    identity, companies, labour market, lenders, cloud and marketplace are all there), which makes multi-week
    attacks cheap; probes that need paying simulated customers ask for `consumers=True`."""
    import crucible
    crucible.load_all_modules()
    from ..worldgen import generate
    fam = family or ("open_digital" if open_world else "tiny")
    if fam == "__first_mech_family__":      # worlds hosting a hidden mechanism (Crucible 4.0 `rig` module), if present
        import importlib
        for name in ("crucible.mechanisms.module", "crucible.worldgen.mech_families"):   # registered on import
            try:
                importlib.import_module(name)
            except ImportError:
                pass
        from ..worldgen.generator import FAMILIES
        mech = sorted(f for f in FAMILIES if f.startswith("mech_") and not f.endswith("_tools"))
        if not mech:
            raise NotApplicable("no `mech_*` world family is registered in this build")
        fam, consumers = mech[0], True
    ov = dict(overrides or {})
    if open_world:
        ov.setdefault("n_consumers", 300)
    cfg: dict = {"family": fam, "overrides": ov}
    dropped = list(drop)
    if not consumers:
        dropped += ["consumers", "shocks"]
    if dropped:
        cfg["drop_modules"] = dropped
    if extra_modules:
        cfg["extra_modules"] = list(extra_modules)
    p = dict(params or {})
    if p:
        ov["params"] = {**ov.get("params", {}), **p}
    spec = generate(cfg, seed)
    tmp = None
    if store == "memory":
        from ..storage.memory import MemoryStore
        store = MemoryStore()
    elif store == "sqlite":
        import tempfile
        from ..storage.sqlite import SQLiteStore
        tmp = tempfile.mkdtemp(prefix="crucible-redteam-")
        store = SQLiteStore(os.path.join(tmp, "world.db"))
    w = World.create(spec, world_id=world_id, store=store, strict=strict)
    if tmp is not None:
        w.runtime["redteam_tmpdir"] = tmp
    return w


_SANDBOX_OK: list = []


def sandbox_available() -> tuple[bool, str]:
    """Whether agent code can really be executed here (Landlock + seccomp + privilege drop)."""
    if _SANDBOX_OK:
        return _SANDBOX_OK[0]
    try:
        from ..sandbox.runner import run_program
        out = run_program({"p.py": "def f(a, c):\n    return 7\n"}, "p:f", None, timeout=20)
        res = (bool(out.get("ok")) and out.get("result") == 7, "" if out.get("ok") else str(out.get("error"))[:200])
    except Exception as e:  # noqa: BLE001
        res = (False, f"{type(e).__name__}: {e}"[:200])
    _SANDBOX_OK.append(res)
    return res


def _call_factory(factory: Callable[..., World], req: dict) -> World:
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        params = {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return factory(**req)
    kw = {k: v for k, v in req.items() if k in params}
    missing = sorted(k for k, v in req.items() if k not in params and v)
    if missing:
        raise NotApplicable(f"the world factory cannot provide {missing}")
    return factory(**kw)


@contextlib.contextmanager
def _no_operator_powers(world: World):
    def refuse(*a, **k):
        raise ProbeViolation("the attack phase used operator authority (World.system_call)")
    orig = world.system_call
    world.system_call = refuse  # type: ignore[method-assign]
    try:
        yield
    finally:
        world.system_call = orig  # type: ignore[method-assign]


def run_probe(world_factory: Callable[..., World], probe: Probe) -> dict:
    p = copy.copy(probe)
    t0 = time.perf_counter()
    rec: dict[str, Any] = {"name": p.name, "attack_class": p.attack_class, "attack": " ".join(p.description.split())}
    world = None
    attacker = None
    try:
        if p.needs_sandbox:
            ok, why = sandbox_available()
            if not ok:
                raise NotApplicable(f"needs the real sandbox, which cannot start here ({why})")
        world = _call_factory(world_factory, dict(p.world))
        attackers: set[str] = set()
        bystanders: set[str] = set()
        p.setup(Operator(world, attackers, bystanders))
        baseline = {"seq": world.seq, "t": world.now,
                    "attacker_wealth": Evaluator(world, attackers, {}).wealth(controlled_closure(world, attackers))}
        attacker = Attacker(world, attackers, bystanders)
        with _no_operator_powers(world):
            p.attack(attacker)
        ev = Evaluator(world, attackers, baseline)
        v = p.verdict(ev)
        post = integrity_postcheck(world)
        if not post["ok"] and v.outcome != INTEGRITY_BREAK:
            v.also = sorted(set(v.also) | ({v.outcome} if v.outcome in FINDINGS else set()))
            v.summary = "integrity post-check failed after the attack; " + v.summary
            v.outcome = INTEGRITY_BREAK
        rec.update(result=v.outcome, also=v.also, severity=v.severity, summary=v.summary, evidence=_clip(v.evidence, 4000),
                   fix=v.fix, postcheck={k: post[k] for k in ("ok", "invariants", "ledger", "sum_of_balances", "replay")},
                   events=world.seq, sim_days=round(world.now / DAY, 2))
    except AttackRefused as e:
        post = integrity_postcheck(world) if world is not None else {"ok": True}
        rec.update(result=HELD if post["ok"] else INTEGRITY_BREAK, also=[], severity="", fix="",
                   summary=f"the decisive step `{e.tool}` was refused by the world: "
                           f"{e.error.get('code')}: {str(e.error.get('message', ''))[:200]}",
                   evidence={"refused_step": e.tool, "error": _clip(e.error)},
                   postcheck={k: post.get(k) for k in ("ok", "invariants", "ledger", "sum_of_balances", "replay")})
    except NotApplicable as e:
        rec.update(result=NOT_APPLICABLE, also=[], severity="", summary=str(e), evidence={}, fix="")
    except ProbeViolation as e:
        rec.update(result=ERROR, also=[], severity="", summary=f"probe is not a valid in-world attack: {e}",
                   evidence={}, fix="")
    except Exception as e:  # noqa: BLE001 - a crashed probe must show up as an error, never as "held"
        rec.update(result=ERROR, also=[], severity="", summary=f"{type(e).__name__}: {e}"[:500],
                   evidence={"traceback": traceback.format_exc(limit=8)[-2500:]}, fix="")
    finally:
        if attacker is not None:
            rec["steps"] = len(attacker.transcript)
            rec["transcript"] = attacker.transcript[:12] + ([{"more": len(attacker.transcript) - 12}]
                                                            if len(attacker.transcript) > 12 else [])
            if attacker.notes:
                rec["notes"] = attacker.notes[:10]
        if world is not None:
            tmp = world.runtime.get("redteam_tmpdir")
            try:
                world.close()
            except Exception:  # noqa: BLE001
                pass
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    rec["wall_s"] = round(time.perf_counter() - t0, 3)
    return rec


def run_probes(world_factory: Callable[..., World] | None, probes: Iterable[Probe], progress: Callable | None = None) -> dict:
    """Run every probe in its own fresh world and return the report (JSON-serialisable)."""
    factory = world_factory or default_world_factory
    rows = []
    for p in probes:
        r = run_probe(factory, p)
        rows.append(r)
        if progress is not None:
            progress(r)
    counts = {o: sum(1 for r in rows if r["result"] == o) for o in OUTCOMES}
    by_class: dict[str, dict] = {}
    for r in rows:
        c = by_class.setdefault(r["attack_class"].split("/")[0], {o: 0 for o in OUTCOMES})
        c[r["result"]] += 1
    return {"schema": SCHEMA, "n": len(rows), "counts": counts, "by_class": by_class,
            "findings": [{"name": r["name"], "result": r["result"], "also": r.get("also", []),
                          "severity": r.get("severity", ""), "summary": r["summary"], "fix": r.get("fix", "")}
                         for r in rows if r["result"] in FINDINGS],
            "errors": [r["name"] for r in rows if r["result"] == ERROR],
            "probes": rows}


def table(report: dict) -> str:
    """Plain-text table of probe outcomes."""
    w = max((len(r["name"]) for r in report["probes"]), default=10)
    lines = [f"{'probe'.ljust(w)}  {'result'.ljust(19)} {'sev'.ljust(8)} summary"]
    for r in report["probes"]:
        lines.append(f"{r['name'].ljust(w)}  {r['result'].ljust(19)} {(r.get('severity') or '-').ljust(8)} "
                     f"{r['summary'][:150]}")
    return "\n".join(lines)


def _clip(x: Any, budget: int = 600) -> Any:
    """Keep evidence and transcripts readable: long strings and long lists are truncated."""
    if isinstance(x, str):
        return x if len(x) <= budget else x[:budget] + f"...(+{len(x) - budget} chars)"
    if isinstance(x, dict):
        return {str(k): _clip(v, budget) for k, v in list(x.items())[:60]}
    if isinstance(x, (list, tuple)):
        xs = [_clip(v, budget) for v in list(x)[:40]]
        return xs + ([f"...(+{len(x) - 40} items)"] if len(x) > 40 else [])
    if isinstance(x, (set, frozenset)):
        return _clip(sorted(x, key=str), budget)
    if isinstance(x, (int, float, bool)) or x is None:
        return x
    return str(x)[:budget]
