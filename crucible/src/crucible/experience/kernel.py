"""`KernelExperienceEnv`: any kernel world (any `WorldSpec`) behind the `ExperienceEnv` interface.

This is the adapter for Tier 2 domains, Tier 3 civilisations, the 4.0 world families and anything else built on
`crucible.core.kernel.World`. It adds nothing to the world's rules: actions are `World.execute` tool calls (so
permissions, prices, rate limits, quotas and visibility are the world's own), time passes only through `advance`,
and observations are assembled from what the agent may see:

* `numbers`/`arrays`/`text` from the agent's own module summaries (`World.observe(agent)`), exact;
* `events`: every event visible to the agent, delivered losslessly - `observe` delivers up to `max_events` and says
  in `missing` how many more wait after its `cursor`; `observe_since(agent, cursor)` returns all of them;
* `cause` on an event: when the event happened in the transaction of one of the agent's own commands
  (``via: "direct"``) or in a scheduled process that such a command (transitively) scheduled (``via:
  "scheduled"``), the agent's command (`action_seq`, `tool`, `t`) and the delay. Only the agent's own actions are
  ever named: a consequence of someone else's action carries no cause;
* `artifacts`: content references carried by visible events (v5 ``sha256:`` artifacts, 4.0 content objects);
* `entities`, `resources` (the agent's own action counts, time used, rate limit, money), `analytics` (counts of
  public events), `missing` (pending results of the agent's own experiments and self-goals, hidden mechanisms
  present in this world, truncation notices), `objective` (the public description of the agent's goal).

Reward modes (Clarification 2): ``observation_only`` is the default - it pays 0.0 and the world is built (or adopted)
with ``evaluator_access="evaluator"`` (`crucible.core.evaluator_only`), so evaluator tools such as `goals.check_now`
and `use.report` are refused, evaluator fields (relatedness verdicts, the mechanism's objective quantity) are removed
from tool results, events and summaries, and no participant-authored predicate can read an evaluator metric. An
agent's objective is a `goals.contracts.GoalContract` (or any object with `begin/step_reward/status`); only its public
text is shown. ``verified_terminal_outcome`` (explicit) pays 1.0 once when the verifier confirms success;
``legacy_shaped_reward`` (explicit) pays the 4.0 `RewardContract` (potential shaping on verified progress plus a
success bonus; without an objective, the change of the agent's own balances, as the 4.0 `WorldEnv` does) and keeps
the 4.0 evaluator-access contract. In every mode the verifier runs after each `advance` and `execute`; its status is
returned only by `verify()`, which also reports evaluator-side outcome measures when no objective is set.
"""
from __future__ import annotations

import copy
from bisect import bisect_right
from typing import Any, Callable, Iterable

from ..core import evaluator_only as EO
from .env import BaseExperienceEnv, error_result
from .features import annotate_tools, artifact_refs, entities, flatten, texts
from .modes import LEGACY, OBSERVATION_ONLY, VERIFIED, RewardGate, check_mode, prepare_spec, prepare_world
from .observation import TIME_UNIT, Observation, json_copy

# What each module keeps from its participants (stated in `missing` when the module is present). Facts about the
# world's rules, never values.
HIDDEN = {
    "rig": "rig: how a facility's levers act on its gauges, revenue and cost is not observable (find out by acting)",
    "consumers": "consumers: customers' needs, budgets and preferences are not observable; their purchases are",
    "npcs": "npcs: other participants' private state and intentions are not observable",
    "society": "society: firm minds' internal books and plans are not observable",
    "use": "use: the hidden test batteries that score usefulness are not observable",
    "shocks": "shocks: future shocks are not announced in advance",
    "pop": "pop: individuals' interests, habits and attention budgets are not observable",
    "lab": "lab: the hidden traffic process of the engineering lab is not observable",
}
PAGE = 100_000
# event types that record an action without changing what the world is (the command itself, recorded external
# outcomes, idempotency records, counters, knowledge of a host)
BOOKKEEPING = frozenset({"cmd", "cmd.rejected", "ext", "idem", "ctr", "know", "note"})


class KernelExperienceEnv(BaseExperienceEnv):
    kind = "kernel"

    def __init__(self, world: Any, seed: int = 0, *, agents: Iterable[str] | int = ("agent",), endowment: int = 1_000_000,
                 reward_mode: str | None = None, objectives: Any = None, tier: int = 0, name: str | None = None,
                 extra_modules: Iterable[str] = (), experiments: bool = True, store=None, strict: bool = False,
                 max_events: int = 500, max_text: int = 200, max_artifacts: int = 200, join_kind: str = "agent",
                 evaluator: Callable[[Any, str], dict] | None = None, world_id: str | None = None,
                 semantic: bool | str = "auto", max_semantic: int = 8, controllers: dict | str | None = None):
        """`controllers` declares who controls the admitted participants (K_selfplay 3.4): one label for all of them
        (a learner driving several seats), or ``{handle: label}`` per participant; participants with one label are
        one controller for the evaluator's relatedness analysis (`commerce.integrity`), so revenue between them is
        never independent. Unlabelled participants are independent of each other (the 4.0 behaviour)."""
        from ..core.kernel import World
        import crucible
        crucible.load_all_modules()
        self.tier = int(tier)
        mode = check_mode(reward_mode)
        if isinstance(world, World):
            self.world = prepare_world(world, mode)
            label = name or world.spec.family
        else:
            spec = self._spec(world, seed)
            spec = prepare_spec(copy.deepcopy(spec), mode)
            have = {m["name"] for m in spec.modules}
            # self-generated experiments need identities (every world with participants has them)
            for m in list(extra_modules) + (["exp"] if experiments and "identity" in have else []):
                if m not in have:
                    spec.modules.append({"name": m, "config": {}})
                    have.add(m)
            from ..api.local import ensure_modules
            ensure_modules([m["name"] for m in spec.modules])
            self.world = World.create(spec, world_id=world_id, store=store, strict=strict)
            label = name or spec.family
        super().__init__(mode, label, self.world.spec.seed)
        self.max_events, self.max_text, self.max_artifacts = int(max_events), int(max_text), int(max_artifacts)
        self._encode = _semantic_encoder() if semantic in ("auto", True) else None
        if semantic is True and self._encode is None:
            raise ImportError("semantic=True needs crucible.semantic.encode (workstream D)")
        self.max_semantic = int(max_semantic)
        self._sem: dict[str, list] = {}
        self.gate = RewardGate(self.reward_mode)
        self.evaluator = evaluator
        self._agents: list[str] = []
        self._gone: dict[str, int] = {}          # visible events that left the in-memory window unobserved (no store)
        self._vis: dict[str, list[int]] = {}
        self._cause: dict[str, dict[int, dict]] = {}
        self._cursor: dict[str, int] = {}
        self._last_result: dict[str, dict | None] = {}
        self._artifacts: dict[str, dict[str, dict]] = {}
        self._objectives: dict[str, Any] = {}
        self._objective_text: dict[str, str | None] = {}
        self._cash: dict[str, int] = {}
        self._status: dict[str, dict] = {}
        self._counts: dict[str, dict] = {}
        self._job_cause: dict[str, dict] = {}
        self._tx: tuple[str, dict | None] = ("none", None)
        self._scanned = self.world.seq
        self._public: list[tuple[int, str]] = []
        self._public_total = 0
        self._tools_cache: dict[str, tuple[int, list[dict]]] = {}
        names = [f"agent{i}" for i in range(int(agents))] if isinstance(agents, int) else list(agents)
        for h in names:
            ctl = controllers.get(h, "") if isinstance(controllers, dict) else (controllers or "")
            self.add_agent(h, endowment=endowment, kind=join_kind, controller=str(ctl))
        objs = objectives if isinstance(objectives, dict) else ({a: objectives for a in self._agents}
                                                                 if objectives is not None else {})
        for a, o in objs.items():
            self.set_objective(a if a in self._agents else f"p:{a}", o)

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _spec(world: Any, seed: int):
        from ..core.kernel import WorldSpec
        if isinstance(world, WorldSpec):
            return world
        if isinstance(world, dict) and "modules" in world:
            d = dict(world)
            d.setdefault("family", "custom")
            d.setdefault("seed", int(seed))
            return WorldSpec.from_dict(d)
        from ..worldgen import generate
        return generate(world if isinstance(world, (dict, str)) else str(world), int(seed))

    def add_agent(self, handle: str, *, endowment: int = 1_000_000, kind: str = "agent", controller: str = "") -> str:
        """Admit a participant (recorded `admin.join`, with its `controller` label when one is given) or adopt an
        existing principal (``"p:..."``)."""
        w = self.world
        if handle.startswith(("p:", "svc:")) and w.principal(handle) is not None:
            p = handle
        else:
            args = {"handle": handle, "endowment": int(endowment), "kind": kind}
            if controller:
                args["controller"] = str(controller)
            p = w.system_call("admin.join", args)["principal"]
        if p in self._vis:
            return p
        self._agents.append(p)
        self._vis[p], self._cause[p], self._artifacts[p] = [], {}, {}
        self._gone[p] = 0
        self._cursor[p] = w.seq
        self._last_result[p] = None
        self._objective_text[p] = None
        self._counts[p] = {"executes": 0, "rejected": 0, "cost_s": 0}
        self._cash[p] = self._balance(p)
        self._scan()
        return p

    def set_objective(self, agent: str, objective: Any) -> None:
        """Give `agent` an objective: a `GoalContract` (wrapped in a `RewardContract` for the reward mode) or an
        object with ``begin/step_reward/status``. Its public text becomes the observation's `objective`."""
        from ..goals.contracts import GoalContract, RewardContract
        if isinstance(objective, GoalContract):
            text = objective.description
            obj = RewardContract(objective, shaping=0.0, success_bonus=1.0) if self.reward_mode == VERIFIED \
                else RewardContract(objective)
        else:
            obj = objective
            text = getattr(getattr(objective, "goal", None), "description", None) or getattr(objective, "text", None)
        obj.begin(self.world, agent)
        self._objectives[agent] = obj
        self._objective_text[agent] = text if isinstance(text, str) else None
        self._status[agent] = dict(obj.status(self.world, agent) or {})

    # ------------------------------------------------------------------ event scanning (lossless, incremental)
    def _scan(self) -> None:
        w = self.world
        if self._scanned >= w.seq:
            return
        evs = w.events_range(self._scanned, w.seq)
        ps = {a: w.principals_of(a) for a in self._agents}
        mine = set(self._agents)
        for e in evs:
            typ = e.type
            if typ == "cmd":
                if e.payload.get("n") is not None:
                    c = {"agent": e.actor, "action_seq": e.seq, "tool": e.payload.get("tool"), "t": e.t} \
                        if e.actor in mine else None
                    self._tx = ("direct", c)
                else:
                    self._tx = ("system", None)
            elif typ == "proc":
                self._tx = ("scheduled", self._job_cause.get(e.payload.get("id")))
            elif typ in ("time", "cmd.rejected"):
                self._tx = (typ, None)
            via, cause = self._tx
            if typ == "sched.add" and cause is not None:
                self._job_cause[e.payload["id"]] = cause
            if e.vis == "public":
                self._public.append((e.t, typ))
                self._public_total += 1
            for a in self._agents:
                if not w.visible(e, ps[a]):
                    continue
                if typ in ("cmd", "cmd.rejected") and e.actor == a:
                    continue                    # the agent's own command is not news to it
                self._vis[a].append(e.seq)
                if cause is not None and cause["agent"] == a:
                    self._cause[a][e.seq] = {"action_seq": cause["action_seq"], "tool": cause["tool"],
                                             "t": cause["t"], "delay_s": int(e.t - cause["t"]), "via": via}
                for ref in artifact_refs({"type": typ, "payload": e.payload, "t": e.t}):
                    self._artifacts[a].setdefault(ref["ref"], ref)
        self._scanned = w.seq
        cut = w.now - 86400
        if self._public and self._public[0][0] < cut:
            self._public = [x for x in self._public if x[0] >= cut]

    def _retained(self, agent: str) -> None:
        """Forget the visible events that left the world's in-memory window (`params.log_retain`) when the world keeps
        no event store: they cannot be delivered any more. How many were never observed is counted for the next
        observation's `missing` note; the bookkeeping of the forgotten ones is dropped (bounded memory in long runs)."""
        w = self.world
        if w.store is not None:
            return
        vis = self._vis[agent]
        k = bisect_right(vis, int(w.log_base))
        if not k:
            return
        self._gone[agent] += sum(1 for q in vis[:k] if q > self._cursor[agent])
        for q in vis[:k]:
            self._cause[agent].pop(q, None)
        del vis[:k]

    def _event(self, agent: str, seq: int) -> dict:
        e = self.world.participant_event(self.world.events_range(seq - 1, seq)[0])
        return {"seq": e.seq, "t": e.t, "type": e.type, "actor": e.actor, "payload": json_copy(e.payload),
                "cause": dict(self._cause[agent][seq]) if seq in self._cause[agent] else None}

    # ------------------------------------------------------------------ objectives and rewards
    def _balance(self, agent: str) -> int:
        st = self.world.state
        if "accounts" not in st.tables:
            return 0
        return int(sum(st.tables["accounts"][a]["balance"] for a in st.find_ids("accounts", "owner", agent)))

    def _tick(self) -> None:
        w = self.world
        for a in self._agents:
            obj = self._objectives.get(a)
            if obj is not None:
                r = float(obj.step_reward(w, a))
                st = dict(obj.status(w, a) or {})
                self._status[a] = st
                self.gate.add_legacy(a, r)
                self.gate.verified(a, bool(st.get("success")))
            elif self.reward_mode == LEGACY:
                cash = self._balance(a)
                self.gate.add_legacy(a, float(cash - self._cash[a]))
                self._cash[a] = cash
        self._scan()

    # ------------------------------------------------------------------ protocol
    def agents(self) -> list[str]:
        return list(self._agents)

    @property
    def now(self) -> int:
        return int(self.world.now)

    def _check(self, agent: str) -> None:
        if agent not in self._vis:
            raise KeyError(f"unknown agent {agent!r}; agents: {self._agents}")

    def tools(self, agent: str) -> list[dict]:
        self._check(agent)
        c = self._tools_cache.get(agent)
        if c is not None and c[0] == self.world.seq:
            return [dict(t, annotations=dict(t["annotations"])) for t in c[1]]
        tools = [t for t in self.world.available_tools(agent) if t["name"].split(".", 1)[0] != "admin"]
        ann = annotate_tools(json_copy(tools))
        self._tools_cache[agent] = (self.world.seq, ann)
        return [dict(t, annotations=dict(t["annotations"])) for t in ann]

    def execute(self, agent: str, action: dict) -> dict:
        self._check(agent)
        self.last_changed = False
        if not isinstance(action, dict):
            res = error_result(self.world.now, "", "invalid_action", "action must be {'tool': str, 'args': object}")
        else:
            act = {k: action[k] for k in ("tool", "args", "idempotency_key") if k in action}
            act.setdefault("args", {})
            ar = self.world.execute(agent, act)
            res = ar.to_dict()
            if ar.events:
                res["action_seq"] = int(ar.events[0])      # what `cause.action_seq` of its consequences names
            if ar.ok and ar.events:
                # changed: the action's transaction recorded more than its own bookkeeping
                evs = self.world.events_range(min(ar.events) - 1, max(ar.events))
                self.last_changed = any(e.type not in BOOKKEEPING for e in evs)
                self.changes += int(self.last_changed)
        cnt = self._counts[agent]
        cnt["executes"] += 1
        cnt["rejected"] += 0 if res.get("ok") else 1
        cnt["cost_s"] += int(res.get("cost_s") or 0)
        self._last_result[agent] = res
        self.steps += 1
        self._tick()
        return json_copy(res)

    def advance(self, seconds: int) -> None:
        if int(seconds) > 0:
            self.world.advance_time(self.world.now + int(seconds))
        self._tick()

    def reward(self, agent: str) -> float:
        self._check(agent)
        return 0.0 if self.reward_mode == OBSERVATION_ONLY else self.gate.take(agent)

    def observe_since(self, agent: str, cursor: int) -> tuple[list[dict], int]:
        """Every event visible to `agent` with seq > `cursor`, in order (a page of at most 100,000; the returned
        cursor is then the last returned seq, otherwise the world's current seq)."""
        self._check(agent)
        self._scan()
        self._retained(agent)
        vis = self._vis[agent]
        i = bisect_right(vis, int(cursor))
        page = vis[i:i + PAGE]
        nxt = page[-1] if len(vis) - i > PAGE else self.world.seq
        return [self._event(agent, q) for q in page], int(nxt)

    def observe(self, agent: str) -> Observation:
        self._check(agent)
        self._scan()
        self._retained(agent)
        w = self.world
        vis = self._vis[agent]
        i = bisect_right(vis, self._cursor[agent])
        pending = vis[i:]
        deliver = pending[:self.max_events]
        cursor = deliver[-1] if len(deliver) < len(pending) else w.seq
        self._cursor[agent] = cursor
        events = [self._event(agent, q) for q in deliver]
        summary = {k: v for k, v in w.observe(agent).items() if k not in ("time", "principal")}
        numbers, arrays = flatten(summary)
        numbers["world.time_s"] = int(w.now)
        numbers["world.day"] = int(w.now // 86400)
        missing: list[str] = []
        if EO.hidden(w):
            missing.append("evaluator: goal progress, verified quality, relatedness verdicts and integrity metrics are "
                           "evaluator data and are not observable (outcomes are verified separately)")
        txt = texts(summary, "summary", [])
        for e in events:
            texts(e["payload"], f"event[{e['seq']}].{e['type']}", txt)
        if len(txt) > self.max_text:
            missing.append(f"text: {len(txt) - self.max_text} more text items in this observation's summary and "
                           f"events (read the events with observe_since)")
            txt = txt[:self.max_text]
        arts = sorted(self._artifacts[agent].values(), key=lambda a: (a.get("t") or 0, a["ref"]))
        if len(arts) > self.max_artifacts:
            missing.append(f"artifacts: {len(arts)} known, the {self.max_artifacts} most recent are listed")
            arts = arts[-self.max_artifacts:]
        if len(deliver) < len(pending):
            missing.append(f"events: {len(pending) - len(deliver)} more visible events after cursor {cursor}; read "
                           f"them with observe_since(agent, {cursor})")
        if self._gone[agent]:
            missing.append(f"events: {self._gone[agent]} visible events left the world's in-memory window "
                           f"(params.log_retain = {w.spec.params.get('log_retain')}) before they were observed; this world "
                           f"keeps no event store to read them from")
            self._gone[agent] = 0
        if self._encode is not None:
            arrays.update(self._semantic(arts, missing))
        missing += self._pending_notes(agent)
        missing += [HIDDEN[m] for m in sorted(w.modules) if m in HIDDEN]
        p = w.principal(agent) or {}
        limit = int(w.spec.params.get("max_actions_per_hour", 0))
        hour = w.now // 3600
        cnt = self._counts[agent]
        accounts = w.state.find_ids("accounts", "owner", agent) if "accounts" in w.state.tables else []
        resources = {"time_unit": TIME_UNIT, "actions": int(p.get("actions", 0)),
                     "time_used_s": int(p.get("time_used", 0)),
                     "actions_this_hour": int(p.get("hour_actions", 0)) if p.get("hour") == hour else 0,
                     "action_limit_per_hour": limit, "executes": cnt["executes"], "rejected": cnt["rejected"],
                     "cost_s_total": cnt["cost_s"], "balance_cents": self._balance(agent), "accounts": len(accounts)}
        by_type: dict[str, int] = {}
        for _, typ in self._public:
            by_type[typ] = by_type.get(typ, 0) + 1
        analytics = {"public_events_total": self._public_total, "public_events_24h": len(self._public),
                     "public_events_24h_by_type": dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))[:25]),
                     "world_day": int(w.now // 86400)}
        ents = entities(agent, summary, events, w.principals_of(agent))
        return Observation(t=int(w.now), agent=agent, numbers=numbers, arrays=arrays, text=txt,
                           artifacts=[json_copy(a) for a in arts], tools=self.tools(agent), events=events,
                           entities=ents, resources=resources, analytics=analytics,
                           last_result=json_copy(self._last_result[agent]), missing=missing, cursor=int(cursor),
                           objective=self._objective_text.get(agent))

    def _semantic(self, arts: list[dict], missing: list[str]) -> dict[str, list]:
        """Semantic representations of the readable v5 artifacts (``sha256:`` refs) through the shared encoders
        (`crucible.semantic.encode`, recorded inference); at most `max_semantic` new encodings per observation."""
        out, new = {}, 0
        for a in arts:
            ref = a["ref"]
            if not ref.startswith("sha256:"):
                continue
            if ref not in self._sem:
                if new >= self.max_semantic:
                    missing.append(f"semantic: {ref} not encoded yet (encoding budget per observation reached)")
                    continue
                vec = self._encode(self.world, ref, _modality(a.get("mime")))
                self._sem[ref] = [float(x) for x in vec]
                new += 1
            out[f"semantic.{ref}"] = list(self._sem[ref])
        return out

    def _pending_notes(self, agent: str) -> list[str]:
        st = self.world.state
        out = []
        if "experiments" in st.tables:
            for i in st.find_ids("experiments", "owner", agent):
                g = st.tables["experiments"][i]
                if g["status"] == "open":
                    out.append(f"exp:{i}: outcome of your experiment pending until t={g['due']}")
        if "self_goals" in st.tables:
            for i in st.find_ids("self_goals", "owner", agent):
                g = st.tables["self_goals"][i]
                if g["status"] == "open":
                    out.append(f"goals:{i}: your self-goal is checked at t={g['due']}")
        return out

    # ------------------------------------------------------------------ evaluator side
    def verify(self) -> dict:
        """Independent verification (evaluator side): every agent's objective status from its verifier, the
        domain evaluator's verdict when one was given, invariants, and the world's digest."""
        w = self.world
        self.verifications += 1
        out = {"verifier": "goals.contracts.GoalVerifier" if self._objectives else
               "crucible.experience.kernel evaluator measures", "t": int(w.now),
               "seq": int(w.seq), "digest": w.digest(), "invariant_violations": w.check_invariants(),
               "agents": {a: dict(self._status.get(a) or {}) for a in self._agents},
               "measures": {a: self._measures(a) for a in self._agents}}
        if self.evaluator is not None:
            out["evaluation"] = {a: self.evaluator(w, a) for a in self._agents}
            out["verifier"] = out["verifier"] or "domain evaluator"
        hook = w.hooks.get("evaluate")
        if hook is not None:
            out["world_evaluation"] = hook(w, None)
        return out

    def _measures(self, agent: str) -> dict:
        """Evaluator-side outcome measures of a participant that hold with or without an objective: its own cash and,
        for every organisation it administers, the evaluator's independent revenue over the last 30 days and the
        organisation's status. Computed from authoritative state (never shown to participants)."""
        from ..goals.metrics import METRICS, evaluate_metric
        w = self.world
        out: dict = {}
        if "cash" in METRICS:
            out["cash_cents"] = float(evaluate_metric(w, "cash", {"principal": agent}))
        orgs = sorted(p for p in w.principals_of(agent) if p != agent and p.startswith("o:"))
        if orgs and "independent_revenue" in METRICS:
            out["orgs"] = {o: {"independent_revenue_30d_cents": float(evaluate_metric(
                w, "independent_revenue", {"principal": o, "window_s": 30 * 86400}))} for o in orgs}
        return out

    def digest(self) -> str:
        return self.world.digest()

    def describe(self) -> dict:
        d = super().describe()
        d.update(time_unit=TIME_UNIT, evaluator_access=EO.policy(self.world),
                 world_id=self.world.world_id, spec_hash=self.world.spec.hash(), family=self.world.spec.family,
                 agents=list(self._agents), seq=int(self.world.seq), t=int(self.world.now))
        return d

    def close(self) -> None:
        self.world.close()


def _semantic_encoder():
    """`crucible.semantic.encode` when workstream D's encoders are present, else None."""
    try:
        from .. import semantic
    except ImportError:
        return None
    fn = getattr(semantic, "encode", None)
    return fn if callable(fn) else None


def _modality(mime: Any) -> str:
    m = str(mime or "")
    for prefix, mod in (("image/", "image"), ("audio/", "audio"), ("video/", "video")):
        if m.startswith(prefix):
            return mod
    if m in ("text/x-python", "application/x-python", "text/x-script.python", "application/javascript"):
        return "code"
    return "text"
