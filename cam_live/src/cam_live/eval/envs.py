"""cam_live.eval.envs — evaluation families, the independent ground-truth evaluator and a software-utility domain.

Families (all controlled, owned, no external service is ever contacted):

``F1``  ``lab/first``  — the *first objective* (experience phase): LabWorld (6-day horizon) with 4 channels whose rates follow the
        family prior ``w ~ N((1, -0.5, 0.3), 0.5)``, a key-door channel and a noisy gauge (``validation.lab_config``).
``T1``  ``lab/door``   — an unfamiliar objective with a *different mechanism class*: inverted feature effects, five
        channels, weak direct channels and a delayed dependency that dominates (the last channel yields 6x after
        ``lab.unlock``, 0.05x before). Same adapter interface, different causal structure.
``T2``  ``codec``      — a genuinely different domain: a software-utility tuning task on real software. The agent
        chooses a compressor (zlib / bz2 / lzma, real ``zlib``/``bz2``/``lzma`` runs on a seeded corpus) and a
        level to reach a compression-ratio target. Outcomes are real measurements; an independent verifier
        re-runs the compressor on the recorded setting and signs the result. No adopters, no channels, no delays.

:class:`TruthTap` wraps an environment and records, for every executed action, the ground-truth quality the agent
never sees (expected adopters from the hidden rates in LabWorld; the measured ratio relative to the best available
setting in CodecWorld). It is the independent evaluator of action quality; it never changes what the agent sees.
"""
from __future__ import annotations

import bz2
import hashlib
import hmac
import json
import lzma
import random
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import (ActionResult, ActionSpec, AuthorizationDecision, AuthVerdict, Cost, EXEC_INTERFACE_VERSION,
                         FailureClass, GoalStatus, Level, Limits, Modality, Observation, OutcomeKind,
                         OutcomeObservation, OutcomeQuery, OutcomeStatus, PrimitiveOp, Provenance, Quantity,
                         ResultStatus, TaskSpec, ToolDescriptor, Trust, VerifiedOutcome, new_id)
from ..diagnostics.world import DAY, LabConfig, LabVerifier, LabWorld, lab_task

FAMILIES = ("F1", "T1", "T2")


def lab_family_config(family: str, seed: int) -> LabConfig:
    if family == "F1":
        return LabConfig(seed=seed, n_channels=4, target=12, budget=40.0, horizon_s=6 * DAY,
                         outcome_horizon_s=6 * 3600.0, mean_delay_s=(1800.0, 4 * 3600.0), at_most_once=True,
                         family="lab/first")
    if family == "T1":
        return LabConfig(seed=seed, n_channels=5, target=10, budget=40.0, horizon_s=6 * DAY,
                         outcome_horizon_s=6 * 3600.0, mean_delay_s=(1800.0, 4 * 3600.0), at_most_once=True,
                         w_mean=(-1.0, 0.5, -0.3), rate_scale=1.0, door_boost=6.0, door_locked=0.05,
                         family="lab/door")
    raise KeyError(f"{family!r} is not a LabWorld family")


#: decision budget and candidate count the per-world targets are calibrated with (docs/S/PROTOCOLS.md)
LAB_DECISIONS, LAB_K, TARGET_FRACTION = 16, 16, 0.5
_TARGETS: Dict[Tuple[str, int], int] = {}


def lab_target(family: str, seed: int) -> int:
    """Per-world objective: half of what the oracle policy is expected to produce with the same candidates and
    decision budget (so the target is attainable but not by chance; uniform choice reaches it in a minority)."""
    key = (family, int(seed))
    if key not in _TARGETS:
        from .runner import run_anchor
        e = run_anchor("oracle", family, seed, max_decisions=LAB_DECISIONS, k=LAB_K, target=1)["expected_adopters"]
        _TARGETS[key] = max(4, int(round(TARGET_FRACTION * float(e))))
    return _TARGETS[key]


def make_world(family: str, seed: int, clock: Any, *, target: Optional[int] = None) -> Tuple[Any, Any, TaskSpec]:
    """(environment, independent verifier, task) of one lifetime of ``family`` with world seed ``seed``."""
    if family in ("F1", "T1"):
        cfg = lab_family_config(family, seed)
        cfg.target = int(target if target is not None else lab_target(family, seed))
        world = LabWorld(cfg, clock)
        task = lab_task(world, f"obj_{family}_{seed}")
        return world, LabVerifier(world), task
    if family == "T2":
        world = CodecWorld(seed, clock)
        return world, CodecVerifier(world), world.task(f"obj_T2_{seed}")
    raise KeyError(family)


# ================================================================================================ ground truth
class TruthTap:
    """Proxy of an ``EnvironmentAdapter`` that records the ground-truth quality of every executed action.

    LabWorld: expected adopters of ``lab.act`` = hidden rate x effort at execution time (Poisson mean); the oracle
    rate is the best channel's rate with the door unlocked. CodecWorld: the measured ratio of the chosen setting and
    the best ratio over all settings (computed by the tap, not visible to the agent)."""

    def __init__(self, env: Any):
        self._env = env
        self.rows: List[Dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def execute(self, action: ActionSpec, authorization: AuthorizationDecision) -> ActionResult:
        row = {"i": len(self.rows), "operation": action.operation, "arguments": dict(action.arguments),
               "t": float(self._env.clock.now())}
        env = self._env
        if isinstance(env, LabWorld):
            c = env.cfg
            row["oracle_rate"] = max(oracle_rates(env))
            if action.operation == "lab.act":
                ch, e = int(action.arguments.get("channel", -1)), int(action.arguments.get("effort", 0) or 0)
                if 0 <= ch < c.n_channels and 1 <= e <= c.max_effort:
                    row["rate"] = env._rate(ch)
                    row["expected_adopters"] = row["rate"] * e
                    row["money"] = c.act_cost * e
        elif isinstance(env, CodecWorld):
            row["best_ratio"] = env.best_ratio()
            r = env.ratio_of(action)
            if r is not None:
                row["ratio"] = r
                row["quality"] = r / row["best_ratio"]
        res = env.execute(action, authorization)
        row["status"] = res.status
        row["failure"] = res.failure_class
        if res.status != ResultStatus.OK:
            row.pop("expected_adopters", None)
            row.pop("quality", None)
        self.rows.append(row)
        return res


def oracle_rates(world: LabWorld) -> List[float]:
    """Hidden per-effort rates with the key-door channel unlocked (the best any policy could use)."""
    c = world.cfg
    out = []
    for ch in range(c.n_channels):
        r = world._rate(ch)
        if c.key_door and ch == c.n_channels - 1 and world._unlocked_at is None:
            r = r / c.door_locked * c.door_boost
        out.append(r)
    return out


def lab_efficiency(rows: Sequence[Dict[str, Any]]) -> float:
    """Money efficiency of the executed interventions: sum(rate x effort) / sum(best rate x effort), the best rate
    being the best channel with the door counted as unlocked (what an oracle would use). In [0, 1]; 0 if the agent
    never intervened. Independent of arrival noise and of how early the objective was reached."""
    num = sum(r.get("expected_adopters", 0.0) for r in rows)
    den = sum(r["oracle_rate"] * int(r["arguments"].get("effort", 0) or 0) for r in rows
              if "expected_adopters" in r)
    return float(num / den) if den > 0 else 0.0


# ================================================================================================ CodecWorld
_WORDS = ("data system model learning signal compress window block entropy token stream archive delta "
          "buffer index record value cache vector matrix sample batch layer energy field").split()


def make_corpus(seed: int, size: int = 48_000) -> bytes:
    """A seeded corpus mixing (a) word salad, (b) blocks repeated at long distance (> 32 KiB: invisible to zlib's
    window, visible to lzma), (c) small-alphabet noise (favours entropy coding). The mixture weights vary by seed,
    so the best compressor differs between worlds."""
    rng = random.Random(seed * 7_368_787 + 11)
    w = [rng.random() + 0.05 for _ in range(3)]
    tot = sum(w)
    parts: List[bytes] = []
    n_text = int(size * w[0] / tot)
    words = []
    while sum(len(x) + 1 for x in words) < n_text:
        words.append(rng.choice(_WORDS))
    parts.append(" ".join(words).encode())
    n_rep = int(size * w[1] / tot)
    block = bytes(rng.randrange(256) for _ in range(max(64, n_rep // 6)))
    pad = bytes(rng.randrange(256) for _ in range(34_000))
    parts.append(block + pad[: max(0, min(len(pad), n_rep))] + block)
    n_noise = size - sum(len(p) for p in parts)
    alpha = bytes(rng.sample(range(256), 6))
    parts.append(bytes(rng.choice(alpha) for _ in range(max(0, n_noise))))
    return b"".join(parts)


#: operation -> (argument, offered settings). Four settings per compressor, listed in the tool schema as an enum.
CODECS = {"zlib.compress": ("level", (1, 4, 6, 9)), "bz2.compress": ("level", (1, 3, 6, 9)),
          "lzma.compress": ("preset", (0, 3, 6, 9))}


def run_codec(op: str, level: int, data: bytes) -> int:
    if op == "zlib.compress":
        return len(zlib.compress(data, level))
    if op == "bz2.compress":
        return len(bz2.compress(data, compresslevel=level))
    if op == "lzma.compress":
        return len(lzma.compress(data, preset=level))
    raise KeyError(op)


class CodecWorld:
    """``contracts.EnvironmentAdapter`` for the compression-tuning objective. Real compressor runs; the ratio of a
    run is observed immediately and is also available as an outcome query (``codec.ratio.from.<action id>``)."""
    interface_version = EXEC_INTERFACE_VERSION

    def __init__(self, seed: int, clock: Any, *, budget: float = 30.0, max_decisions: int = 24):
        self.seed, self.clock = int(seed), clock
        self.data = make_corpus(seed)
        self.budget, self.max_decisions = float(budget), int(max_decisions)
        self.t0 = clock.now()
        self._spent = 0.0
        self._runs: Dict[str, Dict[str, Any]] = {}            # action id -> {op, level, ratio}
        self._pending: Dict[str, OutcomeQuery] = {}
        self._first = True
        self._results: Dict[str, ActionResult] = {}
        self._cache: Dict[Tuple[str, int], float] = {}
        self._best = max(self._ratio(op, lv) for op, (_, levels) in CODECS.items() for lv in levels)
        # the target: the best ratio any offered setting reaches, minus 0.05 % (one setting or exact ties qualify)
        self.target = round(self._best * 0.9995, 5)

    def _ratio(self, op: str, level: int) -> float:
        k = (op, int(level))
        if k not in self._cache:
            self._cache[k] = len(self.data) / run_codec(op, int(level), self.data)
        return self._cache[k]

    def best_ratio(self) -> float:
        return self._best

    def ratio_of(self, action: ActionSpec) -> Optional[float]:
        spec = CODECS.get(action.operation)
        if spec is None:
            return None
        lv = action.arguments.get(spec[0])
        if not isinstance(lv, int) or lv not in spec[1]:
            return None
        return self._ratio(action.operation, lv)

    def task(self, objective_id: str) -> TaskSpec:
        return TaskSpec(objective_id=objective_id,
                        objective=f"Find a compressor setting that compresses the corpus by a ratio of at least "
                                  f"{self.target:.3f}.",
                        success={"metric": "codec.best_ratio", "target": self.target}, family="codec",
                        interfaces=[Modality.API, Modality.INTERNAL],
                        limits=Limits(duration_s=2 * DAY, money=self.budget, max_decisions=self.max_decisions),
                        measurement={"source": "CodecVerifier"}, mode="diagnostic")

    # ---- adapter ----
    def _obs(self, kind: str, text: str, q: Sequence[Quantity] = (), parent: str = "") -> Observation:
        now = self.clock.now()
        return Observation(obs_id=new_id("obs"), ts=now, kind=kind, text=text, quantities=list(q),
                           provenance=Provenance(source="codecworld", retrieved_at=now, trust=Trust.UNTRUSTED),
                           parent_action_id=parent)

    def best_so_far(self) -> float:
        return max((r["ratio"] for r in self._runs.values()), default=0.0)

    def get_observable_metrics(self) -> List[Quantity]:
        return [Quantity("codec.best_ratio", round(self.best_so_far(), 4)),
                Quantity("codec.budget.remaining", round(self.budget - self._spent, 4), unit="credits")]

    def observe(self) -> List[Observation]:
        out = []
        if self._first:
            self._first = False
            out.append(self._obs("document", f"a corpus of {len(self.data)} bytes to be archived",
                                 [Quantity("corpus.bytes", float(len(self.data)))]))
        out.append(self._obs("metric", "archive tuning state", self.get_observable_metrics()))
        return out

    def discover_tools(self) -> List[ToolDescriptor]:
        tools = []
        for op, (arg, levels) in CODECS.items():
            tools.append(ToolDescriptor(op, Modality.API, f"Compress the corpus with {op.split('.')[0]} at a chosen "
                                        f"{arg}; reports the compression ratio.",
                                        {"type": "object", "properties": {arg: {"type": "integer",
                                                                                "enum": list(levels)}},
                                         "required": [arg]},
                                        Level.INTERVENTION, "codec.run", "compute", Cost(seconds=60.0, money=1.0)))
        tools.append(ToolDescriptor("corpus.inspect", Modality.API, "Read statistics of a corpus sample.",
                                    {"type": "object", "properties": {}}, Level.INTERVENTION, "codec.read",
                                    "external_read", Cost(seconds=30.0, money=0.1)))
        tools.append(ToolDescriptor("wait", Modality.INTERNAL, "Do nothing until later.",
                                    {"type": "object", "properties": {"seconds": {"type": "number",
                                                                                  "enum": [60, 600]}},
                                     "required": ["seconds"]}, Level.INTERVENTION, "", "none", Cost()))
        return tools

    def validate_action(self, action: ActionSpec) -> List[str]:
        if action.operation in CODECS:
            return [] if self.ratio_of(action) is not None else ["level out of range"]
        if action.operation in ("corpus.inspect",):
            return []
        if action.operation == "wait":
            s = action.arguments.get("seconds")
            return [] if isinstance(s, (int, float)) and s in (60, 600) else ["seconds out of range"]
        return [f"unknown operation {action.operation!r}"]

    def execute(self, action: ActionSpec, authorization: AuthorizationDecision) -> ActionResult:
        if action.action_id in self._results:
            return ActionResult.from_dict(self._results[action.action_id].to_dict())
        now = self.clock.now()
        res = ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=now, ended=now)
        if authorization is None or authorization.verdict != AuthVerdict.ALLOW:
            res.status, res.failure_class = ResultStatus.DENIED, FailureClass.MISSING_PERMISSION
            return res
        errs = self.validate_action(action)
        if errs:
            res.status, res.failure_class, res.failure_detail = (ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                                                                 "; ".join(errs))
            return res
        op, a = action.operation, action.arguments
        cost = 1.0 if op in CODECS else (0.1 if op == "corpus.inspect" else 0.0)
        if self._spent + cost > self.budget + 1e-9:
            res.status, res.failure_class, res.failure_detail = (ResultStatus.FAILED, FailureClass.RESOURCE_LIMIT,
                                                                 "budget exhausted")
            return res
        self._spent += cost
        dur = 60.0 if op in CODECS else (30.0 if op == "corpus.inspect" else float(a.get("seconds", 60.0)))
        obs: List[Observation] = []
        if op in CODECS:
            lv = int(a[CODECS[op][0]])
            out = run_codec(op, lv, self.data)                # the real run, every time
            ratio = len(self.data) / out
            self._runs[action.action_id] = {"op": op, "level": lv, "ratio": ratio, "at": now}
            q = OutcomeQuery(query_id=new_id("q"), metric=f"codec.ratio.from.{action.action_id}",
                             kind=OutcomeKind.REAL, horizon_s=120.0, issued_at=now + dur,
                             description="compression ratio reached by this setting", goal_relevant=True)
            self._pending[q.query_id] = q
            res.expected_outcomes.append(q)
            obs.append(self._obs("tool_result", f"{op} at {CODECS[op][0]} {lv}: {out} bytes",
                                 [Quantity("codec.ratio", round(ratio, 5)), Quantity("codec.bytes", float(out))],
                                 action.action_id))
        elif op == "corpus.inspect":
            sample = self.data[: 4096]
            hist: Dict[int, int] = {}
            for b in sample:
                hist[b] = hist.get(b, 0) + 1
            import math
            ent = -sum(v / len(sample) * math.log2(v / len(sample)) for v in hist.values())
            obs.append(self._obs("tool_result", "corpus sample statistics",
                                 [Quantity("corpus.sample_entropy_bits", round(ent, 4)),
                                  Quantity("corpus.sample_distinct", float(len(hist)))], action.action_id))
        else:
            obs.append(self._obs("tool_result", f"waited {dur:.0f} seconds", [], action.action_id))
        self.clock.advance(dur)
        res.ended = self.clock.now()
        res.observations = obs
        res.cost = Cost(seconds=dur, money=cost, primitive_ops=1)
        res.ops = [PrimitiveOp(op=op, modality=action.modality, arguments=dict(a), started=now, ended=res.ended)]
        res.verified = True
        self._results[action.action_id] = ActionResult.from_dict(res.to_dict())
        return res

    def poll_events(self, cursor: str) -> Tuple[List[Observation], str]:
        return [], cursor or "0"

    def get_resource_state(self) -> Dict[str, Any]:
        return {"money_spent": self._spent, "money_remaining": self.budget - self._spent,
                "time_elapsed_s": self.clock.now() - self.t0, "external_actions": len(self._runs)}

    def get_pending_outcomes(self) -> List[OutcomeQuery]:
        return list(self._pending.values())

    def get_capabilities(self) -> Dict[str, Any]:
        return {"name": "codecworld", "interface_version": self.interface_version, "provenance": "diagnostic",
                "modalities": [Modality.API, Modality.INTERNAL], "simulated_time": True, "live": False,
                "features": {"at_most_once_by_action_id": True}}

    def measure(self, queries: Sequence[OutcomeQuery], *, _verified: bool = False) -> List[OutcomeObservation]:
        now, out = self.clock.now(), []
        for q in queries:
            if now < q.due_at:
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.PENDING, source="codecworld"))
                continue
            aid = q.metric[len("codec.ratio.from."):] if q.metric.startswith("codec.ratio.from.") else ""
            run = self._runs.get(aid)
            if run is None:
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.UNOBSERVABLE, source="codecworld"))
                continue
            v = len(self.data) / run_codec(run["op"], run["level"], self.data) if _verified else run["ratio"]
            self._pending.pop(q.query_id, None)
            out.append(OutcomeObservation(q.query_id, OutcomeStatus.OBSERVED, float(v), observed_at=now,
                                          verified=_verified, source="codecworld",
                                          dedup_key=f"{q.metric}@{q.due_at:.0f}"))
        return out


class CodecVerifier:
    """Independent verifier: re-runs the compressor on every recorded setting and signs the best ratio."""

    def __init__(self, world: CodecWorld, secret: bytes = b"codec-verifier-key"):
        self._w, self._secret = world, secret

    def _sign(self, o: Any) -> str:
        d = o.to_dict()
        d.pop("signature", None)
        return hmac.new(self._secret, json.dumps(d, sort_keys=True).encode(), hashlib.sha256).hexdigest()

    def verify(self, task: TaskSpec) -> VerifiedOutcome:
        w, now = self._w, self._w.clock.now()
        best = max((len(w.data) / run_codec(r["op"], r["level"], w.data) for r in w._runs.values()), default=0.0)
        target = float(task.success.get("target", w.target))
        if best >= target:
            status = GoalStatus.ACHIEVED
        elif w._spent + 1.0 > w.budget + 1e-9:
            status = GoalStatus.FAILED
        else:
            status = GoalStatus.PENDING
        o = VerifiedOutcome(objective_id=task.objective_id, status=status, value=float(best), target=target,
                            evaluated_at=now, notes="codecworld: independent re-run of every recorded setting")
        o.signature = self._sign(o)
        return o

    def check_signature(self, outcome: VerifiedOutcome) -> bool:
        return (isinstance(outcome, VerifiedOutcome) and bool(outcome.signature)
                and hmac.compare_digest(outcome.signature, self._sign(outcome)))

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        out = self._w.measure(queries, _verified=True)
        for o in out:
            if o.status == OutcomeStatus.OBSERVED:
                o.source = "verifier:codecworld"
                o.signature = self._sign(o)
        return out

    def check_measurement(self, obs: OutcomeObservation) -> bool:
        return (isinstance(obs, OutcomeObservation) and bool(obs.signature) and obs.verified
                and hmac.compare_digest(obs.signature, self._sign(obs)))
