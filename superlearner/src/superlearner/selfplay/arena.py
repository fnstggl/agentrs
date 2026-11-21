"""Several CAM learners in one persistent Crucible world.

Each learner is an ordinary actor of the same world with its own controller weights, learning algorithm, episodic
memory, tool library, endowment, goal contract and evaluator. They meet only through world mechanics: the same
consumers, compute providers, market listings and search index. What one of them builds or prices changes the
problem the others face.

Time. The world is persistent and never reset. In every round each awake learner takes one decision (its commands
execute at the current time), then the world advances by `round_s`. A learner that chose to wait sleeps until its
wake-up time. There are no episodes; for learning, every `segment` decisions of a learner form a truncated
trajectory (bootstrapped from its own critic) on which its algorithm takes an update, so different learners can
learn with different algorithms inside the same world.

Controls. A participant may be a scripted policy or idle. `counterfactual` re-runs the same world and seeds with
one participant kept idle and reports how the others' verified outcomes change: the measured effect of that
participant's presence on everybody else. Nothing rewards a particular invention; rewards are each learner's own
goal contract on authoritative state.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .. import ensure_crucible
from ..cam.agent import AgentConfig, CAMAgent, Chooser, ModelChooser
from ..cam.memory import EpisodicMemory
from ..cam.model import CAMController, EmbStore, ModelConfig
from ..grounding.toolsmith import ToolLibrary
from ..world.core import CoreConfig, Evaluator
from ..world.tasks import TaskSpec, build_goal, run_prefix

ensure_crucible()
from crucible.goals.metrics import evaluate_metric  # noqa: E402
from crucible.integrations import ActorView, agent_tools  # noqa: E402

DAY = 86400


class SharedCore:
    """The session protocol for one actor of a shared world (trusted side). The clock belongs to the arena."""

    def __init__(self, world, actor: str, goal, cfg: CoreConfig, start_info: dict):
        self.world, self.actor, self.goal, self.cfg = world, actor, goal, cfg
        self.view = ActorView(world, actor, cfg.max_events)
        self.evaluator = Evaluator(goal, actor, cfg)
        self.evaluator.begin(world)
        self.start_info = start_info
        self.t_start = world.now
        self.wake_at = world.now
        self.n_decisions = self.n_commands = self.n_ok = 0
        self.commands: list = []
        self._sig = None

    def info(self) -> dict:
        hist = [{"tool": a["tool"], "args": a.get("args", {}), "result": r} for a, r in self.start_info.get("history", [])]
        return {"goal": self.goal.description, "actor": self.actor, "horizon_s": self.goal.horizon_s, "time": self.world.now,
                "start": self.start_info.get("stage", "scratch"), "start_history": hist, "max_decisions": 10**6,
                "task_family": self.goal.family.replace("sl.", ""), "world_family": self.goal.world["family"], "split": "train"}

    def observe(self) -> dict:
        return self.view.observe()

    def tools(self, force: bool = False):
        tl = agent_tools(self.world, self.actor)
        sig = (len(tl), tl[-1]["name"] if tl else "")
        if not force and sig == self._sig:
            return None
        self._sig = sig
        return tl

    def execute(self, tool, args) -> dict:
        r = self.world.execute(self.actor, {"tool": tool, "args": args})
        self.n_commands += 1
        self.n_ok += int(r.ok)
        self.commands.append((self.world.now, r.tool, bool(r.ok)))
        return r.to_dict()

    def wait(self, seconds: int) -> int:
        s = int(min(max(1, int(seconds)), self.cfg.max_wait_s))
        self.wake_at = self.world.now + s
        return s

    def end_decision(self) -> dict:
        self.n_decisions += 1
        r = self.evaluator.step(self.world)
        el = self.world.now - self.t_start
        return {"reward": r, "terminated": False, "truncated": False, "time": self.world.now,
                "elapsed_frac": min(1.0, el / max(1, self.goal.horizon_s)), "decisions": self.n_decisions}

    def summary(self) -> dict:
        s = self.evaluator.status(self.world)
        return {"task": f"arena:{self.actor}", "arm": [self.goal.family.replace("sl.", ""), 0, "arena"], "split": "train",
                "world_family": self.goal.world["family"], "success": bool(s["success"]), "progress": float(s["progress"]),
                "best_progress": float(s["best_progress"]), "return": float(s["return"]), "elapsed_s": int(s["elapsed_s"]),
                "decisions": self.n_decisions, "commands": self.n_commands, "commands_ok": self.n_ok, "world_errors": {},
                "sim_wall_s": 0.0, "build_wall_s": 0.0, "world_events": self.world.seq}


class IdleChooser(Chooser):
    name = "idle"

    def head(self, ag, items, logits):
        return next(i for i, m in enumerate(items.meta) if m[0] == "wait" and m[1] == DAY)

    def option(self, ag, tool, slot, items, logits):
        return 0


def outcomes(world, actor: str) -> dict:
    """Verified outcomes of one participant, from authoritative state."""
    me = {"principal": actor}
    from ..evalkit.metrics import owned_orgs
    orgs = owned_orgs(world, actor)
    st = world.state
    hosts = [d["host"] for _, d in st.scan("deployments") if d["owner"] in orgs] if "deployments" in st.tables else []
    served = sum(r["ok"] for k, r in st.scan("svc_days") if k.split("|")[0] in hosts) if "svc_days" in st.tables else 0
    rev = sum(evaluate_metric(world, "independent_revenue", {"principal": o, "window_s": 60 * DAY}) for o in orgs)
    cash = evaluate_metric(world, "cash", me) + sum(evaluate_metric(world, "cash", {"principal": o}) for o in orgs)
    listings = [(l["need"], l["price"]) for _, l in st.scan("listings") if l["seller"] in orgs and l["status"] == "active"] if "listings" in st.tables else []
    return {"companies": evaluate_metric(world, "sl_companies", me), "live_services": evaluate_metric(world, "sl_live_services", me),
            "listed_services": evaluate_metric(world, "sl_listed_services", me), "requests_served": int(served),
            "independent_revenue": float(rev), "cash": float(cash), "listings": listings}


def market(world) -> dict:
    """Active API listings per need: number of sellers and the lowest price (the shared problem everybody faces)."""
    st = world.state
    out: dict = {}
    if "listings" in st.tables:
        for _, l in st.scan("listings"):
            if l["status"] == "active" and l["kind"] == "api" and l.get("need"):
                m = out.setdefault(l["need"], {"sellers": 0, "min_price": None})
                m["sellers"] += 1
                m["min_price"] = l["price"] if m["min_price"] is None else min(m["min_price"], l["price"])
    return out


class Arena:
    def __init__(self, world_family: str, seed: int, participants: list[dict], round_s: int = 1800, core: dict | None = None):
        """`participants`: [{"handle", "goal": task family, "start": start state, "endowment", "need"}]."""
        from crucible.core.kernel import World
        from crucible.worldgen import generate
        self.world = World.create(generate({"family": world_family}, int(seed)), world_id=f"arena-{world_family}-{seed}")
        self.round_s = round_s
        self.cores: dict[str, SharedCore] = {}
        self.spec = {"world_family": world_family, "seed": int(seed), "participants": copy.deepcopy(participants)}
        cfg = CoreConfig(**(core or {}))
        for p in participants:
            goal = build_goal(TaskSpec(p.get("goal", "service"), world_family, 10_000_000 + int(seed), int(p.get("level", 0))))
            goal = copy.deepcopy(goal)
            goal.horizon_s = int(p.get("horizon_days", 30) * DAY)
            r = self.world.system_call("admin.join", {"handle": p["handle"], "endowment": int(p.get("endowment", goal.endowment)), "kind": "agent"})
            actor = r["principal"]
            info = {"stage": "scratch", "history": []}
            if p.get("start", "scratch") != "scratch":
                info = run_prefix(self.world, actor, p["start"], p.get("need"))
            self.cores[actor] = SharedCore(self.world, actor, goal, cfg, info)
        self.log: list[dict] = []

    def run(self, players: dict, days: float, seed: int = 0, on_segment=None, segment: int = 24, snapshot_every_days: float = 1.0) -> dict:
        """`players`: actor -> (CAMAgent, Chooser). Returns per-participant outcomes and the market history."""
        rng = np.random.default_rng(seed)
        t_end = self.world.now + int(days * DAY)
        order = sorted(players)
        for a in order:
            players[a][0].begin(self.cores[a], f"arena-{a}", learn=True)
        seg_start = {a: 0 for a in order}
        next_snap = self.world.now
        t0 = time.time()
        while self.world.now < t_end:
            if self.world.now >= next_snap:
                self.log.append({"t": self.world.now, "day": round((self.world.now) / DAY, 2), "market": market(self.world),
                                 "outcomes": {a: outcomes(self.world, a) for a in order}})
                next_snap += int(snapshot_every_days * DAY)
            for i in rng.permutation(len(order)):
                a = order[int(i)]
                ag, ch = players[a]
                if self.cores[a].wake_at > self.world.now:
                    continue
                ag.step(ch)
                if on_segment is not None and len(ag.decisions) - seg_start[a] >= segment:
                    seg = ag.decisions[seg_start[a]:]
                    seg_start[a] = len(ag.decisions)
                    on_segment(a, ag, seg)
            self.world.advance_time(self.world.now + self.round_s)
        self.log.append({"t": self.world.now, "day": round(self.world.now / DAY, 2), "market": market(self.world),
                         "outcomes": {a: outcomes(self.world, a) for a in order}})
        res = {"spec": self.spec, "days": days, "wall_s": round(time.time() - t0, 1), "final": self.log[-1], "history": self.log,
               "agents": {a: {"decisions": len(players[a][0].decisions), "valid": players[a][0].exec.stats(),
                              "tools_made": players[a][0].library.names(), "chooser": players[a][1].name,
                              "evaluator": self.cores[a].evaluator.status(self.world),
                              "heads": _count([d["kind"] for d in players[a][0].decisions]),
                              "tools_called": _count([d["tool"] for d in players[a][0].decisions if d["kind"] == "tool"])}
                          for a in order},
               "invariants": self.world.check_invariants()[:3], "events": self.world.seq}
        return res


def _count(xs: list) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1])[:12])


def make_learner(ckpt: dict | None, llm, algo_name: str = "ppo", algo_cfg: dict | None = None, agent_cfg: dict | None = None,
                 seed: int = 0, device: str = "cpu"):
    """A CAM participant with its own weights, optimiser, memory and tools (optionally from a checkpoint)."""
    import torch
    from ..algo.base import make
    torch.manual_seed(seed)
    model = CAMController(ModelConfig(**((ckpt or {}).get("model_cfg") or {}))).to(device)
    store = EmbStore()
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        for k, row in zip(ckpt["store_keys"], ckpt["store_mat"]):
            if int(k):
                store.put(int(k), row)
    model.eval()
    mem, lib = EpisodicMemory(), ToolLibrary()
    ag = CAMAgent(model, store, llm, AgentConfig(**(agent_cfg or {})), lib, mem, device)
    algo = make(algo_name, model, dict(algo_cfg or {}, seed=seed), device)
    return ag, algo, ModelChooser(False, 1.0, seed)


def online_update(algo, store):
    """`on_segment` callback factory: a truncated-trajectory update of one learner's own algorithm."""
    stats: dict = {}

    def cb(actor, ag, seg):
        seg = [dict(d) for d in seg]
        seg[-1]["trunc"] = True
        ep = {"decisions": seg, "summary": {"success": False}}
        m = algo[actor].update([ep], ag.store)
        stats.setdefault(actor, []).append({k: m.get(k) for k in ("kl", "entropy", "param_delta", "v_loss", "grad_steps")})
    cb.stats = stats
    return cb
