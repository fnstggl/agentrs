"""Task protocols: a declared world distribution with splits, an outcome-only goal with an independent evaluator, a
horizon, thresholds, scripted controls and the list of what the environment does not support.

`TaskProtocol.calibrate` checks feasibility the way spec Part 16 asks: every control runs on the same world in its
own **ephemeral fork** (one base world per task, `open_world(fork_identical=True)`), through the same
`OpenWorldAdapter` records a learned agent gets, under the same SMDP clock, and is judged by the same evaluator.
Controls have roles:

* ``ceiling``   privileged scripted builder (hidden reference solvers, e.g. Crucible's `Builder`): shows the goal is
                reachable under the world's laws; never a fair demonstration, never training data;
* ``control``   non-privileged competent control (public information only, e.g. `ExampleInductor`): shows the goal
                is reachable without hidden knowledge;
* ``shortcut``  idle / random / garbage policies: must FAIL, otherwise the task is trivial or exploitable.

A task is *feasible* when a ceiling or control succeeds and no shortcut does; *fairly feasible* when a
non-privileged control succeeds. The record keeps evaluator-only details (integrity report, verifier state) for the
experimenter; nothing of it is a learner input.

Feasibility requirements travel **with the task**: `TaskProtocol.make_task` returns tasks carrying the seat the
actor must hold (``actor_handle``, e.g. OW-3's engineer ``eng``), the independently controlled participants the
world needs (``participant_factory``, e.g. OW-7's partner) and the declared split of their (family, seed); and
`open_world` honours the first two. `TaskProtocol.open(split, i)` builds (adapter, operator, evaluator) for one
declared task, in-process or process-isolated (learner runs).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

DAY = 86400


@dataclass
class Control:
    name: str
    factory: Callable[[], Any]         # -> a crucible agents.base.Policy
    role: str                          # ceiling | control | shortcut
    privileged: bool = False
    idle_wait_s: int = 3000
    note: str = ""


@dataclass
class TaskProtocol:
    name: str
    title: str
    spec_ref: str
    make_task: Callable[[str, int], Any]          # (world family, seed) -> GoalContract-like task
    splits: dict                                  # {"train" | "val" | "test" | "dev": [(world family, seed), ...]}
    horizon_days: float
    thresholds: dict = field(default_factory=dict)
    controls: list = field(default_factory=list)
    handles: Optional[Sequence[str]] = None
    participants: Optional[Callable[[], list]] = None
    unsupported: list = field(default_factory=list)
    notes: str = ""
    arms: list = field(default_factory=list)      # comparison conditions the experiment runs (spec / Part 19)

    def __post_init__(self):
        raw = self.make_task
        if getattr(raw, "_cam_protocol_task", False):
            return

        def make_task(world_family: str, seed: int, *a, _raw=raw, **kw):
            t = _raw(world_family, seed, *a, **kw)
            if self.handles:
                t.actor_handle = self.handles[0]
            if self.participants is not None:
                t.participant_factory = self.participants
            sp = self.split_of(world_family, seed)
            if sp is not None:
                t.split = sp
            return t
        make_task._cam_protocol_task = True
        make_task.__doc__ = raw.__doc__
        self.make_task = make_task

    def split_of(self, world_family: str, seed: int) -> Optional[str]:
        """The declared split of one (world family, seed), or None when it is not declared."""
        for sp, rows in self.splits.items():
            if (world_family, int(seed)) in [(f, int(x)) for f, x in rows]:
                return sp
        return None

    def tasks(self, split: str = "train") -> list:
        return [self.make_task(f, s) for f, s in self.splits.get(split, [])]

    def split_declarations(self) -> dict:
        """{world family: split} for `worldgen.SplitGuard(extra=...)`; a family declared in several splits gets
        the most restrictive one (test > val > train/dev)."""
        rank = {"test": 3, "heldout": 3, "val": 2, "train": 1, "dev": 0}
        out: dict = {}
        for sp, rows in self.splits.items():
            for f, _ in rows:
                if f not in out or rank.get(sp, 3) > rank.get(out[f], 3):
                    out[f] = sp
        return out

    def pair_declarations(self) -> dict:
        """{(world family, seed): split} for `worldgen.SplitGuard(extra_pairs=...)`."""
        return {(f, int(x)): sp for sp, rows in self.splits.items() for f, x in rows}

    def open(self, split: str = "train", i: int = 0, *, isolated: bool = False, evaluator: Optional[dict] = None,
             **kw) -> tuple:
        """(adapter, operator, evaluator) for the i-th declared task of `split`, with the seat and participants the
        task needs. ``isolated=True`` (learner runs) puts worlds, operator and evaluator in a spawned process and
        returns (adapter, RemoteOperator, RemoteEvaluator). `evaluator`: OpenWorldEvaluator keyword arguments."""
        fam, seed = self.splits[split][i]
        task = self.make_task(fam, seed)
        if isolated:
            from ..openworld import open_world_isolated
            return open_world_isolated([task], evaluator=dict(evaluator or {}), **kw)
        from ..openworld import open_world
        from ..openworld_eval import OpenWorldEvaluator
        ad, op = open_world([task], **kw)
        return ad, op, OpenWorldEvaluator(op, **(evaluator or {}))

    def describe(self) -> dict:
        return {"name": self.name, "title": self.title, "spec_ref": self.spec_ref, "horizon_days": self.horizon_days,
                "thresholds": self.thresholds, "splits": self.splits,
                "controls": [{"name": c.name, "role": c.role, "privileged": c.privileged, "note": c.note}
                             for c in self.controls],
                "unsupported": self.unsupported, "notes": self.notes, "arms": self.arms}

    def calibrate(self, split: str = "train", n_tasks: int = 1, controls: Optional[Sequence[str]] = None,
                  days: Optional[float] = None, think_s: float = 600.0, record: Optional[str] = None,
                  stop_on_success: bool = False, seed: int = 0) -> dict:
        """Run the controls in forks of each of the first `n_tasks` worlds of `split` and return the feasibility
        record (also appended as one JSON line to `record`)."""
        from ..agent_protocol import PolicyAgent
        from ..harness import run_lifetime
        from ..openworld import open_world
        from ..openworld_eval import OpenWorldEvaluator
        ctrls = [c for c in self.controls if controls is None or c.name in controls]
        if not ctrls:
            raise ValueError("no controls selected")
        out_tasks = []
        for (fam, s) in self.splits[split][:n_tasks]:
            task = self.make_task(fam, s)       # carries its seat and participants (one actor per fork)
            t0 = time.time()
            ad, op = open_world([task] * len(ctrls), fork_identical=True, seed=seed)
            ev = OpenWorldEvaluator(op, integrity=True)
            agent = PolicyAgent(lambda b: ctrls[b].factory(), idle_wait_s=ctrls[0].idle_wait_s)
            try:
                res = run_lifetime(agent, ad, ev, think_s=think_s, days=days, stop_on_success=stop_on_success)
            finally:
                ad.close()
            rows = []
            for b, c in enumerate(ctrls):
                f = res.final[b]
                det = f.get("details", {})
                integ = det.get("integrity") or {}
                rows.append({"control": c.name, "role": c.role, "privileged": c.privileged,
                             "achieved": bool(f["achieved"]), "success_day": res.success_day[b],
                             "score": f["score"], "metrics": f["metrics"], "return": res.returns[b],
                             "elapsed_days": res.elapsed_days[b],
                             "integrity_flags": {k: v for k, v in integ.items() if isinstance(v, (bool, int, float))}
                             if isinstance(integ, dict) else {}})
            ok_ceiling = any(r["achieved"] for r in rows if r["role"] in ("ceiling", "control"))
            ok_fair = any(r["achieved"] for r in rows if r["role"] == "control")
            short = [r["control"] for r in rows if r["role"] == "shortcut" and r["achieved"]]
            out_tasks.append({"world_family": fam, "seed": s, "split": split, "controls": rows,
                              "feasible": bool(ok_ceiling and not short), "fairly_feasible": bool(ok_fair and not short),
                              "shortcuts_succeeded": short, "wall_s": round(time.time() - t0, 2),
                              "decisions": res.steps, "accounting": {k: res.accounting[k] for k in
                                                                     ("wall_s", "cpu_s", "decisions", "actions",
                                                                      "rejected", "phase_wall_s")}})
        rec = {"protocol": self.name, "title": self.title, "split": split, "think_s": think_s,
               "days": days, "tasks": out_tasks, "unsupported": self.unsupported,
               "t_wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if record:
            os.makedirs(os.path.dirname(os.path.abspath(record)), exist_ok=True)
            with open(record, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        return rec


def idle_control() -> Control:
    from crucible.agents.scripted import IdlePolicy
    return Control("idle", IdlePolicy, "shortcut", note="does nothing; must fail")


def random_control() -> Control:
    from crucible.agents.scripted import RandomPolicy
    return Control("random", RandomPolicy, "shortcut", note="uniformly random tool with schema-valid arguments")


def builder_control(**kw) -> Control:
    from crucible.agents.scripted import Builder
    return Control("builder", lambda: Builder(**kw), "ceiling", privileged=True,
                   note="deploys the hidden reference solver of the weakest-incumbent need (privileged)")


def inductor_control(name: str = "example_inductor", **kw) -> Control:
    from crucible.usefulness.agents import ExampleInductor
    return Control(name, lambda: ExampleInductor(**kw), "control",
                   note="induces a wire format from public worked examples, deploys, lists (non-privileged)"
                        + (f"; {kw}" if kw else ""))


def garbage_control(**kw) -> Control:
    from crucible.usefulness.agents import GarbageSeller
    return Control("garbage_seller", lambda: GarbageSeller(**kw), "shortcut",
                   note="well-formed worthless service behind an attractive listing; must fail")


# Environment gaps that hold for every protocol (spec 16, "Important environment limitation")
COMMON_UNSUPPORTED = [
    "No social-media / viral-video / community dynamics: Crucible 4.0 has no browsing population (simulated people "
    "do not visit websites) and no content virality mechanism; no OW task claims to test them.",
    "Free text never changes NPC decisions (minds negotiate with reservation logic; optional LLM minds only draft "
    "replies), so persuasion and community building are not measurable.",
    "Simulated dynamics are plausible hypotheses, not calibrated to the real world (Crucible fidelity register).",
]
