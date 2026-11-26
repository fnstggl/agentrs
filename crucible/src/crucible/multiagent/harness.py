"""Civilisation self-play matches: many participants of many kinds in one shared world (MASTER_SPEC Part 12).

A `Match` is one world - a Tier 3 civilisation (`crucible.civ`, through `crucible.experience.registry.civ_world`)
or, for unit tests, any kernel world - in observation-only mode, plus a roster of participants admitted as
ordinary actors (`admin.join`, each with the `controller` of the operator that controls it, so the evaluator's
related-party analysis knows which identities are one party), all driven together by the multi-agent driver of
`crucible.experience` (`run_learners`: every participant acts on its own observation, the actions are executed,
then time passes once). The participants are

* identities of neural `PolicyGroup`s (`crucible.multiagent.policy`) in the roles *current* (learning), *historical*
  (a frozen earlier version of a current group), *independent* (separately updated groups with unrelated
  objectives) and *heldout* (frozen policies of a family never met in training, for evaluation);
* semantic generalists (`SemanticAgent`, class 4);
* and, inside the world, the civilisation's own population: statistical behavioural individuals (class 1),
  compact adaptive neural individuals (class 2), strategic neural groups (class 3), incumbent firms.

Nothing is shared between participants but the world. The harness is the operator: it admits identities, runs
time, freezes versions into the registry, and records composition, actions and evaluator outcomes
(`crucible.multiagent.evaluate`) - none of which any participant can read.
"""
from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from ..experience.loop import run_learners
from ..experience.modes import OBSERVATION_ONLY
from .agents import LearnerAgent, SemanticAgent
from .objectives import Objective
from .policy import PolicyGroup
from .registry import PolicyRegistry

ROLES = ("current", "historical", "independent", "heldout", "semantic")
HOUR = 3600
DAY = 86400
CPU_EXTRA = ("products", "subs", "pkg", "inst", "xchg", "ventures", "labor")


@dataclass
class Member:
    handle: str
    role: str
    controller: str
    group: str | None = None          # policy group id (learners)
    drive: str = "profit"
    endowment: int = 2_000_000
    goal: str | None = None           # semantic participants
    every: int = 4                    # semantic participants: decide every n-th step

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")


# ====================================================================== worlds
def civ_env(profile: str = "balanced", seed: int = 0, *, population: str = "development",
            extra_modules: Iterable[str] = (), store=None):
    """An observation-only Tier 3 civilisation behind `KernelExperienceEnv` (no participants yet)."""
    from ..experience.kernel import KernelExperienceEnv
    from ..experience.registry import LEARNER_MODULES, civ_world
    world = civ_world(profile, int(seed), population=population, mode=OBSERVATION_ONLY,
                      learner_modules=tuple(LEARNER_MODULES) + tuple(extra_modules), store=store)
    return KernelExperienceEnv(world, int(seed), agents=[], reward_mode=OBSERVATION_ONLY, tier=3,
                               name=f"civ:{profile}", semantic=False)


def kernel_env(family: str = "open_digital", seed: int = 0, *, extra_modules: Iterable[str] = CPU_EXTRA):
    """A 4.0 kernel world family with the v5 commerce/software modules added (CPU unit tests; not a civilisation)."""
    from ..experience.kernel import KernelExperienceEnv
    return KernelExperienceEnv(family, int(seed), agents=[], reward_mode=OBSERVATION_ONLY,
                               extra_modules=list(extra_modules), semantic=False)


def is_civilisation(world) -> bool:
    return str(world.spec.family).startswith("civ_") and "pop" in world.modules


# ====================================================================== the match
class Match:
    """One self-play match (see the module docstring)."""

    def __init__(self, env, groups: dict[str, PolicyGroup], members: list[Member], *,
                 registry: PolicyRegistry | None = None, match_id: str | None = None, think_s: int = HOUR,
                 semantic_service=None, store_root: str | None = None, seed: int = 0, wrap=None):
        self.env, self.world = env, env.world
        self.groups = dict(groups)
        self.registry = registry if registry is not None else PolicyRegistry()
        self.match_id = match_id or f"m-{self.world.world_id}"
        self.think_s = int(think_s)
        self.seed = int(seed)
        self.t0, self.seq0 = int(self.world.now), int(self.world.seq)
        self.agents: dict[str, Any] = {}
        self.members: dict[str, Member] = {}
        self.endowments: dict[str, int] = {}
        self.service = semantic_service
        self.wrap = wrap                     # operator-side instrument around each participant (audits), or None
        self._sem_group = None
        self.store = None
        if store_root:
            from ..experience.store import ExperienceStore
            os.makedirs(store_root, exist_ok=True)
            self.store = ExperienceStore(os.path.join(store_root, self.match_id))
        self.windows: list[dict] = []
        self.summaries: list[dict] = []
        self.wall_s = 0.0
        for gid, g in self.groups.items():
            if self.registry.vid(g) not in self.registry.versions:
                self.registry.register_group(g, world_id=self.world.world_id, seed=self.seed)
            else:
                g.on_update = self.registry.on_update
                g._registry_world = (self.world.world_id, self.seed)
        self.roster_versions: dict[str, str] = {}
        for m in members:
            self.admit(m)

    # ------------------------------------------------------------------ admission
    def admit(self, m: Member) -> str:
        w = self.world
        p = w.system_call("admin.join", {"handle": m.handle, "endowment": int(m.endowment),
                                         "controller": m.controller})["principal"]
        self.env.add_agent(p)
        if m.role == "semantic":
            agent = SemanticAgent(p, self._semantic_group().participant(goal=m.goal or "run a useful business"),
                                  every=m.every, seed=self.seed)
            vid = agent.info()["name"]
        else:
            g = self.groups[m.group]
            agent = LearnerAgent(p, g, Objective(m.drive), role=m.role)
            vid = self.registry.vid(g)
        self.agents[p] = agent if self.wrap is None else self.wrap(agent, p, m)
        self.members[p] = m
        self.endowments[p] = int(m.endowment)
        self.roster_versions[p] = vid
        self.registry.add_identity(p, handle=m.handle, controller=m.controller, group=m.group, role=m.role,
                                   objective=(m.drive if m.role != "semantic" else m.goal),
                                   world_id=w.world_id, admitted_t=int(w.now), version=vid, match=self.match_id)
        return p

    def _semantic_group(self):
        if self._sem_group is None:
            if self.service is None:
                raise RuntimeError("semantic members need a SemanticService (crucible.semantic.service)")
            from ..semantic.participants import SemanticGroup
            self._sem_group = SemanticGroup(self.service, world=self.world, record=True)
        return self._sem_group

    # ------------------------------------------------------------------ running
    def run(self, hours: float, *, chunk_hours: float = 6, learn: bool = True) -> dict:
        steps_total = int(round(hours * HOUR / self.think_s))
        per = max(1, int(round(chunk_hours * HOUR / self.think_s)))
        done = 0
        t_wall = time.perf_counter()
        while done < steps_total:
            n = min(per, steps_total - done)
            ta, seqa = int(self.world.now), int(self.world.seq)
            out = run_learners(self.env, self.agents, n, think_s=self.think_s, update_every=1, learn=learn,
                               store=self.store, check_schema=True)
            self.summaries.append({k: v for k, v in out.items() if k != "verification"})
            self.windows.append(self.composition(ta, int(self.world.now), seqa))
            done += n
        self.wall_s += time.perf_counter() - t_wall
        return self.totals()

    def totals(self) -> dict:
        agg: dict[str, Counter] = {}
        for s in self.summaries:
            for a, r in s["agents"].items():
                c = agg.setdefault(a, Counter())
                for k in ("steps", "actions", "ok", "rejected", "schema_failures", "nonzero_rewards"):
                    c[k] += int(r.get(k, 0))
        return {a: dict(c) for a, c in agg.items()}

    # ------------------------------------------------------------------ composition (operator side)
    def composition(self, ta: int, tb: int, seqa: int) -> dict:
        """Which kinds of participants were present and acted in [ta, tb): admitted identities by role (acted =
        at least one accepted state-changing action), and the civilisation's population classes (acted = an
        individual's last activity falls in the window)."""
        kinds: dict[str, dict] = {}
        for p, m in self.members.items():
            a = self.agents[p]
            acted = any(h.get("ok") and not h.get("read_only") and ta <= int(h.get("t", -1)) <= tb + self.think_s
                        for h in getattr(a, "history", []))
            role = m.role
            d = kinds.setdefault(role, {"present": 0, "acting": 0, "groups": set()})
            d["present"] += 1
            d["acting"] += int(acted)
            d["groups"].add(m.group or "semantic")
        pop = population_activity(self.world, ta, tb)
        for name, (present, acting) in pop.items():
            kinds[name] = {"present": present, "acting": acting, "groups": {name}}
        snap = {"match": self.match_id, "t0": ta, "t1": tb,
                "kinds": {k: {"present": v["present"], "acting": v["acting"], "groups": sorted(v["groups"])}
                          for k, v in kinds.items()},
                "versions": {p: self.registry.vid(self.agents[p].group) if hasattr(self.agents[p], "group")
                             else self.roster_versions[p] for p in self.agents}}
        self.registry.record_composition(snap)
        return snap

    # ------------------------------------------------------------------ end
    def finish(self) -> dict:
        from .evaluate import match_outcomes
        outs = match_outcomes(self)
        self.registry.record_outcomes(self.match_id, outs)
        for p, a in self.agents.items():
            self.registry.record_actions(p, getattr(a, "history", []))
        if self.store is not None:
            self.store.flush()
            self.registry.record_experience(self.store.root, self.world.world_id, self.store.digest(),
                                            len(self.store))
        rec = {"match": self.match_id, "world_id": self.world.world_id, "family": self.world.spec.family,
               "seed": int(self.world.spec.seed), "spec_hash": self.world.spec.hash(), "t0": self.t0,
               "t1": int(self.world.now), "think_s": self.think_s, "roster": dict(self.roster_versions),
               "members": {p: asdict(m) for p, m in self.members.items()}, "digest": self.world.digest(),
               "events": int(self.world.seq), "wall_s": round(self.wall_s, 2)}
        self.registry.record_match(rec)
        return outs

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
        self.env.close()


def population_activity(world, ta: int, tb: int) -> dict[str, tuple[int, int]]:
    """Population classes present and acting in [ta, tb) (evaluator side; read from the population's columns)."""
    if "pop" not in world.modules:
        return {}
    from ..population import core
    from ..substrate import columns
    out = {}
    last = columns.col(world, core.TABLE, "last_activity")
    names = {1: "statistical", 2: "compact_neural", 3: "strategic_neural"}
    for cls, name in names.items():
        idx = core.class_members(world, cls)
        if idx.size == 0:
            continue
        la = np.asarray(last[idx], dtype=np.int64)
        out[name] = (int(idx.size), int(((la >= ta) & (la <= tb)).sum()))
    return out
