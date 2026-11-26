"""A civilisation self-play campaign end to end (what the Gate Q and SP-12 tests run).

1. **Demonstrations** (`demonstrations`): SCRIPTED demonstrator identities (`crucible.multiagent.demonstrator`,
   labelled scripted) of several controllers act in CPU training worlds (a 4.0 open-economy family with the v5
   commerce and software modules - never the civilisations the policies are evaluated in). Their decisions are
   recorded in the policy's own candidate format.
2. **Pre-training** (`pretrain`): every policy group is initialised independently (its own seed) and trained by
   behaviour cloning (its own minibatch order) on the demonstration corpus - shared by default, like a common
   pre-training corpus, or dealt out disjointly; the training-data provenance (worlds, identities, decision times)
   is registered as the group's first versions. Everything after that - the self-play experience - is each
   group's own.
3. **Held-out family** (`heldout_family`): a recurrent (GRU) PPO policy pre-trained on demonstrations from other
   worlds and then trained by self-play among its own identities in yet another world - an architecture and
   algorithm no training group has, never admitted to the training civilisation: an unfamiliar opponent.
4. **Civilisation self-play** (`civ_run`): the learning groups (shared-training group of several identities,
   independently updated groups with other algorithms and unrelated objectives), semantic generalists and, from
   half-time, a frozen historical checkpoint of the main group act in one Tier 3 civilisation with its whole
   population; every learning group also feeds a *no-signal control* twin (same batches, signal 0).
5. **Strategy switch** (`strategy_switch`): the world and every participant are forked; in one branch a competitor
   group's policy is replaced, in two others nothing changes; the main group's decisions are compared across the
   branches (it responds to the competitor's strategy exactly when its decisions differ from the unchanged branches,
   which must agree with each other).
"""
from __future__ import annotations

import copy
import time
from collections import Counter
from typing import Any

import numpy as np

from . import analysis as AN
from .agents import LearnerAgent
from .demonstrator import Demonstrator
from .harness import HOUR, Match, Member, kernel_env
from .policy import PolicyGroup
from .registry import PolicyRegistry

TRAINING_WORLD = "open_digital"


# ====================================================================== 1-3: training before the civilisation
def demonstrations(seeds, *, hours: float = 48, n: int = 6, think_s: int = 1800) -> dict:
    """Recorded scripted demonstrations from CPU training worlds (one per seed). Returns {"demos": [records],
    "worlds": [world ids], "by_identity": {principal: [records]}}."""
    out = {"demos": [], "worlds": [], "by_identity": {}, "capabilities": [], "scripted": True}
    for seed in seeds:
        env = kernel_env(TRAINING_WORLD, int(seed))
        g = PolicyGroup(f"demo{seed}", "reinforce", seed=int(seed))
        m = Match(env, {g.gid: g}, [Member(f"dm{i}", "current", f"scripted:demo{seed}:{i % 3}", g.gid)
                                    for i in range(n)], seed=int(seed), think_s=think_s)
        for i, a in enumerate(m.agents.values()):
            a.chooser = Demonstrator(seed=int(seed) * 100 + i)
        m.run(hours, chunk_hours=12, learn=False)
        ctrl = {p: mm.controller for p, mm in m.members.items()}
        inter = AN.interactions(m.world, ctrl, m.t0)
        caps = AN.capabilities(inter, ventures=AN.ventures_of(m.world, ctrl, m.t0), groups_learning={},
                               strategy_response=None)
        out["capabilities"].append({k: v["ok"] for k, v in caps.items()})
        for p, a in m.agents.items():
            out["by_identity"][f"{env.world.world_id}:{p}"] = a.demos
            out["demos"] += a.demos
        out["worlds"].append(env.world.world_id)
        m.close()
    return out


def pretrain(registry: PolicyRegistry, specs: dict[str, dict], demos: dict, *, epochs: int = 8,
             disjoint: bool = False, batch: int = 64) -> dict:
    """Create and behaviour-clone every group of `specs` ({gid: {algo, arch, seed}}): each group from its own seed
    and its own minibatch order, on the whole demonstration corpus (a shared pre-training corpus, recorded as such) or,
    with `disjoint`, on a disjoint share of it (identities dealt round-robin). Returns {gid: PolicyGroup}."""
    idents = sorted(demos["by_identity"])
    share = {gid: [] for gid in specs}
    for k, ident in enumerate(idents):
        if disjoint:
            share[list(specs)[k % len(specs)]] += demos["by_identity"][ident]
        else:
            for gid in specs:
                share[gid] += demos["by_identity"][ident]
    groups = {}
    for gid, sp in specs.items():
        g = PolicyGroup(gid, sp.get("algo", "reinforce"), sp.get("arch", "mlp"), seed=int(sp.get("seed", 0)),
                        hp=sp.get("hp"))
        registry.register_group(g, world_id=",".join(demos["worlds"]), seed=None, origin="initialised")
        for _ in range(int(epochs)):
            g.imitate(share[gid], epochs=1, batch=batch)
        registry.checkpoint(g)
        v = registry.versions[registry.vid(g)]
        v["origin"] = "behaviour cloning (scripted demonstrations)"
        v["provenance"] = dict(v.get("provenance") or {}, demonstrations=len(share[gid]),
                               demonstration_worlds=list(demos["worlds"]), scripted_demonstrator=True)
        groups[gid] = g
    return groups


def heldout_family(registry: PolicyRegistry, *, seed: int = 4242, demo_seeds=(31,), hours: float = 24,
                   epochs: int = 6, demo_hours: float = 36, batch: int = 64) -> PolicyGroup:
    """The held-out policy family (see the module docstring): frozen and registered as `heldout_gru@<n>`."""
    d = demonstrations(demo_seeds, hours=demo_hours, n=4)
    g = PolicyGroup("heldout_gru", "ppo", "gru", seed=int(seed))
    registry.register_group(g, world_id=",".join(d["worlds"]), origin="initialised (held-out family)")
    for _ in range(int(epochs)):
        g.imitate(d["demos"], epochs=1, batch=batch)
    env = kernel_env(TRAINING_WORLD, int(seed))
    m = Match(env, {g.gid: g}, [Member(f"ho{i}", "current", f"ctrl:heldout-train:{i}", g.gid) for i in range(3)],
              registry=registry, seed=int(seed), think_s=1800)
    m.run(hours, chunk_hours=12)
    m.finish()
    m.close()
    g.frozen = True
    registry.checkpoint(g)
    registry.versions[registry.vid(g)]["origin"] = "held-out family (gru+ppo, trained outside the league)"
    return g


# ====================================================================== 4: the civilisation run
STANDARD_GROUPS = {"alpha": {"algo": "reinforce", "seed": 11}, "beta": {"algo": "ppo", "seed": 12},
                   "gamma": {"algo": "a2c", "seed": 13}, "delta": {"algo": "a2c", "seed": 14}}


def standard_members(semantic: int = 2, semantic_every: int = 4) -> list[Member]:
    ms = [Member(f"alpha{i}", "current", "ctrl:alpha", "alpha") for i in range(3)] + \
         [Member(f"beta{i}", "current", "ctrl:beta", "beta") for i in range(2)] + \
         [Member("gamma0", "independent", "ctrl:gamma", "gamma", drive="reach"),
          Member("delta0", "independent", "ctrl:delta", "delta", drive="adoption")]
    # goals in the words of the tools a builder uses (a semantic generalist ranks the world's hundreds of tools
    # against its goal text before choosing; "build a tool" alone sends it to the factories of the industrial
    # economy)
    goals = ["Start a software business: create a code repository, commit working code, register and release it as "
             "a package, and sell it to other people as a product.",
             "Write open-source code in a repository, release it as a package other developers can depend on, and "
             "invite other builders into your organisation."]
    for k in range(semantic):
        ms.append(Member(f"sem{k}", "semantic", f"ctrl:semantic{k}", goal=goals[k % 2], every=int(semantic_every)))
    return ms


def civ_run(env, registry: PolicyRegistry, groups: dict[str, PolicyGroup], members: list[Member], *, hours: float,
            think_s: int = 1800, seed: int = 0, semantic_service=None, historical_at: float | None = 0.5,
            controls: bool = True, chunk_hours: float = 6, store_root: str | None = None, before_run=None,
            wrap=None, extra_at: float | None = None, extra_members: list[Member] = (),
            extra_groups: dict | None = None) -> dict:
    """Run the civilisation self-play (see the module docstring). Returns the match and what was measured on the
    side: initial versions, controls, probe situations, the historical checkpoint."""
    initial = {gid: g.clone(f"{gid}_initial") for gid, g in groups.items()}
    for gid, g in initial.items():
        g.frozen = True
        registry.register_group(g, world_id=env.world.world_id, seed=seed, origin=f"frozen copy of {gid} at start",
                                checkpoint=True)
    ctl = {}
    if controls:
        for gid, g in groups.items():
            c = g.clone(f"{gid}_nosignal")
            g.controls.append(c)
            ctl[gid] = c
    m = Match(env, groups, members, registry=registry, seed=seed, think_s=think_s,
              semantic_service=semantic_service, store_root=store_root, wrap=wrap)
    for a in m.agents.values():
        if hasattr(a, "p"):
            a.p.choice_method = "mean"           # one scoring pass per decision (cost); see SemanticParticipant
    if before_run is not None:
        before_run(m)
    t0 = time.perf_counter()
    hist = None
    marks = sorted({x for x in (historical_at, extra_at) if x})
    done = 0.0
    for x in marks + [1.0]:
        if x > done:
            m.run(hours * (x - done), chunk_hours=chunk_hours)
            done = x
        if x == historical_at:
            main = next(iter(groups))
            hist = registry.freeze(groups[main], f"{main}_hist", seed=seed + 77)
            m.groups[hist.gid] = hist
            m.admit(Member(f"{main}hist", "historical", f"ctrl:{main}", hist.gid))
        if x == extra_at:
            for gid, g in (extra_groups or {}).items():
                m.groups[gid] = g
                if registry.vid(g) not in registry.versions:
                    registry.register_group(g, world_id=env.world.world_id, seed=seed, checkpoint=False)
            for mem in extra_members:
                m.admit(mem)
    wall = time.perf_counter() - t0
    for g in groups.values():
        registry.checkpoint(g)
    return {"match": m, "initial": initial, "controls": ctl, "historical": hist, "wall_s": wall}


def _inner(a):
    return a.__dict__.get("agent", a) if hasattr(a, "__dict__") else a


def learning_evidence(run: dict) -> dict:
    """Parameter and behaviour change of every learning group over the civilisation run, against its frozen
    initial copy and its no-signal control; and the early/late action outcomes of its identities."""
    m = run["match"]
    out = {}
    for gid, g in m.groups.items():
        if gid not in run["initial"]:
            continue
        init, ctl = run["initial"][gid], run["controls"].get(gid)
        ags = [_inner(a) for a in m.agents.values()]
        recs = [h for a in ags if isinstance(a, LearnerAgent) and a.group is g for h in getattr(a, "_decisions", [])]
        hist = sorted((h for a in ags if isinstance(a, LearnerAgent) and a.group is g for h in a.history),
                      key=lambda x: x["t"])
        k = max(1, len(hist) // 3)

        def rates(xs):
            n = max(1, len(xs))
            return {"state_changing_accepted": sum(1 for x in xs if x.get("ok") and not x.get("read_only")) / n,
                    "rejected": sum(1 for x in xs if x.get("ok") is False) / n,
                    "waits": sum(1 for x in xs if x.get("tool") is None) / n}
        probe = [r for r in recs[:: max(1, len(recs) // 40)]][:40]
        tv = [AN.tv_distance(init.tool_probs_of(r), g.tool_probs_of(r)) for r in probe]
        tv_ctl = [AN.tv_distance(init.tool_probs_of(r), ctl.tool_probs_of(r)) for r in probe] if ctl else []
        changed = [u for u in g.update_log if u.get("changed") and u.get("algo") != "behaviour_cloning"]
        out[gid] = {"algo": g.algo, "identities": list(g.identities), "updates_in_run": len(changed),
                    "weights_hash_initial": init.hash, "weights_hash_final": g.hash,
                    "weight_hash_changed": init.hash != g.hash,
                    "control_hash_changed": (ctl.hash != init.hash) if ctl else None,
                    "control_updates": ctl.updates if ctl else None,
                    "probe_situations": len(probe),
                    "behaviour_tv_mean": float(np.mean(tv)) if tv else 0.0,
                    "control_behaviour_tv_mean": float(np.mean(tv_ctl)) if tv_ctl else 0.0,
                    "early": rates(hist[:k]), "late": rates(hist[-k:])}
    return out


# ====================================================================== 5: strategy switch (counterfactual)
def clone_agent(a: LearnerAgent, group: PolicyGroup) -> LearnerAgent:
    b = LearnerAgent(a.principal, group, copy.deepcopy(a.objective), role=a.role)
    b.kn = copy.deepcopy(a.kn)
    b.h = None if a.h is None else a.h.copy()
    b.prev_obs, b.prev_total, b.signal_ema = copy.deepcopy(a.prev_obs), a.prev_total, a.signal_ema
    b.last_action, b.last_rec = copy.deepcopy(a.last_action), None
    return b


def strategy_switch(match: Match, *, responder: str, switched: str, replacement: PolicyGroup, hours: float = 12,
                    seed: int = 0) -> dict:
    """See the module docstring. `responder` and `switched` are group ids of `match`; `replacement` is the policy
    the switched group's identities play in the switch branch. Returns the responder's decision divergence."""
    from ..experience.kernel import KernelExperienceEnv
    learners = {p: _inner(a) for p, a in match.agents.items() if isinstance(_inner(a), LearnerAgent)}
    branches = {}
    for name in ("control", "replicate", "switch"):
        w = match.world.fork(f"{match.world.world_id}-{name}")
        env = KernelExperienceEnv(w, seed, agents=[], reward_mode=match.env.reward_mode, tier=match.env.tier,
                                  name=f"{match.env.name}-{name}", semantic=False)
        gmap = {}
        for gid, g in match.groups.items():
            if name == "switch" and gid == switched:
                gmap[gid] = replacement.clone(f"{replacement.gid}-{name}")
            else:
                gmap[gid] = g.clone(f"{gid}-{name}")
            gmap[gid].frozen = True
        agents = {}
        for p, a in learners.items():
            env.add_agent(p)
            agents[p] = clone_agent(a, gmap[a.group.gid])
        from ..experience.loop import run_learners
        steps = int(round(hours * HOUR / match.think_s))
        run_learners(env, agents, steps, think_s=match.think_s, update_every=10**9, learn=False, check_schema=False)
        branches[name] = {p: [(h["tool"], h["ok"]) for h in agents[p].history]
                          for p in agents if learners[p].group.gid == responder}
        env.close()

    def diverged(a, b):
        n, first = 0, None
        for p in a:
            for i, (x, y) in enumerate(zip(a[p], b[p])):
                if x != y:
                    n += 1
                    first = i if first is None else min(first, i)
        return n, first

    def dist(br):
        return AN.usage_distribution([{"tool": t, "ok": ok} for v in br.values() for t, ok in v])
    n_rep, _ = diverged(branches["control"], branches["replicate"])
    n_sw, first = diverged(branches["control"], branches["switch"])
    total = sum(len(v) for v in branches["control"].values())
    return {"responded": n_rep == 0 and n_sw > 0, "replicate_divergent_decisions": n_rep,
            "switch_divergent_decisions": n_sw, "responder_decisions": total, "first_divergence_step": first,
            "host_tv_switch_vs_control": AN.tv_distance(dist(branches["control"]), dist(branches["switch"])),
            "responder": responder, "switched": switched, "replacement": replacement.gid, "hours": hours}
