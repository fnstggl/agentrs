"""League and population-based evaluation over civilisation matches: recorded match specifications, reproducible
cross-play between policy versions, evaluation against held-out opponents on sealed seeds, ratings and
prioritised fictitious self-play (MASTER_SPEC 12.2; Gate Q items 5-6).

A `MatchSpec` fully determines a match: the world (``civ:<profile>`` - a Tier 3 civilisation - or
``kernel:<family>`` for unit tests), its seed, the roster (handle, role, controller, policy version from the
registry, drive, endowment), the decision interval, the duration, whether groups learn, and the seed every loaded
policy's generators are re-seeded with. `play(spec, registry)` builds the world, loads every version from its
content-addressed checkpoint (the weights hash is verified), runs it through `crucible.multiagent.harness.Match`
and records it; `replay(spec, registry, record)` plays it again and compares the event-log digest and every
evaluator outcome - a cross-play result is reproducible exactly when nothing differs.

Held-out evaluation (`heldout_evaluation`): candidate versions play frozen against frozen *held-out* policies -
families (architecture and algorithm) never met in training, trained in other worlds - on sealed seeds no training
run used (`SEALED_SEEDS`; refused if any registered version was trained on one). Outcomes are the evaluator's.

Ratings: Elo over pairwise comparisons of the parties' evaluator outcomes in each match; `pfsp_weights` gives the
prioritised-fictitious-self-play sampling weights of opponents (harder opponents more often).
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .harness import HOUR, Match, Member, civ_env, kernel_env
from .registry import PolicyRegistry

SEALED_SEEDS = tuple(range(90_001, 90_021))
OUTCOME_KEYS = ("cash_cents", "cash_delta_cents", "independent_revenue_cents", "excluded_revenue_cents",
                "accepted", "rejected", "state_changing_accepted", "tools_used")


@dataclass
class Seat:
    handle: str
    role: str
    controller: str
    version: str                   # registry version id "<group>@<n>"
    drive: str = "profit"
    endowment: int = 2_000_000


@dataclass
class MatchSpec:
    world: str                     # "civ:<profile>" or "kernel:<family>"
    seed: int
    seats: list
    hours: float = 24.0
    think_s: int = HOUR
    learn: bool = False
    policy_seed: int = 0
    population: str = "development"
    tag: str = "match"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seats"] = [asdict(s) if isinstance(s, Seat) else dict(s) for s in self.seats]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MatchSpec":
        d = dict(d)
        d["seats"] = [Seat(**s) for s in d["seats"]]
        return cls(**d)


def build_env(spec: MatchSpec):
    kind, name = spec.world.split(":", 1)
    if kind == "civ":
        return civ_env(name, spec.seed, population=spec.population)
    if kind == "kernel":
        return kernel_env(name, spec.seed)
    raise ValueError(f"unknown world {spec.world!r}")


def play(spec: MatchSpec, registry: PolicyRegistry, *, match_id: str | None = None, store_root: str | None = None,
         keep: bool = False) -> dict:
    """Play one match (see the module docstring). Returns the match record with outcomes per seat and party."""
    env = build_env(spec)
    groups = {}
    gid_of = {}
    for i, s in enumerate(spec.seats):
        gid = f"{s.version.replace('@', 'v')}-{spec.tag}"
        gid_of[s.handle] = gid
        if gid not in groups:
            g = registry.load(s.version, gid, seed=int(spec.policy_seed) * 1000 + i, frozen=not spec.learn)
            registry.register_group(g, world_id=None, seed=spec.seed, origin=f"loaded {s.version}", checkpoint=False)
            registry.versions[registry.vid(g)]["copy_of"] = s.version
            groups[gid] = g
    members = [Member(s.handle, s.role, s.controller, gid_of[s.handle], s.drive, s.endowment) for s in spec.seats]
    m = Match(env, groups, members, registry=registry, match_id=match_id or f"{spec.tag}-{spec.seed}",
              think_s=spec.think_s, seed=spec.seed, store_root=store_root)
    m.run(spec.hours, chunk_hours=max(1.0, spec.hours / 4), learn=spec.learn)
    outs = m.finish()
    rec = dict(registry.matches[-1])
    rec["spec"] = spec.to_dict()
    rec["outcomes"] = {p: {k: outs[p][k] for k in OUTCOME_KEYS} for p in outs}
    rec["by_seat"] = {s.handle: rec["outcomes"][p] for s, p in zip(spec.seats, m.agents)}
    party: dict[str, dict] = {}
    for s, p in zip(spec.seats, m.agents):
        d = party.setdefault(s.controller, {k: 0 for k in OUTCOME_KEYS})
        for k in OUTCOME_KEYS:
            d[k] += outs[p][k]
    rec["by_party"] = party
    registry.matches[-1] = rec
    if keep:
        rec["_match"] = m
    else:
        m.close()
    return rec


def replay(spec: MatchSpec, registry: PolicyRegistry, record: dict) -> dict:
    """Play `spec` again and compare digest and outcomes with `record`. Returns the mismatches."""
    again = play(spec, registry, match_id=f"{record['match']}-replay")
    mism = []
    if again["digest"] != record["digest"]:
        mism.append({"what": "world digest", "a": record["digest"], "b": again["digest"]})
    for h, o in record["by_seat"].items():
        for k in OUTCOME_KEYS:
            if again["by_seat"][h][k] != o[k]:
                mism.append({"what": f"{h}.{k}", "a": o[k], "b": again["by_seat"][h][k]})
    return {"mismatches": mism, "digest": again["digest"], "events": again["events"]}


def party_score(o: dict) -> float:
    """The evaluator measure a league compares parties by: money gained plus independent revenue (cents)."""
    return float(o["cash_delta_cents"]) + float(o["independent_revenue_cents"])


# ====================================================================== cross-play and ratings
def crossplay(registry: PolicyRegistry, a: str, b: str, *, world: str, seeds, hours: float, think_s: int = HOUR,
              n_each: int = 2, tag: str = "xp") -> list[dict]:
    """Version `a` (controller ctrl:A) against version `b` (ctrl:B), `n_each` identities each, frozen, on every
    seed. Returns the match records."""
    out = []
    for s in seeds:
        seats = [Seat(f"xa{i}", "current", "ctrl:A", a) for i in range(n_each)] + \
                [Seat(f"xb{i}", "independent", "ctrl:B", b) for i in range(n_each)]
        spec = MatchSpec(world, int(s), seats, hours=hours, think_s=think_s, learn=False, policy_seed=int(s),
                         tag=f"{tag}-{a}-{b}".replace("@", "v").replace("#", "_"))
        out.append(play(spec, registry))
    return out


class Elo:
    def __init__(self, k: float = 24.0, base: float = 1000.0):
        self.k, self.base = float(k), float(base)
        self.r: dict[str, float] = {}
        self.games: dict[str, int] = {}

    def update(self, a: str, b: str, score_a: float) -> None:
        ra, rb = self.r.get(a, self.base), self.r.get(b, self.base)
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        self.r[a] = ra + self.k * (score_a - ea)
        self.r[b] = rb + self.k * ((1 - score_a) - (1 - ea))
        self.games[a] = self.games.get(a, 0) + 1
        self.games[b] = self.games.get(b, 0) + 1


def league_ratings(records: list[dict], label_of: dict[str, str], *, draw_tol: float = 0.02) -> dict:
    """Elo over every pair of parties in every match (party outcome -> label of the version it played)."""
    elo = Elo()
    pairs = 0
    for rec in records:
        ps = sorted(rec["by_party"])
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = ps[i], ps[j]
                sa, sb = party_score(rec["by_party"][a]), party_score(rec["by_party"][b])
                scale = max(1.0, abs(sa) + abs(sb))
                s = 0.5 if abs(sa - sb) / scale <= draw_tol else float(sa > sb)
                elo.update(label_of.get(f"{rec['match']}:{a}", a), label_of.get(f"{rec['match']}:{b}", b), s)
                pairs += 1
    return {"ratings": dict(sorted(elo.r.items())), "games": elo.games, "pairs": pairs}


def pfsp_weights(win_rate: dict[str, float], p: float = 2.0) -> dict[str, float]:
    """Prioritised fictitious self-play: weight (1 - win rate)^p over opponents, normalised."""
    w = {k: (1.0 - float(v)) ** p for k, v in win_rate.items()}
    tot = sum(w.values())
    return {k: (v / tot if tot > 0 else 1.0 / len(w)) for k, v in w.items()}


# ====================================================================== held-out evaluation
def check_sealed(registry: PolicyRegistry, seeds) -> list[str]:
    bad = []
    sealed = set(int(s) for s in seeds)
    for v in registry.versions.values():
        if v.get("seed") is not None and int(v["seed"]) in sealed and v.get("origin") == "update":
            bad.append(v["vid"])
    return bad


def heldout_evaluation(registry: PolicyRegistry, candidates: dict[str, str], heldout: list[str], *, world: str,
                       seeds=SEALED_SEEDS[:2], hours: float = 24.0, think_s: int = HOUR, n_each: int = 2,
                       tag: str = "ho") -> dict:
    """Every candidate version (label -> vid) plays frozen with `n_each` identities against the frozen held-out
    versions (each with `n_each` identities of its own controller) on every sealed seed. Returns per candidate
    the evaluator outcomes per seed, their mean and a paired comparison between the first two candidates."""
    bad = check_sealed(registry, seeds)
    if bad:
        raise ValueError(f"sealed seeds were used in training by {bad[:5]}")
    res: dict[str, list] = {}
    records = []
    for label, vid in candidates.items():
        for s in seeds:
            seats = [Seat(f"c{label[:6]}{i}", "current", f"ctrl:cand:{label}", vid) for i in range(n_each)]
            for k, h in enumerate(heldout):
                seats += [Seat(f"h{k}x{i}", "heldout", f"ctrl:heldout:{k}", h) for i in range(n_each)]
            spec = MatchSpec(world, int(s), seats, hours=hours, think_s=think_s, learn=False, policy_seed=int(s),
                             tag=f"{tag}-{label}")
            rec = play(spec, registry)
            records.append(rec)
            res.setdefault(label, []).append({"seed": int(s), **rec["by_party"][f"ctrl:cand:{label}"],
                                              "heldout": {k: rec["by_party"][f"ctrl:heldout:{k}"]
                                                          for k in range(len(heldout))}})
    summary = {}
    for label, rows in res.items():
        summary[label] = {k: float(np.mean([r[k] for r in rows])) for k in OUTCOME_KEYS}
        summary[label]["score_mean"] = float(np.mean([party_score(r) for r in rows]))
    labels = list(candidates)
    paired = None
    if len(labels) >= 2:
        d = [party_score(a) - party_score(b) for a, b in zip(res[labels[0]], res[labels[1]])]
        paired = {"a": labels[0], "b": labels[1], "diff_mean": float(np.mean(d)), "diffs": d,
                  "rejected_diff_mean": float(np.mean([a["rejected"] - b["rejected"]
                                                       for a, b in zip(res[labels[0]], res[labels[1]])]))}
    return {"per_seed": res, "summary": summary, "paired": paired, "matches": len(records),
            "seeds": [int(s) for s in seeds], "heldout": list(heldout)}
