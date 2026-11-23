"""A persistent self-play population on disk.

`SelfPlayLeague(root, task)` is the population of one persistent world (the hidden mechanism of `task`). It keeps

* **individuals**: live learners with their own checkpoints (`root/individuals/<id>/`), lineage, architecture,
  objective and budget (`arena.Individual`);
* an **archive** of frozen snapshots, one per training stage (`root/archive/<id>@<stage>/`): a snapshot is a copy
  with its own weights, loaded from disk, that never learns again;
* **match records** (who sat where, what each earned from independent demand, the private quantities behind it);
* **ratings**: Elo and the TrueSkill-style rating of the 3.0 league (`training.league`), updated from training
  matches only;
* **matchmaking**: uniform, prioritised fictitious self-play (PFSP: opponents the learner loses to are sampled
  more often) and recency, all from the 3.0 league's `sample_opponents`, all restricted to the training side;
* a **held-out opponent population** that never trains anybody: `play` refuses it in a training match, sampling
  never returns it, and `evaluate_heldout` is a separate call whose every use is appended to
  `root/heldout_log.jsonl`. Its records carry `heldout_eval=True` and stay out of ratings and of the win rates
  that drive PFSP.

This class extends `crucible.training.league.League` (entries, rating math, sampling, lineage, reports) and
replaces only what a match *is*: an arena match between individuals instead of scripted policies in an economy.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Sequence

import numpy as np

from ..core.canonical import digest
from ..research.learner import EnvSpec
from ..research.tasks import Task
from ..training.league import League, MatchRecord
from .arena import Individual, SelfPlayEnv, play as arena_play, summarise
from .learners import learner_config, make_learner

FORMAT = "crucible-selfplay-league/1"
SLOT_STRIDE = 64                 # noise indices of match seed k are k * 64 ... k * 64 + n_slots - 1


def _safe(pid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@-]", "_", pid)


class SelfPlayLeague(League):
    def __init__(self, root: str, task: Task, spec: EnvSpec | None = None, mode: str = "competition",
                 n_slots: int = 4, seed: int = 0, k_elo: float = 24.0, coop_weight: float = 0.5,
                 draw_tol: float = 0.01):
        super().__init__(world=task.family, days=float(task.horizon), seed=seed, mode=mode, k_elo=k_elo,
                         coop_weight=coop_weight, draw_tol=draw_tol, deterministic=False)
        if not 1 <= n_slots <= SLOT_STRIDE:
            raise ValueError(f"n_slots must be between 1 and {SLOT_STRIDE}")
        self.root = str(root)
        self.task = task
        self.spec = spec or EnvSpec()
        self.n_slots = int(n_slots)
        self.env = SelfPlayEnv([task] * self.n_slots, self.spec)
        self.live: dict[str, Individual] = {}        # learners that can still learn
        self.frozen: dict[str, Individual] = {}      # snapshots and fixtures, loaded on demand
        self.info: dict[str, dict] = {}              # per entry: role (live | snapshot | fixture), stage, of
        self.heldout_log: list[dict] = []
        os.makedirs(self.root, exist_ok=True)

    # ------------------------------------------------------------------ population
    def _dir(self, pid: str) -> str:
        sub = "archive" if self.info[pid]["role"] == "snapshot" else "individuals"
        return os.path.join(self.root, sub, _safe(pid))

    def add(self, ind: Individual, heldout: bool = False, tags: Sequence[str] = (), role: str | None = None):
        """Register an individual. A learner that can learn joins as `live`; a frozen learner or a fixture as
        `fixture`. Held-out entries never take part in a training match."""
        role = role or ("fixture" if ind.learner.frozen or ind.privileged or getattr(ind.learner, "kind", "") in
                        ("idle", "random", "fixed") else "live")
        e = self.add_policy(ind.id, factory=lambda: ind, parent=ind.parent, tags=tuple(tags) + tuple(ind.tags),
                            heldout=heldout, generation=ind.generation)
        self.info[ind.id] = {"role": role, "stage": None, "of": ind.parent or ind.id, "arch": ind.arch,
                             "seat": ind.seat, "privileged": ind.privileged, "funds_scale": ind.funds_scale,
                             "objective": ind.objective.kind}
        (self.live if role == "live" else self.frozen)[ind.id] = ind
        ind.world = ind.world or self.task.id
        return e

    def new(self, id: str, algo: str, seed: int = 0, objective="own", seat: int | None = None, heldout: bool = False,
            funds_scale: float = 1.0, bankroll: int | None = None, tags: Sequence[str] = (), **options) -> Individual:
        """Create a fresh individual with its own learner and register it."""
        cfg = learner_config(algo, seed, **options)
        ind = Individual(id, make_learner(self.spec, **cfg), objective, seat=seat, funds_scale=funds_scale,
                         bankroll=bankroll, tags=tags, world=self.task.id, config=cfg)
        self.add(ind, heldout=heldout)
        return ind

    def get(self, pid: str) -> Individual:
        if pid in self.live:
            return self.live[pid]
        if pid not in self.frozen:
            if pid not in self.entries:
                raise KeyError(f"no individual {pid!r} in this league")
            ind = Individual.load(self._dir(pid), self.spec)
            ind.learner.freeze(True)
            self.frozen[pid] = ind
        return self.frozen[pid]

    def snapshot(self, pid: str, stage: int | str, heldout: bool = False) -> str:
        """Archive a frozen copy of a live individual as `<pid>@<stage>`. The copy is written to disk and read
        back, so it shares nothing with the live learner. `heldout=True` sets it aside for evaluation only."""
        src = self.live[pid]
        sid = f"{pid}@{stage}"
        if sid in self.entries:
            raise ValueError(f"snapshot {sid} exists")
        self.add_policy(sid, factory=lambda: None, parent=pid, tags=("snapshot",) + (("set_aside",) if heldout else ()),
                        heldout=heldout, generation=self.entries[pid].generation + 1)
        self.info[sid] = {"role": "snapshot", "stage": stage, "of": self.info[pid]["of"], "arch": src.arch,
                          "seat": src.seat, "privileged": src.privileged, "funds_scale": src.funds_scale,
                          "objective": src.objective.kind}
        d = self._dir(sid)
        was = src.learner.frozen
        src.learner.frozen = True                    # a frozen learner writes weights only
        try:
            src.save(d)
        finally:
            src.learner.frozen = was
        with open(os.path.join(d, "meta.json")) as fh:
            m = json.load(fh)
        m.update({"id": sid, "parent": pid, "generation": self.entries[sid].generation, "frozen": True,
                  "tags": list(src.tags) + ["snapshot"], "history": []})
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(m, fh)
        return sid

    def archive(self, of: str | None = None, include_heldout: bool = False) -> list[str]:
        """Snapshot ids (optionally of one lineage root), oldest first; the training side unless asked."""
        return [e.id for e in sorted(self.entries.values(), key=lambda e: e.added)
                if self.info[e.id]["role"] == "snapshot" and (include_heldout or not e.heldout)
                and (of is None or self.info[e.id]["of"] == of or e.parent == of)]

    def add_heldout_population(self, seats: Sequence[int | None] = (None,), profiles: Sequence[str] = (),
                               funds: Sequence[float] = (0.25, 4.0), climber: bool = True, seed: int = 0) -> list[str]:
        """The standard held-out opponents of a world: an idle holder, a random operator, a competent scripted
        operator, the same operator with less and with more starting funds, and operator-side profiles. One
        set per seat in `seats` (None: seat-free)."""
        from .profiles import profile_individual
        out = []
        for seat in seats:
            sfx = "" if seat is None else f":s{seat}"
            out.append(self.new(f"ho:hold{sfx}", "hold", seed, seat=seat, heldout=True, tags=("heldout",)).id)
            out.append(self.new(f"ho:random{sfx}", "random", seed + 1, seat=seat, heldout=True, tags=("heldout",),
                                hold=6).id)
            if climber:
                out.append(self.new(f"ho:climber{sfx}", "climber", seed + 2, seat=seat, heldout=True,
                                    tags=("heldout", "scripted"), persist=False).id)
                for f in funds:
                    out.append(self.new(f"ho:climber_funds{f:g}{sfx}", "climber", seed + 3, seat=seat, heldout=True,
                                        funds_scale=f, tags=("heldout", "scripted", "resource_variant"),
                                        persist=False).id)
            for name in profiles:
                ind = profile_individual(self.task, 0 if seat is None else seat, name, self.spec,
                                         id=f"ho:{name}{sfx}")
                ind.seat = seat
                self.add(ind, heldout=True, tags=("heldout",))
                out.append(ind.id)
        return out

    # ------------------------------------------------------------------ matches
    def _order(self, participants: Sequence[str]) -> list[str]:
        """Seat order for a set of participants: bound individuals in their seats, the rest in the order given."""
        S = self.env.n_seats
        if len(participants) != S:
            raise ValueError(f"a match of this world has {S} participants")
        seats: list[str | None] = [None] * S
        free = []
        for p in participants:
            s = self.info[p]["seat"]
            if s is None:
                free.append(p)
            elif seats[s] is not None:
                raise ValueError(f"{p} and {seats[s]} are both bound to seat {s}")
            else:
                seats[s] = p
        for i in range(S):
            if seats[i] is None:
                seats[i] = free.pop(0)
        return seats

    def play(self, participants: list[str], seed: int, *, heldout_eval: bool = False, days: float | None = None,
             record_trajectories: bool = False, learn: bool | Sequence[bool] = False, tag: str = "",
             rated: bool = True):
        """One batch of `n_slots` simultaneous matches on noise seed `seed`, participants seated in the order
        given (individuals bound to a seat go to their seat). `learn`: per participant, whether it learns from
        the match. Held-out individuals may only appear with `heldout_eval=True`, and then nobody learns.
        `rated=False` keeps a training-side match out of the ratings (periodic evaluations of a training pair,
        which would otherwise be counted many times)."""
        if len(set(participants)) != len(participants):
            raise ValueError("an individual cannot occupy two seats in one match")
        for p in participants:
            if self.entries[p].heldout and not heldout_eval:
                raise ValueError(f"held-out individual {p} cannot play a training match")
        order = self._order(participants)
        learn_by = dict(zip(participants, [bool(learn)] * len(participants) if isinstance(learn, (bool, np.bool_))
                            else [bool(x) for x in learn]))
        if heldout_eval and any(learn_by.values()):
            raise ValueError("nobody learns in a held-out evaluation match")
        inds = [self.get(p) for p in order]
        t0 = time.perf_counter()
        recs = arena_play(self.env, inds, noise=[int(seed) * SLOT_STRIDE + i for i in range(self.n_slots)],
                          learn=[learn_by[p] and p in self.live for p in order], tag=tag)
        s = summarise(recs)
        units = [s["private"].get(f"units_{i}", s["private"].get(f"completed_line_{i}", 0.0)) for i in range(len(order))]
        tot = sum(units)
        outcomes = {}
        for i, p in enumerate(order):
            outcomes[p] = {"seat": i, "net_per_tick": s["net_per_tick"][i],
                           "revenue": float(np.mean([r["seats"][i]["revenue_cents"] for r in recs])),
                           "cost": float(np.mean([r["seats"][i]["cost_cents"] for r in recs])),
                           "customers": float(units[i]), "share": round(units[i] / tot, 4) if tot else 0.0,
                           "ticks_idle": s["ticks_idle"][i], "ticks_fallback": s["ticks_fallback"][i],
                           "mean_action": s["mean_action"][i], "learned": bool(recs[0]["learned"][i])}
        scores = self._scores(outcomes)
        results = []
        for i, a in enumerate(participants):
            for b in participants[i + 1:]:
                sa, sb = scores[a], scores[b]
                tol = self.draw_tol * max(abs(sa), abs(sb), 1e-9)
                results.append([a, b, 0.5 if abs(sa - sb) <= tol else (1.0 if sa > sb else 0.0)])
        self._n += 1
        rec = MatchRecord(f"m{self._n:05d}", self.task.family, int(seed), digest(self.task.to_dict())[:16], self.mode,
                          float(self.task.horizon), list(participants), {p: i for i, p in enumerate(order)},
                          outcomes, scores, results, heldout_eval, round(time.perf_counter() - t0, 3))
        rec.joint_net_per_tick = s["joint_net_per_tick"]            # operator-side extras (not part of the dataclass)
        rec.private = s["private"]
        rec.tag, rec.rated = tag, bool(rated) and not heldout_eval
        self.records.append(rec)
        if rec.rated:
            self._update_ratings(rec)
        return (rec, recs) if record_trajectories else rec

    def _scores(self, outcomes: dict) -> dict:
        net = {p: float(o["net_per_tick"]) for p, o in outcomes.items()}
        if self.mode == "competition":
            return net
        total = sum(net.values())
        if self.mode == "cooperation":
            return {p: total for p in outcomes}
        return {p: net[p] + self.coop_weight * (total - net[p]) for p in outcomes}

    def train_match(self, participants: list[str], seed: int, tag: str = "train"):
        """A training match: every live participant learns, ratings are updated."""
        return self.play(participants, seed, learn=[p in self.live and not self.live[p].learner.frozen
                                                    for p in participants], tag=tag)

    def evaluate_heldout(self, pid: str, seeds, both_seatings: bool = True, tag: str = "heldout") -> dict:
        """Play `pid` (frozen for the occasion: nobody learns) against every held-out individual it can be seated
        with, on the given noise seeds. Logged in `heldout_log.jsonl`; records carry `heldout_eval=True`."""
        seeds = [int(s) for s in seeds]
        rows = {}
        for h in self.heldout_pool():
            sp, sh = self.info[pid]["seat"], self.info[h]["seat"]
            if sp is not None and sp == sh:
                continue                              # both bound to the same seat: they cannot meet
            recs = []
            for s in seeds:
                recs.append(self.play([pid, h], s, heldout_eval=True, tag=tag))
                if both_seatings and sp is None and sh is None:
                    recs.append(self.play([h, pid], s, heldout_eval=True, tag=tag))
            res = [x[2] if x[0] == pid else 1 - x[2] for r in recs for x in r.results]
            rows[h] = {"games": len(res), "win_rate": float(np.mean(res)),
                       "net_per_tick": float(np.mean([r.outcomes[pid]["net_per_tick"] for r in recs])),
                       "opponent_net_per_tick": float(np.mean([r.outcomes[h]["net_per_tick"] for r in recs])),
                       "joint_net_per_tick": float(np.mean([r.joint_net_per_tick for r in recs])),
                       "ticks_idle": int(sum(r.outcomes[pid]["ticks_idle"] for r in recs)),
                       "opponent_ticks_idle": int(sum(r.outcomes[h]["ticks_idle"] for r in recs))}
        wr = [v["win_rate"] for v in rows.values()]
        out = {"policy": pid, "vs": rows, "mean_win_rate": float(np.mean(wr)) if wr else None,
               "mean_net_per_tick": float(np.mean([v["net_per_tick"] for v in rows.values()])) if rows else None}
        entry = {"n": len(self.heldout_log) + 1, "policy": pid, "seeds": seeds, "opponents": sorted(rows),
                 "matches": int(sum(v["games"] for v in rows.values())), "mean_win_rate": out["mean_win_rate"]}
        self.heldout_log.append(entry)
        with open(os.path.join(self.root, "heldout_log.jsonl"), "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        return out

    def head_to_head(self, ids: Sequence[str], seeds, tag: str = "h2h", rate: bool = True) -> dict:
        """Frozen round robin between seat-free individuals (both seatings, the given noise seeds). Returns the
        mean net of row against column, the win-rate matrix, and whether the win relation contains a cycle.
        With `rate`, the matches count for the ratings (they are training-side matches in which nobody learns)."""
        ids = list(ids)
        n = len(ids)
        net = np.full((n, n), np.nan)
        win = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ids[i], ids[j]
                recs = []
                for s in seeds:
                    recs.append(self.play([a, b], s, tag=tag, rated=rate))
                    if self.info[a]["seat"] is None and self.info[b]["seat"] is None:
                        recs.append(self.play([b, a], s, tag=tag, rated=rate))
                net[i, j] = np.mean([r.outcomes[a]["net_per_tick"] for r in recs])
                net[j, i] = np.mean([r.outcomes[b]["net_per_tick"] for r in recs])
                w = np.mean([x[2] if x[0] == a else 1 - x[2] for r in recs for x in r.results])
                win[i, j], win[j, i] = w, 1 - w
        cycles = []
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    if a < b and a < c and b != c and win[a, b] > 0.5 and win[b, c] > 0.5 and win[c, a] > 0.5:
                        cycles.append([ids[a], ids[b], ids[c]])
        return {"ids": ids, "net": net.tolist(), "win": win.tolist(), "cycles": cycles}

    def rerate(self, tags: Sequence[str] | None = None) -> list[dict]:
        """Recompute every rating from the recorded rated matches (optionally only those with one of `tags`),
        in order. Held-out evaluation matches never count."""
        for e in self.entries.values():
            e.elo, e.mu, e.sigma, e.games = 1000.0, 25.0, 25.0 / 3, 0
        for r in self.records:
            if getattr(r, "rated", not r.heldout_eval) and not r.heldout_eval and \
                    (tags is None or getattr(r, "tag", "") in tags):
                self._update_ratings(r)
        return self.ratings()

    # ------------------------------------------------------------------ persistence
    def save(self) -> str:
        """Write the whole league under `root`: `league.json` (task, entries with ratings and lineage, records,
        the held-out log) and a checkpoint per live individual and fixture (snapshots were written when taken)."""
        for pid, ind in list(self.live.items()) + [(p, i) for p, i in self.frozen.items()
                                                   if self.info[p]["role"] != "snapshot"]:
            ind.save(self._dir(pid))
        recs = []
        for r in self.records:
            d = r.to_dict()
            d["joint_net_per_tick"] = getattr(r, "joint_net_per_tick", None)
            d["private"] = getattr(r, "private", None)
            d["tag"], d["rated"] = getattr(r, "tag", ""), getattr(r, "rated", not r.heldout_eval)
            recs.append(d)
        doc = {"format": FORMAT, "task": self.task.to_dict(), "mode": self.mode, "n_slots": self.n_slots,
               "seed": self.seed, "k_elo": self.k_elo, "coop_weight": self.coop_weight, "draw_tol": self.draw_tol,
               "entries": [e.to_dict() for e in self.entries.values()], "info": self.info, "records": recs,
               "ratings": self.ratings(), "heldout_log": self.heldout_log}
        tmp = os.path.join(self.root, f"league.json.{os.getpid()}.tmp")
        with open(tmp, "w") as fh:
            json.dump(doc, fh)
        os.replace(tmp, os.path.join(self.root, "league.json"))
        return self.root

    @classmethod
    def load(cls, root: str, spec: EnvSpec | None = None) -> "SelfPlayLeague":   # type: ignore[override]
        from ..training.league import PolicyEntry
        with open(os.path.join(root, "league.json")) as fh:
            d = json.load(fh)
        if d.get("format") != FORMAT:
            raise ValueError(f"{root} is not a {FORMAT} directory")
        lg = cls(root, Task.from_dict(d["task"]), spec, d["mode"], d["n_slots"], d["seed"], d["k_elo"],
                 d["coop_weight"], d["draw_tol"])
        lg.info = d["info"]
        for e in sorted(d["entries"], key=lambda e: e["added"]):
            lg.entries[e["id"]] = PolicyEntry(e["id"], None, e.get("parent"), tuple(e.get("tags", ())),
                                              e.get("generation", 0), e.get("added", 0), e.get("heldout", False),
                                              e["elo"], e["mu"], e["sigma"], e.get("games", 0))
        for r in d["records"]:
            extra = {k: r.pop(k, None) for k in ("joint_net_per_tick", "private", "tag", "rated")}
            rec = MatchRecord(**r)
            rec.joint_net_per_tick, rec.private = extra["joint_net_per_tick"], extra["private"]
            rec.tag, rec.rated = extra["tag"] or "", bool(extra["rated"])
            lg.records.append(rec)
        lg._n = len(lg.records)
        lg.heldout_log = list(d.get("heldout_log", []))
        for pid, inf in lg.info.items():             # live learners come back live, with their optimiser state
            if inf["role"] == "live":
                lg.live[pid] = Individual.load(lg._dir(pid), lg.spec)
        return lg

    def export_checkpoints(self, ids: Sequence[str], dest: str) -> list[str]:
        """Copy the checkpoints of some individuals (frozen: weights only) to `dest/<id>/`."""
        out = []
        for pid in ids:
            d = os.path.join(dest, _safe(pid))
            if pid in self.live:
                ind = self.live[pid]
                was = ind.learner.frozen
                ind.learner.frozen = True
                try:
                    ind.save(d)
                finally:
                    ind.learner.frozen = was
            else:
                if os.path.isdir(d):
                    shutil.rmtree(d)
                shutil.copytree(self._dir(pid), d)
            out.append(d)
        return out
