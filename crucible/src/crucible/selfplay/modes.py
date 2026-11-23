"""The six interaction modes of self-play, each a runnable configuration with its own objective and analysis.

    mode          who trains against whom                         objective of an individual      what is measured
    competitive   two individuals, each other                     own net (rig_net)               own nets, the response of one policy to the other's, exploitability
    cooperative   two individuals, each other                     the joint outcome (team mean)   joint net against partners that do not cooperate
    mixed_motive  two individuals, each other, on a shared pool   own net                         where they settle between joint restraint and mutual over-use
    population    several individuals of different versions and   own net                         ratings, the round-robin payoff matrix
                  architectures, matched inside one pool
    historical    current learners against archived stages of     own net                         whether training keeps beating its own past, or cycles
                  earlier training, sampled by a stated rule
    cross_play    nobody trains: every policy meets partners      as trained                      the full matrix; diagonal minus off-diagonal = over-fitting
                  that were trained independently of it                                           to the training partner

`run_mode(name, task, root, ...)` runs one of them in a league directory and returns a JSON-serialisable result.
The building blocks (`train`, `evaluate`, `best_response`, `cross_play`, `lagged_response`,
`commons_position`) are what `examples/ow_selfplay.py` composes into the experiments.

Seeds. A match seed k of a league uses noise indices `k * 64 + slot`. Training matches use seeds from
`TRAIN_SEED0 + run_offset`, evaluation matches the fixed seeds `EVAL_SEEDS`, held-out evaluation `HELDOUT_SEEDS`:
the three ranges never overlap, so no individual is evaluated on a world it trained in.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

import numpy as np

from ..research.learner import EnvSpec
from ..research.tasks import Task
from .arena import Individual, SelfPlayEnv, play, summarise
from .league import SLOT_STRIDE, SelfPlayLeague
from .learners import learner_config, make_learner, policy_summary

TRAIN_SEED0 = 1_000
EVAL_SEEDS = (900_000,)
HELDOUT_SEEDS = (950_000,)
MODE_NAMES = ("competitive", "cooperative", "mixed_motive", "population", "historical", "cross_play")


@dataclasses.dataclass(frozen=True)
class ModeSpec:
    name: str
    objective: str            # the objective of every trained individual
    league_mode: str          # how a match is scored for ratings (training.league modes)
    families: tuple           # mechanism families the mode is meant for
    doc: str


MODES: dict[str, ModeSpec] = {
    "competitive": ModeSpec("competitive", "own", "competition", ("abm_duopoly", "abm_duopoly_fx"),
                            "Two independent learners fight for the same customers; each maximises its own net."),
    "cooperative": ModeSpec("cooperative", "joint", "cooperation", ("tech_handoff", "abm_complement"),
                            "Two independent learners run two halves of one job; each is credited the joint outcome."),
    "mixed_motive": ModeSpec("mixed_motive", "own", "mixed", ("abm_commons", "tech_shared_pool"),
                             "Two independent learners draw on one congestible pool; each maximises its own net."),
    "population": ModeSpec("population", "own", "competition", ("abm_duopoly",),
                           "Several individuals of different versions and architectures are matched in one pool."),
    "historical": ModeSpec("historical", "own", "competition", ("abm_duopoly",),
                           "Current learners meet archived stages of earlier training (PFSP by default)."),
    "cross_play": ModeSpec("cross_play", "own", "competition", ("abm_duopoly", "tech_handoff"),
                           "Policies meet independently trained partners they never met in training."),
}


# ====================================================================== evaluation
_ENVS: dict[tuple, SelfPlayEnv] = {}


def _env(task: Task, n: int, spec: EnvSpec | None = None) -> SelfPlayEnv:
    key = (task.family, task.seed, task.mech_seed, int(n), (spec or EnvSpec()))
    if key not in _ENVS:
        if len(_ENVS) > 32:
            _ENVS.clear()
        _ENVS[key] = SelfPlayEnv([task] * int(n), spec)
    return _ENVS[key]


def noise_of(seed: int, n: int) -> list[int]:
    return [int(seed) * SLOT_STRIDE + i for i in range(n)]


def evaluate(task: Task, individuals: Sequence[Individual | None], seeds: Sequence[int] = EVAL_SEEDS, n_slots: int = 8,
             spec: EnvSpec | None = None, tag: str = "eval") -> dict:
    """Frozen play (no exploration, no learning) of the individuals in their seats on the given noise seeds.
    Returns `summarise` of the matches plus the per-match rows (`rows`: per-seat net per tick and joint net)."""
    env = _env(task, n_slots, spec)
    recs = []
    for s in seeds:
        recs += play(env, individuals, noise=noise_of(s, n_slots), learn=False, explore=False, tag=tag)
    out = summarise(recs)
    out["rows"] = [{"noise": r["noise"], "net": [x["net_per_tick"] for x in r["seats"]],
                    "joint": r["joint_net_per_tick"]} for r in recs]
    out["records"] = recs
    return out


def light(ev: dict) -> dict:
    """An evaluation without its bulky per-match records."""
    return {k: v for k, v in ev.items() if k not in ("records", "rows")}


# ====================================================================== training
def train(league: SelfPlayLeague, seats: Sequence[str], iters: int, *, start: int = 0, seed_offset: int = 0,
          eval_every: int = 5, eval_seeds: Sequence[int] = EVAL_SEEDS, snapshot_every: int | None = None,
          learn: Sequence[bool] | None = None, view: Callable[[int, Sequence[float]], dict] | None = None,
          on_iter: Callable[[int, dict], None] | None = None) -> list[dict]:
    """Independent training of the individuals in `seats` (ids, in seat order) against each other for `iters`
    batches of matches. Each learns only from its own observations and its own objective. Returns the curve:
    one row per iteration with the training nets, each seat's mean settings, its policy vector where it has
    one, and every `eval_every` iterations a frozen evaluation on `eval_seeds`. With `snapshot_every`, a frozen
    stage of every live seat is archived before iterations 0, k, 2k, ... and after the last one."""
    curve = []
    live = [p for p in seats if p in league.live]
    for it in range(start, start + iters):
        if snapshot_every and it % snapshot_every == 0:
            for p in live:
                if f"{p}@{it}" not in league.entries:
                    league.snapshot(p, it)
        lr = [p in league.live and not league.live[p].learner.frozen for p in seats] if learn is None else list(learn)
        rec = league.play(list(seats), TRAIN_SEED0 + seed_offset + it, learn=lr, tag="train")
        row = {"it": it, "net": [rec.outcomes[p]["net_per_tick"] for p in seats],
               "joint": rec.joint_net_per_tick, "action": [rec.outcomes[p]["mean_action"] for p in seats],
               "policy": [policy_summary(league.get(p).learner) for p in seats]}
        if view is not None:
            row["view"] = [view(s, rec.outcomes[p]["mean_action"]) for s, p in enumerate(seats)]
        if eval_every and ((it + 1) % eval_every == 0 or it == start + iters - 1):
            evs = [league.play(list(seats), s, tag="eval", rated=False) for s in eval_seeds]
            row["eval_net"] = [float(np.mean([e.outcomes[p]["net_per_tick"] for e in evs])) for p in seats]
            row["eval_joint"] = float(np.mean([e.joint_net_per_tick for e in evs]))
            row["eval_action"] = [np.mean([e.outcomes[p]["mean_action"] for e in evs], axis=0).tolist() for p in seats]
            if view is not None:
                row["eval_view"] = [view(s, a) for s, a in enumerate(row["eval_action"])]
        curve.append(row)
        if on_iter is not None:
            on_iter(it, row)
    if snapshot_every:
        for p in live:
            if f"{p}@{start + iters}" not in league.entries:
                league.snapshot(p, start + iters)
    return curve


def best_response(task: Task, target: Individual, seat: int, root: str, algo: str = "spsa", seed: int = 0,
                  iters: int = 30, n_slots: int = 8, spec: EnvSpec | None = None, warm: Individual | None = None,
                  eval_seeds: Sequence[int] = EVAL_SEEDS, seed_offset: int = 50_000, **options) -> dict:
    """Exploitability probe: a learner trained in `seat` against the frozen `target` (in the other seat of a
    two-seat world), from scratch or, with `warm`, starting from a copy of that individual's weights. Returns
    what the best response earns against the target, what the target then earns, and the training curve."""
    lg = SelfPlayLeague(root, task, spec, "competition", n_slots)
    was = target.learner.frozen
    target.learner.freeze(True)
    try:
        t_seat, t_bound = target.seat, 1 - seat
        target.seat = t_bound
        lg.add(target, role="fixture")
        if warm is not None:
            import os
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                warm_frozen = warm.learner.frozen
                warm.learner.frozen = False
                try:
                    warm.save(os.path.join(tmp, "w"))
                finally:
                    warm.learner.frozen = warm_frozen
                br = Individual.load(os.path.join(tmp, "w"), lg.spec)
            br.learner.freeze(False)
            br.id, br.parent, br.seat, br.history = f"br:{target.id}", warm.id, seat, []
            lg.add(br, role="live")
        else:
            br = lg.new(f"br:{target.id}", algo, seed, "own", seat=seat, **options)
        seats = [br.id, target.id] if seat == 0 else [target.id, br.id]
        curve = train(lg, seats, iters, seed_offset=seed_offset, eval_every=max(1, iters // 6), eval_seeds=eval_seeds)
        inds = [lg.get(p) for p in seats]
        ev = evaluate(task, inds, eval_seeds, n_slots, spec)
    finally:
        target.seat = t_seat
        target.learner.freeze(was)
    return {"target": target.id, "seat": seat, "algo": br.arch, "warm": warm is not None, "iters": iters,
            "br_net": ev["net_per_tick"][seat], "target_net": ev["net_per_tick"][1 - seat],
            "joint": ev["joint_net_per_tick"], "rows": ev["rows"],
            "curve": [{"it": r["it"], "eval_net": r.get("eval_net")} for r in curve if "eval_net" in r],
            "individual": br}


# ====================================================================== analysis
def lagged_response(x: Sequence[float], y: Sequence[float], max_lag: int = 5) -> dict:
    """Does a change in x come before a change in y? Correlation between the iteration-to-iteration change of
    x at time t and the change of y at time t + lag, for lag = -max_lag ... max_lag. A positive lag with a
    correlation that the negative lags do not show means y follows x. Returns {"lags", "corr", "n"}."""
    dx, dy = np.diff(np.asarray(x, float)), np.diff(np.asarray(y, float))
    lags, corr = list(range(-max_lag, max_lag + 1)), []
    for lag in lags:
        if lag >= 0:
            a, b = dx[:len(dx) - lag], dy[lag:]
        else:
            a, b = dx[-lag:], dy[:len(dy) + lag]
        if len(a) < 3 or a.std() == 0 or b.std() == 0:
            corr.append(None)
        else:
            corr.append(float(np.corrcoef(a, b)[0, 1]))
    return {"lags": lags, "corr": corr, "n": int(len(dx))}


def commons_position(joint: float, joint_restrain: float, joint_grab: float) -> float | None:
    """Where a pair sits between mutual over-use (0) and joint restraint (1), by joint net."""
    gap = joint_restrain - joint_grab
    return None if gap <= 0 else float((joint - joint_grab) / gap)


def cross_play(task: Task, seat0: Sequence[Individual], seat1: Sequence[Individual],
               seeds: Sequence[int] = EVAL_SEEDS, n_slots: int = 8, spec: EnvSpec | None = None) -> dict:
    """The full cross-play matrix: `seat0[i]` with `seat1[j]` for every i, j, frozen, on the same noise.
    `seat0[i]` and `seat1[i]` are assumed to have been trained together (the diagonal). Returns the matrices of
    joint net and of each seat's net, and the cross-play gap: diagonal mean minus off-diagonal mean, for the
    joint net and for each seat (a positive gap is what the policies lose with a partner they did not train
    with)."""
    n = len(seat0)
    if len(seat1) != n:
        raise ValueError("cross-play needs as many seat-1 as seat-0 individuals (pair i = seat0[i], seat1[i])")
    joint, n0, n1 = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for i, a in enumerate(seat0):
        for j, b in enumerate(seat1):
            ev = evaluate(task, [a, b], seeds, n_slots, spec, tag="cross_play")
            joint[i, j], n0[i, j], n1[i, j] = ev["joint_net_per_tick"], ev["net_per_tick"][0], ev["net_per_tick"][1]
    off = ~np.eye(n, dtype=bool)

    def gap(m):
        return float(np.diag(m).mean() - m[off].mean()) if n > 1 else None

    # per individual: its own-pair result minus its mean result with the other partners
    row_gap = [float(joint[i, i] - np.mean([joint[i, j] for j in range(n) if j != i])) for i in range(n)] if n > 1 else []
    col_gap = [float(joint[j, j] - np.mean([joint[i, j] for i in range(n) if i != j])) for j in range(n)] if n > 1 else []
    return {"seat0": [a.id for a in seat0], "seat1": [b.id for b in seat1], "arch0": [a.arch for a in seat0],
            "arch1": [b.arch for b in seat1], "joint": joint.tolist(), "net0": n0.tolist(), "net1": n1.tolist(),
            "diag_joint": float(np.diag(joint).mean()), "offdiag_joint": float(joint[off].mean()) if n > 1 else None,
            "gap_joint": gap(joint), "gap_net0": gap(n0), "gap_net1": gap(n1),
            "gap_joint_by_seat0": row_gap, "gap_joint_by_seat1": col_gap}


# ====================================================================== the modes
def _pair(league: SelfPlayLeague, algos: Sequence[str], seeds: Sequence[int], objective, bound: bool,
          options: Sequence[dict] | None = None) -> list[str]:
    ids = []
    for s in range(league.env.n_seats):
        opt = dict((options or [{}] * league.env.n_seats)[s])
        ind = league.new(f"{algos[s]}{seeds[s]}.s{s}", algos[s], seeds[s], objective, seat=s if bound else None, **opt)
        ids.append(ind.id)
    return ids


def seat_bound(task: Task) -> bool:
    """Whether an individual of this world has to stay in the seat it learned (seats with different levers or
    gauges). Markets of the agent-based engine give every seat the same surface, so their individuals can swap."""
    from .arena import _sim
    eng = _sim(task, [None] * task.n_seats).engine
    surf = [([lv.name for lv in eng.levers(s)], [g.name for g in eng.gauges(s)]) for s in range(eng.n_seats)]
    return any(x != surf[0] for x in surf[1:])


def run_pair(mode: str, task: Task, root: str, *, algos: Sequence[str] = ("spsa", "spsa"), seeds: Sequence[int] = (1, 2),
             iters: int = 30, n_slots: int = 8, objective: Any = None, eval_every: int = 5,
             snapshot_every: int | None = 10, spec: EnvSpec | None = None, seed_offset: int = 0,
             options: Sequence[dict] | None = None, view: Callable | None = None) -> dict:
    """Competitive, cooperative and mixed-motive training of one pair: the three modes differ in the objective
    the individuals are given, in how a match is scored for ratings, and in the analysis attached below."""
    ms = MODES[mode]
    lg = SelfPlayLeague(root, task, spec, ms.league_mode, n_slots)
    obj = ms.objective if objective is None else objective
    ids = _pair(lg, algos, seeds, obj, seat_bound(task), options)
    curve = train(lg, ids, iters, eval_every=eval_every, snapshot_every=snapshot_every, seed_offset=seed_offset,
                  view=view)
    inds = [lg.get(p) for p in ids]
    final = evaluate(task, inds, n_slots=n_slots, spec=spec)
    base = evaluate(task, [None] * task.n_seats, n_slots=n_slots, spec=spec)
    out = {"mode": mode, "task": task.id, "objective": inds[0].objective.describe(), "individuals": ids,
           "arch": [i.arch for i in inds], "iters": iters, "n_slots": n_slots, "curve": curve,
           "final": light(final), "defaults": light(base), "league": lg}
    if mode == "competitive":
        # the lagged measure: does a change in one seat's settings precede a change in the other's?
        a = np.array([r["action"][0] for r in curve])
        b = np.array([r["action"][1] for r in curve])
        out["lagged"] = {f"lever{j}": lagged_response(a[:, j], b[:, j]) for j in range(min(a.shape[1], b.shape[1]))}
    if mode == "cooperative":
        solo = [evaluate(task, [inds[0], None], n_slots=n_slots, spec=spec),
                evaluate(task, [None, inds[1]], n_slots=n_slots, spec=spec)]
        out["with_default_partner"] = [light(s) for s in solo]
        out["joint_gain_over_best_single"] = final["joint_net_per_tick"] - max(s["joint_net_per_tick"] for s in solo)
    if mode == "mixed_motive":
        from .profiles import available, profile_individual
        names = available(task)
        lo, hi = ("grab", "restrain") if "restrain" in names else ("grab", "restraint")
        ref = {}
        for nm in (lo, hi):
            ref[nm] = light(evaluate(task, [profile_individual(task, s, nm, spec) for s in range(task.n_seats)],
                                     n_slots=n_slots, spec=spec))
        out["both_grab"], out["both_restrain"] = ref[lo], ref[hi]
        out["position"] = commons_position(final["joint_net_per_tick"], ref[hi]["joint_net_per_tick"],
                                           ref[lo]["joint_net_per_tick"])
    return out


def run_population(task: Task, root: str, *, members: Sequence[dict], iters: int = 30, n_slots: int = 8,
                   matchmaking: str = "uniform", spec: EnvSpec | None = None, seed: int = 0, eval_every: int = 10,
                   snapshot_every: int | None = None, seed_offset: int = 0) -> dict:
    """Population-based self-play: `members` (each `{"id", "algo", "seed", ...learner options}`; different
    algorithms and versions) live in one pool. Every iteration the pool is split into pairs by the matchmaking
    rule (`uniform`: a random perfect matching; `pfsp` / `recent`: each individual in turn draws its opponent
    with `League.sample_opponents`) and every individual learns from its own match. Seats are drawn at random.
    Only for worlds whose seats share one surface."""
    if seat_bound(task):
        raise ValueError("population matchmaking needs a world whose seats have the same surface")
    lg = SelfPlayLeague(root, task, spec, "competition", n_slots, seed=seed)
    ids = []
    for m in members:
        m = dict(m)
        ids.append(lg.new(m.pop("id"), m.pop("algo"), m.pop("seed", 0), m.pop("objective", "own"), **m).id)
    rng = np.random.default_rng([seed, 0x9091])
    curve = []
    k = 0
    for it in range(iters):
        if snapshot_every and it % snapshot_every == 0:
            for p in ids:
                lg.snapshot(p, it)
        if matchmaking == "uniform":
            order = [ids[i] for i in rng.permutation(len(ids))]
            pairs = [order[i:i + 2] for i in range(0, len(order) - 1, 2)]
        else:
            pairs = []
            for p in ids:
                opp = lg.sample_opponents(p, 1, matchmaking, exclude=[x for x in lg.entries if x not in ids], rng=rng)
                if opp:
                    pairs.append([p, opp[0]] if rng.random() < 0.5 else [opp[0], p])
        row = {"it": it, "pairs": pairs, "net": {}}
        for pr in pairs:
            rec = lg.train_match(pr, TRAIN_SEED0 + seed_offset + k)
            k += 1
            for p in pr:
                row["net"].setdefault(p, []).append(rec.outcomes[p]["net_per_tick"])
        row["net"] = {p: float(np.mean(v)) for p, v in row["net"].items()}
        if eval_every and (it + 1) % eval_every == 0:
            row["elo"] = {p: lg.entries[p].elo for p in ids}
        curve.append(row)
    if snapshot_every:
        for p in ids:
            lg.snapshot(p, iters)
    h2h = lg.head_to_head(ids, EVAL_SEEDS, tag="population_final", rate=False)
    return {"mode": "population", "task": task.id, "members": ids, "arch": [lg.info[p]["arch"] for p in ids],
            "matchmaking": matchmaking, "iters": iters, "curve": curve, "head_to_head": h2h,
            "ratings": [r for r in lg.ratings() if r["id"] in ids], "league": lg}


def run_historical(task: Task, root: str, *, algos: Sequence[str] = ("spsa", "spsa"), seeds: Sequence[int] = (1, 2),
                   iters: int = 30, n_slots: int = 8, snapshot_every: int = 5, p_archive: float = 0.5,
                   sampling: str = "pfsp", spec: EnvSpec | None = None, seed: int = 0, seed_offset: int = 0,
                   eval_every: int = 5) -> dict:
    """Historical-policy self-play: two current learners. Every iteration each of them plays one training
    match: with probability `p_archive` against an archived stage (of either lineage) drawn by `sampling`
    (`pfsp`: weight (1 - P[learner beats it])^2 from the recorded matches, `uniform`, or `recent`), otherwise
    against the other current learner. Stages are archived every `snapshot_every` iterations; an archived
    stage never learns. Only for worlds whose seats share one surface."""
    if seat_bound(task):
        raise ValueError("historical matchmaking needs a world whose seats have the same surface")
    lg = SelfPlayLeague(root, task, spec, "competition", n_slots, seed=seed)
    ids = _pair(lg, algos, seeds, "own", False)
    rng = np.random.default_rng([seed, 0x4157])
    curve, k = [], 0
    for it in range(iters):
        if it % snapshot_every == 0:
            for p in ids:
                lg.snapshot(p, it)
        row = {"it": it, "opponents": {}, "net": {}}
        for p in ids:
            arch = lg.archive()
            if arch and rng.random() < p_archive:
                opp = lg.sample_opponents(p, 1, sampling, exclude=[x for x in lg.entries if x not in arch], rng=rng)[0]
            else:
                opp = ids[1 - ids.index(p)]
            pr = [p, opp] if rng.random() < 0.5 else [opp, p]
            rec = lg.play(pr, TRAIN_SEED0 + seed_offset + k, learn=[x == p for x in pr], tag="train")
            k += 1
            row["opponents"][p] = opp
            row["net"][p] = rec.outcomes[p]["net_per_tick"]
        if eval_every and (it + 1) % eval_every == 0:
            ev = lg.play(ids, EVAL_SEEDS[0], tag="eval", rated=False)
            row["eval_net"] = [ev.outcomes[p]["net_per_tick"] for p in ids]
        curve.append(row)
    for p in ids:
        lg.snapshot(p, iters)
    out = {"mode": "historical", "task": task.id, "individuals": ids, "arch": [lg.info[p]["arch"] for p in ids],
           "sampling": sampling, "p_archive": p_archive, "iters": iters, "curve": curve, "league": lg}
    out["history"] = history_ratings(lg, ids[0])
    return out


def history_ratings(league: SelfPlayLeague, root_id: str, seeds: Sequence[int] = EVAL_SEEDS,
                    stages: Sequence[Any] | None = None) -> dict:
    """Does training keep beating its own past? Frozen round robin between the archived stages of one lineage
    (both seatings, the same noise), rated with the league's Elo and TrueSkill from these matches alone.
    Returns the stages in order with their ratings, the win-rate and net matrices, the fraction of
    (earlier, later) pairs in which the later stage wins, and any cycle in the win relation."""
    snaps = [s for s in league.archive(of=root_id, include_heldout=False) if league.entries[s].parent == root_id]
    if stages is not None:
        snaps = [s for s in snaps if league.info[s]["stage"] in set(stages)]
    snaps.sort(key=lambda s: (float(league.info[s]["stage"]) if str(league.info[s]["stage"]).lstrip("-").isdigit()
                              else 1e18))
    h = league.head_to_head(snaps, seeds, tag=f"history:{root_id}", rate=True)
    league.rerate(tags=[f"history:{root_id}"])
    win = np.array(h["win"], dtype=float)
    later = [win[j, i] for i in range(len(snaps)) for j in range(i + 1, len(snaps))]
    out = {"lineage": root_id, "stages": [league.info[s]["stage"] for s in snaps], "ids": snaps,
           "elo": [league.entries[s].elo for s in snaps], "mu": [league.entries[s].mu for s in snaps],
           "win": h["win"], "net": h["net"], "cycles": h["cycles"],
           "later_beats_earlier": float(np.mean([w > 0.5 for w in later])) if later else None,
           "last_beats_all_earlier": bool(all(win[-1, i] > 0.5 for i in range(len(snaps) - 1))) if len(snaps) > 1 else None}
    league.rerate()                                   # back to the ratings of all rated matches
    return out


def run_cross_play(task: Task, root: str, *, pairs: Sequence[dict], iters: int = 30, n_slots: int = 8,
                   objective: Any = "own", spec: EnvSpec | None = None) -> dict:
    """Train `pairs` (each `{"algos": (a, b), "seeds": (s0, s1)}`) independently, each in its own league
    directory under `root`, then evaluate the full cross-play matrix."""
    import os
    runs = []
    for k, pr in enumerate(pairs):
        mode = "cooperative" if objective == "joint" else "competitive"
        runs.append(run_pair(mode, task, os.path.join(root, f"pair{k}"), algos=pr["algos"], seeds=pr["seeds"],
                             iters=iters, n_slots=n_slots, objective=objective, spec=spec, seed_offset=1000 * k,
                             snapshot_every=None))
    s0 = [r["league"].get(r["individuals"][0]) for r in runs]
    s1 = [r["league"].get(r["individuals"][1]) for r in runs]
    for k, (a, b) in enumerate(zip(s0, s1)):         # ids must be distinct across the independently made pairs
        a.id, b.id = f"p{k}:{a.id}", f"p{k}:{b.id}"
    cp = cross_play(task, s0, s1, n_slots=n_slots, spec=spec)
    return {"mode": "cross_play", "task": task.id, "pairs": [dict(p) for p in pairs], "iters": iters,
            "matrix": cp, "runs": runs}


def run_mode(name: str, task: Task, root: str, **kw) -> dict:
    """Run one of the six modes (see the module docstring). The result's `league` (or `runs[i]["league"]`) is the
    live league object; everything else is JSON-serialisable."""
    if name not in MODES:
        raise ValueError(f"unknown mode {name!r}; modes: {MODE_NAMES}")
    if name in ("competitive", "cooperative", "mixed_motive"):
        return run_pair(name, task, root, **kw)
    if name == "population":
        return run_population(task, root, **kw)
    if name == "historical":
        return run_historical(task, root, **kw)
    return run_cross_play(task, root, **kw)


def strip(result: dict) -> dict:
    """A mode result without live objects (for JSON)."""
    out = {}
    for k, v in result.items():
        if k in ("league", "individual"):
            continue
        if k == "runs":
            out[k] = [strip(r) for r in v]
        else:
            out[k] = v
    return out
