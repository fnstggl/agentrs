"""Reference for the W9 migration of experiments/predict.collect_support to the canonical loop (W1 change request):
the support loop as it was before the migration, verbatim except for module prefixes. Test-only; the package itself
has no act -> execute -> advance loop outside runtime.py (tests/cam_unified/integration/test_single_loop.py)."""
from __future__ import annotations

import random

import numpy as np

from superlearner.cam_unified.experiments import mechrun as M
from superlearner.cam_unified.experiments import predict as P


def legacy_collect_support(tasks, *, k_max: int = 64, n_queries: int = 12, design_seed: int = 0,
                    condition: str = "A", delay: int = 3, hide_frac: float = 0.5, notes_m: int = 0,
                    surface=None, ks=P.K_SET) -> list:
    """The pre-migration support loop of predict.collect_support (its own act -> execute -> advance loop), kept
    verbatim as the reference the canonical-loop version must reproduce exactly."""
    cond = condition.upper()
    world = M.make_world(tasks, surface=surface, seed=design_seed)
    dp = M.DelayedPartial(world, delay=delay, hide_frac=hide_frac, seed=design_seed) if cond == "D" else None
    ob = dp.reset() if dp is not None else world.adapter.reset()
    B = len(tasks)
    op = world.operator
    notes_obs = [[] for _ in range(B)]
    notes_raw = [[] for _ in range(B)]
    if notes_m:
        pn = M.PredecessorNotes()
        for b in range(B):
            r = random.Random(f"notes|{tasks[b].id}|{design_seed}")
            obs = pn.notes(world, b, notes_m, r)
            notes_obs[b] = obs
            r = random.Random(f"notes|{tasks[b].id}|{design_seed}")      # replay the same draws for the numbers
            for _ in range(notes_m):
                e = op.envs[b]
                j = r.randrange(len(e.lever_spec))
                lv = e.lever_spec[j]
                v = r.uniform(lv.lo, lv.hi)
                v = float(int(round(v))) if lv.integer else round(v, 3)
                base, after = pn._experiment(e.task, op.spec, j, v)
                notes_raw[b].append((j, v, base, after))
                r.randrange(3)                                         # the wording draw
    obs_now = [list(ob.observations[b]) + notes_obs[b] for b in range(B)]
    rngs = [random.Random(f"design|{t.id}|{design_seed}") for t in tasks]
    qrngs = [random.Random(f"query|{t.id}|{design_seed}") for t in tasks]
    envs = op.envs
    L = [len(e.lever_spec) for e in envs]
    meta = [P._env_meta(world, b, dp) for b in range(B)]
    steps = [[] for _ in range(B)]
    Xs = [[] for _ in range(B)]
    Gs = [[] for _ in range(B)]
    queries = [dict() for _ in range(B)]
    think = M.THINK_S / M.DAY
    tools0 = world.adapter.available_tools()
    for t in range(k_max + 1):
        if t in ks:
            for b in range(B):
                queries[b][t] = P.make_queries(world, b, qrngs[b], n_queries, tools0[b], meta[b])
        if t == k_max:
            break
        acts = []
        for b, e in enumerate(envs):
            j = rngs[b].randrange(L[b])
            lv = e.lever_spec[j]
            v = rngs[b].uniform(lv.lo, lv.hi)
            acts.append(P._set_action(e.lever_shown[j], v, lv.integer, tools0[b]))
        res = world.adapter.execute(acts)
        dur = [(r.duration if r is not None else 0.0) + think for r in res]
        adv = dp.advance(dur) if dp is not None else world.adapter.advance_time(dur)
        rew = dp.take_rewards(adv.done) if dp is not None else world.evaluator.take_rewards()
        for b, e in enumerate(envs):
            lo, hi, _, gm = meta[b]
            nxt = list(res[b].observations) + list(adv.observations[b])
            steps[b].append({"obs": obs_now[b], "action": acts[b], "result": res[b], "next_obs": nxt,
                             "reward": float(rew[b]), "t": float(ob.t[b]) if t == 0 else float(prev_t[b])})
            Xs[b].append([P._norm(e.raw[j], lo[j], hi[j]) for j in range(L[b])])
            last = e.env.sims[0].last[0]
            g = np.array([float(last["readings"][n]) for n in e.gauge_engine], dtype=np.float64)
            Gs[b].append(np.sign(g) * np.log1p(np.abs(g)) * gm)
            obs_now[b] = nxt
        prev_t = list(adv.t)
        if any(adv.done):
            raise RuntimeError("a support episode ended before k_max decisions; use tasks with horizon > k_max")
    out = []
    for b, task in enumerate(tasks):
        e = envs[b]
        lo, hi, x0, gm = meta[b]
        gap = (task.reference_net - task.default_net) / max(1e-9, float(op.envs[b].goal.meta["reward_scale_cents"])) \
            if task.validated else float("nan")
        out.append(P.SupportEpisode(task, cond, list(e.lever_shown), lo, hi, [lv.integer for lv in e.lever_spec], x0,
                                  np.array(Xs[b]), np.array(Gs[b]), gm, np.array(world.rewards[b][:k_max]),
                                  delay if cond == "D" else 0, steps[b], obs_now[b], queries[b], float(gap),
                                  notes_raw[b], list(tools0[b]), world.adapter.goals()[b]))
    return out
