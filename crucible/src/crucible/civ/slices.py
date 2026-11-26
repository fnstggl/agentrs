"""The seven Phase 3 vertical slices (MASTER_SPEC Part 19, Phase 3; SP-19.3), executed end to end and recorded.

Clarification 1 replaces CAM by an architecture-independent learner: slices 5 and 6 run `crucible.experience`'s
`ReferenceLearner` (a small recurrent network trained by its own update rule) through the public experience
interface (`crucible.experience.registry.make` + `loop.run_learner`), exactly as any external learner would.

    1 discover_consume   an individual of the population discovers a content item through a platform channel (feed,
                         search, follows, a community...) and consumes it (exact audience accounting)
    2 agent_software     software written and deployed by an agent (a scripted reference inventor, labelled) is used
                         by other participants: individuals of the population delegate steps of their own work to it
    3 social_purchase    social discovery leads to a genuine product transaction: an individual learns of a listing
                         from a contact's purchase (word of mouth) and later buys that listing with its own money
    4 neural_adapts      a small neural participant adapts through experience: a population policy group's parameters
                         change by learning from its participants' outcomes (recorded weight hashes differ)
    5 learner_world      the architecture-independent learner acts in a generated semantic world (a Tier 3
                         civilisation): its actions are executed and change world state
    6 learner_obs_only   the learner receives observation-only experience: the environment's reward is 0.0 at every
                         step, observations carry consequences, and the learner still updates its parameters from its
                         own signal
    7 causal_mechanism   a generated technical world (a Tier 1 `crucible.rapid` family) exhibits a causal mechanism
                         that is validated independently, by controlled interventions on copies of the world
                         (`crucible.rapid.claims`, evaluator side)

`run_all()` executes them and returns one record per slice: whether it held, what was measured, and one traced
example (ids, sequence numbers) a reader can look up in the run.
"""
from __future__ import annotations

import time

import numpy as np

DAY, HOUR = 86400, 3600
SLICES = ("discover_consume", "agent_software", "social_purchase", "neural_adapts", "learner_world",
          "learner_obs_only", "causal_mechanism")
CIV_RUN = {"profile": "balanced", "seed": 61, "warm_days": 2, "days": 6}
LEARNER_RUN = {"profile": "balanced", "seed": 62, "steps": 240, "think_s": 1800}
RAPID_FAMILIES = ("net_routing", "team_commons")


def _civ_slices(p: dict) -> dict:
    """Slices 1-4 on one civilisation run (the scripted inventor of slice 2 is admitted at the warm-up's end)."""
    import torch
    torch.set_num_threads(1)
    from ..substrate import columns, ids
    from . import experiments as X, generator
    w = generator.generate(p["profile"], p["seed"])
    w.advance_time(int(p["warm_days"]) * DAY)
    ctx = {"arm": "factual", "seed": p["seed"], "t0": w.now, "seq0": w.seq}
    X._invention_setup(w, ctx)
    s0 = w.seq
    t = w.now
    t1 = t + int(p["days"]) * DAY
    while t < t1:
        t = min(t1, (t // HOUR + 1) * HOUR)
        w.advance_time(t)
        ctx["agent"].act(w)
    evs = w.events_range(0)
    out = {}
    # 1. discovery and consumption, measured as Gate C measures it (crucible.social.experiments.decision_rows over
    #    the recorded impression and consumption batches; the channel of a view is that of the viewer's first
    #    impression of the item)
    from ..social import audience as AUD, experiments as SX
    views, imps = SX.decision_rows(w, "consume", 0), SX.decision_rows(w, "imp", 0)
    n_views = int(views["is_view"].sum()) if "is_view" in views else 0
    ex, chans = None, {}
    if n_views and imps["item"].size:
        first: dict = {}
        for k in np.argsort(imps["seq"], kind="stable"):
            first.setdefault((int(imps["viewer"][k]), int(imps["item"][k])), (int(imps["src"][k]),
                                                                                 int(imps["seq"][k])))
        for k in np.flatnonzero(views["is_view"]):
            key = (int(views["viewer"][k]), int(views["item"][k]))
            if key in first:
                ch = AUD.SOURCES[first[key][0]]
                chans[ch] = chans.get(ch, 0) + 1
                if ex is None:
                    ex = {"viewer": key[0], "item": key[1], "channel": ch, "impression_seq": first[key][1],
                          "view_seq": int(views["seq"][k]), "watch_s": round(float(views["watch_s"][k]), 1)}
    out["discover_consume"] = {"ok": ex is not None, "views": n_views, "views_by_first_channel": chans,
                               "example": ex}
    # 2. agent software used by others
    rows = X._inv_rows(w, ctx)
    users = sorted({int(r["i"]) for r in rows if int(r["uses"]) > 0})
    out["agent_software"] = {"ok": bool(users), "agent": ctx["agent"].actor, "scripted_agent": True,
                             "declaration": ctx["agent"].declaration, "deployed_host": X.INVENTOR["host"],
                             "users": len(users), "uses": sum(int(r["uses"]) for r in rows),
                             "revenue_cents": X.m_inv_revenue(w, ctx["t0"], t1, ctx),
                             "example": {"individual": users[0]} if users else None}
    # 3. word of mouth -> the learner buys that listing with its own money
    wom = [e for e in evs if e.type == "goods.wom"]
    sold = {}
    for e in evs:
        if e.type == "make.retail_sold":
            i = ids.index(str(e.payload.get("buyer", "")))
            if i is not None:
                sold.setdefault((i, e.payload.get("retail") or e.payload.get("id")), []).append(e)
    ex = None
    n_follow = 0
    for e in wom:
        key = (int(e.payload["learner"]), e.payload["retail"])
        later = [x for x in sold.get(key, []) if x.seq > e.seq]
        if later:
            n_follow += 1
            if ex is None:
                ex = {"wom_seq": e.seq, "learner": key[0], "source": int(e.payload["source"]), "retail": key[1],
                      "purchase_seq": later[0].seq, "cents": later[0].payload.get("cents")}
    out["social_purchase"] = {"ok": n_follow > 0, "word_of_mouth_events": len(wom),
                              "purchases_after_word_of_mouth": n_follow, "example": ex}
    # 4. a neural policy group of the population learned
    ups = [e for e in evs if e.type == "pop.policy_update"]
    hashes = {}
    for e in ups:
        hashes.setdefault(e.payload["group"], []).append(e.payload["weights_hash"])
    changed = {g: len(set(h)) for g, h in hashes.items() if len(set(h)) >= 2}
    out["neural_adapts"] = {"ok": bool(changed), "policy_updates": len(ups), "groups_with_changed_weights": changed,
                            "example": {"seq": ups[-1].seq, **{k: ups[-1].payload.get(k) for k in
                                                               ("group", "step", "loss", "n_examples")}}
                            if ups else None}
    out["_run"] = {"events": w.seq, "proc_failed": sum(1 for e in evs if e.type == "proc.failed"),
                   "invariants": w.check_invariants()[:5], "inventor_from_seq": s0}
    w.close()
    return out


def _learner_slices(p: dict) -> dict:
    """Slices 5 and 6: the reference learner on a Tier 3 civilisation through the public experience interface."""
    import torch
    torch.set_num_threads(1)
    from ..experience import registry
    from ..experience.loop import run_learner
    from ..experience.reference import ReferenceLearner
    env = registry.make(3, p["profile"], p["seed"], agents=["learner"])
    mode = getattr(env, "reward_mode", None)
    L = ReferenceLearner(seed=p["seed"])
    h0 = L.params_hash()
    s = run_learner(env, L, int(p["steps"]), think_s=int(p["think_s"]), update_every=32)
    h1 = L.params_hash()
    kinds = sorted(s.get("kinds") or [])
    out = {"learner_world": {"ok": s["ok"] > 0 and s["world_changes"] > 0, "steps": s["steps"],
                             "actions": s["actions"], "ok_actions": s["ok"], "rejected": s["rejected"],
                             "world_changes": s["world_changes"], "learner": "crucible.experience.ReferenceLearner",
                             "world": f"tier3:{p['profile']}"},
           "learner_obs_only": {"ok": mode == "observation_only" and s["nonzero_rewards"] == 0 and
                                s["reward_sum"] == 0.0 and s["params_changed_updates"] > 0 and h1 != h0,
                                "reward_mode": mode, "reward_sum": s["reward_sum"],
                                "nonzero_rewards": s["nonzero_rewards"], "updates": s["updates"],
                                "params_changed_updates": s["params_changed_updates"],
                                "params_hash_before": h0, "params_hash_after": h1,
                                "signal": L.signal_mode(), "observation_kinds": kinds[:20]}}
    env.close()
    return out


def _causal_slice() -> dict:
    """Slice 7: every mechanism claim of a generated Tier 1 world holds under controlled intervention."""
    from ..rapid import claims
    rows = {}
    for fam in RAPID_FAMILIES:
        for seed in (0, 1):
            r = claims.run(fam, seed)
            rows[f"{fam}/{seed}"] = {k: bool(v.get("ok")) for k, v in r.items()}
    held = {k: v for k, v in rows.items() if v and all(v.values())}
    return {"causal_mechanism": {"ok": bool(held), "worlds_checked": len(rows), "worlds_all_claims_hold": len(held),
                                 "claims": rows, "validation": "crucible.rapid.claims (evaluator-side controlled "
                                                               "interventions on copies of the world)"}}


def run_all(civ: dict | None = None, learner: dict | None = None) -> dict:
    t = time.time()
    out = {}
    out.update(_civ_slices({**CIV_RUN, **(civ or {})}))
    out.update(_learner_slices({**LEARNER_RUN, **(learner or {})}))
    out.update(_causal_slice())
    out["vertical_slices_executed"] = sum(1 for k in SLICES if out.get(k, {}).get("ok"))
    out["wall_s"] = round(time.time() - t, 1)
    return out
