"""Counterfactual creation experiments on the one population (Gates J and L; evaluator side).

`creation_world(seed, n)` is a creation world (`popworld.creation_spec`) with three work families:

* `schedule.jobs`, a published format with 1-3 incumbent SaaS firms (adaptive, `commerce.rivals`);
* a procedurally generated category (`usefulness` family `allocate`, a fresh hidden rule per world) that nobody
  documents or serves at genesis;
* a two-step pipeline `route.shortest` -> `pack.knapsack`, both published formats that no firm serves, where the
  second step is a small part of the hand work (so nobody looks for help with it while the first step is done by
  hand).

Arms (same spec and seed; they differ only in what the scripted agents do, so every difference is caused by it):

* `factual`  - a scripted novelty creator (`creators.NoveltyCreator`) reads the public help board from day 1 and
  builds services for tasks people ask about (it skips the formats the SaaS founder covers); a scripted SaaS
  founder (`commerce.reference.ReferenceSaaS`) launches `schedule.jobs` and `route.shortest` products on day
  `saas_day`;
* `no_saas`  - the SaaS founder never launches (removal of the SaaS products);
* `no_novelty` - the novelty creator never builds the generated-category service (removal of that service);
* `garbage`  - both agents deploy input-ignoring programs with the same interfaces (measured behaviour changed,
  everything else equal).

`run_arm` drives an arm day by day and records, every day, the evaluator's needs labelling
(`innovation.relations.label`), the population's work resource use, incumbents' public prices and the rivals'
actions; `mine_chains` extracts multi-stage chains generically from that history and the event log (need ->
public request -> agent artifact -> adoption -> relief, and artifact -> competitor response), and
`confirm` checks each chain's later stages against the arm where its first change was removed.

All agents here are LABELLED SCRIPTED (controller `scripted:...`).
"""
from __future__ import annotations

import numpy as np

from ..core.kernel import World

DAY = 86400
NOVEL_TITLE = "Allocate a budget across options"
SAAS_FORMATS = ("schedule.jobs", "route.shortest")


def creation_world_spec(seed: int, n: int = 900):
    from .popworld import creation_spec
    spec = creation_spec(seed, n=n, published=("schedule.jobs",), novel=("allocate",), novel_share=0.8)
    for m in spec.modules:
        if m["name"] == "work":
            fams = list(m["config"]["families"])
            fams.append({"need": "route.shortest", "then": "pack.knapsack", "share": 1.0, "tasks_day": [1.0, 2.5],
                         "diy_min": [20.0, 45.0], "then_diy_min": [3.0, 7.0], "size": [0.2, 0.5],
                         "standard": [0.9, 0.99], "budget_min": [60.0, 200.0]})
            m["config"] = dict(m["config"], families=fams)
    return spec


def _agents(arm: str):
    from ..commerce.reference import ReferenceSaaS
    from .creators import NoveltyCreator, PROGRAMS  # noqa: F401
    from ..economy.needs import NEEDS
    skip = tuple(NEEDS[f].title for f in SAAS_FORMATS)
    if arm == "no_novelty":
        skip = skip + (NOVEL_TITLE,)
    prog = "garbage" if arm == "garbage" else "good"
    mk = NoveltyCreator(handle="maker", skip=skip, program=prog)
    saas = None if arm == "no_saas" else ReferenceSaaS(handle="saasco", ads_budget_day=2_000, program=prog)
    return mk, saas


def snapshot_day(world) -> dict:
    """Evaluator observation of one day."""
    from ..economy.work import resource_use
    from . import relations as R
    st = world.state
    needs = R.label(world)
    prices = {}
    for pid, p in st.scan("products"):
        if p["status"] == "launched" and p.get("need"):
            ms = [pl["price_cents"] for pl in world.modules["subs"].public_plans(st, pid) if pl["interval"] == "month"]
            if ms:
                prices[pid] = min(ms)
    live = {}
    for _, s in st.scan("sub_subs"):
        if s["status"] in ("trialing", "active", "past_due"):
            live[s["merchant"]] = live.get(s["merchant"], 0) + 1
    return {"t": int(world.now), "needs": {n["id"]: {"rule": n["rule"], "holders": len(n["holders"]),
                                                     "target": n["target"], "path": n["path"]} for n in needs},
            "resources": resource_use(world), "prices": prices, "live_subs": live}


def run_arm(seed: int, arm: str, *, n: int = 900, days: int = 30, saas_day: int = 8) -> dict:
    import crucible
    crucible.load_all_modules()
    w = World.create(creation_world_spec(seed, n=n))
    mk, saas = _agents(arm)
    mk.setup(w)
    history = [snapshot_day(w)]
    for d in range(1, days + 1):
        w.advance_time(d * DAY)
        if saas is not None and d == saas_day:
            saas.setup(w, list(SAAS_FORMATS))
        if saas is not None and d >= saas_day:
            saas.day(w)
        mk.day(w)
        history.append(snapshot_day(w))
    return {"arm": arm, "seed": seed, "world": w, "maker": mk, "saas": saas, "history": history}


# ====================================================================== generic chain mining
def _first(history, pred):
    for h in history:
        if pred(h):
            return h["t"]
    return None


def mine_chains(res: dict) -> list[dict]:
    """Chains of at least three stages across subsystems, found generically:

    * for every need labelled after genesis or present at genesis and later relieved: [need appears (work
      activities)] -> [public request about its task (social: forum)] -> [an agent's provider of the target is
      created (innovation / cloud)] -> [holders adopt it (work)] -> [the need is relieved (work)];
      when the need's own appearance follows an agent provider's adoption for another step of the same activity,
      that earlier adoption is the chain's first stage (adoption creating a need);
    * for every agent provider of a format that incumbents serve: [provider launched] -> [incumbents lose live
      subscriptions (commerce)] -> [an incumbent acts on it: price change or engineering (rivals)]."""
    from . import measure as M
    w, history = res["world"], res["history"]
    st = w.state
    log = w.log
    prods = [p for p in M.agent_products(w)]
    chains = []
    t_gen = history[0]["t"]
    ever = sorted({nid for h in history for nid in h["needs"]})
    for nid in ever:
        t_on = _first(history, lambda h: nid in h["needs"])
        tgt = next(h["needs"][nid]["target"] for h in history if nid in h["needs"])
        need_name = tgt.split("/", 1)[1]
        stages = [{"name": f"need {nid}", "subsystem": "work activities", "t": t_on}]
        # the activity whose other step was automated just before: adoption creating this need
        act = next(h["needs"][nid]["path"][0]["src"] for h in history if nid in h["needs"])
        if t_on is not None and t_on > t_gen + DAY and "+" in act:
            first_step = act.split("/", 1)[1].split("+")[0]
            t_adopt = None
            for e in log:
                if e.type == "work.rely" and e.payload.get("active") and e.payload.get("need") == first_step:
                    t_adopt = e.t
                    break
            if t_adopt is not None and t_adopt <= t_on:
                stages.insert(0, {"name": f"adoption of a {first_step} provider", "subsystem": "commerce", "t": t_adopt})
        t_ask = None
        title = w.modules["work"].task_text(need_name)[:60]
        for e in log:
            if e.type == "forum.post" and e.t >= (t_on or 0) and e.payload["title"].startswith(
                    "Looking for a service: " + title[:40]):
                t_ask = e.t
                break
        if t_ask is not None:
            stages.append({"name": "public request for help", "subsystem": "social (forum)", "t": t_ask})
        relied_for = {r["prov"] for _, r in st.scan("work_exp") if r["step_need"] == need_name}
        mine = [p for p in prods if M._provider_ids(w, p) & relied_for and p["created_t"] >= (t_on or 0)]
        if mine:
            p = min(mine, key=lambda x: x["created_t"])
            stages.append({"name": f"agent artifact {p['id']}", "subsystem": "innovation", "t": p["created_t"],
                           "creator_kind": p["creator_kind"]})
            provs = M._provider_ids(w, p)
            t_ad = next((e.t for e in log if e.type == "work.rely" and e.payload.get("active")
                         and e.payload["prov"] in provs), None)
            if t_ad is not None:
                stages.append({"name": "adoption by holders", "subsystem": "work activities", "t": t_ad})
                t_off = _first([h for h in history if h["t"] > t_ad], lambda h: nid not in h["needs"])
                if t_off is not None:
                    stages.append({"name": "need relieved", "subsystem": "work activities", "t": t_off})
        chains.append(_chain(f"need:{nid}", stages))
    # competition: agent provider of an incumbent-served format -> incumbents lose -> incumbents act
    inc = {o for o, r in st.scan("orgs") if (r.get("info") or {}).get("trusted_runtime")}
    for p in prods:
        fmt = p.get("subcategory")
        rivals = [pid for pid, x in st.scan("products") if x["owner"] in inc and x.get("need") == fmt]
        if not rivals:
            continue
        owners = {st.tables["products"][r]["owner"] for r in rivals}
        stages = [{"name": f"agent artifact {p['id']}", "subsystem": "innovation", "t": p["created_t"],
                   "creator_kind": p["creator_kind"]}]
        peak = {o: 0 for o in owners}
        t_loss = None
        for h in history:
            if h["t"] < p["created_t"]:
                for o in owners:
                    peak[o] = max(peak[o], h["live_subs"].get(o, 0))
                continue
            if any(peak[o] >= 3 and h["live_subs"].get(o, 0) <= 0.85 * peak[o] for o in owners):
                t_loss = h["t"]
                break
        if t_loss is not None:
            stages.append({"name": "incumbents lose subscribers", "subsystem": "commerce", "t": t_loss})
            t_act = next((e.t for e in log if e.type == "rivals.observed" and e.payload["org"] in owners
                          and e.t >= t_loss and any(a[0] in ("price", "engineering_started", "shipped_tier")
                                                    for a in e.payload["actions"])), None)
            if t_act is not None:
                stages.append({"name": "incumbent responds (price or engineering)", "subsystem": "rivals",
                               "t": t_act})
        chains.append(_chain(f"competition:{p['id']}", stages))
    return chains


def _chain(name, stages) -> dict:
    ok = [s for s in stages if s.get("t") is not None]
    ordered = all(a["t"] <= b["t"] for a, b in zip(ok, ok[1:]))
    return {"chain": name, "stages": stages, "n_stages": len(ok), "ordered": ordered,
            "subsystems": sorted({s["subsystem"] for s in ok}),
            "agent_artifact": any(s.get("creator_kind") for s in ok),
            "social": any("social" in s["subsystem"] for s in ok),
            "technical": any(s["subsystem"] in ("innovation", "rivals") for s in ok)}


def outcome(res: dict) -> dict:
    """What the counterfactual comparisons look at, per arm."""
    from . import measure as M
    w = res["world"]
    h = res["history"]
    last = h[-1]
    prods = {p["id"]: p for p in M.agent_products(w)}
    out = {"diy_minutes": last["resources"].get("diy_minutes", 0.0), "minutes": last["resources"].get("minutes", 0.0),
           "lost": last["resources"].get("lost", 0.0), "needs": sorted(last["needs"]),
           "needs_ever": sorted({nid for x in h for nid in x["needs"]}), "products": {}}
    for pid, p in prods.items():
        out["products"][pid] = {"users": len(M.users(w, p)), "adopters": len(M.adopters(w, p)),
                                "discoverers": len(M.discoverers(w, p)), "tx": M.independent_transactions(w, p),
                                "triers": _triers(w, p), "kept": _kept(w, p)}
    inc_live = {}
    st = w.state
    inc = {o for o, r in st.scan("orgs") if (r.get("info") or {}).get("trusted_runtime")}
    for o in inc:
        inc_live[o] = sum(1 for _, s in st.scan("sub_subs") if s["merchant"] == o and s["status"] in
                          ("trialing", "active", "past_due"))
    out["incumbent_live_subs"] = inc_live
    out["incumbent_price_changes"] = sum(1 for e in w.log if e.type == "rivals.observed"
                                         for a in e.payload["actions"] if a[0] == "price")
    out["incumbent_cu"] = float(sum(r["cu"] for k, r in st.scan("svc_days")
                                    if any(st.tables["hosts"].get(r["host"], {}).get("owner") == o for o in inc)))
    return out


def _triers(w, product) -> list[str]:
    from . import measure as M
    provs = M._provider_ids(w, product)
    return sorted({f"p{r['i']}" for _, r in w.state.scan("work_exp") if r["prov"] in provs and r["uses"] > 0})


def _kept(w, product) -> list[str]:
    """Individuals who tried the product and still rely on it (measured behaviour -> adoption)."""
    from . import measure as M
    provs = M._provider_ids(w, product)
    return sorted({f"p{r['i']}" for _, r in w.state.scan("work_exp") if r["prov"] in provs and r["uses"] > 0
                   and r["active"]})
