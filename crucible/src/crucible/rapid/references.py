"""Scripted public-information reference participants (MASTER_SPEC Part 7.4 "scripted positive controls").

Each reference plays a whole episode through a `PublicView` - actions, observations, tool schemas, the public spec,
nothing hidden - with a simple competent strategy: explore then commit on observed net income, designed
experiments with a fitted model, probe-then-dig, test-then-repair, and so on. Admission uses them to show that a
world is solvable by information a participant can get (not only by the private oracle), and that it is not
already saturated by a simple script.

`play(env)` runs the family's reference on `env` (a fresh world) and returns the evaluator outcome.
"""
from __future__ import annotations

import numpy as np

from .engines.search import cell_name
from .identify import PublicView, _fit_predict, _net, experiment_grid


def _run(view, fn):
    while not view.done:
        view.step(fn(view))


# ====================================================================== generic explore-then-commit
def candidates(space) -> list[dict]:
    out: list[dict] = []
    if "levers" in space.kinds:
        props = space.kinds["levers"]["properties"]["values"]["properties"]
        for n, p in sorted(props.items()):
            for f in (0.1, 0.35, 0.65, 0.9):
                out.append({"kind": "levers", "values": {n: p["minimum"] + f * (p["maximum"] - p["minimum"])}})
    if "allocate" in space.kinds:
        names = sorted(space.kinds["allocate"]["properties"]["amounts"]["properties"])
        tot = float(space.limits.get("allocate_total", 1.0))
        out.append({"kind": "allocate", "amounts": {n: tot / len(names) for n in names}})
        out += [{"kind": "allocate", "amounts": {n: tot}} for n in names[:6]]
        out.append({"kind": "allocate", "amounts": {n: tot / (2 * len(names)) for n in names}})
    if "discrete" in space.kinds:
        for d, opts in sorted(space.limits.get("decisions", {}).items()):
            out += [{"kind": "discrete", "decision": d, "choice": o} for o in opts[:8]]
    return out


def explore_commit(view, explore_frac: float = 0.45) -> dict | None:
    """Try candidate actions in turn (levers one at a time, then the best value of each lever together), measure
    observed net income per tick, then commit to the best for the rest of the episode. Returns the action it
    committed to."""
    sp = view.action_space()
    cands = candidates(sp)
    H = view.horizon
    arm = max(2, int(H * explore_frac) // max(1, len(cands)))
    score = []
    for a in cands:
        if view.t + arm >= H:
            break
        vals = []
        for k in range(arm):
            r = view.step(a)
            if k >= arm // 2:
                vals.append(_net(r["observation"]))
        score.append(float(np.mean(vals)))
    if not score:
        _run(view, lambda v: None)
        return None
    best = cands[int(np.argmax(score))]
    if "levers" in sp.kinds:                                 # combine the best level of each lever
        per: dict = {}
        for a, s in zip(cands, score):
            if a["kind"] == "levers":
                (n, v), = a["values"].items()
                if n not in per or s > per[n][1]:
                    per[n] = (v, s)
        combo = {"kind": "levers", "values": {n: v for n, (v, _) in per.items()}}
        trial = []
        for _ in range(arm):
            if not view.done:
                trial.append(_net(view.step(combo)["observation"]))
        if trial and np.mean(trial[len(trial) // 2:]) >= max(score):
            best = combo
    _run(view, lambda v: best)
    return best


# ====================================================================== family-specific references
def science(view, env):
    cfg = env._config
    x1, x2 = [cfg["nodes"][env._engine.idx[n]]["lever"]["name"] for n in cfg["claim"]["levers"]]
    g = cfg["objective"]["gauge"]
    inst = "lab" if cfg.get("experiment", {}).get("instruments") else None
    if inst:
        return science_additive(view, env, x1, x2, g)
    X, y = experiment_grid(view, x1, x2, g, grid=5 if inst is None else 4, reps=2 if inst is None else 1,
                           instrument=inst)
    held = cfg["objective"]["heldout"]
    if len(y) >= 6:
        Xq = np.asarray([[c[x1], c[x2]] for c in held])
        pred = _fit_predict(X, y, Xq)
        view.step({"kind": "message", "to": "journal",
                   "content": {"predictions": [{"set": c, "value": float(v)} for c, v in zip(held, pred)]}})
    _run(view, lambda v: None)


def science_additive(view, env, x1, x2, g):
    """Sweep each setting on its own with the precise instrument (locating a jump in the second), fit an additive
    model - a quadratic in the first, a step in the second - and predict."""
    def measure(a, b):
        r = view.step({"kind": "experiment", "design": {"set": {x1: float(a), x2: float(b)}, "measure": [g],
                                                        "ticks": 1, "replicates": 1, "instrument": "lab"}})
        return r["result"]["results"]["measured"][g]["mean"] if r["result"]["ok"] else None
    b_grid = np.linspace(0.0, 6.0, 7)
    yb = [measure(3.0, b) for b in b_grid]
    a_grid = np.array([0.0, 2.0, 4.0, 6.0])
    ya = [measure(a, 1.0) for a in a_grid]
    held = env._config["objective"]["heldout"]
    if any(v is None for v in yb + ya):
        _run(view, lambda v: None)
        return
    jumps = np.diff(yb)
    k = int(np.argmax(np.abs(jumps)))
    theta = 0.5 * (b_grid[k] + b_grid[k + 1])
    step = float(jumps[k])
    low_b = 1.0 >= theta
    coef = np.polyfit(a_grid, ya, 2)
    preds = []
    for c in held:
        base = float(np.polyval(coef, c[x1]))
        s = (step if c[x2] >= theta else 0.0) - (step if low_b else 0.0)
        preds.append({"set": c, "value": base + s})
    view.step({"kind": "message", "to": "journal", "content": {"predictions": preds}})
    _run(view, lambda v: None)


def field(view, env):
    G = env._config["G"]
    dig_cost = env._config["dig_cost"]
    est = {}
    for x in range(0, G, 2):
        for y in range(0, G, 2):
            if view.done:
                return
            r = view.step({"kind": "query", "target": "probe", "args": {"cell": cell_name(x, y)}})
            if r["result"]["ok"]:
                est[(x, y)] = r["result"]["results"]["value"]
    dug = set()
    while not view.done:
        best = max((c for c in est if c not in dug), key=lambda c: est[c], default=None)
        if best is None or est[best] < 1.5 * dig_cost:
            break
        view.step({"kind": "discrete", "decision": "dig", "choice": cell_name(*best)})
        dug.add(best)
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nb = (best[0] + dx, best[1] + dy)
            if 0 <= nb[0] < G and 0 <= nb[1] < G and nb not in est and not view.done:
                r = view.step({"kind": "query", "target": "probe", "args": {"cell": cell_name(*nb)}})
                if r["result"]["ok"]:
                    est[nb] = r["result"]["results"]["value"]
    _run(view, lambda v: None)


def distractors(view, env):
    c = env._config
    ents = view.step(None)["observation"]["entities"]
    feats = {x["id"]: x["features"] for x in ents if x.get("kind") == "item"}
    names = sorted(feats)
    probed = {}
    for n in names[: max(8, len(names) // 2)]:
        if view.done:
            return
        r = view.step({"kind": "query", "target": "probe", "args": {"item": n}})
        if r["result"]["ok"]:
            probed[n] = r["result"]["results"]["value"]
    F = np.asarray([feats[n] for n in probed])
    y = np.asarray(list(probed.values()))
    lam = 1.0
    Xa = np.c_[np.ones(len(y)), F]
    beta = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ y)
    pred = {n: float(np.r_[1.0, feats[n]] @ beta) for n in names}
    for n in names:
        if n in probed:
            pred[n] = probed[n]
    for n in sorted(names, key=lambda n: -pred[n]):
        if view.done or pred[n] * c["scale"] < 1.2 * c["invest_cost"]:
            break
        view.step({"kind": "discrete", "decision": "invest", "choice": n})
    _run(view, lambda v: None)


def tools(view, env):
    c = env._config
    found = []
    for _ in range(len(c["tool_names"])):
        r = view.step({"kind": "query", "target": "catalogue"})
        if r["result"]["ok"] and r["result"]["results"].get("tool"):
            found.append(r["result"]["results"]["tool"])
    stats: dict = {}
    obs = view.observe()
    jg, sg = c["gauge_names"]
    k = 0
    while not view.done:
        j = int(obs["numbers"].get(jg, 0))
        cands = [(stats.get((t, j), [0, 0]), t) for t in found]
        untried = [t for (n, _), t in cands if n < 2]
        if untried:
            tool = untried[k % len(untried)]
        else:
            rate, tool = max(((s / n, t) for (n, s), t in cands), default=(0, None))
            if rate < 0.3:
                tool = None
        k += 1
        r = view.step({"kind": "discrete", "decision": "use", "choice": tool} if tool else None)
        obs = r["observation"]
        if tool and sg in obs["numbers"]:
            a = stats.setdefault((tool, j), [0, 0])
            a[0] += 1
            a[1] += int(obs["numbers"][sg])


def coop(view, env):
    gx, gy = env._config["gauge_names"]
    G = env._config["G"]
    sx, sy = [], []
    obs = {a: view.observe(a) for a in view.agents()}
    while not view.done:
        if len(sx) >= 6 and len(sy) >= 6:
            plan = {"cell": cell_name(int(np.clip(round(np.mean(sx)), 0, G - 1)),
                                      int(np.clip(round(np.mean(sy)), 0, G - 1)))}
            r = view.step({"a0": {"kind": "coordinate", "with": ["a1"], "plan": plan},
                           "a1": {"kind": "coordinate", "with": ["a0"], "plan": plan}})
            if r["results"]["a0"].get("result", {}).get("hit"):
                sx, sy = [], []
            else:
                sx, sy = sx[-3:], sy[-3:]
        else:
            r = view.step({"a0": {"kind": "message", "to": "a1", "content": {"x": obs["a0"]["numbers"].get(gx)}},
                           "a1": {"kind": "message", "to": "a0", "content": {"y": obs["a1"]["numbers"].get(gy)}}})
            for m in r["observations"]["a1"]["entities"]:
                if m.get("kind") == "message" and (m.get("content") or {}).get("x") is not None:
                    sx.append(m["content"]["x"])
            for m in r["observations"]["a0"]["entities"]:
                if m.get("kind") == "message" and (m.get("content") or {}).get("y") is not None:
                    sy.append(m["content"]["y"])
        obs = r["observations"]


def fault(view, env):
    c = env._config
    g = c["gauge_names"][0]
    sp = view.action_space()
    desc = sp.kinds["query"]["properties"]["target"]["description"]
    tests = [t.split(":", 1)[1] for t in sp.kinds["query"]["properties"]["target"]["enum"] if t.startswith("test:")]
    costs = {t: float(desc.split(f"test:{t}: run test {t} (costs ")[1].split(")")[0]) for t in tests}
    generic = min(costs, key=costs.get)
    specific = [t for t in tests if t != generic]
    repairs = [o for o in sp.limits["decisions"]["repair"] if o != "none"]
    learned: dict = {}
    first = []
    while not view.done:
        r = view.step(None)
        v = r["observation"]["numbers"].get(g)
        if v is None:
            continue
        if len(first) < 3:
            first.append(v)
            continue
        if v > 0.8 * np.mean(first):
            continue
        pos = {t: 0 for t in specific}
        for _ in range(2):
            for t in specific:
                if view.done:
                    return
                rr = view.step({"kind": "query", "target": f"test:{t}"})
                pos[t] += int(rr["result"]["results"]["positive"]) if rr["result"]["ok"] else 0
        sig = max(pos, key=pos.get)
        order = ([learned[sig]] if sig in learned else []) + [x for x in repairs if x != learned.get(sig)]
        for rep in order:
            if view.done:
                return
            view.step({"kind": "discrete", "decision": "repair", "choice": rep})
            nxt = view.step(None)["observation"]["numbers"].get(g, 0.0)
            if nxt > 0.8 * np.mean(first):
                learned[sig] = rep
                break


def unlock(view, env):
    """Stage by stage: inspect every option for the warning sign of a trap, try only the options without one
    (restarting and replaying the known prefix each time), keep the first one that signals progress twice."""
    sp = view.action_space()
    (dname, opts), = sp.limits["decisions"].items()
    options = [o for o in opts if o != "restart"]
    known: list[str] = []
    step = lambda o: {"kind": "discrete", "decision": dname, "choice": o}  # noqa: E731
    max_len = sp.kinds["sequence"]["properties"]["steps"]["maxItems"] - 1
    paying = False
    while not paying and len(known) < max_len // 2 - 1 and not view.done:
        prefix = [step("restart")] + [step(k) for k in known]
        insp = [{"kind": "query", "target": "inspect", "args": {"option": o}} for o in options]
        warn = {o: 0 for o in options}
        for _ in range(2 if known else 1):                   # the first stage has no trap; later ones: inspect twice
            r = view.step({"kind": "sequence", "steps": prefix + insp})
            if not r["result"]["ok"]:
                break
            for x in r["result"]["results"]:
                if isinstance(x, dict) and x.get("warning"):
                    warn[x["option"]] += 1
        safe = sorted((o for o in options if warn[o] < 2), key=lambda o: warn[o])
        clean = [o for o in safe if warn[o] == 0]
        best = None
        for o in clean + safe:                               # unflagged options get a second chance first
            if view.done:
                return
            r = view.step({"kind": "sequence", "steps": (prefix + [step(o)]) * 2})
            if r["result"]["ok"]:
                sig = [x.get("signal", 0) for x in r["result"]["results"] if isinstance(x, dict) and "signal" in x]
                if len(sig) >= 2 and sig[-1] and sig[-1 - len(known) - 1]:
                    best = o
                    break
        if best is None:
            break
        known.append(best)
        if not view.done:                                    # does the sequence known so far already pay?
            view.step({"kind": "sequence", "steps": [step("restart")] + [step(k) for k in known]})
            if not view.done:
                paying = _net(view.step(None)["observation"]) > 0
    if not view.done and not paying:
        view.step({"kind": "sequence", "steps": [step("restart")] + [step(k) for k in known]})
    _run(view, lambda v: None)


def sign_switch(view, env):
    """Probe both ends of the lever every few ticks and play whichever earned more net income last time."""
    name = next(iter(view.action_space().kinds["levers"]["properties"]["values"]["properties"]))
    best, lo_net = 10.0, None
    while not view.done:
        phase = view.t % 6
        x = 0.0 if phase == 0 else (10.0 if phase == 1 else best)
        r = view.step({"kind": "levers", "values": {name: x}})
        if phase == 0:
            lo_net = _net(r["observation"])
        elif phase == 1 and lo_net is not None:
            best = 10.0 if _net(r["observation"]) > lo_net else 0.0


def maintenance(view, env):
    """A practice episode run to failure records each machine's runtime at failure; the scored episode services
    a machine once its runtime reaches a fraction of that machine's typical failure runtime (the fraction is also
    chosen in practice)."""
    import copy

    def episode(v, limits):
        ages: dict = {}
        last: dict = {}
        while not v.done:
            ms = {m["id"]: m for m in v.observe()["entities"] if m.get("kind") == "machine"}
            for mid, m in ms.items():
                if m["status"] == "repair" and last.get(mid, {}).get("status") == "run":
                    ages.setdefault(mid, []).append(last[mid]["runtime"])
            last = ms
            due = [mid for mid, m in ms.items() if m["status"] == "run" and m["runtime"] >= limits.get(mid, 1e9)]
            v.step({"kind": "discrete", "decision": "service", "choice": due[0]} if due else None)
        return ages
    seen = episode(PublicView(copy.deepcopy(env)), {})
    allv = [x for v in seen.values() for x in v] or [1e9]
    best, best_net = {}, -np.inf
    for frac in (0.3, 0.45, 0.6):
        lim = {m: frac * float(np.mean(seen.get(m) or allv)) for m in env._config["machine_names"]}
        e = copy.deepcopy(env)
        episode(PublicView(e), lim)
        if e.outcome()["net"] > best_net:
            best, best_net = lim, e.outcome()["net"]
    episode(view, best)


def bottleneck(view, env):
    names = env._config["station_names"]
    sp = view.action_space()
    alloc = {n: float(x) for n, x in zip(names, env._config["default_alloc"])}
    while not view.done:
        if view.t % 8 == 4:
            q = {}
            for n in names:
                if view.done:
                    return
                r = view.step({"kind": "query", "target": f"station:{n}"})
                if r["result"]["ok"]:
                    q[n] = r["result"]["results"]["queue"]
            hi, lo = max(q, key=q.get), min(q, key=lambda n: (q[n], -alloc[n]))
            if q[hi] > 2 and alloc[lo] > 1 and hi != lo:
                alloc[hi] += 1
                alloc[lo] -= 1
            if not view.done:
                view.step({"kind": "allocate", "amounts": alloc})
        else:
            view.step(None)
    del sp


def inventory(view, env):
    c = env._config
    name = c["order_name"]
    gs, gsold = c["gauge_names"][0], c["gauge_names"][1]
    sold, lead = [], None
    first_order_t = None
    while not view.done:
        obs = view.observe()
        stock = obs["numbers"].get(gs, 0.0)
        on_order = obs["resources"].get("on_order_units", 0.0)
        if gsold in obs["numbers"]:
            sold.append(obs["numbers"][gsold])
        if lead is None and first_order_t is not None and stock > 0 and on_order == 0:
            lead = view.t - first_order_t
        rate = float(np.mean(sold[-10:])) if sold else 5.0
        L = lead if lead is not None else 4
        target = rate * (L + 2)
        q = int(max(0, round(target - stock - on_order)))
        if q > 0 and first_order_t is None:
            first_order_t = view.t
        view.step({"kind": "levers", "values": {name: float(q)}} if q > 0 else None)


def diffusion(view, env):
    sp = view.action_space()
    names = sorted(sp.kinds["allocate"]["properties"]["amounts"]["properties"])
    total = sp.limits["allocate_total"]
    view.step({"kind": "allocate", "amounts": {names[0]: float(total)}})
    _run(view, lambda v: None)


def cascade(view, env):
    names = env._config["zone_names"]
    tot = float(env._config["reinforce_budget"])
    while not view.done:
        obs = view.observe()
        ents = obs["arrays"].get("alive_by_zone")
        if ents:
            weak = np.array([1.0 - a for a in ents]) + 0.05
            w = weak / weak.sum()
            view.step({"kind": "allocate", "amounts": {n: 0.4 * tot * float(x) for n, x in zip(names, w)}})
        else:
            view.step(None)


def repair(view, env):
    while not view.done:
        obs = view.observe()
        broken = [x["id"] for x in obs["entities"] if x.get("kind") == "link" and not x["up"]
                  and not x["id"].startswith("new")]
        view.step({"kind": "discrete", "decision": "repair", "choice": broken[0]} if broken else None)


def controller(view, env):
    """Find heater settings that put the process in specification, then hold the temperature that was in
    specification with a proportional controller."""
    obs = view.observe()
    h = next(iter(view.action_space().kinds["levers"]["properties"]["values"]["properties"]))
    Tg, Sg = env._config["gauges"][0]["name"], env._config["gauges"][1]["name"]
    good = []
    level = 0.0
    while not view.done and view.t < view.horizon // 2:
        level = (level + 1.0) % 11.0
        for _ in range(6):
            if view.done:
                break
            r = view.step({"kind": "levers", "values": {h: level}})
            n = r["observation"]["numbers"]
            if n.get(Sg, 0.0) > 0.5 and Tg in n:
                good.append((n[Tg], level))
    if not good:
        _run(view, lambda v: None)
        return
    target = float(np.median([g[0] for g in good]))
    u = float(np.median([g[1] for g in good]))
    while not view.done:
        T = view.observe()["numbers"].get(Tg, target)
        u = float(np.clip(u + 0.08 * (target - T), 0.0, 10.0))
        view.step({"kind": "levers", "values": {h: u}})
    del obs


SPECIFIC = {"sci_functional_law": science, "sci_instruments": science, "srch_field": field,
            "srch_distractors": distractors, "srch_tools": tools, "coop_split_info": coop,
            "posm_fault_diagnosis": fault, "posm_unlock": unlock, "posm_sign_switch": sign_switch,
            "des_maintenance": maintenance, "des_bottleneck": bottleneck, "des_inventory": inventory,
            "des_perishable": inventory, "net_diffusion": diffusion, "net_cascade": cascade, "net_repair": repair,
            "eng_delayed_controller": controller}


def practice_constant(env) -> dict | None:
    """Learning across episodes: each candidate action is held for a whole practice episode on a copy of the world
    (which tells delayed and irreversible consequences apart from immediate ones); then the best levers are
    combined, and combined with each discrete option. Returns the best constant action found."""
    import copy
    sp = env.action_space()
    cands = candidates(sp)

    def total(a):
        e = copy.deepcopy(env)
        v = PublicView(e)
        _run(v, lambda vv: a)
        return float(e.outcome()["net"])
    scores = [total(a) for a in cands]
    scores.append(total(None))
    cands.append(None)
    if "levers" in sp.kinds:
        per: dict = {}
        for a, s in zip(cands, scores):
            if a is not None and a["kind"] == "levers":
                (n, val), = a["values"].items()
                if n not in per or s > per[n][1]:
                    per[n] = (val, s)
        combo = {"kind": "levers", "values": {n: val for n, (val, _) in per.items()}}
        extra = [combo]
        for d, opts in sorted(sp.limits.get("decisions", {}).items()):
            for o in opts[:8]:
                extra.append({"kind": "sequence", "steps": [{"kind": "discrete", "decision": d, "choice": o}, combo]})
        for a in extra:
            cands.append(a)
            scores.append(total(a))
    return cands[int(np.argmax(scores))]


def play(env) -> dict:
    """The family's reference on `env`: its mechanism-specific public-information policy, which plays the scored
    episode once, from what it observes (closed loop). A family without one gets the generic adaptive learner
    (`refs_v2.adaptive`: extremum seeking on levers and allocations, a discounted bandit on decisions), which knows
    no mechanism. Nothing here replays copies of the scored world (`practice_constant` is a hindsight baseline kept
    for comparison, not a reference)."""
    from .oracles import _family_hook
    fn = SPECIFIC.get(env.family) or _family_hook(env, "reference")
    if fn is not None:
        fn(PublicView(env), env)
        return env.outcome()
    from .refs_v2 import adaptive
    adaptive(PublicView(env), env)
    return env.outcome()
