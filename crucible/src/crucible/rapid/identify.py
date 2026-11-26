"""Identifiability (MASTER_SPEC Part 7.4): can the family's experimental question be answered from what a participant
can do and see?

Each procedure receives a `PublicView` of a fresh world - step, observe, the action space, the public spec and
nothing else - and answers the family's question from its own actions and observations. The evaluator then
computes the true answer from the hidden structure (`truth_*`, or by controlled interventions on copies) and
compares. A family is identifiable when its question is answered correctly on most seeds.

`run(family, seed)` -> {"question", "estimate", "truth", "ok", ...}.
"""
from __future__ import annotations

import numpy as np

from .engines.search import cell_name


class PublicView:
    """What a participant can use: actions, observations, tool schemas and the public spec. Nothing else."""

    __slots__ = ("_PublicView__env",)

    def __init__(self, env):
        self.__env = env

    def step(self, action):
        return self.__env.step(action)

    def observe(self, agent=None):
        return self.__env.observe(agent)

    def action_space(self, agent=None):
        return self.__env.action_space(agent)

    def agents(self):
        return self.__env.agents()

    @property
    def spec(self):
        return self.__env.spec

    @property
    def horizon(self):
        return self.__env.horizon

    @property
    def done(self):
        return self.__env.done

    @property
    def t(self):
        return self.__env.t


def _net(obs) -> float:
    r = obs["resources"]
    return float(r.get("revenue_cents_last_tick", 0.0) - r.get("cost_cents_last_tick", 0.0))


def _levers(view):
    sp = view.action_space()
    return sp.kinds["levers"]["properties"]["values"]["properties"]


# ====================================================================== contrasts (generic)
def contrast_schedule(H: int, block: int) -> list[str]:
    return [("A" if (k // block) % 2 == 0 else "B") for k in range(H)]


def contrast_estimate(view, act, measure, block: int) -> dict:
    """Alternate actions A and B in blocks, measure each tick; the second half of each block counts (transients
    pass). Returns the mean difference A - B and its z-score."""
    H = view.horizon - view.t
    sched = contrast_schedule(H, block)
    xa, xb = [], []
    for k, arm in enumerate(sched):
        r = view.step(act[arm])
        if k % block >= block // 2:
            (xa if arm == "A" else xb).append(measure(r["observation"]))
    xa, xb = np.asarray(xa), np.asarray(xb)
    d = float(xa.mean() - xb.mean())
    se = float(np.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb))) if min(len(xa), len(xb)) > 1 else 1.0
    return {"difference": d, "z": d / max(se, 1e-12)}


def contrast_truth(env, act, measure, block: int) -> float:
    """The same block schedule and its mirror image on two copies of the world (common random numbers): the mean
    per-tick difference between A and B ticks, free of the world's noise."""
    from .oracles import fork
    H = env.horizon - env.t
    sched = contrast_schedule(H, block)
    a, b = fork(env), fork(env)
    diffs = []
    for k, arm in enumerate(sched):
        ra = a.step(act[arm])
        rb = b.step(act["B" if arm == "A" else "A"])
        if k % block >= block // 2:
            v = measure(ra["observation"]) - measure(rb["observation"])
            diffs.append(v if arm == "A" else -v)
    return float(np.mean(diffs))


def _lever_contrast(env):
    cfg = env._config
    view = PublicView(env)
    props = _levers(view)
    cl = cfg.get("claim", {})
    if "lever" in cl and cfg.get("engine") == "primitive_graph":
        eng = env._engine
        name = cfg["nodes"][eng.idx[cl["lever"]]]["lever"]["name"]
    else:
        name = sorted(props)[0] if "lever_name" not in cfg else cfg["lever_name"]
        if cfg.get("engine") == "discrete_event":
            name = cfg["order_name"]
    p = props[name]
    lo, hi = p["minimum"], p["maximum"]
    return name, {"A": {"kind": "levers", "values": {name: lo + 0.75 * (hi - lo)}},
                  "B": {"kind": "levers", "values": {name: lo + 0.25 * (hi - lo)}}}


def lever_sign(env) -> dict:
    """Question: over this horizon, does a higher setting of the family's key lever raise net income per tick?"""
    from .oracles import fork
    block = max(4, env.horizon // 8)
    name, act = _lever_contrast(env)
    truth = contrast_truth(fork(env), act, _net, block)
    est = contrast_estimate(PublicView(env), act, _net, block)
    return {"question": f"sign of the effect of {name} on net income", "truth": float(np.sign(truth)),
            "estimate": float(np.sign(est["difference"])), "z": est["z"], "true_difference": truth,
            "ok": np.sign(truth) == np.sign(est["difference"]) and abs(est["z"]) >= 2.0}


def admission_sign(env) -> dict:
    """Question (routing): does sending all demand earn more than sending a quarter of it, over this network?"""
    from .oracles import fork
    view = PublicView(env)
    props = _levers(view)
    act = {"A": {"kind": "levers", "values": {n: 0.5 for n in props}},
           "B": {"kind": "levers", "values": {n: 0.25 / len(props) for n in props}}}
    block = max(4, env.horizon // 8)
    truth = contrast_truth(fork(env), act, _net, block)
    est = contrast_estimate(view, act, _net, block)
    return {"question": "does admitting all demand pay more than admitting a quarter", "truth": float(np.sign(truth)),
            "estimate": float(np.sign(est["difference"])), "z": est["z"], "true_difference": truth,
            "ok": np.sign(truth) == np.sign(est["difference"]) and abs(est["z"]) >= 2.0}


def regime_switch(env) -> dict:
    """Question (regime switching): does the lever's effect on sales change sign during the episode? Public: alternate
    the lever between its ends every tick and estimate the effect over windows of ten ticks. Truth: the hidden
    regime path of this world visits regimes of both signs for at least ten ticks each."""
    from .oracles import fork
    e = fork(env)                                        # the evaluator's copy, taken before anything happens
    view = PublicView(env)
    props = _levers(view)
    (name, p), = props.items()
    g = list(view.step(None)["observation"]["numbers"])[0]
    xs, ys = [], []
    while not view.done:
        x = p["maximum"] if view.t % 2 == 0 else p["minimum"]
        r = view.step({"kind": "levers", "values": {name: x}})
        v = r["observation"]["numbers"].get(g)
        if v is not None:
            xs.append(x)
            ys.append(v)
    xs, ys = np.asarray(xs), np.asarray(ys)
    signs = []
    for k in range(0, len(xs) - 9, 10):
        a, b = ys[k:k + 10][xs[k:k + 10] == p["maximum"]], ys[k:k + 10][xs[k:k + 10] == p["minimum"]]
        if len(a) > 1 and len(b) > 1:
            d = a.mean() - b.mean()
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) + 1e-9
            if abs(d / se) >= 2.0:
                signs.append(np.sign(d))
    est = bool(signs) and (1.0 in signs) and (-1.0 in signs)
    path = []
    for _ in range(e.horizon - e.t):
        path.append(int(e._state["s"][0]))
        e.step(None)
    slope = env._config["slope"]
    pos = sum(1 for s in path if slope[s] > 0)
    neg = sum(1 for s in path if slope[s] < 0)
    truth = pos >= 10 and neg >= 10
    return {"question": "does the lever's effect change sign during the episode", "estimate": est, "truth": truth,
            "ok": est == truth, "window_signs": [float(s) for s in signs]}


def reinforcement_sign(env) -> dict:
    """Question (cascade): does reinforcing every zone reduce the number of nodes lost per tick?"""
    from .oracles import fork
    cfg = env._config
    tot = float(cfg["reinforce_budget"]) * 0.999
    act = {"A": {"kind": "allocate", "amounts": {n: tot / len(cfg["zone_names"]) for n in cfg["zone_names"]}},
           "B": None}
    g = cfg["gauge_names"][1]
    meas = lambda o: -float(o["numbers"].get(g, 0.0)) + _net(o) / 1e4  # noqa: E731
    block = 6
    truth = contrast_truth(fork(env), act, meas, block)
    est = contrast_estimate(PublicView(env), act, meas, block)
    return {"question": "does reinforcement pay (fewer nodes lost, net income)", "truth": float(np.sign(truth)),
            "estimate": float(np.sign(est["difference"])), "z": est["z"],
            "ok": np.sign(truth) == np.sign(est["difference"])}


def cascade_size(env) -> dict:
    """Question (cascade): can one overloaded node take more than one node down? Public: watch the losses per tick.
    Truth: overload each grid node alone on the hidden model and propagate the failures."""
    view = PublicView(env)
    g = list(view.step(None)["observation"]["numbers"])[1]
    worst = 0.0
    while not view.done:
        worst = max(worst, view.step(None)["observation"]["numbers"].get(g, 0.0))
    eng, c = env._engine, env._config
    base, cap = np.asarray(c["base"]), np.asarray(c["cap"])
    alive = np.ones(len(cap), dtype=bool)
    sizes = []
    for i in range(len(cap)):
        load = base.copy()
        load[i] = cap[i] * 1.05
        sizes.append(int((~eng.propagate(load, cap, alive, np.full(len(c["dep"]), -1))).sum()))
    truth = max(sizes) >= 2
    return {"question": "can one overload take down several nodes", "estimate": bool(worst >= 2), "truth": truth,
            "ok": bool(worst >= 2) == truth, "largest_loss_seen": worst, "largest_single_overload_loss": max(sizes)}


def repair_sign(env) -> dict:
    """Question (repair): does repairing a broken link raise the demand served the next tick?"""
    view = PublicView(env)
    cfg_g = env.spec
    del cfg_g
    obs = view.observe()
    gains_rep, gains_wait = [], []
    served_name = None
    for k in range(view.horizon):
        if view.done:
            break
        broken = [x["id"] for x in obs["entities"] if x.get("kind") == "link" and not x["up"]
                  and not x["id"].startswith("new")]
        before = obs["numbers"]
        if served_name is None and before:
            served_name = sorted(before, key=lambda n: -before[n])[0]
        if broken and k % 2 == 0:
            r = view.step({"kind": "discrete", "decision": "repair", "choice": broken[0]})
            gains_rep.append(r["observation"]["numbers"].get(served_name, 0) - before.get(served_name, 0))
        elif broken:
            r = view.step(None)
            gains_wait.append(r["observation"]["numbers"].get(served_name, 0) - before.get(served_name, 0))
        else:
            r = view.step(None)
        obs = r["observation"]
    est = (np.mean(gains_rep) if gains_rep else 0.0) - (np.mean(gains_wait) if gains_wait else 0.0)
    return {"question": "does repairing a broken link raise served demand", "truth": 1.0,
            "estimate": float(np.sign(est)), "repairs": len(gains_rep),
            "ok": bool(est >= 0 and len(gains_rep) > 0)}


# ====================================================================== the look-alike markets
def market_rule(env) -> dict:
    """Question: which of the four causal rules governs this market? Public procedure: a price sweep, a price sweep
    with the certification, then a late price contrast after holding prices high."""
    view = PublicView(env)
    props = _levers(view)
    sp = view.action_space()
    # the brief names the levers price and promotion and the gauges units, visits, rival price, in that order;
    # the price lever is the one whose range starts at 1
    price = next(n for n, p in props.items() if p["minimum"] == 1.0)
    units_g, visits_g = list(view.step(None)["observation"]["numbers"])[:2]
    dec = sp.limits["decisions"]

    def conversion(o):
        # units per visit: dividing by the observed footfall removes the demand swings both share
        n = o["numbers"]
        return n.get(units_g, np.nan) / max(n.get(visits_g, np.nan), 1e-6)
    (dname, opts), = dec.items()
    grid = [1.5, 4.5, 7.5, 10.0]

    def sweep():
        out = []
        for p in grid:
            vals = []
            for k in range(3):
                r = view.step({"kind": "levers", "values": {price: p}})
                if k:
                    vals.append(conversion(r["observation"]))
            out.append(float(np.nanmean(vals)))
        return out
    plain = sweep()
    view.step({"kind": "discrete", "decision": dname, "choice": opts[1]})
    cert = sweep()
    view.step({"kind": "discrete", "decision": dname, "choice": opts[0]})
    for _ in range(18):
        view.step({"kind": "levers", "values": {price: 9.0}})
    late = []
    while not view.done:
        a = 6.0 if (view.t // 2) % 2 == 0 else 3.0
        r = view.step({"kind": "levers", "values": {price: a}})
        late.append((a, conversion(r["observation"])))
    late = np.array(late)
    late_eff = float(np.nanmean(late[late[:, 0] == 6.0, 1]) - np.nanmean(late[late[:, 0] == 3.0, 1]))
    early_eff = float(np.interp(6.0, grid, plain) - np.interp(3.0, grid, plain))
    if np.mean(cert) > 3 * np.mean(plain) and max(plain) - min(plain) < 0.3 * (max(cert) - min(cert)):
        est = "gate"
    elif 0 < int(np.argmax(plain)) < len(grid) - 1 and plain[0] < 0.8 * max(plain):
        est = "signal"
    elif early_eff < 0 and late_eff < 2.0 * early_eff:
        est = "drift"
    else:
        est = "elastic"
    truth = env._config["rule"]
    return {"question": "which causal rule", "estimate": est, "truth": truth, "ok": est == truth,
            "units_by_price": plain, "units_by_price_certified": cert, "early_effect": early_eff,
            "late_effect": late_eff}


# ====================================================================== science
def _fit_predict(X, y, Xq):
    def feats(A):
        a, b = A[:, 0], A[:, 1]
        return np.c_[np.ones(len(A)), a, b, a * a, b * b, a * b, np.sqrt(np.maximum(a, 0)), a * np.sqrt(
            np.maximum(a, 0)) / (1 + a), (b > np.median(X[:, 1])).astype(float)]
    F = feats(X)
    beta = np.linalg.lstsq(F, y, rcond=None)[0]
    return feats(Xq) @ beta


def experiment_grid(view, x1, x2, g, budget_frac=0.9, grid=4, reps=2, instrument=None):
    sp = view.action_space()
    box = sp.kinds["experiment"]["properties"]["design"]["properties"]["set"]["properties"]
    X, y = [], []
    for a in np.linspace(box[x1]["minimum"], box[x1]["maximum"], grid):
        for b in np.linspace(box[x2]["minimum"], box[x2]["maximum"], grid):
            d = {"set": {x1: float(a), x2: float(b)}, "measure": [g], "ticks": 1, "replicates": reps}
            if instrument:
                d["instrument"] = instrument
            r = view.step({"kind": "experiment", "design": d})
            if not r["result"]["ok"]:
                return np.asarray(X), np.asarray(y)
            m = r["result"]["results"]["measured"][g]
            if m["mean"] is not None:
                X.append([a, b])
                y.append(m["mean"])
    return np.asarray(X), np.asarray(y)


def law_shape(env) -> dict:
    """Question (science): does the reading saturate in the first setting (diminishing returns), and is there a
    jump in the second (a threshold)? Answered from designed experiments."""
    view = PublicView(env)
    cfg = env._config
    x1, x2 = [cfg["nodes"][env._engine.idx[n]]["lever"]["name"] for n in cfg["claim"]["levers"]]
    g = cfg["objective"]["gauge"]
    inst = "lab" if cfg.get("experiment", {}).get("instruments") else None
    X, y = experiment_grid(view, x1, x2, g, grid=4, reps=2 if inst is None else 1, instrument=inst)
    if len(y) < 8:
        return {"question": "law shape", "ok": False, "note": "experiments ran out"}
    by1 = [y[np.isclose(X[:, 0], a)].mean() for a in np.unique(X[:, 0])]
    by2 = [y[np.isclose(X[:, 1], b)].mean() for b in np.unique(X[:, 1])]
    d1 = np.diff(by1)
    d2 = np.diff(by2)
    saturating = bool(d1[-1] < d1[0])
    jump = bool(np.max(np.abs(d2)) > 2.5 * np.median(np.abs(d2)) + 1e-9) if len(d2) > 2 else False
    truth_sat = cfg["claim"]["law"] == "saturating_product"
    truth_jump = cfg["claim"]["law"] != "saturating_product"
    est = {"saturating": saturating, "threshold": jump}
    tru = {"saturating": truth_sat, "threshold": truth_jump}
    key = "saturating" if truth_sat else "threshold"
    return {"question": f"is the law {key}", "estimate": est[key], "truth": tru[key], "ok": est[key] == tru[key],
            "experiments": int(len(y))}


# ====================================================================== discrete-event worlds
def bottleneck(env) -> dict:
    view = PublicView(env)
    sp = view.action_space()
    stations = sorted(t.split(":", 1)[1] for t in sp.kinds["query"]["properties"]["target"]["enum"]
                      if t.startswith("station:"))
    for _ in range(12):
        view.step(None)
    q = {}
    for s in stations:
        r = view.step({"kind": "query", "target": f"station:{s}"})
        q[s] = r["result"]["results"]["queue"]
    names = env._config["station_names"]
    est = max(q, key=q.get)
    truth = names[int(np.argmin(env._config["mu"]))]
    return {"question": "which station is the bottleneck", "estimate": est, "truth": truth, "ok": est == truth,
            "queues": q}


def fragile_machine(env) -> dict:
    view = PublicView(env)
    fails: dict[str, int] = {}
    prev: dict[str, str] = {}
    while not view.done:
        r = view.step(None)
        for m in r["observation"]["entities"]:
            if m.get("kind") == "machine":
                if m["status"] == "repair" and prev.get(m["id"]) != "repair":
                    fails[m["id"]] = fails.get(m["id"], 0) + 1
                prev[m["id"]] = m["status"]
    est = max(fails, key=fails.get) if fails else None
    truth = env._config["machine_names"][int(np.argmin(env._config["eta"]))]
    return {"question": "which machine wears fastest", "estimate": est, "truth": truth, "ok": est == truth,
            "failures_seen": fails}


# ====================================================================== network worlds
def containment(env) -> dict:
    view = PublicView(env)
    sp = view.action_space()
    names = sorted(sp.kinds["allocate"]["properties"]["amounts"]["properties"])
    total = sp.limits["allocate_total"]
    r = view.step(None)
    while not view.done and not any(x.get("id") == names[0] and x.get("members", 0) > 0
                                    for x in r["observation"]["entities"] if x.get("kind") == "community"):
        r = view.step(None)                               # wait until that community is online
    view.step({"kind": "allocate", "amounts": {names[0]: total}})
    counts, sizes = {}, {}
    for _ in range(20):
        if view.done:
            break
        ents = view.step(None)["observation"]["entities"]
        counts = {x["id"]: x["adopters"] for x in ents if x.get("kind") == "community"}
        sizes = {x["id"]: x["members"] for x in ents if x.get("kind") == "community"}
    elsewhere = sum(c for n, c in counts.items() if n != names[0])
    est = bool(elsewhere <= 0.1 * sum(s for n, s in sizes.items() if n != names[0]))
    order = list(env._config["cluster_names"])
    from .claims import concentrated_seeding
    from .families import make
    truth = bool(concentrated_seeding(make(env.family, env.seed, **env.difficulty))["adopters_elsewhere"]
                 <= 0.1 * sum(1 for c in env._config["cluster"] if c != order.index(names[0])))
    return {"question": "does adoption stay in the seeded community", "estimate": est, "truth": truth,
            "ok": est == truth, "adopters_by_community": counts}


# ====================================================================== state machines
def fault(env) -> dict:
    view = PublicView(env)
    sp = view.action_space()
    tests = sorted(t.split(":", 1)[1] for t in sp.kinds["query"]["properties"]["target"]["enum"]
                   if t.startswith("test:"))
    g = env._config["gauge_names"][0]
    first = []
    while not view.done:
        r = view.step(None)
        v = r["observation"]["numbers"].get(g)
        if v is not None and len(first) < 3:
            first.append(v)
        elif v is not None and v < 0.85 * np.mean(first):
            break
    if view.done:
        return {"question": "which fault", "ok": True, "note": "no fault occurred within the horizon",
                "estimate": None, "truth": None}
    pos = {t: 0 for t in tests}
    for _ in range(3):
        for t in tests:
            r = view.step({"kind": "query", "target": f"test:{t}"})
            pos[t] += int(r["result"]["results"]["positive"])
    desc = view.action_space().kinds["query"]["properties"]["target"]["description"]
    costs = {t: float(desc.split(f"test:{t}: run test {t} (costs ")[1].split(")")[0]) for t in tests}
    generic = min(costs, key=costs.get)                  # the brief: one cheap generic test, dearer specific ones
    est = max((t for t in tests if t != generic), key=lambda t: pos[t])
    truth_state = int(env._state["s"][0])
    truth = env._config["test_names"][truth_state] if truth_state > 0 else None
    return {"question": "which fault", "estimate": est, "truth": truth, "ok": est == truth, "positives": pos}


def unlock_code(env) -> dict:
    view = PublicView(env)
    sp = view.action_space()
    (dname, opts), = sp.limits["decisions"].items()
    restart = "restart"
    options = [o for o in opts if o != restart]
    hits = {o: 0 for o in options}
    trial = lambda o: [{"kind": "discrete", "decision": dname, "choice": restart},  # noqa: E731
                       {"kind": "discrete", "decision": dname, "choice": o}]
    for _ in range(2):
        for o in options:
            r = view.step({"kind": "sequence", "steps": trial(o) * 3})
            if r["result"]["ok"]:
                hits[o] += sum(int(x["signal"]) for x in r["result"]["results"] if "signal" in x)
    est = max(hits, key=hits.get)
    truth = env._config["options"][env._config["code"][0]]
    return {"question": "which operation advances the first stage", "estimate": est, "truth": truth,
            "ok": est == truth, "signals": hits}


# ====================================================================== search worlds
def _probe(view, **args):
    r = view.step({"kind": "query", "target": "probe", "args": args})
    return r["result"]["results"]["value"] if r["result"]["ok"] else None


def field_peak(env) -> dict:
    view = PublicView(env)
    G = env._config["G"]
    est = {}
    for x in range(0, G, 2):
        for y in range(0, G, 2):
            est[(x, y)] = _probe(view, cell=cell_name(x, y))
    bx, by = max(est, key=est.get)
    for x in range(max(0, bx - 1), min(G, bx + 2)):
        for y in range(max(0, by - 1), min(G, by + 2)):
            if (x, y) not in est and not view.done:
                est[(x, y)] = _probe(view, cell=cell_name(x, y))
    e = max(est, key=est.get)
    v = np.asarray(env._config["values"])
    t = tuple(int(i) for i in np.unravel_index(int(np.argmax(v)), v.shape))
    return {"question": "where is the best site", "estimate": list(e), "truth": list(t),
            "ok": max(abs(e[0] - t[0]), abs(e[1] - t[1])) <= 1 or v[e] >= 0.8 * v[t], "probes": len(est)}


def informative_features(env) -> dict:
    view = PublicView(env)
    feats = {x["id"]: x["features"] for x in view.step(None)["observation"]["entities"] if x.get("kind") == "item"}
    if not feats:
        feats = {x["id"]: x["features"] for x in view.observe()["entities"] if x.get("kind") == "item"}
    names = sorted(feats)
    y = []
    for n in names:
        y.append(_probe(view, item=n))
        if view.done:
            break
    names = names[:len(y)]
    F = np.asarray([feats[n] for n in names])
    Fz = (F - F.mean(axis=0)) / np.maximum(F.std(axis=0), 1e-9)
    beta = np.linalg.lstsq(np.c_[np.ones(len(y)), Fz], np.asarray(y), rcond=None)[0][1:]
    est = sorted(int(i) for i in np.argsort(-np.abs(beta))[:2])
    truth = sorted(env._config["informative"])
    return {"question": "which features carry value", "estimate": est, "truth": truth, "ok": est == truth}


def tool_map(env) -> dict:
    view = PublicView(env)
    tools = []
    for _ in range(len(env._config["tool_names"])):
        r = view.step({"kind": "query", "target": "catalogue"})
        if r["result"]["ok"] and r["result"]["results"].get("tool"):
            tools.append(r["result"]["results"]["tool"])
    tries: dict = {}
    obs = view.observe()
    jobs = env._config["job_names"]
    succ_g = env._config["gauge_names"][1]
    job_g = env._config["gauge_names"][0]
    k = 0
    while not view.done:
        j = int(obs["numbers"].get(job_g, 0))
        tool = tools[k % len(tools)] if tools else None
        k += 1
        if tool is None:
            break
        r = view.step({"kind": "discrete", "decision": "use", "choice": tool})
        obs = r["observation"]
        s = obs["numbers"].get(succ_g)
        if s is not None:
            a = tries.setdefault((tool, j), [0, 0])
            a[0] += 1
            a[1] += int(s)
    app = np.asarray(env._config["app"])
    names = env._config["tool_names"]
    agree = [((a[1] / a[0]) > 0.5) == bool(app[names.index(t), j]) for (t, j), a in tries.items() if a[0] >= 2]
    del jobs
    return {"question": "which tool works on which job type", "agreement": float(np.mean(agree)) if agree else 0.0,
            "pairs_tested": len(agree), "ok": bool(agree) and float(np.mean(agree)) >= 0.8,
            "estimate": None, "truth": None}


def coop_target(env) -> dict:
    view = PublicView(env)
    gx, gy = env._config["gauge_names"]
    sx, sy = [], []
    obs = {a: view.observe(a) for a in view.agents()}
    for _ in range(16):
        r = view.step({"a0": {"kind": "message", "to": "a1", "content": {"x": obs["a0"]["numbers"].get(gx)}},
                       "a1": {"kind": "message", "to": "a0", "content": {"y": obs["a1"]["numbers"].get(gy)}}})
        obs = r["observations"]
        for m in obs["a1"]["entities"]:
            if m.get("kind") == "message" and (m.get("content") or {}).get("x") is not None:
                sx.append(m["content"]["x"])
        for m in obs["a0"]["entities"]:
            if m.get("kind") == "message" and (m.get("content") or {}).get("y") is not None:
                sy.append(m["content"]["y"])
    G = env._config["G"]
    est = [int(np.clip(round(np.mean(sx)), 0, G - 1)), int(np.clip(round(np.mean(sy)), 0, G - 1))]
    truth = [int(v) for v in env._state["target"]]
    return {"question": "where is the target (each partner sees one coordinate)", "estimate": est, "truth": truth,
            "ok": est == truth}


def run(family: str, seed: int, **difficulty) -> dict:
    import importlib
    from .families import FAMILIES, make
    ref = FAMILIES[family].identify
    mod, fn = ref.split(":")
    try:
        return getattr(importlib.import_module(mod), fn)(make(family, seed, **difficulty))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
