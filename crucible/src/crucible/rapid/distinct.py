"""The machine-readable distinctness registry of Part 7.5, with claimed structural differences verified by controlled
interventions (evaluator side).

For every family the registry records its fingerprint, its nearest registered family, the distance, which
fingerprint components differ and how (`fingerprints.nearest`), the family's own statement of why it is distinct,
and a verification of the difference to the nearest family by an intervention on the public interface:

* interface: when the two families offer different action kinds, an action of a kind one offers is carried out
  in its world and refused in the other's (the difference is a fact of the interface, checked by doing it);
* behaviour: otherwise the same intervention protocol is run on both - each composite canonical setting of the
  shared interface held for a block of ticks, then released - and the qualitative response signature is compared
  (for each reading and for observed net income: direction of the response, whether it is immediate or appears
  later, whether it persists after release, and whether it reverses between an early and a late application).
  The difference counts as verified when the signatures of the two families are further apart than the
  signatures of one family on different seeds (the within-family variability), on both sides.

The four look-alike markets, which share their interface exactly, are in addition separated by their own price
response protocol (`claims.lookalike_matrix`).
"""
from __future__ import annotations

import itertools

import numpy as np

from . import dominance as D
from .families import FAMILIES, make
from .oracles import fork
from .fingerprints import family_fingerprint, nearest

SEEDS = (0, 1, 2)


def _rel(x: float, scale: float) -> str:
    if abs(x) <= 0.1 * scale + 1e-9:
        return "0"
    return "+" if x > 0 else "-"


def signature(family: str, seed: int) -> set[str]:
    """Qualitative responses to interventions on a family's interface, on one world, in a form that does not depend
    on public names (which a seed draws): the uniform lever settings (every lever at the same fraction of its
    range, in increasing order) and every other canonical setting, each held for a quarter of the horizon early
    and again late, compared tick by tick with waiting on the same noise. For observed net income the class of
    the response to each uniform setting and the shape of net income over the uniform settings (increasing,
    decreasing, peaked, valley, flat) are recorded; for readings, the set of response classes over all readings."""
    env = make(family, seed)
    joint = D.Joint(env)
    out: set[str] = _query_responses(env) | _surface(env)
    if joint.empty():
        return out | {"no_world_actions"}
    H = env.horizon
    blk = max(3, H // 4)
    _, base = D.play(env, lambda t, o: {})
    keys = sorted({k for o in base for k in o["numbers"]})

    def series(trace, k):
        return np.array([o["numbers"].get(k, np.nan) for o in trace], dtype=float)

    def net(trace):
        return np.array([D._net(o) for o in trace], dtype=float)
    b_series = {k: series(base, k) for k in keys}
    b_net = net(base)
    canon = joint.canonical()
    has_levers = any(c.levers for c in joint.coords.values())
    uniform = canon[:5] if has_levers else []
    others = canon[5:] if has_levers else canon

    def cls(x, b, start):
        d = x - b
        hold = slice(start, start + blk)
        if not np.isfinite(d[hold]).any():
            return None, 0.0
        scale = float(np.nanstd(b)) + float(np.nanmean(np.abs(b))) * 0.05 + 1e-9
        dh = float(np.nanmean(d[hold]))
        d1 = d[start] if np.isfinite(d[start]) else 0.0
        after = d[start + blk:min(H, start + 2 * blk)]
        da = float(np.nanmean(after)) if after.size and np.isfinite(after).any() else 0.0
        sgn = _rel(dh, scale)
        if sgn == "0":
            return "0", dh
        timing = "immediate" if abs(d1) >= 0.5 * abs(dh) else "delayed"
        persist = "persists" if abs(da) >= 0.5 * abs(dh) else "fades"
        return f"{sgn}:{timing}:{persist}", dh
    for start, tag in ((0, "early"), (min(2 * blk, H - blk), "late")):
        levels = []
        for i, st in enumerate(uniform + others):
            def pol(t, obs, st=st, start=start):
                return D._emit(joint, st, t) if start <= t < start + blk else {}
            _, tr = D.play(env, pol)
            c, dh = cls(net(tr), b_net, start)
            if i < len(uniform):
                levels.append(dh)
                if c is not None:
                    out.add(f"{tag}:net@u{i}:{c}")
            elif c is not None:
                out.add(f"{tag}:net:{c}")
            for k in keys:
                ck, _ = cls(series(tr, k), b_series[k], start)
                if ck is not None:
                    out.add(f"{tag}:reading:{ck}")
        if len(levels) >= 3:
            out.add(f"{tag}:net_shape:{_shape(levels, b_net)}")
        # each lever swept on its own (the others at mid range): the set of response shapes over the levers
        for a, c in joint.coords.items():
            for j in range(len(c.levers)):
                lv = []
                for f in (0.0, 0.25, 0.5, 0.75, 1.0):
                    st = {x: dict(canon[2][x]) for x in joint.agents}
                    u = list(st[a]["u"])
                    u[j] = f
                    st[a]["u"] = u

                    def pol(t, obs, st=st, start=start):
                        return D._emit(joint, st, t) if start <= t < start + blk else {}
                    _, tr = D.play(env, pol)
                    lv.append(cls(net(tr), b_net, start)[1])
                out.add(f"{tag}:lever_shape:{_shape(lv, b_net)}")
    early = {x.split(":", 1)[1] for x in out if x.startswith("early:")}
    late = {x.split(":", 1)[1] for x in out if x.startswith("late:")}
    if early != late:
        out.add("changes_over_time")
    return out


def _structure(x, depth=0) -> str:
    if isinstance(x, dict):
        return "{" + ",".join(f"{k}:{_structure(v, depth + 1)}" for k, v in sorted(x.items())
                              if not k.startswith(("c", "n")) or depth == 0)[:200] + "}" if depth < 2 else "{}"
    if isinstance(x, list):
        return "[" + (_structure(x[0], depth + 1) if x else "") + "]"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, (int, float)):
        return "num"
    return "str" if isinstance(x, str) else "null"


def _query_responses(env) -> set[str]:
    """The structure of what each kind of information query returns when it is asked (an intervention on the
    information channel): which fields come back and of which type, never their values or instance names."""
    from .fingerprints import _query_kind
    out: set[str] = set()
    sp = env.action_space()
    if "query" not in sp.kinds:
        return out
    seen = set()
    for t in sp.kinds["query"]["properties"]["target"]["enum"]:
        k = _query_kind(t)
        if k in seen or t in ("tools", "history", "spec"):
            continue
        seen.add(k)
        import copy
        e = copy.deepcopy(env)
        r = e.step({"kind": "query", "target": t, "args": {}})
        res = r["result"].get("results") if r["result"]["ok"] else {"refused": True}
        out.add(f"query:{k}:{_structure(res)}")
    return out


def _surface(env) -> set[str]:
    """The shape of the first reading's response surface over two levers, held for a while on copies of the same
    world: each lever's curvature class (linear, concave, convex, flat) and whether the levers interact, as one
    name-free feature (the multiset of curvature classes plus the interaction flag)."""
    sp = env.action_space()
    if "levers" not in sp.kinds:
        return set()
    props = sp.kinds["levers"]["properties"]["values"]["properties"]
    names = sorted(props)[:2]
    if len(names) < 2:
        return set()
    hold = max(3, min(10, env.horizon // 4))

    def at(fa, fb):
        vals = {n: float(props[n]["minimum"] + f * (props[n]["maximum"] - props[n]["minimum"]))
                for n, f in zip(names, (fa, fb))}
        e = fork(env)
        xs = []
        for t in range(hold):
            r = e.step({"kind": "levers", "values": vals})
            if t >= hold - 3 and r["observation"]["numbers"]:
                xs.append(list(r["observation"]["numbers"].values())[0])
        return float(np.mean(xs)) if xs else float("nan")
    grid = {(a, b): at(a, b) for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0)}
    if not all(np.isfinite(v) for v in grid.values()):
        return set()
    span = max(grid.values()) - min(grid.values()) + 1e-9

    def curv(vals):
        lo, mid, hi = vals
        if abs(hi - lo) < 0.05 * span and abs(mid - lo) < 0.05 * span:
            return "flat"
        d2 = hi - 2 * mid + lo
        if abs(d2) < 0.1 * abs(hi - lo) + 0.02 * span:
            return "linear"
        return "concave" if d2 < 0 else "convex"
    ca = curv([grid[(a, 0.5)] for a in (0.0, 0.5, 1.0)])
    cb = curv([grid[(0.5, b)] for b in (0.0, 0.5, 1.0)])
    cross = (grid[(1.0, 1.0)] - grid[(1.0, 0.0)]) - (grid[(0.0, 1.0)] - grid[(0.0, 0.0)])
    inter = "interacting" if abs(cross) > 0.1 * span else "separable"
    return {f"surface:{','.join(sorted([ca, cb]))}:{inter}"}


def _shape(levels, b_net) -> str:
    v = np.asarray(levels, dtype=float)
    span = float(v.max() - v.min()) + 1e-9
    dv = np.diff(v)
    if span <= 0.05 * (abs(float(np.mean(b_net))) + 1e-9):
        return "flat"
    if (dv >= -0.05 * span).all():
        return "increasing"
    if (dv <= 0.05 * span).all():
        return "decreasing"
    return "peaked" if int(np.argmax(v)) not in (0, len(v) - 1) else "valley"


def _jac(a: set, b: set) -> float:
    return 1.0 - len(a & b) / len(a | b) if a | b else 0.0


def _interface_check(fa: str, fb: str) -> dict | None:
    """An action kind (or a kind of query target) offered by one family and not the other: the action is valid in
    one world and refused in the other."""
    from .fingerprints import _query_kind
    from .policies import random_action
    ea, eb = make(fa, 0), make(fb, 0)

    def offered(e):
        out = {}
        for ag in e.agents():
            sp = e.action_space(ag)
            for k, sch in sp.kinds.items():
                if k == "query":
                    for t in sch["properties"]["target"]["enum"]:
                        out.setdefault(_query_kind(t), {"kind": "query", "target": t, "args": {}})
                elif k == "experiment" and "instrument" in sch["properties"]["design"]["properties"]:
                    out.setdefault(k, None)
                    rng = np.random.default_rng(0)
                    act = random_action(sp, rng, ["experiment"])
                    act["design"]["instrument"] = sch["properties"]["design"]["properties"]["instrument"]["enum"][0]
                    out.setdefault("experiment:instrument_choice", act)
                else:
                    out.setdefault(k, None)
        return out
    oa, ob = offered(ea), offered(eb)
    only = sorted((set(oa) - set(ob)) | (set(ob) - set(oa)))
    if not only:
        return None
    rng = np.random.default_rng(0)
    for k in only:
        has, lacks, fam = (ea, eb, fa) if k in oa else (eb, ea, fb)
        act = (oa if k in oa else ob)[k]
        if act is None:
            act = random_action(has.action_space(), rng, [k]) if k != "program" else \
                {"kind": "program", "source": "def act(observation, caller):\n    return None\n", "ticks": 1}
        if act is None:
            continue
        valid = not has.action_space().validate(act)
        r = lacks.step(act)
        refused = not r["result"]["ok"]
        if valid and refused:
            return {"method": "interface", "kind": k, "offered_by": fam, "ok": True, "refused_where_absent": True,
                    "valid_where_offered": True, "error_where_absent": str(r["result"].get("error"))[:120]}
    return {"method": "interface", "kind": only[0], "ok": False}


LOOKALIKE = {"mk_elastic", "mk_signal", "mk_gate", "mk_drift"}
# pairwise difference protocols: families whose difference a dedicated intervention shows (evaluator side,
# "module:function" taking a seed and returning {"ok", ...})
PAIR_PROTOCOLS: dict[frozenset, str] = {
    frozenset(("sci_dose_response", "sci_competing_hypotheses")): "crucible.rapid.distinct:law_turns_over",
    frozenset(("des_inventory", "des_perishable")): "crucible.rapid.distinct:overstock_expires",
    frozenset(("net_routing", "net_topology_shift")): "crucible.rapid.distinct:latency_profile_changes",
    frozenset(("plan_serial_stages", "plan_spoilage")): "crucible.rapid.distinct:pulse_reaches_output_later",
    frozenset(("dl_delayed_adoption", "plan_spoilage")): "crucible.rapid.distinct:small_input_is_gated",
}


def _outcome_under(family: str, seed: int, level: float, ticks: int = 25) -> float:
    """Largest scored reading over `ticks` ticks with the lever held at `level` (evaluator side: the noise-free
    node value, so measurement noise cannot hide a zero)."""
    env = make(family, seed)
    sp = env.action_space()
    name = sorted(sp.kinds["levers"]["properties"]["values"]["properties"])[0]
    out = env._config["claim"]["outcome"]
    idx = env._engine.idx[out]
    best = 0.0
    for _ in range(ticks):
        env.step({"kind": "levers", "values": {name: float(level)}})
        best = max(best, float(env._state["h"][0, -1, idx]))
    return best


def small_input_is_gated(seed: int) -> dict:
    """dl_delayed_adoption vs plan_spoilage (both a lever feeding stocks behind saturating flows): the same
    intervention - a small constant input, below what lifts the adoption world's awareness over its hidden gate -
    leaves the adopters at exactly zero but sells in the perishing-stock world from the first ticks."""
    env = make("dl_delayed_adoption", seed)
    c = env._config
    an = next(n for n in c["nodes"] if n["id"] == c["claim"]["awareness"])
    theta = float(next(n for n in c["nodes"] if n.get("prim", "").startswith("threshold"))["params"]["theta"])
    level = 0.5 * theta * float(an["params"]["rate"]) / float(an["ports"]["x"][0][2])
    gated, sells = _outcome_under("dl_delayed_adoption", seed, level), _outcome_under("plan_spoilage", seed, level)
    return {"ok": bool(gated == 0.0 and sells > 0.0), "input": level, "adopters": gated, "sales": sells}


def _first_output_response(family: str, seed: int) -> int:
    """The first tick at which a one-tick pulse of the lever (its top, then zero) moves the reading the objective
    is scored on, against no pulse on the same noise (-1: never within twelve ticks)."""
    from .oracles import fork
    env = make(family, seed)
    sp = env.action_space()
    props = sp.kinds["levers"]["properties"]["values"]["properties"]
    name = sorted(props)[0]
    g = env._config["objective"]["gauge"]
    a, b = fork(env), fork(env)
    for t in range(12):
        xa = a.step({"kind": "levers", "values": {name: float(props[name]["maximum"]) if t == 0 else 0.0}})
        xb = b.step({"kind": "levers", "values": {name: 0.0}})
        if abs(xa["observation"]["numbers"].get(g, 0.0) - xb["observation"]["numbers"].get(g, 0.0)) > 1e-9:
            return t
    return -1


def pulse_reaches_output_later(seed: int) -> dict:
    """plan_serial_stages vs plan_spoilage (both a lever feeding stocks with saturating outflows, scored on the
    final outflow): the same intervention - a one-tick pulse of the lever - reaches the scored reading of the
    serial chain only after its processing steps and transit times, but the perishing stock's sales at once."""
    chain, single = _first_output_response("plan_serial_stages", seed), _first_output_response("plan_spoilage", seed)
    return {"ok": bool(single >= 0 and chain >= single + 2), "first_response_chain": chain,
            "first_response_single_stock": single}


def _expired_after_overstock(family: str, seed: int) -> int:
    """Units that expired after the order lever was held at its top for three ticks and the world then left alone
    to the horizon (evaluator side: the engine's expiry count)."""
    env = make(family, seed)
    sp = env.action_space()
    props = sp.kinds["levers"]["properties"]["values"]["properties"]
    name = sorted(props)[0]
    for _ in range(3):
        env.step({"kind": "levers", "values": {name: float(props[name]["maximum"])}})
    while not env.done:
        env.step({"kind": "levers", "values": {name: 0.0}})
    return int(env._state.get("expired", 0))


def overstock_expires(seed: int) -> dict:
    """des_inventory vs des_perishable (fingerprints differ in a decay motif): the same intervention - order the
    most for three ticks, then nothing - leaves expired units in the perishable world and none in the other."""
    a, b = _expired_after_overstock("des_perishable", seed), _expired_after_overstock("des_inventory", seed)
    return {"ok": a > 0 and b == 0, "expired_perishable": a, "expired_inventory": b}


def _latency_shift(family: str, seed: int) -> float:
    """Largest change, between the first and the last third of the episode, of a route's mean latency relative to
    the mean over routes, with the traffic split held even (public readings; the same intervention in any routing
    world)."""
    env = make(family, seed)
    sp = env.action_space()
    names = sorted(sp.kinds["levers"]["properties"]["values"]["properties"])
    rows = []
    while not env.done:
        r = env.step({"kind": "levers", "values": {n: 1.0 / len(names) for n in names}})
        rows.append(list(r["observation"]["numbers"].values())[1:])
    R = np.asarray(rows, dtype=float)
    k = len(R) // 3
    e, l_ = R[3:k].mean(axis=0), R[-k:].mean(axis=0)
    return float(np.max(np.abs(l_ / max(l_.mean(), 1e-9) - e / max(e.mean(), 1e-9))))


def latency_profile_changes(seed: int) -> dict:
    """net_routing vs net_topology_shift (fingerprints differ in a regime change of the network): under an even
    split held for the whole episode, the routes' relative latencies change between the first and last third in
    the rewired network, by far more than in the static one."""
    a, b = _latency_shift("net_topology_shift", seed), _latency_shift("net_routing", seed)
    return {"ok": a > 3.0 * b and a > 0.1, "relative_latency_change_shift": a, "relative_latency_change_static": b}


def _law_shape(family: str, seed: int) -> dict:
    """The noise-free law of a prediction world along its first setting (second setting at its minimum), from the
    settable range out to twice its top - a controlled intervention beyond what any participant can set (evaluator
    side): whether the response turns over (falls by more than 2% of its range after its maximum)."""
    from .engines.primitive_graph import true_response
    env = make(family, seed)
    eng = env._engine
    lv = eng.public_levers()
    a, b = lv[0], lv[1]
    xs = np.linspace(a["min"], 2.0 * a["max"], 25)
    y = np.asarray(true_response(eng, [{a["name"]: float(x), b["name"]: float(b["min"])} for x in xs],
                                 env._config["objective"]["gauge"]), dtype=float)
    k = int(np.argmax(y))
    drop = float(y[k] - y[k:].min())
    return {"turns_over": bool(drop > 0.02 * (np.ptp(y) + 1e-9)), "argmax": float(xs[k]), "drop": drop}


def law_turns_over(seed: int) -> dict:
    """sci_dose_response vs sci_competing_hypotheses: both are prediction worlds with two settings whose fingerprints
    differ in their motifs (a rise-and-fall response vs a saturating or power law). Verified when, driven past the
    settable range, the dose world's law turns over and the hypotheses world's law (either hypothesis) does not."""
    d = _law_shape("sci_dose_response", seed)
    h = _law_shape("sci_competing_hypotheses", seed)
    return {"ok": d["turns_over"] and not h["turns_over"], "dose": d, "hypotheses": h}


def _observation_check(fa: str, fb: str) -> dict | None:
    """What the two worlds show a participant who waits three ticks: kinds of entities, arrays and resources. A
    kind one world always shows and the other never does verifies a difference in observation structure."""
    def kinds(f, s):
        e = make(f, s)
        out = set()
        obs = [e.observe()] + [e.step(None)["observation"] for _ in range(3)]
        for o in obs:
            out |= {f"entity:{x.get('kind') or 'entity'}" for x in o["entities"]}
            out |= {f"resource:{k.split(':')[0]}" for k in o["resources"]}
            out |= {"arrays"} if o["arrays"] else set()
        return out
    ka = [kinds(fa, s) for s in SEEDS]
    kb = [kinds(fb, s) for s in SEEDS]
    only_a = set.intersection(*ka) - set.union(*kb)
    only_b = set.intersection(*kb) - set.union(*ka)
    if not (only_a or only_b):
        return None
    return {"method": "observation_structure", "ok": True, "only_in_this": sorted(only_a),
            "only_in_nearest": sorted(only_b)}


def verify(fa: str, fb: str, seeds=SEEDS) -> dict:
    """Verify by intervention that families `fa` and `fb` differ."""
    itf = _interface_check(fa, fb)
    if itf is not None and itf["ok"]:
        return itf
    if {fa, fb} <= LOOKALIKE:
        from .claims import lookalike_matrix
        mats = [lookalike_matrix(s) for s in seeds]
        ok = all(m["signatures"][fa] != m["signatures"][fb] and m["all_own"] for m in mats)
        return {"method": "lookalike_protocol", "ok": bool(ok),
                "signatures": [[m["signatures"][fa], m["signatures"][fb]] for m in mats]}
    obs = _observation_check(fa, fb)
    if obs is not None and obs["ok"]:
        return obs
    proto = PAIR_PROTOCOLS.get(frozenset((fa, fb)))
    if proto:
        import importlib
        mod, fn = proto.split(":")
        rs = [getattr(importlib.import_module(mod), fn)(s) for s in seeds]
        return {"method": f"protocol:{fn}", "ok": all(r.get("ok") for r in rs), "runs": rs}
    sa = [signature(fa, s) for s in seeds]
    sb = [signature(fb, s) for s in seeds]
    within_a = float(np.mean([_jac(x, y) for x, y in itertools.combinations(sa, 2)]))
    within_b = float(np.mean([_jac(x, y) for x, y in itertools.combinations(sb, 2)]))
    between = float(np.mean([_jac(x, y) for x in sa for y in sb]))
    # verified: some response is present in every world of one family and in no world of the other
    only_a = sorted(set.intersection(*sa) - set.union(*sb))
    only_b = sorted(set.intersection(*sb) - set.union(*sa))
    return {"method": "behaviour", "ok": bool(only_a or only_b), "between": between, "within": [within_a, within_b],
            "only_in_this": only_a[:8], "only_in_nearest": only_b[:8]}


def registry(families: list[str] | None = None, fps: dict | None = None, verify_nearest: bool = True) -> dict:
    """The distinctness registry: family -> fingerprint, nearest family, distance, differing components, the
    family's own reason, and the intervention verification of the difference to the nearest family."""
    names = families or sorted(FAMILIES)
    fps = fps or {n: family_fingerprint(n) for n in sorted(FAMILIES)}
    out = {}
    for n in names:
        nb = nearest(n, fps)
        rec = {"fingerprint": fps[n], **nb, "distinct_because": FAMILIES[n].distinct_because,
               "doc": FAMILIES[n].doc, "category": FAMILIES[n].category, "item": FAMILIES[n].item}
        if verify_nearest and nb["nearest"]:
            try:
                rec["verification"] = verify(n, nb["nearest"])
            except Exception as e:  # noqa: BLE001 - an unverifiable difference is reported, not hidden
                rec["verification"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out[n] = rec
    return out
