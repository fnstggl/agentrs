"""Information-flow probes: can anything a participant reaches recover what the evaluator knows? (Clarification 2,
OO.2, R.2, R.n1; decision D-19.)

The 4.0/5.0 battery in `leakage.py` checks key names and that observations do not change with the reward mode. A
world tool that hands out evaluator data hands it out identically in every mode, so that battery cannot see it (the
recovery audit proved this with `goals.check_now` and `info["score"]`). These probes instead take the attacker's
position: they collect everything a participant can reach - observations, step infos, rewards, tool responses
(including a sweep over every read-only tool), events and their metadata, free text - and try to recover the
evaluator's values from it, in worlds that are perturbed so that the evaluator's values differ from every legitimate
observable. Probe kinds:

    reward_scalar      every reward the learner receives (and every reward slot of a numeric observation) is 0.0,
                       for environments constructed with their DEFAULT reward mode
    key_name           no evaluator key (goal progress, verifier status, verified quality, relatedness verdicts, ...)
                       anywhere in what is reached; a public objective text is allowed
    canary             strings planted in evaluator-side data, hidden mechanism configuration and other
                       participants' private data never appear
    exact_value        no distinctive evaluator value (a value no legitimate observable equals) appears among the
                       reached numbers, in any unit (x1, x100, /100)
    free_text          ... nor written as a number inside reached text
    regression         across worlds that differ only in evaluator-side data, no reached numeric feature predicts the
                       evaluator value (|r| >= 0.98 with a leave-one-out fit)
    counterfactual     worlds that differ only in evaluator-side data give byte-identical participant views
    predicate_oracle   an adversarial participant cannot get any participant-authored predicate (self-goal, instant
                       goal check, experiment measure, charter condition, contract deliverable) evaluated over an
                       evaluator-only metric - and when it can, the probe binary-searches the metric to show the leak
    termination        episodes do not end because the verifier saw success (a one-bit outcome leak)
    tool_sweep         every evaluator-only tool of the registry is absent from the participant's tool list and
                       refused when called; every read-only tool's response joins the scanned material
    nested_call        evaluator tools and evaluator fields do not come back through calls made on the participant's
                       behalf later or indirectly (flows, deferred workflow jobs, triggers; `probes_tiers`)
    identity           no family, seed, split, domain configuration or civilisation profile of the generated world
                       reaches the participant (`probes_tiers`, OO.16)

Surfaces: the Tier 1 mechanism adapter, the Tier 1 rapid adapter (single- and multi-agent), kernel worlds with v5
modules loaded (goal, experiment, institution and commitment predicates), the innovation registry, the self-play
environments, the numeric environments (`MechVecEnv`, `KernelMechEnv`, `scale.rollout.WorldEnv`), the Gymnasium and
PettingZoo environments (4.0 API wrappers and the `ExperienceEnv` wrappers), and the AGENTRS adapters (sl, CAM 2.0).

Every probe returns ``{"surface", "kinds": {kind: {"checked": n, "leaks": m}}, "leaks": [...]}``; `run_all()`
aggregates them into the OO.2 / R.2 / R.n1 metrics. A probe that cannot run raises; it never reports "no leak".
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import uuid
from dataclasses import replace
from typing import Any, Callable, Iterable

import numpy as np

from .modes import LEGACY, OBSERVATION_ONLY

KINDS = ("reward_scalar", "key_name", "canary", "exact_value", "free_text", "regression", "counterfactual",
         "predicate_oracle", "termination", "tool_sweep", "nested_call", "identity")

# keys that only evaluator data carry (the public objective text is allowed; see `_key_hits`)
EVALUATOR_KEYS = frozenset({
    "progress", "streak", "success_t", "initially_true", "shaped_reward", "reference_net", "default_net",
    "random_net", "solvable", "random_score", "score_first_half", "score_second_half", "objective_quantity",
    "verified_quality", "verified_samples_ok", "currently_true", "independent_trials", "independent_trial_ok_rate",
    "independent_adopters", "excluded_related_requests", "excluded_related_cents", "mean_score_hidden",
    "related_to_author", "independent_consistent"})
DAY = 86400


# ====================================================================== bookkeeping
class Probe:
    def __init__(self, surface: str):
        self.surface = surface
        self.kinds: dict[str, dict] = {}
        self.leaks: list[dict] = []
        self.notes: list[str] = []

    def check(self, kind: str, n: int = 1) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown probe kind {kind}")
        k = self.kinds.setdefault(kind, {"checked": 0, "leaks": 0})
        k["checked"] += int(n)

    def leak(self, kind: str, what: str, **detail) -> None:
        self.check(kind, 0)
        self.kinds[kind]["leaks"] += 1
        if len(self.leaks) < 200:
            self.leaks.append({"kind": kind, "what": what, **{k: _short(v) for k, v in detail.items()}})

    def result(self) -> dict:
        return {"surface": self.surface, "kinds": self.kinds, "leaks": self.leaks,
                "n_leaks": sum(k["leaks"] for k in self.kinds.values()), "notes": self.notes}


def _short(v: Any) -> Any:
    s = v if isinstance(v, (int, float, bool)) or v is None else str(v)
    return s[:300] if isinstance(s, str) else s


def canary(tag: str) -> str:
    return f"CANARY-{tag}-{uuid.uuid4().hex[:12]}"


def dump(x: Any) -> str:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if hasattr(o, "__dict__"):
            return {k: v for k, v in vars(o).items() if not k.startswith("_")}
        return repr(o)
    return json.dumps(x, sort_keys=True, default=default)


def _plain(x: Any) -> Any:
    """`x` as JSON-native data (numpy arrays to lists, objects to their public attributes)."""
    return json.loads(dump(x))


# ====================================================================== scanning
def numbers(x: Any, path: str = "", out: list | None = None, depth: int = 0) -> list[tuple[str, float]]:
    out = [] if out is None else out
    if depth > 40:
        return out
    if isinstance(x, bool) or x is None:
        return out
    if isinstance(x, (int, float)):
        if math.isfinite(float(x)):
            out.append((path, float(x)))
        return out
    if isinstance(x, dict):
        for k, v in x.items():
            if k != "inputSchema":              # static schema limits (maxLength, minimum, ...) are not data
                numbers(v, f"{path}.{k}", out, depth + 1)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            numbers(v, f"{path}[{i}]", out, depth + 1)
    return out


def strings(x: Any, path: str = "", out: list | None = None, depth: int = 0) -> list[tuple[str, str]]:
    out = [] if out is None else out
    if depth > 40:
        return out
    if isinstance(x, str):
        out.append((path, x))
    elif isinstance(x, dict):
        for k, v in x.items():
            strings(v, f"{path}.{k}", out, depth + 1)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            strings(v, f"{path}[{i}]", out, depth + 1)
    return out


_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def text_numbers(x: Any) -> list[tuple[str, float]]:
    out = []
    for p, s in strings(x):
        for m in _NUM.findall(s):
            try:
                out.append((p, float(m.replace(",", "").replace("$", ""))))
            except ValueError:
                pass
    return out


def _close(a: float, b: float, rel: float = 1e-9) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def _variants(v: float) -> list[float]:
    """The value in its own unit and in the other money unit (cents <-> dollars, fraction <-> percent); a rescaled
    variant that is a small integer would match schema limits and counts by coincidence and is left out."""
    return [v] + [x for x in (v * 100.0, v / 100.0) if not (float(x).is_integer() and abs(x) < 1000)]


def distinctive(v: float, legit: Iterable[float]) -> bool:
    """A value worth looking for: not 0, not +-1, not a small integer, and equal to no legitimate observable."""
    if not math.isfinite(v) or abs(v) < 1e-9 or _close(abs(v), 1.0) or (float(v).is_integer() and abs(v) < 10):
        return False
    return not any(_close(v, x) or _close(v, x * 100) or _close(v, x / 100) for x in legit)


def exact_hits(values: dict[str, float], reach: Any, kind_numbers: list[tuple[str, float]] | None = None) -> list:
    nums = numbers(reach) if kind_numbers is None else kind_numbers
    hits = []
    for name, v in values.items():
        for var in _variants(v):
            for p, x in nums:
                if _close(var, x):
                    hits.append((name, v, p, x))
                    break
    return hits


def _key_hits(x: Any, keys=EVALUATOR_KEYS, path: str = "", out: list | None = None, depth: int = 0) -> list[str]:
    out = [] if out is None else out
    if depth > 40:
        return out
    if isinstance(x, dict):
        for k, v in x.items():
            last = str(k).rsplit(".", 1)[-1]
            if last in keys:
                out.append(f"{path}.{k}")
            elif last == "independent" and isinstance(v, (bool, dict)):
                out.append(f"{path}.{k}")       # a relatedness verdict (or a verdict per package)
            elif last == "objective" and not (v is None or isinstance(v, str)):
                out.append(f"{path}.{k}")       # a verifier status, not the public objective text
            _key_hits(v, keys, f"{path}.{k}", out, depth + 1)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _key_hits(v, keys, f"{path}[{i}]", out, depth + 1)
    return out


def scan(probe: Probe, reach: Any, *, values: dict[str, float] | None = None, canaries: Iterable[str] = (),
         label: str = "", extra_keys: Iterable[str] = ()) -> None:
    """The kinds that inspect one body of reached material: key names (plus `extra_keys` that are evaluator data on
    this surface), canaries, exact values, free text."""
    data = _plain(reach)
    probe.check("key_name")
    for h in _key_hits(data, EVALUATOR_KEYS | frozenset(extra_keys))[:20]:
        probe.leak("key_name", f"{label}: evaluator key reached", path=h)
    cs = list(canaries)
    if cs:
        text = dump(data)
        probe.check("canary", len(cs))
        for c in cs:
            if c in text:
                probe.leak("canary", f"{label}: planted canary reached", canary=c.split("-")[1])
    if values:
        probe.check("exact_value", len(values))
        for name, v, p, x in exact_hits(values, data)[:20]:
            probe.leak("exact_value", f"{label}: evaluator value {name}={v!r} reached", path=p, found=x)
        probe.check("free_text", len(values))
        for name, v, p, x in exact_hits(values, data, text_numbers(data))[:20]:
            probe.leak("free_text", f"{label}: evaluator value {name}={v!r} written in text", path=p, found=x)


def regression(probe: Probe, rows: list[tuple[dict[str, float], float]], label: str, r_max: float = 0.98,
               noisy: set | None = None) -> None:
    """`rows`: (reached numeric features by path, evaluator value) per perturbed world. Leak: one feature with
    |r| >= r_max whose leave-one-out linear prediction error is below 5% of the target's spread. Features in `noisy`
    (paths that differ between two runs of the same world: the world's own nondeterminism, e.g. measured CPU time)
    are left out - a handful of worlds cannot tell a leak from chance on them."""
    probe.check("regression")
    y = np.array([r[1] for r in rows], dtype=np.float64)
    if len(rows) < 4 or float(np.std(y)) == 0.0:
        raise ValueError(f"{label}: regression probe needs >= 4 worlds with different evaluator values")
    common = set(rows[0][0])
    for f, _ in rows[1:]:
        common &= set(f)
    common -= {k for k in common if _generic(k) in (noisy or set())}
    best = (0.0, None)
    for k in sorted(common):
        x = np.array([r[0][k] for r in rows], dtype=np.float64)
        if float(np.std(x)) == 0.0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) > abs(best[0]):
            best = (r, k)
        if abs(r) >= r_max:
            errs = []
            for i in range(len(rows)):
                m = np.ones(len(rows), bool)
                m[i] = False
                A = np.vstack([x[m], np.ones(m.sum())]).T
                coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
                errs.append(abs(coef[0] * x[i] + coef[1] - y[i]))
            if max(errs) <= 0.05 * (float(y.max() - y.min())):
                probe.leak("regression", f"{label}: reached feature predicts the evaluator value", feature=k, r=r)
                return
    probe.notes.append(f"{label}: best |r| = {abs(best[0]):.3f} ({best[1]}) over {len(common)} features, "
                       f"{len(rows)} worlds")


def features(reach: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for p, v in numbers(_plain(reach)):
        if p not in out:
            out[p] = v
    return out


def _generic(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def diff_paths(a: Any, b: Any, path: str = "", out: set | None = None) -> set:
    """Generic paths (list indices dropped) where two JSON values differ."""
    out = set() if out is None else out
    if type(a) is not type(b):
        out.add(_generic(path))
    elif isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.add(_generic(f"{path}.{k}"))
            else:
                diff_paths(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.add(_generic(path) + "#len")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.add(_generic(path))
    return out


def counterfactual(probe: Probe, views: list[Any], label: str, replicate: Any = None) -> set:
    """Worlds that differ only in evaluator-side data must give the same participant view. With `replicate` (a
    second run of the first world, unperturbed) the paths that differ between the two runs are the world's own
    nondeterminism (externally measured outcomes such as CPU time) and are not counted; any other difference is a
    leak. Returns the noisy paths."""
    probe.check("counterfactual")
    base = _plain(views[0])
    noisy = diff_paths(base, _plain(replicate)) if replicate is not None else set()
    if noisy:
        probe.notes.append(f"{label}: {len(noisy)} paths vary between two runs of the same world (not compared): "
                           f"{sorted(noisy)[:5]}")
    for i, v in enumerate(views[1:], 1):
        d = diff_paths(base, _plain(v)) - noisy
        if d:
            probe.leak("counterfactual", f"{label}: participant view changes with evaluator-side data", world=i,
                       paths=sorted(d)[:8])
            return noisy
    return noisy


def rewards(probe: Probe, rs: Iterable[float], label: str) -> None:
    rs = [float(r) for r in np.asarray(list(rs), dtype=np.float64).ravel()]
    probe.check("reward_scalar", max(1, len(rs)))
    nz = sum(1 for r in rs if r != 0.0)
    if nz:
        probe.leak("reward_scalar", f"{label}: nonzero reward in the default mode", steps=nz, of=len(rs))


def tool_sweep(probe: Probe, tools: list[dict], call: Callable[[str, dict], dict], label: str,
               fill: Callable[[dict], dict | None] | None = None, max_calls: int = 400) -> list:
    """Registry oracles must be absent and refused; every other read-only tool is called once (with `fill(tool)` args
    or its required arguments left out) and its response returned for scanning."""
    from ..core import evaluator_only as EO
    names = {t["name"] for t in tools}
    probe.check("tool_sweep", len(EO.TOOLS))
    for name in sorted(EO.TOOLS):
        if name in names:
            probe.leak("tool_sweep", f"{label}: evaluator-only tool listed to the participant", tool=name)
        host = name.split(".", 1)[0]
        if host in {n.split(".", 1)[0] for n in names}:
            r = call(name, {})
            err = (r.get("error") or {}).get("code") if isinstance(r, dict) else None
            if r.get("ok") or err not in ("forbidden", "unknown_tool"):
                probe.leak("tool_sweep", f"{label}: evaluator-only tool not refused", tool=name, code=err)
    out = []
    for t in sorted(tools, key=lambda t: t["name"])[:max_calls]:
        if not (t.get("annotations") or {}).get("readOnlyHint") or t["name"] in EO.TOOLS:
            continue
        args = fill(t) if fill else {}
        if args is None:
            continue
        out.append({"tool": t["name"], "response": call(t["name"], args)})
    return out


def _fill_factory(ids: dict[str, str]):
    """Arguments for a read-only tool from the identifiers a participant holds (its own principal, org, account,
    host, repo); tools needing anything else get their optional arguments only."""
    def fill(t: dict) -> dict:
        sch = t.get("inputSchema") or {}
        props = sch.get("properties") or {}
        args = {}
        for k in sch.get("required") or []:
            v = ids.get(k)
            if v is None:
                for key, val in ids.items():
                    if key in k:
                        v = val
                        break
            if v is None:
                return {}
            args[k] = v
        for k in props:
            if k in ids and k not in args:
                args[k] = ids[k]
        return args
    return fill


# ====================================================================== surface: Tier 1 mechanism adapter
def probe_mech(families: tuple = ("scm_additive", "tech_pipeline"), steps: int = 24) -> dict:
    from ..research.tasks import make_task
    from .registry import make
    from .tier1 import MechExperienceEnv
    P = Probe("tier1 mechanism adapter (MechExperienceEnv)")
    for fam in families:
        task = make_task(fam, 3, validate=False)
        env = make(1, f"mech:{fam}", 3, task=task)                 # DEFAULT reward mode
        rs = []
        a = env.agents()[0]
        lv = env.execute(a, {"tool": "rig.spec", "args": {}})["result"]["levers"]
        rng = np.random.default_rng(0)
        for _ in range(steps):
            env.execute(a, {"tool": "rig.set", "args": {"settings": {x["name"]: float(rng.uniform(x["min"], x["max"]))
                                                                     for x in lv}}})
            env.advance(env.tick_s)
            rs.append(env.reward(a))
        rewards(P, rs, f"{fam} default mode")
        views, scores_by_world = [], []
        for k, priv in enumerate((None, 7777.123456, 31337.4242, 123456.789)):
            t = task if priv is None else replace(task, reference_net=priv, default_net=priv / 3.0, random_net=-priv)
            env = MechExperienceEnv(t, reward_mode=OBSERVATION_ONLY)
            got, scores, legit = _drive_mech(env, steps, seed=k)
            ev = env.verify()
            if priv is not None:
                scan(P, got, canaries=[repr(priv), f"{priv:.2f}"], label=f"{fam} private task numbers")
            vals = {f"tick{i}.score": s for i, s in enumerate(scores) if distinctive(s, legit[i])}
            scan(P, got, values=vals, label=f"{fam} tick scores", extra_keys=("score",))
            views.append(got)
            scores_by_world.append(ev)
        counterfactual(P, views[1:], f"{fam} private reference numbers")
    return P.result()


def _drive_mech(env, steps: int, seed: int) -> tuple[list, list[float], list[list[float]]]:
    a = env.agents()[0]
    got = [env.observe(a), env.tools(a)]
    spec = env.execute(a, {"tool": "rig.spec", "args": {}})
    got.append(spec)
    levers = spec["result"]["levers"]
    rng = np.random.default_rng(1000)          # same actions in every perturbed world
    scores, legit = [], []
    for _ in range(steps):
        st = {lv["name"]: float(np.round(rng.uniform(lv["min"], lv["max"]), 4)) for lv in levers}
        got.append(env.execute(a, {"tool": "rig.set", "args": {"settings": st}}))
        env.advance(env.tick_s)
        o = env.sims.last[env._seat[a]]          # evaluator side: the engine's own record of the tick
        scores.append(float(o["score"]))
        legit.append([float(o["revenue"]), float(o["cost"]), float(o["revenue"]) - float(o["cost"]),
                      float(o["shortfall"])] + [float(v) for v in o["readings"].values()] +
                     [float(v) for v in o["settings"].values()])
        got.append(env.execute(a, {"tool": "rig.read", "args": {}}))
        got.append(env.execute(a, {"tool": "rig.history", "args": {"n": 3}}))
        got.append(env.observe(a))
    got.append(env.observe_since(a, 0))
    return got, scores, legit


# ====================================================================== surface: Tier 1 rapid adapter
def probe_rapid(single: tuple = ("net_routing", "hc_confounded", "ch_sign_reversal", "lab_kinetics"),
                multi: tuple = ("team_commons", "coop_split_info"),
                steps: int = 12) -> dict:
    from .registry import make
    P = Probe("tier1 rapid adapter (RapidExperienceEnv)")
    for fam in single + multi:
        env = make(1, f"rapid:{fam}", 2)                            # DEFAULT reward mode
        rs = []
        for k in range(steps):
            for a in env.agents():
                env.execute(a, _rapid_noop(env, a, k))
            env.advance(getattr(env, "tick_s", 60))
            rs += [env.reward(a) for a in env.agents()]
        rewards(P, rs, f"{fam} default mode")
        views = []
        for target in (None, 123.456, -98765.4321):            # evaluator-only: the success threshold
            env = make(1, f"rapid:{fam}", 2, OBSERVATION_ONLY)
            if target is not None:
                env.env._config["objective"] = dict(env.env._config.get("objective") or {}, target=target)
            got = []
            for k in range(steps):
                for a in env.agents():
                    got.append(env.execute(a, _rapid_noop(env, a, k)))
                env.advance(getattr(env, "tick_s", 60))
                got += [env.observe(a) for a in env.agents()]
            got += [env.observe_since(a, 0) for a in env.agents()]
            views.append(got)
            scan(P, got, canaries=[] if target is None else [repr(target)], label=f"{fam} objective threshold")
        counterfactual(P, views, f"{fam} evaluator success threshold")
    for fam in single + multi:                           # the generated world's identity is not an observation
        env = make(1, f"rapid:{fam}", 5)
        a = env.agents()[0]
        names = {t["name"] for t in env.tools(a)}
        if "rapid.query" not in names:
            continue
        env.execute(a, {"tool": "rapid.query", "args": {"target": "spec"}})
        if getattr(env, "multi", False):
            env.advance(env.tick_s)
        res = env.observe(a)["last_result"] or {}
        P.check("key_name")
        spec = ((res.get("result") or {}) if isinstance(res.get("result"), dict) else {}).get("spec") or {}
        if not spec:
            raise RuntimeError(f"{fam}: the spec query returned nothing (the probe proves nothing): {res}")
        found = [k for k in ("family", "seed", "category", "split") if k in spec]
        if found:
            P.leak("key_name", f"{fam}: the spec query reveals the generated world's identity", keys=found)
    bystanders = 0
    for fam in multi:
        env = make(1, f"rapid:{fam}", 4, OBSERVATION_ONLY)
        ags = env.agents()
        if len(ags) < 3:
            continue                                     # a private message needs a third party to hide it from
        bystanders += 1
        secret = canary("PRIVATEMSG")
        r = env.execute(ags[0], {"tool": "rapid.message", "args": {"to": ags[2], "text": f"between us: {secret}"}})
        if not r.get("ok"):
            raise RuntimeError(f"{fam}: could not send the private message: {r}")
        env.advance(getattr(env, "tick_s", 60))
        bystander = [env.observe(ags[1]), env.observe_since(ags[1], 0)]
        scan(P, bystander, canaries=[secret], label=f"{fam} {ags[0]}->{ags[2]} private message seen by {ags[1]}")
        P.check("canary")
        evs = list(bystander[0]["events"]) + list(bystander[1][0])
        for e in evs:
            pl = e.get("payload") or {}
            if e.get("type") == "message" and ags[1] not in (pl.get("to"), e.get("actor"), pl.get("from")):
                P.leak("canary", f"{fam}: {ags[1]} received the metadata of a private message between others",
                       event=e)
                break
        if secret not in dump(_plain([env.observe(ags[2]), env.observe_since(ags[2], 0)])):
            raise RuntimeError(f"{fam}: the recipient never received the message (the probe proves nothing)")
    if not bystanders:
        raise ValueError("no multi-agent family with three participants: the private-message probe did not run")
    return P.result()


def _rapid_noop(env, agent: str, k: int) -> dict:
    """A harmless, valid action of the agent's own space: a query of its history."""
    names = {t["name"] for t in env.tools(agent)}
    if "rapid.query" in names:
        return {"tool": "rapid.query", "args": {"target": "history", "args": {"n": 2}}}
    return {"tool": sorted(names)[0], "args": {}}


# ====================================================================== surface: kernel world with v5 modules
V5_EXTRA = ("exp", "products", "subs", "inst", "commit", "use")
DAILY = (230_000, 170_000, 310_000, 90_000)        # genuine payments on days 1-4 (day 0: 400,000 + a circular flow)


def _revenue_goal(target: int = 1_000_000, window_days: int = 30):
    from ..goals.contracts import GoalContract
    pred = {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": window_days * DAY}, "op": ">=",
            "value": target}
    return GoalContract("revenue", f"Earn at least ${target // 100:,} of independent revenue over a rolling window.",
                        {"family": "tiny", "seed": 0}, "$self", pred,
                        {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": window_days * DAY},
                         "target": target}, 60 * DAY)


def _kernel_world(seed: int, mode: str | None, values: dict[str, str], window_days: int = 30):
    """tiny + v5 modules. The learner founds an org; on day 0 a genuine customer pays 400,000 and the learner pays
    500,000 to a friend who pays it to the org (the evaluator nets that out); on days 1-4 the customer pays `DAILY`.
    Everything the participant does and sees is the same in every call; only evaluator-side data differ: the window
    of the objective's metric (`window_days`) and the strings in `values` (evaluator runtime, an internal event,
    other participants' private mail, a world-spec evaluator parameter)."""
    from .kernel import KernelExperienceEnv
    kw = {} if mode is None else {"reward_mode": mode}
    env = KernelExperienceEnv({"family": "tiny", "overrides": {"n_consumers": 60}}, seed, agents=["learner"],
                              endowment=3_000_000, extra_modules=V5_EXTRA, **kw)
    w, a = env.world, env.agents()[0]

    def call(tool, **args):
        r = env.execute(a, {"tool": tool, "args": args})
        if not r["ok"]:
            raise RuntimeError(f"{tool}: {r.get('error')}")
        return r["result"]
    me = [x for x in call("bank.accounts") if x["owner"] == a][0]["account"]
    org = call("id.create_org", handle="learnco", name="Learn Co", kind="company")["org"]
    call("biz.register", org=org, account=me)
    oacct = call("bank.open_account", owner=org, name="operating")["account"]
    cust = w.system_call("admin.join", {"handle": "customer", "endowment": 5_000_000})["principal"]
    friend = w.system_call("admin.join", {"handle": "friend", "endowment": 10_000})["principal"]
    third = w.system_call("admin.join", {"handle": "third", "endowment": 10_000})["principal"]
    cacct = w.modules["bank"].primary_account(w.state, cust)
    facct = w.modules["bank"].primary_account(w.state, friend)

    def pay(amount):
        r = w.execute(cust, {"tool": "bank.transfer", "args": {"from_account": cacct, "to": oacct, "amount": amount,
                                                               "memo": "invoice"}})
        if not r.ok:
            raise RuntimeError(r.error)
    pay(400_000)
    call("bank.transfer", from_account=me, to=friend, amount=500_000, memo="loan")
    w.execute(friend, {"tool": "bank.transfer", "args": {"from_account": facct, "to": oacct, "amount": 500_000,
                                                         "memo": "invoice"}})
    w.runtime["evaluator_canary"] = values["RUNTIME"]
    w._run_system("canary", lambda ctx: ctx.emit("note", {"canary": values["INTERNAL"]}, vis="internal"))
    w.execute(friend, {"tool": "mail.send", "args": {"to": [third], "subject": "private",
                                                     "body": f"between us: {values['PRIVATEMAIL']}"}})
    w.spec.params["evaluator_note"] = values["SPEC"]
    for amount in DAILY:
        env.advance(DAY)
        pay(amount)
    env.advance(3600)
    env.set_objective(a, _revenue_goal(window_days=window_days))
    env.advance(3600)
    return env, a, {"me": me, "org": org, "oacct": oacct}


def probe_kernel(steps: int = 6) -> dict:
    from ..goals.metrics import evaluate_metric
    from .kernel import KernelExperienceEnv
    P = Probe("kernel world with v5 modules (KernelExperienceEnv)")
    # default mode, no objective: the 4.0 balance-change reward must not be paid
    env = KernelExperienceEnv({"family": "tiny", "overrides": {"n_consumers": 60}}, 1, agents=["learner"])
    a = env.agents()[0]
    acct = [x for x in env.execute(a, {"tool": "bank.accounts", "args": {}})["result"] if x["owner"] == a][0]
    payee = env.world.system_call("admin.join", {"handle": "payee", "endowment": 0})["principal"]
    rs = []
    for k in range(steps):
        env.execute(a, {"tool": "bank.transfer", "args": {"from_account": acct["account"], "to": payee,
                                                          "amount": 1000 + k, "memo": "x"}})
        env.advance(3600)
        rs.append(env.reward(a))
    rewards(P, rs, "kernel default mode, no objective")
    # nested calls (deferred workflow jobs, triggers, flows) in a v5-module world and in the 4.0 mechanism facility
    # (whose rig.read/rig.history results carry the evaluator-only `score`)
    from .probes_tiers import probe_nested
    nenv = KernelExperienceEnv({"family": "tiny", "overrides": {"n_consumers": 60}}, 2, agents=["learner"],
                               extra_modules=V5_EXTRA + ("wf",))
    probe_nested(P, nenv, nenv.agents()[0], "kernel v5")
    import crucible
    crucible.load_all_modules()
    from ..research.tasks import make_task
    from ..worldgen import generate
    from ..worldgen import mech_families  # noqa: F401 - registers the mechanism facility world families
    task = make_task("tech_pipeline", 3, validate=False)
    cfg, wseed = task.world_config()
    fenv = KernelExperienceEnv(generate(cfg, wseed), agents=["agent"], endowment=task.endowment, extra_modules=["wf"])
    fa = fenv.agents()[0]
    lv = fenv.execute(fa, {"tool": "rig.spec", "args": {}})["result"]["levers"][0]
    fenv.execute(fa, {"tool": "rig.set", "args": {"settings": {lv["name"]: lv["max"]}}})
    fenv.advance(3 * 3600)
    probe_nested(P, fenv, fa, "mechanism facility (kernel)", unit=3600)
    tags = ("RUNTIME", "INTERNAL", "PRIVATEMAIL", "SPEC")
    views, rows = [], []
    legit = [400_000.0, 500_000.0, 900_000.0, *map(float, DAILY)]
    for k, window in enumerate((30, 1, 2, 3, 4)):
        vals = {t: canary(t) for t in tags} if k == 0 else {t: "N" * 31 for t in tags}
        env, a, ids = _kernel_world(7, OBSERVATION_ONLY, vals, window_days=window)
        w = env.world
        truth = float(evaluate_metric(w, "independent_revenue", {"principal": ids["org"], "window_s": window * DAY}))
        status = env.verify()["agents"][a]
        got = [env.observe(a), env.tools(a)]
        ids_all = {"principal": a, "org": ids["org"], "account": ids["me"], "owner": ids["org"],
                   "from_account": ids["me"], "handle": "learnco"}
        got += tool_sweep(P, env.tools(a), lambda t, ar: env.execute(a, {"tool": t, "args": ar}), "kernel v5",
                          fill=_fill_factory(ids_all))
        for _ in range(steps):
            env.advance(3600)
            got.append(env.observe(a))
        got.append(env.observe_since(a, 0))
        vals_eval = {"independent_revenue": truth}
        if status.get("progress") is not None:
            vals_eval["verifier.progress"] = float(status["progress"])
        vals_eval = {k2: v for k2, v in vals_eval.items() if distinctive(v, legit)}
        scan(P, got, values=vals_eval, canaries=list(vals.values()) if k == 0 else [], label=f"kernel world {k}")
        if k > 0:
            rows.append((features(got), truth))
            views.append(got)
        if k == 0:
            _predicate_oracles(P, env, a, ids, truth)
    env, a, ids = _kernel_world(7, OBSERVATION_ONLY, {t: "N" * 31 for t in tags}, window_days=1)  # = world 1, again
    rep_got = [env.observe(a), env.tools(a)]
    rep_got += tool_sweep(Probe("replicate"), env.tools(a), lambda t, ar: env.execute(a, {"tool": t, "args": ar}),
                          "kernel v5", fill=_fill_factory({"principal": a, "org": ids["org"], "account": ids["me"],
                                                           "owner": ids["org"], "from_account": ids["me"],
                                                           "handle": "learnco"}))
    for _ in range(steps):
        env.advance(3600)
        rep_got.append(env.observe(a))
    rep_got.append(env.observe_since(a, 0))
    noisy = counterfactual(P, views, "kernel evaluator window and evaluator-side strings", replicate=rep_got)
    regression(P, rows, "kernel independent revenue (evaluator window perturbed, participant view fixed)",
               noisy=noisy)
    return P.result()


def _predicate_oracles(P: Probe, env, a: str, ids: dict, truth: float) -> None:
    """Every participant-authored predicate over an evaluator-only metric must be refused; when one is accepted, an
    instant check is used to binary-search the metric (showing what the channel gives away)."""
    org = ids["org"]
    pred = {"metric": "independent_revenue", "args": {"principal": org, "window_s": 30 * DAY}, "op": ">=",
            "value": 1}

    def ex(tool, **args):
        return env.execute(a, {"tool": tool, "args": args})
    tools = {t["name"] for t in env.tools(a)}
    # 1 self-goal over an evaluator metric
    P.check("predicate_oracle")
    r = ex("goals.propose", title="probe", predicate=pred, due_in_s=3600)
    if r["ok"]:
        P.leak("predicate_oracle", "goals.propose accepted an evaluator-only metric", metric="independent_revenue")
        gid = r["result"]["goal"]
        c = ex("goals.check_now", goal=gid)
        if c["ok"]:
            lo, hi, n = 0, 2_000_000, 0
            while hi - lo > 1 and n < 40:
                mid = (lo + hi) // 2
                g = ex("goals.propose", title=f"p{n}", predicate=dict(pred, value=mid), due_in_s=3600)
                env.advance(3601)
                n += 1
                if not g["ok"]:
                    break
                cc = ex("goals.check_now", goal=g["result"]["goal"])
                if not cc["ok"]:
                    break
                lo, hi = (mid, hi) if cc["result"].get("currently_true") else (lo, mid)
            P.leak("predicate_oracle", "goals.check_now binary search recovered the evaluator metric",
                   recovered=lo, truth=truth, queries=n)
    # 2 the instant check itself
    P.check("predicate_oracle")
    if "goals.check_now" in tools:
        P.leak("predicate_oracle", "goals.check_now (instant evaluation of predicates) is a participant tool")
    # 3 an experiment measured by an evaluator-backed tool
    P.check("predicate_oracle")
    r = ex("exp.register", hypothesis="objective met?", measure={"tool": "goals.check_now", "args": {"goal": "sg_x"},
                                                                 "field": "currently_true"},
           prediction={"op": "==", "value": 1}, deadline_in_s=3600)
    if r["ok"] or (r.get("error") or {}).get("code") not in ("forbidden", "unknown_tool"):
        P.leak("predicate_oracle", "exp.register accepted an evaluator-backed measure",
               code=(r.get("error") or {}).get("code"))
    # 4 a charter condition over an evaluator metric
    P.check("predicate_oracle")
    charter = {"procedures": {"vote": {"type": "majority", "electorate": "members"}},
               "rules": [{"match": {"kind": "tool", "tool": "bank.transfer", "when": dict(pred, value=100_000)},
                          "procedure": "vote"}]}
    r = ex("inst.adopt", org=org, charter=charter)
    if r["ok"]:
        P.leak("predicate_oracle", "inst.adopt accepted a charter condition over an evaluator-only metric")
    elif (r.get("error") or {}).get("code") not in ("forbidden",):
        P.notes.append(f"inst.adopt refused for another reason: {r.get('error')}")
    # 5 a commitment milestone (deliverable predicate) over an evaluator metric
    P.check("predicate_oracle")
    w = env.world
    other = w.system_call("admin.join", {"handle": "counterpart", "endowment": 100_000})["principal"]
    terms = {"title": "probe", "payer": "$you", "milestones": [{"amount_cents": 100, "due_in_s": 2 * DAY,
                                                                 "deliverable": {"kind": "predicate",
                                                                                 "predicate": pred}}]}
    r = ex("commit.offer", to=other, terms=terms, as_org=org)
    if r["ok"]:
        P.leak("predicate_oracle", "commit.offer accepted a deliverable over an evaluator-only metric")
    elif (r.get("error") or {}).get("code") != "forbidden":
        raise RuntimeError(f"commit.offer refused for another reason, the probe proves nothing: {r.get('error')}")


# ====================================================================== surface: innovation registry
def probe_innovation(hours: int = 40) -> dict:
    from ..core.kernel import World
    from ..innovation.reference import ServiceCreator, drive
    from ..innovation.worlds import innovation_spec
    from .kernel import KernelExperienceEnv
    P = Probe("innovation registry (caps) through KernelExperienceEnv")
    spec = innovation_spec(11, participants=12, adopt={"start_hours": [8, 20], "per_day": [6, 12]})
    w = World.create(spec)
    ana = ServiceCreator("ana", "route.shortest", "anaroute", tier=1, price=3, at=3 * 3600)
    ana.ensure(w)
    env = KernelExperienceEnv(w, agents=["p:ana"], reward_mode=OBSERVATION_ONLY)
    a = env.agents()[0]
    drive(w, [ana], hours * 3600)
    got = [env.observe(a), env.observe_since(a, 0)]
    got.append(env.execute(a, {"tool": "caps.search", "args": {"query": "route shortest"}}))
    got.append(env.execute(a, {"tool": "caps.declarations", "args": {}}))
    n_adopt = sum(1 for e in got[1][0] if e["type"] == "caps.adopted")
    P.notes.append(f"caps.adopted events visible to the provider: {n_adopt}")
    scan(P, got, label="innovation provider view")
    if n_adopt == 0:
        raise RuntimeError("no adoption happened: the registry probe saw no verdict-carrying event")
    return P.result()


# ====================================================================== surface: self-play environments
def probe_selfplay(steps: int = 16) -> dict:
    from ..research.tasks import make_task
    from ..selfplay.arena import KernelSelfPlayEnv, SelfPlayEnv
    P = Probe("self-play environments (SelfPlayEnv, KernelSelfPlayEnv)")
    task = make_task("tech_shared_pool", 2, validate=False)    # its objective quantity is not the seat's net
    rng = np.random.default_rng(0)
    acts = [[rng.uniform(-1, 1, size=(1, 12)) for _ in range(2)] for _ in range(steps)]
    for cls in (SelfPlayEnv, KernelSelfPlayEnv):
        env = cls([task]) if cls is SelfPlayEnv else cls(task)         # DEFAULT reward mode
        env.reset()
        rs, slots = [], []
        for a in acts:
            o, r, d, info = env.step(a)
            rs.append(np.asarray(r).ravel())
            slots += [float(np.asarray(x)[0, env.spec.scalars.start]) for x in o]
            if np.asarray(d).all():
                break
        rewards(P, np.concatenate(rs).tolist() + slots, f"{cls.__name__} default mode")
        env = cls([task], reward_mode=OBSERVATION_ONLY) if cls is SelfPlayEnv else cls(task, reward_mode=OBSERVATION_ONLY)
        got = [env.reset()]
        for a in acts:
            out = env.step(a)
            got.append(list(out))
            vals, legit = {}, []
            if cls is SelfPlayEnv:
                sim = env.sims[0]
                for s in range(2):
                    o = sim.last[s]
                    legit += [float(o["revenue"]), float(o["cost"]), float(o["revenue"]) - float(o["cost"])]
                for s in range(2):
                    v = float(sim.last[s]["score"])
                    if distinctive(v, legit):
                        vals[f"seat{s}.score"] = v
            scan(P, out, values=vals, label=f"{cls.__name__} step", extra_keys=("score",))
            if np.asarray(out[2]).all():
                break
    return P.result()


# ====================================================================== surface: numeric environments
def probe_numeric(steps: int = 12) -> dict:
    from ..research.bridge import KernelMechEnv
    from ..research.envs import MechVecEnv
    from ..research.tasks import make_task
    from ..scale.rollout import WorldEnv
    P = Probe("numeric environments (MechVecEnv, KernelMechEnv, WorldEnv)")
    task = make_task("scm_additive", 3, validate=False)
    rng = np.random.default_rng(0)
    for name, env in (("MechVecEnv", MechVecEnv([task, task])), ("KernelMechEnv", KernelMechEnv(task))):
        env.reset()
        rs = []
        for _ in range(steps):
            o, r, d, info = env.step(rng.uniform(-1, 1, size=(env.n, 12)))
            rs += list(np.asarray(r).ravel()) + list(np.asarray(o)[:, env.spec.scalars.start])
            if np.asarray(d).all():
                break
        rewards(P, rs, f"{name} default mode")
    env = WorldEnv({"family": "tiny", "seed": 2, "overrides": {"n_consumers": 60}}, think_s=3600)
    env.world.system_call("admin.join", {"handle": "payee", "endowment": 0})
    acct = env.world.modules["bank"].primary_account(env.world.state, env.actor)
    env.reset()
    rs = []
    for k in range(steps):
        o, r, d, info = env.step({"tool": "bank.transfer", "args": {"from_account": acct, "to": "p:payee",
                                                                    "amount": 500 + k, "memo": "x"}})
        rs.append(r)
    env.close()
    rewards(P, rs, "WorldEnv default mode")
    return P.result()


# ====================================================================== surface: Gymnasium / PettingZoo
def _cash_goal(target: int = 2_000_000):
    from ..goals.contracts import GoalContract
    return GoalContract("cash", f"Hold at least ${target // 100:,} in your own account.", {"family": "tiny", "seed": 0},
                        "$self", {"metric": "cash", "args": {"principal": "$self"}, "op": ">=", "value": target},
                        {"metric": "cash", "args": {"principal": "$self"}, "target": target}, 30 * DAY)


def probe_gym(steps: int = 8) -> dict:
    from ..api.gym_env import CrucibleEnv
    from ..api.pettingzoo_env import CrucibleParallelEnv
    from ..goals.contracts import RewardContract
    P = Probe("Gymnasium / PettingZoo (api.gym_env, api.pettingzoo_env, experience.gym)")
    env = CrucibleEnv(spec={"family": "tiny", "overrides": {"n_consumers": 60}}, seed=4, endowment=1_500_000,
                      objective=RewardContract(_cash_goal()), dt=3600, max_steps=steps)
    env.server.join_world(env.world_id, "payee", endowment=0)
    w = env.server.world(env.world_id)
    acct = w.modules["bank"].primary_account(w.state, env.principal)
    obs, info = env.reset()
    got, rs = [obs, info], []
    for k in range(steps):
        obs, r, term, trunc, info = env.step({"tool": "bank.transfer", "args": {"from_account": acct, "to": "p:payee",
                                                                                "amount": 1000 + 100 * k, "memo": "t"}})
        rs.append(r)
        st = env.verify()
        vals = {"verifier.progress": float(st["progress"])} if st.get("progress") is not None else {}
        cash = float(sum(x["balance"] for x in w.state.tables["accounts"].values() if x["owner"] == env.principal))
        vals = {k2: v for k2, v in vals.items() if distinctive(v, [cash])}
        got += [obs, info]
        scan(P, [obs, info], values=vals, label="CrucibleEnv default mode")
        if term or trunc:
            break
    rewards(P, rs, "CrucibleEnv default mode")
    env.close()
    pz = CrucibleParallelEnv(spec={"family": "tiny", "overrides": {"n_consumers": 60}}, seed=5, agents=["ann", "ben"],
                             endowment=1_500_000, objectives={"ann": RewardContract(_cash_goal()),
                                                              "ben": RewardContract(_cash_goal())},
                             dt=3600, max_steps=steps)
    wz = pz.server.world(pz.world_id)
    accts = {a: wz.modules["bank"].primary_account(wz.state, i.principal) for a, i in pz.identities.items()}
    obs, infos = pz.reset()
    rs = []
    for k in range(steps):
        acts = {a: {"tool": "bank.transfer", "args": {"from_account": accts[a], "to": "p:ben" if a == "ann" else "p:ann",
                                                      "amount": 700 + 37 * k + (11 if a == "ann" else 0), "memo": "x"}}
                for a in pz.agents}
        obs, rew, term, trunc, infos = pz.step(acts)
        rs += list(rew.values())
        scan(P, [obs, infos], label="CrucibleParallelEnv default mode")
        if not pz.agents:
            break
    rewards(P, rs, "CrucibleParallelEnv default mode")
    pz.close()
    try:
        from . import gym as XG
    except ImportError:
        P.notes.append("crucible.experience.gym (ExperienceEnv wrappers) not present")
        return P.result()
    for fam in ("mech:scm_additive", "rapid:net_routing"):
        g = XG.ExperienceGymEnv(1, fam, 3, max_steps=steps)
        o, info = g.reset()
        rs = []
        for _ in range(steps):
            o, r, term, trunc, info = g.step(g.action_space.sample())
            rs.append(r)
            scan(P, [o, info], label=f"ExperienceGymEnv {fam}")
            if term or trunc:
                break
        rewards(P, rs, f"ExperienceGymEnv {fam} default mode")
    return P.result()


# ====================================================================== surface: AGENTRS adapters
def _agentrs() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", "src"))
    if os.path.isdir(os.path.join(root, "agentrs")) and root not in sys.path:
        sys.path.insert(0, root)


def probe_sl(steps: int = 10) -> dict:
    _agentrs()
    from agentrs.sl.world import Task
    from ..integrations.agentrs_sl import CrucibleSLWorld
    P = Probe("AGENTRS superlearner adapter (integrations.agentrs_sl)")
    w = CrucibleSLWorld("solvency", "tiny", 0, 3600, max_decisions=steps)        # DEFAULT reward mode
    w.initialize_world(Task("crucible/solvency", 1, "train"))
    rs, got = [], [w.goal()]
    bad = ["bank", "transfer"]                   # an incomplete (schema-invalid) call
    for k in range(steps):
        sp = w.available_actions()
        act = bad if k % 2 == 0 else ["wait", "END"]
        if act[0] != "wait" and act[0] not in sp.allowed([]):
            act = ["wait", "END"]
        try:
            w.execute("0", act)
        except Exception:                        # noqa: BLE001 - an undecodable call: treat as a wait
            w.execute("0", ["wait", "END"])
        w.advance_time()
        c = w.observe_consequences()
        rs.append(c.reward)
        got += [w.observe(), c]
        st = w.verifier.status(w.world)
        P.check("termination")
        if c.done and st.get("success") and w.n_dec < w.max_decisions:
            P.leak("termination", "sl episode ended because the verifier saw success")
        if c.done:
            break
    # the adapter's native penalty for a schema-invalid call: the world refused the last call as invalid
    w.execute("0", ["wait", "END"])
    w.last_invalid = True
    rs.append(w.observe_consequences().reward)
    rewards(P, rs, "CrucibleSLWorld default mode")
    goal = _plain(w.goal())
    P.check("key_name")
    if any(isinstance(t, str) and t.startswith("metric:") for t in goal.get("tokens", [])):
        P.leak("key_name", "sl goal() names the evaluator's progress metric", tokens=goal.get("tokens"))
    scan(P, got, label="CrucibleSLWorld")
    w.close()
    return P.result()


def probe_cam2(steps: int = 6) -> dict:
    _agentrs()
    from agentrs.cam2.envs.base import TaskRef
    from agentrs.cam2.tools import ToolCall
    from ..integrations.agentrs_cam2 import CrucibleAdapter
    P = Probe("CAM 2.0 adapter (integrations.agentrs_cam2)")
    ad = CrucibleAdapter(max_steps=steps)                                         # DEFAULT reward mode
    obs = ad.reset(TaskRef("crucible", "solvency:tiny:0", 7))
    rs, got = [], [obs]
    for k in range(steps):
        r = ad.step(ToolCall("bank.accounts") if k % 2 else ToolCall("bank.transfer"))
        rs.append(r.reward)
        got += [r.obs, r.info]
        if r.terminated or r.truncated:
            break
    rewards(P, rs, "CAM 2.0 adapter default mode")
    scan(P, got, label="CAM 2.0 adapter")
    return P.result()


# ====================================================================== all
def probe_civ() -> dict:
    """Tier 3 (a `crucible.civ` civilisation): see `crucible.experience.probes_tiers.probe_civ`."""
    from .probes_tiers import probe_civ as _p
    return _p()


SURFACES: dict[str, Callable[[], dict]] = {
    "mech": probe_mech, "rapid": probe_rapid, "kernel": probe_kernel, "innovation": probe_innovation,
    "selfplay": probe_selfplay, "numeric": probe_numeric, "gym": probe_gym, "sl": probe_sl, "cam2": probe_cam2,
    "civ": probe_civ,
}


def run_all(surfaces: Iterable[str] | None = None, errors: str = "raise") -> dict:
    """Every surface's probes and the aggregate metrics. `errors="record"` keeps going after a surface fails to run
    (the failure is reported, and counted as a leak: a probe that could not run proves nothing)."""
    out, errs = [], {}
    for name in (list(surfaces) if surfaces is not None else list(SURFACES)):
        try:
            out.append(dict(SURFACES[name](), name=name))
        except Exception as e:  # noqa: BLE001
            if errors == "raise":
                raise
            errs[name] = f"{type(e).__name__}: {e}"
    kinds_checked = sorted({k for r in out for k, v in r["kinds"].items() if v["checked"] > 0})
    leaks = sum(r["n_leaks"] for r in out) + len(errs)
    default_rewards = sum(r["kinds"].get("reward_scalar", {}).get("leaks", 0) for r in out)
    by_kind = {k: {"checked": sum(r["kinds"].get(k, {}).get("checked", 0) for r in out),
                   "leaks": sum(r["kinds"].get(k, {}).get("leaks", 0) for r in out)} for k in KINDS}
    return {"metrics": {"reward_leak_probe_kinds": len(kinds_checked), "reward_leaks_found": leaks,
                        "nonzero_scalar_rewards_default_mode": default_rewards, "surfaces_probed": len(out),
                        "surfaces_failed_to_run": len(errs)},
            "kinds_checked": kinds_checked, "by_kind": by_kind, "surfaces": out, "errors": errs}


if __name__ == "__main__":          # python -m crucible.experience.probes [surface ...]
    res = run_all(sys.argv[1:] or None, errors="record")
    print(json.dumps(res, indent=1, default=str))


# ====================================================================== positive control
def positive_control() -> dict:
    """The probes must find the leaks the 4.0 contract has: the same probes in an explicitly selected
    ``legacy_shaped_reward`` world (evaluator tools available to participants, the mechanism's score in reports)
    recover the evaluator's values. Returns the leaks found per kind (all must be > 0 for the probes to be
    sensitive)."""
    from ..goals.metrics import evaluate_metric
    from ..research.tasks import make_task
    from ..selfplay.arena import SelfPlayEnv
    P = Probe("positive control (legacy_shaped_reward, the 4.0 contract)")
    tags = ("RUNTIME", "INTERNAL", "PRIVATEMAIL", "SPEC")
    env, a, ids = _kernel_world(7, LEGACY, {t: "N" * 31 for t in tags}, window_days=30)
    truth = float(evaluate_metric(env.world, "independent_revenue", {"principal": ids["org"], "window_s": 30 * DAY}))
    _predicate_oracles(P, env, a, ids, truth)
    tool_sweep(P, env.tools(a), lambda t, ar: env.execute(a, {"tool": t, "args": ar}), "legacy kernel")
    task = make_task("tech_shared_pool", 2, validate=False)
    sp = SelfPlayEnv([task], reward_mode=LEGACY)
    sp.reset()
    rng = np.random.default_rng(0)
    for _ in range(4):
        out = sp.step([rng.uniform(-1, 1, size=(1, 12)) for _ in range(2)])
        vals = {f"seat{s}.score": float(sp.sims[0].last[s]["score"]) for s in range(2)}
        scan(P, out, values=vals, label="legacy SelfPlayEnv", extra_keys=("score",))
    rewards(P, np.asarray(out[1]).ravel().tolist(), "legacy SelfPlayEnv")
    r = P.result()
    r["kinds_with_leaks"] = sorted(k for k, v in r["kinds"].items() if v["leaks"] > 0)
    return r
