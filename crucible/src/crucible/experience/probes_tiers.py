"""Information-flow probes per tier (OO.14, OO.16; R.2/R.3/OO.2 on v5 worlds) and the nested-call probe.

`crucible.experience.probes` takes the attacker's position on each learner-facing surface. This module adds what the
recovery audit and the wave-2 merges made necessary:

* ``nested_call``: evaluator data must not come back through calls a participant does not make itself in the same
  transaction - deferred workflow jobs (`wf.run_later`), triggers (`wf.trigger`), flows (`wf.run`). Every
  evaluator-only tool of the registry is called through each channel (it must fail), and every tool whose result
  carries evaluator-only fields is read through a deferred job (the job's recorded result and its `wf.job_done`
  event must not carry them). Before the kernel fix of this workstream (wave 3), a deferred job returned the
  mechanism's `score` that a direct `rig.read` strips.
* ``identity``: no family, seed, split, domain configuration or civilisation profile of the generated world appears in
  what the participant reaches (OO.16), searched as strings (family/profile/config names, the spec hash and world id)
  and as numbers (a distinctive seed).
* per-tier surfaces: Tier 1 (`make(1, ...)`), Tier 2 (`make(2, ...)`, a `crucible.domains.base.Domain`) and Tier 3
  (`make(3, profile)`, a `crucible.civ` civilisation), each probed on at least four leak channels - observation,
  auxiliary/info fields (the Gymnasium wrapper's `info`), tool responses (a sweep over every read-only tool with
  arguments harvested from what the participant has seen), event metadata (`observe_since`) - plus free text, with
  evaluator values made distinctive by construction (a circular payment flow the evaluator nets out, perturbed
  objective windows), planted canaries, predicate oracles, counterfactual and regression analysis.

A tier whose worlds cannot be built raises (`TierUnavailable` or the generator's own error): a probe that cannot run
never reports "no leak".
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .modes import OBSERVATION_ONLY
from .probes import (DAY, Probe, _plain, _predicate_oracles, canary, counterfactual, distinctive, dump, features,
                     numbers, regression, rewards, scan, strings, tool_sweep)

HOUR = 3600
CHANNELS = ("observation", "info", "tool_response", "event_metadata", "free_text")
CIV_PROBE_PROFILE = "balanced"
IDENTITY_SEED = 918_273                    # a distinctive seed: found as a number = the seed reached the participant
PAYMENTS = (230_000, 170_000, 310_000, 90_000)


class TierProbe(Probe):
    """A probe that also records which leak channels it scanned."""

    def __init__(self, surface: str, tier: int):
        super().__init__(surface)
        self.tier = int(tier)
        self.channels: dict[str, int] = {}

    def scan_channel(self, channel: str, reach: Any, **kw) -> None:
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel}")
        self.channels[channel] = self.channels.get(channel, 0) + 1
        if kw.get("values"):
            self.channels["free_text"] = self.channels.get("free_text", 0) + 1     # scan() also reads the text
        scan(self, reach, label=f"{kw.pop('label', self.surface)} [{channel}]", **kw)

    def result(self) -> dict:
        r = super().result()
        r.update(tier=self.tier, channels=dict(sorted(self.channels.items())),
                 channels_probed=sorted(c for c, n in self.channels.items() if n > 0))
        return r


# ====================================================================== helpers
def _ok(env, agent: str, tool: str, **args) -> Any:
    r = env.execute(agent, {"tool": tool, "args": args})
    if not r["ok"]:
        raise RuntimeError(f"{tool}: {r.get('error')}")
    return r["result"]


def harvest(reach: Any, ids: dict | None = None, per_key: int = 4) -> dict[str, list]:
    """Identifiers by key name from reached material (sorted, deterministic): what a participant could pass back as
    arguments."""
    out: dict[str, list] = {k: list(v) for k, v in (ids or {}).items()}

    def walk(x, key, depth=0):
        if depth > 30:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, str(k), depth + 1)
        elif isinstance(x, list):
            for v in x[:60]:
                walk(v, key, depth + 1)
        elif key and isinstance(x, (str, int)) and not isinstance(x, bool):
            if isinstance(x, str) and (len(x) > 120 or " " in x):
                return
            lst = out.setdefault(key, [])
            if x not in lst and len(lst) < per_key:
                lst.append(x)
    walk(_plain(reach), "")
    return {k: sorted(v, key=lambda z: (isinstance(z, str), str(z))) for k, v in sorted(out.items())}


def harvest_fill(ids: dict[str, list]):
    """`fill(tool)` for `tool_sweep`: each required argument from a harvested identifier of the same (or a
    suffix-matching) key and the schema's type; tools needing anything else are skipped."""
    def fill(t: dict) -> dict | None:
        sch = t.get("inputSchema") or {}
        props = sch.get("properties") or {}
        args = {}
        for k in sch.get("required") or []:
            cand = ids.get(k) or next((v for kk, v in ids.items() if kk.endswith(k) or k.endswith(kk)), None)
            if not cand:
                return None
            typ = (props.get(k) or {}).get("type")
            pick = [c for c in cand if (isinstance(c, int) if typ == "integer" else isinstance(c, str))
                    ] if typ in ("integer", "string") else list(cand)
            if not pick:
                return None
            args[k] = pick[0]
        return args
    return fill


def gym_info(env, agent: str, steps: int, dt: int) -> list:
    """The auxiliary/info channel: the Gymnasium wrapper's `info` dicts (and observations, reward) over the same
    environment, stepping with no-op waits."""
    from .gym import ExperienceGymEnv
    g = ExperienceGymEnv(env=env, agent=agent, think_s=int(dt), max_steps=None)
    out = []
    obs, info = g.reset()
    out.append({"info": info})
    for _ in range(steps):
        obs, r, term, trunc, info = g.step(None)
        out.append({"info": info, "reward": float(r), "terminated": bool(term), "truncated": bool(trunc)})
    return out


# ====================================================================== nested calls
def probe_nested(P: Probe, env, agent: str, label: str, *, unit: int = HOUR) -> None:
    """Evaluator data through calls made on the participant's behalf later or indirectly (workflows)."""
    from ..core import evaluator_only as EO
    names = {t["name"] for t in env.tools(agent)}
    if "wf.define" not in names:
        if getattr(env, "tier", None) == 2 and "wf" not in env.world.modules:
            # a Tier 2 domain is a restriction of the civilisation (D-17): a world without the workflow module has no
            # call-on-behalf channel to probe; recorded, never counted as "no leak"
            P.notes.append(f"{label}: no workflow module in this domain world - nested-call channel not present")
            return
        raise RuntimeError(f"{label}: no workflow module (wf) - the nested-call probe proves nothing here")

    def ex(_tool, **args):
        return env.execute(agent, {"tool": _tool, "args": args})
    jobs = []
    # 1 every evaluator tool through a flow (now and deferred) and a trigger: must never succeed
    for tool in sorted(EO.TOOLS):
        P.check("nested_call", 3)
        d = ex("wf.define", name=f"probe {tool}", steps=[{"id": "s1", "tool": tool, "args": {}}])
        if not d["ok"]:
            continue                    # refused at definition: nothing to run
        fid = d["result"]["flow"]
        r = ex("wf.run", flow=fid)
        if r["ok"]:
            P.leak("nested_call", f"{label}: a flow ran the evaluator tool {tool}", tool=tool)
        r = ex("wf.run_later", flow=fid, in_s=60)
        if r["ok"]:
            jobs.append((r["result"]["job"], tool, None))
        r = ex("wf.trigger", tool=tool, args={}, every_s=600)
        if r["ok"]:
            jobs.append((r["result"].get("trigger"), tool, "trigger"))
    # 2 every tool whose result carries evaluator-only fields, read through a deferred job
    filtered = sorted(t for t in EO.RESULT_FIELDS if t in names)
    for tool in filtered:
        P.check("nested_call")
        d = ex("wf.define", name=f"read {tool}", steps=[{"id": "s1", "tool": tool, "args": {}, "optional": True}])
        if not d["ok"]:
            raise RuntimeError(f"{label}: could not define a flow over {tool}: {d.get('error')}")
        r = ex("wf.run_later", flow=d["result"]["flow"], in_s=60)
        if not r["ok"]:
            raise RuntimeError(f"{label}: could not schedule a flow over {tool}: {r.get('error')}")
        jobs.append((r["result"]["job"], tool, "fields"))
    env.advance(max(int(unit), 900))
    got = [ex("wf.jobs"), ex("wf.triggers"), env.observe_since(agent, 0)]
    text = dump(_plain(got))
    done = {j.get("job"): j for j in (got[0].get("result") or [])} if got[0]["ok"] else {}
    for jid, tool, kind in jobs:
        if kind is None:
            j = done.get(jid)
            if j is None:
                raise RuntimeError(f"{label}: deferred job {jid} over {tool} never ran (the probe proves nothing)")
            if j.get("ok"):
                P.leak("nested_call", f"{label}: a deferred job ran the evaluator tool {tool}", tool=tool)
        elif kind == "trigger":
            fired = [e for e in got[2][0] if e["type"] == "wf.fired" and e["payload"].get("id") == jid]
            if any(e["payload"].get("ok") for e in fired):
                P.leak("nested_call", f"{label}: a trigger ran the evaluator tool {tool}", tool=tool)
    for tool in filtered:
        keys = EO.RESULT_FIELDS[tool]
        if any(f'"{k}"' in text for k in keys):
            hit = [k for k in keys if f'"{k}"' in text]
            P.leak("nested_call", f"{label}: a deferred job over {tool} returned evaluator fields", fields=hit)
    P.notes.append(f"{label}: nested calls over {len(EO.TOOLS)} evaluator tools x 3 channels and "
                   f"{len(filtered)} field-filtered tools ({filtered})")


# ====================================================================== the kernel-world scenario (tiers 0, 2, 3)
def _revenue_goal(window_s: int, target: int = 1_000_000):
    from ..goals.contracts import GoalContract
    pred = {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": int(window_s)}, "op": ">=",
            "value": target}
    return GoalContract("revenue", f"Earn at least ${target // 100:,} of independent revenue over a rolling window.",
                        {"family": "probe", "seed": 0}, "$self", pred,
                        {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": int(window_s)},
                         "target": target}, 60 * DAY)


def revenue_scenario(env, values: dict[str, str], *, unit: int, window_units: int, handle: str = "learnco",
                     target: int = 1_000_000) -> dict:
    """The learner founds an organisation; a genuine customer pays 400,000; the learner pays 500,000 to a friend
    who pays it to the organisation (the evaluator nets that out); then the customer pays `PAYMENTS`, one per `unit`
    seconds. Only evaluator-side data differ between calls: the window of the objective's metric and the strings in
    `values` (evaluator runtime, an internal event, other participants' private mail, a spec evaluator parameter)."""
    w, a = env.world, env.agents()[0]
    me = [x for x in _ok(env, a, "bank.accounts") if x["owner"] == a][0]["account"]
    have = int(w.state.tables["accounts"][me]["balance"])
    if have < 600_000:
        # a domain seat may start with little money (Seat.endowment); the scenario's loan needs 500,000: the operator
        # endows the seat (a recorded admin action, the same in every perturbed world)
        w.system_call("admin.endow", {"principal": a, "amount": 600_000 - have})
    org = _ok(env, a, "id.create_org", handle=handle, name="Learn Co", kind="company")["org"]
    if "biz.register" in {t["name"] for t in env.tools(a)}:
        _ok(env, a, "biz.register", org=org, account=me)
    oacct = _ok(env, a, "bank.open_account", owner=org, name="operating")["account"]
    cust = w.system_call("admin.join", {"handle": f"{handle}-cust", "endowment": 5_000_000})["principal"]
    friend = w.system_call("admin.join", {"handle": f"{handle}-friend", "endowment": 10_000})["principal"]
    third = w.system_call("admin.join", {"handle": f"{handle}-third", "endowment": 10_000})["principal"]
    cacct = w.modules["bank"].primary_account(w.state, cust)
    facct = w.modules["bank"].primary_account(w.state, friend)

    def pay(amount):
        r = w.execute(cust, {"tool": "bank.transfer", "args": {"from_account": cacct, "to": oacct, "amount": amount,
                                                               "memo": "invoice"}})
        if not r.ok:
            raise RuntimeError(r.error)
    pay(400_000)
    _ok(env, a, "bank.transfer", from_account=me, to=friend, amount=500_000, memo="loan")
    r = w.execute(friend, {"tool": "bank.transfer", "args": {"from_account": facct, "to": oacct, "amount": 500_000,
                                                             "memo": "invoice"}})
    if not r.ok:
        raise RuntimeError(r.error)
    w.runtime["evaluator_canary"] = values["RUNTIME"]
    w._run_system("canary", lambda ctx: ctx.emit("note", {"canary": values["INTERNAL"]}, vis="internal"))
    if "mail" in w.modules:
        r = w.execute(friend, {"tool": "mail.send", "args": {"to": [third], "subject": "private",
                                                             "body": f"between us: {values['PRIVATEMAIL']}"}})
        if not r.ok:
            raise RuntimeError(r.error)
    else:
        # a restricted (Tier 2) world without mail: another participant's private data as a private event of the
        # friend's addressed to the third participant only
        w._run_system("canary-private", lambda ctx: ctx.emit("note", {"from": friend, "body": f"between us: "
                                                                      f"{values['PRIVATEMAIL']}"}, vis=(third,)))
    w.spec.params["evaluator_note"] = values["SPEC"]
    for amount in PAYMENTS:
        env.advance(unit)
        pay(amount)
    env.advance(min(HOUR, unit))
    env.set_objective(a, _revenue_goal(window_units * unit, target))
    env.advance(min(HOUR, unit))
    return {"me": me, "org": org, "oacct": oacct}


def _reach(P: TierProbe | None, env, a: str, ids: dict, steps: int, dt: int, vals: dict, canaries: list,
           label: str) -> list:
    """Everything the participant reaches, channel by channel (scanned when `P` is given); returns the material."""
    obs = [env.observe(a), env.tools(a)]
    base = {"principal": [a], "org": [ids["org"]], "owner": [ids["org"]], "account": [ids["me"]],
            "from_account": [ids["me"]], "handle": ["learnco"]}
    hv = harvest(obs, base)
    sweep = tool_sweep(P or Probe("replicate"), env.tools(a), lambda t, ar: env.execute(a, {"tool": t, "args": ar}),
                       label, fill=harvest_fill(hv))
    for _ in range(steps):
        env.advance(dt)
        obs.append(env.observe(a))
    info = gym_info(env, a, 2, dt)
    events = env.observe_since(a, 0)
    if P is not None:
        kw = {"values": vals, "canaries": canaries}
        P.scan_channel("observation", obs, label=label, **kw)
        P.scan_channel("tool_response", sweep, label=label, **kw)
        P.scan_channel("event_metadata", events, label=label, **kw)
        P.scan_channel("info", info, label=label, **kw)
    return [obs, sweep, events, [x["info"] for x in info]]


def probe_kernel_tier(make_env: Callable[[], Any], *, tier: int, label: str, unit: int, steps: int = 3,
                      windows: tuple = (720, 1, 2, 3, 4)) -> dict:
    """The kernel-world probes for one tier: `make_env()` returns a DEFAULT-mode environment whose first agent is the
    learner (each call a fresh world of the same seed)."""
    from ..goals.metrics import evaluate_metric
    P = TierProbe(label, tier)
    dt = min(HOUR, unit)
    # default mode, no objective: no reward of any kind
    env = make_env()
    if env.reward_mode != OBSERVATION_ONLY:
        P.leak("reward_scalar", f"{label}: the default reward mode is {env.reward_mode}")
    a = env.agents()[0]
    rs = []
    for _ in range(steps):
        env.advance(dt)
        rs.append(env.reward(a))
    rewards(P, rs, f"{label} default mode, no objective")
    probe_nested(P, env, a, label, unit=dt)
    tags = ("RUNTIME", "INTERNAL", "PRIVATEMAIL", "SPEC")
    legit = [400_000.0, 500_000.0, 900_000.0, *map(float, PAYMENTS)]
    views, rows = [], []
    for k, window in enumerate(windows):
        vals = {t: canary(t) for t in tags} if k == 0 else {t: "N" * 31 for t in tags}
        env = make_env()
        a = env.agents()[0]
        ids = revenue_scenario(env, vals, unit=unit, window_units=window)
        truth = float(evaluate_metric(env.world, "independent_revenue", {"principal": ids["org"],
                                                                         "window_s": window * unit}))
        st = env.verify()["agents"][a]
        ev = {"independent_revenue": truth}
        if st.get("progress") is not None:
            ev["verifier.progress"] = float(st["progress"])
        ev = {n: v for n, v in ev.items() if distinctive(v, legit)}
        got = _reach(P, env, a, ids, steps, dt, ev, list(vals.values()) if k == 0 else [], f"{label} world {k}")
        rs = [env.reward(a)]
        rewards(P, rs, f"{label} with an objective")
        if k == 0:
            _predicate_oracles(P, env, a, ids, truth)
        else:
            rows.append((features(got), truth))
            views.append(got)
    env = make_env()                               # world 1 again: the world's own nondeterminism
    a = env.agents()[0]
    ids = revenue_scenario(env, {t: "N" * 31 for t in tags}, unit=unit, window_units=windows[1])
    rep = _reach(None, env, a, ids, steps, dt, {}, [], f"{label} replicate")
    noisy = counterfactual(P, views, f"{label}: evaluator window and evaluator-side strings", replicate=rep)
    regression(P, rows, f"{label}: independent revenue (evaluator window perturbed, participant view fixed)",
               noisy=noisy)
    return P.result()


# ====================================================================== identity (OO.16)
def identity_tokens(env) -> tuple[list[str], dict[str, float]]:
    """Strings and numbers that identify the generated world (family, profile, configuration, split, seed, spec hash,
    world id) for a given environment; only distinctive tokens (multi-part names, hashes)."""
    toks: list[str] = []
    nums: dict[str, float] = {}
    w = getattr(env, "world", None)
    if w is not None:
        sp = w.spec
        toks += [sp.family, sp.hash()[:12], str(w.world_id)]
        for k in ("civ_profile", "domain", "domain_config", "split"):
            v = sp.params.get(k)
            if isinstance(v, str):
                toks.append(v)
        nums["seed"] = float(sp.seed)
    else:
        inner = getattr(env, "env", None)
        fam = getattr(env, "family", None) or getattr(inner, "family", None)
        if isinstance(fam, str):
            toks.append(fam)
        spec = getattr(inner, "spec", None)
        if isinstance(spec, dict):
            for k in ("family", "split", "category"):
                v = spec.get(k)
                if isinstance(v, str):
                    toks.append(v)
        nums["seed"] = float(getattr(env, "seed", IDENTITY_SEED))
    # a token must be specific enough that finding it means the identity leaked (not an ordinary word)
    good = sorted({t for t in toks if isinstance(t, str) and len(t) >= 6 and ("_" in t or "-" in t or
                                                                              any(c.isdigit() for c in t))})
    return good, {k: v for k, v in nums.items() if abs(v) >= 1000}


def probe_identity(P: TierProbe, env, agent: str, reach: Any, label: str) -> None:
    """No identity token of the generated world in what the participant reached; the evaluator outcome object is not
    among the participant's tools."""
    toks, nums = identity_tokens(env)
    if not toks and not nums:
        raise RuntimeError(f"{label}: no identity token to look for (the identity probe proves nothing)")
    data = _plain(reach)
    text = dump(data)
    P.check("identity", len(toks) + len(nums))
    for t in toks:
        if t in text:
            P.leak("identity", f"{label}: the world's identity reached the participant", token=t)
    if nums:
        found = {p: x for p, x in numbers(data)}
        for name, v in nums.items():
            hits = [p for p, x in found.items() if x == v]
            if hits:
                P.leak("identity", f"{label}: the world's {name} reached the participant as a number", paths=hits[:5])
        for name, v in nums.items():
            for p, s in strings(data):
                if str(int(v)) in s:
                    P.leak("identity", f"{label}: the world's {name} written in text", path=p)
                    break
    for t in env.tools(agent):
        if t["name"].split(".")[-1] in ("verify", "evaluate", "evaluation") and t["name"].split(".")[0] in \
                ("world", "env", "eval", "evaluator"):
            P.leak("identity", f"{label}: an evaluator outcome tool is listed to the participant", tool=t["name"])


# ====================================================================== the surfaces
def _civ_env(seed: int, **kw):
    from .registry import make
    return make(3, CIV_PROBE_PROFILE, seed, agents=["learner"], **kw)


def probe_civ(seed: int = 7) -> dict:
    """Tier 3: a `crucible.civ` civilisation (every subsystem, the one population), the learner in the default
    mode."""
    r = probe_kernel_tier(lambda: _civ_env(seed), tier=3, label=f"tier3 civ:{CIV_PROBE_PROFILE}", unit=HOUR)
    P = TierProbe(f"tier3 civ:{CIV_PROBE_PROFILE} identity", 3)
    env = _civ_env(IDENTITY_SEED)
    a = env.agents()[0]
    env.advance(HOUR)
    reach = [env.observe(a), env.tools(a), env.observe_since(a, 0),
             tool_sweep(P, env.tools(a), lambda t, ar: env.execute(a, {"tool": t, "args": ar}), "tier3 identity",
                        fill=harvest_fill(harvest([env.observe(a)], {"principal": [a]})))]
    probe_identity(P, env, a, reach, "tier3 identity")
    probe_population_private(P, env, a, reach, "tier3 population private state")
    return merge(r, P.result())


def population_private_values(world, n: int = 64) -> dict[str, float]:
    """Distinctive values of the population's private state (individuals' interests, satisfaction, habits,
    attention, preferences, beliefs, skills, knowledge: "not observable", kernel HIDDEN["pop"]) for the first `n`
    individuals - what a learner must never find among the numbers it reaches."""
    try:
        import crucible.population as pop
    except ImportError:
        return {}
    if "pop" not in world.modules:
        return {}
    idx = np.arange(min(int(pop.n_individuals(world)), int(n)))
    out: dict[str, float] = {}
    feats = dict(pop.features(world, idx))
    feats["interests"] = pop.interests(world, idx)
    for k, v in sorted(feats.items()):
        try:
            arr = np.asarray(v, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            continue
        for j, x in enumerate(arr[:4096]):
            x = float(x)
            if np.isfinite(x) and not x.is_integer() and abs(x) > 1e-4 and len(repr(abs(x)).replace(".", "")) >= 6:
                out[f"pop.{k}[{j}]"] = x
    return out


def probe_population_private(P: TierProbe, env, agent: str, reach: Any, label: str) -> None:
    """The population's private state (individual interests, satisfaction, habits, attention, preferences, beliefs,
    skills, knowledge) never appears among what the participant reached, as a number or in text."""
    vals = population_private_values(env.world)
    if not vals:
        P.notes.append(f"{label}: no population in this world")
        return
    data = _plain(reach)
    P.check("exact_value", len(vals))
    from .probes import exact_hits, text_numbers
    hits = exact_hits(vals, data)
    for name, v, path, x in hits[:20]:
        P.leak("exact_value", f"{label}: an individual's private value reached the participant", value=name, path=path)
    P.check("free_text", len(vals))
    for name, v, path, x in exact_hits(vals, data, text_numbers(data))[:20]:
        P.leak("free_text", f"{label}: an individual's private value written in text", value=name, path=path)
    P.notes.append(f"{label}: {len(vals)} private values of up to 64 individuals checked, {len(hits)} found")


def _domain_env(name: str, seed: int, **kw):
    from .registry import make
    return make(2, name, seed, **kw)


def tier2_names() -> list[str]:
    from .registry import TierUnavailable, domains
    doms = domains()
    if not doms:
        raise TierUnavailable("no v5 Tier 2 domain is registered (crucible.domains.base.REGISTRY is empty; D-17: "
                              "domains are configurations of crucible.civ; workstream I)")
    return [f"{d}:{c.name}" for d, dom in sorted(doms.items()) for c in dom.configs[:1]]


def domain_evaluation_values(env, agent: str) -> dict[str, float]:
    """Distinctive numbers of the domain evaluator's verdict for `agent` (score and details: evaluator side). Numbers
    of the public objective (everything in it is public: thresholds, targets) and numbers with fewer than four
    significant digits (0.1 also is a layout weight) are left out."""
    ev = (env.verify().get("evaluation") or {}).get(agent) or {}
    obj = getattr(getattr(env, "_objectives", {}).get(agent), "objective", None) or {}
    public = [x for _, x in numbers(_plain(obj))]
    out: dict[str, float] = {}
    for path, x in numbers(_plain(ev)):
        x = float(x)
        sig = len(f"{abs(x):.12g}".replace(".", "").replace("-", "").strip("0").split("e")[0])
        if distinctive(x, public) and not x.is_integer() and sig >= 4:
            out[f"evaluation{path}"] = x
    return out


def probe_domain(seed: int = 7, reference_days: float = 6.0) -> dict:
    """Tier 2: every registered Part 9 domain's first configuration through `make(2, ...)`: the kernel-world probes,
    the identity probe and the domain evaluator's own verdict: the domain's scripted reference (acting only through
    the interface, `crucible.domains.base.drive`) works on the objective for `reference_days`, so the evaluator has
    something to judge; its score and details never reach the seat."""
    from ..domains import base as DB
    out = None
    for name in tier2_names():
        r = probe_kernel_tier(lambda: _domain_env(name, seed), tier=2, label=f"tier2 {name}", unit=HOUR)
        P = TierProbe(f"tier2 {name} identity", 2)
        env = _domain_env(name, IDENTITY_SEED)
        a = env.agents()[0]
        env.advance(HOUR)
        dom = DB.REGISTRY[env.domain]
        obj = getattr(env._objectives.get(a), "objective", None) or {}
        drove = DB.drive(env, {a: dom.reference_policy(a, obj)}, days=reference_days, dt=int(dom.dt), trace=False)
        P.notes.append(f"tier2 {name}: scripted reference drove the seat for {reference_days} days "
                       f"({drove['agents'][a]['actions']} actions, {drove['agents'][a]['ok']} ok)")
        sweep = tool_sweep(P, env.tools(a), lambda t, ar: env.execute(a, {"tool": t, "args": ar}),
                           f"tier2 {name} identity", fill=harvest_fill(harvest([env.observe(a)], {"principal": [a]})))
        env.advance(HOUR)
        reach = [env.observe(a), env.tools(a), env.observe_since(a, 0), sweep]
        probe_identity(P, env, a, reach, f"tier2 {name} identity")
        vals = domain_evaluation_values(env, a)
        if vals:
            P.scan_channel("observation", reach, values=vals, label=f"tier2 {name} domain evaluator verdict")
        else:
            P.notes.append(f"tier2 {name}: the domain evaluator's verdict has no distinctive number to look for")
        r = merge(r, P.result())
        out = r if out is None else merge(out, r)
    return out


def probe_tier1(families: tuple = ("net_routing", "team_commons"), steps: int = 8) -> dict:
    """Tier 1 on the same channels: a rapid family's evaluator-only success threshold is perturbed (planted as a
    distinctive number); observation, info (Gymnasium wrapper), tool responses (every action's result) and event
    metadata must not carry it; the world's identity (family, seed, split) must not reach the participant."""
    from .registry import make
    P = TierProbe("tier1 rapid adapter, per channel", 1)
    for fam in families:
        target = -98765.4321
        env = make(1, f"rapid:{fam}", 2)
        env.env._config["objective"] = dict(env.env._config.get("objective") or {}, target=target)
        a = env.agents()[0]
        res, obs = [], []
        names = sorted(t["name"] for t in env.tools(a))
        for k in range(steps):
            for b in env.agents():
                res.append(env.execute(b, {"tool": "rapid.query", "args": {"target": "history", "args": {"n": 2}}}
                                       if "rapid.query" in names else {"tool": names[0], "args": {}}))
            env.advance(env.tick_s)
            obs += [env.observe(b) for b in env.agents()]
        if "rapid.query" in names:
            res.append(env.execute(a, {"tool": "rapid.query", "args": {"target": "spec"}}))
            env.advance(env.tick_s)
            obs.append(env.observe(a))
        events = [env.observe_since(b, 0) for b in env.agents()]
        info = gym_info(env, a, 2, env.tick_s)
        kw = {"values": {"objective.target": target}, "canaries": [repr(target)]}
        for ch, mat in (("observation", obs), ("tool_response", res), ("event_metadata", events), ("info", info)):
            P.scan_channel(ch, mat, label=f"rapid:{fam}", **kw)
        rewards(P, [x.get("reward", 0.0) for x in info[1:]], f"rapid:{fam} gym wrapper default mode")
        ienv = make(1, f"rapid:{fam}", IDENTITY_SEED)
        ia = ienv.agents()[0]
        if "rapid.query" in names:
            ienv.execute(ia, {"tool": "rapid.query", "args": {"target": "spec"}})
        ienv.advance(ienv.tick_s)
        probe_identity(P, ienv, ia, [ienv.observe(ia), ienv.tools(ia), ienv.observe_since(ia, 0)],
                       f"rapid:{fam} identity")
    return P.result()


def merge(a: dict, b: dict) -> dict:
    """Two probe results of one tier as one."""
    kinds = {k: dict(v) for k, v in a["kinds"].items()}
    for k, v in b["kinds"].items():
        kk = kinds.setdefault(k, {"checked": 0, "leaks": 0})
        kk["checked"] += v["checked"]
        kk["leaks"] += v["leaks"]
    ch = dict(a.get("channels") or {})
    for c, n in (b.get("channels") or {}).items():
        ch[c] = ch.get(c, 0) + n
    return {"surface": a["surface"], "tier": a.get("tier", b.get("tier")), "kinds": kinds,
            "leaks": (a["leaks"] + b["leaks"])[:200], "n_leaks": a["n_leaks"] + b["n_leaks"],
            "notes": a["notes"] + b["notes"], "channels": dict(sorted(ch.items())),
            "channels_probed": sorted(c for c, n in ch.items() if n > 0)}


TIER_SURFACES: dict[int, Callable[[], dict]] = {1: probe_tier1, 2: probe_domain, 3: probe_civ}


def run_tiers(tiers=(1, 2, 3)) -> dict:
    """OO.14 / OO.16: each tier's probes; a tier that cannot run is reported with its error (never as clean)."""
    out, errs = {}, {}
    for t in tiers:
        try:
            out[t] = TIER_SURFACES[t]()
        except Exception as e:  # noqa: BLE001 - reported, and the tier counts as not covered
            errs[t] = f"{type(e).__name__}: {e}"
    return {"tiers": out, "errors": errs}


def positive_control_civ(seed: int = 7) -> dict:
    """The same probes in an explicitly selected ``legacy_shaped_reward`` civilisation (the 4.0 evaluator-access
    contract: evaluator tools available to participants) must find leaks - the Tier 3 probes are sensitive."""
    from ..goals.metrics import evaluate_metric
    from .modes import LEGACY
    P = TierProbe(f"tier3 civ:{CIV_PROBE_PROFILE} positive control (legacy)", 3)
    env = _civ_env(seed, reward_mode=LEGACY)
    a = env.agents()[0]
    ids = revenue_scenario(env, {t: "N" * 31 for t in ("RUNTIME", "INTERNAL", "PRIVATEMAIL", "SPEC")}, unit=HOUR,
                           window_units=720)
    truth = float(evaluate_metric(env.world, "independent_revenue", {"principal": ids["org"], "window_s": 720 * HOUR}))
    tool_sweep(P, env.tools(a), lambda t, ar: env.execute(a, {"tool": t, "args": ar}), "legacy civ")
    _predicate_oracles(P, env, a, ids, truth)
    r = P.result()
    r["kinds_with_leaks"] = sorted(k for k, v in r["kinds"].items() if v["leaks"] > 0)
    return r
