"""Probes against generated civilisations and Tier 2 domain worlds (D-17: every v5 Tier 2 and Tier 3 world is a
configuration of `crucible.civ.generate`; Gate W.2, W.4, W.5, W.8, W.9).

The substrate, population, commerce and network probes of the other modules run in small purpose-built worlds; these
re-run the decisive ones inside real civilisations, where every subsystem, coupling and the unified population are
present at once (the configuration a learner is evaluated in):

* the world process stays offline while a civilisation runs and an agent asks for hostile URLs;
* no read-only tool reveals an individual's private traits (the unified population's private columns);
* no tool writes a measured quantity, among every tool of a civilisation;
* nothing passes between two civilisations in one process (artifact pool, existence oracle);
* Tier 2 domain worlds (`crucible.domains`): one probe per registered domain configuration of the evaluation
  splits - the world stays offline under the domain's reference policy and the evaluator's measurements of that run
  are not readable through any participant tool. The catalogue grows with the registry (empty today: no domain
  is registered on the integrated tree yet).

A civilisation that cannot be generated makes these probes ERROR (never not-applicable), so Gate W fails until the
generator works.
"""
from __future__ import annotations

from ..framework import HELD, LEAK, Attacker, Evaluator, NotApplicable, Operator, Verdict
from ..leakage import distinctive
from .base import V5Probe, scan_hits, sweep_read_only
from .commerce import NoMetricWriteTools
from .crossworld import ArtifactsDoNotCrossWorlds
from .network import HOSTILE, _Guarded
from .population import PopulationPrivateColumns
from .private import SocialPrivateState
from .social import AudienceCounterWrites

PROFILE = "balanced"              # the default civilisation (every subsystem of the generator)
OTHER = "digital_first"           # a structurally different one, for the cross-world probe


def _civ(profile: str = PROFILE) -> dict:
    return {"v5": "civ", "profile": profile, "seed": 3}


class CivWorldOffline(_Guarded):
    name = "v5.civ.world_is_offline"
    attack_class = "network/egress"
    world = _civ()
    target, owner = "civ", "J"
    slow = True
    description = ("Under a guard recording every resolve/connect/sendto of the world process, a generated "
                   "civilisation (profile balanced, development population) runs 6 simulated hours while an agent "
                   "asks `net.fetch` for an external site, the cloud metadata address and the instance's management "
                   "API. Checked: no network attempt at all.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("civnet", 10_000)

    def attack(self, a: Attacker) -> None:
        def go():
            self.calls = []
            names = {t["name"] for t in a.tools(self.att)}
            for url in HOSTILE:
                if "net.fetch" in names:
                    r = a.act(self.att, "net.fetch", url=url)
                    self.calls.append([url, r.ok, (r.error or {}).get("code")])
            a.wait(6 * 3600)
        self.run_guarded(go)


class CivPopulationPrivateTraits(PopulationPrivateColumns):
    name = "v5.civ.population_private_traits_not_readable"
    world = _civ()
    target, owner = "civ population", "B"
    slow = True
    description = ("In a generated civilisation (the unified population on every platform), canaries are taken from "
                   "the private columns of the first 60 individuals; an agent sweeps every read-only tool of the "
                   "civilisation with every identifier it can find (population principals included) and reads its "
                   "observation stream. Checked: no canary appears.")


class CivNoMetricWriteTools(NoMetricWriteTools):
    name = "v5.civ.no_tool_writes_a_measured_quantity"
    world = _civ()
    target, owner = "civ", "J"
    slow = True
    description = ("An agent lists every tool it can call in a generated civilisation and tries each one whose name "
                   "suggests a direct write of a measured quantity (set/adjust/override of revenue, MRR, rating, "
                   "views, followers, adoption, quality, score, balance). Checked: none exists or each is refused.")


class CivArtifactsDoNotCross(ArtifactsDoNotCrossWorlds):
    name = "v5.civ.artifacts_do_not_cross_civilisations"
    world = _civ()
    target, owner = "civ artifacts", "A"
    slow = True
    description = ("A victim in a second civilisation of another profile in the same process stores private bytes; "
                   "the attacker in this civilisation reads them by REF, asks for their info and stores the same and "
                   "fresh bytes looking for an existence oracle. Checked: nothing distinguishes them from bytes "
                   "stored nowhere.")

    def setup(self, op: Operator) -> None:
        from ... import civ
        from .base import b64
        self.att = op.join("xciv-attacker", 1000)
        self.other = civ.generate(OTHER, 11, population="development", world_id="xciv-victim-world")
        va = self.other.system_call("admin.join", {"handle": "victim", "endowment": 1000})["principal"]
        self.secret = b"civilisation V private: launch code 4417-2291-XQ"
        self.ref = self.other.system_call("artifact.put", {"data_b64": b64(self.secret), "mime": "text/plain"},
                                          as_principal=va)["ref"]


class CivSocialPrivateState(SocialPrivateState):
    name = "v5.civ.social_private_state_not_readable"
    world = _civ()
    target, owner = "civ social", "C"
    slow = True
    description = ("In a generated civilisation (every platform, the unified population, commerce), a victim sends a "
                   "direct message, publishes a members-only post behind a paid tier, keeps a draft and posts in a "
                   "private community; an unrelated agent sweeps every read-only tool of the civilisation with every "
                   "identifier it finds, guesses conversation ids and reads its observation stream. Checked: none "
                   "of the four secrets reaches it.")


class CivAudienceCounterWrites(AudienceCounterWrites):
    name = "v5.civ.audience_counters_cannot_be_written"
    world = _civ()
    target, owner = "civ audience", "C"
    slow = True
    description = ("In a generated civilisation (the unified population behind the audience accounting), an agent "
                   "publishes a post and tries every route to its view counters: a raw column batch, consumption and "
                   "watch tools as a non-population principal, extra `views` arguments, switching the platform's "
                   "recommender mixture. Checked: all are refused or have no effect, and the exact recount agrees.")


# ---------------------------------------------------------------------------------------------- Tier 2 domains
# The domain framework (`crucible.domains.base`, domains-framework/2): a learner reaches a configuration through
# `make_env(domain, config, seed)` (observation_only), occupies a seat, reads the public objective; the evaluator's
# outcome is `Domain.evaluate` / `env.verify()`; the scripted reference acts only through the learner interface.
class _EnvParticipant:
    """The red team's sweep (`base.sweep_read_only`) over a learner environment: the seat's tools, actions and
    observation stream - exactly what a learner occupying the seat receives."""

    def __init__(self, env):
        self.env = env

    def tools(self, actor):
        return list(self.env.observe(actor).get("tools") or [])

    def observe(self, actor):
        o = dict(self.env.observe(actor))
        o["events"] = [{"type": str(e.get("type", "")), "payload": e} if isinstance(e, dict) else
                       {"type": "event", "payload": e} for e in (o.get("events") or [])]
        return o

    def act(self, actor, tool, /, **args):
        r = self.env.execute(actor, {"tool": tool, "args": args})
        return _Result(bool(r.get("ok")), r.get("result"), r.get("error"))


class _Result:
    def __init__(self, ok, result, error):
        self.ok, self.result, self.error = ok, result, error


class _Tier2(V5Probe):
    target, owner = "domains", "I"
    slow = True
    world = {"v5": "substrate"}       # the attack runs in the learner's own environment of the configuration

    def __init__(self, domain: str, config: str):
        self.domain, self.config = domain, config

    def setup(self, op: Operator) -> None:
        self.att = op.join("t2-rt", 10)

    def _env(self):
        from ...domains import REGISTRY, make_env
        d = REGISTRY[self.domain]
        diff = int(getattr(d, "reference_difficulty", 1))
        env = make_env(self.domain, self.config, 0, difficulty=diff)
        me = env.agents()[0]
        return d, env, me, d.objective(env.world, me, diff)

    def _reference_day(self, d, env, me, obj, policy=None, days: float = 1.0) -> dict:
        from ...domains import drive
        policy = policy if policy is not None else d.reference_policy(me, obj)
        return drive(env, {me: policy}, days=days, dt=int(getattr(d, "dt", 3600)), trace=False)


class Tier2OfflineAndIsolated(_Tier2):
    attack_class = "evaluator_independence/hidden_answer"
    gates = ("W.2", "W.8")
    EXTEND_S = 90.0              # wall-clock budget for driving beyond the first day

    def __init__(self, domain: str, config: str):
        super().__init__(domain, config)
        self.name = f"v5.tier2.{domain}.{config}.offline_and_evaluator_isolated"
        self.description = (f"Tier 2 domain {domain}, evaluation configuration {config}, built through the learner's own "
                            "path: under the network guard the domain's scripted reference acts for a simulated day "
                            "(and on, in chunks within the horizon and a time budget, until the evaluator's outcome "
                            "has distinctive numbers); the evaluator's measurements of the seat (Domain.evaluate and "
                            "the environment's verify()) are then taken as canaries and the seat sweeps every read-only tool it has "
                            "with every identifier it finds and reads its observation stream. Checked: no network "
                            "attempt at all, and no evaluator value reaches the seat.")

    def attack(self, a: Attacker) -> None:
        from ...realworld import NetworkGuard
        self.canaries, self.received, self.drive = {}, [], {}
        with NetworkGuard() as g:
            try:
                d, env, me, obj = self._env()
                try:
                    import time as _time
                    policy = d.reference_policy(me, obj)
                    self.drive = self._reference_day(d, env, me, obj, policy)
                    self.days = 1.0

                    def walk(x, path):
                        if isinstance(x, dict):
                            for k, v in x.items():
                                walk(v, f"{path}.{k}")
                        elif isinstance(x, (list, tuple)):
                            for i, v in enumerate(x[:50]):
                                walk(v, f"{path}[{i}]")
                        elif isinstance(x, (int, float)) and not isinstance(x, bool):
                            dd = distinctive(x)
                            if dd:
                                self.canaries.setdefault(dd, path)

                    def measure():
                        # the evaluator's own measurements only (verify()'s bookkeeping - event count, digest - is
                        # public)
                        walk({"evaluate": d.evaluate(env.world, me, obj),
                              "verify_evaluation": (env.verify().get("evaluation") or {})}, "evaluator")
                    measure()
                    # one simulated day often leaves the evaluator's outcome without a distinctive number (the
                    # isolation half would be vacuous): the same reference keeps acting, in chunks of up to five
                    # days within the domain's horizon, until the outcome has distinctive values or the time
                    # budget is spent
                    horizon = float(getattr(d, "horizon_days", 30.0) or 30.0)
                    t0 = _time.monotonic()
                    while not self.canaries and self.days < horizon and _time.monotonic() - t0 < self.EXTEND_S:
                        chunk = min(5.0, horizon - self.days)
                        more = self._reference_day(d, env, me, obj, policy, days=chunk)
                        for a, x in (more.get("agents") or {}).items():
                            y = self.drive.setdefault("agents", {}).setdefault(a, {"actions": 0})
                            y["actions"] = int(y.get("actions") or 0) + int(x.get("actions") or 0)
                        self.days += chunk
                        measure()
                    # the seat's own funds are its own knowledge (its accounts' balances, read through ordinary bank
                    # tools): an evaluator detail that equals one of them is not a leak of the evaluator's
                    # measurement (e.g. a cash target checked against the seat's own balance)
                    self.own = self._own_funds(env, me)
                    self.self_observable = {k: v for k, v in self.canaries.items() if k in self.own}
                    for k in self.self_observable:
                        self.canaries.pop(k, None)
                    self.received = sweep_read_only(_EnvParticipant(env), me, rounds=1, max_calls=3000)
                finally:
                    env.close()
            finally:
                self.attempts = list(g.attempts)

    @staticmethod
    def _own_funds(env, me) -> set:
        """Distinctive renderings (cents and dollars) of the balances of the accounts the seat itself can read."""
        out = set()
        try:
            accts = env.world.query(me, "bank.accounts", {})
        except Exception:  # noqa: BLE001 - a world without the bank's account listing: nothing to exclude
            return out
        rows = accts.get("accounts", accts) if isinstance(accts, dict) else accts
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            for k in ("balance", "balance_cents", "available"):
                v = r.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    for x in (v, v / 100.0, int(round(v))):
                        dd = distinctive(x)
                        if dd:
                            out.add(dd)
        return out

    def verdict(self, ev: Evaluator) -> Verdict:
        hits = scan_hits(self.received, self.canaries)
        acted = sum(int(x.get("actions") or 0) for x in (self.drive.get("agents") or {}).values())
        evd = {"network_attempts": self.attempts[:20], "network_escapes": len(self.attempts), "reference_actions": acted,
               "canaries": len(self.canaries), "hits": dict(list(hits.items())[:10]), "evaluator_leaks": len(hits),
               "texts_examined": len(self.received), "simulated_days": getattr(self, "days", None),
               "isolation_vacuous": not self.canaries,
               "sweep_calls": next((int(t) for w, t in self.received if w == "calls"), 0),
               "sweep_capped": any(w == "capped" and t == "True" for w, t in self.received),
               "self_observable_excluded": dict(list(getattr(self, "self_observable", {}).items())[:10])}
        if self.attempts:
            from ..framework import INTEGRITY_BREAK
            return Verdict(INTEGRITY_BREAK, f"a Tier 2 world tried to reach the network: {self.attempts[:3]}", evd,
                           severity="critical")
        if hits:
            return Verdict(LEAK, f"{len(hits)} evaluator measurement(s) of the domain reached the participant", evd,
                           severity="high")
        if not self.canaries:
            # offline (W.8) is measured; evaluator isolation (W.2) is not: gate_metrics does not count this row as a
            # W.2 probe run and reports it as evaluator_isolation_vacuous (wave-3b verifier)
            return Verdict(HELD, f"offline ({acted} reference actions in {self.days:g} days); the evaluator's outcome "
                                 "held no distinctive value to look for within the time budget (the isolation half "
                                 "is vacuous here and is not counted for W.2)", evd)
        return Verdict(HELD, f"offline ({acted} reference actions in {self.days:g} days); none of {len(self.canaries)} "
                             "evaluator values appeared in any participant tool", evd)


class Tier2NoMetricWriteTools(NoMetricWriteTools):
    def __init__(self, domain: str, config: str):
        self.domain, self.config = domain, config
        self.name = f"v5.tier2.{domain}.{config}.no_tool_writes_a_measured_quantity"
        self.world = {"v5": "tier2", "domain": domain, "config": config}
        self.target, self.owner = "domains", "I"
        self.slow = True
        self.description = (f"Tier 2 domain {domain}, evaluation configuration {config}: an agent lists every tool of the "
                            "configuration's world and tries each one whose name suggests a direct write of a "
                            "measured quantity. Checked: none exists or each is refused.")


class Tier2DomainDoesNotLoad(V5Probe):
    """A domain module that cannot be imported (or registered) is reported as an error of the Gate W categories its
    probes would measure - never silently left out of the catalogue."""
    attack_class = "network/egress"
    gates = ("W.2", "W.8")
    target, owner = "domains", "I"
    world = {"v5": "substrate"}

    def __init__(self, module: str, error: str):
        self.module, self.error = module, error
        self.name = f"v5.tier2.{module}.does_not_load"
        self.description = f"Tier 2 domain module {module} could not be loaded, so its configurations cannot be probed."

    def setup(self, op: Operator) -> None:
        raise RuntimeError(f"domain module crucible.domains.{self.module} does not load: {self.error}")


def _load_domains() -> tuple[dict, list]:
    """Import every domain module of `crucible.domains` one by one (each registers itself); returns (registry,
    [(module, error)] of the modules that failed)."""
    import importlib
    import pkgutil
    from ... import domains
    skip = set(getattr(domains, "_FRAMEWORK", ())) | {"base"}
    failed = []
    for info in pkgutil.iter_modules(domains.__path__):
        if info.name in skip or info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{domains.__name__}.{info.name}")
        except Exception as e:  # noqa: BLE001 - reported as an erroring probe, never dropped
            failed.append((info.name, f"{type(e).__name__}: {str(e)[:300]}"))
    return dict(domains.REGISTRY), failed


def tier2_probes() -> list:
    """Per registered Tier 2 domain (test fixtures excluded): an offline + evaluator-isolation probe for each
    configuration of the evaluation splits (the configurations a learner is evaluated in) and a metric-write probe
    on the domain's tools; an erroring probe per domain module that does not load."""
    reg, failed = _load_domains()
    out = [Tier2DomainDoesNotLoad(m, e) for m, e in failed]
    for name, d in sorted(reg.items()):
        if getattr(d, "fixture", False):
            continue
        evals = [c for c in d.configs if c.split != "train"]
        out += [Tier2OfflineAndIsolated(name, c.name) for c in evals]
        if evals:                                       # the tool set is the domain's: one configuration suffices
            out.append(Tier2NoMetricWriteTools(name, evals[0].name))
    return out


def probes() -> list:
    return [CivWorldOffline(), CivPopulationPrivateTraits(), CivNoMetricWriteTools(), CivArtifactsDoNotCross(),
            CivSocialPrivateState(), CivAudienceCounterWrites()] + tier2_probes()
