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

from ..framework import HELD, LEAK, Attacker, Evaluator, Operator, Verdict
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
class _Tier2(V5Probe):
    target, owner = "domains", "I"
    slow = True

    def __init__(self, domain: str, config: str):
        self.domain, self.config = domain, config
        self.world = {"v5": "tier2", "domain": domain, "config": config}


class Tier2DomainOffline(_Tier2):
    attack_class = "network/egress"
    gates = ("W.8",)

    def __init__(self, domain: str, config: str):
        super().__init__(domain, config)
        self.name = f"v5.tier2.{domain}.{config}.world_is_offline"
        self.description = (f"Tier 2 domain {domain}, evaluation configuration {config}: the domain's reference policy "
                            "acts for one simulated day under the network guard. Checked: no network attempt.")

    def setup(self, op: Operator) -> None:
        from ...domains import REGISTRY
        d = REGISTRY[self.domain]
        self.principal = d.participant_principal(op.world) if d.participant_principal else op.join("t2net", 10_000)
        op.control(self.principal)

    def attack(self, a: Attacker) -> None:
        from ...domains import REGISTRY
        from ...realworld import NetworkGuard
        d = REGISTRY[self.domain]
        with NetworkGuard() as g:
            try:
                obj = d.objective(a._w, self.principal, 0)
                pol = d.reference(a._w, self.principal, obj)
                if callable(pol):
                    for _ in range(24):
                        pol(a._w, self.principal)
                        a.wait(3600)
                else:
                    a.wait(24 * 3600)
            finally:
                self.attempts = list(g.attempts)

    def verdict(self, ev: Evaluator) -> Verdict:
        evd = {"network_attempts": self.attempts[:20], "network_escapes": len(self.attempts)}
        if self.attempts:
            from ..framework import INTEGRITY_BREAK
            return Verdict(INTEGRITY_BREAK, f"a Tier 2 world tried to reach the network: {self.attempts[:3]}", evd,
                           severity="critical")
        return Verdict(HELD, "no name resolution, connection or datagram left the world process", evd)


class Tier2EvaluatorNotReadable(_Tier2):
    attack_class = "evaluator_independence/hidden_answer"
    gates = ("W.2",)

    def __init__(self, domain: str, config: str):
        super().__init__(domain, config)
        self.name = f"v5.tier2.{domain}.{config}.evaluator_measurements_not_readable"
        self.description = (f"Tier 2 domain {domain}, evaluation configuration {config}: after a simulated day the "
                            "evaluator's measurements of the participant (domain evaluate()) are taken as canaries; "
                            "the participant sweeps every read-only tool. Checked: no evaluator value appears.")

    def setup(self, op: Operator) -> None:
        from ...domains import REGISTRY
        d = REGISTRY[self.domain]
        w = op.world
        self.principal = d.participant_principal(w) if d.participant_principal else op.join("t2eval", 10_000)
        op.control(self.principal)
        obj = d.objective(w, self.principal, 0)
        op.advance(24 * 3600)
        res = d.evaluate(w, self.principal, obj)
        self.canaries = {}

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
        walk(res, "evaluate")

    def attack(self, a: Attacker) -> None:
        self.received = sweep_read_only(a, self.principal, rounds=1, max_calls=2000)

    def verdict(self, ev: Evaluator) -> Verdict:
        hits = scan_hits(self.received, self.canaries)
        evd = {"canaries": len(self.canaries), "hits": dict(list(hits.items())[:10]), "evaluator_leaks": len(hits)}
        if hits:
            return Verdict(LEAK, f"{len(hits)} evaluator measurement(s) of the domain reached the participant", evd,
                           severity="high")
        return Verdict(HELD, f"none of {len(self.canaries)} evaluator values appeared in any participant tool", evd)


def tier2_probes() -> list:
    """One offline and one evaluator-isolation probe per registered Tier 2 domain configuration of the evaluation
    splits (the configurations a learner is evaluated in)."""
    from ... import domains
    out = []
    for name, d in sorted(domains.load_all().items()):
        for c in d.configs:
            if c.split != "train":
                out += [Tier2DomainOffline(name, c.name), Tier2EvaluatorNotReadable(name, c.name)]
    return out


def probes() -> list:
    return [CivWorldOffline(), CivPopulationPrivateTraits(), CivNoMetricWriteTools(), CivArtifactsDoNotCross(),
            CivSocialPrivateState(), CivAudienceCounterWrites()] + tier2_probes()
