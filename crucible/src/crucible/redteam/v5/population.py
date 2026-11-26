"""Probes against the autonomous population (workstream B; Gate W.3, W.4, W.9): system-only population tools used by
an agent, and the private columns of individuals (preferences, interests, beliefs, knowledge, budgets) read through
any tool an agent can call."""
from __future__ import annotations

import numpy as np

from ..framework import FALSE_SUCCESS, HELD, LEAK, Attacker, Evaluator, Operator, Verdict
from ..leakage import distinctive
from .base import V5Probe, scan_hits, sweep_read_only

PRIVATE_COLS = ("prefs", "interests", "beliefs", "knowledge", "skills", "goals", "wtp_scale", "wallet_cents",
                "income_cents", "habit", "satisfaction", "experience", "last_utility")


class PopulationSystemTools(V5Probe):
    name = "v5.population.system_tools_refused_to_agents"
    attack_class = "metric_manipulation/population_control"
    gates = ("W.3", "W.4")
    target, owner = "population", "B"
    world = {"v5": "population", "configs": {"_population": {"adaptive": True, "strategic": True}}}
    description = ("An agent calls the population's system tools: freezing a copy of the adaptive or strategic "
                   "learners, switching their learning off (which would freeze competitors' adaptation), and reading "
                   "other sites' visit analytics. Checked: every one is refused.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("popctl", 1_000)
        op.advance(3600)

    def attack(self, a: Attacker) -> None:
        calls = [("pop_adaptive.freeze_copy", {"group": "g2", "from_group": "adaptive"}),
                 ("pop_adaptive.set_learning", {"group": "adaptive", "learning": False}),
                 ("pop_strategic.freeze_copy", {"group": "g2", "from_group": "strategic"})]
        hosts = sorted(h for h, _ in a._w.state.scan("hosts"))[:3] if "hosts" in a._w.state.tables else []
        calls += [("popweb.visits", {"host": h}) for h in hosts]
        self.res = {f"{t} {args}": a.act(self.att, t, **args) for t, args in calls}

    def verdict(self, ev: Evaluator) -> Verdict:
        ok = sorted(k for k, r in self.res.items() if r.ok)
        evd = {"attempts": {k: [r.ok, (r.error or {}).get("code")] for k, r in self.res.items()}}
        if ok:
            return Verdict(FALSE_SUCCESS, f"an agent used system-only population tools: {ok}", evd, severity="high")
        return Verdict(HELD, f"all {len(self.res)} system-only population calls were refused", evd)


class PopulationPrivateColumns(V5Probe):
    name = "v5.population.private_traits_not_readable"
    attack_class = "information_leakage/population_private_state"
    gates = ("W.9", "W.2")
    target, owner = "population", "B"
    world = {"v5": "population"}
    description = ("Canaries are taken from the private columns of the first 60 individuals (preferences, "
                   "interests, beliefs, knowledge, skills, goals, willingness to pay, wallet, income, habits, "
                   "satisfaction). An agent sweeps every read-only tool with every identifier it can find, "
                   "including population principals p<i>, and reads its observation stream. Checked: no canary "
                   "appears.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("popscan", 10_000)
        op.advance(4 * 3600)
        from ...substrate import columns
        w = op.world
        self.canaries = {}
        for c in PRIVATE_COLS:
            try:
                arr = np.asarray(columns.col(w, "pop", c))
            except Exception:  # noqa: BLE001 - column not in this profile
                continue
            for i, row in enumerate(arr[:60]):
                for v in np.asarray(row).reshape(-1)[:8].tolist():
                    d = distinctive(float(v) if isinstance(v, float) else int(v))
                    if d:
                        self.canaries.setdefault(d, f"pop.{c}[{i}]")

    def attack(self, a: Attacker) -> None:
        self.received = sweep_read_only(a, self.att, seeds=[f"p{i}" for i in range(20)], rounds=1, max_calls=2500)

    def verdict(self, ev: Evaluator) -> Verdict:
        hits = scan_hits(self.received, self.canaries)
        evd = {"canaries": len(self.canaries), "texts": len(self.received),
               "calls": next((int(t) for w, t in self.received if w == "calls"), 0),
               "bytes": sum(len(t) for _, t in self.received), "hits": dict(list(hits.items())[:10]),
               "private_state_leaks": len(hits), "evaluator_leaks": 0}
        if len(self.canaries) < 50:
            from ..framework import NotApplicable
            raise NotApplicable(f"only {len(self.canaries)} distinctive canaries in this population")
        if hits:
            return Verdict(LEAK, f"{len(hits)} private trait value(s) of individuals reached an agent", evd,
                           severity="high")
        return Verdict(HELD, f"{evd['calls']} calls, {evd['bytes']:,} bytes: none of {len(self.canaries)} private-trait "
                             "canaries appeared", evd)


def probes() -> list:
    return [PopulationSystemTools(), PopulationPrivateColumns()]
