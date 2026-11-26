"""Estimated large-population mode and its validation against individual simulation (MASTER_SPEC 13.6, Gate V item 9,
SP-13.6).

`CohortModel` is a statistical periphery for routine population activity: individuals are grouped into cohorts
(participant class x segment) and each cohort's routine dynamics are estimated from a high-fidelity individual run of
the one population (crucible.population): a two-state daily activity chain (active / inactive, which gives retention),
sessions per active day, a first-purchase hazard (adoption), purchases and spend per active day, and how visits split
over web resources (by resource kind, with the within-kind concentration over resource ranks). `predict` then rolls
the cohorts forward with aggregate binomial / Poisson transitions for any population size and resource mix.

Everything it produces is an ESTIMATE: returned in a dict labelled {"estimated": True, "mode": MODE_ID, "fitted_on"},
never written into a world. It records no event, moves no money and bumps no counter, so low-resolution participants
can never become a source of fabricated outcomes (13.6); distinct-viewer claims stay with the individual simulation.

`validate()` measures the error of the estimate against independent high-fidelity runs at tractable scales (fit on
one size and seed, predict a larger population on other seeds, and a resource-mix intervention) for the seven
quantities of 13.6: adoption trajectories, audience distributions, revenue, network diffusion (resource reach
curves), retention, intervention effects and rare-event frequencies; with intervals over seeds.

Limits (documented and measured by `validate`): cohorts are exchangeable inside (no individual memory, no contact
graph, so word-of-mouth cascades and social diffusion are captured only through fitted reach curves); resource
attractiveness is a per-kind constant fitted on the control mix; feedback from the learner or from prices changing
during the run is not modelled; the fit is only valid inside the configuration range it was fitted on.
"""
from __future__ import annotations

import math
import time

import numpy as np

MODE_ID = "approx-cohort/1"
DAY = 86400
RARE_SESSIONS = 4          # an individual-day with at least this many sessions is a rare event
RARE_AUDIENCE = 0.05       # a resource reaching at least this share of the population is a rare event
QUANTITIES = ("adoption_trajectory", "audience_distribution", "revenue", "network_diffusion", "retention",
              "intervention_effect", "rare_event_frequency")


# ====================================================================== high-fidelity panel
def snapshot(world) -> dict:
    """Day-boundary observation of the individual simulation (columns of `pop` and of the web resources)."""
    from ..substrate import columns
    tabs = columns.tables(world)
    c = lambda t, n: columns.col(world, t, n).copy()   # noqa: E731
    out = {"t": int(world.now), "class": c("pop", "class"), "segment": c("pop", "segment"),
           "n_sessions": c("pop", "n_sessions").astype(np.int64),
           "n_purchases": c("pop", "n_purchases").astype(np.int64) if "n_purchases" in tabs["pop"]["columns"]
           else np.zeros(columns.rows(world, "pop"), np.int64),
           "spent_cents": c("pop", "spent_cents").astype(np.int64) if "spent_cents" in tabs["pop"]["columns"]
           else np.zeros(columns.rows(world, "pop"), np.int64)}
    if "res" in tabs and columns.rows(world, "res"):
        for k in ("kind", "visits", "uniq", "purchases", "revenue_cents"):
            out[f"res_{k}"] = c("res", k).astype(np.int64)
    else:
        for k in ("kind", "visits", "uniq", "purchases", "revenue_cents"):
            out[f"res_{k}"] = np.zeros(0, np.int64)
    return out


def hifi_run(n: int, seed: int, days: int, *, shop_share: float = 0.4, sites: int = 24, profile: str = "development",
             tick_s: int = 900) -> dict:
    """A high-fidelity individual run of the one population: `n` individuals, daily snapshots."""
    import crucible
    crucible.load_all_modules()
    from ..core.kernel import World
    from ..population.worlds import population_world
    t0 = time.time()
    w = World.create(population_world(profile, seed, n=n, sites=sites, shop_share=shop_share, tick_s=tick_s))
    snaps = [snapshot(w)]
    for d in range(days):
        w.advance_time(w.now + DAY)
        snaps.append(snapshot(w))
    events = int(w.seq)
    digest = w.digest()
    w.close()
    return {"n": int(n), "seed": int(seed), "days": int(days), "shop_share": shop_share, "sites": sites,
            "snaps": snaps, "events": events, "digest": digest, "wall_s": round(time.time() - t0, 2)}


def outcomes(run: dict) -> dict:
    """The seven observed quantities of a high-fidelity run (from its snapshots)."""
    s = run["snaps"]
    N = int(s[0]["class"].size)
    D = len(s) - 1
    sess = np.stack([s[d + 1]["n_sessions"] - s[d]["n_sessions"] for d in range(D)])        # [D, N]
    active = sess > 0
    adopted = np.stack([s[d + 1]["n_purchases"] > 0 for d in range(D)])
    rev = np.array([int(s[d + 1]["res_revenue_cents"].sum() - s[d]["res_revenue_cents"].sum()) for d in range(D)])
    end = s[-1]
    uniq = end["res_uniq"].astype(np.float64)
    reach = np.stack([s[d + 1]["res_uniq"] for d in range(D)]).astype(np.float64) / max(N, 1)     # [D, R]
    ret = [float((active[d] & active[d + 1]).sum() / max(1, active[d].sum())) for d in range(D - 1)]
    return {"N": N, "days": D,
            "adoption": [float(a.mean()) for a in adopted],
            "audience_shares": sorted((uniq / max(N, 1)).tolist(), reverse=True),
            "revenue_per_day": rev.astype(float).tolist(), "revenue_total": float(rev.sum()),
            "reach_top": reach[:, np.argsort(-uniq)[:5]].mean(1).tolist() if uniq.size else [0.0] * D,
            "retention": float(np.mean(ret)) if ret else 0.0,
            "rare_individual_days": float((sess >= RARE_SESSIONS).mean()),
            "rare_resources": float((uniq / max(N, 1) >= RARE_AUDIENCE).mean()) if uniq.size else 0.0,
            "active_share": float(active.mean()), "sessions_per_active": float(sess[active].mean()) if active.any() else 0.0}


# ====================================================================== the estimated mode
class CohortModel:
    """Cohort-level routine dynamics fitted on high-fidelity runs (see the module docstring)."""

    def __init__(self):
        self.params: dict = {}
        self.fitted_on: list = []

    @staticmethod
    def _cohort(cls, seg):
        return np.asarray(cls, np.int64) * 100 + np.asarray(seg, np.int64)

    def fit(self, runs: list[dict]) -> "CohortModel":
        stats: dict = {}
        res_kind_visits, res_kind_count, conc = {}, {}, {}
        rev_per_purchase, purchases_per_visit_shop = [], []
        for run in runs:
            s = run["snaps"]
            D = len(s) - 1
            coh = self._cohort(s[0]["class"], s[0]["segment"])
            sess = np.stack([s[d + 1]["n_sessions"] - s[d]["n_sessions"] for d in range(D)])
            act = sess > 0
            pur = np.stack([s[d + 1]["n_purchases"] - s[d]["n_purchases"] for d in range(D)])
            ever = np.stack([s[d]["n_purchases"] > 0 for d in range(D + 1)])
            for c in np.unique(coh):
                m = coh == c
                st = stats.setdefault(int(c), {"n": 0, "a11": 0, "a1": 0, "a01": 0, "a0": 0, "act0": 0, "sess": [],
                                                "new_adopt": 0, "at_risk": 0, "pur": 0, "act_days": 0,
                                                "heavy": 0})
                st["n"] += int(m.sum())
                st["act0"] += int(act[0, m].sum())
                for d in range(D - 1):
                    a, b = act[d, m], act[d + 1, m]
                    st["a11"] += int((a & b).sum())
                    st["a1"] += int(a.sum())
                    st["a01"] += int((~a & b).sum())
                    st["a0"] += int((~a).sum())
                st["sess"].extend(sess[:, m][act[:, m]].tolist())
                for d in range(D):
                    risk = m & ~ever[d] & act[d]
                    st["at_risk"] += int(risk.sum())
                    st["new_adopt"] += int((risk & ever[d + 1]).sum())
                st["pur"] += int(pur[:, m].sum())
                st["act_days"] += int(act[:, m].sum())
                st["heavy"] += int((sess[:, m] >= RARE_SESSIONS).sum())
            end = s[-1]
            kinds = end["res_kind"]
            for k in np.unique(kinds):
                sel = kinds == k
                res_kind_visits[int(k)] = res_kind_visits.get(int(k), 0) + float(end["res_visits"][sel].sum())
                res_kind_count[int(k)] = res_kind_count.get(int(k), 0) + int(sel.sum())
                v = np.sort(end["res_visits"][sel].astype(np.float64))[::-1]
                if v.sum() > 0:
                    conc.setdefault(int(k), []).append(v / v.sum())
            if end["res_purchases"].sum() > 0:
                rev_per_purchase.append(float(end["res_revenue_cents"].sum() / end["res_purchases"].sum()))
            shop = end["res_purchases"] > 0
            if end["res_visits"][shop].sum() > 0:
                purchases_per_visit_shop.append(float(end["res_purchases"][shop].sum() / end["res_visits"].sum()))
            self.fitted_on.append({"n": run["n"], "seed": run["seed"], "days": run["days"],
                                   "shop_share": run["shop_share"], "sites": run["sites"]})
        p = {}
        for c, st in stats.items():
            sess = np.asarray(st["sess"], np.float64)
            p[c] = {"p11": st["a11"] / max(1, st["a1"]), "p01": st["a01"] / max(1, st["a0"]),
                    "p_act0": st["act0"] / max(1, st["n"]),
                    "sess_mean": float(sess.mean()) if sess.size else 0.0,
                    "sess_hist": np.bincount(sess.astype(np.int64), minlength=12)[:12] / max(1, sess.size) if sess.size
                    else np.zeros(12),
                    "adopt_hazard": st["new_adopt"] / max(1, st["at_risk"]), "pur_per_act": st["pur"] / max(1, st["act_days"])}
        self.params = {"cohorts": p,
                       "kind_attr": {k: res_kind_visits[k] / max(1, res_kind_count[k]) for k in res_kind_visits},
                       "kind_conc": {k: self._mean_profile(v) for k, v in conc.items()},
                       "rev_per_purchase": float(np.mean(rev_per_purchase)) if rev_per_purchase else 0.0,
                       "N_fit": int(np.mean([r["n"] for r in runs]))}
        return self

    @staticmethod
    def _mean_profile(profiles: list[np.ndarray], L: int = 64) -> np.ndarray:
        out = np.zeros(L)
        for v in profiles:
            x = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, v.size), np.cumsum(v)) if v.size > 1 else np.ones(L)
            out += x
        cum = out / max(1, len(profiles))
        return np.diff(np.concatenate([[0.0], cum]))

    def predict(self, cohorts: dict[int, int], res_kinds: np.ndarray, days: int, seed: int = 0) -> dict:
        """Estimated outcomes for a population with `cohorts` {cohort id: size} and web resources of `res_kinds`."""
        rng = np.random.default_rng(seed)
        P = self.params["cohorts"]
        N = int(sum(cohorts.values()))
        act_days = []
        adopt = np.zeros(days)
        heavy = 0.0
        tot_act = np.zeros(days)
        tot_sess = np.zeros(days)
        purchases = np.zeros(days)
        ret_num = ret_den = 0.0
        for c, n in cohorts.items():
            q = P.get(int(c)) or P[min(P, key=lambda x: abs(x - int(c)))]
            a = rng.binomial(n, q["p_act0"])
            adopted = 0
            prev = a
            for d in range(days):
                if d > 0:
                    stay = rng.binomial(prev, q["p11"])
                    a = stay + rng.binomial(n - prev, q["p01"])
                    ret_num += stay
                    ret_den += prev
                    prev = a
                tot_act[d] += a
                s = rng.poisson(q["sess_mean"] * a) if a else 0
                tot_sess[d] += s
                at_risk = a * (n - adopted) / max(1, n)
                new = rng.binomial(int(round(at_risk)), min(1.0, q["adopt_hazard"])) if at_risk >= 1 else 0
                adopted = min(n, adopted + new)
                adopt[d] += adopted
                purchases[d] += rng.poisson(q["pur_per_act"] * a) if a else 0
                heavy += a * float(np.asarray(q["sess_hist"])[RARE_SESSIONS:].sum())
            act_days.append(n)
        # visits over resources: attractiveness by kind, within-kind concentration profile
        kinds = np.asarray(res_kinds, np.int64)
        visits = np.zeros(kinds.size)
        total_visits = float(tot_sess.sum())
        attr = np.array([self.params["kind_attr"].get(int(k), 0.0) for k in kinds])
        if attr.sum() > 0:
            for k in np.unique(kinds):
                sel = np.flatnonzero(kinds == k)
                share = attr[sel].sum() / attr.sum()
                prof = self.params["kind_conc"].get(int(k))
                w = np.interp(np.linspace(0, 1, sel.size), np.linspace(0, 1, prof.size), prof) if prof is not None \
                    else np.ones(sel.size)
                w = w / max(w.sum(), 1e-12)
                visits[sel] = total_visits * share * w
        # distinct visitors from visits (occupancy: each visit draws a visitor among the active population)
        mean_active = max(1.0, float(tot_act.mean()))
        uniq = mean_active * (1.0 - np.exp(-visits / mean_active))
        reach = []
        top = np.argsort(-uniq)[:5]
        for d in range(days):
            vd = visits[top] * (tot_sess[:d + 1].sum() / max(total_visits, 1e-12))
            reach.append(float((mean_active * (1.0 - np.exp(-vd / mean_active)) / max(N, 1)).mean()))
        rev = purchases * self.params["rev_per_purchase"]
        return {"estimated": True, "mode": MODE_ID, "fitted_on": list(self.fitted_on), "N": N, "days": days,
                "adoption": (adopt / max(N, 1)).tolist(), "audience_shares": sorted((uniq / max(N, 1)).tolist(), reverse=True),
                "revenue_per_day": rev.tolist(), "revenue_total": float(rev.sum()), "reach_top": reach,
                "retention": float(ret_num / max(1.0, ret_den)),
                "rare_individual_days": float(heavy / max(1.0, N * days)),
                "rare_resources": float((uniq / max(N, 1) >= RARE_AUDIENCE).mean()) if uniq.size else 0.0,
                "active_share": float(tot_act.mean() / max(N, 1)),
                "sessions_per_active": float(tot_sess.sum() / max(1.0, tot_act.sum()))}


def cohorts_of(run: dict) -> dict[int, int]:
    s0 = run["snaps"][0]
    coh = CohortModel._cohort(s0["class"], s0["segment"])
    u, n = np.unique(coh, return_counts=True)
    return {int(a): int(b) for a, b in zip(u, n)}


# ====================================================================== errors
def _ks(a, b) -> float:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if a.size == 0 or b.size == 0:
        return 1.0 if a.size != b.size else 0.0
    xs = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, xs, "right") / a.size - np.searchsorted(b, xs, "right") / b.size)))


def errors(est: dict, obs: dict, est_effect: float | None = None, obs_effect: float | None = None) -> dict:
    """Error of the estimate against the observed individual simulation, per quantity."""
    rel = lambda e, o: abs(e - o) / max(abs(o), 1e-9)     # noqa: E731
    D = min(len(est["adoption"]), len(obs["adoption"]))
    out = {
        "adoption_trajectory": float(np.max(np.abs(np.array(est["adoption"][:D]) - np.array(obs["adoption"][:D])))),
        "audience_distribution": _ks(est["audience_shares"], obs["audience_shares"]),
        "revenue": rel(est["revenue_total"], obs["revenue_total"]),
        "network_diffusion": float(np.mean(np.abs(np.array(est["reach_top"][:D]) - np.array(obs["reach_top"][:D])))),
        "retention": abs(est["retention"] - obs["retention"]),
        "rare_event_frequency": rel(est["rare_individual_days"], obs["rare_individual_days"]),
    }
    if est_effect is not None and obs_effect is not None:
        out["intervention_effect"] = abs(est_effect - obs_effect) / max(abs(obs_effect), 1e-9)
    return out


def validate(*, n_fit: int = 400, n_target: int = 1600, days: int = 4, fit_seeds=(0,), target_seeds=(1, 2, 3),
             shop_share: float = 0.4, treat_share: float = 0.7, sites: int = 24) -> dict:
    """Fit on high-fidelity runs of n_fit individuals, predict n_target individuals (other seeds) and an intervention
    (a different shop share of the web), and measure the error of every quantity against high-fidelity runs."""
    t0 = time.time()
    fits = [hifi_run(n_fit, s, days, shop_share=shop_share, sites=sites) for s in fit_seeds]
    model = CohortModel().fit(fits)
    per_seed = []
    for s in target_seeds:
        ctl = hifi_run(n_target, s, days, shop_share=shop_share, sites=sites)
        trt = hifi_run(n_target, s, days, shop_share=treat_share, sites=sites)
        o_c, o_t = outcomes(ctl), outcomes(trt)
        e_c = model.predict(cohorts_of(ctl), ctl["snaps"][-1]["res_kind"], days, seed=100 + s)
        e_t = model.predict(cohorts_of(trt), trt["snaps"][-1]["res_kind"], days, seed=200 + s)
        err = errors(e_c, o_c, e_t["revenue_total"] - e_c["revenue_total"], o_t["revenue_total"] - o_c["revenue_total"])
        per_seed.append({"seed": s, "errors": err, "observed": {k: o_c[k] for k in ("revenue_total", "retention",
                                                                                     "active_share", "rare_individual_days")},
                         "estimated": {k: e_c[k] for k in ("revenue_total", "retention", "active_share",
                                                           "rare_individual_days")},
                         "observed_effect": o_t["revenue_total"] - o_c["revenue_total"],
                         "estimated_effect": e_t["revenue_total"] - e_c["revenue_total"],
                         "hifi_wall_s": ctl["wall_s"] + trt["wall_s"]})
    summary = {}
    for q in QUANTITIES:
        v = np.array([p["errors"][q] for p in per_seed if q in p["errors"]], float)
        if v.size:
            m, sd = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
            half = 1.96 * sd / math.sqrt(v.size) if v.size > 1 else float("nan")
            summary[q] = {"mean": round(m, 5), "ci95": [round(m - half, 5), round(m + half, 5)] if v.size > 1 else None,
                          "per_seed": [round(float(x), 5) for x in v]}
    return {"mode": MODE_ID, "estimated": True, "n_fit": n_fit, "n_target": n_target, "days": days,
            "fit_seeds": list(fit_seeds), "target_seeds": list(target_seeds), "shop_share": shop_share,
            "treat_share": treat_share, "errors": summary, "per_seed": per_seed,
            "params_fitted": {"cohorts": len(model.params["cohorts"]), "kinds": len(model.params["kind_attr"])},
            "wall_s": round(time.time() - t0, 1),
            "limits": ["cohorts are exchangeable inside: no individual memory or contact graph, so cascades enter only "
                       "through fitted reach curves",
                       "resource attractiveness is a per-kind constant fitted on the control mix",
                       "no feedback from prices or learners changing during the run",
                       "valid only inside the configuration range it was fitted on (sizes, site mixes, horizons)"]}
