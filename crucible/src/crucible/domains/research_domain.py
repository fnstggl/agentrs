"""Tier 2 domain 9.5 - Scientific research (MASTER_SPEC Part 9.5; workstream I-1).

Example objective: discover an unknown environmental or technical relationship and demonstrate its validity.

A configuration is a slice of the civilisation's scientific-research subsystem (D-17): the `sci` module (phenomena with
hidden laws, paid and queued apparatus with hidden calibration, certified standards, datasets, papers whose models
are executable, replication reports, the standards bureau's blind prediction challenges at novel conditions), the
substrate, documents and search, and the ONE population (`pop`), whose researchers take part through the kernel
module `scipop` defined here. Nothing else acts: no NPC labs (D-15).

Population researchers (`scipop`, fidelity record FR-SCIPOP-1, plausible, unvalidated). At genesis the individuals
with the strongest evaluating, technical, reading and planning skills (a configured share of the population) become
researchers, each with a method of its own drawn from its skills and preferences: which of the public handbook's
response forms it knows (`TEXTBOOK`, published in-world as a document anyone can read), the highest polynomial
degree it tries, whether it calibrates against certified standards, how many points and replicates it measures,
whether it designs grids or random designs, how much of its own money it spends, and how much it prefers
replicating others to new studies. In its `research` sessions (an activity of the population, woken when its runs
complete) a researcher either studies a phenomenon (books runs with its own money, waits for them, fits its models
to its own corrected readings, publishes the best by AIC) or replicates a paper it has not yet replicated (calibrates
on an apparatus the paper did not use when it can, measures a design over the apparatus' range and files the report
the module computes). Every choice is a recorded decision (`population.record_decisions`, model `sci-research@1`).
Researchers read only tool results; none reads a law, a gain or an evaluator verdict.

Evaluator (hidden): the claimed law of each of the principal's papers on the target phenomenon is compared with the
true law at evaluator-drawn novel conditions (outside every bookable apparatus range, inside the physical range):
normalised RMSE. Success needs a paper whose error is within the difficulty's threshold, published before any other
paper on the phenomenon reached that accuracy (a discovery, not a copy), with enough independent consistent
replications by others (validity demonstrated in the world).

Reference (hidden, LABELLED SCRIPTED): a tool-only researcher that calibrates one precise apparatus, measures a grid,
fits every handbook form and polynomial by BIC and publishes. Null baseline: doing nothing.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

import json

from ..core.kernel import Module, applier, register_module
from ..substrate import columns, ids
from . import _t2kit as K
from . import lawlang
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, Seat, register

DAY = 86400
DECISION = {"model_id": "sci-research", "model_version": "1", "fidelity_id": "FR-SCIPOP-1"}
CHOICE = {"wait": 0, "study": 1, "replicate": 2, "publish": 3, "report": 4, "unaffordable": 5, "none": 6}

# ====================================================================== the public handbook (participant knowledge)
# Response forms of the world's methods handbook (published as a public document at genesis). `{x}`, `{z}`, `{y}` are
# the variables in the order a researcher assigns them; p0.. are fitted constants; bounds keep the forms defined.
TEXTBOOK = [
    {"name": "linear", "n": 2, "expr": "p0 + p1 * {x} + p2 * {z}", "lb": [-1e3] * 3, "ub": [1e3] * 3},
    {"name": "saturating", "n": 2, "expr": "{x} / (p0 + {x}) * (p1 + p2 * {z})", "lb": [1e-3, -1e3, -1e3],
     "ub": [1e3, 1e3, 1e3]},
    {"name": "gaussian_peak", "n": 2, "expr": "p0 * exp(-(({x} - p1) / p2) ** 2) * (1 + p3 * {z}) + p4",
     "lb": [-1e3, -1e3, 0.05, -10.0, -1e3], "ub": [1e3, 1e3, 1e3, 10.0, 1e3]},
    {"name": "threshold_quadratic", "n": 2, "expr": "p0 * step({z} - p1) + p2 + p3 * {x} + p4 * {x} ** 2",
     "smooth": "p0 * sigmoid(30 * ({z} - p1)) + p2 + p3 * {x} + p4 * {x} ** 2",
     "lb": [-1e3, -1e3, -1e3, -1e3, -1e3], "ub": [1e3, 1e3, 1e3, 1e3, 1e3]},
    {"name": "power_activation", "n": 2, "expr": "p0 * {x} ** p1 * exp(-p2 / ({z} + p3))",
     "lb": [-1e3, 0.0, 0.0, 0.05], "ub": [1e3, 3.0, 50.0, 20.0]},
    {"name": "exponential", "n": 2, "expr": "p0 * exp(p1 * {x}) + p2 * {z} + p3", "lb": [-1e3, -3.0, -1e3, -1e3],
     "ub": [1e3, 3.0, 1e3, 1e3]},
    {"name": "linear3", "n": 3, "expr": "p0 + p1 * {x} + p2 * {z} + p3 * {y}", "lb": [-1e3] * 4, "ub": [1e3] * 4},
    {"name": "saturating_decay", "n": 3, "expr": "{x} / (p0 + {x}) * (p1 + p2 * {z}) * exp(-p3 * {y})",
     "lb": [1e-3, -1e3, -1e3, -5.0], "ub": [1e3, 1e3, 1e3, 5.0]},
    {"name": "damped_interaction", "n": 3, "expr": "p0 + p1 * {x} * {z} / (1 + p2 * {y} ** 2)",
     "lb": [-1e3, -1e3, 0.0], "ub": [1e3, 1e3, 1e3]},
]
FORMS = {f["name"]: f for f in TEXTBOOK}


def handbook_text() -> str:
    lines = ["Handbook of empirical response forms", "",
             "Common functional forms for a response measured against condition variables. x, z, y stand for the "
             "conditions in whatever order fits; p0, p1, ... are constants fitted to data (weighted least squares on "
             "readings corrected with certified standards; choose between forms by an information criterion).", ""]
    for f in TEXTBOOK:
        lines.append(f"- {f['name']} ({f['n']} variables): " + f["expr"].format(x="x", z="z", y="y"))
    lines += ["", "Polynomials of degree 1 or 2 in every variable are the general-purpose fallback."]
    return "\n".join(lines)


def _env(points: list[dict], names: list[str]) -> dict:
    return {k: np.array([float(p[k]) for p in points], dtype=float) for k in names}


def _init(form: dict, X: dict, y: np.ndarray, x: str, z: str) -> list[list[float]]:
    """A few deterministic starting points for a form (from the data's scale)."""
    ym, ys = float(np.mean(y)), float(np.std(y)) or 1.0
    xv = X[x]
    name = form["name"]
    if name == "linear":
        return [[ym, 0.0, 0.0]]
    if name == "linear3":
        return [[ym, 0.0, 0.0, 0.0]]
    if name == "saturating":
        return [[max(0.5, float(np.median(xv))), ym, 0.0], [2.0, ym * 0.7, ys * 0.1]]
    if name == "gaussian_peak":
        xm = float(xv[int(np.argmax(y))])
        return [[float(np.ptp(y)) or 1.0, xm, max(0.5, float(np.ptp(xv)) / 4), 0.0, float(np.min(y))],
                [float(np.ptp(y)) or 1.0, xm, max(0.5, float(np.ptp(xv)) / 2), 0.1, float(np.min(y))]]
    if name == "threshold_quadratic":
        zv = X[z]
        return [[ys, float(np.median(zv)), ym, 0.0, 0.0], [ys, float(np.percentile(zv, 70)), ym, 0.5, -0.05]]
    if name == "power_activation":
        return [[ym, 0.5, 1.0, 0.5], [ym * 2, 0.4, 2.0, 1.0]]
    if name == "exponential":
        return [[ym, 0.0, 0.0, 0.0], [ym * 0.5, 0.1, 0.0, ym * 0.5]]
    if name == "saturating_decay":
        return [[max(0.5, float(np.median(xv))), ym, 0.0, 0.0], [2.0, ym, ys * 0.1, 0.1]]
    if name == "damped_interaction":
        return [[ym, 0.1, 0.05], [float(np.min(y)), 0.4, 0.1]]
    return [[0.0] * len(form["lb"])]


def fit_form(form: dict, points: list[dict], y: np.ndarray, w: np.ndarray, assign: tuple[str, ...],
             names: list[str]) -> tuple[float, str] | None:
    """Weighted least squares of one handbook form with one variable assignment. Returns (weighted RSS, model text)."""
    from scipy.optimize import least_squares
    k = len(form["lb"])
    pn = [f"p{j}" for j in range(k)]
    sub = dict(zip(("x", "z", "y"), assign))
    text_fit = form.get("smooth", form["expr"]).format(**sub)
    try:
        ex = lawlang.compile_expr(text_fit, list(names) + pn)
    except lawlang.LawError:
        return None
    X = _env(points, names)

    def resid(p):
        env = dict(X, **{pn[j]: float(p[j]) for j in range(k)})
        m = np.broadcast_to(ex(env), y.shape)
        r = (m - y) * w
        return np.where(np.isfinite(r), r, 1e6)
    best = None
    lb, ub = np.array(form["lb"], float), np.array(form["ub"], float)
    for p0 in _init(form, X, y, sub["x"], sub["z"]):
        p0 = np.clip(np.array(p0, float), lb + 1e-9, ub - 1e-9)
        try:
            sol = least_squares(resid, p0, bounds=(lb, ub), max_nfev=400)
        except (ValueError, FloatingPointError):
            continue
        rss = float(np.sum(sol.fun ** 2))
        if math.isfinite(rss) and (best is None or rss < best[0]):
            best = (rss, sol.x)
    if best is None:
        return None
    text = form["expr"].format(**sub)
    for j in reversed(range(k)):
        text = text.replace(f"p{j}", f"({lawlang.fmt(best[1][j])})")
    try:
        ex2 = lawlang.compile_expr(text, names)
    except lawlang.LawError:
        return None
    m = ex2.at(points)
    if not np.all(np.isfinite(m)):
        return None
    return float(np.sum(((m - y) * w) ** 2)), text


def fit_poly(points: list[dict], y: np.ndarray, w: np.ndarray, names: list[str], deg: int) -> tuple[float, str, int]:
    X = np.array([[float(p[k]) for k in names] for p in points])
    cols, terms = [np.ones(len(y))], ["1"]
    for i, k in enumerate(names):
        cols.append(X[:, i])
        terms.append(k)
    if deg >= 2:
        for i, k in enumerate(names):
            cols.append(X[:, i] ** 2)
            terms.append(f"{k} ** 2")
        for i, j in itertools.combinations(range(len(names)), 2):
            cols.append(X[:, i] * X[:, j])
            terms.append(f"{names[i]} * {names[j]}")
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    rss = float(np.sum(((A @ coef) - y) ** 2 * w ** 2))
    parts = [f"({lawlang.fmt(c)})" if t == "1" else f"({lawlang.fmt(c)}) * {t}" for c, t in zip(coef, terms)]
    return rss, " + ".join(parts), A.shape[1]


def best_model(points: list[dict], y, w, names: list[str], forms: list[str], max_deg: int,
               criterion: str = "bic") -> str | None:
    """The model a scientist with this method would publish: every known handbook form under every variable
    assignment plus polynomials up to `max_deg`, the lowest information criterion on weighted residuals."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    n = len(y)
    cands = []
    for deg in range(1, max_deg + 1):
        rss, text, k = fit_poly(points, y, w, names, deg)
        if n > k + 1:
            cands.append((rss, k, text))
    for fname in forms:
        f = FORMS[fname]
        if f["n"] > len(names):
            continue
        k = len(f["lb"])
        if n <= k + 1:
            continue
        for assign in itertools.permutations(names, f["n"]):
            r = fit_form(f, points, y, w, assign, names)
            if r is not None:
                cands.append((r[0], k, r[1]))
    if not cands:
        return None
    pen = math.log(max(n, 2)) if criterion == "bic" else 2.0
    return min(cands, key=lambda c: (n * math.log(max(c[0], 1e-12) / n) + pen * c[1], c[2]))[2]


def poly_forms(n_vars: int, deg: int) -> list[dict]:
    """Polynomials of degree `deg` as handbook-style forms (for the joint fit)."""
    vs = ["{x}", "{z}", "{y}"][:n_vars]
    terms = ["1"] + vs
    if deg >= 2:
        terms += [f"{v} ** 2" for v in vs] + [f"{a} * {b}" for a, b in itertools.combinations(vs, 2)]
    expr = " + ".join(f"p{j}" if t == "1" else f"p{j} * {t}" for j, t in enumerate(terms))
    return [{"name": f"poly{deg}", "n": n_vars, "expr": expr, "lb": [-1e4] * len(terms), "ub": [1e4] * len(terms)}]


def fit_joint(forms: list[dict], blocks: list[dict], names: list[str], anchors: list[tuple[dict, float]],
              criterion: str = "bic") -> str | None:
    """Joint weighted least squares of a response form and each apparatus' gain and offset on RAW readings, with the
    certified standards as exact anchors of the law (their certified value is the law at their conditions).
    `blocks`: one per apparatus {points, raw, sd, rel, std_raw: [(certified, reading)]}. The forms' constants and
    every (gain, offset) are fitted together, so a gain that two close standards cannot pin down is resolved by the
    law's shape and the anchors. Returns the model text with the lowest information criterion."""
    from scipy.optimize import least_squares
    n_obs = sum(len(b["raw"]) + len(b["std_raw"]) for b in blocks) + len(anchors)
    best = None
    for form in forms:
        if form["n"] > len(names):
            continue
        k = len(form["lb"])
        if n_obs <= k + 2 * len(blocks) + 1:
            continue
        pn = [f"p{j}" for j in range(k)]
        for assign in itertools.permutations(names, form["n"]):
            sub = dict(zip(("x", "z", "y"), assign))
            try:
                ex = lawlang.compile_expr(form.get("smooth", form["expr"]).format(**sub), list(names) + pn)
            except lawlang.LawError:
                continue
            envs = [_env(b["points"], names) for b in blocks]
            aenv = _env([a for a, _ in anchors], names) if anchors else None
            acert = np.array([c for _, c in anchors], float)

            def resid(q, ex=ex, k=k, pn=pn, envs=envs, aenv=aenv, acert=acert):
                th = {pn[j]: float(q[j]) for j in range(k)}
                out = []
                for bi, b in enumerate(blocks):
                    g, o = float(q[k + 2 * bi]), float(q[k + 2 * bi + 1])
                    m = np.broadcast_to(ex(dict(envs[bi], **th)), (len(b["raw"]),))
                    pred = g * m + o
                    sig = np.sqrt(b["sd"] ** 2 + (b["rel"] * np.abs(pred)) ** 2)
                    out.append((pred - b["raw"]) / sig)
                    if b["std_raw"]:
                        cs = np.array([c for c, _ in b["std_raw"]])
                        rs = np.array([r for _, r in b["std_raw"]])
                        ps = g * cs + o
                        out.append((ps - rs) / np.sqrt(b["sd"] ** 2 + (b["rel"] * np.abs(ps)) ** 2))
                if aenv is not None:
                    ma = np.broadcast_to(ex(dict(aenv, **th)), acert.shape)
                    out.append((ma - acert) / (0.002 * np.abs(acert) + 1e-3))
                r = np.concatenate(out)
                return np.where(np.isfinite(r), r, 1e6)
            X0 = _env(blocks[0]["points"], names)
            y0 = np.asarray(blocks[0]["raw"], float)
            lb = list(form["lb"]) + [0.2, -1e3] * len(blocks)
            ub = list(form["ub"]) + [5.0, 1e3] * len(blocks)
            for p0 in _init(form, X0, y0, sub["x"], sub["z"]) if form["name"] in FORMS else [[0.0] * k]:
                q0 = np.clip(np.array(list(p0) + [1.0, 0.0] * len(blocks), float), np.array(lb) + 1e-9,
                             np.array(ub) - 1e-9)
                try:
                    sol = least_squares(resid, q0, bounds=(lb, ub), max_nfev=600)
                except (ValueError, FloatingPointError):
                    continue
                rss = float(np.sum(sol.fun ** 2))
                if not math.isfinite(rss):
                    continue
                pen = math.log(n_obs) if criterion == "bic" else 2.0
                ic = n_obs * math.log(max(rss, 1e-12) / n_obs) + pen * (k + 2 * len(blocks))
                text = form["expr"].format(**sub)
                for j in reversed(range(k)):
                    text = text.replace(f"p{j}", f"({lawlang.fmt(sol.x[j])})")
                if best is None or ic < best[0]:
                    best = (ic, text)
    return None if best is None else best[1]


def corrected(readings: list[dict], certified: dict[str, float], app: dict) -> tuple[list[dict], np.ndarray,
                                                                                      np.ndarray]:
    """Participant-side calibration: readings at conditions corrected by a weighted line through the measured
    certified standards (offset only with one distinct standard); weights from the stated precision."""
    sd, rel = float(app.get("stated_sd", 1.0)), float(app.get("stated_rel", 0.0))
    xs = [certified[r["standard"]] for r in readings if r.get("standard") in certified]
    ys = [float(r["value"]) for r in readings if r.get("standard") in certified]
    g, o = 1.0, 0.0
    if xs:
        xa, ya = np.array(xs), np.array(ys)
        wt = 1.0 / (sd ** 2 + (rel * np.abs(xa)) ** 2)
        if len(set(np.round(xa, 6))) >= 2:
            A = np.stack([np.ones(len(xa)), xa], 1)
            o, g = np.linalg.solve(A.T @ (A * wt[:, None]), A.T @ (wt * ya))
            if not 0.2 < g < 5.0:
                g, o = 1.0, float(np.sum(wt * (ya - xa)) / np.sum(wt))
        else:
            o = float(np.sum(wt * (ya - xa)) / np.sum(wt))
    pts = [r["at"] for r in readings if not r.get("standard")]
    y = (np.array([float(r["value"]) for r in readings if not r.get("standard")]) - o) / g
    w = 1.0 / np.sqrt(sd ** 2 + (rel * np.abs(y)) ** 2)
    return pts, y, w


# ====================================================================== population researchers (kernel module)
RES_DEFAULTS = {"share": 0.03, "min": 4, "cooldown_days": [4, 12], "replicate_max_reports": 4, "max_runs": 3,
                "handbook": True, "start_hours": [24, 120]}


@register_module
class PopResearchers(Module):
    """Researchers of the one population (see the module doc)."""
    name = "scipop"
    host = "scipop"
    depends = ("identity", "bank", "pop", "cols", "sci", "docs")

    def cfg(self, k):
        return self.config.get(k, RES_DEFAULTS[k])

    def install(self, world):
        from .. import population
        n = population.n_individuals(world)
        columns.declare(world, "scipop", n, {"choice": ("int16", (), -1), "studies": ("int32", (), 0),
                                             "reports": ("int32", (), 0), "spent": ("int64", (), 0)})
        world.state.declare_table("scipop_r")
        population.register_activity(world, "research", self.act_research, owner="scipop", prior=-1.5,
                                     goals=(1, 5))

    def genesis(self, ctx):
        from .. import population
        w = ctx.world
        n = population.n_individuals(w)
        if not n:
            return
        rng = w.rng_for("scipop-genesis")
        f = population.features(w, np.arange(n))
        sk = f["skills"].astype(np.float64)
        pr = f["prefs"].astype(np.float64)
        score = sk[:, 2] + sk[:, 4] + sk[:, 6] + sk[:, 7] + rng.normal(0.0, 0.1, n)
        k = max(int(self.cfg("min")), int(round(float(self.cfg("share")) * n)))
        chosen = np.sort(np.argsort(-score, kind="stable")[:k])
        forms2 = [t["name"] for t in TEXTBOOK if t["n"] == 2 and t["name"] != "linear"]
        forms3 = [t["name"] for t in TEXTBOOK if t["n"] == 3]
        rows = []
        for i in chosen:
            i = int(i)
            tech, plan, ev = float(sk[i, 4]), float(sk[i, 7]), float(sk[i, 2])
            nf = int(min(len(forms2), math.floor(tech * 3.0)))
            known = ["linear"] + sorted(str(x) for x in rng.choice(forms2, size=nf, replace=False))
            if nf >= 2:
                known += sorted(str(x) for x in rng.choice(forms3, size=min(len(forms3), nf - 1), replace=False))
            rows.append({"i": i, "forms": known, "deg": 1 + int(plan > 0.45),
                         "calib": bool(rng.random() < 0.35 + 0.55 * ev), "reps": 1 + int(rng.random() < 0.5 * ev),
                         "k": int(8 + round(20 * plan)), "grid": bool(plan > 0.5),
                         "replicate": float(np.clip(0.25 + 0.45 * ev - 0.25 * pr[i, 1], 0.05, 0.9)),
                         "budget": float(np.clip(0.15 + 0.45 * (1.0 - pr[i, 0]), 0.1, 0.7)),
                         "stage": "idle", "runs": [], "cal": [], "paper": None, "phen": None,
                         "next": int(ctx.now + int(rng.integers(self.cfg("start_hours")[0],
                                                                self.cfg("start_hours")[1] + 1)) * 3600),
                         "done": [], "published": 0, "reported": 0})
        ctx.emit("scipop.researchers", {"rows": rows}, vis="internal")
        for r in rows:
            population.wake(ctx, np.array([r["i"]]), r["next"], "research")
        if self.cfg("handbook") and "docs" in w.modules:
            idm = w.modules["identity"]
            org = idm.create_org(ctx, "methods-society", "Methods Society", None, "group")
            w.modules["docs"].create_doc(ctx, org, "Handbook of empirical response forms", handbook_text(), "public")

    # ------------------------------------------------------------------ helpers
    def _call(self, ctx, p, tool, args):
        from ..core.errors import CrucibleError
        try:
            return ctx.call(tool, args, as_principal=p)
        except CrucibleError:
            return None

    def _set(self, ctx, i, **ch):
        ctx.emit("scipop.state", {"i": int(i), "changes": ch}, vis="internal")

    def _wake(self, ctx, i, at):
        from .. import population
        population.wake(ctx, np.array([int(i)]), int(at), "research")

    def _fund(self, ctx, i) -> int:
        from ..population import core
        ids.ensure_principal(ctx, int(i), account=True)
        core.money_deposit(ctx, int(i))
        bank = ctx.world.modules["bank"]
        acct = bank.primary_account(ctx.state, ids.pid(int(i)))
        return int(bank.balance(ctx.state, acct)) if acct else 0

    @staticmethod
    def _grid(app: dict, names: list[str], k: int, reps: int, rng, grid: bool) -> list[dict]:
        rngs = app["range"]
        if grid:
            per = max(2, int(round(k ** (1.0 / max(1, len(names))))))
            axes = [np.linspace(rngs[v][0], rngs[v][1], per) for v in names]
            pts = [dict(zip(names, map(float, c))) for c in itertools.product(*axes)]
        else:
            pts = [{v: float(rng.uniform(*rngs[v])) for v in names} for _ in range(k)]
        return [{"at": {v: round(p[v], 4) for v in names}, "n": int(reps)} for p in pts]

    # ------------------------------------------------------------------ the research session
    def act_research(self, ctx, idx: np.ndarray, now: int):
        st = ctx.state
        util = np.full(idx.size, -0.4)
        dec_rows, dec_choice = [], []
        for r_, i in enumerate(idx.tolist()):
            row = st.get("scipop_r", str(i))
            if row is None:
                continue
            choice, u = self._session(ctx, int(i), row, now)
            util[r_] = u
            dec_rows.append(int(i))
            dec_choice.append(CHOICE[choice])
        if dec_rows:
            from .. import population
            population.record_decisions(ctx, "scipop.decide", np.array(dec_rows, np.int64),
                                        {"choice": np.array(dec_choice, np.int16)}, table="scipop", **DECISION)
        return util

    def _session(self, ctx, i: int, r: dict, now: int) -> tuple[str, float]:
        p = ids.pid(i)
        if r["stage"] in ("measuring", "replicating"):
            runs = self._call(ctx, p, "sci.runs", {}) or []
            mine = {x["run"]: x for x in runs}
            need = list(r["runs"]) + list(r["cal"])
            if any(mine.get(x, {}).get("status") != "done" for x in need):
                ready = max([int(mine[x]["ready_at"]) for x in need if x in mine] or [now + DAY])
                self._wake(ctx, i, max(ready + 600, now + 3600))
                return "wait", 0.05
            return (self._publish(ctx, i, p, r, now) if r["stage"] == "measuring" else
                    self._report(ctx, i, p, r, now))
        if now < int(r["next"]):
            self._wake(ctx, i, int(r["next"]))
            return "wait", 0.0
        rng = ctx.rng("scipop", i, now)
        papers = self._call(ctx, p, "sci.papers", {"limit": 200}) or []
        cands = [x for x in papers if x["author"] != p and x["paper"] not in r["done"]
                 and sum(x["replications"][k] for k in ("consistent", "refuted")) < int(self.cfg("replicate_max_reports"))
                 and x["t"] <= now - DAY // 2]
        if cands and rng.random() < float(r["replicate"]):
            return self._start_replication(ctx, i, p, r, now, cands, rng)
        return self._start_study(ctx, i, p, r, now, papers, rng)

    def _affordable(self, ctx, i, r, price_per, n) -> int:
        bal = self._fund(ctx, i)
        budget = int(float(r["budget"]) * bal)
        return int(min(n, budget // max(1, int(price_per))))

    def _book(self, ctx, p, app, design):
        res = self._call(ctx, p, "sci.run", {"apparatus": app["apparatus"], "design": design})
        return res

    def _start_study(self, ctx, i, p, r, now, papers, rng):
        phens = self._call(ctx, p, "sci.phenomena", {}) or []
        if not phens:
            return "none", -0.2
        chal = self._call(ctx, p, "sci.challenges", {"limit": 100}) or []
        passed = {res["paper"] for c in chal for res in c["results"] if res["verdict"] == "passed"}
        solved = {x["phenomenon"] for x in papers if x["paper"] in passed}
        open_ = [ph for ph in phens if ph["id"] not in solved] or phens
        ph = open_[int(rng.integers(0, len(open_)))]
        apps = self._call(ctx, p, "sci.apparatus", {"phenomenon": ph["id"]}) or []
        if not apps:
            return "none", -0.2
        pr = float(np.clip(r["budget"], 0.1, 0.9))
        # value for money: precision per (price ^ thrift), thrift from the researcher's own budget preference
        apps.sort(key=lambda a: (-(1.0 / max(1e-6, a["stated_sd"])) / (a["price_cents"] ** (1.2 - pr)),
                                 a["apparatus"]))
        app = apps[0]
        names = [v["name"] for v in ph["vars"]]
        design = self._grid(app, names, int(r["k"]), int(r["reps"]), rng, bool(r["grid"]))
        cal = [{"standard": s["id"], "n": 2} for s in ph["standards"]] if r["calib"] and ph["standards"] else []
        n_meas = sum(int(d["n"]) for d in design) + sum(int(d["n"]) for d in cal)
        if self._affordable(ctx, i, r, app["price_cents"], n_meas) < n_meas:
            design = design[: max(0, self._affordable(ctx, i, r, app["price_cents"], n_meas) - 2 * len(cal))]
            if len(design) < 5:
                self._set(ctx, i, next=now + 7 * DAY)
                self._wake(ctx, i, now + 7 * DAY)
                return "unaffordable", -0.2
        res = self._book(ctx, p, app, design[:240])
        if res is None:
            self._set(ctx, i, next=now + 2 * DAY)
            self._wake(ctx, i, now + 2 * DAY)
            return "none", -0.1
        runs, cals, ready = [res["run"]], [], int(res["ready_at"])
        if cal:
            c = self._book(ctx, p, app, cal)
            if c is not None:
                cals.append(c["run"])
                ready = max(ready, int(c["ready_at"]))
        self._set(ctx, i, stage="measuring", runs=runs, cal=cals, phen=ph["id"], paper=None)
        self._wake(ctx, i, ready + 600)
        return "study", 0.1

    def _start_replication(self, ctx, i, p, r, now, cands, rng):
        cands.sort(key=lambda x: (sum(x["replications"][k] for k in ("consistent", "refuted")), x["t"], x["paper"]))
        paper = cands[0]
        apps = self._call(ctx, p, "sci.apparatus", {"phenomenon": paper["phenomenon"]}) or []
        if not apps:
            return "none", -0.2
        apps.sort(key=lambda a: (a["apparatus"] in paper["apparatus"], a["price_cents"], a["apparatus"]))
        app = apps[0]
        phens = {ph["id"]: ph for ph in (self._call(ctx, p, "sci.phenomena", {}) or [])}
        ph = phens[paper["phenomenon"]]
        names = [v["name"] for v in ph["vars"]]
        design = self._grid(app, names, max(6, int(r["k"]) // 2), 1, rng, True)
        cal = [{"standard": s["id"], "n": 2} for s in ph["standards"]] if r["calib"] and ph["standards"] else []
        n_meas = sum(int(d["n"]) for d in design) + sum(int(d["n"]) for d in cal)
        if self._affordable(ctx, i, r, app["price_cents"], n_meas) < n_meas:
            self._set(ctx, i, next=now + 7 * DAY, done=list(r["done"]) + [paper["paper"]])
            self._wake(ctx, i, now + 7 * DAY)
            return "unaffordable", -0.2
        res = self._book(ctx, p, app, design)
        if res is None:
            self._set(ctx, i, next=now + 2 * DAY, done=list(r["done"]) + [paper["paper"]])
            self._wake(ctx, i, now + 2 * DAY)
            return "none", -0.1
        cals, ready = [], int(res["ready_at"])
        if cal:
            c = self._book(ctx, p, app, cal)
            if c is not None:
                cals.append(c["run"])
                ready = max(ready, int(c["ready_at"]))
        self._set(ctx, i, stage="replicating", runs=[res["run"]], cal=cals, paper=paper["paper"],
                  phen=paper["phenomenon"], done=list(r["done"]) + [paper["paper"]])
        self._wake(ctx, i, ready + 600)
        return "replicate", 0.1

    def _cooldown(self, ctx, i, now) -> int:
        lo, hi = self.cfg("cooldown_days")
        return now + int(ctx.rng("scipop-cool", i, now).integers(int(lo), int(hi) + 1)) * DAY

    def _publish(self, ctx, i, p, r, now):
        phens = {ph["id"]: ph for ph in (self._call(ctx, p, "sci.phenomena", {}) or [])}
        ph = phens.get(r["phen"])
        apps = {a["apparatus"]: a for a in (self._call(ctx, p, "sci.apparatus", {"phenomenon": r["phen"]}) or [])}
        nxt = self._cooldown(ctx, i, now)
        if ph is None:
            self._set(ctx, i, stage="idle", runs=[], cal=[], next=nxt)
            return "none", -0.2
        names = [v["name"] for v in ph["vars"]]
        certified = {s["id"]: float(s["certified"]) for s in ph["standards"]}
        pts, ys, ws = [], [], []
        for run in r["runs"]:
            d = self._call(ctx, p, "sci.data", {"run": run})
            if d is None:
                continue
            readings = list(d["readings"])
            for c in r["cal"]:
                cd = self._call(ctx, p, "sci.data", {"run": c})
                if cd is not None and cd["apparatus"] == d["apparatus"]:
                    readings += cd["readings"]
            a, y, w = corrected(readings, certified, apps.get(d["apparatus"], {}))
            pts += a
            ys += list(y)
            ws += list(w)
        row = self.world.state.tables["scipop_r"][str(i)]
        model = best_model(pts, ys, ws, names, list(row["forms"]), int(row["deg"]), criterion="aic") if pts else None
        if model is None:
            self._set(ctx, i, stage="idle", runs=[], cal=[], next=nxt)
            self._wake(ctx, i, nxt)
            return "none", -0.2
        res = self._call(ctx, p, "sci.publish", {"phenomenon": r["phen"], "title": f"A response law for {r['phen']}",
                                                 "abstract": "Weighted least-squares fit on calibrated readings."
                                                 if r["cal"] else "Least-squares fit on raw readings.",
                                                 "model": model, "data": list(r["runs"]),
                                                 "calibration_runs": list(r["cal"])})
        self._set(ctx, i, stage="idle", runs=[], cal=[], next=nxt, published=int(r["published"]) + int(res is not None))
        self._wake(ctx, i, nxt)
        return ("publish", 0.6) if res is not None else ("none", -0.2)

    def _report(self, ctx, i, p, r, now):
        res = self._call(ctx, p, "sci.replicate", {"paper": r["paper"], "runs": list(r["runs"]),
                                                   "calibration_runs": list(r["cal"])})
        nxt = self._cooldown(ctx, i, now)
        self._set(ctx, i, stage="idle", runs=[], cal=[], paper=None, next=nxt,
                  reported=int(r["reported"]) + int(res is not None))
        self._wake(ctx, i, nxt)
        return ("report", 0.5) if res is not None else ("none", -0.2)

    # ------------------------------------------------------------------ appliers
    @applier("scipop.researchers")
    def ap_researchers(self, ev):
        for r in ev.payload["rows"]:
            self.world.state.insert("scipop_r", str(r["i"]), dict(r))

    @applier("scipop.state")
    def ap_state(self, ev):
        self.world.state.patch("scipop_r", str(ev.payload["i"]), **ev.payload["changes"])


# ====================================================================== configurations (world generation)
VAR_RANGES = {2: [(14.0, 10.0), (7.0, 5.0)], 3: [(14.0, 10.0), (7.0, 5.0), (6.0, 4.0)]}


def _phenomenon(rng, pid: str, family: str, noise_rel: float) -> tuple[dict, dict]:
    nv = lawlang.n_vars(family)
    xs = lawlang.draw_variables(rng, nv)
    law = lawlang.ALL_FAMILIES[family](rng, xs)
    vr = VAR_RANGES[nv]
    phys = [{"name": x, "lo": 0.0, "hi": float(vr[j][0])} for j, x in enumerate(xs)]
    reach = {x: [0.0, float(vr[j][1])] for j, x in enumerate(xs)}
    s1 = {x: round(0.2 * vr[j][1] + 0.5, 3) for j, x in enumerate(xs)}
    s2 = {x: round(0.7 * vr[j][1], 3) for j, x in enumerate(xs)}
    ph = {"id": pid, "title": f"{', '.join(xs)} response of sample {pid}",
          "doc": "Response of a sample as a function of its conditions.", "vars": phys, "response": "response",
          "law": law["law"], "form": law["form"], "noise_rel": float(noise_rel),
          "standards": [{"id": f"{pid}s1", "at": s1}, {"id": f"{pid}s2", "at": s2}]}
    return ph, {"reach": reach, "vars": xs, "form": law["form"]}


def _apparatus(rng, pid: str, reach: dict, kind: str, *, drift: bool, optimistic: bool, capacity: int) -> list[dict]:
    """Bookable apparatus of one phenomenon. `full`: three instruments over the whole reachable box (cheap/biased,
    precise/expensive, middle); `tiled`: two instruments each covering part of the first variable's range plus one
    noisy full-range instrument."""
    first = sorted(reach)[0] if kind == "tiled" else None
    out = []
    specs = [("a", "instr", 300, 0.35, 0.03, capacity), ("b", "instr2", 700, 0.15, 0.01, max(10, capacity * 2 // 3)),
             ("c", "instr3", 450, 0.25, 0.02, capacity)]
    for j, (suffix, op, price, sd, rel, cap) in enumerate(specs):
        rg = {k: list(v) for k, v in reach.items()}
        if first is not None and suffix in ("a", "b"):
            lo, hi = reach[first]
            rg[first] = [lo, round(lo + 0.6 * (hi - lo), 3)] if suffix == "a" else [round(lo + 0.45 * (hi - lo), 3), hi]
        if first is not None and suffix == "c":
            sd, rel, price = sd * 2.0, rel * 1.5, price + 200
        a = {"id": f"{pid}{suffix}", "phenomenon": pid, "operator": op, "range": rg, "price_cents": int(price),
             "sd": round(sd, 4), "rel": round(rel, 4), "capacity_per_day": int(cap),
             "gain": round(float(rng.uniform(0.95, 1.06)), 4), "offset": round(float(rng.normal(0.0, 0.3)), 4)}
        if drift:
            a["drift_per_measure"] = round(float(rng.uniform(2e-4, 6e-4)), 6)
        if optimistic and suffix == "a":
            a["stated_rel"] = round(rel * 0.25, 4)       # the datasheet understates the relative noise
        out.append(a)
    return out


STRUCTURES = {
    "bench": {"law_forms": ["peak", "sat_lin"], "variables": 2, "phenomena": 1, "apparatus": "full",
              "drift": False, "datasheet": "honest", "standards": 2, "competition": "low"},
    "drift": {"law_forms": ["power_exp", "thresh_quad"], "variables": 2, "phenomena": 2, "apparatus": "full",
              "drift": True, "datasheet": "honest", "standards": 2, "competition": "low"},
    "survey": {"law_forms": ["interact3", "sat_exp3"], "variables": 3, "phenomena": 1, "apparatus": "tiled",
               "drift": False, "datasheet": "honest", "standards": 2, "competition": "high"},
    "drift_survey": {"law_forms": ["interact3", "sat_exp3"], "variables": 3, "phenomena": 2, "apparatus": "tiled",
                     "drift": True, "datasheet": "honest", "standards": 2, "competition": "low"},
    "heldout": {"law_forms": list(lawlang.HELD_OUT), "variables": 2, "phenomena": 1, "apparatus": "full",
                "drift": False, "datasheet": "optimistic", "standards": 1, "competition": "low"},
}
DOCS = {
    "bench": "one two-variable phenomenon, three full-range instruments, two certified standards",
    "drift": "two phenomena (the target and another), instruments whose gain drifts with use",
    "survey": "a three-variable phenomenon, instruments that each reach part of the range, scarce capacity, many "
              "competing researchers",
    "drift_survey": "compositional: three variables, tiled instruments AND drifting gains AND two phenomena",
    "heldout": "structural: a response form outside the handbook, one certified standard (offset-only "
               "calibration), an instrument whose datasheet understates its noise",
}


def build(config: DomainConfig, seed: int, overrides: dict) -> dict:
    """A slice of the civilisation's scientific-research subsystem (`base.civ_recipe`): `sci` with this
    configuration's phenomena, apparatus and bureau, documents and search, the population and its researchers."""
    import zlib
    from . import base
    st = STRUCTURES[config.name]
    ov = dict(overrides or {})
    n_pop = int(ov.pop("population", 900))
    rng = np.random.default_rng([int(seed), 9050, zlib.crc32(config.name.encode())])
    phen, apps, meta = [], [], {}
    forms = list(st["law_forms"])
    for k in range(int(st["phenomena"])):
        fam = forms[int(rng.integers(0, len(forms)))] if k == 0 else forms[(forms.index(meta["ph0"]["form"]) + 1)
                                                                             % len(forms)]
        ph, m = _phenomenon(rng, f"ph{k}", fam, noise_rel=0.02)
        if st["standards"] == 1:
            ph["standards"] = ph["standards"][:1]
        phen.append(ph)
        meta[ph["id"]] = m
        apps += _apparatus(rng, ph["id"], m["reach"], st["apparatus"], drift=st["drift"],
                           optimistic=st["datasheet"] == "optimistic",
                           capacity=25 if st["competition"] == "high" else 60)
    sci = {"phenomena": phen, "apparatus": apps,
           "operators": [{"handle": "instr", "name": "Instrument Works", "cash": 100_000},
                         {"handle": "instr2", "name": "Precision Labs", "cash": 100_000},
                         {"handle": "instr3", "name": "Field Instruments", "cash": 100_000}],
           "rivals": {"n": 0}, "replicators": {"n": 0},
           "challenge": {"every_days": 10, "start_day": 8, "points": 8}}
    if ov:
        raise ValueError(f"research domain: unknown overrides {sorted(ov)}")
    return base.civ_recipe(int(seed), subsystems=["scientific_research"], modules=["search", "docs", "scipop"],
                           population_modules=("pop",),
                           module_config={"sci": sci, "scipop": {"share": 0.05 if st["competition"] == "high"
                                                                 else 0.025},
                                          "pop": {"profile": "development", "n": n_pop, "tick_s": 1800,
                                                  "classes": {"adaptive": 0, "strategic": 0, "semantic": 0}}},
                           params={"target_phenomenon": "ph0", "log_retain": 3_000_000})
# ====================================================================== objective, evaluator
THRESHOLD = {1: 0.10, 2: 0.05, 3: 0.025}
REPLICATIONS = {1: 1, 2: 2, 3: 2}
CO_DISCOVERY_DAYS = 2.0          # independent discoveries this close together are simultaneous (credited to both)
HORIZON_DAYS = 30.0


def objective(world, principal: str, difficulty: int = 1) -> dict:
    d = int(min(3, max(1, difficulty)))
    target = world.spec.params.get("target_phenomenon", "ph0")
    text = (f"Discover the law that relates the response of phenomenon {target} to its conditions and demonstrate "
            f"its validity within {HORIZON_DAYS:.0f} days: your latest paper on {target} (sci.publish) must state a "
            f"model that predicts the response well, including at conditions beyond what the bookable apparatus "
            f"reach; you must have found an accurate law no later than {CO_DISCOVERY_DAYS:.0f} days after anyone "
            f"else did, in your own words (not a copy of another paper's model); and at least {REPLICATIONS[d]} "
            f"independent lab(s) must have replicated it as consistent.")
    return {"text": text, "description": text, "domain": "research", "difficulty": d, "phenomenon": target,
            "window_days": HORIZON_DAYS}


def _eval_points(world, phenomenon: str, n: int = 240) -> list[dict]:
    """Evaluator-side novel conditions: outside the union of every apparatus' range, inside the physical range
    (seeded from the world seed; never shown in-world)."""
    st = world.state
    ph = st.tables["sci_phen"][phenomenon]
    names = [v["name"] for v in ph["vars"]]
    lo = np.array([float(v["lo"]) for v in ph["vars"]])
    hi = np.array([float(v["hi"]) for v in ph["vars"]])
    apps = [a for _, a in st.scan("sci_app") if a["phenomenon"] == phenomenon]
    alo = np.array([min(a["range"][k][0] for a in apps) for k in names])
    ahi = np.array([max(a["range"][k][1] for a in apps) for k in names])
    rng = np.random.default_rng([int(world.spec.seed), 90517, len(phenomenon)])
    out = []
    for _ in range(200 * n):
        if len(out) >= n:
            break
        x = rng.uniform(lo, hi)
        if np.any((x < alo - 1e-9) | (x > ahi + 1e-9)):
            out.append({k: float(v) for k, v in zip(names, x)})
    return out


def novel_error(world, phenomenon: str, model_text: str) -> float:
    """Normalised RMSE of a model against the true law at the evaluator's novel conditions (inf if invalid)."""
    sci = world.modules["sci"]
    pts = _eval_points(world, phenomenon)
    if not pts:
        return float("inf")
    truth = sci.truth(phenomenon, pts)
    try:
        m = lawlang.compile_expr(model_text, sci.vars[phenomenon]).at(pts)
    except lawlang.LawError:
        return float("inf")
    if not np.all(np.isfinite(m)):
        return float("inf")
    scale = float(np.std(truth)) or 1.0
    return float(np.sqrt(np.mean((m - truth) ** 2)) / scale)


def evaluate(world, principal: str, obj: dict) -> dict:
    """Evaluator side. The principal's claim is its latest paper on the target phenomenon. Success: the claim's
    novel-condition error is within the difficulty's threshold; the principal's first accurate paper came no later
    than `CO_DISCOVERY_DAYS` after the first accurate paper by anyone else; the claim's model text is not a copy of
    an earlier paper by another author; and independent labs replicated the claim as consistent often enough."""
    d = int(obj.get("difficulty", 1))
    target = obj.get("phenomenon", "ph0")
    st = world.state
    thr, need = THRESHOLD[d], REPLICATIONS[d]
    mine_p = set(world.principals_of(principal)) | {principal} if world.principal(principal) else {principal}
    papers = sorted(((p["t"], pid, p) for pid, p in st.scan("sci_papers") if p["phenomenon"] == target),
                    key=lambda x: (x[0], x[1]))
    first_other, first_mine, rows, seen_texts = None, None, [], {}
    for t, pid, p in papers:
        err = novel_error(world, target, p["model"])
        mine = p["author"] in mine_p
        copied = mine and any(a not in mine_p and t0 <= t for a, t0 in seen_texts.get(p["model"], []))
        seen_texts.setdefault(p["model"], []).append((p["author"], t))
        if err <= thr:
            if mine and not copied and first_mine is None:
                first_mine = t
            if not mine and first_other is None:
                first_other = t
        if mine:
            reps = [st.tables["sci_reps"][r] for r in st.find_ids("sci_reps", "paper", pid)]
            ind = sum(1 for r in reps if r["independent"] and r["verdict"] == "consistent"
                      and r["reporter"] not in mine_p)
            rows.append({"paper": pid, "t": t, "novel_nrmse": round(err, 5) if math.isfinite(err) else None,
                         "independent_consistent": ind, "copied": copied, "reports": len(reps)})
    claim = rows[-1] if rows else None
    priority = first_mine is not None and (first_other is None or
                                           first_mine <= first_other + CO_DISCOVERY_DAYS * DAY)
    accurate = bool(claim) and claim["novel_nrmse"] is not None and claim["novel_nrmse"] <= thr
    ok = bool(claim) and accurate and not claim["copied"] and priority and claim["independent_consistent"] >= need
    best = min([r["novel_nrmse"] for r in rows if r["novel_nrmse"] is not None] or [float("inf")])
    score = 0.0 if not math.isfinite(best) else float(max(0.0, 1.0 - best / (4 * thr)))
    return {"success": bool(ok), "score": round(score, 5),
            "details": {"papers": rows, "claim": claim["paper"] if claim else None, "threshold": thr,
                        "replications_needed": need, "priority": bool(priority),
                        "best_novel_nrmse": None if not math.isfinite(best) else round(best, 5),
                        "first_accurate_other_t": first_other, "first_accurate_mine_t": first_mine}}


# ====================================================================== the hidden reference (LABELLED SCRIPTED)
def reference_script(target: str = "ph0", reps: int = 1):
    """A tool-only researcher: one precise apparatus (several when they tile the range), calibration standards,
    a grid design, every handbook form and polynomial by BIC, publish; then wait (see `_t2kit.ScriptPolicy`)."""
    def script(pol):
        phs = K.ok((yield {"tool": "sci.phenomena", "args": {}})) or []
        ph = next(p for p in phs if p["id"] == target)
        names = [v["name"] for v in ph["vars"]]
        apps = K.ok((yield {"tool": "sci.apparatus", "args": {"phenomenon": target}})) or []
        full = [a for a in apps if all(abs(a["range"][k][0] - min(b["range"][k][0] for b in apps)) < 1e-9 and
                                       abs(a["range"][k][1] - max(b["range"][k][1] for b in apps)) < 1e-9
                                       for k in names)]
        if full and len(full) >= 3:
            # the two most precise full-range instruments (the third stays free for independent replication)
            use = sorted(full, key=lambda a: (a["stated_sd"] + a["stated_rel"] * 5, a["price_cents"]))[:2]
        elif full:
            use = [min(full, key=lambda a: (a["stated_sd"] + a["stated_rel"] * 5, a["price_cents"]))]
        else:
            use = sorted(apps, key=lambda a: (a["stated_sd"], a["apparatus"]))[:2]
        per = 7 if len(names) == 2 else 4
        runs, cals = [], []
        for a in use:
            axes = [np.linspace(a["range"][k][0], a["range"][k][1], per) for k in names]
            design = [{"at": {k: round(float(v), 4) for k, v in zip(names, c)}, "n": int(reps)}
                      for c in itertools.product(*axes)][:240]
            r = K.ok((yield {"tool": "sci.run", "args": {"apparatus": a["apparatus"], "design": design}}))
            if r:
                runs.append((r["run"], a))
            if ph["standards"]:
                c = K.ok((yield {"tool": "sci.run", "args": {"apparatus": a["apparatus"], "design": [
                    {"standard": s["id"], "n": 6} for s in ph["standards"]]}}))
                if c:
                    cals.append((c["run"], a))
        while True:
            mine = K.ok((yield {"tool": "sci.runs", "args": {}})) or []
            status = {x["run"]: x["status"] for x in mine}
            if all(status.get(r) == "done" for r, _ in runs + cals):
                break
            for _ in range(6):
                yield None
        certified = {s["id"]: float(s["certified"]) for s in ph["standards"]}
        blocks = []
        for r, a in runs:
            readings = list((K.ok((yield {"tool": "sci.data", "args": {"run": r}})) or {}).get("readings") or [])
            for c, ca in cals:
                if ca["apparatus"] == a["apparatus"]:
                    readings += (K.ok((yield {"tool": "sci.data", "args": {"run": c}})) or {}).get("readings") or []
            blocks.append({"points": [x["at"] for x in readings if not x.get("standard")],
                           "raw": np.array([float(x["value"]) for x in readings if not x.get("standard")]),
                           "std_raw": [(certified[x["standard"]], float(x["value"])) for x in readings
                                       if x.get("standard") in certified],
                           "sd": float(a["stated_sd"]), "rel": float(a["stated_rel"])})
        anchors = [(s["at"], float(s["certified"])) for s in ph["standards"]]
        forms = [f for f in TEXTBOOK] + poly_forms(len(names), 2)
        model = fit_joint(forms, blocks, names, anchors, criterion="bic")
        if model is not None:
            yield {"tool": "sci.publish", "args": {"phenomenon": target, "title": f"The response law of {target}",
                                                   "abstract": "Calibrated grid measurements; model chosen by BIC "
                                                               "among the handbook forms and polynomials.",
                                                   "model": model, "data": [r for r, _ in runs],
                                                   "calibration_runs": [c for c, _ in cals]}}
        while True:
            yield None
    return script


def reference(principal: str, obj: dict):
    """The hidden reference policy factory (LABELLED SCRIPTED): tool-only, never handed the World."""
    return K.ScriptPolicy(reference_script(obj.get("phenomenon", "ph0")), label="research-reference")


def null_policy(principal=None, obj=None):
    """The null baseline: never acts."""
    def script(pol):
        while True:
            yield None
    return K.ScriptPolicy(script, label="null")


# ---------------------------------------------------------------------- degenerate strategies (SP-9.16)
def _constant_script(target: str):
    """Measure a small grid on the cheapest apparatus and claim the mean response as the law (a constant)."""
    def script(pol):
        phs = K.ok((yield {"tool": "sci.phenomena", "args": {}})) or []
        ph = next(p for p in phs if p["id"] == target)
        names = [v["name"] for v in ph["vars"]]
        apps = K.ok((yield {"tool": "sci.apparatus", "args": {"phenomenon": target}})) or []
        a = min(apps, key=lambda x: (x["price_cents"], x["apparatus"]))
        design = [{"at": {k: round(float(a["range"][k][0] + f * (a["range"][k][1] - a["range"][k][0])), 4)
                          for k in names}, "n": 1} for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
        r = K.ok((yield {"tool": "sci.run", "args": {"apparatus": a["apparatus"], "design": design}}))
        while r:
            mine = K.ok((yield {"tool": "sci.runs", "args": {}})) or []
            if any(x["run"] == r["run"] and x["status"] == "done" for x in mine):
                break
            for _ in range(4):
                yield None
        d = K.ok((yield {"tool": "sci.data", "args": {"run": r["run"]}})) if r else None
        if d:
            mean = float(np.mean([x["value"] for x in d["readings"]]))
            yield {"tool": "sci.publish", "args": {"phenomenon": target, "title": "A constant response",
                                                   "model": lawlang.fmt(mean), "data": [r["run"]]}}
        while True:
            yield None
    return script


def _copy_script(target: str):
    """Book a small run at once, then republish the first model someone else publishes (a copy)."""
    def script(pol):
        apps = K.ok((yield {"tool": "sci.apparatus", "args": {"phenomenon": target}})) or []
        phs = K.ok((yield {"tool": "sci.phenomena", "args": {}})) or []
        ph = next(p for p in phs if p["id"] == target)
        names = [v["name"] for v in ph["vars"]]
        a = min(apps, key=lambda x: (x["price_cents"], x["apparatus"]))
        design = [{"at": {k: round(float(sum(a["range"][k]) / 2), 4) for k in names}, "n": 3}]
        r = K.ok((yield {"tool": "sci.run", "args": {"apparatus": a["apparatus"], "design": design}}))
        copied = set()
        while r:
            papers = K.ok((yield {"tool": "sci.papers", "args": {"phenomenon": target, "limit": 200}})) or []
            mine = K.ok((yield {"tool": "sci.runs", "args": {}})) or []
            ready = any(x["run"] == r["run"] and x["status"] == "done" for x in mine)
            for p in papers:
                if ready and p["paper"] not in copied and p["author"] != pol.actor:
                    copied.add(p["paper"])
                    yield {"tool": "sci.publish", "args": {"phenomenon": target, "title": "Our law",
                                                           "model": p["model"], "data": [r["run"]]}}
            for _ in range(2):
                yield None
        while True:
            yield None
    return script


def constant_model(principal: str, obj: dict):
    return K.ScriptPolicy(_constant_script(obj.get("phenomenon", "ph0")), label="constant-model")


def copy_model(principal: str, obj: dict):
    return K.ScriptPolicy(_copy_script(obj.get("phenomenon", "ph0")), label="copy-model")


# ====================================================================== mechanics (recorded-event probes)
def _events(world, since: int, prefix: tuple[str, ...] = ("sci.", "bank.txn", "artifact.stored")):
    return [e for e in world.events_range(int(since), world.seq) if e.type.startswith(prefix)]


def _done_runs(world, since: int) -> list[dict]:
    ids_ = {e.payload["id"] for e in world.events_range(int(since), world.seq) if e.type == "sci.completed"}
    return [r for rid, r in world.state.scan("sci_runs") if rid in ids_]


def count_lawful_readings(world, since: int) -> int:
    """Completed runs whose readings follow the hidden law through the apparatus' hidden calibration (median
    absolute residual below half the response spread), provided no public event payload carries a law's text."""
    sci = world.modules["sci"]
    laws = [sci._phen_cfg(pid)["law"] for pid in sci.laws]
    if any(e.vis == "public" and any(law in str(e.payload) for law in laws)
           for e in world.events_range(0, world.seq)):
        return 0
    n = 0
    for r in _done_runs(world, since):
        pts = [p for p in r["points"] if not p.get("standard")]
        if len(pts) < 3:
            continue
        tr = sci.truth(r["phenomenon"], [p["at"] for p in pts])
        vals = np.array([v for p, v in zip(r["points"], r["values"]) if not p.get("standard")])
        a = sci.apps[r["apparatus"]]
        res = np.abs(vals - (float(a.get("gain", 1.0)) * tr + float(a.get("offset", 0.0))))
        n += int(float(np.median(res)) < 0.5 * (float(np.std(tr)) or 1.0))
    return n


def count_noisy_replicates(world, since: int) -> int:
    """Groups of repeated readings at the same condition in one run whose values differ (noise is real)."""
    n = 0
    for r in _done_runs(world, since):
        seen: dict = {}
        for p, v in zip(r["points"], r["values"]):
            seen.setdefault((p.get("standard"), tuple(sorted(p["at"].items()))), []).append(v)
        n += sum(1 for v in seen.values() if len(v) > 1 and float(np.std(v)) > 0)
    return n


def count_experiment_payments(world, since: int) -> int:
    return sum(1 for e in world.events_range(int(since), world.seq)
               if e.type == "bank.txn" and e.payload.get("kind") == "experiment")


def count_datasets(world, since: int) -> int:
    return sum(1 for e in world.events_range(int(since), world.seq)
               if e.type == "artifact.stored" and e.payload.get("kind") == "dataset")


def count_independent_replications(world, since: int) -> int:
    return sum(1 for e in world.events_range(int(since), world.seq)
               if e.type == "sci.replication" and e.payload.get("independent"))


def count_challenge_tests(world, since: int) -> int:
    return sum(len(e.payload["results"]) for e in world.events_range(int(since), world.seq)
               if e.type == "sci.challenge")


MECHANICS = (
    Mechanic("Experimental apparatus", ("sci.booked",), "runs booked on paid, queued, capacity-limited apparatus"),
    Mechanic("Hidden laws", (), "readings generated by the phenomena's hidden laws, never published",
             count=count_lawful_readings),
    Mechanic("Measurement noise", (), "repeated readings at one condition differ (absolute and relative noise)",
             count=count_noisy_replicates),
    Mechanic("Experimental costs", (), "measurements paid to the apparatus operators through the bank",
             count=count_experiment_payments),
    Mechanic("Data collection", ("sci.completed",), "completed runs stored as dataset artifacts",
             count=count_datasets),
    Mechanic("Hypothesis testing", ("sci.paper", "sci.replication"),
             "claimed models tested against data by the generalised chi-square statistic"),
    Mechanic("Independent replication", (), "replication reports by unrelated labs on other apparatus",
             count=count_independent_replications),
    Mechanic("Novel-condition evaluation", ("sci.challenge",),
             "blind challenges outside every bookable range; papers tested against them",
             count=count_challenge_tests),
)
REQUIRED = tuple(m.name for m in MECHANICS)


def mechanics(world, since: int = 0) -> dict:
    """Each required mechanic with its recorded-event count after `since` (the framework's probes) - a readable
    report for tests and notes; the gates use `base.mechanics_exercised`."""
    from . import base
    rep = base.mechanics_exercised(world, DOMAIN, since)
    return {m: K.mechanic(rep["counts"][m] > 0, count=rep["counts"][m]) for m in REQUIRED}


def structure(world) -> dict:
    """The domain's measured structure of a built world (discrete: names and categories, never magnitudes)."""
    cfg = world.modules["sci"].config
    phen = cfg.get("phenomena") or []
    apps = cfg.get("apparatus") or []
    if not phen:
        return {}
    t = phen[0]
    names = [v["name"] for v in t["vars"]]
    tapps = [a for a in apps if a["phenomenon"] == t["id"]]
    layouts = {json.dumps({k: a["range"][k] for k in names}, sort_keys=True) for a in tapps}
    rel_true = [float(a.get("rel", t.get("noise_rel", 0.0))) for a in apps]
    rel_stated = [float(a["stated_rel"]) if a.get("stated_rel") is not None else r for a, r in zip(apps, rel_true)]
    return {"phenomena": {1: "one", 2: "two"}.get(len(phen), "many"), "variables": f"v{len(names)}",
            "apparatus_layout": "tiled" if len(layouts) > 1 else "full",
            "drift": any(float(a.get("drift_per_measure", 0.0)) > 0 for a in apps),
            "datasheet": "optimistic" if any(s < r - 1e-12 for s, r in zip(rel_stated, rel_true)) else "honest",
            "standards": f"s{len(t['standards'])}",
            "law_form_in_handbook": all(str(ph.get("form")) in HANDBOOK_FAMILIES for ph in phen),
            "challenge_bureau": bool(cfg.get("challenge"))}


# the hidden law families whose functional form the public handbook lists (a researcher who knows the form can fit
# it exactly); the structural evaluation configuration draws from families outside it
HANDBOOK_FAMILIES = frozenset({"sat_lin", "thresh_quad", "peak", "power_exp", "sat_exp3", "interact3"})

def tier3(config, merged: dict) -> dict:
    """Inside a civilisation the civilisation's own `sci` configuration stays (its phenomena may be the physics of
    its industry); the configuration adds the standards bureau's challenges (novel-condition evaluation) and the
    population researchers."""
    import copy
    out = copy.deepcopy(merged)
    sci = out["modules"].setdefault("sci", {})
    sci.setdefault("challenge", {"every_days": 10, "start_day": 8, "points": 8})
    out["modules"].setdefault("scipop", {"share": 0.025})
    out["params"].setdefault("target_phenomenon", "ph0")
    return out


CONFIGS = (
    DomainConfig("bench", "train", STRUCTURES["bench"], DOCS["bench"]),
    DomainConfig("drift", "train", STRUCTURES["drift"], DOCS["drift"]),
    DomainConfig("survey", "train", STRUCTURES["survey"], DOCS["survey"]),
    DomainConfig("drift_survey", "eval_compositional", STRUCTURES["drift_survey"], DOCS["drift_survey"]),
    DomainConfig("heldout", "eval_structural", STRUCTURES["heldout"], DOCS["heldout"]),
)

DOMAIN = register(Domain(
    name="research", title="Scientific research", spec_part="9.5", required_mechanics=REQUIRED, configs=CONFIGS,
    build=build, objective=objective, evaluate=evaluate, reference=reference, mechanics=MECHANICS,
    seats=(Seat("researcher", 800_000, "Independent researcher", "a researcher with its own lab budget"),),
    actors=(Actor("population researchers", ("cols.batch:scipop.decide",),
                  "individuals of the population who study, publish and replicate with their own money"),
            Actor("standards bureau", ("sci.challenge",), "blind prediction challenges at novel conditions")),
    null_baseline=null_policy,
    degenerate=(Probe("constant_model", "claim the mean of a few readings as the law", constant_model),
                Probe("copy_model", "republish another lab's model as one's own", copy_model)),
    structure=structure, tier3=tier3, horizon_days=HORIZON_DAYS, slowest_effect_days=10.0, dt=6 * 3600,
    reference_seeds=(0, 1), sources=("research_domain.py", "science.py", "lawlang.py"),
    doc="Phenomena with hidden laws studied by the population's researchers with paid, noisy, capacity-limited "
        "apparatus; papers, independent replication and blind challenges at novel conditions."))
