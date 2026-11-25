"""Shared fitting machinery for calibration workstreams: candidate fitting, the prespecified selection rule,
held-out reports and model cards. Offline code only (scikit-learn / torch are used here, never by `load`)."""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np

from . import metrics as M
from . import predictors as P
from .registry import CRUCIBLE_ROOT, REPORTS_DIR, save_model


# ------------------------------------------------------------------ provenance of the plan and the fit
def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=CRUCIBLE_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def plan_record(rel_path: str) -> dict:
    """{path, sha256, commit} of a committed, unmodified evaluation plan. Refuses an uncommitted or edited plan:
    metrics must be prespecified in version control before a fit."""
    full = os.path.join(CRUCIBLE_ROOT, rel_path)
    if _git("status", "--porcelain", "--", rel_path):
        raise RuntimeError(f"evaluation plan {rel_path} has uncommitted changes; commit it before fitting")
    commit = _git("log", "-1", "--format=%H", "--", rel_path)
    if not commit:
        raise RuntimeError(f"evaluation plan {rel_path} is not committed")
    return {"path": rel_path, "sha256": M_sha(full), "commit": commit,
            "committed_at": _git("log", "-1", "--format=%cI", commit)}


def M_sha(path: str) -> str:
    from .registry import sha256_file
    return sha256_file(path)


def head_commit() -> str:
    return _git("rev-parse", "HEAD")


# ------------------------------------------------------------------ candidates
def fit_constant(y) -> P.Constant:
    return P.Constant(float(np.mean(y)))


def fit_logistic(X, y, C: float = 1.0, sample_weight=None) -> P.Logistic:
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(X, float)
    mean = X.mean(0)
    scale = X.std(0)
    scale[scale < 1e-12] = 1.0
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    lr.fit((X - mean) / scale, y, sample_weight=sample_weight)
    return P.Logistic(lr.coef_[0], lr.intercept_[0], mean, scale)


def fit_gbdt(X, y, seed: int = 0) -> P.TreeEnsemble:
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=200,
                                       l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=20, random_state=seed)
    X = np.asarray(X, float)
    m.fit(X, y)
    te = P.TreeEnsemble.from_sklearn(m)
    k = min(20000, len(X))
    diff = np.max(np.abs(te.predict(X[:k]) - m.predict_proba(X[:k])[:, 1]))
    if diff > 1e-9:
        raise AssertionError(f"exported trees differ from scikit-learn by {diff}")
    te.n_iter = int(m.n_iter_)
    te._sk = m                 # evaluation inside the fitting process uses scikit-learn's C predictor
    return te


def fit_offset(pred, X, y) -> float:
    """One intercept shift on the logit scale minimising log loss (recalibration-in-the-large)."""
    from scipy.optimize import minimize_scalar
    z = pred.logit(X)
    yy = np.asarray(y, float)

    def nll(c):
        return M.log_loss(yy, 1 / (1 + np.exp(-np.clip(z + c, -40, 40))))
    return float(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x)


def n_params(pred) -> int:
    if isinstance(pred, P.Constant):
        return 1
    if isinstance(pred, P.Logistic):
        return len(pred.w) + 1
    if isinstance(pred, P.TreeEnsemble):
        return int((~pred.is_leaf).sum() * 2 + pred.is_leaf.sum())
    if isinstance(pred, P.QuantileNet):
        return int(sum(W.size + b.size for W, b in pred.weights) + pred.head_w.size + pred.head_b.size)
    if isinstance(pred, P.BinnedQuantiles):
        return int(pred.table.size)
    if isinstance(pred, P.GammaPoisson):
        return 2
    return -1


# ------------------------------------------------------------------ quantile networks (torch, CPU)
def fit_quantile_net(Xtr, ytr, Xva, yva, levels, hidden: tuple = (), seed: int = 0, lr: float = 3e-3,
                     batch: int = 4096, epochs: int = 25, patience: int = 3, threads: int = 2) -> tuple:
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    Xtr = np.asarray(Xtr, np.float32)
    mean = Xtr.mean(0)
    scale = Xtr.std(0)
    scale[scale < 1e-12] = 1.0
    Ztr = torch.tensor((Xtr - mean) / scale)
    Zva = torch.tensor((np.asarray(Xva, np.float32) - mean) / scale)
    ytr_t = torch.tensor(np.asarray(ytr, np.float32))
    yva_t = torch.tensor(np.asarray(yva, np.float32))
    tau = torch.tensor(np.asarray(levels, np.float32))
    layers = []
    d = Ztr.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.Tanh()]
        d = h
    body = torch.nn.Sequential(*layers)
    head = torch.nn.Linear(d, len(levels))
    with torch.no_grad():                       # start near the unconditional quantiles
        q0 = np.quantile(np.asarray(ytr, float), levels)
        inc = np.maximum(np.diff(q0), 1e-3)
        head.weight.mul_(0.01)
        head.bias.copy_(torch.tensor(np.r_[q0[0], np.log(np.expm1(inc))], dtype=torch.float32))
    params = list(body.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    def quant(Z):
        raw = head(body(Z))
        return torch.cat([raw[:, :1], raw[:, :1] + torch.cumsum(torch.nn.functional.softplus(raw[:, 1:]), 1)], 1)

    def loss_fn(Q, y):
        e = y[:, None] - Q
        return torch.mean(torch.maximum(tau * e, (tau - 1) * e))

    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, history = np.inf, None, 0, []
    for ep in range(epochs):
        perm = torch.randperm(len(Ztr), generator=gen)
        for i in range(0, len(Ztr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(quant(Ztr[idx]), ytr_t[idx]).backward()
            opt.step()
        with torch.no_grad():
            v = float(loss_fn(quant(Zva), yva_t))
        history.append(v)
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = ([p.detach().clone() for p in params])
        else:
            bad += 1
            if bad >= patience:
                break
    with torch.no_grad():
        for p, s in zip(params, best_state):
            p.copy_(s)
    lin = [m for m in body if isinstance(m, torch.nn.Linear)]
    weights = [(m.weight.detach().numpy().T.astype(float), m.bias.detach().numpy().astype(float)) for m in lin]
    qn = P.QuantileNet(mean, scale, weights, head.weight.detach().numpy().T.astype(float),
                       head.bias.detach().numpy().astype(float), levels,
                       y_min=float(np.min(ytr)), y_max=float(np.max(ytr)))
    with torch.no_grad():
        ref = np.clip(quant(Zva[:5000]).numpy(), qn.y_min, qn.y_max)
    diff = float(np.max(np.abs(qn.predict(np.asarray(Xva, float)[:5000]) - ref)))
    if diff > 1e-3:
        raise AssertionError(f"NumPy quantile network differs from torch by {diff}")
    return qn, {"epochs": len(history), "val_pinball_by_epoch": history}


def fit_binned_quantiles(X, y, col: int, levels, n_bins: int = 10, group_col: int = -1) -> P.BinnedQuantiles:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if n_bins > 1:
        edges = np.unique(np.quantile(X[:, col], np.linspace(0, 1, n_bins + 1)[1:-1]))
    else:
        edges = np.zeros(0)
    B = len(edges) + 1
    groups = [0] if group_col < 0 else sorted(set(X[:, group_col].astype(int).tolist()))
    G = max(groups) + 1
    table = np.zeros((G, B, len(levels)))
    b = np.searchsorted(edges, X[:, col], side="left") if len(edges) else np.zeros(len(X), int)
    g = np.zeros(len(X), int) if group_col < 0 else X[:, group_col].astype(int)
    overall = np.quantile(y, levels)
    for gi in range(G):
        for bi in range(B):
            m = (g == gi) & (b == bi)
            table[gi, bi] = np.quantile(y[m], levels) if m.sum() >= 50 else overall
    return P.BinnedQuantiles(col, edges, table, levels, group_col)


# ------------------------------------------------------------------ selection rule (prespecified)
def select(cands: dict, primary: str, baselines: list[str], lower_is_better: bool = True,
           complexity_order: list[str] | None = None) -> dict:
    """cands: name -> {"val": {...}, "test": {...}}. Returns the selection record of the plan's rule."""
    sign = 1 if lower_is_better else -1
    order = complexity_order or list(cands)
    vals = {k: sign * cands[k]["val"][primary] for k in cands}
    best_val = min(vals.values())
    tied = [k for k in order if vals[k] <= best_val + 1e-4]
    selected = tied[0]
    tests = {k: sign * cands[k]["test"][primary] for k in cands}
    best_base = min(baselines, key=lambda k: tests[k])
    kept = selected
    reason = "selected on validation and confirmed on test: no simple baseline is better on the primary metric"
    if tests[best_base] < tests[selected]:
        kept = best_base
        reason = (f"{selected} was best on validation but the baseline {best_base} is better on test; the baseline "
                  "is kept (see the report's investigation)")
    elif selected in baselines:
        reason = "a simple baseline was best on validation and is kept"
    return {"rule": "argmin validation primary metric (ties to the simpler); keep only if no simple baseline is "
                    "better on test", "primary_metric": primary, "lower_is_better": lower_is_better,
            "validation": {k: cands[k]["val"][primary] for k in cands},
            "test": {k: cands[k]["test"][primary] for k in cands},
            "baselines": list(baselines), "candidates": list(cands),
            "selected_on_validation": selected, "best_baseline_on_test": best_base, "kept": kept,
            "kept_is_baseline": kept in baselines, "reason": reason}


# ------------------------------------------------------------------ reports and cards
def write_report(name: str, report: dict) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    p = os.path.join(REPORTS_DIR, f"{name}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=_j)
        fh.write("\n")
    with open(os.path.join(REPORTS_DIR, f"{name}.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return os.path.relpath(p, CRUCIBLE_ROOT)


def _j(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e4 else f"{v:.4g}"
    return str(v)


def render_markdown(r: dict) -> str:
    lines = [f"# {r['model']}@{r['version']} held-out evaluation report", "",
             f"Domain: {r['domain']}. Dataset: {', '.join(r['datasets'])}. Identification: {r['identification']}.",
             f"Evaluation plan: `{r['plan']['path']}` (sha256 {r['plan']['sha256'][:12]}, commit "
             f"{r['plan']['commit'][:9]}); fit at commit {r['fit']['commit'][:9]} in {r['fit']['seconds']:.0f} s.", "",
             "## Estimand", "", r["estimand"], "", "## Splits", "", "| split | definition | rows | users |",
             "|---|---|---|---|"]
    for k, s in r["splits"].items():
        lines.append(f"| {k} | {s['definition']} | {s.get('rows', '')} | {s.get('users', '')} |")
    prim = r["selection"]["primary_metric"]
    metric_names = sorted({m for c in r["candidates"].values() for m in c["test"]
                           if isinstance(c["test"][m], (int, float)) and m != "n"})
    lines += ["", "## Held-out evaluation", "", "Every candidate on the validation and test splits (primary metric "
              f"`{prim}`, {'lower' if r['selection']['lower_is_better'] else 'higher'} is better).", "",
              "| candidate | complexity | params | val " + prim + " | " + " | ".join(f"test {m}" for m in metric_names)
              + " |", "|---|---|---|---|" + "---|" * len(metric_names)]
    for k, c in r["candidates"].items():
        lines.append(f"| {k} | {c['complexity']} | {c['n_params']} | {_fmt(c['val'][prim])} | "
                     + " | ".join(_fmt(c["test"].get(m, "")) for m in metric_names) + " |")
    s = r["selection"]
    b = r["baseline_comparison"]
    lines += ["", "## Baseline comparison", "", f"Selected on validation: **{s['selected_on_validation']}**; best "
              f"simple baseline on test: **{s['best_baseline_on_test']}**; kept: **{s['kept']}**. {s['reason']}.", "",
              f"Test {prim}: kept {_fmt(b['kept_test'])}, baseline {_fmt(b['baseline_test'])}, difference "
              f"{_fmt(b['difference'])} (95% user-cluster bootstrap interval {_fmt(b['ci95'][0])} to "
              f"{_fmt(b['ci95'][1])}).", "", "## Calibration", ""]
    for k, v in r["calibration"].items():
        if not isinstance(v, (list, dict)):
            lines.append(f"* {k}: {_fmt(v)}")
    if r.get("heldout_extra"):
        lines += ["", "## Further held-out checks", ""]
        for k, v in r["heldout_extra"].items():
            lines.append(f"### {k}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(v, indent=1, default=_j)[:6000])
            lines.append("```")
            lines.append("")
    lines += ["", "## Identification", "", r["identification_note"], "", "## Extrapolation limits", "",
              r["extrapolation_limits"], "", "## Simulator feature mapping", "", "| input | simulator quantity |",
              "|---|---|"]
    for k, v in r["feature_map"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)


def finish(name: str, version: int, domain: str, kept_pred, features: list[str], target: str, report: dict,
           card_extra: dict) -> str:
    """Write the report and the model card + checkpoint of the kept model."""
    rp = write_report(name, report)
    kept = report["selection"]["kept"]
    base = report["selection"]["best_baseline_on_test"]
    card = {
        "domain": domain, "features": features, "target": target,
        "feature_map": report["feature_map"],
        "metrics": report["candidates"][kept]["test"],
        "baseline_metrics": dict(report["candidates"][base]["test"], baseline=base),
        "candidates": {k: {"complexity": c["complexity"], "n_params": c["n_params"], "val": c["val"],
                           "test": c["test"]} for k, c in report["candidates"].items()},
        "selection": report["selection"],
        "uncertainty": report["baseline_comparison"],
        "provenance": {"datasets": report["datasets"], "identification": report["identification"],
                       "population": report["population"], "platform": report["platform"],
                       "period": report["period"], "evaluation_plan": report["plan"], "fit": report["fit"],
                       "domain": domain, "kept_candidate": kept},
        "limits": report["extrapolation_limits"],
        "report": rp,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    card.update(card_extra)
    return save_model(name, version, kept_pred, card)


def feature_quantiles(X, names) -> dict:
    X = np.asarray(X, float)
    qs = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    return {"levels": qs, **{n: np.quantile(X[:, i], qs).round(6).tolist() for i, n in enumerate(names)}}


def binary_candidate(pred, complexity: str, Xva, yva, Xte, yte) -> dict:
    return {"complexity": complexity, "n_params": n_params(pred), "pred": pred,
            "val": M.binary_report(yva, pred.predict(Xva)), "test": M.binary_report(yte, pred.predict(Xte))}


def strip(cands: dict) -> dict:
    return {k: {kk: vv for kk, vv in c.items() if kk != "pred"} for k, c in cands.items()}
