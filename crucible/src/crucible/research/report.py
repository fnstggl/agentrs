"""Reports from protocol results: Markdown tables and PNG figures.

`render(result, out_dir, name)` takes the JSON-serialisable result of any protocol in `research.protocols` (or of
`research.sealed.evaluate`, or one of the experiment bundles `examples/ow_benchmark.py` writes) and produces
`<out_dir>/<name>.md` plus one PNG per quantity. Rules the output keeps:

* one figure per quantity (score, success probability, cost, ...), each with its interval drawn; two quantities
  of different scale never share an axis, and nothing is ever combined into a single score;
* every figure has a table next to it with the same numbers (the figure is never the only way to read a value);
* a learner keeps its colour and marker across every figure (`Styles`): the registry's learners have fixed
  slots, other names take free slots in sorted order, and past eight series the rest are drawn in grey with
  their own markers rather than in invented hues. The categorical palette and its order are the validated
  defaults of the data-visualisation guide this project uses (colour-vision-deficiency separation holds for
  adjacent slots; three slots are below 3:1 contrast on the light surface, which is why markers, a legend and the
  table always accompany them);
* results from sealed sets are headed as such, with the set, generation and access number.

Matplotlib is used with the Agg backend only.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
SURFACE, INK, INK2, MUTED, GRID, AXIS, OTHER = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#898781"
KNOWN_LEARNERS = ("hold", "random", "probe_exploit", "hill_climb", "ppo_gru", "ppo_gru_flat", "model_based",
                  "evo_probe")
ROLES = {"pretrained": 0, "scratch": 1, "trained": 0, "untrained": 1, "source_trained": 0, "original": 1}


class Styles:
    """Entity -> (colour, marker), stable across figures: colour follows the entity, never its rank."""

    def __init__(self, names: Sequence[str]):
        self.slot: dict[str, int] = {}
        names = list(dict.fromkeys(names))
        for n in names:
            if n in KNOWN_LEARNERS:
                self.slot[n] = KNOWN_LEARNERS.index(n)
            elif n in ROLES:
                self.slot[n] = ROLES[n]
        free = [i for i in range(64) if i not in set(self.slot.values())]
        for n in sorted(x for x in names if x not in self.slot):
            self.slot[n] = free.pop(0)

    def __call__(self, name: str) -> tuple[str, str]:
        i = self.slot.get(name, len(PALETTE))
        return (PALETTE[i] if i < len(PALETTE) else OTHER), MARKERS[i % len(MARKERS)]


# ====================================================================== text helpers
def _num(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(v):
        return "-"
    if v != 0 and abs(v) < 10 ** -nd:
        return f"{v:.1e}"
    return f"{v:,.{nd}f}" if abs(v) < 1000 else f"{v:,.0f}"


def _ci(d: Mapping | None, key: str = "mean", nd: int = 3) -> str:
    """`mean [lo, hi]` of an interval dict (marked ~ when it rests on fewer than five runs)."""
    if not d or d.get(key) is None:
        return "-"
    s = f"{_num(d[key], nd)} [{_num(d.get('lo'), nd)}, {_num(d.get('hi'), nd)}]"
    return s + (" ~" if d.get("low_n") else "")


def _p(v: Any) -> str:
    if v is None:
        return "-"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |")
    return out + [""]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "x"


# ====================================================================== figure helpers
def _ax(title: str, xlabel: str, ylabel: str, w: float = 6.6, h: float = 3.9):
    fig, ax = plt.subplots(figsize=(w, h), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.set_title(textwrap.fill(title, int(12 * w)), loc="left", color=INK, fontsize=10.5, pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    return fig, ax


def _save(fig, out_dir: str, stem: str, figs: list[str]) -> str:
    path = os.path.join(out_dir, stem + ".png")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    figs.append(path)
    return os.path.basename(path)


def _legend(ax, n: int) -> None:
    """Two or more series always get a legend: rows between the title and the plot, never on top of the data,
    with as many columns as the labels leave room for."""
    if n < 2:
        return
    _, labels = ax.get_legend_handles_labels()
    n = len(labels)
    budget = 12.5 * ax.figure.get_figwidth()             # characters of 8.5 pt text across the figure
    cols = min(n, 4)
    while cols > 1:
        q, r = divmod(n, cols)                           # matplotlib fills the first r columns with q + 1 entries
        sizes = [q + 1] * r + [q] * (cols - r)
        starts = [sum(sizes[:j]) for j in range(cols)]
        if sum(max(len(x) for x in labels[a:a + k]) + 7 for a, k in zip(starts, sizes)) <= budget:
            break
        cols -= 1
    rows = -(-n // cols)
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                    ncol=cols, handlelength=3.4, columnspacing=1.4, borderaxespad=0.2)
    leg.set_in_layout(False)                             # the title above it reserves the room
    ax.set_title(ax.get_title(loc="left"), loc="left", color=INK, fontsize=10.5, pad=12 + 15 * rows)


def _vals(xs: Sequence) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in xs], dtype=float)


def _curve(ax, x, mean, lo, hi, color: str, marker: str, label: str, dashed: bool = False) -> None:
    x, mean, lo, hi = _vals(x), _vals(mean), _vals(lo), _vals(hi)
    ok = np.isfinite(mean)
    if not ok.any():
        return
    band = ok & np.isfinite(lo) & np.isfinite(hi)
    if band.any():
        ax.fill_between(x[band], lo[band], hi[band], color=color, alpha=0.13, linewidth=0)
    ax.plot(x[ok], mean[ok], color=color, linewidth=2.0, linestyle="--" if dashed else "-", marker=marker,
            markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.4, label=label, solid_capstyle="round")


def _end_labels(ax, series: Sequence[tuple[str, float, float]]) -> None:
    """Name the line ends directly when there are at most four and they are far enough apart to stay attached."""
    series = [(n, x, y) for n, x, y in series if np.isfinite(y)]
    if not 1 < len(series) <= 4:
        return
    y0, y1 = ax.get_ylim()
    ys = sorted(y for _, _, y in series)
    if any(b - a < 0.07 * (y1 - y0) for a, b in zip(ys, ys[1:])):
        return
    for n, x, y in series:
        ax.annotate(n, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2,
                    annotation_clip=False)


def _intervals(out_dir: str, stem: str, figs: list[str], title: str, xlabel: str, rows: Sequence[tuple[str, Mapping]],
               styles: Styles | None = None, key: str = "mean", ref: float | None = None,
               style_keys: Sequence[str] | None = None) -> str | None:
    """Dot and interval per row (the form for 'an estimate with its uncertainty'), rows top to bottom."""
    rows = [(n, d) for n, d in rows if d and d.get(key) is not None]
    if not rows:
        return None
    fig, ax = _ax("", xlabel, "", h=max(1.9, 0.42 * len(rows) + 1.3) + 0.2 * (len(title) > 78))
    fig.suptitle(textwrap.fill(title, 78), x=0.02, ha="left", color=INK, fontsize=10.5)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    for i, (n, d) in enumerate(rows):
        y = len(rows) - 1 - i
        color, marker = styles((style_keys or [r[0] for r in rows])[i]) if styles else (PALETTE[0], "o")
        if d.get("lo") is not None and d.get("hi") is not None:
            ax.plot([d["lo"], d["hi"]], [y, y], color=color, linewidth=2.0, solid_capstyle="round")
        ax.plot([d[key]], [y], marker=marker, markersize=7.5, color=color, markeredgecolor=SURFACE,
                markeredgewidth=1.4, linestyle="none")
    if ref is not None:
        ax.axvline(ref, color=AXIS, linewidth=1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows][::-1], color=INK2, fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return _save(fig, out_dir, stem, figs)


def _img(md: list[str], file: str | None, alt: str) -> None:
    if file:
        md += [f"![{alt}]({file})", ""]


# ====================================================================== shared blocks
def _sealed_header(md: list[str], res: Mapping) -> None:
    s = res.get("sealed")
    if isinstance(s, Mapping) and s.get("sealed"):
        md += [f"> **SEALED TEST SET** `{s['set']}`, generation {s['generation']}, access {s['access']} of "
               f"{s['budget']} (task definition {s.get('task_version', '-')}). Purpose stated at access: "
               f"\"{s.get('purpose', '')}\". Tasks are named by alias; seeds are not recorded.", ""]


def _versions(md: list[str], res: Mapping) -> None:
    v = res.get("versions")
    if v:
        md += [f"Measured under task definition `{v.get('task')}`, mechanism contract `{v.get('mech_contract')}`, "
               f"protocol schema `{v.get('protocols')}`.", ""]


def _pairwise(md: list[str], rows: Sequence[Mapping], alpha: float) -> None:
    if not rows:
        return
    md += [f"Paired differences (a minus b, per task; sign-flip permutation p; Holm-adjusted over the "
           f"{rows[0].get('family_size', len(rows))} comparisons of this table; significant at {alpha}):", ""]
    md += _table(["quantity", "set", "a", "b", "a - b [95% CI]", "p", "p (Holm)", "better", "tasks"],
                 [[r["quantity"], r["set"], r["a"], r["b"], _ci(r, "mean_diff"), _p(r["p_value"]), _p(r.get("p_holm")),
                   r.get("direction") or "not distinguishable", r["n"]] for r in rows])


def _diagnosis(md: list[str], name: str, d: Mapping) -> None:
    if not d:
        return
    if d.get("n_failed"):
        md += [f"**{name}**: {d['n_failed']} of {d['n']} episodes failed. " +
               "; ".join(f"{c}: {v}" for c, v in sorted(d["classes"].items(), key=lambda kv: -kv[1])) + "."]
    else:
        md += [f"**{name}**: no failed episode among {d.get('n', 0)}."]
    for s in d.get("statements", []):
        md += [f"- {s}"]
    for k in d.get("kernel") or []:
        md += [f"- through tool calls on {k['task']}: primary cause `{k['primary']}` ({k['primary_reason']}); "
               f"{k['tool_calls']} calls, score {_num(k['score'])}"]
    md += [""]


COST_HEAD = ["env steps", "updates", "train CPU s", "infer CPU s", "train GPU s", "infer GPU s", "env sim s",
             "infer tokens", "search evals", "sim cost (cents)", "sim net (cents)"]
COST_KEYS = ["env_steps", "updates", "train_cpu_s", "infer_cpu_s", "train_gpu_s", "infer_gpu_s", "env_sim_s",
             "infer_tokens", "search_evals", "sim_cost_cents", "sim_net_cents"]


def _cost_cells(c: Mapping) -> list[str]:
    def one(v):
        if isinstance(v, Mapping):
            v = v.get("mean")
        if isinstance(v, float) and float(v).is_integer() and abs(v) >= 100:
            v = int(v)
        return _num(v, 2)
    return [one(c.get(k)) for k in COST_KEYS]


# ====================================================================== baseline / sealed
def _r_baseline(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str]) -> None:
    per = res["per_learner"]
    st = Styles(list(per))
    sealed = isinstance(res.get("sealed"), Mapping)
    md += [f"# {'Sealed test' if sealed else 'Frozen evaluation'}: {res.get('label') or name}", ""]
    _sealed_header(md, res)
    _versions(md, res)
    md += [f"{res['n_tasks']} tasks, families: {', '.join(res.get('families', []))}. Weights frozen, context reset at "
           "the start of every episode, one episode per task and learner seed, identical tasks for every learner. "
           "Score: 0 = the default settings left alone, 1 = the reference policy. Success: the public objective "
           "was reached. Intervals are over tasks.", ""]
    md += _table(["learner", "seeds", "score mean [95% CI]", "score IQM [95% CI]", "success [Wilson 95%]",
                  "late - early segment", "infer CPU s", "frozen state unchanged"],
                 [[n, len(p.get("seeds", [0])), _ci(p["score"]), _ci(p["iqm"], "iqm"), _ci(p["success"], "p"),
                   _ci(p["adaptation"]), _num(p["cost"]["infer_cpu_s"], 2), _num(p["frozen_unchanged"])]
                  for n, p in per.items()])
    _img(md, _intervals(out_dir, f"{name}__score", figs, "Normalised score, mean over tasks with 95% interval",
                        "score (0 = defaults, 1 = reference)", [(n, p["score"]) for n, p in per.items()], st),
         "score per learner")
    _img(md, _intervals(out_dir, f"{name}__success", figs, "Probability of reaching the public objective (Wilson 95%)",
                        "success probability", [(n, p["success"]) for n, p in per.items()], st, key="p"),
         "success per learner")
    for key, label in (("by_transfer", "relation of the family to the training set"), ("by_family", "family")):
        groups = sorted({g for p in per.values() for g in p[key]})
        if len(groups) < 1:
            continue
        md += [f"Score by {label} (mean [95% CI], n episodes):", ""]
        md += _table(["learner"] + groups, [[n] + [(_ci(p[key][g]["score"]) + f" (n={p[key][g]['n']})")
                                                   if g in p[key] else "-" for g in groups] for n, p in per.items()])
        if key == "by_transfer" and len(groups) > 1:
            rows = [(f"{g} / {n}", p[key][g]["score"]) for g in groups for n, p in per.items() if g in p[key]]
            _img(md, _intervals(out_dir, f"{name}__score_by_transfer", figs,
                                "Score by relation to the training set (95% interval)", "score", rows, st,
                                style_keys=[r[0].split(" / ", 1)[1] for r in rows]), "score by transfer kind")
    k = max((len(p["segments"]) for p in per.values()), default=0)
    if k > 1:
        fig, ax = _ax("Within-episode profile: score in successive segments of the episode",
                      "segment of the episode", "score")
        ends = []
        for n, p in per.items():
            c, m = st(n)
            seg = p["segments"]
            x = np.arange(1, len(seg) + 1)
            _curve(ax, x, [s["mean"] for s in seg], [s["lo"] for s in seg], [s["hi"] for s in seg], c, m, n)
            ends.append((n, x[-1], seg[-1]["mean"] if seg[-1]["mean"] is not None else np.nan))
        ax.set_xticks(range(1, k + 1))
        _legend(ax, len(per))
        _end_labels(ax, ends)
        _img(md, _save(fig, out_dir, f"{name}__adaptation_profile", figs), "within-episode profile")
        md += _table(["learner"] + [f"segment {j + 1}" for j in range(k)],
                     [[n] + [_ci(s) for s in p["segments"]] for n, p in per.items()])
    _pairwise(md, res.get("pairwise", []), res.get("alpha", 0.05))
    md += ["## What the episode records establish about failures", ""]
    for n, p in per.items():
        _diagnosis(md, n, p.get("diagnosis", {}))


# ====================================================================== lifetime
def _points_xy(points: Sequence[Mapping], set_name: str, metric: str):
    e = [p["eval"][set_name][metric] for p in points]
    key = "mean" if metric == "score" else "p"
    return [x[key] for x in e], [x["lo"] for x in e], [x["hi"] for x in e]


def _r_lifetime(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str], heading: str = "#") -> None:
    pts = res["points"]
    lr = res["learner"]
    md += [f"{heading} Learning curve: {lr['name']} (seed {lr['seed']})", ""]
    if heading == "#":
        _versions(md, res)
    cfg = res["config"]
    md += [f"Budget {cfg['budget_env_steps']:,} environment steps, frozen evaluation every {cfg['eval_every']:,}, "
           f"{cfg['n_envs']} parallel environments. Complete: {_num(res['complete'])}. Resumed after an interruption "
           f"at points: {res['resumed'] or 'never'}. Frozen state unchanged by every evaluation: "
           f"{_num(res['frozen_unchanged'])}. Training tasks consumed: {res['train_tasks']['n']:,} "
           f"({', '.join(f'{k}: {v}' for k, v in res['train_tasks']['families'].items())}).", ""]
    for s, d in res["disjoint"].items():
        if not d["ok"]:
            md += [f"**Evaluation set `{s}` overlaps the training tasks ({len(d['overlap'])} tasks): it measures "
                   "performance on seen tasks, not generalisation.**", ""]
    x = [p["env_steps"] for p in pts]
    xc = [p["compute_s"] for p in pts]
    st = Styles(list(pts[0]["eval"]))
    for metric, ylabel, title in (("score", "score (0 = defaults, 1 = reference)", "Frozen score"),
                                  ("success", "success probability", "Probability of reaching the objective")):
        for xs, xlabel, tag in ((x, "environment steps of training", "steps"),
                                (xc, "learner compute, CPU + GPU seconds", "compute")):
            if tag == "compute" and (metric != "score" or xc[-1] <= 0):
                continue
            fig, ax = _ax(f"{title} against {'experience' if tag == 'steps' else 'compute'} "
                          f"(95% interval over tasks)", xlabel, ylabel)
            ends = []
            for s in pts[0]["eval"]:
                c, m = st(s)
                mean, lo, hi = _points_xy(pts, s, metric)
                _curve(ax, xs, mean, lo, hi, c, m, s)
                ends.append((s, xs[-1], mean[-1] if mean[-1] is not None else np.nan))
            _legend(ax, len(pts[0]["eval"]))
            _end_labels(ax, ends)
            _img(md, _save(fig, out_dir, f"{name}__{metric}_vs_{tag}", figs), f"{metric} against {tag}")
    for s in pts[0]["eval"]:
        md += [f"Evaluation set `{s}` ({pts[0]['eval'][s]['n']} tasks):", ""]
        md += _table(["env steps", "updates", "compute s", "score [95% CI]", "IQM", "success [Wilson]",
                      "late - early", "training episodes: score / reward per tick"],
                     [[_num(p["env_steps"]), _num(p["updates"]), _num(p["compute_s"], 2), _ci(p["eval"][s]["score"]),
                       _num(p["eval"][s]["iqm"]["iqm"]), _ci(p["eval"][s]["success"], "p"),
                       _ci(p["eval"][s]["adaptation"]),
                       "-" if not p["train"] else f"{_num(p['train']['score'])} / {_num(p['train']['reward_per_tick'])}"]
                      for p in pts])
        d = res["derived"][s]
        rows = []
        for metric in ("score", "success"):
            m = d[metric]
            if not m.get("n_tasks"):
                continue
            rows.append([f"{metric}: area under the curve against steps (mean height)", _ci(m["auc_steps"])])
            rows.append([f"{metric}: area under the curve against compute (to {_num(m['auc_compute'].get('x_max'), 2)} s)",
                         _ci(m["auc_compute"])])
            for b, v in m["at_budget"].items():
                rows.append([f"{metric} after {int(b):,} steps", _ci(v) + ("" if v.get("reached") else " (not reached)")])
            for t, v in m["to_threshold"].items():
                if v["steps"] is None:
                    txt = f"not reached (in {_num(100 * (v['reached_frac'] or 0), 0)}% of task resamples)"
                else:
                    txt = (f"{_num(v['steps'], 0)} steps [{_num(v.get('lo'), 0)}, {_num(v.get('hi'), 0)}], "
                           f"{_num(v.get('compute_s'), 2)} s compute, {_num(v.get('wall_s'), 1)} s wall")
                rows.append([f"{metric}: steps to {t} held for {v['sustain']} points", txt])
        r = d["simulated_resources"]
        rows.append(["simulated resources in training", f"cost {_num(r['sim_cost_cents'], 0)} cents, net "
                     f"{_num(r['sim_net_cents'], 0)} cents, {r['ticks_fallback']} ticks at fallback, "
                     f"{r['ticks_idle']} idle"])
        md += _table(["derived measure", "value [95% CI over tasks]"], rows)
    fig, ax = _ax("Seconds consumed, by component", "environment steps of training", "seconds (cumulative)")
    comp = [("train_cpu_s", "learner training CPU"), ("infer_cpu_s", "learner inference CPU"),
            ("env_sim_s", "environment simulation")]
    if any(p["train_gpu_s"] or p["infer_gpu_s"] for p in pts):
        comp += [("train_gpu_s", "training GPU (reported)"), ("infer_gpu_s", "inference GPU (reported)")]
    for i, (k, label) in enumerate(comp):
        v = [p[k] for p in pts]
        _curve(ax, x, v, [None] * len(v), [None] * len(v), PALETTE[i], MARKERS[i], label)
    _legend(ax, len(comp))
    _img(md, _save(fig, out_dir, f"{name}__seconds", figs), "seconds by component")
    md += ["Cost of training (cumulative at the last point) and, kept apart, of the frozen evaluations:", ""]
    md += _table([""] + COST_HEAD, [["training"] + _cost_cells(res["cost"]["train"]),
                                    ["evaluation"] + _cost_cells(res["cost"]["eval"])])


def _r_lifetime_set(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str]) -> None:
    runs = res["runs"]
    agg = res.get("aggregate") or {}
    st = Styles(list(runs))
    md += [f"# Learning curves: {res.get('label') or name}", ""]
    _versions(md, next(iter(runs.values()))[0])
    first = next(iter(runs.values()))[0]
    sets = list(first["points"][0]["eval"])
    n_seeds = max(len(v) for v in runs.values())
    md += [f"{len(runs)} learners, {n_seeds} seed(s) each, the same training stream and evaluation sets. "
           + ("Bands are bootstrap intervals over seeds" + (" (fewer than five: indicative only)" if n_seeds < 5 else "")
              if n_seeds > 1 else "Bands are 95% intervals over tasks (one seed).") + "", ""]
    for s in sets:
        for metric, ylabel in (("score", "score (0 = defaults, 1 = reference)"), ("success", "success probability")):
            fig, ax = _ax(f"{'Frozen score' if metric == 'score' else 'Success probability'} on `{s}` against experience",
                          "environment steps of training", ylabel)
            ends = []
            for n, rs in runs.items():
                c, m = st(n)
                if len(rs) > 1 and n in agg:
                    b = agg[n]["sets"][s][metric]
                    _curve(ax, b["x"], b["mean"], b["lo"], b["hi"], c, m, n)
                    ends.append((n, b["x"][-1], b["mean"][-1] if b["mean"] and b["mean"][-1] is not None else np.nan))
                else:
                    mean, lo, hi = _points_xy(rs[0]["points"], s, metric)
                    xs = [p["env_steps"] for p in rs[0]["points"]]
                    _curve(ax, xs, mean, lo, hi, c, m, n)
                    ends.append((n, xs[-1], mean[-1] if mean[-1] is not None else np.nan))
            _legend(ax, len(runs))
            _end_labels(ax, ends)
            _img(md, _save(fig, out_dir, f"{name}__{_slug(s)}__{metric}_vs_steps", figs), f"{metric} on {s}")
        rows = []
        for n, rs in runs.items():
            d = [r["derived"][s] for r in rs]

            def over(get):
                v = [get(x) for x in d]
                v = [float(x) for x in v if x is not None]
                if not v:
                    return "-"
                return _num(float(np.mean(v))) + (f" (sd {_num(float(np.std(v, ddof=1)))} over {len(v)} seeds)"
                                                  if len(v) > 1 else "")
            thr = next(iter(d[0]["score"]["to_threshold"]))
            rows.append([n, len(rs), over(lambda x: x["score"]["final"]["mean"]),
                         over(lambda x: x["success"]["final"]["mean"]), over(lambda x: x["score"]["auc_steps"]["mean"]),
                         over(lambda x: x["score"]["auc_compute"].get("mean")),
                         over(lambda x: x["score"]["to_threshold"][thr]["steps"]),
                         f"{sum(x['score']['to_threshold'][thr]['steps'] is not None for x in d)}/{len(d)}"])
        thr = next(iter(first["derived"][s]["score"]["to_threshold"]))
        md += [f"Evaluation set `{s}`:", ""]
        md += _table(["learner", "seeds", "final score", "final success", "AUC score vs steps", "AUC score vs compute",
                      f"steps to score {thr}", "runs that reached it"], rows)
    md += ["Cost of training per run (mean over seeds):", ""]
    md += _table(["learner"] + COST_HEAD,
                 [[n] + _cost_cells({k: float(np.mean([r["cost"]["train"][k] for r in rs])) for k in COST_KEYS})
                  for n, rs in runs.items()])
    for n, rs in runs.items():
        _r_lifetime(rs[0], out_dir, f"{name}__{_slug(n)}", md, figs, heading="##")


# ====================================================================== timescales
def _r_timescales(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str]) -> None:
    md += [f"# The three timescales of learning: {res['learner']['name']}", ""]
    _versions(md, res)
    a = res["attribution"]
    md += [f"{res['n_tasks']} fresh tasks, {res['budget_env_steps']:,} environment steps of training "
           f"({res['training']['updates']:,} weight updates), context cleared every {res['k_reset']} tick(s) in the "
           f"ablated copy. Training tasks disjoint from evaluation tasks: {_num(res['disjoint']['ok'])} "
           f"({res['disjoint']['checked_training_tasks']} checked). A timescale is attributed when the paired interval "
           f"excludes zero and the Holm-adjusted p over the three tests is below {res['alpha']}.", ""]
    f, p, m = res["fast"], res["parametric"], res["meta"]
    md += _table(["timescale", "measure", "paired difference [95% CI]", "p (Holm)", "present"], [
        ["fast adaptation", "score with context intact minus context cleared (trained checkpoint)",
         _ci(f["trained"]["context"], "mean_diff"), _p(res["p_holm"]["fast"]), _num(a["fast"])],
        ["", "the same at the untrained checkpoint", _ci(f["untrained"]["context"], "mean_diff"), "", ""],
        ["", "late minus early segment, context intact (trained)", _ci(f["trained"]["late_minus_early"]), "", ""],
        ["parametric learning", "frozen score, fresh context: trained minus untrained", _ci(p["gain"], "mean_diff"),
         _p(res["p_holm"]["parametric"]), _num(a["parametric"])],
        ["", "of which without any context", _ci(p["without_context"], "mean_diff"), "", ""],
        ["", "success probability: trained minus untrained", _ci(p["success_gain"], "mean_diff"), "", ""],
        ["meta-learning", "what context carries: trained minus untrained", _ci(m["context_gain"], "mean_diff"),
         _p(res["p_holm"]["meta"]), _num(a["meta"])],
        ["", "rise of the within-episode profile (mean minus first segment)", _ci(m["rise_gain"], "mean_diff"), "", ""],
        ["", "late minus early segment", _ci(m["late_minus_early_gain"], "mean_diff"), "", ""]])
    s2 = m.get("segments_to_untrained_final")
    if s2:
        md += [f"The trained checkpoint is at the untrained checkpoint's final level "
               f"({_num(s2['untrained_final'])}) " + (f"by segment {s2['segment']} of {s2['of']}." if s2["segment"]
                                                      else "in no segment.") + "", ""]

    def pd(d):
        return {"mean": d["mean_diff"], "lo": d["lo"], "hi": d["hi"]}
    _img(md, _intervals(out_dir, f"{name}__fast", figs, "Fast adaptation: what the context carries (paired, 95%)",
                        "score, context intact minus context cleared",
                        [("untrained checkpoint", pd(f["untrained"]["context"])),
                         ("trained checkpoint", pd(f["trained"]["context"]))], ref=0.0), "fast adaptation")
    _img(md, _intervals(out_dir, f"{name}__parametric", figs,
                        "Parametric learning: frozen score, trained minus untrained (paired, 95%)", "score difference",
                        [("fresh context", pd(p["gain"])), ("no context at all", pd(p["without_context"]))], ref=0.0),
         "parametric learning")
    _img(md, _intervals(out_dir, f"{name}__meta", figs,
                        "Meta-learning: change of within-episode adaptation, trained minus untrained (paired, 95%)",
                        "score difference", [("what context carries", pd(m["context_gain"])),
                                             ("rise of the profile", pd(m["rise_gain"]))], ref=0.0), "meta-learning")
    fig, ax = _ax("Within-episode profile at both checkpoints (95% interval over tasks)", "segment of the episode",
                  "score")
    n_series = 0
    for ck in ("trained", "untrained"):
        c, mk = Styles([ck])(ck)
        for cond, dashed in (("segments_intact", False), ("segments_context_reset", True)):
            seg = f[ck][cond]
            x = np.arange(1, len(seg) + 1)
            _curve(ax, x, [s["mean"] for s in seg], [s["lo"] for s in seg], [s["hi"] for s in seg], c, mk,
                   f"{ck}, context {'cleared' if dashed else 'intact'}", dashed)
            n_series += 1
    ax.set_xticks(range(1, len(f["trained"]["segments_intact"]) + 1))
    _legend(ax, n_series)
    _img(md, _save(fig, out_dir, f"{name}__profile", figs), "within-episode profile")
    md += _table(["checkpoint", "context"] + [f"segment {j + 1}" for j in range(len(f["trained"]["segments_intact"]))],
                 [[ck, "cleared" if "reset" in cond else "intact"] + [_ci(s) for s in f[ck][cond]]
                  for ck in ("untrained", "trained") for cond in ("segments_intact", "segments_context_reset")])
    md += ["Training cost:", ""]
    md += _table(COST_HEAD, [_cost_cells(res["training"])])


def _r_timescales_set(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str]) -> None:
    results = res["results"]
    md += [f"# The three timescales of learning: {res.get('label') or name}", ""]
    _versions(md, next(iter(results.values()))[0])
    rows, pts = [], {"fast": [], "parametric": [], "meta": []}
    for n, rs in results.items():
        for r in rs:
            a = r["attribution"]
            q = {"fast": r["fast"]["trained"]["context"], "parametric": r["parametric"]["gain"],
                 "meta": r["meta"]["context_gain"]}
            rows.append([n, r["learner"]["seed"]] + [f"{_ci(q[k], 'mean_diff')} {'(present)' if a[k] else ''}"
                                                     for k in ("fast", "parametric", "meta")] +
                        [_ci(r["parametric"]["without_context"], "mean_diff"), _num(r["training"]["updates"])])
            for k in pts:
                pts[k].append((f"{n} / seed {r['learner']['seed']}",
                               {"mean": q[k]["mean_diff"], "lo": q[k]["lo"], "hi": q[k]["hi"]}, n))
    md += _table(["learner", "seed", "fast: what context carries (trained)", "parametric: frozen gain",
                  "meta: gain in what context carries", "parametric gain without context", "updates"], rows)
    st = Styles(list(results))
    for k, title in (("fast", "Fast adaptation: score with context intact minus context cleared"),
                     ("parametric", "Parametric learning: frozen score, trained minus untrained"),
                     ("meta", "Meta-learning: what context carries, trained minus untrained")):
        _img(md, _intervals(out_dir, f"{name}__{k}", figs, title + " (paired by task, 95%)", "score difference",
                            [(a, b) for a, b, _ in pts[k]], st, ref=0.0, style_keys=[c for _, _, c in pts[k]]), k)
    for n, rs in results.items():
        sub: list[str] = []
        _r_timescales(rs[0], out_dir, f"{name}__{_slug(n)}", sub, figs)
        md += ["#" + x if x.startswith("# ") else x for x in sub]


# ====================================================================== internalisation
def _r_internalisation(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str],
                       heading: str = "#") -> None:
    md += [f"{heading} Parametric internalisation: {res['learner']['name']} (seed {res['seed']})", ""]
    if heading == "#":
        _versions(md, res)
    g = res["gain"]
    md += [f"**Verdict: {res['verdict']}.** Frozen score on {res['n_tasks']} fresh tasks: original weights "
           f"{_ci(res['original']['score'])}, trained checkpoint {_ci(res['trained']['score'])}; paired gain "
           f"{_ci(g, 'mean_diff')}, p={_p(g['p_value'])}, better on {_num(100 * (g.get('frac_better') or 0), 0)}% of "
           f"tasks. Success probability {_ci(res['original']['success'], 'p')} -> {_ci(res['trained']['success'], 'p')}.",
           ""]
    if res["violations"]:
        md += ["**The result is invalid:** " + "; ".join(res["violations"]) + ".", ""]
    for w in res.get("warnings", []):
        md += [f"Warning: {w}.", ""]

    def desc(s: Mapping) -> str:
        n = s["name"]
        if n == "evaluate_original_weights":
            return f"score {_ci(s['score'])}; state unchanged by evaluation: {_num(s['frozen_unchanged'])}"
        if n == "learning_interactions":
            return (f"{s['env_steps']:,} environment steps in {s['episodes']:,} episodes on "
                    f"{s['training_tasks']['n']:,} tasks; inference {_num(s['infer_cpu_s'], 2)} CPU s, "
                    f"{s['infer_tokens']:,} tokens, {s['search_evals']:,} search evaluations")
        if n == "update_weights":
            return f"{s['updates']:,} updates; training {_num(s['train_cpu_s'], 2)} CPU s, {_num(s['train_gpu_s'], 2)} GPU s"
        if n == "save_checkpoint":
            return (f"hash {str(s['checkpoint_hash'])[:12]} ({s['hash_method']}, {s['bytes']} bytes); differs from the "
                    f"original: {_num(s['weights_changed_by_training'])}")
        if n == "clear_task_context":
            return f"{s['method']}; checkpoint round-trips: {_num(s['checkpoint_round_trips'])}"
        if n == "disable_retrieval":
            return f"{s['call']}; frozen flag set: {_num(s['frozen_flag'])}"
        if n == "evaluate_on_fresh_tasks":
            d = s["disjoint_from_training"]
            rp = s.get("repeat") or {}
            return (f"score {_ci(s['score'])}; disjoint from the {d['checked_training_tasks']} training tasks by "
                    f"{d['by']}: {_num(d['ok'])}; state unchanged by evaluation: {_num(s['frozen_unchanged'])}; "
                    f"evaluation reproduces itself: {_num(rp.get('identical'))}")
        return f"score {_ci(s['score_gain'], 'mean_diff')}, success {_ci(s['success_gain'], 'mean_diff')}"
    md += _table(["step", "what", "recorded"], [[s["step"], s["name"].replace("_", " "), desc(s)] for s in res["steps"]])
    c = res["compute"]
    md += _table(["updates", "environment interactions", "training CPU s", "training GPU s", "inference CPU s",
                  "inference GPU s", "environment simulation s", "tokens", "search evaluations"],
                 [[_num(res["update_count"]), _num(res["env_interactions"]), _num(c["train_cpu_s"], 2),
                   _num(c["train_gpu_s"], 2), _num(c["infer_cpu_s"], 2), _num(c["infer_gpu_s"], 2),
                   _num(c["env_sim_s"], 2), _num(c["infer_tokens"]), _num(c["search_evals"])]])
    bt = res["transfer_performance"]["by_transfer"]
    md += ["Trained checkpoint by relation of the family to the training set:", ""]
    md += _table(["relation", "episodes", "score [95% CI]", "success [Wilson]"],
                 [[k, v["n"], _ci(v["score"]), _ci(v["success"], "p")] for k, v in bt.items()])
    x, y = _vals(res["original"]["scores"]), _vals(res["trained"]["scores"])
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.any():
        fig, ax = _ax("Each fresh task, before and after training",
                      "frozen score, original weights", "frozen score, trained checkpoint", w=4.8, h=4.4)
        lo, hi = float(min(x[ok].min(), y[ok].min())), float(max(x[ok].max(), y[ok].max()))
        pad = 0.05 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=AXIS, linewidth=1.0)
        ax.plot(x[ok], y[ok], linestyle="none", marker="o", markersize=7, color=PALETTE[0], markeredgecolor=SURFACE,
                markeredgewidth=1.4, alpha=0.8)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.annotate("no change", (hi + pad, hi + pad), xytext=(-4, -12), textcoords="offset points", ha="right",
                    fontsize=8.5, color=MUTED)
        _img(md, _save(fig, out_dir, f"{name}__paired", figs), "paired scores per task")
    _img(md, _intervals(out_dir, f"{name}__gain", figs, "Internalised gain on fresh tasks (paired, 95%)",
                        "frozen score, trained minus original",
                        [("score", {"mean": g["mean_diff"], "lo": g["lo"], "hi": g["hi"]})], ref=0.0), "paired gain")
    r = res.get("retention")
    if r:
        ff, k = r["first_family"], r["kept_fraction"]
        md += [f"{heading}# Retention", "",
               f"After {r['second_training']['env_steps']:,} further environment steps on a second family "
               f"({r['second_training']['updates']:,} updates), the first family was evaluated again: "
               + (f"{_num(100 * k['ratio'], 1)}% [{_num(100 * k['lo'], 1)}, {_num(100 * k['hi'], 1)}] of the first gain is kept"
                  if k.get("defined") else f"kept fraction undefined ({k.get('reason')})")
               + f"; forgetting (score lost) {_ci(r['forgetting'], 'mean_diff')}.", ""]
        rows = [("original weights", ff["original"]), ("after training on the first family", ff["after_first_training"]),
                ("after training on the second family", ff["after_second_training"])]
        md += _table(["first family, frozen score", "mean [95% CI]"], [[n, _ci(d)] for n, d in rows])
        if r.get("second_family"):
            sf = r["second_family"]
            md += _table(["second family, frozen score", "mean [95% CI]"],
                         [["before its training", _ci(sf["before"])], ["after its training", _ci(sf["after"])],
                          ["paired gain", _ci(sf["gain"], "mean_diff")]])
        _img(md, _intervals(out_dir, f"{name}__retention", figs, "First family, frozen score at three stages (95%)",
                            "score", rows), "retention")


def _r_internalisation_set(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str]) -> None:
    results = res["results"]
    md += [f"# Parametric internalisation: {res.get('label') or name}", ""]
    _versions(md, next(iter(results.values()))[0])
    rows, pts = [], []
    for n, rs in results.items():
        for r in rs:
            g = r["gain"]
            rows.append([n, r["seed"], r["verdict"], _ci(r["original"]["score"]), _ci(r["trained"]["score"]),
                         _ci(g, "mean_diff"), _p(g["p_value"]), _num(r["update_count"]), _num(r["env_interactions"]),
                         _num(r["compute"]["compute_s"], 2), _num(r["valid"])])
            pts.append((f"{n} / seed {r['seed']}", {"mean": g["mean_diff"], "lo": g["lo"], "hi": g["hi"]}, n))
    md += _table(["learner", "seed", "verdict", "original score", "trained score", "paired gain [95% CI]", "p",
                  "updates", "env interactions", "compute s", "valid"], rows)
    st = Styles(list(results))
    _img(md, _intervals(out_dir, f"{name}__gain", figs, "Internalised gain on fresh tasks (paired by task, 95%)",
                        "frozen score, trained minus original", [(a, b) for a, b, _ in pts], st, ref=0.0,
                        style_keys=[c for _, _, c in pts]), "internalised gain per learner")
    for n, rs in results.items():
        _r_internalisation(rs[0], out_dir, f"{name}__{_slug(n)}", md, figs, heading="##")


# ====================================================================== transfer
def _r_transfer(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str], heading: str = "#") -> None:
    lr = (res.get("learner") or {}).get("name", "learner")
    md += [f"{heading} Transfer: {lr}", ""]
    if heading == "#":
        _versions(md, res)
    md += [f"Source: {', '.join(res['source'].get('families') or ['given tasks'])}, "
           f"{res['source']['budget_env_steps']:,} environment steps; seeds {res['seeds']}. Zero-shot: the "
           "source-trained checkpoint, frozen, against the untrained learner on the same tasks. Adaptation: the same "
           "interaction budget on the target, starting from the source-trained checkpoint or from scratch (same seeds, "
           "same tasks). Efficiency above 1 means the source helped.", ""]
    if res["negative_transfer"] or res["negative_zero_shot"]:
        md += [f"**Negative transfer** in adaptation: {res['negative_transfer'] or 'none'}; zero-shot: "
               f"{res['negative_zero_shot'] or 'none'}.", ""]
    rows, zs_rows, eff_rows = [], [], []
    st = Styles(["pretrained", "scratch"])
    for tname, t in res["targets"].items():
        z, ad = t["zero_shot"], t["adaptation"]
        row = [tname, t["category"], t["n_pairs"], _ci(z["source_trained"]["score"]), _ci(z["untrained"]["score"]),
               _ci(z["gain"], "mean_diff")]
        zs_rows += [(f"{tname}: source-trained", z["source_trained"]["score"], "pretrained"),
                    (f"{tname}: untrained", z["untrained"]["score"], "scratch")]
        if ad:
            ra, sp = ad["auc"]["ratio"], ad["steps_to_threshold"]
            ratio = _ci(ra, "ratio") if ra.get("defined") else f"undefined ({ra.get('reason', '')})"
            if sp["ratio"] is not None:
                steps = f"{_num(sp['ratio'], 2)} [{_num(sp.get('lo'), 2)}, {_num(sp.get('hi'), 2)}]"
            else:
                steps = sp.get("note") or "-"
            row += [_ci(ad["auc"]["pretrained"]), _ci(ad["auc"]["scratch"]), _ci(ad["auc"]["diff"], "mean_diff"), ratio,
                    steps, "NEGATIVE" if t["negative_transfer"] else "no"]
            if ra.get("defined"):
                eff_rows.append((tname, {"mean": ra["ratio"], "lo": ra["lo"], "hi": ra["hi"]}))
            fig, ax = _ax(f"Adaptation on `{tname}` ({t['category']}): frozen score against target experience",
                          "environment steps on the target", "score")
            ends = []
            for arm in ("pretrained", "scratch"):
                c, m = st(arm)
                cv = ad["curves"][arm]
                _curve(ax, cv["x"], cv["mean"], cv["lo"], cv["hi"], c, m,
                       "from the source-trained checkpoint" if arm == "pretrained" else "from scratch")
                ends.append((arm, cv["x"][-1], cv["mean"][-1]))
            _legend(ax, 2)
            _end_labels(ax, ends)
            _img(md, _save(fig, out_dir, f"{name}__adaptation_{_slug(tname)}", figs), f"adaptation on {tname}")
        else:
            row += ["-"] * 5 + ["zero-shot only"]
        rows.append(row)
    md += _table(["target", "category", "pairs (seed x task)", "zero-shot, source-trained", "zero-shot, untrained",
                  "zero-shot gain [95% CI]", "AUC from checkpoint", "AUC from scratch", "AUC difference",
                  "AUC ratio (efficiency)", f"steps-to-{res['threshold']} ratio", "negative transfer"], rows)
    _img(md, _intervals(out_dir, f"{name}__zero_shot", figs, "Zero-shot frozen score on each target (95%)", "score",
                        [(a, b) for a, b, _ in zs_rows], st, style_keys=[c for _, _, c in zs_rows]), "zero-shot")
    _img(md, _intervals(out_dir, f"{name}__efficiency_auc", figs,
                        "Transfer efficiency: ratio of areas under the adaptation curves (95%)",
                        "area from the source-trained checkpoint / area from scratch", eff_rows, ref=1.0),
         "transfer efficiency")


def _r_transfer_set(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str]) -> None:
    md += [f"# Transfer: {res.get('label') or name}", ""]
    results = res["results"]
    _versions(md, next(iter(results.values())))
    rows = []
    for n, r in results.items():
        for tname, t in r["targets"].items():
            ad = t["adaptation"]
            ra = ad["auc"]["ratio"] if ad else {}
            rows.append([n, tname, t["category"], _ci(t["zero_shot"]["gain"], "mean_diff"),
                         _ci(ra, "ratio") if ra.get("defined") else "-",
                         "-" if t["negative_transfer"] is None else ("NEGATIVE" if t["negative_transfer"] else "no")])
    md += _table(["learner", "target", "category", "zero-shot gain", "AUC ratio (efficiency)", "negative transfer"], rows)
    for n, r in results.items():
        _r_transfer(r, out_dir, f"{name}__{_slug(n)}", md, figs, heading="##")


# ====================================================================== compare
def _r_compare(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str]) -> None:
    per = res["per_learner"]
    st = Styles(res["learners"])
    md += [f"# Comparison under equal budgets: {res.get('label') or name}", ""]
    _versions(md, res)
    ch = res["checks"]
    md += [f"{len(res['learners'])} learners, seeds {res['seeds']}, {int(res['budget_env_steps']):,} environment steps "
           f"each. Identical evaluation tasks: {_num(ch['same_eval_tasks'])}; identical training tasks: "
           f"{_num(ch['same_train_tasks'])}; identical step grid: {_num(ch['same_step_grid'])}; all runs complete: "
           f"{_num(ch['all_complete'])}; frozen state unchanged by evaluations: {_num(ch['frozen_unchanged'])}. "
           f"Statistics are paired over tasks (mean over seeds per task) and Holm-corrected.", ""]
    if not ch.get("equal_budgets", True):
        md += ["**The runs did not consume identical training tasks on an identical schedule: budgets are not equal.**", ""]
    for s in res["sets"]:
        md += [f"## Evaluation set `{s}`", ""]
        md += _table(["learner", "final score [95% CI]", "final IQM", "final success [Wilson 95%, all episodes]",
                      "AUC score vs steps", "AUC success vs steps", "sd of the final score over seeds"],
                     [[n, _ci(p["final"][s]["score"]), _ci(p["final"][s]["iqm"], "iqm"),
                       _ci(p["final"][s]["success_pooled"], "p"), _ci(p["auc_steps"][s]), _ci(p["auc_success_steps"][s]),
                       _num(p["final"][s]["seed_sd"])] for n, p in per.items()])
        for metric, ylabel in (("score", "score (0 = defaults, 1 = reference)"), ("success", "success probability")):
            fig, ax = _ax(f"Frozen score on `{s}` at equal experience (95% interval over tasks)" if metric == "score"
                          else f"Success probability on `{s}` at equal experience (Wilson 95%, all episodes)",
                          "environment steps of training", ylabel)
            ends = []
            for n, p in per.items():
                c, m = st(n)
                cv = p["curves"][s][metric]
                _curve(ax, cv["x"], cv["mean"], cv["lo"], cv["hi"], c, m, n)
                ends.append((n, cv["x"][-1], cv["mean"][-1] if cv["mean"][-1] is not None else np.nan))
            _legend(ax, len(per))
            _end_labels(ax, ends)
            _img(md, _save(fig, out_dir, f"{name}__{_slug(s)}__{metric}_vs_steps", figs), f"{metric} vs steps")
        if res["equal_compute"]["measurable"]:
            fig, ax = _ax(f"Frozen score on `{s}` against learner compute", "learner compute, CPU + GPU seconds", "score")
            for n, p in per.items():
                c, m = st(n)
                cv = p["curves"][s]["score"]
                _curve(ax, p["curves"][s]["compute_s"], cv["mean"], cv["lo"], cv["hi"], c, m, n)
            ax.axvline(res["equal_compute"]["budget_s"], color=AXIS, linewidth=1.0)
            _legend(ax, len(per))
            _img(md, _save(fig, out_dir, f"{name}__{_slug(s)}__score_vs_compute", figs), "score vs compute")
        _img(md, _intervals(out_dir, f"{name}__{_slug(s)}__final_score", figs,
                            f"Final frozen score on `{s}` at equal experience (95%)", "score",
                            [(n, p["final"][s]["score"]) for n, p in per.items()], st), "final score")
        _img(md, _intervals(out_dir, f"{name}__{_slug(s)}__auc_steps", figs,
                            f"Area under the learning curve on `{s}` against steps (95%)", "mean height of the curve",
                            [(n, p["auc_steps"][s]) for n, p in per.items()], st), "area under the curve")
    eq = res["equal_compute"]
    md += ["## Equal compute", ""]
    if eq["measurable"]:
        md += [f"Every learner is read at {_num(eq['budget_s'], 3)} s of learner compute (the smallest total any run "
               "used, so nobody is extrapolated).", ""]
        md += _table(["learner", "total compute s"] + [f"score on {s} at the budget" for s in res["sets"]] +
                     ["env steps it had used by then"],
                     [[n, _num(eq["totals_s"][n], 3)] + [_ci(eq["per_learner"][n][s]) for s in res["sets"]] +
                      [_num(eq["per_learner"][n][res["sets"][0]]["env_steps_at_budget"], 0)] for n in res["learners"]])
    else:
        md += ["Learner compute was not measurable for at least one learner (no CPU time registered): no "
               "equal-compute comparison is made.", ""]
    md += ["## Costs (mean per run)", ""]
    md += _table(["learner"] + COST_HEAD, [[n] + _cost_cells(p["cost"]) for n, p in per.items()])
    md += ["## Paired statistics", ""]
    _pairwise(md, res["pairwise"], res["alpha"])
    md += ["## What the records establish about failures", ""]
    for n in res["learners"]:
        for s in res["sets"]:
            _diagnosis(md, f"{n} on {s}", res["diagnosis"][n][s])


def _r_evaluate(res: Mapping, out_dir: str, name: str, md: list[str], figs: list[str]) -> None:
    a = res["aggregate"]
    md += [f"# Frozen evaluation: {res['learner']['name']}", ""]
    _versions(md, res)
    md += _table(["tasks", "score mean [95% CI]", "IQM", "success [Wilson]", "late - early", "state unchanged"],
                 [[res["n_tasks"], _ci(a["score"]), _ci(a["iqm"], "iqm"), _ci(a["success"], "p"), _ci(a["adaptation"]),
                   _num(res["frozen"]["unchanged"])]])
    for key in ("by_transfer", "by_family"):
        md += _table([key.replace("by_", ""), "episodes", "score [95% CI]", "success [Wilson]"],
                     [[g, v["n"], _ci(v["score"]), _ci(v["success"], "p")] for g, v in res[key].items()])
    _img(md, _intervals(out_dir, f"{name}__score_by_family", figs, "Frozen score by family (95%)", "score",
                        [(g, v["score"]) for g, v in res["by_family"].items()]), "score by family")
    _diagnosis(md, res["learner"]["name"], _safe_diag(res))


def _safe_diag(res: Mapping) -> dict:
    from .protocols import diagnose_records
    return diagnose_records(res["records"])


_RENDERERS = {"baseline": _r_baseline, "sealed": _r_baseline, "lifetime": _r_lifetime, "lifetime_set": _r_lifetime_set,
              "timescales": _r_timescales, "timescales_set": _r_timescales_set, "internalisation": _r_internalisation,
              "internalisation_set": _r_internalisation_set, "transfer": _r_transfer, "transfer_set": _r_transfer_set,
              "compare": _r_compare, "evaluate": _r_evaluate}


def render(result: Mapping | str, out_dir: str, name: str | None = None) -> dict:
    """Write the report of one protocol result (a dict, or the path of its JSON file) into `out_dir`.
    Returns `{"markdown": path, "figures": [paths]}`."""
    if isinstance(result, str):
        name = name or os.path.splitext(os.path.basename(result))[0]
        with open(result) as fh:
            result = json.load(fh)
    proto = result.get("protocol")
    if proto not in _RENDERERS:
        raise ValueError(f"no report for protocol {proto!r}; known: {sorted(_RENDERERS)}")
    name = _slug(name or proto)
    os.makedirs(out_dir, exist_ok=True)
    md: list[str] = []
    figs: list[str] = []
    _RENDERERS[proto](result, out_dir, name, md, figs)
    path = os.path.join(out_dir, name + ".md")
    with open(path, "w") as fh:
        fh.write("\n".join(md).rstrip() + "\n")
    return {"markdown": path, "figures": figs}
