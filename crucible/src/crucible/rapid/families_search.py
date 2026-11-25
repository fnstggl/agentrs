"""Families on the `search` engine: structured search, informative versus distracting features, discoverable tools
and a two-participant search that needs communication and joint commitment."""
from __future__ import annotations

import numpy as np

from ..mechanisms.base import neutral_names
from .families import family

C = "crucible.rapid.claims:"
I = "crucible.rapid.identify:"


def u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


@family("srch_field", engine="search", category="8.5", item="8.5.2", split="train",
        doc="A grid of sites whose values form a few smooth hidden peaks. Probes return noisy values cheaply; "
            "digging collects a site's value once.",
        distinct_because="spatially structured hidden values (smooth bumps) revealed by costly noisy probes and "
                         "exhausted by commitment",
        axes={"graph_size": 8, "hidden_state_complexity": 3, "experiment_cost": 40.0, "observation_noise": 1.0,
              "horizon": 30}, objective="maximize_net", claims=(C + "field_structure",), identify=I + "field_peak")
def srch_field(rng, d):
    G, nb = int(d["graph_size"]), int(d["hidden_state_complexity"])
    xs, ys = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    v = np.zeros((G, G))
    bumps = []
    for _ in range(nb):
        cx, cy = u(rng, 0, G - 1), u(rng, 0, G - 1)
        amp, w = u(rng, 600, 2000), u(rng, 0.8, 1.8)
        v += amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * w * w))
        bumps.append([cx, cy, amp, w])
    names = neutral_names(rng, 1)
    return {"engine": "search", "scenario": "field", "horizon": int(d["horizon"]), "G": G,
            "values": np.round(v, 6).tolist(), "bumps": bumps, "probe_cost": float(d["experiment_cost"]),
            "probe_noise": 100.0 * float(d["observation_noise"]), "dig_cost": 400.0, "gauge_names": names,
            "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise what you dig up minus what probing and digging "
                                                          "cost."},
            "brief": f"A {G}x{G} grid of sites named c<x>_<y>. Probe a site to get a noisy estimate of its value; dig "
                     "a site to collect its value (once)."}


@family("srch_distractors", engine="search", category="8.5", item="8.5.6", split="val",
        doc="Items with public features of which only two determine value; the rest are distractors, some of them "
            "strikingly variable. Probes reveal noisy values.",
        distinct_because="value is linear in a hidden subset of public features; salient high-variance distractor "
                         "features carry no information (useful discovery versus novelty)",
        axes={"action_complexity": 20, "n_variables": 6, "experiment_cost": 40.0, "horizon": 25},
        objective="maximize_net", claims=(C + "distractor_features",), identify=I + "informative_features")
def srch_distractors(rng, d):
    N, D = int(d["action_complexity"]), max(3, int(d["n_variables"]))
    inf = sorted(int(i) for i in rng.choice(D, size=2, replace=False))
    w = [u(rng, 0.7, 1.5) * (1 if rng.random() < 0.5 else -1) for _ in inf]
    F = rng.normal(0, 1, (N, D))
    for j in range(D):
        if j not in inf and j % 2 == 1:
            F[:, j] *= 3.0                                # novel, salient, uninformative
    vals = F[:, inf] @ np.asarray(w) + rng.normal(0, 0.3, N)
    names = neutral_names(rng, N)
    return {"engine": "search", "scenario": "distractors", "horizon": int(d["horizon"]), "item_names": names,
            "features": np.round(F, 6).tolist(), "values": np.round(vals, 6).tolist(), "informative": inf,
            "weights": w, "scale": 600.0, "probe_cost": float(d["experiment_cost"]), "probe_noise": 0.5,
            "invest_cost": 300.0, "gauge_names": [], "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise the returns of the items you invest in minus "
                                                          "costs."},
            "brief": f"{N} items, each with {D} public features. Probe an item for a noisy estimate of its return; "
                     "invest in an item to collect its return (once)."}


@family("srch_tools", engine="search", category="8.5", item="8.5.3", split="train",
        doc="Tools are found one at a time in a catalogue (at a cost); each works only on a hidden subset of job "
            "types. Jobs of observable types arrive every tick.",
        distinct_because="discoverable actions: the action space grows as tools are found; tool-by-job "
                         "applicability is a hidden interaction",
        axes={"action_complexity": 6, "hidden_state_complexity": 4, "horizon": 60}, objective="maximize_net",
        claims=(C + "tool_applicability",), identify=I + "tool_map")
def srch_tools(rng, d):
    T, J = int(d["action_complexity"]), int(d["hidden_state_complexity"])
    names = neutral_names(rng, T + J + 2)
    app = np.zeros((T, J))
    for k in range(T):
        n = int(rng.integers(0, J + 1))
        app[k, rng.choice(J, size=n, replace=False)] = 1.0
    for j in range(J):
        if app[:, j].sum() == 0:
            app[int(rng.integers(T)), j] = 1.0
    return {"engine": "search", "scenario": "tools", "horizon": int(d["horizon"]), "tool_names": names[:T],
            "job_names": names[T:T + J], "app": app.astype(int).tolist(), "order": [int(x) for x in rng.permutation(T)],
            "lookup_cost": 150.0, "use_cost": 30.0, "payoff": 300.0, "p_success": 0.9,
            "gauge_names": names[T + J:], "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Maximise your net income from jobs done."},
            "brief": "Jobs arrive one per tick; each needs a tool that works on its type. You start with no tools; "
                     "look them up in the catalogue (query target 'catalogue')."}


@family("coop_split_info", engine="search", category="8.9", item="8.9.2", split="train",
        doc="Two participants search one grid for a hidden target; each sees a noisy signal of only one coordinate. "
            "Payment needs both to commit jointly to the same cell, so they must share what they see.",
        distinct_because="information distributed across two participants with joint commitment (coordinate) and "
                         "communication (message); neither can succeed alone",
        axes={"graph_size": 6, "observation_noise": 0.7, "horizon": 40}, objective="maximize_net",
        participants=("teammate:shared_payoff",), claims=(C + "needs_both",), identify=I + "coop_target")
def coop_split_info(rng, d):
    G = int(d["graph_size"])
    names = neutral_names(rng, 2)
    return {"engine": "search", "scenario": "coop", "horizon": int(d["horizon"]), "G": G, "agents": ["a0", "a1"],
            "signal_noise": float(d["observation_noise"]), "try_cost": 50.0, "payoff": 2000.0, "gauge_names": names,
            "endowment": 50_000,
            "objective": {"type": "maximize_net", "text": "Find the hidden cell with your partner: a joint "
                                                          "commitment to the right cell pays both of you."},
            "brief": f"A {G}x{G} grid (cells c<x>_<y>). You see a noisy signal of one coordinate of a hidden target; "
                     "your partner sees the other. Message each other and commit jointly (coordinate)."}
