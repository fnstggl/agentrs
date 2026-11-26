"""Evaluator-side learning experiments on the population's class-2 adaptive participants (Gate G, SP-3.4).

`adaptation(seed, phase_days, arms)` builds one world of the one population with 10 hand-specified web sites (5 rich,
2 broken, 2 thin, 1 spam), 150 lightweight individuals and 256 adaptive ones split into policy groups (`arms`:
{name: group config}); groups with the same `seed` start from IDENTICAL weights. Phase 1 lasts `phase_days`; then the
controlled new condition: the two most visited rich sites are redeployed as broken and the two broken ones as rich
(their operators' ordinary tools); phase 2 lasts `phase_days`. Measured per group from the recorded log: weight hashes
at genesis / switch / end, policy updates and optimizer steps, held-out value-prediction error on examples never used
for training (learned vs the initial weights), the realised utility of the group's web visits in each phase with a
bootstrap CI of the difference between arms, and the share of visits to failing sites.

Scripted parts (labelled in evidence): the site specifications and the switch are the experimenter's; every visit,
choice, utility and gradient step is the population's own.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from ..core.kernel import World
from ..substrate import artifacts, columns
from . import analysis as A
from . import sites as SITES
from .worlds import population_world

DAY = 86400
TOPICS = ["cooking", "cycling", "gardening", "coding", "music", "fitness", "travel", "finance", "games", "learning"]
BEHAVIOURS = ["rich", "rich", "rich", "rich", "rich", "broken", "broken", "thin", "thin", "spam"]


def _site_specs(seed: int):
    rng = np.random.default_rng(100 + seed)
    names = [f"s{k}{t[:5]}" for k, t in enumerate(TOPICS)]
    specs = []
    for k, (t, b, nm) in enumerate(zip(TOPICS, BEHAVIOURS, names)):
        links = [names[(k + j) % 10] for j in (1, 2, 3)]
        specs.append(SITES.make_spec(rng, nm, t, b, n_articles=8, n_products=3 if k % 3 == 0 else 0, links=links))
    return names, specs


def redeploy(ctx, host: str, spec: dict, behaviour: str) -> None:
    st = ctx.state
    dep = st.get("deployments", st.get("hosts", host)["deployment"])
    person = st.get("orgs", dep["owner"])["founder"]
    acct = ctx.world.modules["bank"].primary_account(st, dep["owner"])
    sp = copy.deepcopy(spec)
    sp["behaviour"] = behaviour
    ctx.world.modules["code"].commit_files(ctx, dep["repo"], SITES.app_files(sp, acct), "site update", person,
                                           trusted=True)
    ctx.call("cloud.deploy", {"repo": dep["repo"], "host": host, "instance": dep["instance"]}, as_principal=person)


def _boot(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    if a.size == 0 or b.size == 0:
        return {"mean": 0.0, "ci95": [0.0, 0.0]}
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean() for _ in range(n)]
    return {"mean": float(a.mean() - b.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def adaptation(seed: int, phase_days: int = 5, arms: dict | None = None, *, n_light: int = 150,
               min_examples: int = 32) -> dict:
    """See the module docstring. `arms` maps an arm name to a group config ({id, share, learning, memory, seed})."""
    import torch
    torch.set_num_threads(1)
    arms = arms or {"learning": {"id": 0, "share": 0.5, "seed": 5},
                    "frozen": {"id": 1, "share": 0.5, "seed": 5, "learning": False}}
    names, specs = _site_specs(seed)
    spec = population_world("development", seed, n=n_light, site_specs=specs, tick_s=1800,
                            classes={"adaptive": 256, "strategic": 0, "semantic": 0}, strategic=False,
                            adaptive={"groups": list(arms.values()), "min_examples": min_examples})
    t0 = time.time()
    w = World.create(spec)
    ad = w.modules["pop_adaptive"]
    gid = {a: int(c["id"]) for a, c in arms.items()}
    out = {"seed": seed, "phase_days": phase_days, "arms": {a: dict(c) for a, c in arms.items()},
           "scripted": "site specifications and the switch at the end of phase 1 (experimenter)"}
    hashes = lambda: {a: ad.policy(w, g).weights_hash() for a, g in gid.items()}  # noqa: E731
    out["hash_genesis"] = hashes()
    w.advance_time(w.now + phase_days * DAY)
    out["hash_switch"] = hashes()
    log1 = A.visit_log(w)
    hosts = A.res_hosts(w)
    row = {h: r for r, h in enumerate(hosts)}
    vis = {h: int((log1["res"] == row[h]).sum()) if h in row else 0 for h in names}
    rich = sorted([h for h, b in zip(names, BEHAVIOURS) if b == "rich"], key=lambda h: (-vis[h], h))
    to_broken, to_rich = rich[:2], [h for h, b in zip(names, BEHAVIOURS) if b == "broken"]

    def switch(ctx):
        for h in to_broken:
            redeploy(ctx, h, specs[names.index(h)], "broken")
        for h in to_rich:
            redeploy(ctx, h, specs[names.index(h)], "rich")
    w._run_system("experiment.switch", switch)
    t_switch = w.now
    w.advance_time(w.now + phase_days * DAY)
    out["hash_end"] = hashes()
    out["switch"] = {"to_broken": to_broken, "to_rich": to_rich}
    ups = [e.payload for e in w.log if e.type == "pop.policy_update"]
    for a, g in gid.items():
        s = [u for u in ups if u["group"] == f"adaptive:{g}"]
        out.setdefault("updates", {})[a] = max(0, len(s) - 1)
        out.setdefault("optimizer_steps", {})[a] = int(sum(int(u.get("optimizer_steps") or 0) for u in s))
        out.setdefault("loss_curve", {})[a] = [(float(u.get("loss_before") or 0), float(u["loss"]),
                                                int(u["n_examples"])) for u in s[1:]]
    log = A.visit_log(w)
    pg = columns.col(w, "pop", "policy_group").astype(int)
    m2 = (log["cls"] == 2) & (log["activity"] == "web")
    grp = pg[log["ind"].astype(int)]
    bad_after = {row[h] for h in to_broken if h in row} | \
        {row[h] for h, b in zip(names, BEHAVIOURS) if b in ("thin", "spam") and h in row}
    util = {}
    for a, g in gid.items():
        r = {}
        for ph, sel in (("phase1", log["t"] < t_switch), ("phase2", log["t"] >= t_switch)):
            m = m2 & (grp == g) & sel
            r[ph] = {"visits": int(m.sum()), "mean_util": float(log["util"][m].mean()) if m.any() else 0.0}
            if ph == "phase2":
                r[ph]["share_failing"] = float(np.isin(log["res"][m], list(bad_after)).mean()) if m.any() else 0.0
            util[(a, ph)] = log["util"][m].astype(np.float64)
        out.setdefault("behaviour", {})[a] = r
    names_ = list(gid)
    if len(names_) >= 2:
        a, b = names_[0], names_[1]
        out["phase2_difference"] = _boot(util[(a, "phase2")], util[(b, "phase2")])
        out["phase1_difference"] = _boot(util[(a, "phase1")], util[(b, "phase1")], seed=1)
    st = w.state
    unused = [k for k, r in st.tables["pop_examples"].items() if r.get("used") is None]
    if unused:
        batch = ad._load_examples(w, unused)
        out["heldout_examples"] = int(len(batch["reward"]))
        out["heldout_mse"] = {a: float(ad.policy(w, g).loss_on(batch)) for a, g in gid.items()}
        out["_heldout"] = {"batch": batch, "policies": {a: ad.policy(w, g) for a, g in gid.items()}}
    mem = columns.col(w, "pop", "memory")[columns.col(w, "pop", "class") == 2].astype(np.float32)
    out["distinct_personal_states"] = int(np.unique(mem.round(3), axis=0).shape[0])
    out["adaptive_rows"] = int(mem.shape[0])
    out["policy_objects"] = len(w.runtime.get("pop_adaptive.nets", {}))
    out["adaptive_stats"] = {k: (v if isinstance(v, (int, float)) else str(v))
                             for k, v in dict(w.runtime.get("pop_adaptive.stats", {})).items()}
    # simultaneously active adaptive individuals: distinct ones that took part in the same simulated day, and the
    # largest number activated in one decision round
    days, rounds = {}, {}
    cls = columns.col(w, "pop", "class")
    for e in w.log:
        if e.type == "cols.batch" and e.payload["kind"] == "pop.session":
            idx = artifacts.get_array_blob(w, e.payload["ref"])["idx"].astype(np.int64)
            a2 = idx[cls[idx] == 2]
            t = int(e.payload["extra"]["t"])
            days.setdefault(t // DAY, set()).update(a2.tolist())
            rounds[t] = rounds.get(t, 0) + int(a2.size)
    out["max_adaptive_active_same_day"] = max((len(v) for v in days.values()), default=0)
    out["max_adaptive_per_decision_round"] = max(rounds.values(), default=0)
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(spearmanr(a, b).correlation)


def _describe(c: np.ndarray) -> str:
    if c[6] > 0.5:
        return ("You are deciding whether to try a website you have not visited before"
                + (f", recommended by a contact you trust {c[8]:.2f} (0-1)." if c[7] > 0.5 else "."))
    return (f"You are deciding whether to visit a website you already know. How well its content matched your "
            f"interests last time: {c[0]:.2f} (-1 to 1). Your satisfaction with it so far: {c[1]:.2f} (-1 to 1). "
            f"Visits so far: {int(round(np.expm1(c[2] * 3.0)))}. Visited recently: {'yes' if c[3] > 0.5 else 'no'}. "
            f"Share of your contacts who like it: {c[4]:.2f}.")


def compact_vs_language(run: dict, seed: int = 0, *, n: int = 200, model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
                        ) -> list[dict]:
    """The same held-out class-2 decisions (the chosen candidate's features and the realised utility) valued by
    the compact policies of the run (learned and its identical-init frozen twin) and by a small pretrained language
    model asked zero-shot whether the visit is worthwhile (log P(yes) - log P(no) through crucible.semantic's
    inference service, CPU). Returns one row per model: Spearman correlation of value and realised utility,
    parameters and milliseconds per decision."""
    import torch
    h = run.get("_heldout")
    if not h:
        return []
    b = h["batch"]
    rng = np.random.default_rng(seed)
    ix = np.sort(rng.choice(len(b["reward"]), size=min(n, len(b["reward"])), replace=False))
    sub = {k: v[ix] for k, v in b.items()}
    rows = []
    for name, pol in h["policies"].items():
        t0 = time.time()
        with torch.no_grad():
            obs = torch.from_numpy(sub["obs"].astype(np.float32))
            mem = torch.from_numpy(sub["mem"].astype(np.float32)) if pol.use_memory else \
                torch.zeros(len(ix), pol.mem_dim)
            hh, m2 = pol.net.memory(obs, mem)
            pred = pol.net.score(hh, m2, torch.from_numpy(sub["cand"].astype(np.float32))[:, None, :])[:, 0].numpy()
        dt = (time.time() - t0) / max(1, len(ix)) * 1000
        rows.append({"model": f"compact_{name}", "params": int(pol.params), "spearman": _spearman(pred, sub["reward"]),
                     "ms_per_decision": dt, "n": int(len(ix))})
    try:
        from ..semantic.service import SemanticService
        svc = SemanticService(model, device="cpu", max_batch=16)
        reqs = [{"messages": [{"role": "system", "content": "Answer yes or no."},
                              {"role": "user", "content": _describe(c) + " Is the visit likely to be worthwhile?"}],
                 "choices": [" yes", " no"], "normalize": "sum"} for c in sub["cand"]]
        t0 = time.time()
        res = svc.choose(None, reqs)
        dt = (time.time() - t0) / max(1, len(ix)) * 1000
        val = np.array([r["logprobs"][0] - r["logprobs"][1] for r in res])
        params = int(sum(p.numel() for p in svc.model.parameters())) if hasattr(svc, "model") else 134_515_008
        rows.append({"model": "smollm2_135m_zero_shot", "params": params, "spearman": _spearman(val, sub["reward"]),
                     "ms_per_decision": dt, "n": int(len(ix))})
    except Exception as e:  # noqa: BLE001  (reported as a row: the language model could not run here)
        rows.append({"model": "smollm2_135m_zero_shot", "params": 0, "spearman": 0.0, "ms_per_decision": 0.0,
                     "n": 0, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return rows
