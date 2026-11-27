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


def _paired_boot(d: np.ndarray, strata: np.ndarray | None = None, n: int = 2000, seed: int = 0) -> dict:
    """Mean of paired differences with a bootstrap 95% CI (resampling units within each stratum)."""
    d = np.asarray(d, np.float64)
    if d.size == 0:
        return {"mean": 0.0, "ci95": [0.0, 0.0], "units": 0}
    strata = np.zeros(d.size, np.int64) if strata is None else np.asarray(strata)
    groups = [d[strata == s] for s in np.unique(strata)]
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([rng.choice(g, g.size) for g in groups]).mean() for _ in range(n)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "units": int(d.size)}


def _member_phase(log: dict, members: np.ndarray, t_switch: int, bad: set) -> dict[int, tuple[float, float, int]]:
    """Per class-2 member: (mean realised utility, share of visits to failing sites, visits) of its phase-2 web
    visits."""
    m = (log["cls"] == 2) & (log["activity"] == "web") & (log["t"] >= t_switch) & np.isin(log["ind"], members)
    ind, u, res = log["ind"][m].astype(np.int64), log["util"][m].astype(np.float64), log["res"][m]
    out = {}
    for i in np.unique(ind):
        k = ind == i
        out[int(i)] = (float(u[k].mean()), float(np.isin(res[k], list(bad)).mean()), int(k.sum()))
    return out


def adaptation(seed: int, phase_days: int = 5, arms: dict | None = None, *, n_light: int = 150,
               min_examples: int = 32, window_s: int = 14400, counterfactual: bool = False) -> dict:
    """See the module docstring. `arms` maps an arm name to a group config ({id, share, learning, memory, seed}).

    `counterfactual`: at the switch the world is forked and, in the fork only, the first learning arm stops learning
    (`pop_adaptive.set_learning`; the main world records the same system call with learning on, so both worlds
    keep identical random streams until the first post-switch policy update). Both run phase 2. Per member of that
    group, the phase-2 utility in the main world minus the utility in the fork is what learning AFTER the change
    did under the new condition - change-specific adaptation, with the weights, the memories, the state of the world
    and every random draw at the switch held equal (`adaptation_after_change`, paired bootstrap over members)."""
    import torch
    torch.set_num_threads(1)
    arms = arms or {"learning": {"id": 0, "share": 0.5, "seed": 5},
                    "frozen": {"id": 1, "share": 0.5, "seed": 5, "learning": False}}
    names, specs = _site_specs(seed)
    spec = population_world("development", seed, n=n_light, site_specs=specs, tick_s=1800,
                            classes={"adaptive": 256, "strategic": 0, "semantic": 0}, strategic=False,
                            adaptive={"groups": list(arms.values()), "min_examples": min_examples},
                            pop={"adaptive_window_s": window_s})
    t0 = time.time()
    w = World.create(spec, strict=True)
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
    cf, la = None, None
    if counterfactual:
        la = next(a for a, c in arms.items() if c.get("learning", True))
        cf = w.fork(f"{w.world_id}-frozen-at-switch")
        w.system_call("pop_adaptive.set_learning", {"group": gid[la], "learning": True})
        cf.system_call("pop_adaptive.set_learning", {"group": gid[la], "learning": False})
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
    if cf is not None:
        try:
            cf.advance_time(cf.now + phase_days * DAY)
            members = np.flatnonzero(pg == gid[la])
            a_main = _member_phase(log, members, t_switch, bad_after)
            a_cf = _member_phase(A.visit_log(cf), members, t_switch, bad_after)
            both = sorted(set(a_main) & set(a_cf))
            d = np.array([a_main[i][0] - a_cf[i][0] for i in both])
            g = f"adaptive:{gid[la]}"
            after = lambda lg: sum(1 for e in lg if e.type == "pop.policy_update" and  # noqa: E731
                                   e.payload["group"] == g and e.t >= t_switch)
            out["adaptation_after_change"] = {
                **_paired_boot(d, seed=2), "arm": la, "members_compared": len(both),
                "member_utility_diffs": d.tolist(),
                "share_failing_learning": float(np.mean([a_main[i][1] for i in both])) if both else 0.0,
                "share_failing_frozen_at_switch": float(np.mean([a_cf[i][1] for i in both])) if both else 0.0,
                "visits_learning": int(sum(a_main[i][2] for i in both)),
                "visits_frozen_at_switch": int(sum(a_cf[i][2] for i in both)),
                "policy_updates_after_switch": int(after(w.log)),
                "policy_updates_after_switch_frozen": int(after(cf.log)),
                "hash_switch": out["hash_switch"][la], "hash_end_learning": out["hash_end"][la],
                "hash_end_frozen_at_switch": cf.modules["pop_adaptive"].policy(cf, gid[la]).weights_hash()}
        finally:
            cf.close()
    names_ = list(gid)
    if len(names_) >= 2:
        a, b = names_[0], names_[1]
        out["phase2_difference"] = _boot(util[(a, "phase2")], util[(b, "phase2")])
        out["phase1_difference"] = _boot(util[(a, "phase1")], util[(b, "phase1")], seed=1)
    st = w.state
    unused = [k for k, r in st.tables["pop_examples"].items() if r.get("used") is None]
    if unused:
        batch = ad._load_examples(w, unused)
        cnt = {k: (0 if b is None else int(np.count_nonzero(np.asarray(b["w"]) > 0)) if "w" in b else
                   int(len(b["reward"]))) for k, b in batch.items()}     # day reviews (zero weight) are not examples
        out["heldout_examples"] = int(sum(cnt.values()))
        out["heldout_examples_by_kind"] = cnt
        out["heldout_mse"] = {a: float(ad.policy(w, g).loss_on(batch)) for a, g in gid.items()}
        out["_heldout"] = {"batch": batch, "policies": {a: ad.policy(w, g) for a, g in gid.items()}}
    mem = columns.col(w, "pop", "memory")[columns.col(w, "pop", "class") == 2].astype(np.float32)
    out["distinct_personal_states"] = int(np.unique(mem, axis=0).shape[0])
    out["adaptive_rows"] = int(mem.shape[0])
    out["personal_states_changed"] = int(np.count_nonzero(np.abs(mem).sum(axis=1) > 0))
    out["policy_objects"] = len(w.runtime.get("pop_adaptive.nets", {}))
    out["adaptive_stats"] = {k: (v if isinstance(v, (int, float)) else str(v))
                             for k, v in dict(w.runtime.get("pop_adaptive.stats", {})).items()}
    # simultaneously active adaptive individuals: distinct ones that took part in the same simulated day, and the
    # largest number activated in one decision round
    days, rounds, policy_days, passes = {}, {}, {}, {}
    cls = columns.col(w, "pop", "class")
    for e in w.log:
        # individuals whose own personal state their policy stepped (a day review or a session's activity decision)
        if e.type == "cols.batch" and e.payload["kind"] == "pop_adaptive.memory":
            idx = artifacts.get_array_blob(w, e.payload["ref"])["idx"].astype(np.int64)
            t = int(e.payload["extra"]["t"])
            policy_days.setdefault(t // DAY, set()).update(idx[cls[idx] == 2].tolist())
            k = "day_review" if (e.payload["extra"] or {}).get("review") else "activity"
            passes[k] = max(passes.get(k, 0), int(idx.size))
        # individuals who took a session (the population's learning batch of the session's takers)
        if e.type == "cols.batch" and e.payload["kind"] == "pop.learn":
            idx = artifacts.get_array_blob(w, e.payload["ref"])["idx"].astype(np.int64)
            a2 = idx[cls[idx] == 2]
            t = int(e.payload["extra"]["t"])
            days.setdefault(t // DAY, set()).update(a2.tolist())
            rounds[t] = rounds.get(t, 0) + int(a2.size)
    out["max_adaptive_active_same_day"] = max((len(v) for v in days.values()), default=0)
    ph2 = set()
    for d_, v in days.items():
        if d_ * DAY >= t_switch:
            ph2 |= v
    out["adaptive_active_phase2"] = len(ph2)
    ds = sorted(days)
    out["max_adaptive_active_two_days"] = max((len(days[d] | days.get(d + 1, set())) for d in ds), default=0)
    out["max_adaptive_per_decision_round"] = max(rounds.values(), default=0)
    run = set()
    for v in days.values():
        run |= v
    out["adaptive_active_run"] = len(run)
    # simultaneously active: the largest number of class-2 individuals whose policy decided for them (stepped their
    # own memory: the day review at the first activation of the day, or a session's activity decision) on one day
    out["max_adaptive_policy_active_same_day"] = max((len(v) for v in policy_days.values()), default=0)
    out["adaptive_policy_active_days"] = {int(d_): len(v) for d_, v in sorted(policy_days.items())}
    out["largest_memory_batch"] = passes
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


LANGUAGE_MODELS = ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen3-0.6B")


def compact_vs_language(run: dict, seed: int = 0, *, n: int = 200, models=LANGUAGE_MODELS, device: str = "cpu"
                        ) -> list[dict]:
    """The same held-out class-2 resource decisions (the chosen candidate's features, the realised utility) valued
    by the compact policies of the run (the learned one and its identical-init frozen twin) and by small pretrained
    language models asked zero-shot whether the visit is worthwhile (log P(yes) - log P(no), through
    crucible.semantic's inference service; local checkpoints, evaluator side, nothing enters a world). One row per
    model: Spearman correlation of its value with the realised utility, parameters, milliseconds per decision; a
    language model that cannot run here is a row with `error` (never dropped)."""
    import torch
    h = run.get("_heldout")
    if not h or h["batch"].get("web") is None:
        return []
    b = h["batch"]["web"]
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
            pred = pol.net.score(pol.net.encode(obs), mem,
                                 torch.from_numpy(sub["cand"].astype(np.float32))[:, None, :])[:, 0].numpy()
        dt = (time.time() - t0) / max(1, len(ix)) * 1000
        rows.append({"model": f"compact_{name}", "kind": "compact", "params": int(pol.params),
                     "spearman": _spearman(pred, sub["reward"]), "ms_per_decision": dt, "n": int(len(ix))})
    from ..semantic.service import SemanticService
    for mid in models:
        tag = mid.split("/")[-1].lower().replace("-instruct", "").replace("-", "_").replace(".", "_")
        try:
            svc = SemanticService(mid, device=device, max_batch=16)
            reqs = [{"messages": [{"role": "system", "content": "Answer yes or no."},
                                  {"role": "user", "content": _describe(c) + " Is the visit likely to be worthwhile?"}],
                     "choices": [" yes", " no"], "normalize": "sum"} for c in sub["cand"]]
            t0 = time.time()
            res = svc.choose(None, reqs)
            dt = (time.time() - t0) / max(1, len(ix)) * 1000
            val = np.array([r["logprobs"][0] - r["logprobs"][1] for r in res])
            params = int(sum(p.numel() for p in svc.model.parameters())) if hasattr(svc, "model") else 0
            rows.append({"model": f"{tag}_zero_shot", "kind": "language", "params": params,
                         "spearman": _spearman(val, sub["reward"]), "ms_per_decision": dt, "n": int(len(ix))})
        except Exception as e:  # noqa: BLE001  (reported as a row with the error: the model could not run here)
            rows.append({"model": f"{tag}_zero_shot", "kind": "language", "params": 0, "spearman": 0.0,
                         "ms_per_decision": 0.0, "n": 0, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        finally:
            import gc
            svc = None
            gc.collect()
    return rows
