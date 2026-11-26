"""Gate J and Gate L measurements over the invention arms (`innovation.civexp`; EVALUATOR SIDE, wave 3b).

`arms(kind, seed, ...)` runs the four arms (factual, garbage, ban_novel, ban_published) on one spec and seed and
mines chains; `j_measure(A)` and `l_measure(A)` compute the Gate J and Gate L metrics the gate tests record. Every
product is made by an autonomous developer of the population; every participant is an individual of the
population doing its own work; nothing is scripted. "Independent" is the evaluator's verdict
(`measure.independent`), never computed in-world.

Gate J (per product *category* = the task a product's users rely on it for; `civexp.product_category`):

* J.1 agent_created_executable_products - products created during the run by in-world creators (here: the
  population's developers, `creator_kind` "population") that executed for at least one independent user;
* J.2 published_products - live and declared (or released) at the end;
* J.3 independent_discoverers - min over categories of the best product's individuals who considered it for their
  own task after the interface check, or tried it (a bare search hit does not count);
* J.4 independent_users - min over categories: individuals for whom it executed on their own input;
* J.5 measured_behavior_adoption_pvalue - Fisher exact test, triers who kept relying on the category's products in
  the factual arm vs in the garbage arm (input-ignoring programs, everything else equal);
* J.6 independent_transactions - min over categories: paid calls net of refunds from independent payers;
* J.7 resource_consumption_delta - min over categories: hand-work minutes of the activities with that step in the
  arm where nobody builds for it, minus the factual arm (a pre-existing resource: people's working time); the
  compute the developers' services draw from the providers is reported too;
* J.n1 product_categories - categories whose best product passes J.1-J.7 (and whether one is a generated,
  undocumented category: `novel`).

Gate L:

* L.1 capability_effects_measured - pre-existing conditions that differ between the factual arm and the removal
  arms (hand-work minutes of novel and of documented activities, tasks lost for lack of time, incumbents' live
  subscriptions, compute drawn from the providers);
* L.2 emerged_needs - needs the evaluator's relation graph labels after genesis that were not there at genesis;
* L.3 needs_addressed_by_new_services - needs whose target was later served by a provider created after the need's
  chain began, which holders adopted;
* L.4 independent_chains / min_chain_stages - mined chains (>= 3 stages, causally ordered, >= 2 subsystems) with
  distinct first changes; among the top three: an agent (developer) artifact, social activity, technical activity;
* L.5 counterfactual_removals_confirming - chains whose later stages do not happen in the arm where the chain's
  first change (the invention it starts from) is removed;
* L.n1 special_case_bonuses - system-account credits to creators after admission (`measure.special_case_bonuses`;
  population income deposits verified against the population's own deposit records) plus the static scan.
"""
from __future__ import annotations

import functools

from . import civexp as CX
from . import experiments as X
from . import measure as M

DAY = 86400


def _arm_summary(r: dict) -> dict:
    """What the gate measures read from a counterfactual arm's world, so that only the factual world stays open (a
    civilisation world holds a few GB): individuals relying on in-world creations per step, and the system credits
    to its creators."""
    w = r["world"]
    steps = sorted({row["step_need"] for _, row in w.state.scan("work_exp") if row["step_need"]})
    creators = sorted({p["creator"] for p in M.agent_products(w)})
    return {"relying": {need: relying_on_creations(w, need) for need in steps},
            "bonuses": M.special_case_bonuses(w, creators)}


@functools.lru_cache(maxsize=2)
def arms(kind: str = "civ", seed: int = 3, days: int = 24, profile: str = "balanced", n: int = 600) -> dict:
    """The four arms (factual first). Counterfactual arms are summarised (`_arm_summary`) and their worlds closed;
    the factual arm keeps its world (`A["factual"]["world"]`)."""
    out = {}
    for arm in CX.ARMS:
        r = CX.run_arm(kind, arm, seed, days=days, profile=profile, n=n, log=print)
        r["outcome"] = CX.outcome(r)
        r["chains"] = X.mine_chains(r) + CX.package_chains(r)
        r["summary"] = _arm_summary(r)
        if arm != "factual":
            r["world"].close()
            r["world"] = None
        out[arm] = r
    return out


def _relying(A: dict, arm: str, need: str) -> int:
    r = A[arm]
    if r.get("world") is not None:
        return relying_on_creations(r["world"], need)
    return int(r["summary"]["relying"].get(need, 0))


def _cat_totals(out: dict, cat: str) -> tuple[int, int]:
    tried = kept = 0
    for o in out["products"].values():
        if o.get("category") == cat and o.get("kind") == "service":
            tried += len(o["triers"])
            kept += len(o["kept"])
    return tried, kept


# ====================================================================== Gate J
def j_measure(A: dict) -> dict:
    f = A["factual"]["outcome"]
    rows = {}
    for pid, o in sorted(f["products"].items()):
        cat = o.get("category")
        if cat is None:
            continue
        rows[pid] = {"category": cat, "kind": o["kind"], "creator_kind": o["creator_kind"],
                     "published": bool(o["published"]), "executable": bool(o["executable"]),
                     "discoverers": int(o["discoverers"]), "users": int(o["users"]), "tx": int(o["tx"]["n"]),
                     "tx_cents": int(o["tx"]["cents"]), "payers": int(o["tx"]["payers"])}
    cats = sorted({r["category"] for r in rows.values() if r["kind"] == "service"})
    best, per = {}, {}
    for c in cats:
        svc = [r for r in rows.values() if r["category"] == c and r["kind"] == "service"]
        b = max(svc, key=lambda r: (r["users"], r["tx"], r["discoverers"]))
        rm = CX.removal_arm_for(c)
        delta = CX.minutes_for(A[rm]["outcome"], c) - CX.minutes_for(f, c)
        tk, kk = _cat_totals(f, c)
        tg, kg = _cat_totals(A["garbage"]["outcome"], c)
        best[c] = dict(b, delta_minutes=round(float(delta), 1), removal_arm=rm, tried=tk, kept=kk,
                       tried_garbage=tg, kept_garbage=kg, novel=c.startswith("gen."))
        per[c] = M.fisher_exact(kk, tk - kk, kg, tg - kg) if tk and tg else 1.0
    a = sum(b["kept"] for b in best.values())
    b_ = sum(b["tried"] - b["kept"] for b in best.values())
    c_ = sum(b["kept_garbage"] for b in best.values())
    d = sum(b["tried_garbage"] - b["kept_garbage"] for b in best.values())
    passing = sorted(c for c, r in best.items() if r["executable"] and r["published"] and r["discoverers"] >= 10
                     and r["users"] >= 10 and r["tx"] >= 1 and r["delta_minutes"] > 0)
    ok = [best[c] for c in passing] or list(best.values())
    fc, rc = A["factual"]["outcome"]["dev_compute"], A["ban_published"]["outcome"]["dev_compute"]
    executable = [r for r in rows.values() if r["executable"] and r["users"] > 0]
    return {
        "rows": rows, "best": best, "passing": passing,
        "J.1": {"agent_created_executable_products": len(executable),
                "creators_population": sum(1 for r in executable if r["creator_kind"] == "population"),
                "creators_scripted": sum(1 for r in executable if r["creator_kind"] == "scripted"),
                "creators_agent": sum(1 for r in executable if r["creator_kind"] == "agent"),
                "services": sum(1 for r in executable if r["kind"] == "service"),
                "packages": sum(1 for r in executable if r["kind"] == "package")},
        "J.2": {"published_products": sum(1 for r in rows.values() if r["published"])},
        "J.3": {"independent_discoverers": min((r["discoverers"] for r in ok), default=0), "categories": len(best)},
        "J.4": {"independent_users": min((r["users"] for r in ok), default=0)},
        "J.5": {"measured_behavior_adoption_pvalue": M.fisher_exact(a, b_, c_, d), "kept_good": a,
                "dropped_good": b_, "kept_garbage": c_, "dropped_garbage": d,
                **{f"pvalue_{c.replace('.', '_')}": v for c, v in per.items()}},
        "J.6": {"independent_transactions": min((r["tx"] for r in ok), default=0),
                "independent_payers_min": min((r["payers"] for r in ok), default=0)},
        "J.7": {"resource_consumption_delta": float(min((r["delta_minutes"] for r in ok), default=0.0)),
                "developer_compute_cu_delta": float(fc["cu"] - rc["cu"]),
                "developer_cores_delta": int(fc["cores"] - rc["cores"])},
        "J.n1": {"product_categories": len(passing), "novel_categories": sum(1 for c in passing if c.startswith("gen.")),
                 "categories": ",".join(passing)},
    }


def j_note(J: dict) -> str:
    return "Autonomous population developers (no scripted creator). Best product per category: " + "; ".join(
        f"{c}{' [novel]' if r['novel'] else ''}: discoverers {r['discoverers']}, users {r['users']}, tx {r['tx']} "
        f"({r['tx_cents']}c, {r['payers']} payers), kept {r['kept']}/{r['tried']} vs garbage "
        f"{r['kept_garbage']}/{r['tried_garbage']}, hand-work minutes saved {r['delta_minutes']:.0f} "
        f"(removal arm {r['removal_arm']})" for c, r in sorted(J["best"].items()))


# ====================================================================== Gate L
def good_chains(chains: list[dict]) -> list[dict]:
    return [c for c in chains if c["n_stages"] >= 3 and c["ordered"] and len(c["subsystems"]) >= 2]


def chain_removal_arm(chain: dict) -> str | None:
    """The arm in which the chain's first change does not happen."""
    first = chain["stages"][0]["name"] if chain["stages"] else ""
    if first.startswith("adoption of a "):
        need = first[len("adoption of a "):].split(" provider", 1)[0]
        return CX.removal_arm_for(need)
    name = chain["chain"]
    if name.startswith("need:"):
        target = name.split("capability/", 1)[-1]
        return CX.removal_arm_for(target)
    if name.startswith("competition:") or name.startswith("package:"):
        return "ban_published"
    return None


def relying_on_creations(world, need: str) -> int:
    """Individuals who rely (now) on a product of an in-world creator for the step `need` of their work."""
    provs = set()
    for p in M.agent_products(world):
        provs |= M._provider_ids(world, p)
    return len({r["i"] for _, r in world.state.scan("work_exp") if r["active"] and r["step_need"] == need
                and r["prov"] in provs})


def confirmed(A: dict, chain: dict) -> bool:
    """The chain's later stages do not happen in the arm where its first change was removed."""
    arm = chain_removal_arm(chain)
    if arm is None:
        return False
    out = A[arm]["outcome"]
    name = chain["chain"]
    if name.startswith("need:"):
        nid = name[5:]
        if chain["stages"][0]["name"].startswith("adoption of a "):
            # adoption created the need: without the earlier adoption, the need never arises
            return nid not in out["needs_ever"]
        target = name.split("capability/", 1)[-1]
        later = [s for s in chain["stages"] if s["name"] in ("adoption by holders", "need relieved")]
        # without the invention nobody adopts anything created for the step and the need is still there at the end
        return bool(later) and _relying(A, "factual", target) > 0 and _relying(A, arm, target) == 0 and \
            nid in out["needs"]
    if name.startswith("competition:"):
        return out["incumbent_price_changes"] < A["factual"]["outcome"]["incumbent_price_changes"] or \
            not any(c["chain"] == name and c["n_stages"] >= 3 for c in A[arm]["chains"])
    if name.startswith("package:"):
        return not any(c["chain"] == name for c in A[arm]["chains"])
    return False


def l_measure(A: dict) -> dict:
    f = A["factual"]
    fo = f["outcome"]
    hist = f["history"]
    at_genesis = set(hist[0]["needs"])
    emerged = sorted({nid for h in hist[1:] for nid in h["needs"]} - at_genesis)
    effects = {}
    novel = [k for k in fo["families"] if k.startswith("gen.")]
    pub = [k for k in fo["families"] if not k.startswith("gen.")]

    def hand(out, keys):
        return sum(out["families"][k]["diy_minutes"] for k in keys if k in out["families"])

    def lost(out, keys):
        return sum(out["families"][k]["lost"] for k in keys if k in out["families"])
    effects["novel_hand_minutes"] = hand(A["ban_novel"]["outcome"], novel) - hand(fo, novel)
    effects["documented_hand_minutes"] = hand(A["ban_published"]["outcome"], pub) - hand(fo, pub)
    effects["tasks_lost"] = (lost(A["ban_novel"]["outcome"], novel) + lost(A["ban_published"]["outcome"], pub)
                             - lost(fo, novel) - lost(fo, pub))
    inc_f = sum(fo["incumbent_live_subs"].values())
    inc_r = sum(A["ban_published"]["outcome"]["incumbent_live_subs"].values())
    effects["incumbent_live_subscriptions"] = inc_r - inc_f
    effects["provider_compute_cu"] = fo["dev_compute"]["cu"] - A["ban_published"]["outcome"]["dev_compute"]["cu"]
    measured = {k: v for k, v in effects.items() if (k.endswith("minutes") and v > 0) or
                (k == "provider_compute_cu" and v > 0) or (k not in ("novel_hand_minutes", "documented_hand_minutes",
                                                                    "provider_compute_cu") and v != 0)}
    good = good_chains(f["chains"])
    good.sort(key=lambda c: (-c["n_stages"], c["chain"]))
    roots = {}
    for c in good:
        key = (c["stages"][0]["name"], c["stages"][0]["t"])
        roots.setdefault(key, c)
    indep = list(roots.values())
    top = indep[:3]
    addressed = [c for c in good if c["chain"].startswith("need:") and
                 any(s["name"].startswith("agent artifact") for s in c["stages"]) and
                 any(s["name"] == "adoption by holders" for s in c["stages"])]
    conf = [(c["chain"], chain_removal_arm(c), confirmed(A, c)) for c in indep]
    creators = sorted({p["creator"] for p in M.agent_products(f["world"])})
    bon = {a: A[a]["summary"]["bonuses"] for a in A}
    return {
        "effects": effects, "emerged": emerged, "good": good, "independent": indep, "confirmations": conf,
        "L.1": {"capability_effects_measured": len(measured), **{f"effect_{k}": float(v) for k, v in effects.items()}},
        "L.2": {"emerged_needs": len(emerged), "needs_at_genesis": len(at_genesis)},
        "L.3": {"needs_addressed_by_new_services": len({c["chain"] for c in addressed})},
        "L.4": {"independent_chains": len(indep),
                "min_chain_stages": min((c["n_stages"] for c in top), default=0),
                "chains_with_agent_artifact": sum(1 for c in indep if c["agent_artifact"]),
                "chains_with_social": sum(1 for c in indep if c["social"]),
                "chains_with_technical": sum(1 for c in indep if c["technical"]),
                "top3_covers_artifact_social_technical": bool(any(c["agent_artifact"] for c in top) and
                                                             any(c["social"] for c in top) and
                                                             any(c["technical"] for c in top)),
                "scripted_stages": sum(1 for c in indep for s in c["stages"] if s.get("creator_kind") == "scripted")},
        "L.5": {"counterfactual_removals_confirming": sum(1 for _, _, ok in conf if ok), "chains_tested": len(conf)},
        "L.n1": {"runtime_bonuses": int(sum(b["bonuses"] for b in bon.values())),
                 "income_deposits_verified": int(sum(b["income_deposits"] for b in bon.values())),
                 "creators": len(creators)},
    }


# ====================================================================== SP-6.3b (usefulness, novel categories, workflows)
USEFULNESS_DECISIONS = {
    # module path (under crucible/) -> functions that decide whether something is useful to a participant
    "economy/work.py": ("_accepted", "_adopt", "act_shop", "_after_session", "belief", "_public_prior"),
    "innovation/caps.py": ("ep_verify", "ep_adopt_service", "ep_try"),
    "innovation/pkg.py": ("_exercised", "_repro_check", "ep_upgrade"),
    "innovation/devs.py": ("_maintain", "_compose_plan", "_competitors", "converts", "_explore"),
}
LM_MODULES = ("semantic.space.lm", "semantic.participants", "semantic.agents", "substrate.inference", "transformers",
              "torch", "openai", "anthropic")
LM_CALLS = frozenset({"lm", "llm", "judge", "grade", "chat", "generate_text", "complete", "logprob", "perplexity"})


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast_mod().Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast_mod().Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def ast_mod():
    import ast
    return ast


def single_judge_paths(root: str) -> list[str]:
    """Usefulness decision functions that reach a language-model judgement: the decision functions themselves and,
    transitively, the functions of the same module they call (static AST scan of names actually used: imports of
    language-model modules, calls through them, calls named like a judge). A usefulness decision must come from
    the participant's own measured outcome (world physics, own tests, own experience), never from one model
    grading for everyone."""
    import os
    ast = ast_mod()
    out = []
    for rel, fns in sorted(USEFULNESS_DECISIONS.items()):
        path = os.path.join(root, rel)
        tree = ast.parse(open(path).read())
        defs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

        def lm_aliases(scope) -> set[str]:
            al = set()
            for n in ast.walk(scope):
                if isinstance(n, ast.ImportFrom):
                    mod = n.module or ""
                    for a in n.names:
                        if any(m in f"{mod}.{a.name}" for m in LM_MODULES):
                            al.add(a.asname or a.name)
                elif isinstance(n, ast.Import):
                    for a in n.names:
                        if any(m in a.name for m in LM_MODULES):
                            al.add((a.asname or a.name).split(".")[0])
            return al
        module_aliases = lm_aliases(tree)
        for fn in fns:
            if fn not in defs:
                out.append(f"{rel}:{fn} (missing: the scan's list is stale)")
                continue
            seen, todo, hit = set(), [fn], None
            while todo and hit is None:
                f = todo.pop()
                if f in seen or f not in defs:
                    continue
                seen.add(f)
                al = module_aliases | lm_aliases(defs[f])
                for n in ast.walk(defs[f]):
                    if not isinstance(n, ast.Call):
                        continue
                    name = _dotted(n.func)
                    parts = name.split(".") if name else []
                    if (parts and parts[0] in al) or (set(parts) & LM_CALLS):
                        hit = f"{rel}:{fn} -> {f}: {name}"
                        break
                    last = parts[-1] if parts else None
                    if last in defs:
                        todo.append(last)
            if hit:
                out.append(hit)
    return out


def grounded_adoptions(world) -> dict:
    """Every adoption by an individual (`work.rely` active) is preceded by the individual's own accepted outcome
    with that provider (`work.outcomes`, or the subscription trial it started); per product, keep and drop
    decisions differ between individuals (diverse, grounded responses)."""
    ok_seen: set[tuple[int, str]] = set()
    adopt, grounded = 0, 0
    keep: dict[str, set] = {}
    drop: dict[str, set] = {}
    for e in world.log:
        if e.type == "work.outcomes":
            for row in e.payload["rows"]:
                if int(row[4]) > 0:
                    ok_seen.add((int(row[0]), row[1]))
        elif e.type == "work.rely":
            pv, i = e.payload["prov"], int(e.payload["i"])
            if e.payload.get("active"):
                if pv.startswith("cdecl_"):
                    adopt += 1
                    grounded += int((i, pv) in ok_seen)
                keep.setdefault(pv, set()).add(i)
            else:
                drop.setdefault(pv, set()).add(i)
    diverse = sum(1 for pv in set(keep) | set(drop) if keep.get(pv) and drop.get(pv))
    return {"capability_adoptions": adopt, "grounded": grounded, "providers_with_keep_and_drop": diverse}


def sp63b_measure(A: dict, root: str) -> dict:
    from ..substrate import ids as _ids
    f = A["factual"]
    w = f["world"]
    st = w.state
    novel = {g["need"] for g in w.modules["work"].fams if g["need"].startswith("gen.")}
    docs = [d["title"] for _, d in st.scan("documents")] if "documents" in st.tables else []
    undocumented = sorted(n for n in novel if not any(n in t for t in docs))
    novel_adopt = []
    for pid, o in f["outcome"]["products"].items():
        if o.get("category") in undocumented and o.get("kind") == "service" and o["created_t"] > 0:
            novel_adopt.append((pid, len(o["kept"])))
    wfs = []
    for wid, wf in st.scan("cap_workflows") if "cap_workflows" in st.tables else ():
        comps = {c["provider"] for c in wf["components"]}
        indep = all(M.independent(w, wf["composer"], c) for c in comps) and \
            all(M.independent(w, a, b) for a in comps for b in comps if a < b)
        users = len({r["i"] for _, r in st.scan("work_exp") if r["prov"] == wf["declaration"] and r["uses"] > 0})
        kept = len({r["i"] for _, r in st.scan("work_exp") if r["prov"] == wf["declaration"] and r["active"]})
        wfs.append({"workflow": wid, "status": wf["status"], "components": len(wf["components"]),
                    "component_creators": len(comps), "independent": bool(indep), "users": users, "kept": kept,
                    "composer_kind": M.creator_kind(w, wf["composer"])})
    good_wf = [x for x in wfs if x["status"] == "verified" and x["component_creators"] >= 2 and x["independent"]]
    paths = single_judge_paths(root)
    g = grounded_adoptions(w)
    return {"novel": sorted(novel), "undocumented": undocumented, "novel_products": novel_adopt, "workflows": wfs,
            "paths": paths, "grounded": g,
            "metrics": {"single_judge_usefulness_paths": len(paths),
                        "novel_category_adoptions": int(sum(k for _, k in novel_adopt)),
                        "composed_workflows_registered": len(good_wf),
                        "composed_workflows_used": sum(1 for x in good_wf if x["kept"] > 0),
                        "workflow_users": int(sum(x["users"] for x in good_wf)),
                        "novel_categories_undocumented": len(undocumented),
                        "capability_adoptions": g["capability_adoptions"],
                        "grounded_adoptions": g["grounded"],
                        "providers_with_keep_and_drop": g["providers_with_keep_and_drop"]}}


# ====================================================================== SP-6.5b (adoption integrity, developers, sandbox)
FORGERIES = ("TracerCounterForgery", "ShadowPathForgery", "ExistenceCheckForgery", "ContentInspectionForgery",
             "NondeterministicOutputForgery", "BoundaryInspectionForgery")


def forged_adoptions() -> dict:
    """The adoption forgeries of the red team (tracer counter, shadow copy, existence check, reading the source,
    nondeterministic output, inspecting the measurement runs), each in its own fresh world against the real
    software ecosystem, plus the honest control (a dependent that really calls the package must be credited)."""
    from ..redteam.v5 import innovation as RI
    from ..redteam.v5 import run
    ps = [getattr(RI, n)() for n in FORGERIES] + [RI.HonestAdoptionCounts()]
    rep = run(ps)
    rows = {r["name"]: r for r in rep["probes"]}
    forged, credited = 0, {}
    for p in ps[:-1]:
        ev = rows[p.name].get("evidence") or {}
        n = len(ev.get("evaluator_adoptions") or []) + int(int(ev.get("adoption_executions") or 0) > 0)
        credited[p.name] = n
        forged += n
    honest = rows[ps[-1].name]
    errors = [n for n, r in rows.items() if r["result"] in ("error", "not_applicable")]
    return {"forged_adoptions_counted": forged, "forgeries_run": len(ps) - 1, "per_forgery": credited,
            "honest_control_credited": honest["result"] == "held", "probe_errors": errors}


def star_only() -> dict:
    """A package starred by many unrelated principals and locked by a dependent that never runs it: the evaluator
    must count no adoption (stars and declared dependencies are not adoption); the same package then executed by a
    dependent is counted (the measurement is not vacuous)."""
    import crucible
    from ..core.kernel import World
    from . import reference as R
    from .worlds import innovation_spec
    crucible.load_all_modules()
    w = World.create(innovation_spec(17, with_adopters=False))
    try:
        for h in ("eve", "ben", "cat", "dan", "fay", "gus"):
            R.join(w, h, 1_000_000)

        def must(who, tool, **args):
            r = w.execute(f"p:{who}", {"tool": tool, "args": args})
            assert r.ok, (tool, r.error)
            return r.result
        must("eve", "code.create_repo", name="vecmath", visibility="public", license="open")
        must("eve", "code.commit", repo="eve/vecmath", files={"lib.py": R.VEC_LIB, "tests/test_lib.py": R.VEC_TESTS})
        must("eve", "pkg.register", name="vecmath", repo="eve/vecmath", description="vector maths")
        must("eve", "pkg.release", package="vecmath", version="1.0.0")
        for h in ("ben", "cat", "dan", "fay", "gus"):
            must(h, "pkg.star", package="vecmath")
        must("ben", "code.create_repo", name="app")
        must("ben", "code.commit", repo="ben/app", files={"app.py": "def run(a, c):\n    return 1\n"})
        must("ben", "pkg.depend", repo="ben/app", package="vecmath")
        must("ben", "pkg.exec", repo="ben/app", target="app:run", input=None)
        stars = int(w.state.tables["pkg_packages"]["vecmath"]["stars"])
        counted_before = len([a for a in M.adoptions_by_execution(w) if a["package"] == "vecmath"])
        must("cat", "code.create_repo", name="user")
        must("cat", "code.commit", repo="cat/user", files={
            "app.py": "from deps.vecmath.lib import norm\n\n\ndef run(a, c):\n    return norm(a)\n"})
        must("cat", "pkg.depend", repo="cat/user", package="vecmath")
        must("cat", "pkg.exec", repo="cat/user", target="app:run", input=[3, 4])
        counted_after = len([a for a in M.adoptions_by_execution(w) if a["package"] == "vecmath"])
        return {"star_only_adoptions_counted": counted_before, "stars": stars,
                "executed_adoptions_counted": counted_after, "invariants": len(w.check_invariants())}
    finally:
        w.close()


def sandbox_coverage(world) -> dict:
    """Programs of in-world creators run only in the sandbox: no deployment of a creator (agent or population
    developer) is a trusted in-process runtime, and every recorded program execution of theirs went through the
    sandbox bridge (cloud requests of their hosts, pkg executions/CI of their repositories)."""
    st = world.state
    outside, deps = 0, 0
    for _, d in st.scan("deployments"):
        if M.is_creator(world, d["owner"]):
            deps += 1
            outside += int(bool(d.get("trusted")))
    execs = sum(1 for _, e in st.scan("pkg_execs") if M.is_creator(world, e["owner"])) if "pkg_execs" in st.tables else 0
    from ..sandbox.runner import capabilities
    caps = capabilities()
    return {"agent_programs_outside_sandbox": outside, "creator_deployments": deps, "creator_pkg_executions": execs,
            "sandbox_landlock_abi": int(caps.get("landlock_abi") or 0), "sandbox_seccomp": bool(caps.get("seccomp"))}


def developer_adoptions(world) -> dict:
    """Adoptions by execution (attested boundary) in which an autonomous population developer's repository runs a
    package of another, independent creator; and maintenance: releases after the first, upgrades, redeploys."""
    st = world.state
    from ..substrate import ids as _ids
    auto = [a for a in M.adoptions_by_execution(world)
            if _ids.index(a["owner"]) is not None and M.is_creator(world, st.tables["pkg_packages"][a["package"]]["owner"])]
    rel = [r for _, r in st.scan("pkg_releases")] if "pkg_releases" in st.tables else []
    later = sum(1 for r in rel if r["version"] != "1.0.0")
    upgrades = sum(1 for _, c in st.scan("pkg_compat") if c["compatible"]) if "pkg_compat" in st.tables else 0
    return {"autonomous_developer_adoptions": len(auto), "later_releases": later, "compatible_upgrades": upgrades,
            "packages": len(st.tables.get("pkg_packages", {}))}


# ====================================================================== SP-6.4b (need mechanisms from the one graph)
# The example mechanisms of Part 6.4. LISTED are the five SP-6.4b names; the other two (reliability bottleneck,
# computing demand) are reported separately and never counted towards need_mechanism_examples.
LISTED = ("workflow_change", "transport_market_access", "coordination", "content_communities_tools",
          "education_skill_availability")
EXTRA = ("reliability_bottleneck", "computing_demand")
NOT_LIVE = {
    "coordination": "no module models organisational coordination load (organisations' size, overhead or "
                    "coordination failures are not world state), so the graph has nothing to read",
    "content_communities_tools": "communities hold no tool requirement or moderation load the population produces "
                                 "(population members join, leave and post; nobody reports or moderates)",
    "education_skill_availability": "the education module (domains/learning.py) is inert in civilisations (learn n=0) "
                                    "and its learners never study in a standalone world (learn.study raises "
                                    "KeyError 'id' at learning.py:324); no relation provider reads it yet",
}


def _first_seen(history: list[dict]) -> tuple[dict, dict]:
    first, rec = {}, {}
    for h in history:
        for nid, n in h["needs"].items():
            first.setdefault(nid, h["t"])
            rec.setdefault(nid, n)
    return first, rec


def need_mechanisms(A: dict) -> dict:
    """Needs of the factual civilisation, labelled by the one relation graph (`innovation.relations`: providers that
    read the world's state, generic rules), that arose after genesis, classified by the mechanism the derivation
    path shows (rule and the providers of its edges; no node name is consulted):

    * workflow_change - a bottleneck on a later step of a multi-step work activity, first labelled after holders
      adopted a provider for an earlier step, and never labelled in the arm where nobody builds for that step;
    * transport_market_access - scarcity of a transport link (freight provider: load against daily capacity, or
      shipments waiting for capacity) giving access between regions;
    * reliability_bottleneck - scarcity of a service's rented compute under its users' load (cloud provider);
    * computing_demand - scarcity of a compute provider's cores (compute_market provider)."""
    f = A["factual"]
    hist = f["history"]
    t0 = hist[0]["t"]
    first, rec = _first_seen(hist)
    first_rely: dict[str, int] = {}
    for e in f["world"].log:
        if e.type == "work.rely" and e.payload.get("active") and e.payload.get("need"):
            first_rely.setdefault(e.payload["need"], e.t)
    found: dict[str, list] = {k: [] for k in LISTED + EXTRA}
    for nid in sorted(first):
        if first[nid] <= t0 + DAY:
            continue
        n = rec[nid]
        prov = [p.get("provider") for p in n["path"]]
        if n["rule"] == "bottleneck" and prov == ["work"]:
            act = n["path"][0]["src"].split("/", 1)[1]
            steps = act.split("+")
            tgt = n["target"].split("/", 1)[1]
            if tgt in steps and steps.index(tgt) > 0:
                earlier = steps[:steps.index(tgt)]
                ta = min((first_rely[s] for s in earlier if s in first_rely), default=None)
                arm = CX.removal_arm_for(earlier[0])
                if ta is not None and ta <= first[nid] and nid not in A[arm]["outcome"]["needs_ever"]:
                    found["workflow_change"].append({"need": nid, "first_day": first[nid] // DAY,
                                                     "earlier_step_adopted_day": ta // DAY, "absent_in": arm})
        elif n["rule"] == "scarcity":
            row = {"need": nid, "first_day": first[nid] // DAY, "holders": n["holders"]}
            if prov[0] == "freight":
                found["transport_market_access"].append(row)
            elif prov[0] == "compute_market":
                found["computing_demand"].append(row)
            elif prov[0] == "cloud":
                found["reliability_bottleneck"].append(row)
    return found


def designer_supplied_demand(A: dict) -> dict:
    """Paths by which demand would be supplied by the designer instead of arising from world state: labelled needs
    (any arm, any day) whose path uses relations given as data (provider `data`), measured worlds whose spec
    carries relations or relation observables as data, and world configuration that names a workflow's composer,
    components or declaration (a demand planted for the combination)."""
    import json as _json
    data_paths = sum(1 for r in A.values() for h in r["history"] for n in h["needs"].values()
                     if any(p.get("provider") == "data" for p in n["path"]))
    w = A["factual"]["world"]
    spec_txt = _json.dumps(w.spec.to_dict(), sort_keys=True)
    data_specs = int(bool(w.spec.params.get("relations") or w.spec.params.get("relation_nodes")))
    st = w.state
    named = 0
    for _, wf in (st.scan("cap_workflows") if "cap_workflows" in st.tables else ()):
        names = [wf["declaration"], wf["host"]] + [c["host"] for c in wf["components"]] + \
            [c["declaration"] for c in wf["components"]]
        for x in names:
            if x and f'"{x}"' in spec_txt:
                named += 1
    return {"designer_supplied_demand_paths": int(data_paths + data_specs + named), "data_relation_paths": data_paths,
            "data_relation_specs": data_specs, "configured_combination_names": named}


def novel_combinations(A: dict) -> dict:
    """Agent-created combinations of capabilities: verified workflow registry objects (`caps.compose`) of a population
    developer combining live services of independent other creators, the individuals who rely on them, and the
    future state they change - hand-work minutes of the composed activity in the factual arm against the arm in
    which nobody builds for the generated categories (no components, no workflow)."""
    w = A["factual"]["world"]
    st = w.state
    fo, ro = A["factual"]["outcome"], A["ban_novel"]["outcome"]
    rows = []
    for wid, wf in (st.scan("cap_workflows") if "cap_workflows" in st.tables else ()):
        comps = {c["provider"] for c in wf["components"]}
        indep = all(M.independent(w, wf["composer"], c) for c in comps) and \
            all(M.independent(w, a, b) for a in comps for b in comps if a < b)
        relied = {r["i"] for _, r in st.scan("work_exp") if r["prov"] == wf["declaration"] and r["active"]}
        need = next((r["step_need"] for _, r in st.scan("work_exp") if r["prov"] == wf["declaration"]
                     and r["step_need"]), None)
        delta = (CX.minutes_for(ro, need) - CX.minutes_for(fo, need)) if need else 0.0
        rows.append({"workflow": wid, "status": wf["status"], "components": len(comps), "independent": bool(indep),
                     "relied": len(relied), "need": need, "hand_minutes_saved_vs_removal": round(float(delta), 1),
                     "composer_kind": M.creator_kind(w, wf["composer"]), "created_day": int(wf.get("t", 0)) // DAY})
    good = [r for r in rows if r["status"] == "verified" and r["components"] >= 2 and r["independent"]
            and r["relied"] > 0 and r["hand_minutes_saved_vs_removal"] > 0 and r["composer_kind"] == "population"]
    return {"rows": rows, "novel_combinations_changing_state": len(good)}


def sp64b_measure(A: dict) -> dict:
    found = need_mechanisms(A)
    dem = designer_supplied_demand(A)
    nc = novel_combinations(A)
    listed = [k for k in LISTED if found[k]]
    m = {"need_mechanism_examples": len(listed),
         **{f"mechanism_{k}": len(found[k]) for k in LISTED + EXTRA},
         "extra_mechanisms_demonstrated": sum(1 for k in EXTRA if found[k]),
         "designer_supplied_demand_paths": dem["designer_supplied_demand_paths"],
         "data_relation_paths": dem["data_relation_paths"],
         "configured_combination_names": dem["configured_combination_names"],
         "novel_combinations_changing_state": nc["novel_combinations_changing_state"]}
    missing = {k: NOT_LIVE.get(k, "did not arise in the measured run") for k in LISTED if not found[k]}
    return {"metrics": m, "found": found, "missing": missing, "combinations": nc["rows"]}
