"""The numbered experiments (Part 12). Each writes a JSON result under ``docs/experiments/`` and prints a summary.

    superlearner experiment baselines --checkpoint RUN/ckpt/latest.pt     Part 5: the seven-way comparison
    superlearner experiment a --checkpoint CKPT                           A: real actions, a functioning service
    superlearner experiment b --run-dir RUN                               B: before / after RL, parameter change
    superlearner experiment c --checkpoint CKPT                           C: transfer to unfamiliar worlds + adaptation
    superlearner experiment d --config CFG                                D: retention after unrelated training
    superlearner experiment worldmodel --run-dir RUN                      learned world model: held-out accuracy, planning
    superlearner experiment selfplay --checkpoint CKPT                    several learners in one persistent world

Experiments E and F are the Meta-Researcher's own runs (`superlearner meta-search`); `experiment e` and
`experiment f` summarise their ledgers. Every number comes from the evaluator on authoritative world state.
"""
from __future__ import annotations

import copy
import glob
import gzip
import json
import os
import pickle
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "docs", "experiments")


def _save(name: str, res: dict) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".json")
    json.dump(res, open(p, "w"), indent=1, default=str)
    print(f"wrote {p}")
    return p


def _trainer(ckpt_path: str | None, cfg: dict | None = None, run_dir: str | None = None, workers: int | None = None):
    import torch
    from .cli import RUNS, resolve_llm
    from .train.trainer import Trainer
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path else None
    cfg = copy.deepcopy(cfg or ck["cfg"])
    cfg = resolve_llm(cfg)
    if workers is not None:
        cfg["workers"] = workers
    tr = Trainer(cfg, run_dir or os.path.join(RUNS, "exp_" + time.strftime("%m%d_%H%M%S")))
    if ckpt_path:
        tr.load(ckpt_path)
    return tr, ck


def heldout_tasks(tr, n: int, offset: int = 3000) -> list:
    """Held-out comparison set: unseen seeds; half the worlds from families the controller trained in, half from
    world families it never saw (entrenched incumbents, fragmented networks)."""
    from .world.tasks import task_list
    arms = [x.key for x in tr.curriculum.arms]
    trained = tr.cfg["curriculum"].get("world_families") or ["tiny", "bazaar"]
    fams = [f for pair in zip(trained, ["entrenched_incumbents", "fragmented_networks"]) for f in pair]
    return task_list(arms, fams, n, "select", offset)


def _row(r: dict) -> dict:
    return {k: r.get(k) for k in ("episodes", "success", "success_ci", "progress", "valid_rate", "decisions", "calls",
                                  "failures", "llm_propose", "llm_gen", "by_arm", "wall_s")}


# ================================================================================================ Part 5 baselines
def raw_llm_episode(core, llm, max_new_tokens: int = 256) -> dict:
    """Baseline 1: the frozen LLM alone, through Crucible's ORIGINAL agent interface (`crucible.agents.llm.LLMPolicy`:
    one JSON tool call per turn, the full tool list rendered into the prompt, no grounding layer, no controller)."""
    from . import ensure_crucible
    ensure_crucible()
    from crucible.agents.llm import GenResult as CG, LLMPolicy

    class Backend:
        name = getattr(llm, "name", "llm")

        def generate(self, messages, max_new_tokens=256, temperature=0.0, seed=None):
            g = llm.generate(messages, max_new_tokens, 0)
            return CG(g.text, g.prompt_tokens, g.gen_tokens, False, g.wall_s)
    info = core.info()
    pol = LLMPolicy(goal=info["goal"], backend=Backend(), max_new_tokens=max_new_tokens, tool_order="interleave",
                    tool_chars=24000)
    pol.reset(info["actor"], np.random.default_rng(0))
    tools = core.tools(True)
    n_ok = n_calls = 0
    fails: dict = {}
    while True:
        obs = core.observe()
        tl = core.tools()
        tools = tl if tl is not None else tools
        a = pol.act(obs, tools)
        if a is None:
            core.wait(3600)
        else:
            r = core.execute(a["tool"], a.get("args", {}))
            pol.observe_result(a, r)
            n_calls += 1
            n_ok += int(bool(r.get("ok")))
            if not r.get("ok"):
                c = (r.get("error") or {}).get("code", "error")
                fails[c] = fails.get(c, 0) + 1
        ed = core.end_decision()
        if ed["terminated"] or ed["truncated"]:
            break
    s = core.summary()
    s.update(agent={"calls": n_calls, "sent": n_calls, "ok": n_ok, "valid_rate": n_ok / max(1, n_calls), "failures": fails},
             llm={"propose": pol.stats["calls"], "gen": 0, "propose_fail": pol.stats["parse_failures"], "gen_fail": 0},
             heads={}, chooser="frozen LLM alone (raw interface)", tools_used=[])
    return {"summary": s, "decisions": [], "emb_new": []}


def run_raw_llm(tasks: list, llm_spec: dict, core_cfg: dict, threads: int = 8) -> dict:
    import concurrent.futures as cf
    from .llm.server import connect
    from .train.trainer import summarize
    from .world.core import CoreConfig, SessionCore

    def one(spec):
        try:
            return raw_llm_episode(SessionCore(spec, CoreConfig(**core_cfg)), connect(llm_spec))
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        eps = list(ex.map(one, tasks))
    res = summarize([e for e in eps if "summary" in e])
    res["errors"] = [e["error"] for e in eps if "error" in e][:3]
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def exp_baselines(a) -> int:
    """Seven conditions on the same held-out tasks, same world mechanics, same frozen model, same budgets."""
    import torch
    from .algo.ppo import flat_params
    run = os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint)))
    tr, ck = _trainer(a.checkpoint, workers=a.workers)
    n = a.episodes or 64
    try:
        tasks = heldout_tasks(tr, n)
        res: dict = {"checkpoint": a.checkpoint, "tasks": {"n": len(tasks), "split": "select", "arms": sorted({"/".join(map(str, t.arm)) for t in tasks}),
                                                           "world_families": sorted({t.world_family for t in tasks})},
                     "core": tr.cfg["core"], "conditions": {}}
        c = res["conditions"]
        c["4 CAM after RL"] = _row(tr.evaluate(tasks, "greedy", "b4_after_rl", save=False))
        theta_rl = flat_params(tr.model).clone()
        mem_state, lib_state = tr.memory.state_dict(), tr.library.state_dict()
        # 5: learned weights restored, episodic memory and tool library reset
        tr.memory.items, tr.library.tools = [], {}
        tr.pool.sync([], tr.library.state_dict(), reset=True)
        tr.mem_synced = 0
        c["5 weights restored, memory reset"] = _row(tr.evaluate(tasks, "greedy", "b5_weights_only", save=False))
        # 3: warm-started CAM before RL
        bc = os.path.join(run, "ckpt", "bc.pt")
        if os.path.exists(bc):
            tr.load(bc)
            c["3 warm-started CAM before RL"] = _row(tr.evaluate(tasks, "greedy", "b3_before_rl", save=False))
            res["param_distance_rl_vs_warm_start"] = float((flat_params(tr.model) - theta_rl).norm())
        # 6: memory and tools preserved, trainable weights reset to a fresh initialisation
        from .cam.model import CAMController, ModelConfig
        torch.manual_seed(12345)
        tr.model.load_state_dict(CAMController(ModelConfig(**tr.cfg["model"])).state_dict())
        tr.memory.load_state_dict(mem_state)
        tr.library.load_state_dict(lib_state)
        tr.mem_synced = 0
        tr.pool.sync(tr.memory.items, tr.library.state_dict(), reset=True)
        tr.mem_synced = len(tr.memory.items)
        c["6 memory preserved, weights reset"] = _row(tr.evaluate(tasks, "greedy", "b6_memory_only", save=False))
        res["memory_items"], res["library_tools"] = len(mem_state["items"]), len(lib_state["tools"])
        # 7: competent scripted control (through the same action interface)
        c["7 scripted control"] = _row(tr.evaluate(tasks, "teacher:auto", "b7_scripted", save=False))
        c["0 uniform random (same action space)"] = _row(tr.evaluate(tasks, "random", "b0_random", save=False))
        llm_ok = tr.cfg["llm"].get("kind") == "remote"
        if llm_ok and not getattr(a, "no_llm", False):
            # 2: frozen LLM with the grounding layer, no neural controller
            c["2 frozen LLM + grounding, no learning"] = _row(tr.evaluate(tasks, "llm", "b2_llm_grounded", agent={"use_llm": True, "llm_budget": 6}, save=False))
            # 1: frozen LLM alone through Crucible's original interface
            raw_core = dict(tr.cfg["core"], max_decisions=min(24, int(tr.cfg["core"].get("max_decisions", 40))))
            c["1 frozen LLM alone"] = _row(run_raw_llm(tasks[: min(len(tasks), 16)], tr.cfg["llm"], raw_core, threads=3))
            c["1 frozen LLM alone"]["note"] = "first 16 tasks, 24 decisions each (one tool call per decision)"
        res["llm"] = tr.llm_totals()
    finally:
        tr.close()
    for k in sorted(c):
        print(f"{k:42s} success {c[k]['success']:.3f} {c[k]['success_ci']}  valid {c[k]['valid_rate']:.2f}  n={c[k]['episodes']}")
    _save("baselines_" + os.path.basename(run), res)
    return 0


# ================================================================================================ B
def exp_b(a) -> int:
    """Before / after RL on held-out tasks, learning curve, and the evidence that the weights changed."""
    import torch
    from .algo.ppo import flat_params
    from .train.report import report
    run = a.run_dir
    rep = report(run)
    tr, _ = _trainer(os.path.join(run, "ckpt", "latest.pt"), workers=a.workers)
    n = a.episodes or 96
    try:
        tasks = heldout_tasks(tr, n)
        after = tr.evaluate(tasks, "greedy", "expB_after", save=False)
        th_after = flat_params(tr.model).clone()
        tr.load(os.path.join(run, "ckpt", "bc.pt"))
        before = tr.evaluate(tasks, "greedy", "expB_before", save=False)
        th_before = flat_params(tr.model)
        diff = th_after - th_before
        groups: dict = {}
        sd_b = torch.load(os.path.join(run, "ckpt", "bc.pt"), map_location="cpu", weights_only=False)["model"]
        sd_a = torch.load(os.path.join(run, "ckpt", "latest.pt"), map_location="cpu", weights_only=False)["model"]
        for k in sd_a:
            g = k.split(".")[0]
            d = (sd_a[k].float() - sd_b[k].float())
            e = groups.setdefault(g, [0.0, 0.0, 0])
            e[0] += float((d ** 2).sum())
            e[1] += float((sd_b[k].float() ** 2).sum())
            e[2] += int((d != 0).sum())
    finally:
        tr.close()
    k1, n1, k0, n0 = round(after["success"] * after["episodes"]), after["episodes"], round(before["success"] * before["episodes"]), before["episodes"]
    rng = np.random.default_rng(0)
    boots = rng.binomial(n1, k1 / n1, 20000) / n1 - rng.binomial(n0, k0 / n0, 20000) / n0
    res = {"run": run, "tasks": {"n": len(tasks), "split": "select: unseen seeds; half the worlds from families never trained on",
                                  "world_families": sorted({t.world_family for t in tasks})},
           "before_rl": _row(before), "after_rl": _row(after),
           "difference": {"success": after["success"] - before["success"],
                          "ci95_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                          "p_after_better": float((boots > 0).mean())},
           "parameters": {"trainable": rep["params"]["trainable"], "l2_distance_before_to_after": float(diff.norm()),
                          "fraction_of_weights_changed": float((diff != 0).float().mean()),
                          "relative_change_by_component": {g: round((v[0] / max(v[1], 1e-12)) ** 0.5, 5) for g, v in groups.items()},
                          "updates_that_changed_weights": rep["weights"]["updates_that_changed_weights"], "rl_iterations": rep["rl_iters"]},
           "learning_curve_during_training": rep["eval_curve"], "aulc": rep["aulc_eval_success"],
           "experience": {"env_decisions": rep["env_decisions"], "episodes": rep["episodes"], "demos": rep["demos"]},
           "cost": rep["cost"]}
    print(json.dumps({k: res[k] for k in ("before_rl", "after_rl", "difference")}, indent=1, default=str)[:2500])
    _save("exp_b_" + os.path.basename(os.path.normpath(run)), res)
    return 0


def exp_b_multi(a) -> int:
    """Experiment B over several seeds: every seed has ONE warm-start checkpoint from which the arms branch.

    For each seed the warm start and each arm's final checkpoint are evaluated greedily on the same held-out task
    list (unseen seeds, half of the worlds from unfamiliar families). Reported: success before and after RL per
    arm, paired differences over seeds with bootstrap intervals, the in-training learning curves and their AULC,
    experience used, and how far the weights moved."""
    import torch
    from .cli import RUNS
    from .meta.compare import compare
    from .train.report import aulc, read_jsonl
    arms = {"PPO alone": "exp_b_fewdemo", "PPO + replay of own successes": "exp_b_fewdemo_sil"}
    seeds = [int(x) for x in str(getattr(a, "seeds", None) or "0,1,2").split(",")]
    warm0 = os.path.join(RUNS, f"warm_s{seeds[0]}", "ckpt", "bc.pt")
    tr, _ = _trainer(warm0, workers=a.workers)
    n = a.episodes or 96
    res: dict = {"arms": arms, "seeds": seeds, "per_seed": {}, "tasks": {}}
    try:
        tasks = heldout_tasks(tr, n)
        res["tasks"] = {"n": len(tasks), "split": "select: unseen seeds", "world_families": sorted({t.world_family for t in tasks}),
                        "arms": sorted({"/".join(map(str, t.arm)) for t in tasks})}
        for sd in seeds:
            warm = os.path.join(RUNS, f"warm_s{sd}", "ckpt", "bc.pt")
            tr.load(warm)
            before = tr.evaluate(tasks, "greedy", f"B_before_s{sd}", save=False)
            sd_b = torch.load(warm, map_location="cpu", weights_only=False)["model"]
            row = {"before_rl": _row(before), "arms": {}}
            for label, name in arms.items():
                run = os.path.join(RUNS, f"{name}_s{sd}")
                ck = os.path.join(run, "ckpt", "latest.pt")
                if not os.path.exists(ck):
                    continue
                tr.load(ck)
                after = tr.evaluate(tasks, "greedy", f"B_after_{name}_s{sd}", save=False)
                sd_a = torch.load(ck, map_location="cpu", weights_only=False)
                d2 = b2 = 0.0
                groups: dict = {}
                for k in sd_a["model"]:
                    d = (sd_a["model"][k].float() - sd_b[k].float())
                    d2 += float((d ** 2).sum())
                    b2 += float((sd_b[k].float() ** 2).sum())
                    g = groups.setdefault(k.split(".")[0], [0.0, 0.0])
                    g[0] += float((d ** 2).sum())
                    g[1] += float((sd_b[k].float() ** 2).sum())
                ev = [e for e in read_jsonl(os.path.join(run, "eval.jsonl")) if e.get("tag") == "eval"]
                rl = [m for m in read_jsonl(os.path.join(run, "metrics.jsonl")) if m.get("phase") == "rl"]
                row["arms"][label] = {"after_rl": _row(after), "delta_success": after["success"] - before["success"],
                                      "in_training_curve": [(e["env_decisions"], round(e["success"], 4)) for e in ev],
                                      "aulc_in_training": aulc([e["env_decisions"] for e in ev], [e["success"] for e in ev]),
                                      "env_decisions": sd_a["env_decisions"], "episodes": sd_a["episodes"], "rl_iterations": sd_a["iter"],
                                      "weights_l2_distance_from_warm_start": d2 ** 0.5, "relative_change": (d2 / max(b2, 1e-12)) ** 0.5,
                                      "relative_change_by_component": {g: round((v[0] / max(v[1], 1e-12)) ** 0.5, 4) for g, v in groups.items()},
                                      "updates_that_changed_weights": int(sum(bool(m.get("weights_changed")) for m in rl)),
                                      "rollout_s": round(sum(m.get("rollout_s", 0) for m in rl)), "update_s": round(sum(m.get("update_s", 0) for m in rl)),
                                      "replay_steps": int(sum(m.get("replay_steps", 0) for m in rl))}
            res["per_seed"][sd] = row
            print(f"seed {sd}: before {before['success']:.3f} | " + " | ".join(
                f"{l}: {r['after_rl']['success']:.3f} ({r['delta_success']:+.3f})" for l, r in row["arms"].items()), flush=True)
    finally:
        tr.close()
    ok = [sd for sd in seeds if len(res["per_seed"].get(sd, {}).get("arms", {})) == len(arms)]
    agg: dict = {"seeds_complete": ok}
    if ok:
        bef = {sd: res["per_seed"][sd]["before_rl"]["success"] for sd in ok}
        for label in arms:
            aft = {sd: res["per_seed"][sd]["arms"][label]["after_rl"]["success"] for sd in ok}
            agg[f"{label}: after - before (held-out success)"] = compare(aft, bef)
        l1, l2 = list(arms)
        agg[f"{l2} - {l1}: held-out success after RL"] = compare(
            {sd: res["per_seed"][sd]["arms"][l2]["after_rl"]["success"] for sd in ok},
            {sd: res["per_seed"][sd]["arms"][l1]["after_rl"]["success"] for sd in ok})
        agg[f"{l2} - {l1}: AULC of the in-training curve"] = compare(
            {sd: res["per_seed"][sd]["arms"][l2]["aulc_in_training"] for sd in ok},
            {sd: res["per_seed"][sd]["arms"][l1]["aulc_in_training"] for sd in ok})
        by_arm: dict = {}
        for sd in ok:
            for cond, r in (("before", res["per_seed"][sd]["before_rl"]),) + tuple((l, res["per_seed"][sd]["arms"][l]["after_rl"]) for l in arms):
                for arm, v in r["by_arm"].items():
                    by_arm.setdefault(arm, {}).setdefault(cond, []).append(v)
        agg["mean_success_by_task_arm"] = {arm: {c: round(float(np.mean(v)), 3) for c, v in d.items()} for arm, d in sorted(by_arm.items())}
    res["aggregate"] = agg
    print(json.dumps(agg, indent=1, default=str)[:3000])
    _save("exp_b_multi", res)
    return 0


# ================================================================================================ C
def exp_c(a) -> int:
    """Transfer: evaluate without new training on unfamiliar seeds / world families / objectives, then adapt with
    a small amount of additional experience and compare adaptation speed with the warm start."""
    run = os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint)))
    tr, _ = _trainer(a.checkpoint, workers=a.workers)
    n = a.episodes or 64
    from .world.tasks import FINAL_FAMILIES, SELECT_FAMILIES, TRAIN_FAMILIES
    try:
        arms = [x.key for x in tr.curriculum.arms]
        trained_fams = tr.cfg["curriculum"].get("world_families") or list(TRAIN_FAMILIES)
        sets = {
            "training families, unseen seeds": tr.eval_tasks(n, "select", arms),
            "unfamiliar world families (validation)": _tasks(arms, [f for f in SELECT_FAMILIES if f not in trained_fams], n, "select", 6000),
            "unfamiliar world families (test: fragile infrastructure, novel needs)": _tasks(arms, [f for f in FINAL_FAMILIES if f not in TRAIN_FAMILIES], n, "final", 800_000),
            "harder objective (service level 1, never trained)": _tasks(["service/1/listed", "service/1/deployed"], trained_fams, max(16, n // 2), "select", 7000),
        }
        res: dict = {"checkpoint": a.checkpoint, "trained_world_families": trained_fams, "zero_shot": {}}
        for name, tasks in sets.items():
            r = tr.evaluate(tasks, "greedy", "expC", save=False)
            res["zero_shot"][name] = _row(r)
            print(f"zero-shot {name:70s} success {r['success']:.3f} {r['success_ci']} n={r['episodes']}")
        # adaptation: a few iterations of RL on the unfamiliar families, starting from the trained and the warm-start weights
        new_fams = [f for f in SELECT_FAMILIES if f not in trained_fams] or list(SELECT_FAMILIES)
        adapt_eval = sets["unfamiliar world families (validation)"]
        res["adaptation"] = {}
        for label, ck in (("from RL-trained weights", a.checkpoint), ("from warm start (before RL)", os.path.join(run, "ckpt", "bc.pt"))):
            if not os.path.exists(ck):
                continue
            tr.load(ck)
            tr.curriculum.world_families = [f for f in new_fams if f in ("entrenched_incumbents", "fragmented_networks")] or new_fams
            tr.curriculum.split = "select"            # the unfamiliar families exist only in the held-out splits;
            tr.curriculum.counter = 500_000           # adaptation uses a seed range far from the evaluation lists
            curve = [{"decisions": 0, "success": tr.evaluate(adapt_eval, "greedy", "expC_adapt", save=False)["success"]}]
            d0 = tr.env_decisions
            for _ in range(int(getattr(a, "adapt_iters", None) or 4)):
                tr.train_iter()
                curve.append({"decisions": tr.env_decisions - d0, "success": tr.evaluate(adapt_eval, "greedy", "expC_adapt", save=False)["success"]})
            res["adaptation"][label] = curve
            print(f"adaptation {label}: {[(c['decisions'], round(c['success'], 3)) for c in curve]}")
    finally:
        tr.close()
    _save("exp_c_" + os.path.basename(run), res)
    return 0


def _tasks(arms, fams, n, split, offset):
    from .world.tasks import task_list
    return task_list(arms, list(fams), n, split, offset)


# ================================================================================================ D
def exp_d(a) -> int:
    """Retention: learn skill set A, then train only on unrelated tasks B, then test A again.

    Phase A is run once; four strategies for phase B branch from the same checkpoint and are compared as competing
    hypotheses: no protection; replay of the agent's own earlier successes (the buffer filled in phase A is kept);
    rehearsal (A tasks sampled at a low rate); a behaviour-cloning regulariser on A's demonstrations."""
    import copy as _copy
    from .cli import RUNS, load_config
    from .train.curriculum import Curriculum
    from .train.trainer import merge
    base = load_config(a.config)
    A = [{"family": "found"}, {"family": "fund"}, {"family": "list", "start": "deployed"}]
    Bset = [{"family": "repair"}, {"family": "deploy", "start": "coded"}, {"family": "service", "start": "listed"}]
    armsA = ["found/0/scratch", "fund/0/scratch", "list/0/deployed"]
    armsB = ["repair/0/scratch", "deploy/0/coded", "service/0/listed"]
    itersA, itersB = int(base.get("retention_iters_a", 8)), int(base.get("retention_iters_b", 10))
    cfg = merge(base, {"name": "exp_d", "curriculum": {"arms": A, "adaptive": True, "gate": False}, "eval_every": 0, "ckpt_every": 0})
    rd = a.run_dir or os.path.join(RUNS, "exp_d")
    tr, _ = _trainer(None, cfg, rd, a.workers)
    out: dict = {"skill_set_A": armsA, "unrelated_set_B": armsB, "iterations": {"A": itersA, "B": itersB}, "strategies": {}}
    try:
        tr.collect_demos(int(cfg["demos"]["episodes"]), A)             # demonstrations for A only: B is learned by RL
        demosA = list(tr.demos)
        tr.pretrain_bc(int(cfg["demos"]["bc_epochs"]))
        n = a.episodes or 48
        tA, tB = tr.eval_tasks(n, "select", armsA), tr.eval_tasks(n, "select", armsB)
        a0 = tr.evaluate(tA, "greedy", "D_A_warm", save=False)
        for _ in range(itersA):
            tr.train_iter()
        a1, b1 = tr.evaluate(tA, "greedy", "D_A_after_A", save=False), tr.evaluate(tB, "greedy", "D_B_after_A", save=False)
        out["phase_A"] = {"A_at_warm_start": _row(a0), "A_after_learning_A": _row(a1), "B_before_any_B_training": _row(b1)}
        print(f"phase A: A {a0['success']:.3f} -> {a1['success']:.3f}; B (untrained) {b1['success']:.3f}", flush=True)
        ck = tr.save("after_A.pt")
        buf = list(getattr(tr.algo, "buffer", []))
        counter = tr.curriculum.counter
        for name, arms_b, keep_buffer, extra in (
                ("no protection", Bset, False, {}),
                ("replay of own earlier successes", Bset, True, {}),
                ("rehearsal (A sampled at a low rate)", Bset + A, False, {}),
                ("demonstration regulariser on A (bc_coef 0.5)", Bset, False, {"bc_coef": 0.5})):
            tr.load(ck)
            tr.algo.buffer = list(buf) if keep_buffer else []
            tr.algo.c.update({"bc_coef": 0.0}, **extra)
            cur = Curriculum(arms_b, cfg["curriculum"].get("world_families"), "train", cfg["seed"] + 1, gate=False)
            cur.counter = counter + 200_000
            if len(arms_b) > len(Bset):
                w = np.array([1.0] * len(Bset) + [0.15] * len(A))
                cur.weights = lambda w=w: w / w.sum()
            tr.curriculum = cur
            tr.demos = _copy.copy(demosA) if extra.get("bc_coef") else []
            for _ in range(itersB):
                tr.train_iter()
            a2, b2 = tr.evaluate(tA, "greedy", "D_A_after_B", save=False), tr.evaluate(tB, "greedy", "D_B_after_B", save=False)
            out["strategies"][name] = {"A_after_training_on_B": _row(a2), "B_after_training_on_B": _row(b2),
                                       "forgetting_on_A": a1["success"] - a2["success"], "gain_on_B": b2["success"] - b1["success"]}
            print(f"{name:48s} A: {a1['success']:.3f} -> {a2['success']:.3f}   B: {b1['success']:.3f} -> {b2['success']:.3f}", flush=True)
    finally:
        tr.close()
    _save("exp_d_retention", out)
    return 0


# ================================================================================================ A
def exp_a(a) -> int:
    """Real actions: build and operate a functioning service from scratch in new worlds; inspect the artifacts."""
    tr, _ = _trainer(a.checkpoint, workers=a.workers)
    n = a.episodes or 24
    try:
        fams = tr.cfg["curriculum"].get("world_families") or ["tiny", "bazaar"]
        tasks = _tasks(["deploy/0/scratch", "service/0/scratch"], fams, n, "select", 9000)
        res: dict = {"checkpoint": a.checkpoint, "tasks": [t.key for t in tasks], "conditions": {}}
        for label, mode, agent in (("CAM controller (greedy)", "greedy", {}), ("CAM controller (sampled)", "sample", {}),
                                   ("scripted control through the same interface", "teacher:auto", {}),
                                   ("frozen LLM + grounding, no controller", "llm", {"llm_budget": 6})):
            tr.sync_workers()
            jobs = [{"spec": s.to_dict(), "mode": mode, "seed": 500 + i, "episode_id": f"A{i}", "learn": False, "human": True,
                     "keep_decisions": False, "agent": agent} for i, s in enumerate(tasks)]
            eps = tr.absorb(tr.pool.run(jobs))
            from .train.trainer import summarize
            r = summarize(eps)
            arts = []
            for e in eps:
                steps = [d["human"] for d in e["decisions"] if "human" in d]
                arts.append({"task": e["summary"]["task"], "success": e["summary"]["success"], "progress": e["summary"]["best_progress"],
                             "decisions": e["summary"]["decisions"], "failures": e["summary"]["agent"]["failures"],
                             "llm": e["summary"]["llm"], "actions": [s["action"] + ("" if s["ok"] else f" !{s['fclass']}") for s in steps][:40]})
            res["conditions"][label] = {"summary": _row(r), "episodes": arts}
            tr.save_traj("expA_" + mode.replace(":", "_"), eps[:6])
            print(f"{label:48s} success {r['success']:.3f} {r['success_ci']} valid {r['valid_rate']:.2f} failures {r['failures']}")
    finally:
        tr.close()
    _save("exp_a_real_actions", res)
    return 0


# ================================================================================================ world model
def exp_worldmodel(a) -> int:
    import torch
    from .cam.worldmodel import WorldModel, evaluate_model, fit, transitions
    run = a.run_dir
    tr, _ = _trainer(os.path.join(run, "ckpt", "latest.pt"), workers=a.workers)
    try:
        files = sorted(glob.glob(os.path.join(run, "replay", "iter*.pkl.gz")))
        eps = []
        for f in files:
            d = pickle.load(gzip.open(f, "rb"))
            for k, v in d["emb"]:
                tr.store.put(k, v)
            eps += d["episodes"]
        if len(eps) < 8:
            raise SystemExit("this run stored no training episodes (set store_episodes_iters in its config)")
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(eps))
        n_te = max(4, len(eps) // 5)
        te, trn = [eps[i] for i in idx[:n_te]], [eps[i] for i in idx[n_te:]]
        dev = tr.device
        d_tr, d_te = transitions(tr.model, trn, tr.store, dev), transitions(tr.model, te, tr.store, dev)
        wm = WorldModel(tr.model.cfg.d)
        hist = fit(wm, d_tr, epochs=40, device=dev)
        res = {"run": run, "train_transitions": int(len(d_tr["ok"])), "heldout_transitions": int(len(d_te["ok"])),
               "heldout_episodes": len(te), "train_loss_first_last": [hist[0], hist[-1]],
               "heldout": evaluate_model(wm, d_te, d_tr, dev), "train_fit": evaluate_model(wm, d_tr, d_tr, dev),
               "parameters": sum(p.numel() for p in wm.parameters())}
        torch.save({"wm": wm.state_dict(), "d": tr.model.cfg.d}, os.path.join(run, "worldmodel.pt"))
        print(json.dumps(res["heldout"], indent=1))
        # planning: does one-step lookahead with the model change real success at equal environment budget?
        tasks = tr.eval_tasks(a.episodes or 64, "select")
        base = tr.evaluate(tasks, "greedy", "wm_model_free", save=False)
        plan = tr.evaluate(tasks, "greedy", "wm_planning", agent={"plan_with_world_model": os.path.join(run, "worldmodel.pt")}, save=False)
        res["planning"] = {"model_free": _row(base), "with_one_step_lookahead": _row(plan),
                           "difference": plan["success"] - base["success"],
                           "note": "same tasks, same environment decisions budget; lookahead adds one world-model forward per decision"}
        print(f"model-free {base['success']:.3f} {base['success_ci']} valid {base['valid_rate']:.2f} | "
              f"lookahead {plan['success']:.3f} {plan['success_ci']} valid {plan['valid_rate']:.2f}")
    finally:
        tr.close()
    _save("exp_worldmodel_" + os.path.basename(os.path.normpath(run)), res)
    return 0


# ================================================================================================ self-play
def exp_selfplay(a) -> int:
    import torch
    from .cli import resolve_llm
    from .llm.server import connect
    from .selfplay.arena import Arena, IdleChooser, make_learner, online_update
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = resolve_llm(copy.deepcopy(ck["cfg"]))
    llm = connect(cfg["llm"])
    days = float(getattr(a, "days", None) or 12)
    parts = [{"handle": "ada", "goal": "revenue", "start": "coded", "need": None, "endowment": 800_000},
             {"handle": "bea", "goal": "revenue", "start": "coded", "need": None, "endowment": 800_000},
             {"handle": "cy", "goal": "service", "start": "coded", "need": None, "endowment": 800_000}]
    res: dict = {"checkpoint": a.checkpoint, "days": days, "participants": parts, "runs": {}}
    algos = {"p:ada": ("ppo", {"lr": 3e-5, "epochs": 2, "minibatch_eps": 1, "clip": 0.1, "target_kl": 0.05, "detach_critic": True}),
             "p:bea": ("ppo_sil", {"lr": 3e-5, "epochs": 2, "minibatch_eps": 1, "clip": 0.1, "target_kl": 0.05, "detach_critic": True, "sil_steps": 1}),
             "p:cy": ("ppo", {"lr": 1e-5, "epochs": 1, "minibatch_eps": 1, "clip": 0.1, "target_kl": 0.05, "detach_critic": True})}
    for label, idle in (("all three learners", ()), ("counterfactual: ada idle", ("p:ada",)), ("counterfactual: bea idle", ("p:bea",)),
                        ("counterfactual: cy idle", ("p:cy",))):
        arena = Arena("bazaar", 31, parts, round_s=1800, core=cfg["core"])
        players, al = {}, {}
        for i, actor in enumerate(sorted(arena.cores)):
            ag, algo, ch = make_learner(ck, llm, algos[actor][0], algos[actor][1], dict(cfg["agent"]), seed=10 + i)
            players[actor] = (ag, IdleChooser() if actor in idle else ch)
            al[actor] = algo
        cb = online_update(al, None)
        out = arena.run(players, days, seed=3, on_segment=lambda actor, ag, seg: cb(actor, ag, seg) if actor not in idle else None)
        out["learning"] = {k: v[-3:] for k, v in cb.stats.items()}
        res["runs"][label] = {k: out[k] for k in ("final", "agents", "wall_s", "events", "invariants", "learning")}
        res["runs"][label]["history"] = [{"day": h["day"], "market": h["market"],
                                          "outcomes": {k: {m: v[m] for m in ("live_services", "listed_services", "requests_served", "independent_revenue")} for k, v in h["outcomes"].items()}}
                                         for h in out["history"]]
        print(label, {k: {m: round(v[m], 1) if isinstance(v[m], float) else v[m] for m in ("live_services", "listed_services", "requests_served", "independent_revenue")}
                      for k, v in out["final"]["outcomes"].items()}, "market", out["final"]["market"])
    full = res["runs"]["all three learners"]["final"]["outcomes"]
    res["effects_of_presence"] = {}
    for who in ("p:ada", "p:bea", "p:cy"):
        cf_ = res["runs"][f"counterfactual: {who.split(':')[1]} idle"]["final"]["outcomes"]
        res["effects_of_presence"][who] = {other: {m: full[other][m] - cf_[other][m] for m in ("requests_served", "independent_revenue")}
                                           for other in full if other != who}
    print("effect of each learner's presence on the others (with - without):", json.dumps(res["effects_of_presence"], indent=1))
    _save("exp_selfplay", res)
    return 0


# ================================================================================================ E, F (ledger summaries)
def exp_meta_summary(a, name: str) -> int:
    from .meta.ledger import Ledger
    rd = a.run_dir
    led = Ledger(os.path.join(rd, "ledger.jsonl"))
    exps = led.experiments()
    res = {"run": rd, "experiments": len(exps), "table": led.table(),
           "proposals_rejected_before_running": led.entries("proposal_rejected"),
           "selections": led.entries("selection"), "promotions": led.entries("promotion"),
           "meta_updates": led.entries("meta_update")[-3:], "learned_rule_updates": led.entries("learned_rule_update"),
           "diagnoses": [d["text"] for d in led.entries("diagnosis")][-3:], "details": exps}
    for r in res["table"]:
        print(r)
    _save(name, res)
    return 0


def run_experiment(name: str, a) -> int:
    name = name.lower()
    fn = {"baselines": exp_baselines, "a": exp_a, "b": exp_b, "b_multi": exp_b_multi, "c": exp_c, "d": exp_d,
          "worldmodel": exp_worldmodel, "selfplay": exp_selfplay}.get(name)
    if fn is not None:
        return fn(a)
    if name in ("e", "f"):
        return exp_meta_summary(a, f"exp_{name}_meta")
    raise SystemExit(f"unknown experiment {name!r}")
