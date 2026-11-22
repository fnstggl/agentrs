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


class Partial:
    """The finished parts of a long experiment, written to ``runs/partial/<name>.json`` after every part. When the
    experiment is started again after an interruption, finished parts are read back instead of being run again.
    `key` identifies the inputs (checkpoint, task count, ...): with different inputs the file is ignored. The file
    is removed when the experiment completes."""

    def __init__(self, name: str, key=None):
        from .cli import RUNS
        self.path = os.path.join(RUNS, "partial", name + ".json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.d: dict = {}
        if os.path.exists(self.path):
            try:
                self.d = json.load(open(self.path))
            except ValueError:
                self.d = {}
        key = json.loads(json.dumps(key, default=str))
        if self.d.get("__key__") != key:
            self.d = {"__key__": key}
        self.reused: list[str] = []

    def has(self, k: str) -> bool:
        return k in self.d

    def put(self, k: str, v):
        self.d[k] = json.loads(json.dumps(v, default=str))
        tmp = self.path + ".tmp"
        json.dump(self.d, open(tmp, "w"))
        os.replace(tmp, self.path)
        return self.d[k]

    def get(self, k: str, fn):
        if k in self.d:
            self.reused.append(k)
            print(f"(already done, reused: {k})", flush=True)
            return self.d[k]
        return self.put(k, fn())

    def finish(self) -> list[str]:
        if os.path.exists(self.path):
            os.replace(self.path, self.path + ".done")
        return self.reused


def _ckpt_key(path: str | None, *more) -> list:
    return [os.path.abspath(path), round(os.path.getmtime(path))] + list(more) if path else list(more)


def _trainer(ckpt_path: str | None, cfg: dict | None = None, run_dir: str | None = None, workers: int | None = None,
             device: str | None = None):
    """`device="cpu"` for experiments that only evaluate: rollout workers act on the CPU anyway, and the GPU is
    shared with the language-model server and the training runs."""
    import torch
    from .cli import RUNS, resolve_llm
    from .train.trainer import Trainer
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path else None
    cfg = copy.deepcopy(cfg or ck["cfg"])
    cfg = resolve_llm(cfg)
    if workers is not None:
        cfg["workers"] = workers
    if device or os.environ.get("SUPERLEARNER_DEVICE"):
        cfg["device"] = device or os.environ["SUPERLEARNER_DEVICE"]
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


def _eval_per_task(tr, tasks: list, mode: str, tag: str) -> tuple[dict, list]:
    """Like Trainer.evaluate (same jobs and seeds), also returning success per task in task order (None = no episode)."""
    from .train.trainer import summarize
    tr.sync_workers()
    jobs = [{"spec": s.to_dict(), "mode": mode, "seed": 10_000 + i, "episode_id": f"{tag}{i}", "learn": False,
             "keep_decisions": False, "human": False, "agent": {}, "tag": tag} for i, s in enumerate(tasks)]
    eps = tr.absorb(tr.pool.run(jobs))
    by_id = {e["summary"].get("episode_id", ""): float(e["summary"]["success"]) for e in eps if e is not None}
    by_task = {e["summary"]["task"]: float(e["summary"]["success"]) for e in eps if e is not None}
    per = [by_id.get(f"{tag}{i}", by_task.get(s.key)) for i, s in enumerate(tasks)]
    return summarize(eps), per


def paired_tasks(after: dict, before: dict, n_boot: int = 4000, seed: int = 0) -> dict:
    """`after`, `before`: seed -> per-task success (same task order). Paired difference over (seed, task) pairs with a
    two-level bootstrap (seeds, then tasks within a seed), plus the counts of tasks that changed in each direction."""
    rng = np.random.default_rng(seed)
    d = {}
    for sd in after:
        pairs = [(a, b) for a, b in zip(after[sd], before[sd]) if a is not None and b is not None]
        if pairs:
            d[sd] = np.array([a - b for a, b in pairs], float)
    if not d:
        return {"n_pairs": 0}
    keys = sorted(d)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ks = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
        boots[i] = float(np.mean([d[k][rng.integers(0, len(d[k]), len(d[k]))].mean() for k in ks]))
    allv = np.concatenate([d[k] for k in keys])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_pairs": int(len(allv)), "n_seeds": len(keys), "diff": float(np.mean([d[k].mean() for k in keys])),
            "ci": [float(lo), float(hi)], "p_improve": float((boots > 0).mean()),
            "tasks_gained": int((allv > 0).sum()), "tasks_lost": int((allv < 0).sum()), "tasks_unchanged": int((allv == 0).sum()),
            "established": bool(lo > 0 or hi < 0)}


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
    tr, ck = _trainer(a.checkpoint, workers=a.workers, device="cpu")
    n = a.episodes or 64
    try:
        tasks = heldout_tasks(tr, n)
        res: dict = {"checkpoint": a.checkpoint, "tasks": {"n": len(tasks), "split": "select", "arms": sorted({"/".join(map(str, t.arm)) for t in tasks}),
                                                           "world_families": sorted({t.world_family for t in tasks})},
                     "core": tr.cfg["core"], "conditions": {}}
        c = res["conditions"]
        P = Partial("baselines_" + os.path.basename(run), _ckpt_key(a.checkpoint, n))

        def ev(label, *args, **kw):
            return P.get(label, lambda: _row(tr.evaluate(*args, save=False, **kw)))
        c["4 CAM after RL"] = ev("4 CAM after RL", tasks, "greedy", "b4_after_rl")
        theta_rl = flat_params(tr.model).clone()
        mem_state, lib_state = tr.memory.state_dict(), tr.library.state_dict()
        # 5: learned weights restored, episodic memory and tool library reset
        tr.memory.items, tr.library.tools = [], {}
        tr.pool.sync([], tr.library.state_dict(), reset=True)
        tr.mem_synced = 0
        c["5 weights restored, memory reset"] = ev("5 weights restored, memory reset", tasks, "greedy", "b5_weights_only")
        # 3: warm-started CAM before RL
        bc = os.path.join(run, "ckpt", "bc.pt")
        if os.path.exists(bc):
            tr.load(bc)
            c["3 warm-started CAM before RL"] = ev("3 warm-started CAM before RL", tasks, "greedy", "b3_before_rl")
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
        c["6 memory preserved, weights reset"] = ev("6 memory preserved, weights reset", tasks, "greedy", "b6_memory_only")
        res["memory_items"], res["library_tools"] = len(mem_state["items"]), len(lib_state["tools"])
        # 7: competent scripted control (through the same action interface)
        c["7 scripted control"] = ev("7 scripted control", tasks, "teacher:auto", "b7_scripted")
        c["0 uniform random (same action space)"] = ev("0 uniform random (same action space)", tasks, "random", "b0_random")
        llm_ok = tr.cfg["llm"].get("kind") == "remote"
        if llm_ok and not getattr(a, "no_llm", False):
            # 2: frozen LLM with the grounding layer, no neural controller
            c["2 frozen LLM + grounding, no learning"] = ev("2 frozen LLM + grounding, no learning", tasks, "llm", "b2_llm_grounded", agent={"use_llm": True, "llm_budget": 6})
            # 1: frozen LLM alone through Crucible's original interface
            raw_core = dict(tr.cfg["core"], max_decisions=min(24, int(tr.cfg["core"].get("max_decisions", 40))))
            c["1 frozen LLM alone"] = P.get("1 frozen LLM alone", lambda: _row(run_raw_llm(tasks[: min(len(tasks), 16)], tr.cfg["llm"], raw_core, threads=3)))
            c["1 frozen LLM alone"]["note"] = "first 16 tasks, 24 decisions each (one tool call per decision)"
        res["llm"] = tr.llm_totals()
        res["parts_reused_after_interruption"] = P.finish()
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
    tr, _ = _trainer(warm0, workers=a.workers, device="cpu")
    n = a.episodes or 96
    res: dict = {"arms": arms, "seeds": seeds, "per_seed": {}, "tasks": {}}
    try:
        tasks = heldout_tasks(tr, n)
        res["tasks"] = {"n": len(tasks), "split": "select: unseen seeds", "world_families": sorted({t.world_family for t in tasks}),
                        "arms": sorted({"/".join(map(str, t.arm)) for t in tasks})}
        P = Partial("exp_b_multi", [n, seeds] + [_ckpt_key(os.path.join(RUNS, f"{name}_s{sd}", "ckpt", "latest.pt"))
                                                  for name in arms.values() for sd in seeds
                                                  if os.path.exists(os.path.join(RUNS, f"{name}_s{sd}", "ckpt", "latest.pt"))])
        for sd in seeds:
            warm = os.path.join(RUNS, f"warm_s{sd}", "ckpt", "bc.pt")
            tr.load(warm)
            before, pt_b = P.get(f"s{sd} before greedy", lambda: _eval_per_task(tr, tasks, "greedy", f"B_before_s{sd}"))
            before_s, pt_bs = P.get(f"s{sd} before sampled", lambda: _eval_per_task(tr, tasks, "sample", f"B_before_sampled_s{sd}"))
            sd_b = torch.load(warm, map_location="cpu", weights_only=False)["model"]
            row = {"before_rl": _row(before), "before_rl_sampled": _row(before_s), "arms": {},
                   "per_task": {"before": pt_b, "before_sampled": pt_bs}}
            for label, name in arms.items():
                run = os.path.join(RUNS, f"{name}_s{sd}")
                ck = os.path.join(run, "ckpt", "latest.pt")
                if not os.path.exists(ck):
                    continue
                tr.load(ck)
                after, pt_a = P.get(f"s{sd} {name} greedy", lambda: _eval_per_task(tr, tasks, "greedy", f"B_after_{name}_s{sd}"))
                after_s, pt_as = P.get(f"s{sd} {name} sampled", lambda: _eval_per_task(tr, tasks, "sample", f"B_after_sampled_{name}_s{sd}"))
                row["per_task"][label] = pt_a
                row["per_task"][label + " (sampled)"] = pt_as
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
                evs = [e for e in read_jsonl(os.path.join(run, "eval.jsonl")) if e.get("tag") == "eval_sample"]
                rl = [m for m in read_jsonl(os.path.join(run, "metrics.jsonl")) if m.get("phase") == "rl"]
                row["arms"][label] = {"after_rl": _row(after), "delta_success": after["success"] - before["success"],
                                      "after_rl_sampled": _row(after_s), "delta_success_sampled": after_s["success"] - before_s["success"],
                                      "in_training_curve": [(e["env_decisions"], round(e["success"], 4)) for e in ev],
                                      "in_training_curve_sampled": [(e["env_decisions"], round(e["success"], 4)) for e in evs],
                                      "train_success_by_iter": [round(m.get("train_success", 0.0), 3) for m in rl],
                                      "lr_scale_by_iter": [m.get("lr_scale") for m in rl],
                                      "rollbacks": int(sum(m.get("rollbacks", 0) or 0 for m in rl)),
                                      "aulc_in_training": aulc([e["env_decisions"] for e in ev], [e["success"] for e in ev]),
                                      "aulc_in_training_sampled": aulc([e["env_decisions"] for e in evs], [e["success"] for e in evs]) if evs else None,
                                      "env_decisions": sd_a["env_decisions"], "episodes": sd_a["episodes"], "rl_iterations": sd_a["iter"],
                                      "weights_l2_distance_from_warm_start": d2 ** 0.5, "relative_change": (d2 / max(b2, 1e-12)) ** 0.5,
                                      "relative_change_by_component": {g: round((v[0] / max(v[1], 1e-12)) ** 0.5, 4) for g, v in groups.items()},
                                      "updates_that_changed_weights": int(sum(bool(m.get("weights_changed")) for m in rl)),
                                      "rollout_s": round(sum(m.get("rollout_s", 0) for m in rl)), "update_s": round(sum(m.get("update_s", 0) for m in rl)),
                                      "replay_steps": int(sum(m.get("replay_steps", 0) for m in rl))}
            res["per_seed"][sd] = row
            print(f"seed {sd}: before {before['success']:.3f} (sampled {before_s['success']:.3f}) | " + " | ".join(
                f"{l}: {r['after_rl']['success']:.3f} ({r['delta_success']:+.3f}), sampled {r['after_rl_sampled']['success']:.3f} ({r['delta_success_sampled']:+.3f})"
                for l, r in row["arms"].items()), flush=True)
        res["parts_reused_after_interruption"] = P.finish()
    finally:
        tr.close()
    ok = [sd for sd in seeds if len(res["per_seed"].get(sd, {}).get("arms", {})) == len(arms)]
    agg: dict = {"seeds_complete": ok}
    if ok:
        bef = {sd: res["per_seed"][sd]["before_rl"]["success"] for sd in ok}
        for label in arms:
            aft = {sd: res["per_seed"][sd]["arms"][label]["after_rl"]["success"] for sd in ok}
            agg[f"{label}: after - before (held-out success)"] = compare(aft, bef)
            agg[f"{label}: after - before (held-out success, sampled actions)"] = compare(
                {sd: res["per_seed"][sd]["arms"][label]["after_rl_sampled"]["success"] for sd in ok},
                {sd: res["per_seed"][sd]["before_rl_sampled"]["success"] for sd in ok})
            ps = res["per_seed"]
            agg[f"{label}: paired over tasks (greedy)"] = paired_tasks(
                {sd: ps[sd]["per_task"][label] for sd in ok}, {sd: ps[sd]["per_task"]["before"] for sd in ok})
            agg[f"{label}: paired over tasks (sampled)"] = paired_tasks(
                {sd: ps[sd]["per_task"][label + " (sampled)"] for sd in ok}, {sd: ps[sd]["per_task"]["before_sampled"] for sd in ok})
            agg[f"{label}: pooled episodes"] = {
                "n": len(ok) * res["tasks"]["n"],
                "before": float(np.mean([bef[sd] for sd in ok])), "after": float(np.mean([aft[sd] for sd in ok])),
                "before_sampled": float(np.mean([res["per_seed"][sd]["before_rl_sampled"]["success"] for sd in ok])),
                "after_sampled": float(np.mean([res["per_seed"][sd]["arms"][label]["after_rl_sampled"]["success"] for sd in ok]))}
        l1, l2 = list(arms)
        agg[f"{l2} - {l1}: held-out success after RL"] = compare(
            {sd: res["per_seed"][sd]["arms"][l2]["after_rl"]["success"] for sd in ok},
            {sd: res["per_seed"][sd]["arms"][l1]["after_rl"]["success"] for sd in ok})
        agg[f"{l2} - {l1}: held-out success after RL (sampled actions)"] = compare(
            {sd: res["per_seed"][sd]["arms"][l2]["after_rl_sampled"]["success"] for sd in ok},
            {sd: res["per_seed"][sd]["arms"][l1]["after_rl_sampled"]["success"] for sd in ok})
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
        P = Partial("exp_c_" + os.path.basename(run), _ckpt_key(a.checkpoint, n, int(getattr(a, "adapt_iters", None) or 4)))
        for name, tasks in sets.items():
            r = P.get("zero-shot: " + name, lambda tasks=tasks: _row(tr.evaluate(tasks, "greedy", "expC", save=False)))
            res["zero_shot"][name] = r
            print(f"zero-shot {name:70s} success {r['success']:.3f} {r['success_ci']} n={r['episodes']}", flush=True)
        # adaptation: a few iterations of RL on the unfamiliar families, starting from the trained and the warm-start weights
        new_fams = [f for f in SELECT_FAMILIES if f not in trained_fams] or list(SELECT_FAMILIES)
        adapt_eval = sets["unfamiliar world families (validation)"]
        res["adaptation"] = {}
        for label, ck in (("from RL-trained weights", a.checkpoint), ("from warm start (before RL)", os.path.join(run, "ckpt", "bc.pt"))):
            if not os.path.exists(ck):
                continue

            def adapt(ck=ck):
                tr.load(ck)
                tr.curriculum.world_families = [f for f in new_fams if f in ("entrenched_incumbents", "fragmented_networks")] or new_fams
                tr.curriculum.split = "select"            # the unfamiliar families exist only in the held-out splits;
                tr.curriculum.counter = 500_000           # adaptation uses a seed range far from the evaluation lists
                curve = [{"decisions": 0, "success": tr.evaluate(adapt_eval, "greedy", "expC_adapt", save=False)["success"]}]
                d0 = tr.env_decisions
                for _ in range(int(getattr(a, "adapt_iters", None) or 4)):
                    rec = tr.train_iter()
                    curve.append({"decisions": tr.env_decisions - d0, "success": tr.evaluate(adapt_eval, "greedy", "expC_adapt", save=False)["success"],
                                  "train_success": rec.get("train_success"), "lr_scale": rec.get("lr_scale"), "param_delta": rec.get("param_delta")})
                return curve
            curve = P.get("adaptation: " + label, adapt)
            res["adaptation"][label] = curve
            print(f"adaptation {label}: {[(c['decisions'], round(c['success'], 3)) for c in curve]}", flush=True)
        res["parts_reused_after_interruption"] = P.finish()
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
        n = a.episodes or 48
        P = Partial("exp_d", [itersA, itersB, int(cfg["episodes_per_iter"]), n])
        tA, tB = tr.eval_tasks(n, "select", armsA), tr.eval_tasks(n, "select", armsB)
        ck = os.path.join(rd, "ckpt", "after_A.pt")
        if P.has("phase_A") and os.path.exists(ck):
            tr.load(ck)                                            # phase A was finished by an earlier, interrupted run
            demosA = list(tr.demos)
            print("(phase A already done: continuing from its checkpoint)", flush=True)
        else:
            tr.collect_demos(int(cfg["demos"]["episodes"]), A)         # demonstrations for A only: B is learned by RL
            demosA = list(tr.demos)
            tr.pretrain_bc(int(cfg["demos"]["bc_epochs"]))
            a0 = tr.evaluate(tA, "greedy", "D_A_warm", save=False)
            for _ in range(itersA):
                tr.train_iter()
            a1, b1 = tr.evaluate(tA, "greedy", "D_A_after_A", save=False), tr.evaluate(tB, "greedy", "D_B_after_A", save=False)
            ck = tr.save("after_A.pt")
            P.put("phase_A", {"A_at_warm_start": _row(a0), "A_after_learning_A": _row(a1), "B_before_any_B_training": _row(b1)})
        out["phase_A"] = P.d["phase_A"]
        a1, b1 = out["phase_A"]["A_after_learning_A"], out["phase_A"]["B_before_any_B_training"]
        print(f"phase A: A {out['phase_A']['A_at_warm_start']['success']:.3f} -> {a1['success']:.3f}; B (untrained) {b1['success']:.3f}", flush=True)
        buf = list(getattr(tr.algo, "buffer", []))
        counter = tr.curriculum.counter
        for name, arms_b, keep_buffer, extra in (
                ("no protection", Bset, False, {}),
                ("replay of own earlier successes", Bset, True, {}),
                ("rehearsal (A sampled at a low rate)", Bset + A, False, {}),
                ("demonstration regulariser on A (bc_coef 0.5)", Bset, False, {"bc_coef": 0.5})):
            def phase_b(arms_b=arms_b, keep_buffer=keep_buffer, extra=extra):
                from .algo.ppo import flat_params
                tr.load(ck)
                theta = flat_params(tr.model).clone()
                tr.algo.buffer = list(buf) if keep_buffer else []
                tr.algo.c.update({"bc_coef": 0.0}, **extra)
                cur = Curriculum(arms_b, cfg["curriculum"].get("world_families"), "train", cfg["seed"] + 1, gate=False)
                cur.counter = counter + 200_000
                if len(arms_b) > len(Bset):
                    w = np.array([1.0] * len(Bset) + [0.15] * len(A))
                    cur.weights = lambda w=w: w / w.sum()
                tr.curriculum = cur
                tr.demos = _copy.copy(demosA) if extra.get("bc_coef") else []
                recs = [tr.train_iter() for _ in range(itersB)]
                a2, b2 = tr.evaluate(tA, "greedy", "D_A_after_B", save=False), tr.evaluate(tB, "greedy", "D_B_after_B", save=False)
                return {"A_after_training_on_B": _row(a2), "B_after_training_on_B": _row(b2),
                        "forgetting_on_A": a1["success"] - a2["success"], "gain_on_B": b2["success"] - b1["success"],
                        "weights_moved_l2": float((flat_params(tr.model) - theta).norm()), "weights_norm": float(theta.norm()),
                        "train_success_on_B_by_iter": [round(r.get("train_success", 0.0), 3) for r in recs],
                        "lr_scale_by_iter": [r.get("lr_scale") for r in recs]}
            r = P.get("strategy: " + name, phase_b)
            out["strategies"][name] = r
            print(f"{name:48s} A: {a1['success']:.3f} -> {r['A_after_training_on_B']['success']:.3f}   "
                  f"B: {b1['success']:.3f} -> {r['B_after_training_on_B']['success']:.3f}", flush=True)
        out["parts_reused_after_interruption"] = P.finish()
    finally:
        tr.close()
    _save("exp_d_retention", out)
    return 0


def exp_d_clone(a) -> int:
    """Retention, supervised variant. Starting from experiment D's checkpoint after phase A, the controller is trained
    on the unrelated set B by cloning verified demonstrations of B (which moves the weights far more than a few RL
    iterations do), and A is tested again. Two strategies branch from the same checkpoint: B's demonstrations only,
    and B's demonstrations with A's mixed in (rehearsal)."""
    from .algo.ppo import flat_params
    from .cli import RUNS
    src = a.run_dir or os.path.join(RUNS, "exp_d")
    ck = os.path.join(src, "ckpt", "after_A.pt")
    Bset = [{"family": "repair"}, {"family": "deploy", "start": "coded"}, {"family": "service", "start": "listed"}]
    armsA = ["found/0/scratch", "fund/0/scratch", "list/0/deployed"]
    armsB = ["repair/0/scratch", "deploy/0/coded", "service/0/listed"]
    tr, _ = _trainer(ck, run_dir=os.path.join(RUNS, "exp_d_clone"), workers=a.workers)
    out: dict = {"checkpoint": ck, "skill_set_A": armsA, "unrelated_set_B": armsB, "strategies": {}}
    try:
        dA = pickle.load(open(os.path.join(src, "demos.pkl"), "rb"))
        for k, v in dA.get("emb", []):
            tr.store.put(k, v)
        demosA = list(dA["episodes"])
        n = a.episodes or 48
        P = Partial("exp_d_clone", _ckpt_key(ck, n))
        tA, tB = tr.eval_tasks(n, "select", armsA), tr.eval_tasks(n, "select", armsB)
        a1 = P.get("A before", lambda: _row(tr.evaluate(tA, "greedy", "Dc_A_before", save=False)))
        b1 = P.get("B before", lambda: _row(tr.evaluate(tB, "greedy", "Dc_B_before", save=False)))
        out["before"] = {"A": a1, "B": b1}
        theta_a = flat_params(tr.model).clone()
        mine = os.path.join(tr.dir, "demos.pkl")
        if P.has("demonstrations") and os.path.exists(mine):   # B's demonstrations were collected by an interrupted run
            dB = pickle.load(open(mine, "rb"))
            for k, v in dB.get("emb", []):
                tr.store.put(k, v)
            demosB = list(dB["episodes"])
        else:
            prov = tr.collect_demos(60, Bset)                  # verified demonstrations of B (writes this run's demos.pkl)
            demosB = list(tr.demos)
            P.put("demonstrations", {"A": len(demosA), "B": len(demosB), "B_by_arm": prov["by_arm_kept"]})
        out["demonstrations"] = P.d["demonstrations"]
        epochs = int(tr.cfg["demos"]["bc_epochs"])
        print(f"before: A {a1['success']:.3f}  B {b1['success']:.3f}; demonstrations A {len(demosA)} B {len(demosB)}", flush=True)
        for name, dem in (("clone B only", demosB), ("clone B with A's demonstrations mixed in", demosB + demosA)):
            def clone(dem=dem):
                tr.load(ck)
                tr.demos = list(dem)
                hist = tr.pretrain_bc(epochs)
                a2, b2 = tr.evaluate(tA, "greedy", "Dc_A_after", save=False), tr.evaluate(tB, "greedy", "Dc_B_after", save=False)
                return {"A_after": _row(a2), "B_after": _row(b2), "forgetting_on_A": a1["success"] - a2["success"],
                        "gain_on_B": b2["success"] - b1["success"], "epochs": epochs,
                        "weights_moved_l2": float((flat_params(tr.model) - theta_a).norm()), "weights_norm": float(theta_a.norm()),
                        "nll_first_last": [hist[0]["bc_nll"], hist[-1]["bc_nll"]] if hist else None}
            r = P.get("strategy: " + name, clone)
            out["strategies"][name] = r
            print(f"{name:44s} A: {a1['success']:.3f} -> {r['A_after']['success']:.3f}   B: {b1['success']:.3f} -> {r['B_after']['success']:.3f}   "
                  f"weights moved {r['weights_moved_l2']:.2f}", flush=True)
        out["parts_reused_after_interruption"] = P.finish()
    finally:
        tr.close()
    _save("exp_d_clone", out)
    return 0


# ================================================================================================ A
def exp_a(a) -> int:
    """Real actions: build and operate a functioning service from scratch in new worlds; inspect the artifacts."""
    tr, _ = _trainer(a.checkpoint, workers=a.workers, device="cpu")
    n = a.episodes or 24
    try:
        fams = tr.cfg["curriculum"].get("world_families") or ["tiny", "bazaar"]
        tasks = _tasks(["deploy/0/scratch", "service/0/scratch"], fams, n, "select", 9000)
        res: dict = {"checkpoint": a.checkpoint, "tasks": [t.key for t in tasks], "conditions": {}}
        P = Partial("exp_a", _ckpt_key(a.checkpoint, n))
        for label, mode, agent in (("CAM controller (greedy)", "greedy", {}), ("CAM controller (sampled)", "sample", {}),
                                   ("scripted control through the same interface", "teacher:auto", {}),
                                   ("frozen LLM + grounding, no controller", "llm", {"llm_budget": 6})):
            def cond(mode=mode, agent=agent):
                from .train.trainer import summarize
                tr.sync_workers()
                jobs = [{"spec": s.to_dict(), "mode": mode, "seed": 500 + i, "episode_id": f"A{i}", "learn": False, "human": True,
                         "keep_decisions": False, "agent": agent} for i, s in enumerate(tasks)]
                eps = tr.absorb(tr.pool.run(jobs))
                arts = []
                for e in eps:
                    steps = [d["human"] for d in e["decisions"] if "human" in d]
                    arts.append({"task": e["summary"]["task"], "success": e["summary"]["success"], "progress": e["summary"]["best_progress"],
                                 "decisions": e["summary"]["decisions"], "failures": e["summary"]["agent"]["failures"],
                                 "llm": e["summary"]["llm"], "actions": [s["action"] + ("" if s["ok"] else f" !{s['fclass']}") for s in steps][:40]})
                tr.save_traj("expA_" + mode.replace(":", "_"), eps[:6])
                return {"summary": _row(summarize(eps)), "episodes": arts}
            c_ = P.get(label, cond)
            res["conditions"][label] = c_
            r = c_["summary"]
            print(f"{label:48s} success {r['success']:.3f} {r['success_ci']} valid {r['valid_rate']:.2f} failures {r['failures']}", flush=True)
        res["parts_reused_after_interruption"] = P.finish()
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
    fam, wseed = (str(getattr(a, "world", None) or "bazaar:31").split(":") + ["31"])[:2]
    res: dict = {"checkpoint": a.checkpoint, "days": days, "world": {"family": fam, "seed": int(wseed)}, "participants": parts, "runs": {}}
    algos = {"p:ada": ("ppo", {"lr": 3e-5, "epochs": 2, "minibatch_eps": 1, "clip": 0.1, "target_kl": 0.05, "detach_critic": True}),
             "p:bea": ("ppo_sil", {"lr": 3e-5, "epochs": 2, "minibatch_eps": 1, "clip": 0.1, "target_kl": 0.05, "detach_critic": True, "sil_steps": 1}),
             "p:cy": ("ppo", {"lr": 1e-5, "epochs": 1, "minibatch_eps": 1, "clip": 0.1, "target_kl": 0.05, "detach_critic": True})}
    tag = "exp_selfplay" if (fam, wseed) == ("bazaar", "31") else f"exp_selfplay_{fam}{wseed}"
    P = Partial(tag, _ckpt_key(a.checkpoint, days, fam, wseed) if tag != "exp_selfplay" else _ckpt_key(a.checkpoint, days))
    from .train.demos import TeacherChooser
    for label, idle in (("all three learners", ()), ("counterfactual: ada idle", ("p:ada",)), ("counterfactual: bea idle", ("p:bea",)),
                        ("counterfactual: cy idle", ("p:cy",)),
                        ("reference: scripted control for all three (same world, same interface, no learning)", "scripted")):
        def one(idle=idle):
            arena = Arena(fam, int(wseed), parts, round_s=1800, core=cfg["core"])
            players, al = {}, {}
            scripted = idle == "scripted"
            for i, actor in enumerate(sorted(arena.cores)):
                ag, algo, ch = make_learner(ck, llm, algos[actor][0], algos[actor][1], dict(cfg["agent"]), seed=10 + i)
                players[actor] = (ag, TeacherChooser("auto", 10 + i) if scripted else IdleChooser() if actor in idle else ch)
                al[actor] = algo
            cb = online_update(al, None)
            out = arena.run(players, days, seed=3, on_segment=lambda actor, ag, seg: cb(actor, ag, seg) if (not scripted and actor not in idle) else None)
            out["learning"] = {k: v[-3:] for k, v in cb.stats.items()}
            r = {k: out[k] for k in ("final", "agents", "wall_s", "events", "invariants", "learning")}
            r["history"] = [{"day": h["day"], "market": h["market"],
                             "outcomes": {k: {m: v[m] for m in ("live_services", "listed_services", "requests_served", "independent_revenue")} for k, v in h["outcomes"].items()}}
                            for h in out["history"]]
            return r
        res["runs"][label] = P.get(label, one)
        fin = res["runs"][label]["final"]
        print(label, {k: {m: round(v[m], 1) if isinstance(v[m], float) else v[m] for m in ("live_services", "listed_services", "requests_served", "independent_revenue")}
                      for k, v in fin["outcomes"].items()}, "market", fin["market"], flush=True)
    res["parts_reused_after_interruption"] = P.finish()
    full = res["runs"]["all three learners"]["final"]["outcomes"]
    res["effects_of_presence"] = {}
    for who in ("p:ada", "p:bea", "p:cy"):
        cf_ = res["runs"][f"counterfactual: {who.split(':')[1]} idle"]["final"]["outcomes"]
        res["effects_of_presence"][who] = {other: {m: full[other][m] - cf_[other][m] for m in ("requests_served", "independent_revenue")}
                                           for other in full if other != who}
    print("effect of each learner's presence on the others (with - without):", json.dumps(res["effects_of_presence"], indent=1))
    _save(tag, res)
    return 0


# ================================================================================================ E, F (ledger summaries)
def _spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def offline_selection(rd: str, rounds: list, k: int = 2) -> dict:
    """Experiment F, second view. In the search itself the surrogate learns only from the candidates it picked. Because
    every pool was screened in full, a surrogate can also be replayed with full feedback: before each round it is
    fitted on ALL screened candidates of the earlier rounds and asked to rank the new pool (structural features only).
    Its picks are compared with the pool mean (what uniform random selection gets in expectation) and with the best."""
    from .meta.surrogate import Surrogate
    hist: list = []
    rows = []
    for fr in rounds:
        pool = []
        for e in fr["pool"]:
            pf = os.path.join(rd, "candidates", e["candidate"], "proposal_full.json")
            if os.path.exists(pf):
                pool.append((json.load(open(pf)), e))
        if len(hist) >= 4 and len(pool) >= 3:
            sur = Surrogate(None, seed=0)
            for p_, e in hist:
                sur.add(p_, e["effect"], e["valid"])
            sur.fit()
            props = [p_ for p_, _ in pool]
            mu, sd, pv = sur.predict(props)
            vals = np.array([e["value"] for _, e in pool], float)
            picks = sur.select(props, k, rng=np.random.default_rng(0))
            ok = [i for i, (_, e) in enumerate(pool) if e["valid"]]
            rows.append({"round": fr["round"], "trained_on": len(hist), "pool": len(pool),
                         "picked": float(vals[picks].mean()), "random": float(vals.mean()), "best": float(vals.max()),
                         "spearman_predicted_vs_measured": _spearman(mu[ok], vals[ok]) if len(ok) >= 3 else None,
                         "spearman_score_vs_value_all": _spearman(pv * mu, vals)})
        hist += pool
    if not rows:
        return {"rounds": [], "note": "not enough screened rounds"}
    d = np.array([r["picked"] - r["random"] for r in rows])
    rng = np.random.default_rng(0)
    b = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(1) if len(d) > 1 else np.full(4000, d[0])
    sp = [r["spearman_predicted_vs_measured"] for r in rows if r["spearman_predicted_vs_measured"] is not None]
    return {"rounds": rows, "k": k, "advantage_over_random": {"mean": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n_rounds": len(d)},
            "mean_spearman": float(np.mean(sp)) if sp else None,
            "established": bool(len(d) >= 5 and np.percentile(b, 2.5) > 0)}


def exp_meta_summary(a, name: str) -> int:
    from .meta.ledger import Ledger
    rd = a.run_dir
    led = Ledger(os.path.join(rd, "ledger.jsonl"))
    exps = led.experiments()
    cfg = json.load(open(os.path.join(rd, "config.json")))
    rounds = led.entries("full_feedback")
    screened = []
    for fr in rounds:
        for e in fr["pool"]:
            diff = ""
            pf = os.path.join(rd, "candidates", e["candidate"], "proposal.diff")
            if os.path.exists(pf):
                diff = open(pf).read()
            screened.append({"round": fr["round"], "candidate": e["candidate"], "source": e["source"], "slots": e["slots"],
                             "valid": e["valid"], "effect_aulc": e["effect"], "predicted": e.get("predicted"), "p_valid": e.get("p_valid"),
                             "picked_by_surrogate": e["proposal"] in fr["surrogate_picks"], "reject_reason": e.get("reject_reason", ""),
                             "hypothesis": e["hypothesis"], "measures": e.get("measures"), "wall_s": e.get("wall_s"),
                             "diff_lines": len([l for l in diff.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")])})
    # which screened candidates do nothing? (confined probe: one update on fixed data, compared with the champion's)
    inert: dict = {}
    try:
        from .meta.researcher import probe_search
        pr = probe_search(rd)
        inert = {cid: v.get("inert") for cid, v in pr["candidates"].items()}
    except Exception as e:  # noqa: BLE001 - the summary is still useful without the probe
        print(f"inert-change probe not run: {type(e).__name__}: {e}")
    for x in screened:
        x["inert"] = inert.get(x["candidate"])
    sel_path = os.path.join(rd, "selection_report.json")
    base = [{"stage": b["stage"], "seed": b["seed"], **{k: b["measures"].get(k) for k in ("aulc_eval", "first_eval", "final_eval", "aulc_train", "train_decisions", "wall_s")}}
            for b in led.entries("baseline_run")]
    valid = [x for x in screened if x["valid"]]
    by_source: dict = {}
    for x in screened:
        g = by_source.setdefault(x["source"], {"n": 0, "valid": 0, "effects": []})
        g["n"] += 1
        g["valid"] += int(x["valid"])
        if x["valid"]:
            g["effects"].append(x["effect_aulc"])
    for src_, g in by_source.items():
        g["mean_effect"] = float(np.mean(g["effects"])) if g["effects"] else None
        g["best_effect"] = float(np.max(g["effects"])) if g["effects"] else None
        g["inert"] = int(sum(1 for x in screened if x["source"] == src_ and x.get("inert")))
    # an inert candidate is the champion's algorithm run again: its "effect" is run-to-run noise at one seed
    noise = [x["effect_aulc"] for x in valid if x.get("inert")]
    active = [x["effect_aulc"] for x in valid if x.get("inert") is False]
    noise_summary = {"n": len(noise), "effects": [round(v, 4) for v in noise],
                     "mean_abs": float(np.mean(np.abs(noise))) if noise else None, "max_abs": float(np.max(np.abs(noise))) if noise else None,
                     "sd_around_zero": float(np.sqrt(np.mean(np.square(noise)))) if noise else None,
                     "active_candidates": len(active),
                     "active_effects_beyond_max_noise": int(sum(1 for v in active if noise and abs(v) > np.max(np.abs(noise))))}
    res = {"run": rd, "primary": cfg.get("primary"), "stages": cfg.get("stages"), "inner": cfg.get("inner"),
           "champion_baseline_runs": base, "screened": screened, "candidates_screened": len(screened), "valid_screened": len(valid),
           "by_source": by_source, "noise_from_inert_candidates": noise_summary,
           "rounds": [{k: fr.get(k) for k in ("round", "surrogate_fitted", "k", "surrogate_picks", "value_surrogate", "value_random_expected", "value_best", "valid_fraction", "diagnosis_tags")} for fr in rounds],
           "selection_online": json.load(open(sel_path)) if os.path.exists(sel_path) else None,
           "selection_offline_full_feedback": offline_selection(rd, rounds, int(cfg.get("per_round", 1))),
           "experiments": len(exps), "table": led.table(),
           "proposals_rejected_before_running": led.entries("proposal_rejected"),
           "selections": led.entries("selection"), "promotions": led.entries("promotion"),
           "meta_updates": led.entries("meta_update")[-3:], "learned_rule_updates": led.entries("learned_rule_update"),
           "diagnoses": [d["text"] for d in led.entries("diagnosis")], "details": exps}
    for x in screened:
        print(f"r{x['round']} {x['candidate']} {x['source']:12s} {','.join(x['slots']):28s} valid={x['valid']} inert={x.get('inert')} "
              f"effect={'-' if x['effect_aulc'] is None else format(x['effect_aulc'], '+.4f')} {'*' if x['picked_by_surrogate'] else ' '} {x['hypothesis'][:70]}")
    for r in res["table"]:
        print(r)
    print("noise from inert candidates:", noise_summary)
    print(json.dumps(res["selection_offline_full_feedback"], indent=1, default=str)[:1500])
    _save(name, res)
    return 0


# ================================================================================================ sealed final list
def exp_final(a) -> int:
    """The sealed final task list, used once, after every choice has been made on the other splits.

    The learner to test is chosen from `docs/experiments/exp_b_multi.json` (the arm with the higher mean held-out
    success on the selection split; `--run-dir` overrides with a run-name prefix). For every seed its warm start and
    its checkpoint after RL are evaluated on the sealed tasks of the task arms the learner was trained on (the other
    sealed tasks need LLM-written services, which these runs did not use; they are listed as not attempted)."""
    from .cli import RUNS
    from .world.tasks import load_final
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "experiments")
    path = a.config or os.path.join(here, "final_tasks.json")
    tasks = load_final(path)                                   # raises if the list was altered after sealing
    seeds = [int(x) for x in str(getattr(a, "seeds", None) or "0,1,2").split(",")]
    prefix, rule = a.run_dir, "given on the command line"
    if not prefix:
        bm = json.load(open(os.path.join(here, "exp_b_multi.json")))
        means = {lab: float(np.mean([r["arms"][lab]["after_rl"]["success"] for r in bm["per_seed"].values() if lab in r["arms"]])) for lab in bm["arms"]}
        best = max(means, key=means.get)
        prefix, rule = bm["arms"][best], f"highest mean held-out success on the selection split: {means}"
    tr, _ = _trainer(os.path.join(RUNS, f"warm_s{seeds[0]}", "ckpt", "bc.pt"), workers=a.workers, device="cpu")
    res: dict = {"sealed_list": os.path.basename(path), "seal": json.load(open(path))["seal"], "learner": prefix, "selection_rule": rule,
                 "seeds": seeds, "per_seed": {}}
    try:
        trained = {x.key for x in tr.curriculum.arms}
        sub = [t for t in tasks if "/".join(map(str, t.arm)) in trained]
        res["tasks"] = {"sealed": len(tasks), "attempted": len(sub), "arms_attempted": sorted({"/".join(map(str, t.arm)) for t in sub}),
                        "arms_not_attempted": sorted({"/".join(map(str, t.arm)) for t in tasks} - trained),
                        "world_families": sorted({t.world_family for t in sub})}
        per: dict = {"before": {}, "after": {}, "before_sampled": {}, "after_sampled": {}}
        P = Partial("exp_final", [prefix, seeds, res["seal"]] + [_ckpt_key(os.path.join(RUNS, f"{prefix}_s{sd}", "ckpt", "latest.pt")) for sd in seeds])
        for sd in seeds:
            tr.load(os.path.join(RUNS, f"warm_s{sd}", "ckpt", "bc.pt"))
            b, per["before"][sd] = P.get(f"s{sd} before greedy", lambda: _eval_per_task(tr, sub, "greedy", f"F_before_s{sd}"))
            bs, per["before_sampled"][sd] = P.get(f"s{sd} before sampled", lambda: _eval_per_task(tr, sub, "sample", f"F_before_sampled_s{sd}"))
            tr.load(os.path.join(RUNS, f"{prefix}_s{sd}", "ckpt", "latest.pt"))
            af, per["after"][sd] = P.get(f"s{sd} after greedy", lambda: _eval_per_task(tr, sub, "greedy", f"F_after_s{sd}"))
            afs, per["after_sampled"][sd] = P.get(f"s{sd} after sampled", lambda: _eval_per_task(tr, sub, "sample", f"F_after_sampled_s{sd}"))
            res["per_seed"][sd] = {"before_rl": _row(b), "after_rl": _row(af), "before_rl_sampled": _row(bs), "after_rl_sampled": _row(afs)}
            print(f"seed {sd}: greedy {b['success']:.3f} -> {af['success']:.3f}; sampled {bs['success']:.3f} -> {afs['success']:.3f}", flush=True)
        res["per_task"] = per
        res["aggregate"] = {"after - before (greedy)": paired_tasks(per["after"], per["before"]),
                            "after - before (sampled)": paired_tasks(per["after_sampled"], per["before_sampled"]),
                            "mean_before": float(np.mean([r["before_rl"]["success"] for r in res["per_seed"].values()])),
                            "mean_after": float(np.mean([r["after_rl"]["success"] for r in res["per_seed"].values()])),
                            "mean_before_sampled": float(np.mean([r["before_rl_sampled"]["success"] for r in res["per_seed"].values()])),
                            "mean_after_sampled": float(np.mean([r["after_rl_sampled"]["success"] for r in res["per_seed"].values()]))}
        by_fam: dict = {}
        for sd in seeds:
            for i, t in enumerate(sub):
                g = by_fam.setdefault(t.world_family, {"before": [], "after": []})
                if per["before"][sd][i] is not None and per["after"][sd][i] is not None:
                    g["before"].append(per["before"][sd][i])
                    g["after"].append(per["after"][sd][i])
        res["aggregate"]["by_world_family"] = {k: {"n": len(v["before"]), "before": round(float(np.mean(v["before"])), 3), "after": round(float(np.mean(v["after"])), 3)}
                                                for k, v in sorted(by_fam.items())}
        res["parts_reused_after_interruption"] = P.finish()
    finally:
        tr.close()
    print(json.dumps(res["aggregate"], indent=1, default=str))
    _save("exp_final", res)
    return 0


def run_experiment(name: str, a) -> int:
    name = name.lower()
    fn = {"baselines": exp_baselines, "a": exp_a, "b": exp_b, "b_multi": exp_b_multi, "c": exp_c, "d": exp_d,
          "worldmodel": exp_worldmodel, "selfplay": exp_selfplay, "final": exp_final, "d_clone": exp_d_clone}.get(name)
    if fn is not None:
        return fn(a)
    if name in ("e", "f"):
        return exp_meta_summary(a, f"exp_{name}_meta")
    raise SystemExit(f"unknown experiment {name!r}")
