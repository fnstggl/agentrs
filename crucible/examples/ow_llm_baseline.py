"""Crucible 4.0: the frozen language-model baseline, on a corrected interface, with automatic failure diagnosis.

    cd crucible
    export CRUCIBLE_CACHE=<dir>                                   # validated tasks are cached there
    PYTHONPATH=src python examples/ow_llm_baseline.py controls    # no GPU
    PYTHONPATH=src python examples/ow_llm_baseline.py llm ablation sens open open_v3 summary

Stages (any subset, in the order given; every finished run is stored, so a stage that is run again only does what
is missing - and a run whose task or configuration changed is done again):

    controls   every mechanism task: idle, random settings, random tools, the scripted experimenter (the competent
               control), the private reference played through the tools
    llm        the frozen model on the v4 interface, every task
    ablation   the same model with the 3.0 prompt (`interface="v3"`, 9000-character tool list, 8k context) on the
               first two tasks
    sens       a smaller frozen model on the v4 interface, first `--sens-tasks` tasks (3: one task per engine)
    stale      the main model with the stock v4 observation line ("last action: none" on every turn), first task
    short      on the first `--short-tasks` tasks cut to `--short-ticks` ticks (renormalised, `llm_eval.shorten`):
               the controls, the main model on the v4 interface (greedy, as everywhere else), the same with
               thinking enabled, and the same sampled at temperature 0.7 (seed 0)
    open       open economy: idle, random tools, the scripted example-inductor (competent control), the frozen
               model on the v4 interface
    open_v3    the open economy with the 3.0 prompt
    summary    tables (printed, and written to ow_llm_baseline_summary.json)

What a run is: `crucible.research.llm_eval.run_mechanism` / `run_open_economy`. The model gets the public
objective text and the ordinary tool interface; the outcome is read from the world by the operator-side
evaluator; `research.diagnostics.diagnose` classifies every action, with the controls on the same task as
references.

Output (default `docs/experiments/`): `ow_llm_baseline.json` (one record per run), `ow_llm_<run>.jsonl`
(transcripts: complete for model runs, the actions only for the scripted controls), and
`ow_llm_baseline_summary.json`.

GPU. The machine is shared: before weights are loaded the script requires `--need-gb` of free GPU memory
(`nvidia-smi`), waits and re-checks (`--gpu-tries` x `--gpu-pause` seconds), then falls back to the next smaller
model (and finally to the smallest on CPU), and records which model actually ran. One set of weights is on the
GPU at a time. Within one invocation, consecutive stages that use the same weights (the v4 and v3 prompts,
thinking on or off: prompt-side settings of one loaded model) share that instance; it is deleted and the cache
emptied before other weights are loaded and at the end. A hash over all weights is taken at load and before the
delete (frozen means: equal).

`--backend echo` replaces the model by the scripted test double (`agents.llm.EchoLLM`, one fixed reply) to try
the pipeline without a GPU; such records are marked `test_double` and must never be reported.
"""
import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.llm import LLMPolicy  # noqa: E402
from crucible.core.canonical import digest  # noqa: E402
from crucible.research import diagnostics as X  # noqa: E402
from crucible.research import llm_eval as L  # noqa: E402
from crucible.research import tasks as T  # noqa: E402

V = T.VAL_SEED0
# two tasks per engine, four families; the first three cover the three engines (`--tasks 3`)
TASKS = [("scm_additive", V), ("abm_utility", V), ("tech_pipeline", V), ("scm_delay", V), ("abm_utility", V + 1),
         ("tech_pipeline", V + 1)]
# the corrected interface: complete catalogue, complete results, structured error detail, room to remember reports,
# and no "last action: none" line after an action (see `llm_eval.EvalLLMPolicy`)
V4 = dict(interface="v4", tool_chars=40_000, result_chars=12_000, history=8, history_chars=1200, payload_chars=600,
          max_new_tokens=512, last_action_line=False)
# the same with the stock v4 observation, which says "last action: none" on every turn
V4_STALE = {**V4, "last_action_line": True}
# the 3.0 prompt with its defaults (`LLMPolicy()` of 3.0): 9000-character alphabetical tool list, first-line
# descriptions, 600 characters of the last result, six history lines of 240 characters, no error detail
V3 = dict(interface="v3", tool_chars=9000, tool_order="alpha", history=6, max_new_tokens=256)
FALLBACK = {"Qwen/Qwen3-4B": "Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B": "Qwen/Qwen3-0.6B"}
NEED_GB = {"Qwen/Qwen3-4B": 14.0, "Qwen/Qwen3-1.7B": 8.0, "Qwen/Qwen3-0.6B": 5.0}
OPEN = dict(world_family="ow_tiny", seed=1, overrides={"n_consumers": 800}, days=8, window_days=3,
            revenue_cents=10_000, requests=100, sustain_checks=2, endowment=3_000_000)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def code_digest() -> str:
    """Hash of the package source that is about to produce a run (the worktree is shared and moving: a record
    says which code state measured it)."""
    import hashlib
    h = hashlib.sha256()
    src = os.path.join(ROOT, "src", "crucible")
    for d, _, files in sorted(os.walk(src)):
        for f in sorted(files):
            if f.endswith(".py"):
                h.update(os.path.relpath(os.path.join(d, f), src).encode())
                with open(os.path.join(d, f), "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()[:16]


def stamp(rec: dict) -> dict:
    rec["code_digest"] = code_digest()
    rec["recorded"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return rec


# ====================================================================== the model (one at a time)
def gpu_free_mib():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                                      text=True, timeout=30)
        used, total = (int(x.split()[0]) for x in out.strip().splitlines()[0].split(","))
        return total - used
    except Exception:  # noqa: BLE001 - no GPU, no nvidia-smi
        return None


class Model:
    """One set of frozen weights on the GPU, serving views that differ only in prompt-side settings (context
    limit, thinking). `ensure` checks free memory first (waits, then falls back to a smaller model); `free`
    deletes the weights. `fingerprint` (a hash over all weights) is taken at load and again before the delete,
    and written into every record made with this instance."""

    def __init__(self, name: str, need_gb: float | None, tries: int, pause_s: int, echo: bool = False):
        self.requested, self.name, self.backend, self.notes = name, name, None, []
        self.need_gb, self.tries, self.pause_s, self.echo = need_gb, tries, pause_s, echo
        self.fingerprint = None
        self.runs: list[str] = []                  # run ids made with this instance

    def ensure(self):
        if self.backend is not None:
            return self
        if self.echo:
            from crucible.agents.llm import EchoLLM
            self.backend = EchoLLM(['{"tool": "rig.read", "args": {}}'])
            self.backend.device, self.backend.max_context, self.backend.enable_thinking = "none", None, False
            self.name, self.fingerprint = self.backend.name, "test-double"
            self.notes.append("TEST DOUBLE (EchoLLM): not a model, not a result")
            return self
        from crucible.agents.llm import HFBackend
        name, device = self.name, "cuda"
        while True:
            need = (self.need_gb if self.need_gb is not None and name == self.requested else NEED_GB.get(name, 14.0)) * 1024
            free = None
            for k in range(self.tries):
                free = gpu_free_mib()
                if free is None or free >= need:
                    break
                log(f"GPU has {free} MiB free, {name} needs {need:.0f}: waiting {self.pause_s}s ({k + 1}/{self.tries})")
                time.sleep(self.pause_s)
            if free is None:
                device, name = "cpu", "Qwen/Qwen3-0.6B"
                self.notes.append("no GPU visible: the smallest model on CPU")
                break
            if free >= need:
                break
            if name in FALLBACK:
                self.notes.append(f"{name} not loaded: {free} MiB free after {self.tries} checks, {need:.0f} needed")
                name = FALLBACK[name]
            else:
                device = "cpu"
                self.notes.append(f"{name} on CPU: {free} MiB free on the GPU")
                break
        t0 = time.time()
        self.name = name
        self.backend = HFBackend(name, device)
        self.fingerprint = self.backend.fingerprint()
        log(f"loaded {name} on {self.backend.device} in {time.time() - t0:.0f}s, weights {self.fingerprint}",
            *(self.notes or []))
        return self

    def view(self, max_context: int, thinking: bool):
        """The loaded weights with another context limit / thinking switch (a shallow copy: no second model)."""
        b = copy.copy(self.ensure().backend)
        if not self.echo:
            b.max_context, b.enable_thinking = int(max_context), bool(thinking)
        return b

    def info(self, backend) -> dict:
        return {"requested": self.requested, "model": self.name, "device": backend.device,
                "max_context": backend.max_context, "thinking": backend.enable_thinking,
                "weights_fingerprint": self.fingerprint, "notes": list(self.notes)}

    def free(self, store=None) -> None:
        """Delete the weights and empty the cache; write the fingerprint taken just before into this instance's
        records (equal to the one at load: the weights were never updated)."""
        if self.backend is None:
            return
        after = self.fingerprint
        if not self.echo:
            after = self.backend.fingerprint()
            import torch
            self.backend.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.backend = None
        log(f"freed {self.name}; weights unchanged: {after == self.fingerprint}; GPU free {gpu_free_mib()} MiB")
        if store is not None:
            store.reload()
            for rid in self.runs:
                if store.get(rid) is not None and "model" in store.runs[rid]:
                    store.runs[rid]["model"]["weights_fingerprint_after"] = after
                    store.touch(rid)
            store.flush()
        self.runs = []


class Models:
    """The one model that is loaded, kept across stages that ask for the same weights."""

    def __init__(self, store, need_gb, tries, pause_s, echo):
        self.store, self.args, self.current = store, (need_gb, tries, pause_s, echo), None

    def get(self, name: str) -> Model:
        if self.current is not None and self.current.requested != name:
            self.free()
        if self.current is None:
            self.current = Model(name, *self.args)
        return self.current

    def free(self) -> None:
        if self.current is not None:
            self.current.free(self.store)
            self.current = None


# ====================================================================== tasks
def load_tasks(pairs, ticks: int):
    """[(full task, task as evaluated)] - the second is `llm_eval.shorten(full, ticks)` when ticks > 0."""
    full = T.build_tasks(pairs)
    out = []
    for t in full:
        s = L.shorten(t, ticks) if ticks and ticks < t.horizon else t
        out.append((t, s))
    return out


def tid(task) -> str:
    return f"mech.{task.family}-{task.seed}.h{task.horizon}"


def task_key(task) -> str:
    return digest(task.to_dict(), 8)


def controls_for(task):
    return [("idle", lambda: L.IdlePolicy(), "none"),
            ("random_settings", lambda: L.RandomRigPolicy(), "none"),
            ("random_tools", lambda: L.RandomPolicy(), "none"),
            ("scripted_experimenter", lambda: L.RigExperimenter(horizon=task.horizon), "actions"),
            ("privileged_reference", lambda: L.SchedulePolicy(L.reference_schedule(task)), "actions")]


def _trim(rec: dict, keep: str) -> dict:
    """Transcripts on disk: complete for model runs, the turns with an action for scripted controls, none for
    the idle and random controls (the record itself is always complete)."""
    if keep == "none":
        rec.pop("transcript", None)
    elif keep == "actions":
        rec["transcript"] = [r for r in rec["transcript"] if r.get("action")]
    return rec


def line(rec: dict) -> str:
    o, d, acc = rec["outcome"], rec["diagnosis"], rec["accounting"]
    s = f"{rec['run']:58s} "
    if rec["kind"] == "mechanism":
        s += (f"score {o['score']:7.3f} (halves {o['score_first_half']:6.2f} {o['score_second_half']:6.2f}) "
              f"success {str(o['success']):5s} sim_equal {rec['sim_check'].get('equal')} ")
    else:
        s += f"success {str(o['success']):5s} progress {o['best_progress']:.3f} revenue {o['independent_revenue_cents']:.0f} "
    return s + (f"primary {d['primary']} | turns {acc['turns']} tokens {acc['prompt_tokens']}+{acc['gen_tokens']} "
                f"wall {acc['wall_s']:.0f}s")


def stage_controls(store, tasks):
    for full, short in tasks:
        for task in ([short, full] if short is not full else [full]):
            for name, make, keep in controls_for(task):
                rid = f"{tid(task)}.{name}"
                cfg = {"task": task_key(task), "control": name, "dt": L.RUNNER_DT}
                rec = store.run(rid, lambda: stamp(_trim(L.run_mechanism(task, make(), run=rid), keep)), cfg)
                log(line(rec))


def references(store, task):
    """The controls on the same task, as references for the diagnosis: the scripted experimenter is the competent
    one; idle, random and the privileged reference are context only."""
    out = []
    for name, _, _ in controls_for(task):
        rec = store.get(f"{tid(task)}.{name}")
        if rec is None or rec.get("config_key") != store._key({"task": task_key(task), "control": name, "dt": L.RUNNER_DT}):
            raise SystemExit(f"run the `controls` stage first: {tid(task)}.{name} is missing or out of date")
        out.append(L.reference_of(rec, competent=name == "scripted_experimenter"))
    return out


def make_policy(backend, cfg: dict):
    if cfg["interface"] == "v4":
        return L.EvalLLMPolicy(backend=backend, **cfg)
    return LLMPolicy(backend=backend, **cfg)             # the 3.0 policy class itself


def stage_llm(store, tasks, tag: str, model: Model, cfg: dict, max_context: int, thinking: bool = False):
    """The model with one prompt configuration on every task that is not stored yet."""
    todo = []
    for _, task in tasks:
        rid = f"{tid(task)}.{tag}"
        key = {"task": task_key(task), "model": model.requested, "policy": cfg, "dt": L.RUNNER_DT,
               "max_context": max_context, "thinking": thinking}
        store.reload()
        if store.has(rid, key):
            log("stored:", line(store.get(rid)))
        else:
            todo.append((task, rid, key))
    if not todo:
        return
    backend = model.view(max_context, thinking)
    for task, rid, key in todo:
        refs = references(store, task)
        t0 = time.time()

        def fn():
            pol = make_policy(backend, cfg)
            rec = L.run_mechanism(task, pol, run=rid, references=refs,
                                  on_tick=lambda k, w: (k + 1) % 8 == 0 and log(f"  {rid} tick {k + 1}/{task.horizon} "
                                                                                f"{time.time() - t0:.0f}s"))
            rec["model"] = model.info(backend)
            return stamp(rec)
        model.runs.append(rid)
        log(line(store.run(rid, fn, key)))


def stage_open(store, tag: str | None, model: Model | None, cfg: dict | None, max_context: int = 0):
    """The open economy: the controls (always), then the model with one prompt configuration."""
    from crucible.agents.scripted import IdlePolicy, RandomPolicy
    from crucible.usefulness.agents import ExampleInductor
    goal = L.open_economy_goal(**OPEN)
    base = f"open.{OPEN['world_family']}-{OPEN['seed']}.d{OPEN['days']}"
    ctl = [("idle", IdlePolicy, "none", False), ("random_tools", RandomPolicy, "none", False),
           ("example_inductor", ExampleInductor, "actions", True)]
    refs = []
    for name, make, keep, competent in ctl:
        rid = f"{base}.{name}"
        rec = store.run(rid, lambda: stamp(_trim(L.run_open_economy(goal, make(), run=rid), keep)),
                        {"goal": goal.to_dict(), "control": name})
        log(line(rec))
        refs.append(L.reference_of(rec, competent))
    if model is None:
        return
    rid = f"{base}.{tag}"
    key = {"goal": goal.to_dict(), "model": model.requested, "policy": cfg, "dt": L.OPEN_DT,
           "max_context": max_context, "thinking": False}
    if store.has(rid, key):
        log("stored:", line(store.get(rid)))
        return
    backend = model.view(max_context, False)
    t0 = time.time()

    def fn():
        rec = L.run_open_economy(goal, make_policy(backend, cfg), run=rid, references=refs,
                                 on_day=lambda d, w: log(f"  {rid} day {d}/{OPEN['days']} {time.time() - t0:.0f}s"))
        rec["model"] = model.info(backend)
        return stamp(rec)
    model.runs.append(rid)
    log(line(store.run(rid, fn, key)))


# ====================================================================== summary
MECH_COLS = ["task", "ticks", "policy_tag", "score", "score_first_half", "score_second_half", "success",
             "setting_changes", "levers_moved", "levers_total", "read_spec", "hypotheses", "sign_accuracy", "primary",
             "conduct", "reference_score"]
DIAG_COLS = ["run", "turns", "attempts", "accepted_rate", "distinct_tools", "rate_valid", "rate_idle", "rate_form",
             "rate_discovery", "rate_belief", "rate_rule", "rate_resource", "loop_rate", "top_call_share",
             "effect_share", "primary", "conduct", "strategy", "difficulty", "not_determinable"]
IFACE_COLS = ["run", "interface", "catalogue", "prompts_truncated", "max_prompt_tokens", "results_cut", "gen_at_limit",
              "interface_clean", "interface_share", "prompt_tokens", "gen_tokens", "wall_s", "model_s"]
OPEN_COLS = ["policy_tag", "success", "progress", "organisations", "commits", "live_deployments", "listings",
             "independent_revenue_cents", "real_usage", "primary", "conduct", "top_call_share", "loop_rate"]


def stage_summary(store, out_dir: str, tasks):
    recs = list(store.runs.values())
    rows = L.summary_rows(recs)
    for r, rec in zip(rows, recs):
        r["policy_tag"] = ".".join(rec["run"].split(".")[3:])        # mech.<task>.h<ticks>.<tag> / open.<world>.d<n>.<tag>
    mech = sorted((r for r in rows if r["kind"] == "mechanism"), key=lambda r: (r["task"], r["ticks"], r["policy_tag"]))
    open_rows = [r for r in rows if r["kind"] == "open_economy"]
    llm = [r for r in rows if r.get("interface")]
    print("\n== Mechanism tasks: outcome (score: 0 = defaults left alone, 1 = private reference; success = public objective)\n")
    print(L.render_table(mech, MECH_COLS))
    print("\n== Model runs: diagnosis breakdown (rates over turns)\n")
    print(L.render_table(llm, DIAG_COLS))
    print("\n== Model runs: what the interface cut, tokens, seconds\n")
    print(L.render_table(llm, IFACE_COLS))
    if open_rows:
        print("\n== Open economy\n")
        print(L.render_table(open_rows, OPEN_COLS))
    agg = {}
    full_h = {f.id: f.horizon for f, _ in tasks}
    for r in mech:
        a = agg.setdefault(f"{r['policy_tag']}@{'full' if r['ticks'] == full_h.get(r['task']) else 'short'}",
                           {"n": 0, "score": 0.0, "second_half": 0.0, "successes": 0, "primaries": {}})
        a["n"] += 1
        a["score"] += r["score"]
        a["second_half"] += r["score_second_half"]
        a["successes"] += bool(r["success"])
        a["primaries"][r["primary"]] = a["primaries"].get(r["primary"], 0) + 1
    for a in agg.values():
        a["mean_score"] = round(a.pop("score") / a["n"], 4)
        a["mean_second_half_score"] = round(a.pop("second_half") / a["n"], 4)
    print("\n== Mechanism tasks: per policy\n")
    print(L.render_table([{"policy": k, **{x: v[x] for x in ("n", "mean_score", "mean_second_half_score", "successes")},
                           "primary": json.dumps(v["primaries"])} for k, v in sorted(agg.items())]))
    summary = {"format": "crucible-llm-eval-summary/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "runner": {"mechanism_dt_s": L.RUNNER_DT, "turns_per_tick": 3600 // L.RUNNER_DT, "open_dt_s": L.OPEN_DT},
               "configs": {"v4": V4, "v3": V3, "open": OPEN},
               "tasks": [{"task": s.id, "evaluated_ticks": s.horizon, "full_ticks": f.horizon, "engine": s.engine,
                          "income_cents_per_tick": s.income, "starting_balance_cents": s.endowment,
                          "goal_net_cents": s.goal_net_cents, "private": {
                              "default_net_per_tick": s.default_net, "reference_net_per_tick": s.reference_net,
                              "random_net_per_tick": s.random_net, "full_goal_net_cents": f.goal_net_cents,
                              "full_default_net_per_tick": f.default_net, "full_reference_net_per_tick": f.reference_net}}
                         for f, s in tasks],
               "code_digests": sorted({r.get("code_digest") for r in recs if r.get("code_digest")}),
               "per_policy": agg, "rows": rows,
               "conduct": {r["run"]: L.conduct(r) for r in recs},
               "diagnosis_compare": X.compare({r["run"]: r["diagnosis"] for r in recs if r["policy"].get("interface")}),
               "diagnosis_text": {r["run"]: X.render(r["diagnosis"]) for r in recs if r["policy"].get("interface")}}
    path = os.path.join(out_dir, "ow_llm_baseline_summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    log("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", choices=["controls", "llm", "ablation", "sens", "stale", "short", "open",
                                                  "open_v3", "open_controls", "summary"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--small-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--ticks", type=int, default=0, help="evaluate on the first N ticks, renormalised with "
                                                     "llm_eval.shorten (default 0: the task's full horizon)")
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--need-gb", type=float, default=None, help="free GPU memory required to load --model")
    ap.add_argument("--gpu-tries", type=int, default=8)
    ap.add_argument("--gpu-pause", type=int, default=60)
    ap.add_argument("--tasks", type=int, default=len(TASKS), help="use the first N tasks")
    ap.add_argument("--sens-tasks", type=int, default=3, help="tasks of the `sens` stage (the first N)")
    ap.add_argument("--short-ticks", type=int, default=32)
    ap.add_argument("--short-tasks", type=int, default=1)
    ap.add_argument("--think-tokens", type=int, default=2048, help="max_new_tokens with thinking enabled")
    ap.add_argument("--backend", choices=["hf", "echo"], default="hf")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "experiments"))
    a = ap.parse_args()
    store = L.RunStore(os.path.join(a.out, "ow_llm_baseline.json"), prefix="ow_llm_")
    tasks = load_tasks(TASKS[:a.tasks], a.ticks)
    for f, s in tasks:
        log(f"task {s.id}: {s.horizon} of {f.horizon} ticks, objective {s.goal_net_cents} cents, engine {s.engine}")
    short = lambda n: n.split("/")[-1].lower()                                            # noqa: E731
    models = Models(store, a.need_gb, a.gpu_tries, a.gpu_pause, a.backend == "echo")
    main_tag = short(a.model)
    failed: list[str] = []
    try:
        for stage in a.stages:
            log(f"=== stage {stage}")
            try:
                if stage == "controls":
                    stage_controls(store, tasks)
                elif stage == "llm":
                    stage_llm(store, tasks, f"{main_tag}-v4", models.get(a.model), V4, a.max_context)
                elif stage == "ablation":
                    stage_llm(store, tasks[:2], f"{main_tag}-v3", models.get(a.model), V3, 8192)
                elif stage == "stale":
                    stage_llm(store, tasks[:1], f"{main_tag}-v4-staleline", models.get(a.model), V4_STALE,
                              a.max_context)
                elif stage == "sens":
                    stage_llm(store, tasks[:a.sens_tasks], f"{short(a.small_model)}-v4", models.get(a.small_model), V4,
                              a.max_context)
                elif stage == "short":
                    cut = load_tasks(TASKS[:a.short_tasks], a.short_ticks)
                    stage_controls(store, [(s, s) for _, s in cut])
                    stage_llm(store, cut, f"{main_tag}-v4", models.get(a.model), V4, a.max_context)
                    stage_llm(store, cut, f"{main_tag}-v4-think", models.get(a.model),
                              {**V4, "max_new_tokens": a.think_tokens}, a.max_context, thinking=True)
                    stage_llm(store, cut, f"{main_tag}-v4-sampled", models.get(a.model),
                              {**V4, "temperature": 0.7, "seed": 0}, a.max_context)
                elif stage == "open_controls":
                    stage_open(store, None, None, None)
                elif stage == "open":
                    stage_open(store, f"{main_tag}-v4", models.get(a.model), V4, a.max_context)
                elif stage == "open_v3":
                    stage_open(store, f"{main_tag}-v3", models.get(a.model), V3, 8192)
                elif stage == "summary":
                    stage_summary(store, a.out, tasks)
            except Exception:  # noqa: BLE001 - one failed stage must not cost the stages after it
                traceback.print_exc()
                failed.append(stage)
    finally:
        models.free()
    log(f"done: {len(store.executed)} runs executed, {len(store.skipped)} already stored"
        + (f"; FAILED stages: {failed}" if failed else ""))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
