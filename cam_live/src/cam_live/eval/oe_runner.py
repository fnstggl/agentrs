"""cam_live.eval.oe_runner — substantive runs (Protocol v3): the full-size agent with open-ended generation on the
open-ended families, one objective per GPU-lock hold, the learner state carried between holds by its own checkpoint.

    python -m cam_live.eval.oe_runner sequence --condition E --seed 0 --plan SW:rle,SW:rpn,SW:merge,SW:topk \
        --decisions 20 --out R.json [--resume] [--save-exports] [--max-objectives 1 --no-lock]
    python -m cam_live.eval.oe_runner sequence --from-export DIR ...        (16.8 child: weights-only start)
    python -m cam_live.eval.oe_runner transfer --writer-root ROOT_{seed} --seeds 0,1 --plan SW:unique --out-dir D
    python -m cam_live.eval.oe_batch ...                                    (scheduler: one GPU lease, <= 3 children)

``sequence`` runs the listed objectives in order with ONE agent (weights carry over: spec 10.4). Before each
objective the shared GPU lock is taken (unless ``--no-lock``: a child of ``eval.oe_batch``, whose parent holds
the lease), the agent is rebuilt from the configuration and restored from its last
checkpoint (``CamLiveAgent.load``: complete learner state, same run directory), the objective runs under
``runtime.loop.LiveRuntime`` until it is verified achieved or its decision budget is spent, the checkpoint is
written, the lock released. ``DiskGuard.check`` runs before every checkpoint / export and ``apply_retention``
after every objective. The result file is rewritten after every objective (resumable with ``--resume``).

``sequence --from-export`` is the 16.8 transfer child: a new process gets only a weights-only export, builds a new
agent, loads the tensors strictly, resets fast beliefs and runs the plan (steps as in ``eval.transfer``);
``transfer`` starts such children in empty working directories with an empty ``CAM_LIVE_HOME``.

A run is refused (``UnavailableTools``) when a tool of a modality the task permits is reported unavailable by the
environment (e.g. a sandbox workspace the sandbox uid cannot reach): the unavailable tools are recorded per objective.

Recorded per objective: verified outcome (status, value, the verifier's score history), decisions, the provenance
of every candidate and of every chosen action (backbone-written vs schema walker), execution status of every
chosen action (executable fraction), failure classes, repeats of earlier failed actions, distinct strategies
(operation + argument-shape signatures), prediction NLL of resolved predictions, learning and meta-update
records (applied / skipped with reasons, meta-gradient norms, reward-version changes), compute and wall time.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

#: substantive agent settings: open-ended generation on, enough tokens for program text
SUBSTANTIVE_AGENT = dict(use_backbone=True, proposer_max_new_tokens=512, proposer_k=4, durable=False,
                         policy_update_every=3, policy_update_batch=12, planner_measure_every=4,
                         experiment_rate=0.1)
SUBSTANTIVE_LEARNER = {"min_independent_outcomes": 2, "warmup_steps": 2, "batch_size": 8, "slow_batch_size": 3,
                       "grad_last": 2, "adapter_every": 3}


def substantive_config(condition: str, seed: int, *, real: bool = True) -> Any:
    from ..config import CamLiveConfig
    from .conditions import eval_config
    cfg = CamLiveConfig.load() if real else CamLiveConfig.debug()
    cfg.device = "cuda" if real else "cpu"
    if not real:
        cfg.backbone.tiny_vocab = 4096
    cfg.seed = int(seed)
    cfg.optim.meta_min_independent_outcomes = 4
    a = dict(cfg.sections.get("agent", {}))
    a.update(SUBSTANTIVE_AGENT)
    if not real:
        a["proposer_max_new_tokens"] = 32
    cfg.sections["agent"] = a
    cred = dict(cfg.sections.get("credit", {}))
    cred["learner"] = {**cred.get("learner", {}), **SUBSTANTIVE_LEARNER}
    cfg.sections["credit"] = cred
    cfg = eval_config(cfg, condition, seed=seed)
    cfg.sections["agent"]["use_backbone"] = True          # eval_config's matched-walker switch does not apply here
    return cfg


def describe(agent: Any) -> Dict[str, Any]:
    """Model configuration, active parameter groups and learning pathways of a substantive run."""
    from .conditions import FastOnlyAgent, FrozenBackboneChooser, FrozenEvalAgent
    m = agent.model
    pc = {k: v for k, v in m.count_parameters().items() if k != "experiential_breakdown"}
    cls = type(agent).__name__
    weights_learn = not isinstance(agent, (FrozenBackboneChooser, FastOnlyAgent, FrozenEvalAgent)) and \
        bool(getattr(agent.settings, "learning", True))
    return {"agent_class": cls, "backbone": agent.cfg.backbone.name, "revision": agent.cfg.backbone.revision,
            "scale": agent.cfg.core.scale, "device": str(m.device), "parameters": pc,
            "agent_settings": {k: getattr(agent.settings, k) for k in ("use_backbone", "selection_rule",
                                                                         "proposer_max_new_tokens", "proposer_k",
                                                                         "experiment_rate", "adapters",
                                                                         "bridge_grounding", "retention")
                               if hasattr(agent.settings, k)},
            "loss_providers": [p.name for p in agent.learner.providers],
            "credit_assigner": agent.learner.credit is not None,
            "pathways": {"fast_beliefs": not isinstance(agent, (FrozenBackboneChooser, FrozenEvalAgent)),
                         "weights": weights_learn, "policy_update": weights_learn,
                         "meta_learning": weights_learn and int(agent.meta.mc.min_independent) < 10 ** 8,
                         "bootstrap_mix": float(agent.meta.mixer.mix),
                         "chooser": "frozen backbone" if isinstance(agent, FrozenBackboneChooser) else
                         agent.settings.selection_rule},
            "trainable_groups": sorted({g for _, _, g in m._named_groups()})}


def _sig(a: Any) -> str:
    def shape(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: shape(x) for k, x in sorted(v.items())}
        if isinstance(v, list):
            return [shape(x) for x in v[:2]]
        return type(v).__name__
    return a.operation + json.dumps(shape(a.arguments), sort_keys=True)


def _content_hash(a: Any) -> str:
    return hashlib.sha256(json.dumps({"op": a.operation, "args": a.arguments}, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


class UnavailableTools(RuntimeError):
    """A tool the task permits is unavailable in this environment: the run would not test the agent (INT3A-2)."""


_MODALITY_OF_INTERFACE = {"browser": "structured_browser"}
#: the tools each family cannot be attempted without (all unavailable permitted tools are recorded either way; e.g. the
#: publishing tools are legitimately unavailable where no publishing destination exists)
ESSENTIAL_PREFIXES = {"SW": ("sandbox.",), "MECH": ("sandbox.", "http."), "CU": ("browser.",),
                      "GOALFREE": ("sandbox.",)}


def unavailable_permitted_tools(env: Any, task: Any) -> List[Dict[str, str]]:
    permitted = {_MODALITY_OF_INTERFACE.get(i, i) for i in (task.interfaces or [])}
    return [{"tool": t.name, "modality": t.modality, "reason": getattr(t, "unavailable_reason", "")}
            for t in env.discover_tools() if not t.available and (not permitted or t.modality in permitted)]


def run_objective(agent: Any, oc: Any, family: str, problem: str, wseed: int, objective_id: str, run_dir: str,
                  clock: Any, *, max_decisions: int, k: int = 10, budget_s: float = 840.0,
                  allow_unavailable: bool = False) -> Dict[str, Any]:
    from ..runtime.loop import LiveRuntime, RuntimeConfig
    from .conditions import Timed
    env, policy, verifier, task = oc.open(family, problem, wseed, objective_id, max_decisions=max_decisions)
    unavailable = unavailable_permitted_tools(env, task)
    essential = [u for u in unavailable if u["tool"].startswith(ESSENTIAL_PREFIXES.get(family, ()))]
    if essential and not allow_unavailable:
        raise UnavailableTools(f"{objective_id}: tools the objective needs are unavailable: {essential[:5]}")
    agent.attach_verifier(verifier)
    timed = Timed(agent)
    rt = LiveRuntime(timed, env, policy, verifier, task, run_dir, clock,
                     RuntimeConfig(fsync=False, candidates_k=k, keep_checkpoints=1, verify_every_s=0.0,
                                   checkpoint_every_s=1e12, journal_every=10 ** 9, journal_min_interval_s=1e12))
    t0 = time.time()
    rt.start()
    steps = 0
    while rt.status == "running" and steps < 3 * max_decisions + 10:
        if time.time() - t0 > budget_s:
            rt._finish("stopped", f"wall budget {budget_s:.0f} s of this lock hold reached")
            break
        rt.step()
        steps += 1
    if rt.status == "running":
        rt._finish("stopped", "step cap reached")
    rt.close()
    final = verifier.verify(task)
    recs = [r for r in agent.store.trajectory(task.objective_id) if r.decision is not None]
    rows, failed, repeats = [], set(), 0
    for r in recs:
        ch = r.decision.chosen
        ok = r.result is not None and r.result.status == "ok"
        h = _content_hash(ch)
        if h in failed:
            repeats += 1
        if not ok:
            failed.add(h)
        rows.append({"operation": ch.operation, "provenance": ch.provenance, "ok": ok,
                     "status": r.result.status if r.result else None,
                     "failure": r.result.failure_class if r.result else None,
                     "detail": (r.result.failure_detail or "")[:200] if r.result else "",
                     "primitive_ops": int(r.result.cost.primitive_ops or 0) if r.result else 0,
                     "seconds": float(r.result.cost.seconds or 0.0) if r.result else 0.0,
                     "signature": _sig(ch), "n_candidates": len(r.decision.candidates),
                     "n_backbone_candidates": sum(1 for c in r.decision.candidates
                                                  if "backbone" in c.action.provenance.split("+")),
                     "args_preview": json.dumps(ch.arguments, default=str)[:300]})
    preds = [p for p in agent.predictions.records() if p.nll is not None and p.decision_id.startswith("dec_")
             and p.query.query_id in {q.query_id for r in recs for q in r.queries}]
    hist = getattr(verifier, "history", [])
    target = task.success.get("target") if task.success else None        # a goal-free period has none (X5)
    first_hit = None if target is None else next(
        (i for i, h in enumerate(hist) if (h.get("pass_fraction", h.get("value", 0.0)) or 0.0) >= float(target)), None)
    return {"family": family, "problem": problem, "world_seed": wseed, "objective_id": objective_id,
            "unavailable_permitted_tools": unavailable, "n_tools": len(env.discover_tools()),
            "runtime_status": rt.status, "reason": rt.status_reason[:200], "verified_status": final.status,
            "verified_value": float(final.value), "target": float(final.target), "achieved": final.status == "achieved",
            "decisions": len(recs), "steps": steps, "wall_s": time.time() - t0,
            "decisions_to_achieve": len(recs) if final.status == "achieved" else None,
            "executable_fraction": (sum(1 for x in rows if x["ok"]) / len(rows)) if rows else None,
            "backbone_chosen_fraction": (sum(1 for x in rows if "backbone" in x["provenance"].split("+")) /
                                         len(rows)) if rows else None,
            "repeats_of_failed_actions": repeats, "distinct_strategies": len({x["signature"] for x in rows}),
            "primitive_ops_total": sum(x["primitive_ops"] for x in rows),
            "skill_runs": sum(1 for x in rows if x["operation"] == "skill.run"),
            "skill_runs_ok": sum(1 for x in rows if x["operation"] == "skill.run" and x["ok"]),
            "eval_code": os.environ.get("CAM_LIVE_EVAL_COMMIT", ""),
            "mean_prediction_nll": (sum(p.nll for p in preds) / len(preds)) if preds else None,
            "n_predictions_resolved": len(preds), "verifier_history": hist[-50:], "first_verified_hit": first_hit,
            "compute": {"seconds": dict(timed.seconds), "cpu_seconds": dict(timed.cpu_seconds)},
            "counters": dict(rt.counters), "errors": rt._errors[-5:], "decisions_detail": rows}


def load_export(agent: Any, export_dir: str) -> Dict[str, Any]:
    """Spec 16.8 steps 4-8 inside a fresh process: only the weights-only export's tensors are loaded (strict) into a
    newly built agent; fast beliefs reset to the loaded prior. Returns what the process had before its first
    decision (empty store / registries / fast state, the files in its working directory)."""
    from ..credit.checkpoint import collect_weights, load_weights, load_weights_only, tensor_hash
    weights, meta = load_weights_only(export_dir, map_location="cpu")
    rep = load_weights(agent.model, weights, strict=True)
    agent.model.mark_adapters_updated()
    agent.learner.reset_fast()
    agent.predictor.set_fast_state(agent.learner.fast_state)
    hashes = {g: tensor_hash(d) for g, d in collect_weights(agent.model).items() if d}
    return {"export": os.path.abspath(export_dir), "cwd_files_at_start": sorted(os.listdir(os.getcwd())),
            "store_records": agent.store.count(), "predictions_registered": len(agent.predictions),
            "experiments": len(agent.experiments.records()),
            "fast_state_updates": {k: agent.learner.fast_state.n_updates(k) for k in ("count", "real", "binary")},
            "load_clean": rep.clean, "tensors_loaded": len(rep.loaded), "pid": os.getpid(),
            "hashes_match_export": all(hashes.get(g) == h for g, h in meta.get("group_hashes", {}).items())}


def cmd_transfer(a: argparse.Namespace) -> int:
    """Fresh vs experienced in independent processes (16.8): each (version, seed) is a new Python process started in
    an empty working directory with an empty CAM_LIVE_HOME, given only a copy of one weights-only export."""
    import subprocess
    import tempfile
    from .experiments import EVAL_HOME
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    codes = []
    for seed in [int(x) for x in a.seeds.split(",")]:
        for version in ("initial", "experienced"):
            src = os.path.join(a.writer_root.format(seed=seed), "export_pre" if version == "initial" else
                               "export_post")
            res = os.path.join(out_dir, f"{version}_s{seed}.json")
            if os.path.exists(res) and json.load(open(res)).get("executed"):
                continue
            work = tempfile.mkdtemp(prefix=f"camlive_oe_transfer_{version}_{seed}_", dir=os.path.join(EVAL_HOME, "tmp")
                                    if os.path.isdir(os.path.join(EVAL_HOME, "tmp")) else None)
            os.chmod(work, 0o755)                        # the sandbox uid must be able to reach its workspace
            own = os.path.join(work, "export")
            shutil.copytree(src, own)
            cwd, home = os.path.join(work, "cwd"), os.path.join(work, "home")
            os.makedirs(cwd)
            os.makedirs(home)
            env = {k: v for k, v in os.environ.items() if not k.startswith("CAM_LIVE_") or k.startswith("CAM_LIVE_EVAL")
                   or k == "CAM_LIVE_GPU_LOCK"}
            env["CAM_LIVE_HOME"] = home
            cmd = [sys.executable, "-m", "cam_live.eval.oe_runner", "sequence", "--condition", "E", "--seed",
                   str(seed), "--plan", a.plan, "--decisions", str(a.decisions), "--out", res, "--root",
                   os.path.join(work, "run"), "--from-export", own, "--tag", f"transfer_{version}"]
            if a.cpu:
                cmd.append("--cpu")
            print(f"{time.strftime('%H:%M:%S')} {version} s{seed}", flush=True)
            p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=6 * 3600)
            codes.append(p.returncode)
            if p.returncode != 0:
                print(p.stderr[-2000:], file=sys.stderr)
            elif os.path.exists(res):
                d = json.load(open(res))
                d["version"] = version
                json.dump(d, open(res, "w"), default=str)
            shutil.rmtree(os.path.join(work, "run", "agent", "learner", "embed_cache"), ignore_errors=True)
    return 0 if all(c == 0 for c in codes) else 1


def _meta_records(run_dir: str) -> List[Dict[str, Any]]:
    from ..contracts import RunFiles
    from .experiments import _records
    out = []
    for e in _records(run_dir, RunFiles.META):
        m = e.get("data", {})
        out.append({k: m.get(k) for k in ("applied", "skipped_reason", "meta_grad_norm", "n_independent_outcomes",
                                           "reward_version_before", "reward_version_after", "bootstrap_mix",
                                           "estimator")})
    return out


def _learning_records(run_dir: str) -> List[Dict[str, Any]]:
    from ..contracts import RunFiles
    from .experiments import _records
    return [{k: e.get("data", {}).get(k) for k in ("timescale", "param_groups", "weight_delta")}
            for e in _records(run_dir, RunFiles.LEARNING)]


def _plan(s: str) -> List[Tuple[str, str]]:
    out = []
    for part in s.split(","):
        fam, _, prob = part.partition(":")
        out.append((fam, prob))
    return out


def cmd_sequence(a: argparse.Namespace) -> int:
    torch.set_num_threads(int(a.threads))            # a shared machine: never oversubscribe the CPU
    from ..storage import DiskGuard, apply_retention
    from .experiments import EVAL_HOME, code_identity, export_checkpoint
    root = os.path.abspath(a.root or os.path.join(EVAL_HOME, "oe", f"{a.condition}_s{a.seed}_{a.tag}"))
    if not a.resume and os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    guard = DiskGuard(root)
    res: Dict[str, Any] = {"executed": False, "condition": a.condition, "seed": a.seed, "plan": a.plan,
                           "decisions_per_objective": a.decisions, "real": not a.cpu, "root": root, "objectives": [],
                           "code": code_identity()}
    if a.resume and os.path.exists(a.out):
        with open(a.out) as f:
            res = json.load(f)
    plan = _plan(a.plan)
    done = len(res["objectives"])
    t_start = time.time()
    clock_state = os.path.join(root, "clock.json")
    from ..validation import OffsetClock
    clock = OffsetClock()                                 # one clock and one computer for the whole process
    if os.path.exists(clock_state):
        clock.offset = json.load(open(clock_state))["offset"]
    computer = make_computer(root, clock, a)
    for i in range(done, len(plan)):
        if a.max_objectives and i - done >= a.max_objectives:
            break
        fam, prob = plan[i]
        guard.check(need_bytes=2 * 2 ** 30, what="objective checkpoint")
        lock = contextlib.nullcontext()
        if not a.cpu and not a.no_lock:
            from ..nn.gpu import gpu_lock
            lock = gpu_lock(min_free_gb=a.min_free_gb, timeout_s=a.lock_timeout_s)
        with lock:
            t_lock = time.time()
            cfg = substantive_config(a.condition, a.seed, real=not a.cpu)
            agent = _build(a.condition, cfg, os.path.join(root, "agent"), clock)
            ck = os.path.join(root, "ckpt_latest")
            if i > 0 and os.path.exists(ck):
                agent.load(ck)
            elif i == 0 and a.from_export:
                res["fresh_process"] = load_export(agent, a.from_export)
                res["configuration"] = describe(agent)
            elif i == 0:
                res["configuration"] = describe(agent)
                if a.save_exports:
                    res["checkpoints"] = {"pre": export_checkpoint(agent, os.path.join(root, "ckpt_pre"),
                                                                   os.path.join(root, "export_pre"),
                                                                   note="before any experience")}
            h0 = agent.model.group_hashes()
            # world seeds by position among the assigned objectives (a goal-free period before them - X5 - does
            # not shift them, so X5's objectives are the same instances as X1's)
            k = i - sum(1 for f, _ in plan[:i] if f == "GOALFREE")
            wseed = 1000 * (k + 1) + a.seed
            if fam == "MECH" and prob:
                wseed = int(prob) * 100 + a.seed
            ob = run_objective(agent, computer, fam, prob, wseed, f"obj_{fam}_{prob}_{i}_s{a.seed}",
                               os.path.join(root, f"obj{i:02d}"), clock, max_decisions=a.decisions,
                               budget_s=a.hold_s - (time.time() - t_lock) - 60.0,
                               allow_unavailable=a.allow_unavailable)
            ob["index"] = i
            ob["group_changed"] = {g: h0.get(g) != h for g, h in agent.model.group_hashes().items()}
            ob["meta_updates"] = _meta_records(os.path.join(root, f"obj{i:02d}"))
            ob["learning_updates"] = _learning_records(os.path.join(root, f"obj{i:02d}"))
            ob["mix"] = float(agent.meta.mixer.mix)
            ob["reward_version"] = agent.meta.reward.version
            ob["lock_hold_s"] = time.time() - t_lock
            ob["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
            guard.check(need_bytes=2 * 2 ** 30, what="checkpoint")
            tmp = ck + ".new"
            shutil.rmtree(tmp, ignore_errors=True)
            agent.save(tmp)
            shutil.rmtree(ck, ignore_errors=True)
            os.rename(tmp, ck)
            json.dump({"offset": clock.offset}, open(clock_state, "w"))
            if a.save_exports and i == len(plan) - 1:
                res.setdefault("checkpoints", {})["post"] = export_checkpoint(
                    agent, os.path.join(root, "ckpt_post"), os.path.join(root, "export_post"),
                    note=f"after {len(plan)} objectives")
            agent.close()
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        res["objectives"].append(ob)
        res["retention"] = apply_retention(root, keep=1, keep_optim=1, drop_protected_optim=True,
                                           drop_groups=("*/obj[0-9][0-9]/checkpoints",))["bytes"]
        with open(a.out, "w") as f:
            json.dump(res, f, default=str)
        print(f"[{a.condition} s{a.seed}] objective {i} {fam}:{prob} {ob['verified_status']} value "
              f"{ob['verified_value']:.3f} decisions {ob['decisions']} exec {ob['executable_fraction']} "
              f"backbone {ob['backbone_chosen_fraction']} hold {ob['lock_hold_s']:.0f}s", flush=True)
    computer.close()
    res["executed"] = len(res["objectives"]) == len(plan)
    res["wall_seconds"] = res.get("wall_seconds", 0.0) + time.time() - t_start
    with open(a.out, "w") as f:
        json.dump(res, f, default=str)
    return 0


class CUComputer:
    """The staging computer-use families (``validation.StagingComputer``: one persistent computer, browser and skill
    library across objectives) behind the same ``open`` interface as ``OpenEndedComputer``."""

    def __init__(self, root: str, clock: Any, *, skills: bool = True):
        from ..validation import StagingComputer
        self.sc = StagingComputer(os.path.join(root, "cu"), clock, exclude=() if skills else ("skill",))

    def open(self, family: str, problem: str, seed: int, objective_id: str, *, max_decisions: int) -> Tuple:
        task = self.sc.task(problem, max_decisions=max_decisions, objective_id=objective_id)
        env, policy, verifier = self.sc.open(task, self.sc.start_url(problem))
        return env, policy, verifier, task

    def close(self) -> None:
        self.sc.close()


class _Computers:
    """Dispatch by family: SW / MECH -> OpenEndedComputer, CU -> CUComputer (both persistent)."""

    def __init__(self, root: str, clock: Any, a: argparse.Namespace):
        self.root, self.clock, self.a = root, clock, a
        self._oe: Any = None
        self._cu: Any = None

    def open(self, family: str, *args: Any, **kw: Any) -> Tuple:
        if family == "CU":
            if self._cu is None:
                self._cu = CUComputer(self.root, self.clock, skills=not self.a.no_skills)
            return self._cu.open(family, *args, **kw)
        if self._oe is None:
            from .openended import OpenEndedComputer
            self._oe = OpenEndedComputer(os.path.join(self.root, "computer"), self.clock)
        return self._oe.open(family, *args, **kw)

    def close(self) -> None:
        for c in (self._oe, self._cu):
            if c is not None:
                with contextlib.suppress(Exception):
                    c.close()


def make_computer(root: str, clock: Any, a: argparse.Namespace) -> _Computers:
    return _Computers(root, clock, a)


def _build(condition: str, cfg: Any, run_dir: str, clock: Any) -> Any:
    from .conditions import _CLASSES
    return _CLASSES[condition](cfg, run_dir, clock=clock)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cam_live.eval.oe_runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sequence")
    s.add_argument("--condition", default="E")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--plan", required=True)
    s.add_argument("--decisions", type=int, default=20)
    s.add_argument("--out", required=True)
    s.add_argument("--root", default="")
    s.add_argument("--tag", default="seq")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--save-exports", dest="save_exports", action="store_true")
    s.add_argument("--max-objectives", dest="max_objectives", type=int, default=0)
    s.add_argument("--hold-s", dest="hold_s", type=float, default=870.0, help="max seconds per GPU-lock hold")
    s.add_argument("--min-free-gb", dest="min_free_gb", type=float, default=14.0)
    s.add_argument("--lock-timeout-s", dest="lock_timeout_s", type=float, default=7200.0)
    s.add_argument("--cpu", action="store_true", help="tiny CPU backbone: plumbing check only, never substantive")
    s.add_argument("--no-skills", dest="no_skills", action="store_true",
                   help="CU: primitive-only operation (skill executor excluded) - arm of X4")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--no-lock", dest="no_lock", action="store_true",
                   help="do not take the GPU lock: the parent (eval.oe_batch) holds the lease")
    s.add_argument("--allow-unavailable", dest="allow_unavailable", action="store_true",
                   help="run even when permitted tools are unavailable (diagnostics only)")
    s.add_argument("--from-export", dest="from_export", default="",
                   help="16.8 child: start from this weights-only export only (fresh process)")
    s.set_defaults(fn=cmd_sequence)
    t = sub.add_parser("transfer", help="fresh vs experienced in independent processes (16.8)")
    t.add_argument("--writer-root", dest="writer_root", required=True,
                   help="run root of the writer with {seed}, holding export_pre / export_post")
    t.add_argument("--seeds", default="0,1,2")
    t.add_argument("--plan", required=True)
    t.add_argument("--decisions", type=int, default=20)
    t.add_argument("--out-dir", dest="out_dir", required=True)
    t.add_argument("--cpu", action="store_true")
    t.set_defaults(fn=cmd_transfer)
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
