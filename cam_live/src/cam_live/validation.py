"""cam_live.validation — Phase-4 staging validation of the integrated agent (spec Part 19, Phase 4; LX-11, IN-07).

    run_staging(cfg, *, out_dir=None, real_model=True, n_decisions=40, seed=0) -> report (dict)

One :class:`cam_live.agent.CamLiveAgent` (real Qwen3-4B + 50m core on the GPU under the shared lock, or the tiny CPU
configuration with ``real_model=False``) is driven by the real persistent runtime (``runtime.loop.LiveRuntime``)
through two controlled environments, one after the other, with the same weights:

A. the general-purpose computer environment (``harness.env.ComputerEnvironment``) with the local staging sites
   (Chromium through Playwright), the generic HTTP executor and the staging Discord service, authorized by the real
   ``harness.auth.PolicyService`` in staging mode and judged by the runtime's independent
   ``MetricThresholdVerifier`` reading the staging application's own state;
B. a family of LabWorld lifetimes (delayed consequences, several independently verified objectives) with the
   signing ``LabVerifier`` and the same authorization service.

Measured, on a fixed held-out context that the agent never acted in: the trainable policy's distribution over a
fixed candidate set before any experience and after both phases, and what restoring the old weights does to it.
Everything is written to ``<evidence_dir>/staging_<UTC timestamp>.json``. Results are staging / diagnostic results;
nothing here contacts any external service.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import statistics
import subprocess
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import PROJECT_ROOT, CamLiveConfig
from .contracts import Limits, Quantity, RunFiles, TaskSpec
from .runtime.logs import iter_jsonl

DAY = 86400.0
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "docs", "evidence", "validation")
#: requirement ids this validation produces evidence for (each one is checked in ``checks`` of the report)
REQUIREMENTS = ("IN-07", "LX-11", "IN-01", "NA-06", "CU-22", "CU-30", "CL-02", "CL-03", "CL-09", "MR-01", "MR-03",
                "MR-05", "MR-06", "PI-02", "RC-01", "AE-05", "AE-08")


class OffsetClock:
    """Wall-clock time that can also jump forward (simulated waiting, matured delayed consequences). Monotone."""

    def __init__(self) -> None:
        self.offset = 0.0
        self._last = 0.0

    def now(self) -> float:
        t = time.time() + self.offset
        self._last = max(self._last, t)
        return self._last

    def advance(self, dt: float) -> float:
        if dt < 0:
            raise ValueError("time does not go backwards")
        self.offset += float(dt)
        return self.now()


class SiteMetrics:
    """``runtime.verifier.MetricSource`` over a staging site's own state (read directly, not through the agent)."""

    live = False

    def __init__(self, site: Any, name: str):
        self.site, self.name = site, name

    def read(self) -> List[Quantity]:
        st = self.site.state()
        return [Quantity("staging.items.count", float(len(st.get("items", []))), kind="count"),
                Quantity("staging.uploads.count", float(len(st.get("uploads", []))), kind="count")]


def lab_config(seed: int, **kw: Any) -> Any:
    """The LabWorld family used for delayed consequences: short horizons so several lifetimes fit a validation."""
    from .diagnostics.world import LabConfig
    base = dict(seed=seed, n_channels=4, target=12, budget=40.0, horizon_s=2 * DAY, outcome_horizon_s=6 * 3600.0,
                mean_delay_s=(1800.0, 4 * 3600.0), at_most_once=True, family="lab/staging")
    base.update(kw)
    return LabConfig(**base)


class LookOnlyGrants:
    """The authorization service as the runtime sees it: ``check_grant`` only looks (``consume=False``); the
    computer environment, which executes, consumes the grant (EA-2: single use, one defined point of consumption).
    Workaround for change request M-int-6: ``runtime.loop`` consumes the grant itself before ``env.execute``, after
    which the environment refuses the action ("the grant does not verify")."""

    def __init__(self, policy: Any):
        self._p = policy

    def check_grant(self, action: Any, decision: Any) -> bool:
        return self._p.check_grant(action, decision, consume=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._p, name)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _tv(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * (p - q).abs().sum())


def _stats(xs: Sequence[float]) -> Dict[str, float]:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": statistics.fmean(xs), "max": max(xs), "min": min(xs),
            "nonzero": sum(1 for x in xs if x > 0)}


def _gpu_peak() -> Optional[float]:
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None


# --------------------------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------------------------
def staging_config(cfg: Optional[CamLiveConfig], real_model: bool, seed: int) -> CamLiveConfig:
    if real_model:
        cfg = cfg or CamLiveConfig.load()
        cfg.device = "cuda"
        agent = dict(proposer_max_new_tokens=160, proposer_k=4, policy_update_every=3, policy_update_batch=12,
                     planner_measure_every=4, experiment_rate=0.1)
        learner = {"min_independent_outcomes": 2, "warmup_steps": 2, "batch_size": 8, "slow_batch_size": 3,
                   "grad_last": 2, "adapter_every": 3}
    else:
        cfg = cfg or CamLiveConfig.debug()
        cfg.backbone.tiny_vocab = 4096
        agent = dict(proposer_max_new_tokens=16, durable=False, policy_update_every=3, policy_update_batch=12,
                     planner_measure_every=4, experiment_rate=0.1)
        learner = {"min_independent_outcomes": 2, "warmup_steps": 0, "slow_batch_size": 3, "grad_last": 2,
                   "adapter_every": 3}
    cfg.seed = int(seed)
    cfg.optim.meta_min_independent_outcomes = 4
    sec = dict(cfg.sections.get("agent", {}))
    sec.update({k: v for k, v in agent.items() if k not in sec})
    cfg.sections["agent"] = sec
    cred = dict(cfg.sections.get("credit", {}))
    cred["learner"] = {**learner, **cred.get("learner", {})}
    cfg.sections["credit"] = cred
    return cfg


# --------------------------------------------------------------------------------------------------------------
# held-out behaviour probe
# --------------------------------------------------------------------------------------------------------------
class HeldOutProbe:
    """A fixed context and candidate set from a LabWorld seed the agent never acts in."""

    def __init__(self, agent: Any, seed: int = 9_999):
        from .agent import observation_item, tools_item
        from .contracts import ExperienceItem, ItemKind, SimClock, Trust
        from .diagnostics.world import LabWorld, lab_task
        from .predict.candidates import schema_candidates
        world = LabWorld(lab_config(seed), SimClock())
        task = lab_task(world, "obj_heldout_probe")
        now = world.clock.now()
        self.items = [ExperienceItem(kind=ItemKind.GOAL, text=task.objective, ts=now, ref="probe:goal",
                                     trust=Trust.OPERATOR)]
        self.items.append(tools_item(world.discover_tools(), now, "probe"))
        for o in world.observe():
            self.items.append(observation_item(o))
        self.goal = task.objective
        self.candidates = schema_candidates(world.discover_tools(), per_tool=4, id_prefix="probe")[:12]
        self.agent = agent

    def distribution(self) -> torch.Tensor:
        return self.agent.policy_distribution(self.items, self.candidates, goal_text=self.goal)


def weight_swap(model: Any, state: Dict[str, torch.Tensor], groups: Sequence[str]) -> None:
    """Load the tensors of ``groups`` from ``state`` (a ``trainable_state()`` snapshot) into ``model``."""
    names = {n for n, p, g in model._named_groups() if g in groups}
    sub = {k: v for k, v in state.items() if k in names}
    model.load_trainable_state(sub, strict=False)


# --------------------------------------------------------------------------------------------------------------
# phase A: the computer environment with the staging sites and the staging Discord service
# --------------------------------------------------------------------------------------------------------------
def phase_computer(agent: Any, clock: OffsetClock, root: str, n_decisions: int, budget_s: float) -> Dict[str, Any]:
    from .harness.auth import PolicyService
    from .harness.discord import DiscordExecutor
    from .harness.http_api import ApiModality, HttpApiExecutor
    from .harness.service import build_environment
    from .harness.staging import StagingSites
    from .harness.staging_discord import StagingDiscord, StagingDiscordServer
    from .runtime.loop import LiveRuntime, RuntimeConfig
    from .runtime.verifier import MetricThresholdVerifier
    run = os.path.join(root, "computer")
    os.makedirs(run, exist_ok=True)
    out: Dict[str, Any] = {"run_dir": run}
    sites = StagingSites().start()
    sd = StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0)))
    st = sd.state
    env = None
    try:
        hosts = [u.replace("http://", "") for u in sites.urls.values()]
        discord_host = sd.base_url.split("/api")[0].replace("http://", "")
        base_items = len(sites.structured.state()["items"])
        task = TaskSpec(
            objective_id="obj_staging_computer",
            objective=f"Increase the number of items recorded by the structured staging application to "
                      f"{base_items + 2}.",
            success={"verifier": "metric_threshold", "metric": "staging.items.count", "target": base_items + 2},
            family="staging/computer",
            resources={"owned": hosts + [f"channel:discord:{st.general_id}", f"account:discord:{st.guild_id}"],
                       "preauthorized": hosts + [f"channel:discord:{st.general_id}"],
                       "read": hosts + [discord_host], "sites": dict(sites.urls),
                       "discord": {"guild_id": st.guild_id, "channels": {"general": st.general_id}}},
            interfaces=["structured_browser", "api", "internal"],
            limits=Limits(duration_s=2 * 3600.0, money=1.0, compute_s=0.0, max_external_actions=200,
                          max_decisions=int(n_decisions)),
            data_permissions={"aggregate_metrics": True}, mode="staging")
        policy = PolicyService(os.path.join(run, "auth_state"), clock=clock)
        policy.register_task(task, "operator")
        policy.put_secret("discord_bot_token", st.token, [discord_host])
        exec_dir = os.path.join(run, "exec")
        try:                                               # every executor of the harness, configured from the task
            from .harness.env import ComputerEnvironment
            from .harness.factories import all_executors
            env = ComputerEnvironment(exec_dir, policy, clock=clock, secrets=policy.secrets_handle(),
                                      provenance="staging")
            for ex in all_executors(exec_dir, clock, env=env, policy=policy, secrets=policy.secrets_handle(),
                                    task=task, discord_base_url=sd.base_url):
                env.register(ex)
            out["executors"] = "harness.factories.all_executors"
        except ImportError:                                # older harness: register the executors directly
            env = build_environment(exec_dir, policy, provenance="staging", clock=clock,
                                    secrets=policy.secrets_handle(), browser=True)
            disc = DiscordExecutor(sd.base_url, guild_id=st.guild_id, secrets=policy.secrets_handle(),
                                   credential_probe=lambda name: name == "discord_bot_token")
            env.register(ApiModality([HttpApiExecutor(), disc]))
            out["executors"] = "internal, structured_browser, api(http, discord) registered directly"
        verifier = MetricThresholdVerifier(SiteMetrics(sites.structured, "staging-structured"), clock,
                                           os.path.join(run, "verifier"), pinned_task=task)
        agent.attach_verifier(verifier)
        rt = LiveRuntime(agent, env, LookOnlyGrants(policy), verifier, task, os.path.join(run, "runtime"), clock,
                         RuntimeConfig(fsync=True, candidates_k=10, keep_checkpoints=2, verify_every_s=0.0))
        rt.start()
        if rt.status != "running":
            raise RuntimeError(f"the runtime did not start (status {rt.status!r}): use a fresh out_dir")
        t0 = time.time()
        steps = 0
        while rt.status == "running" and steps < 4 * n_decisions and time.time() - t0 < budget_s:
            rt.step()
            steps += 1
            if rt.counters["decisions"] >= n_decisions:
                break
        rt._commit(force=True)
        rt.close()
        out.update(status=rt.status, steps=steps, seconds=time.time() - t0, counters=dict(rt.counters),
                   tools=sorted(t.name for t in env.discover_tools())[:80],
                   verified=rt._last_verified.get("outcome", {}), errors=rt._errors[-10:],
                   staging_items_after=len(sites.structured.state()["items"]), staging_items_before=base_items,
                   discord_messages=sum(len(v) for v in st.messages.values()),
                   site_hits={k: dict(s.hits) for k, s in sites.sites.items()}, runtime_dir=rt.run_dir)
    finally:
        with contextlib.suppress(Exception):
            if env is not None:
                env.close()
        with contextlib.suppress(Exception):
            sd.close()
        with contextlib.suppress(Exception):
            sites.stop()
    return out


# --------------------------------------------------------------------------------------------------------------
# phase B: a LabWorld family (delayed consequences, several verified lifetimes)
# --------------------------------------------------------------------------------------------------------------
def phase_lab(agent: Any, clock: OffsetClock, root: str, seeds: Sequence[int], per_lifetime: int,
              budget_s: float, target: int = 24) -> Dict[str, Any]:
    from .diagnostics.world import LabVerifier, LabWorld, lab_task
    from .harness.auth import PolicyService
    from .runtime.loop import LiveRuntime, RuntimeConfig
    out: Dict[str, Any] = {"lifetimes": []}
    t0 = time.time()
    for seed in seeds:
        if time.time() - t0 > budget_s:
            out["stopped"] = f"time budget of {budget_s:.0f} s reached before lifetime {seed}"
            break
        run = os.path.join(root, f"lab_{seed}")
        world = LabWorld(lab_config(seed, target=int(target)), clock)
        task = lab_task(world, f"obj_staging_lab_{seed}")
        task.limits.max_decisions = int(per_lifetime)
        verifier = LabVerifier(world)
        agent.attach_verifier(verifier)
        policy = PolicyService(os.path.join(run, "auth_state"), clock=clock)
        policy.register_task(task, "operator")
        rt = LiveRuntime(agent, world, policy, verifier, task, os.path.join(run, "runtime"), clock,
                         RuntimeConfig(fsync=True, candidates_k=10, keep_checkpoints=2))
        rt.start()
        t1, steps = time.time(), 0
        while rt.status == "running" and steps < 6 * per_lifetime and time.time() - t0 < budget_s:
            rt.step()
            steps += 1
        if rt.status == "running":
            rt._finish("stopped", "validation step budget reached")
        rt.close()
        out["lifetimes"].append({"seed": seed, "status": rt.status, "steps": steps, "seconds": time.time() - t1,
                                 "counters": dict(rt.counters), "verified": rt._last_verified.get("outcome", {}),
                                 "errors": rt._errors[-5:], "runtime_dir": rt.run_dir})
    return out


# --------------------------------------------------------------------------------------------------------------
# evidence from the logs
# --------------------------------------------------------------------------------------------------------------
def _records(run_dirs: Sequence[str], name: str) -> List[Dict[str, Any]]:
    out = []
    for d in run_dirs:
        p = os.path.join(d, name)
        if os.path.exists(os.path.join(d, "runtime_logs.json")):
            with open(os.path.join(d, "runtime_logs.json")) as f:
                p = os.path.join(d, json.load(f).get("log_dir", ""), name)
        out += [e for e in iter_jsonl(p)]
    return out


def collect_evidence(agent: Any, run_dirs: Sequence[str]) -> Dict[str, Any]:
    ups = [e["data"] for e in _records(run_dirs, RunFiles.LEARNING)]
    metas = [e["data"] for e in _records(run_dirs, RunFiles.META)]
    decisions = _records(run_dirs, RunFiles.DECISIONS)
    res = [e for e in _records(run_dirs, RunFiles.PREDICTIONS) if e.get("type") == "resolution"]
    deltas: Dict[str, List[float]] = {}
    for u in ups:
        for g, v in (u.get("weight_delta") or {}).items():
            deltas.setdefault(f"{u.get('timescale')}:{g}", []).append(float(v))
    reg = [p for p in agent.predictions.records() if p.resolved_status != "pending"]
    resolutions = [{"metric": p.query.metric.split(".act_")[0][:60], "kind": p.query.kind,
                    "predicted_mean": p.dist.mean, "predicted_std": p.dist.std, "observed": p.resolved_value,
                    "status": p.resolved_status, "nll": p.nll, "made_at": p.made_at, "resolved_at": p.resolved_at}
                   for p in reg]
    by_kind: Dict[str, List[float]] = {}
    for r in resolutions:
        if r["nll"] is not None:
            by_kind.setdefault(r["kind"], []).append(float(r["nll"]))
    recs = agent.store.all()
    rewards = [r.intrinsic_reward for r in recs if r.intrinsic_reward is not None]
    links = sum(len(agent.store.links(exp_id=r.exp_id)) for r in recs)
    return {
        "n_decisions": len(decisions), "n_records": len(recs),
        "decision_rules": dict(_count(d["data"]["decision"].get("exploration", "") for d in decisions)),
        "candidate_provenance": dict(_count(p for d in decisions for c in d["data"]["decision"]["candidates"]
                                            for p in c["action"]["provenance"].split("+"))),
        "chosen_operations": dict(_count(d["data"]["decision"]["chosen"]["operation"] for d in decisions)),
        "learning_updates": dict(_count(u.get("timescale", "") for u in ups)),
        "weight_delta_by_timescale_group": {k: _stats(v) for k, v in sorted(deltas.items())},
        "learning_update_records": [{k: u.get(k) for k in ("timescale", "trigger", "param_groups", "weight_delta",
                                                           "losses", "prediction_change", "model_version")}
                                    for u in ups if u.get("timescale") != "fast"][-40:],
        "meta_updates": [{k: m.get(k) for k in ("applied", "skipped_reason", "meta_grad_norm", "n_independent_outcomes",
                                                "n_query", "n_support", "estimator", "reward_version_before",
                                                "reward_version_after", "bootstrap_mix", "outer_objective")}
                         for m in metas],
        "meta_applied": sum(1 for m in metas if m.get("applied")),
        "meta_skipped_reasons": dict(_count(m.get("skipped_reason", "").split(":")[0] for m in metas
                                            if not m.get("applied"))),
        "intrinsic_rewards": _stats([abs(x) for x in rewards]) | {"n_records_with_reward": len(rewards),
                                                                   "values_first_20": rewards[:20]},
        "prediction_resolutions": {"n_registered": len(agent.predictions), "n_resolved": len(reg),
                                   "runtime_resolution_lines": len(res),
                                   "mean_nll_by_kind": {k: statistics.fmean(v) for k, v in by_kind.items()},
                                   "all_resolved_after_registration": all(
                                       r["resolved_at"] is not None and r["resolved_at"] > r["made_at"]
                                       for r in resolutions),
                                   "examples": resolutions[:25]},
        "delayed_outcome_links": links,
        "agent_status": agent.status(),
    }


def _count(xs: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# --------------------------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------------------------
def run_staging(cfg: Optional[CamLiveConfig] = None, *, out_dir: Optional[str] = None, real_model: bool = True,
                n_decisions: int = 40, seed: int = 0, evidence_dir: Optional[str] = None,
                budget_s: float = 780.0, lab_seeds: Sequence[int] = (101, 102),
                computer_share: float = 0.15, lab_target: int = 24) -> Dict[str, Any]:
    """Run the staging validation and write its evidence file. ``budget_s`` bounds the time the GPU lock is held
    for the learning phases (the lock is held for at most ~15 minutes in total)."""
    from .agent import CamLiveAgent
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    cfg = staging_config(cfg, real_model, seed)
    root = os.path.abspath(out_dir or os.path.join(cfg.resolved_home(), "validation", f"staging_{stamp}"))
    os.makedirs(root, exist_ok=True)
    report: Dict[str, Any] = {"executed": False, "ok": False, "kind": "staging_validation", "started_utc": stamp,
                              "settings": {"n_decisions": n_decisions, "budget_s": budget_s, "lab_seeds": list(lab_seeds),
                                           "computer_share": computer_share, "lab_target": lab_target,
                                           "agent": cfg.section("agent"), "credit": cfg.section("credit"),
                                           "meta_min_independent_outcomes": cfg.optim.meta_min_independent_outcomes,
                                           "backbone": cfg.backbone.name, "scale": cfg.core.scale},
                              "git_head": _git_head(), "real_model": bool(real_model), "seed": seed,
                              "root": root, "requirements": list(REQUIREMENTS), "failures": []}
    lock = contextlib.nullcontext()
    if real_model:
        if not torch.cuda.is_available():
            raise RuntimeError("real_model=True needs a CUDA GPU (use real_model=False for the CPU plumbing check)")
        from .nn.gpu import gpu_lock
        lock = gpu_lock(min_free_gb=18.0)
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    with lock:
        t_lock = time.time()
        clock = OffsetClock()
        agent = CamLiveAgent(cfg, os.path.join(root, "agent"), clock=clock)
        try:
            report["build_seconds"] = time.time() - t_lock
            report["parameters"] = agent.model.count_parameters()
            report["parameters"].pop("experiential_breakdown", None)
            probe = HeldOutProbe(agent)
            p0 = probe.distribution()
            s0 = agent.model.trainable_state()
            n_a = max(1, int(round(n_decisions * computer_share)))
            n_b = max(1, (n_decisions - n_a) // max(1, len(lab_seeds)))
            left = budget_s - (time.time() - t_lock)
            try:
                report["phase_computer"] = phase_computer(agent, clock, root, n_a, left * computer_share)
            except Exception as e:  # noqa: BLE001 — recorded honestly; phase B still runs
                report["failures"].append({"phase": "computer", "error": f"{type(e).__name__}: {e}",
                                           "trace": traceback.format_exc(limit=12)})
            left = budget_s - (time.time() - t_lock)
            report["phase_lab"] = phase_lab(agent, clock, root, lab_seeds, n_b, max(0.0, left - 30.0),
                                            target=lab_target)
            p1 = probe.distribution()
            s1 = agent.model.trainable_state()
            # behaviour change on the held-out context, and what the weights alone explain
            weight_swap(agent.model, s0, ("experiential", "semantic_adapters", "heads", "reward"))
            p_restored = probe.distribution()
            weight_swap(agent.model, s1, ("experiential", "semantic_adapters", "heads", "reward"))
            weight_swap(agent.model, s0, ("experiential",))
            p_old_exp = probe.distribution()
            weight_swap(agent.model, s1, ("experiential",))
            p_back = probe.distribution()
            delta_norm = agent.model.group_delta_norms(s0)
            report["behaviour_change"] = {
                "context": "held-out LabWorld seed 9999 (never acted in); 12 fixed schema candidates",
                "candidates": [c.render()[:120] for c in probe.candidates],
                "p_before": [round(float(x), 6) for x in p0], "p_after": [round(float(x), 6) for x in p1],
                "tv_before_after": _tv(p0, p1),
                "tv_after_all_trainable_groups_restored_vs_before": _tv(p_restored, p0),
                "tv_only_experiential_restored_vs_after": _tv(p_old_exp, p1),
                "tv_only_experiential_restored_vs_before": _tv(p_old_exp, p0),
                "tv_reapplied_vs_after": _tv(p_back, p1),
                "weight_delta_norm_by_group": delta_norm,
                "note": "restoring every trainable group (frozen backbone untouched, store / fast state / registries "
                        "kept) must give back the old distribution exactly; restoring only the experiential group "
                        "shows the part of the change carried by the experiential weights",
            }
            dirs = [d for d in ([report.get("phase_computer", {}).get("runtime_dir", "")]
                                + [lt["runtime_dir"] for lt in report["phase_lab"]["lifetimes"]]) if d]
            report["evidence"] = collect_evidence(agent, dirs)
        finally:
            report["gpu_peak_gb"] = _gpu_peak()
            report["lock_seconds"] = time.time() - t_lock
            agent.close()
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report["wall_seconds"] = time.time() - t_start
    report["executed"] = True
    report["checks"] = checks(report)
    report["ok"] = report["passed"] = all(c["holds"] for c in report["checks"].values()) and not report["failures"]
    evidence_dir = evidence_dir or root               # the committed evidence is written only when asked for
    if evidence_dir:
        os.makedirs(evidence_dir, exist_ok=True)
        path = os.path.join(evidence_dir, f"staging_{stamp}{'' if real_model else '_cpu'}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True, default=str)
        report["evidence_file"] = path
    return report


def checks(r: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ev = r.get("evidence", {})
    bc = r.get("behaviour_change", {})
    wd = ev.get("weight_delta_by_timescale_group", {})
    pa = r.get("phase_computer", {})
    lab = r.get("phase_lab", {}).get("lifetimes", [])
    verified_lab = [lt for lt in lab if lt.get("verified", {}).get("signature")]

    def c(holds: bool, **detail: Any) -> Dict[str, Any]:
        return {"holds": bool(holds), **detail}
    return {
        "computer_environment_decisions": c(pa.get("counters", {}).get("decisions", 0) > 0,
                                            decisions=pa.get("counters", {}).get("decisions", 0)),
        "lab_lifetimes_verified": c(len(verified_lab) >= 2, n=len(verified_lab),
                                    closed=sum(1 for lt in verified_lab
                                               if lt["verified"].get("status") in ("achieved", "failed"))),
        "predictions_before_outcomes": c(ev.get("prediction_resolutions", {}).get("n_resolved", 0) > 0 and
                                         ev["prediction_resolutions"]["all_resolved_after_registration"],
                                         n_resolved=ev.get("prediction_resolutions", {}).get("n_resolved", 0)),
        "internal_rewards": c(ev.get("intrinsic_rewards", {}).get("n_records_with_reward", 0) > 0),
        "experiential_weights_changed": c(wd.get("frequent:experiential", {}).get("nonzero", 0) > 0,
                                          stats=wd.get("frequent:experiential")),
        "adapter_weights_changed": c(wd.get("slow_adapters:semantic_adapters", {}).get("nonzero", 0) > 0,
                                     stats=wd.get("slow_adapters:semantic_adapters")),
        "meta_update_applied": c(any(m.get("applied") and (m.get("meta_grad_norm") or 0.0) > 0
                                     and m.get("reward_version_before") != m.get("reward_version_after")
                                     for m in ev.get("meta_updates", [])),
                                 applied=ev.get("meta_applied", 0),
                                 applied_with_nonzero_gradient=sum(1 for m in ev.get("meta_updates", [])
                                                                   if m.get("applied") and
                                                                   (m.get("meta_grad_norm") or 0.0) > 0),
                                 skipped=ev.get("meta_skipped_reasons")),
        "delayed_outcomes_linked": c(ev.get("delayed_outcome_links", 0) > 0, n=ev.get("delayed_outcome_links", 0)),
        "behaviour_changed_on_held_out_context": c(bc.get("tv_before_after", 0.0) > 1e-6,
                                                   tv=bc.get("tv_before_after")),
        "weights_explain_the_change": c(bc.get("tv_after_all_trainable_groups_restored_vs_before", 1.0) < 1e-6,
                                        tv=bc.get("tv_after_all_trainable_groups_restored_vs_before")),
    }
