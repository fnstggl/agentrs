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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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
    """``runtime.verifier.MetricSource`` over the staging sites' own state (read directly by the verifier, never
    through the agent or the browser)."""

    live = False

    def __init__(self, site: Any, name: str, variants: Any = None):
        self.site, self.name, self.variants = site, name, variants

    def read(self) -> List[Quantity]:
        st = self.site.state()
        items = st.get("items", [])
        out = [Quantity("staging.items.count", float(len(items)), kind="count"),
               Quantity("staging.items.archived", float(sum(1 for i in items if i.get("status") == "archived")),
                        kind="count"),
               Quantity("staging.uploads.count", float(len(st.get("uploads", []))), kind="count")]
        if self.variants is not None:
            out.append(Quantity("staging.entries.count", float(len(self.variants.state().get("entries", []))),
                                kind="count"))
        return out


class StagingHttpMetrics:
    """``runtime.verifier.MetricSource`` for the verifier *service*: reads a staging site's own state over its local
    ``/api/state`` endpoint (the verifier process has no access to the site object; it never asks the agent or the
    browser). Metric names as :class:`SiteMetrics`."""

    live = False

    def __init__(self, base_url: str):
        self.base_url, self.name = base_url.rstrip("/"), "staging-http"

    def read(self) -> List[Quantity]:
        import urllib.request
        with urllib.request.urlopen(self.base_url + "/api/state", timeout=10) as r:     # noqa: S310 — localhost
            st = json.loads(r.read().decode())
        items = st.get("items", [])
        out = [Quantity("staging.items.count", float(len(items)), kind="count"),
               Quantity("staging.items.archived", float(sum(1 for i in items if i.get("status") == "archived")),
                        kind="count"),
               Quantity("staging.uploads.count", float(len(st.get("uploads", []))), kind="count")]
        if "entries" in st:
            out.append(Quantity("staging.entries.count", float(len(st["entries"])), kind="count"))
        return out


def staging_http_source(task: TaskSpec, run_dir: str) -> StagingHttpMetrics:
    """``success.evidence_source: cam_live.validation:staging_http_source`` — the site named by the task's
    ``resources.staging_site``."""
    url = str((task.resources or {}).get("staging_site") or "")
    if "://" not in url:
        raise ValueError("resources.staging_site must be the staging site's base URL")
    return StagingHttpMetrics(url)


#: objectives on the staging sites: (task key, site, path, metric, objective text). Plain outcome statements — what
#: should become true, never how; the agent receives the site's address only as an authorized resource.
STAGING_OBJECTIVES = {
    "record_item": ("structured", "/", "staging.items.count",
                    "Increase the number of items recorded by the records application to {target}."),
    "archive_item": ("structured", "/", "staging.items.archived",
                     "Increase the number of archived items of the records application to {target}."),
    "store_entry": ("variants", "/a/", "staging.entries.count",
                    "Increase the number of entries stored in the entry book to {target}."),
}


class StagingComputer:
    """The general-purpose computer environment (``harness.env.ComputerEnvironment`` with every executor of
    ``harness.factories.all_executors``) over the local staging sites, with the real authorization service and a
    signing ``MetricThresholdVerifier`` that reads the sites' own state. One environment (one browser, one skill
    library) persists across the objectives given to it, like the persistent computer of spec 12A.1.

    ``exclude``: executor modalities left out (``("skill",)`` = primitive-only operation)."""

    def __init__(self, root: str, clock: Any, *, exclude: Sequence[str] = (), browser: bool = True,
                 approvals: bool = False):
        from .harness.staging import StagingSites
        self.root, self.clock, self.exclude, self.browser = os.path.abspath(root), clock, tuple(exclude), browser
        self.approvals = approvals
        os.makedirs(self.root, exist_ok=True)
        self.sites = StagingSites().start()
        self.env: Any = None
        self.policy: Any = None
        self.metrics = SiteMetrics(self.sites.structured, "staging-sites", variants=self.sites.variants)
        self._n = 0

    def task(self, key: str, *, max_decisions: int, objective_id: str = "", target: Optional[int] = None,
             duration_s: float = 2 * 3600.0) -> TaskSpec:
        site, path, metric, text = STAGING_OBJECTIVES[key]
        now = {q.name: q.value for q in self.metrics.read()}[metric]
        target = int(now) + 1 if target is None else int(target)
        self._n += 1
        hosts = [u.replace("http://", "") for u in self.sites.urls.values()]
        url = self.sites.urls[site].rstrip("/") + path
        return TaskSpec(
            objective_id=objective_id or f"obj_{key}_{self._n:04d}", objective=text.format(target=target),
            success={"verifier": "metric_threshold", "metric": metric, "target": target},
            family=f"staging/{key.replace('_', '-')}",
            resources={"owned": hosts + [url], "preauthorized": hosts, "read": hosts},
            interfaces=["structured_browser", "api", "internal", "skill"],
            limits=Limits(duration_s=duration_s, money=1.0, compute_s=0.0, max_external_actions=10 * max_decisions,
                          max_decisions=int(max_decisions)),
            data_permissions={"aggregate_metrics": True}, mode="staging")

    def open(self, task: TaskSpec, start_url: str = "about:blank") -> Tuple[Any, Any, Any]:
        """(env, policy, verifier) for ``task``. The environment and its skill library are created once; the
        browser is put on ``start_url`` before the objective starts."""
        from .harness.auth import PolicyService
        from .harness.env import ComputerEnvironment
        from .harness.factories import all_executors
        from .runtime.verifier import MetricThresholdVerifier
        if self.policy is None:
            self.policy = PolicyService(os.path.join(self.root, "auth_state"), self._policy_file(), clock=self.clock)
        self.policy.register_task(task, "operator")
        if self.env is None:
            exec_dir = os.path.join(self.root, "exec")
            self.env = ComputerEnvironment(exec_dir, self.policy, clock=self.clock,
                                           secrets=self.policy.secrets_handle(), provenance="staging")
            for ex in all_executors(exec_dir, self.clock, env=self.env, policy=self.policy,
                                    secrets=self.policy.secrets_handle(), task=task, browser_enabled=self.browser,
                                    exclude=self.exclude):
                self.env.register(ex)
        self.reset_browser(task, start_url)
        verifier = MetricThresholdVerifier(self.metrics, self.clock, os.path.join(self.root, "verifier", task.objective_id),
                                           pinned_task=task)
        return self.env, self.policy, verifier

    def _policy_file(self) -> Optional[str]:
        """With ``approvals``: the default authorization policy with nothing pre-authorized on loopback in staging
        mode, so every interaction with the (owned) staging pages is queued for the operator's approval."""
        if not self.approvals:
            return None
        from .harness.auth.policy import run_policy_path
        src = open(run_policy_path()).read()
        old = '    preauthorized: ["127.0.0.1", "localhost", "[::1]", "channel:staging:*", "account:staging:*"]'
        if old not in src:
            raise RuntimeError("the default policy's staging pre-authorization line was not found")
        path = os.path.join(self.root, "policy_approvals.yaml")
        with open(path, "w") as f:
            f.write(src.replace(old, '    preauthorized: ["channel:staging:*", "account:staging:*"]', 1))
        return path

    def start_url(self, key: str) -> str:
        site, path, _m, _t = STAGING_OBJECTIVES[key]
        return self.sites.urls[site].rstrip("/") + path

    def reset_browser(self, task: TaskSpec, url: str = "about:blank") -> None:
        """Experimental control between objectives: the active tab is put on ``url`` (the operator's harness does
        this, not the agent; it is not one of the agent's decisions and the agent sees only the resulting page)."""
        from .contracts import ActionSpec, new_id
        tool = next((t for t in self.env.discover_tools() if t.name == "browser.navigate"), None)
        if tool is None:
            return
        a = ActionSpec(action_id=new_id("act"), operation=tool.name, modality=tool.modality,
                       arguments={"url": url}, auth_scope=tool.auth_scope)
        with contextlib.suppress(Exception):
            self.env.execute(a, self.policy.authorize(a, task, {"actor_id": "operator-harness"}))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self.env is not None:
                self.env.close()
        with contextlib.suppress(Exception):
            self.sites.stop()


def run_episode(agent: Any, sc: StagingComputer, key: str, clock: Any, root: str, *, max_decisions: int,
                start: str = "site", rt_cfg: Any = None, max_steps: int = 0) -> Dict[str, Any]:
    """One objective of the task family ``key`` driven by the real runtime until it ends (verified achieved,
    decision budget exhausted, or failed). Returns what was measured: verified success, decisions, primitive
    operations of the executed actions, skill use, failure classes, wall time."""
    from .runtime.loop import LiveRuntime, RuntimeConfig
    task = sc.task(key, max_decisions=max_decisions)
    env, policy, verifier = sc.open(task, sc.start_url(key) if start == "site" else "about:blank")
    agent.attach_verifier(verifier)
    rt = LiveRuntime(agent, env, policy, verifier, task, os.path.join(root, task.objective_id), clock,
                     rt_cfg or RuntimeConfig(fsync=False, candidates_k=10, keep_checkpoints=1, verify_every_s=0.0,
                                             checkpoint_every_s=1e12))
    t0 = time.time()
    rt.start()
    steps = 0
    while rt.status == "running" and steps < (max_steps or 3 * max_decisions + 10):
        rt.step()
        steps += 1
    if rt.status == "running":
        rt._finish("stopped", "episode step budget reached")
    rt.close()
    recs = [r for r in agent.store.trajectory(task.objective_id) if r.result is not None]
    ops = [r.result.cost.primitive_ops for r in recs]
    fails: Dict[str, int] = {}
    for r in recs:
        if r.result.status != "ok":
            fails[r.result.failure_class] = fails.get(r.result.failure_class, 0) + 1
    verified = rt._last_verified.get("outcome", {})
    return {"objective_id": task.objective_id, "key": key, "status": rt.status, "reason": rt.status_reason[:200],
            "achieved": verified.get("status") == "achieved", "verified_value": verified.get("value"),
            "target": task.success["target"], "decisions": len(recs), "primitive_ops": int(sum(ops)),
            "skill_runs": sum(1 for r in recs if r.decision.chosen.operation == "skill.run"),
            "skill_runs_ok": sum(1 for r in recs if r.decision.chosen.operation == "skill.run"
                                 and r.result.status == "ok"),
            "induce_attempts": sum(1 for r in recs if r.decision.chosen.operation == "skill.induce"),
            "operations": [r.decision.chosen.operation for r in recs],
            "provenance": [r.decision.chosen.provenance for r in recs], "failures": fails,
            "seconds": time.time() - t0, "steps": steps}


def lab_config(seed: int, **kw: Any) -> Any:
    """The LabWorld family used for delayed consequences: short horizons so several lifetimes fit a validation."""
    from .diagnostics.world import LabConfig
    base = dict(seed=seed, n_channels=4, target=12, budget=40.0, horizon_s=2 * DAY, outcome_horizon_s=6 * 3600.0,
                mean_delay_s=(1800.0, 4 * 3600.0), at_most_once=True, family="lab/staging")
    base.update(kw)
    return LabConfig(**base)


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


@contextlib.contextmanager
def shared_gpu(min_free_gb: float = 0.0) -> Iterator[None]:
    """The machine's shared GPU lock for one bounded job: a queued wait (``runtime.gpu_cycle.GpuLease``), then the
    free-memory check of ``nn.gpu.gpu_lock``; the caller bounds how long it holds it (``budget_s``)."""
    from .nn.gpu import GpuUnavailable
    from .runtime.gpu_cycle import GpuLease
    lease = GpuLease(hold_s=0.0)
    lease.acquire()
    try:
        if min_free_gb and torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
            if free < min_free_gb:
                raise GpuUnavailable(f"only {free:.1f} GB GPU memory free, need {min_free_gb}")
        yield
    finally:
        lease.release()


def _gpu_peak() -> Optional[float]:
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None


# --------------------------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------------------------
def staging_config(cfg: Optional[CamLiveConfig], real_model: bool, seed: int,
                   meta_min_independent: int = 4) -> CamLiveConfig:
    if real_model:
        cfg = cfg or CamLiveConfig.load()
        cfg.device = "cuda"
        agent = dict(proposer_max_new_tokens=160, proposer_k=4, policy_update_every=3, policy_update_batch=12,
                     planner_measure_every=4, experiment_rate=0.1)
        learner = {"min_independent_outcomes": 2, "warmup_steps": 2, "batch_size": 8, "slow_batch_size": 3,
                   "grad_last": 2, "adapter_every": 3}
    else:
        # INT3B (master item B): the debug / CPU validation never loads the pretrained backbone, whatever
        # configuration file was given (``cam-live validate --debug`` passes the operator's real configuration);
        # only the state root of that configuration is kept
        if cfg is None or cfg.backbone.name != "tiny" or cfg.resolved_device() != "cpu":
            home = cfg.home if cfg is not None else ""
            cfg = CamLiveConfig.debug()
            cfg.home = home
        cfg.backbone.tiny_vocab = 4096
        agent = dict(proposer_max_new_tokens=16, durable=False, policy_update_every=3, policy_update_batch=12,
                     planner_measure_every=4, experiment_rate=0.1)
        learner = {"min_independent_outcomes": 2, "warmup_steps": 0, "slow_batch_size": 3, "grad_last": 2,
                   "adapter_every": 3}
    cfg.seed = int(seed)
    cfg.optim.meta_min_independent_outcomes = int(meta_min_independent)
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
# phase A: the general-purpose computer environment on a staging site, with operator approvals and a resource limit
# --------------------------------------------------------------------------------------------------------------
def phase_computer(agent: Any, clock: OffsetClock, root: str, n_decisions: int, budget_s: float, *,
                   key: str = "store_entry", approvals: bool = True, external_limit: int = 0,
                   reject_every: int = 3, compute_limit_s: float = 0.0) -> Dict[str, Any]:
    """One objective of ``key`` in the computer environment (``StagingComputer``: every executor of the harness,
    Chromium, the real authorization service, the signing verifier reading the site's own state).

    ``approvals``: the site is owned but *not* pre-authorized, so every interaction with it is queued for the operator;
    the validation acts as the operator (``PolicyService.resolve_approval``): every ``reject_every``-th request is
    rejected, the others approved — the runtime executes an approved action at most once, after re-checking the
    limits. ``external_limit``: the task's ``max_external_actions`` (the runtime and the authorization service refuse
    beyond it). ``compute_limit_s``: the task's ``compute_s`` (learner time); when it is used up the runtime ends the
    objective as exhausted and checkpoints — a resource limit reached before the decision budget (INT3B)."""
    from .runtime.loop import LiveRuntime, RuntimeConfig
    from .runtime.logs import RunReader
    sc = StagingComputer(os.path.join(root, "computer"), clock, approvals=approvals)
    out: Dict[str, Any] = {"key": key}
    try:
        task = sc.task(key, max_decisions=n_decisions, objective_id="obj_staging_computer")
        if approvals:
            task.resources["preauthorized"] = []
        if external_limit:
            task.limits.max_external_actions = int(external_limit)
        if compute_limit_s:
            task.limits.compute_s = float(compute_limit_s)
        env, policy, verifier = sc.open(task, sc.start_url(key))
        agent.attach_verifier(verifier)
        rt = LiveRuntime(agent, env, policy, verifier, task, os.path.join(root, "computer", "runtime"), clock,
                         RuntimeConfig(fsync=True, candidates_k=10, keep_checkpoints=2, verify_every_s=0.0))
        rt.start()
        t0, steps, resolved = time.time(), 0, []
        while rt.status == "running" and steps < 4 * n_decisions and time.time() - t0 < budget_s:
            rt.step()
            steps += 1
            for row in policy.approvals():
                if row.get("status") != "pending":
                    continue
                approve = (len(resolved) + 1) % max(1, reject_every) != 0
                policy.resolve_approval(row["approval_id"], approve, "operator", "staging validation operator")
                resolved.append({"approval_id": row["approval_id"], "approved": approve,
                                 "operation": str((row.get("action") or {}).get("operation", "")
                                                  if isinstance(row.get("action"), dict) else row.get("preview", ""))[:80]})
        if rt.status == "running":
            rt._finish("stopped", "validation step budget reached")
        rt.close()
        r = RunReader(rt.run_dir)
        results = [e["data"] for e in r.records(RunFiles.EVENTS, types=["action_result"])]
        out.update(status=rt.status, reason=rt.status_reason[:200], steps=steps, seconds=time.time() - t0,
                   counters=dict(rt.counters), verified=rt._last_verified.get("outcome", {}), errors=rt._errors[-10:],
                   runtime_dir=rt.run_dir, tools=len(env.discover_tools()), approvals_resolved=resolved,
                   result_kinds=_count(f"{d.get('kind')}:{d.get('status')}" for d in results),
                   resource_limit_refusals=sum(1 for d in results if d.get("failure_class") == "resource_limit"),
                   grant_refusals=sum(1 for d in results if "grant does not verify" in str(d.get("failure_detail"))),
                   executed_after_approval=sum(1 for d in results if d.get("approval_id") and d.get("kind") == "execute"),
                   metrics=[q.to_dict() for q in sc.metrics.read()],
                   chosen=[(x["data"]["decision"]["chosen"]["operation"], x["data"]["decision"]["chosen"]["provenance"])
                           for x in r.records(RunFiles.DECISIONS)][:80],
                   journal=os.path.exists(os.path.join(rt.run_dir, "journal.md")),
                   limits={k: getattr(task.limits, k) for k in ("max_decisions", "max_external_actions", "compute_s",
                                                                "money")},
                   learning_skipped_for_compute=len(r.records(RunFiles.EVENTS, types=["learning_skipped"])))
    finally:
        sc.close()
    return out


# --------------------------------------------------------------------------------------------------------------
# phase B: a LabWorld family (delayed consequences, several verified lifetimes), one of them interrupted
# --------------------------------------------------------------------------------------------------------------
def phase_lab(agent: Any, clock: OffsetClock, root: str, seeds: Sequence[int], per_lifetime: int,
              budget_s: float, target: Any = 24, *, restart: Optional[Any] = None,
              lab_kw: Optional[Dict[str, Any]] = None, drain: bool = True) -> Dict[str, Any]:
    """Lifetimes of the LabWorld family, each a new objective for the same agent. ``target`` is one value or one
    per seed (different targets give lifetimes whose verified achievement differs). ``restart(agent) -> agent``:
    in the second lifetime the runtime and the agent are discarded half-way without a final commit and rebuilt
    from disk (spec 15.2: actual recovery); the rebuilt agent continues. Returns the report and the agent in use."""
    from .diagnostics.world import LabVerifier, LabWorld, lab_task
    from .harness.auth import PolicyService
    from .runtime.logs import RunReader
    from .runtime.loop import LiveRuntime, RuntimeConfig
    out: Dict[str, Any] = {"lifetimes": []}
    t0 = time.time()
    targets = list(target) if isinstance(target, (list, tuple)) else [target] * len(seeds)
    for i, seed in enumerate(seeds):
        if time.time() - t0 > budget_s:
            out["stopped"] = f"time budget of {budget_s:.0f} s reached before lifetime {seed}"
            break
        run = os.path.join(root, f"lab_{seed}")
        world = LabWorld(lab_config(seed, target=int(targets[i]), **(lab_kw or {})), clock)
        task = lab_task(world, f"obj_staging_lab_{seed}")
        task.limits.max_decisions = int(per_lifetime)
        verifier = LabVerifier(world)
        agent.attach_verifier(verifier)
        policy = PolicyService(os.path.join(run, "auth_state"), clock=clock)
        policy.register_task(task, "operator")
        # checkpoints only when forced (before the interruption, at the end): what happens in between is held by the
        # write-ahead journal alone, which the restart must replay
        # drain (S-1) + settle (INT3B): after the decision budget the runtime keeps measuring the lifetime's due
        # outcomes until the verifier's signed verdict is terminal (or the deadline), so every lifetime is an
        # independent verified group for the meta-learner — no decision is added
        rcfg = RuntimeConfig(fsync=True, candidates_k=10, keep_checkpoints=2, checkpoint_every_s=1e12,
                             drain_after_exhaustion=bool(drain))
        rt = LiveRuntime(agent, world, policy, verifier, task, os.path.join(run, "runtime"), clock, rcfg)
        rt.start()
        t1, steps, rec = time.time(), 0, None
        while rt.status == "running" and steps < 6 * per_lifetime and time.time() - t0 < budget_s:
            rt.step()
            steps += 1
            if restart is not None and i == 1 and rec is None and rt.counters["decisions"] >= max(2, per_lifetime // 2):
                before = {"decisions": rt.counters["decisions"], "committed": (rt._committed.get("ckpt") or {}).get("id"),
                          "model_version": agent.learner.model_version, "store_records": agent.store.count()}
                rt._commit(force=True)                 # the last commit before the interruption
                n_after = 0
                while rt.status == "running" and n_after < 2:  # work after the commit that only the journal holds
                    rt.step()
                    n_after += 1
                before["decisions_at_interruption"] = rt.counters["decisions"]
                del rt                                  # no close(), no final commit: the process is gone
                agent = restart(agent)
                agent.attach_verifier(verifier)
                rt = LiveRuntime(agent, world, policy, verifier, task, os.path.join(run, "runtime"), clock, rcfg)
                rt.start()                              # resume: checkpoint + journal replay
                rec = {**before, "resumed_status": rt.status, "decisions_after_resume": rt.counters["decisions"],
                       "model_version_after": agent.learner.model_version, "store_records_after": agent.store.count(),
                       "resumed_events": len(RunReader(rt.run_dir).records(RunFiles.EVENTS, types=["run_resumed"]))}
        if rt.status == "running":
            rt._finish("stopped", "validation step budget reached")
        rt.close()
        out["lifetimes"].append({"seed": seed, "target": int(targets[i]), "status": rt.status, "steps": steps,
                                 "seconds": time.time() - t1, "counters": dict(rt.counters),
                                 "verified": rt._last_verified.get("outcome", {}), "errors": rt._errors[-5:],
                                 "runtime_dir": rt.run_dir, "restart": rec})
    out["agent"] = agent
    return out


class MetaStepProbe:
    """Wraps ``MetaLearner.meta_update`` to record, for every applied outer step: the meta-gradient norm, the
    reward-parameter delta, and the policy change *caused by the meta-step* — the inner update of the latest policy
    update event recomputed from its snapshot theta_0 with psi before and with psi after the step, compared on that
    event's support decisions (mean total variation and largest probability difference)."""

    def __init__(self, agent: Any):
        self.agent, self.rows = agent, []
        self._orig = agent.meta.meta_update
        agent.meta.meta_update = self._wrapped

    def _wrapped(self, now: Optional[float] = None) -> Any:
        meta = self.agent.meta
        psi0 = {n: p.detach().clone() for n, p in meta.reward.psi().items()}
        rec = self._orig(now)
        if rec.applied:
            try:
                self.rows.append(self._measure(psi0, rec))
            except Exception as e:  # noqa: BLE001 — the measurement is evidence, never a reason to stop
                self.rows.append({"meta_id": rec.meta_id, "error": f"{type(e).__name__}: {e}"})
        return rec

    def _measure(self, psi0: Dict[str, torch.Tensor], rec: Any) -> Dict[str, Any]:
        meta = self.agent.meta
        psi1 = {n: p.detach().clone() for n, p in meta.reward.psi().items()}
        delta = math.sqrt(sum(float((psi1[n] - psi0[n]).double().pow(2).sum()) for n in psi0))
        ev = next((e for e in reversed(meta.events) if e.theta0), None)
        row = {"meta_id": rec.meta_id, "meta_grad_norm": rec.meta_grad_norm, "estimator": rec.estimator,
               "n_independent_outcomes": rec.n_independent_outcomes, "n_query": rec.n_query,
               "reward_version_before": rec.reward_version_before, "reward_version_after": rec.reward_version_after,
               "reward_param_delta_norm": delta, "outer_objective": rec.outer_objective}
        if ev is None:
            return row
        steps, _ = meta._steps_of([ev])
        dev = next(iter(psi1.values())).device
        theta0 = {n: v.to(dev) for n, v in ev.theta0.items()}
        lr = ev.inner_lr
        tb = meta.engine.inner_update(theta0, {n: v.to(dev) for n, v in psi0.items()}, steps, lr, len(steps),
                                      create_graph=False).detached()
        ta = meta.engine.inner_update(theta0, psi1, steps, lr, len(steps), create_graph=False).detached()
        with torch.no_grad():
            pb, pa = meta.inner.probs(tb, steps[0]).double(), meta.inner.probs(ta, steps[0]).double()
            p0 = meta.inner.probs(theta0, steps[0]).double()
            r_b = meta.inner.learned_reward({n: v.to(dev) for n, v in psi0.items()}, steps[0]).double()
            r_a = meta.inner.learned_reward(psi1, steps[0]).double()
        tv = 0.5 * (pa - pb).abs().sum(-1)
        tv_inner = 0.5 * (pa - p0).abs().sum(-1)
        row.update(policy_change_by_meta_step={"rows": int(pb.shape[0]), "mean_tv": float(tv.mean()),
                                               "max_tv": float(tv.max()),
                                               "max_abs_prob_diff": float((pa - pb).abs().max())},
                   internal_reward_on_support={"before": [round(float(x), 6) for x in r_b.tolist()],
                                               "after": [round(float(x), 6) for x in r_a.tolist()],
                                               "mean_abs_change": float((r_a - r_b).abs().mean()),
                                               "std_before": float(r_b.std()) if r_b.numel() > 1 else 0.0},
                   inner_update_policy_change={"mean_tv_theta0_to_thetaK": float(tv_inner.mean()),
                                               "max_tv": float(tv_inner.max())},
                   query_evidence={k: rec.diagnostics.get(k) for k in ("utility_mean", "n_distinct_utilities",
                                                                        "max_abs_advantage", "ess", "n_episodes",
                                                                        "policy_prob_change_on_query",
                                                                        "psi_tensors_with_gradient")},
                   event_id=ev.event_id)
        return row


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
                budget_s: float = 780.0, lab_seeds: Sequence[int] = (101, 102, 103, 104),
                computer_share: float = 0.25, lab_target: Any = (6, 30, 10, 40), computer_key: str = "store_entry",
                approvals: bool = True, external_limit: int = 0, restart: bool = True,
                lab_kw: Optional[Dict[str, Any]] = None, compute_limit_s: Optional[float] = None,
                lab_drain: bool = True, meta_min_independent: int = 4) -> Dict[str, Any]:
    """Run the staging validation (spec Phase 4) and write its evidence file.

    observation -> latent -> prediction -> action -> consequence -> internal reward -> neural update -> changed
    behaviour, through the real runtime, with operator approvals and a resource limit in the computer environment,
    an interruption + restart from disk in the second LabWorld lifetime, and a probe of every applied meta-update
    (meta-gradient norm, reward-parameter delta, policy change caused by the meta-step). LabWorld lifetimes get
    different targets, so their verified achievement differs (contrast for the outer objective). ``budget_s`` bounds
    the time the GPU lock is held for the learning phases (at most ~15 minutes in total)."""
    from .agent import CamLiveAgent
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    cfg = staging_config(cfg, real_model, seed, meta_min_independent)
    root = os.path.abspath(out_dir or os.path.join(cfg.resolved_home(), "validation", f"staging_{stamp}"))
    os.makedirs(root, exist_ok=True)
    lab_kw = dict(outcome_horizon_s=2 * 3600.0, mean_delay_s=(900.0, 3600.0)) if lab_kw is None else dict(lab_kw)
    n_a = max(1, int(round(n_decisions * computer_share)))
    n_b = max(2, (n_decisions - n_a) // max(1, len(lab_seeds)))
    if not external_limit:
        external_limit = max(2, n_a // 4)
    if compute_limit_s is None:
        # learner time of the computer phase: about half of what its decision budget takes on the real configuration
        # (measured 16 s per decision, staging_20261127T223510Z.json), so this limit, not the decision budget, ends it
        compute_limit_s = 8.0 * n_a
    report: Dict[str, Any] = {"executed": False, "ok": False, "kind": "staging_validation", "started_utc": stamp,
                              "settings": {"n_decisions": n_decisions, "budget_s": budget_s, "lab_seeds": list(lab_seeds),
                                           "computer_share": computer_share, "lab_target": lab_target,
                                           "computer_decisions": n_a, "lab_decisions_per_lifetime": n_b,
                                           "computer_key": computer_key, "approvals": approvals,
                                           "external_limit": external_limit, "restart": restart, "lab_kw": lab_kw,
                                           "computer_compute_limit_s": compute_limit_s, "lab_drain": lab_drain,
                                           "agent": cfg.section("agent"), "credit": cfg.section("credit"),
                                           "meta_min_independent_outcomes": cfg.optim.meta_min_independent_outcomes,
                                           "backbone": cfg.backbone.name, "scale": cfg.core.scale},
                              "git_head": _git_head(), "real_model": bool(real_model), "seed": seed,
                              "root": root, "requirements": list(REQUIREMENTS), "failures": []}
    lock = contextlib.nullcontext()
    if real_model and not torch.cuda.is_available():
        raise RuntimeError("real_model=True needs a CUDA GPU (use real_model=False for the CPU plumbing check)")
    if cfg.backbone.name != "tiny" and cfg.resolved_device() == "cuda":
        # any configuration that puts the pretrained backbone on the GPU holds the shared lock while it does; the
        # wait is the queued one of runtime.gpu_cycle (no timeout: a tenant that re-polls at once cannot starve it)
        lock = shared_gpu(min_free_gb=18.0)
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    probes: List[MetaStepProbe] = []
    with lock:
        t_lock = time.time()
        clock = OffsetClock()
        agent = CamLiveAgent(cfg, os.path.join(root, "agent"), clock=clock)
        probes.append(MetaStepProbe(agent))

        def rebuild(old: Any) -> Any:
            """The learner process is gone: a new agent is built from the configuration and restored from disk."""
            old.close()
            del old
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new = CamLiveAgent(cfg, os.path.join(root, "agent"), clock=clock)
            probes.append(MetaStepProbe(new))
            return new
        try:
            report["build_seconds"] = time.time() - t_lock
            report["parameters"] = agent.model.count_parameters()
            report["parameters"].pop("experiential_breakdown", None)
            probe = HeldOutProbe(agent)
            p0 = probe.distribution()
            s0 = agent.model.trainable_state()
            left = budget_s - (time.time() - t_lock)
            try:
                report["phase_computer"] = phase_computer(agent, clock, root, n_a, left * computer_share,
                                                          key=computer_key, approvals=approvals,
                                                          external_limit=external_limit,
                                                          compute_limit_s=float(compute_limit_s or 0.0))
            except Exception as e:  # noqa: BLE001 — recorded honestly; phase B still runs
                report["failures"].append({"phase": "computer", "error": f"{type(e).__name__}: {e}",
                                           "trace": traceback.format_exc(limit=12)})
            left = budget_s - (time.time() - t_lock)
            # consequences mature within a lifetime (an action takes about an hour of the world's time), so the
            # query rows after a policy update have their verified measurements before the lifetime ends
            lab = phase_lab(agent, clock, root, lab_seeds, n_b, max(0.0, left - 45.0), target=lab_target,
                            restart=rebuild if restart else None, lab_kw=lab_kw, drain=lab_drain)
            agent = lab.pop("agent")
            report["phase_lab"] = lab
            probe.agent = agent
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
            report["meta_step_effects"] = [r for p in probes for r in p.rows]
            report["reward_param_delta_norm_total"] = delta_norm.get("reward", 0.0)
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
    report["requirements_evidenced"] = sorted({rid for name, c in report["checks"].items() if c["holds"]
                                               for rid in CHECK_REQUIREMENTS.get(name, ())})
    evidence_dir = evidence_dir or root               # the committed evidence is written only when asked for
    if evidence_dir:
        os.makedirs(evidence_dir, exist_ok=True)
        path = os.path.join(evidence_dir, f"staging_{stamp}{'' if real_model else '_cpu'}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True, default=str)
        report["evidence_file"] = path
    return report


#: which requirement ids of the completion manifest each check evidences
CHECK_REQUIREMENTS = {
    "computer_environment_decisions": ("IN-01", "CU-22", "CU-30"),
    "no_grant_refusals": ("IN-01", "LX-11"),
    "approvals_exercised": ("LX-11", "IN-07"),
    "resource_limit_exercised": ("LX-11", "IN-07"),
    "recovery_exercised": ("LX-11", "IN-07", "CL-06"),
    "logging": ("LX-11",),
    "lab_lifetimes_verified": ("IN-07", "MR-05"),
    "predictions_before_outcomes": ("PI-02",),
    "internal_rewards": ("MR-01",),
    "experiential_weights_changed": ("CL-02", "CU-22"),
    "adapter_weights_changed": ("CL-03",),
    "meta_update_applied": ("MR-03", "MR-05", "MR-06"),
    "meta_step_changes_policy": ("MR-03", "MR-06"),
    "delayed_outcomes_linked": ("RC-01",),
    "behaviour_changed_on_held_out_context": ("NA-06", "IN-07"),
    "weights_explain_the_change": ("NA-06", "CL-09"),
}


def checks(r: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ev = r.get("evidence", {})
    bc = r.get("behaviour_change", {})
    wd = ev.get("weight_delta_by_timescale_group", {})
    pa = r.get("phase_computer", {})
    lab = r.get("phase_lab", {}).get("lifetimes", [])
    verified_lab = [lt for lt in lab if lt.get("verified", {}).get("signature")]

    def c(holds: bool, **detail: Any) -> Dict[str, Any]:
        return {"holds": bool(holds), **detail}
    restarts = [lt["restart"] for lt in lab if lt.get("restart")]
    mse = r.get("meta_step_effects", [])
    return {
        "computer_environment_decisions": c(pa.get("counters", {}).get("decisions", 0) > 0,
                                            decisions=pa.get("counters", {}).get("decisions", 0),
                                            verified=pa.get("verified", {}).get("status")),
        "no_grant_refusals": c(pa.get("grant_refusals", 1) == 0 and pa.get("counters", {}).get("actions_executed", 0) > 0,
                               refusals=pa.get("grant_refusals"), executed=pa.get("counters", {}).get("actions_executed")),
        "approvals_exercised": c(any(a["approved"] for a in pa.get("approvals_resolved", []))
                                 and any(not a["approved"] for a in pa.get("approvals_resolved", []))
                                 and pa.get("executed_after_approval", 0) > 0,
                                 resolved=len(pa.get("approvals_resolved", [])),
                                 executed_after_approval=pa.get("executed_after_approval", 0)),
        # a configured resource limit other than the decision budget was reached and enforced: an action refused for
        # it, or the objective ended by it (the runtime checkpoints and stops deciding)
        "resource_limit_exercised": c(pa.get("resource_limit_refusals", 0) > 0
                                      or any(f"limit {k} exhausted" in pa.get("reason", "")
                                             for k in ("max_external_actions", "compute_s", "money")),
                                      refusals=pa.get("resource_limit_refusals", 0), reason=pa.get("reason", ""),
                                      limits=pa.get("limits")),
        "recovery_exercised": c(bool(restarts) and all(x.get("resumed_status") == "running"
                                                       and x.get("resumed_events", 0) >= 1
                                                       and x.get("decisions_after_resume", 0)
                                                       >= x.get("decisions_at_interruption", 1) for x in restarts),
                                restarts=restarts),
        "logging": c(bool(pa.get("journal")) and ev.get("n_decisions", 0) > 0, journal=pa.get("journal"),
                     decisions_logged=ev.get("n_decisions", 0)),
        "meta_step_changes_policy": c(any((m.get("policy_change_by_meta_step") or {}).get("max_abs_prob_diff", 0.0) > 0
                                          and m.get("reward_param_delta_norm", 0.0) > 0 for m in mse),
                                      rows=len(mse)),
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
