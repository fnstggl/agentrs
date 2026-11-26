"""Distributed CAM Unified runs on the real Crucible OpenWorld, and the evidence extracted from them (spec 14, 18, 22;
manifest A16 / F02 / F03 / F04 / X06 / A20).

`cam_run` launches one central learner (`cam_learner.CamLearnerModel`, GPU) and N actor processes (each: its own
`cam_policy.CamActorPolicy` on the GPU and its own process-isolated ow_tiny world, `cam_env`) through the
`launcher.Launcher`, runs them for a wall-clock budget, stops them cleanly (actors close their episodes, the learner
drains the queue) and writes ``<run>/evidence.json`` (`run_evidence`) — computed only from the run's own durable
files:

* **shared learning** — per actor: decisions generated (actor status), decisions archived, decisions trained by the
  learner according to the durable consumed index (``learner/consumed.sqlite`` joined with the archive index on
  event id → actor), the updates whose batches contained the actor's decisions (``learner/updates.jsonl``
  ``by_actor``), and the policy objective's per-actor counts (model metrics);
* **version propagation** — per actor: the distinct ``model_version`` values of its decision records over time and
  the versions it adopted (``versions_used``), refreshes and latent-state rebuilds;
* **policy lag / off-policy correction** — the lag histogram of every routed decision and of the trained ones, the
  number of decisions D's objective trained with V-trace (stale rows) and with PPO (fresh rows), the V-trace ρ
  statistics, and the exactness check of the lag-0 recomputation (|log π_recomputed − log μ_recorded|);
* **throughput and compute** — decisions/s (steady window), trained decisions/s, updates, per-phase actor and learner
  wall time (bottleneck analysis), GPU / CPU seconds per process, semantic-model tokens (encoder prefill, decoder
  samples), environment steps, simulated days, peak RSS and GPU memory.

`behaviour_change` re-evaluates one actor's recorded decisions (stored exact inputs) under two published versions
and reports how the log-probability of the very actions it took changed — the measurement behind "an actor's
experience produced a central update that changed another actor's later behaviour" when the learner trained only on
the other actor (``train_actors``). The same measurement on a run whose learner had lr = 0 is the control.

CLI::

    python -m superlearner.cam_unified.dist.cam_runs run --actors 4 --seconds 600 --out <dir> [--label L]
    python -m superlearner.cam_unified.dist.cam_runs cross --seconds 600 --out <dir> [--lr 0]
    python -m superlearner.cam_unified.dist.cam_runs evidence <run_dir>
"""
from __future__ import annotations

import collections
import json
import os
import sqlite3
import time
from typing import Optional

from ..observe.accounting import atomic_write_json, merge_snapshots, read_snapshots
from .actor import ActorSpec, run_paths
from .launcher import Launcher, steady_rate
from .learner import LearnerSpec

CAM_LEARNER = "superlearner.cam_unified.dist.cam_learner:make_cam_learner"
CAM_POLICY = "superlearner.cam_unified.dist.cam_policy:make_cam_policy"
OW_ENV = "superlearner.cam_unified.dist.cam_env:make_openworld_env"


def cam_specs(run: str, n_actors: int, cfg: Optional[dict] = None, seed: int = 0, lr: Optional[float] = None,
              train_actors: Optional[list] = None, max_lag: Optional[int] = None,
              lag_validation: Optional[str] = None, min_decisions: int = 32, segment_len: int = 32, burn_in: int = 8,
              settle: int = 2, keep_versions: int = 5, refresh_at: str = "decision", rebuild_threshold: float = 0.05,
              days: int = 8, learner_kwargs: Optional[dict] = None, actor_seed0: int = 1000,
              batch_records: int = 8, max_seconds: Optional[float] = None) -> tuple:
    """(LearnerSpec, [ActorSpec]) of a distributed CAM run (see module docstring)."""
    mk = {"run_dir": run, "cfg": dict(cfg or {}), "seed": seed}
    if lr is not None:
        mk["lr"] = float(lr)
        mk["lr_sem"] = float(lr)
    mk.update(learner_kwargs or {})
    L = LearnerSpec(run_dir=run, model_factory=CAM_LEARNER, model_kwargs=mk, max_lag=max_lag, lag_mode="vtrace",
                    min_decisions=min_decisions, max_batch_decisions=min_decisions, publish_every=1,
                    train_actors=train_actors, segment_len=segment_len, burn_in=burn_in, settle=settle,
                    stale_s=900.0, torch_threads=2, cuda=True, lag_validation=lag_validation,
                    keep_versions=keep_versions)
    A = [ActorSpec(actor_id=f"actor{i}", run_dir=run, env_factory=OW_ENV, env_kwargs={"days": days},
                   policy_factory=CAM_POLICY, policy_kwargs={"cfg": dict(cfg or {})}, seed=actor_seed0 + i,
                   refresh_at=refresh_at, rebuild_threshold=rebuild_threshold, batch_records=batch_records,
                   cuda=True, torch_threads=1, wait_version_s=1800.0, drift_mode="meta_sum", status_every_s=2.0,
                   max_seconds=max_seconds)
         for i in range(n_actors)]
    return L, A


def cam_run(out_dir: str, n_actors: int, seconds: float, label: Optional[str] = None, stop_timeout_s: float = 600.0,
            **kw) -> dict:
    """Launch, run for ``seconds`` of wall time after every actor is running, stop, write evidence."""
    run = os.path.abspath(os.path.join(out_dir, label or f"cam_actors{n_actors}"))
    if os.path.exists(run):
        raise FileExistsError(f"{run} exists (runs are never overwritten)")
    L, A = cam_specs(run, n_actors, **kw)
    env = {"OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    launch = Launcher(run, L, A, max_restarts=1, sample_every_s=2.0, env=env)
    t0 = time.time()
    launch.start()
    started = None
    try:
        while True:
            launch.poll()
            st = launch._read_status()
            if started is None and all((st.get(a.actor_id) or {}).get("state") == "running" for a in A):
                started = time.time()
                launch._event("all_actors_running", "launcher", after_s=started - t0)
            if started is not None and time.time() - started >= seconds:
                break
            if launch.actors_done():
                break
            if time.time() - t0 > seconds + 1800:          # never hang: start-up took more than 30 minutes
                launch._event("startup_timeout", "launcher")
                break
            time.sleep(1.0)
    finally:
        summary = launch.stop(timeout_s=stop_timeout_s)
    ev = run_evidence(run, summary)
    ev["spec"] = {"n_actors": n_actors, "seconds": seconds, "kw": {k: v for k, v in kw.items()},
                  "startup_s": None if started is None else started - t0}
    atomic_write_json(os.path.join(run, "evidence.json"), ev)
    return ev


# ============================================================================ evidence

def _jsonl(path: str) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _read(path: str, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def _hist(xs) -> dict:
    c = collections.Counter(xs)
    return {str(k): v for k, v in sorted(c.items())}


def run_evidence(run: str, summary: Optional[dict] = None) -> dict:
    """Evidence of one run from its files (module docstring)."""
    p = run_paths(run)
    status = {}
    for fn in sorted(os.listdir(p["status"])) if os.path.isdir(p["status"]) else []:
        if fn.endswith(".json"):
            status[fn[:-5]] = _read(os.path.join(p["status"], fn), {})
    actors = {k: v for k, v in status.items() if "actor_id" in v}
    learner = status.get("learner", {})
    updates = [u for u in _jsonl(os.path.join(p["learner"], "updates.jsonl")) if "update_id" in u]
    trained_ups = [u for u in updates if u.get("trained")]
    # ---- durable consumption by actor (consumed index x archive index)
    by_status_actor = collections.Counter()
    versions_by_actor = collections.defaultdict(list)
    arch_db = os.path.join(p["archive"], "index.sqlite")
    cons_db = os.path.join(p["learner"], "consumed.sqlite")
    archived_by_actor = collections.Counter()
    if os.path.exists(arch_db):
        a = sqlite3.connect(arch_db)
        for actor, n in a.execute("SELECT actor_id, COUNT(*) FROM ev GROUP BY actor_id"):
            archived_by_actor[actor] = n
        if os.path.exists(cons_db):
            a.execute(f"ATTACH DATABASE '{cons_db}' AS c")
            for actor, stt, n in a.execute("SELECT ev.actor_id, c.consumed.status, COUNT(*) FROM ev JOIN c.consumed "
                                           "ON ev.event_id = c.consumed.event_id GROUP BY ev.actor_id, "
                                           "c.consumed.status"):
                by_status_actor[(actor, stt)] = n
        for actor, step, mv, t in a.execute("SELECT actor_id, step, model_version, t_sim FROM ev WHERE step >= 0 "
                                            "ORDER BY actor_id, rowid"):
            versions_by_actor[actor].append(int(mv))
        a.close()
    per_actor = {}
    for aid in sorted(set(actors) | set(archived_by_actor)):
        st = actors.get(aid, {})
        ups_with = [u for u in trained_ups if (u.get("by_actor") or {}).get(aid)]
        vs = versions_by_actor.get(aid, [])
        per_actor[aid] = {
            "decisions": st.get("decisions"), "records_written": st.get("records"),
            "records_archived": archived_by_actor.get(aid, 0),
            "trained_decisions_durable": by_status_actor.get((aid, "trained"), 0),
            "skipped_decisions_durable": by_status_actor.get((aid, "skipped"), 0),
            "aux_records_consumed": by_status_actor.get((aid, "aux"), 0),
            "trained_decisions_update_log": sum((u.get("by_actor") or {}).get(aid, 0) for u in trained_ups),
            "policy_objective_decisions": sum(((u.get("metrics") or {}).get("policy_trained_by_actor") or {})
                                              .get(aid, 0) for u in trained_ups),
            "updates_with_actor": len(ups_with), "first_update_with_actor": ups_with[0]["version"] if ups_with
            else None,
            "record_versions_hist": _hist(vs), "max_version_acted": max(vs) if vs else None,
            "versions_used": st.get("versions_used"), "refreshes": st.get("refreshes"), "rebuilds": st.get("rebuilds"),
            "episodes": st.get("episodes"), "env_failures": st.get("env_failures"), "incarnation": st.get("incarnation"),
            "phase_s": st.get("phase_s"), "semantic": st.get("semantic"),
            "writer_blocked_s": (st.get("writer") or {}).get("blocked_s"), "wait_weights_s": st.get("wait_weights_s"),
            "recent_score": st.get("recent_score"), "episode_log": st.get("episode_log")}
    # ---- lag and off-policy correction
    trained_lags = collections.Counter()
    for u in trained_ups:
        h = (u.get("lag") or {}).get("lag_hist") or {}
        for k, v in h.items():
            trained_lags[int(k)] += v
    mets = [u.get("metrics") or {} for u in trained_ups]

    def tot(key):
        return sum(int(m.get(key) or 0) for m in mets)

    lag0 = [m.get("lag0_logp_dev_max") for m in mets if m.get("lag0_logp_dev_max") is not None]
    within = [m.get("lag0_logp_within_1e-3") for m in mets if m.get("lag0_logp_within_1e-3") is not None]
    vt = [m.get("vtrace") or {} for m in mets]
    vt_n = sum(int(v.get("n", 0) or 0) for v in vt)
    rho_mean = (sum(float(v.get("rho_mean", 0)) * int(v.get("n", 0)) for v in vt if v.get("n")) / vt_n) if vt_n else None
    ess = (sum(float(v.get("ess_frac", 0)) * int(v.get("n", 0)) for v in vt if v.get("n")) / vt_n) if vt_n else None
    clipped = (sum(float(v.get("clipped_frac", 0)) * int(v.get("n", 0)) for v in vt if v.get("n")) / vt_n) \
        if vt_n else None
    router = learner.get("router") or {}
    lag = {"max_lag": router.get("max_lag"), "max_lag_source": router.get("max_lag_source"),
           "routed_lag_hist": router.get("lag_hist"), "router_counts": router.get("counts"),
           "trained_lag_hist": {str(k): v for k, v in sorted(trained_lags.items())},
           "policy_ppo_fresh_decisions": tot("n_fresh_steps"), "policy_vtrace_stale_decisions": tot("n_stale_steps"),
           "rerouted_by_kl_rows": tot("rerouted_by_kl"), "vtrace_rho_mean": rho_mean, "vtrace_ess_frac": ess,
           "vtrace_clipped_frac": clipped,
           "lag0_logp_dev_max_per_update": {"max": max(lag0) if lag0 else None,
                                            "median": sorted(lag0)[len(lag0) // 2] if lag0 else None},
           "lag0_logp_within_1e-3_mean": (sum(within) / len(within)) if within else None,
           "z_recompute_dev_max": max([m.get("z_recompute_dev_max") or 0.0 for m in mets] or [0.0]),
           "recomputed_pieces": tot("n_recomputed_pieces"), "pieces": tot("n_pieces")}
    # ---- throughput, compute, bottlenecks
    series = _jsonl(os.path.join(run, "series.jsonl"))
    snaps = read_snapshots(p["compute"])
    comp = merge_snapshots(snaps) if snaps else {}
    phases_l = collections.Counter()
    for m in mets:
        for k, v in (m.get("phases") or {}).items():
            phases_l[k] += float(v)
    phases_a = collections.Counter()
    for a in per_actor.values():
        for k, v in (a.get("phase_s") or {}).items():
            phases_a[k] += float(v)
    sem = collections.Counter()
    for a in per_actor.values():
        for k, v in (a.get("semantic") or {}).items():
            sem[k] += int(v)
    wall = (series[-1]["t"] - series[0]["t"]) if len(series) >= 2 else None
    n_dec = sum(int(a.get("decisions") or 0) for a in per_actor.values())
    thr = {"wall_s_sampled": wall, "decisions": n_dec,
           "decisions_per_s_steady": steady_rate(series, "decisions"),
           "trained_decisions_per_s_steady": steady_rate(series, "trained_decisions"),
           "ingested_records_per_s_steady": steady_rate(series, "ingested"),
           "updates": len(trained_ups), "learner_version": learner.get("version"),
           "update_wall_s_mean": (phases_l.get("total_s", 0.0) / len(trained_ups)) if trained_ups else None,
           "decisions_per_update_mean": (sum(int(u.get("n_decisions") or 0) for u in trained_ups) /
                                         len(trained_ups)) if trained_ups else None,
           "learner_phase_s": dict(phases_l), "actor_phase_s_sum": dict(phases_a),
           "actor_s_per_decision": {k: v / max(1, n_dec) for k, v in phases_a.items()},
           "learner_s_per_trained_decision": {k: v / max(1, sum(int(u.get("n_decisions") or 0) for u in trained_ups))
                                              for k, v in phases_l.items()},
           "pending_segments_at_end": (learner.get("queue") or {}).get("pending"),
           "learner_counts": learner.get("counts"),
           "untrained_backlog_decisions": max(0, n_dec - sum(a["trained_decisions_durable"]
                                                             for a in per_actor.values()))}
    compute = {k: comp.get(k) for k in ("wall_s", "cpu_s", "gpu_s", "env_steps", "decisions", "sim_days",
                                        "tokens_prompt", "tokens_generated", "records", "updates", "samples_trained",
                                        "peak_rss_mb_max", "peak_rss_mb_sum", "peak_gpu_mb_max", "peak_gpu_mb_sum",
                                        "cost_usd", "n_processes", "by_role")}
    compute["semantic_counters"] = dict(sem)
    compute["processes"] = [{k: s.get(k) for k in ("name", "role", "pid", "wall_s", "cpu_s", "gpu_s", "decisions",
                                                   "records", "updates", "samples_trained", "tokens_prompt",
                                                   "tokens_generated", "sim_days", "peak_rss_mb", "peak_gpu_mb")}
                            for s in snaps]
    upd_rows = [{"version": u["version"], "t_rel": u.get("t_rel"), "n_decisions": u.get("n_decisions"),
                 "by_actor": u.get("by_actor"), "lag_hist": (u.get("lag") or {}).get("lag_hist"),
                 "fresh": (u.get("metrics") or {}).get("n_fresh_steps"),
                 "stale": (u.get("metrics") or {}).get("n_stale_steps"),
                 "loss": (u.get("metrics") or {}).get("loss"), "losses": (u.get("metrics") or {}).get("losses"),
                 "lag0_dev_max": (u.get("metrics") or {}).get("lag0_logp_dev_max"),
                 "mean_reward": (u.get("metrics") or {}).get("mean_reward"),
                 "published": u.get("published")} for u in trained_ups]
    ev = {"run_dir": run, "t": time.time(), "per_actor": per_actor, "lag": lag, "throughput": thr,
          "compute": compute, "updates": upd_rows,
          "checkpoint": _read(os.path.join(p["ckpt"], "LATEST")),
          "launcher_events": [e for e in _jsonl(os.path.join(run, "launcher.jsonl"))
                              if e.get("event") not in ("start",)][:50]}
    if summary is not None:
        ev["launcher_summary"] = {k: summary.get(k) for k in ("wall_s", "n_actors", "restarts", "decisions",
                                                             "decisions_per_s_steady",
                                                             "trained_decisions_per_s_steady", "learner_version")}
    ev["checks"] = shared_learning_checks(ev)
    return ev


def shared_learning_checks(ev: dict) -> dict:
    """Boolean summary of the A16 conditions on one run's evidence."""
    pa = ev["per_actor"]
    lag = ev["lag"]
    return {"every_actor_trained": bool(pa) and all(a["trained_decisions_durable"] > 0 for a in pa.values()),
            "every_actor_acted_with_updated_weights": bool(pa) and all((a["max_version_acted"] or 0) > 0
                                                                      for a in pa.values()),
            "updates_mix_actors": sum(1 for u in ev["updates"] if len(u.get("by_actor") or {}) > 1),
            "vtrace_corrections_applied": (lag["policy_vtrace_stale_decisions"] or 0) > 0,
            "lag_gt0_trained": sum(v for k, v in (lag["trained_lag_hist"] or {}).items() if int(k) > 0)}


# ============================================================================ behaviour change probe

def behaviour_change(run: str, actor_id: str, v_from: int, v_to: Optional[int] = None, max_decisions: int = 64,
                     behaviour_version: Optional[int] = None, device: Optional[str] = None) -> dict:
    """Re-evaluate ``actor_id``'s recorded decisions (those taken under ``behaviour_version``, default ``v_from``)
    under the published weights of ``v_from`` and ``v_to`` (default LATEST): per decision, log π_v(a|s) of the action
    the actor actually took, from the stored exact inputs (one-step, per-row evaluation). Returns the deviations
    |log π_to − log π_from| and |log π_from − log μ| (recomputation exactness)."""
    import torch
    from .cam_learner import CamLearnerModel
    from .publish import load_weights, read_latest
    from .sinks import JsonlArchive
    p = run_paths(run)
    rj = _read(os.path.join(run, "run.json"), {})
    mk = dict((rj.get("learner") or {}).get("model_kwargs") or {})
    lt = read_latest(p["ckpt"])
    v_to = int(lt["version"]) if v_to is None else int(v_to)
    bv = v_from if behaviour_version is None else behaviour_version
    arch = JsonlArchive(p["archive"])
    recs = []
    for chunk in arch.scan(actor_id=actor_id):
        for r in chunk:
            if r.action is not None and int(r.model_version) == bv and (r.extra or {}).get("inputs_ref"):
                recs.append(r)
    arch.close()
    recs = sorted(recs, key=lambda r: (r.episode_id, r.step))[:max_decisions]
    model = CamLearnerModel(run_dir=run, cfg=mk.get("cfg"), seed=int(mk.get("seed", 0)), device=device,
                            world_objectives=False)
    m = model.model
    out = {}
    for v in (v_from, v_to):
        path = os.path.join(p["ckpt"], f"v{v:08d}", "weights.pt")
        model.load_state_dict(load_weights(path)["state_dict"])
        m.version = v
        m.eval()
        lps = []
        for r in recs:
            d = model.store.inputs(r.extra["inputs_ref"])
            tools = model.store.tools(r.extra["tools_ref"])
            m.tokenizer.register_tools(tools)
            if d.get("last_tool"):
                from ..interfaces import ToolSchema
                m.tokenizer.register_tools([ToolSchema.from_dict(d["last_tool"])])
            with torch.no_grad():
                o = m.policy_evaluate(model._inputs(d, r, tools, None), [r.action])
            lps.append(float(o["logp"][0]))
        out[v] = lps
    beh = [float(r.behavior_logp) for r in recs]
    dlt = [abs(a - b) for a, b in zip(out[v_to], out[v_from])]
    exact = [abs(a - b) for a, b in zip(out[v_from], beh)]
    return {"run": run, "actor_id": actor_id, "v_from": v_from, "v_to": v_to, "behaviour_version": bv,
            "n_decisions": len(recs), "mean_abs_delta_logp": (sum(dlt) / len(dlt)) if dlt else None,
            "max_abs_delta_logp": max(dlt) if dlt else None,
            "frac_changed_gt_1e-3": (sum(1 for x in dlt if x > 1e-3) / len(dlt)) if dlt else None,
            "mean_logp_from": (sum(out[v_from]) / len(recs)) if recs else None,
            "mean_logp_to": (sum(out[v_to]) / len(recs)) if recs else None,
            "recompute_vs_behaviour_max": max(exact) if exact else None,
            "recompute_vs_behaviour_median": sorted(exact)[len(exact) // 2] if exact else None,
            "tools": [r.action.tool for r in recs], "logp_from": out[v_from], "logp_to": out[v_to],
            "behaviour_logp": beh}


def _main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="cam_unified.dist.cam_runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one distributed CAM run with N actors")
    r.add_argument("--actors", type=int, required=True)
    r.add_argument("--seconds", type=float, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--label")
    r.add_argument("--max-lag", type=int)
    r.add_argument("--lag-validation")
    r.add_argument("--keep-versions", type=int, default=5)
    r.add_argument("--min-decisions", type=int, default=32)
    r.add_argument("--lr", type=float)
    r.add_argument("--train-actors")
    r.add_argument("--cfg", default="{}", help="JSON dict of cam_config overrides")
    r.add_argument("--learner-kwargs", default="{}")
    e = sub.add_parser("evidence", help="(re)compute evidence.json of a finished run")
    e.add_argument("run_dir")
    b = sub.add_parser("behaviour", help="behaviour change of one actor between two versions")
    b.add_argument("run_dir")
    b.add_argument("--actor", required=True)
    b.add_argument("--v-from", type=int, default=0)
    b.add_argument("--v-to", type=int)
    b.add_argument("--n", type=int, default=48)
    b.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        ta = a.train_actors.split(",") if a.train_actors else None
        ev = cam_run(a.out, a.actors, a.seconds, label=a.label, max_lag=a.max_lag, lag_validation=a.lag_validation,
                     keep_versions=a.keep_versions, min_decisions=a.min_decisions, segment_len=a.min_decisions,
                     lr=a.lr, train_actors=ta, cfg=json.loads(a.cfg), learner_kwargs=json.loads(a.learner_kwargs))
        print(json.dumps({"checks": ev["checks"], "throughput": {k: ev["throughput"][k] for k in (
            "decisions", "decisions_per_s_steady", "trained_decisions_per_s_steady", "updates")},
                          "lag": {k: ev["lag"][k] for k in ("trained_lag_hist", "policy_vtrace_stale_decisions",
                                                            "policy_ppo_fresh_decisions")}}, indent=1, default=str))
    elif a.cmd == "evidence":
        ev = run_evidence(a.run_dir)
        old = _read(os.path.join(a.run_dir, "evidence.json"), {}) or {}
        if "spec" in old:
            ev["spec"] = old["spec"]
        atomic_write_json(os.path.join(a.run_dir, "evidence.json"), ev)
        print(json.dumps(ev["checks"], indent=1))
    elif a.cmd == "behaviour":
        res = behaviour_change(a.run_dir, a.actor, a.v_from, a.v_to, a.n)
        if a.out:
            atomic_write_json(a.out, res)
        print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))


if __name__ == "__main__":
    _main()
