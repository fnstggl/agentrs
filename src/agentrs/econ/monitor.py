"""Watchdog + live dashboard for the overnight run (``python -m agentrs.econ.monitor``).

Every ``interval`` seconds it merges the trainer heartbeat, the live observer and GitHub observer
status, GPU (nvidia-smi), container memory (cgroup, not host totals) and disk into
``<run>/latest_status.json`` and ``<run>/latest_status.md``; snapshots go to ``status_history.jsonl``.
It enforces the safety rails that must not depend on the trainer being healthy:
  * stale trainer heartbeat while supervisor says RUNNING -> restart the trainer (bounded)
  * disk nearly full or spending cap reached -> create STOP (trainer then finishes cleanly)
  * hard deadline (+grace) passed and trainer still running -> stop it via supervisor
  * once the trainer is done/stopped after the deadline -> stop the observers and exit
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def sh(cmd: list[str], timeout: int = 30) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def gpu() -> dict:
    out = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
              "--format=csv,noheader,nounits"])
    try:
        u, mu, mt, temp, pw = [x.strip() for x in out.strip().split("\n")[0].split(",")]
        return {"util_pct": float(u), "vram_used_gb": float(mu) / 1024, "vram_total_gb": float(mt) / 1024,
                "temp_c": float(temp), "power_w": float(pw)}
    except ValueError:
        return {}


def cgroup_mem() -> dict:
    for cur, mx in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                    ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            c = int(Path(cur).read_text())
            m = Path(mx).read_text().strip()
            m = int(m) if m != "max" else None
            return {"used_gb": c / 2**30, "limit_gb": m / 2**30 if m else None}
        except (OSError, ValueError):
            continue
    return {}


def supervisor_state(name: str) -> str:
    out = sh(["supervisorctl", "status", name])
    parts = out.split()
    return parts[1] if len(parts) > 1 else "UNKNOWN"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/overnight")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--stale-min", type=float, default=35.0)
    ap.add_argument("--grace-min", type=float, default=45.0)
    ap.add_argument("--max-trainer-restarts", type=int, default=3)
    ap.add_argument("--disk-stop-pct", type=float, default=92.0)
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    restarts = 0
    spend_strikes = 0
    last_hist = 0.0
    events: list[dict] = []

    def event(msg: str):
        events.append({"time": time.time(), "msg": msg})
        print(f"[monitor {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    event("monitor started")
    while True:
        t = time.time()
        st = read_json(run / "state.json") or {}
        tr = read_json(run / "trainer_status.json") or {}
        live = read_json(run / "live" / "status.json") or {}
        ghs = read_json(run / "github" / "status.json") or {}
        g = gpu()
        mem = cgroup_mem()
        du = shutil.disk_usage(run)
        disk_pct = 100 * (du.total - du.free) / du.total
        trainer = supervisor_state("agentrs_trainer")
        started = st.get("session_started_at") or st.get("started_at", t)  # spend cap / elapsed are per session
        deadline = st.get("deadline")
        dph = (st.get("config") or {}).get("dph_usd") or 0.0
        max_spend = (st.get("config") or {}).get("max_spend_usd")
        spent = (t - started) / 3600 * dph
        hb_age = t - tr.get("heartbeat", t)
        warnings = []
        ghb = read_json(run / "gh_backup_status.json") or {}
        gh_last = ghb.get("last_success") or {}
        gh_age_h = (t - gh_last["uploaded_at"]) / 3600 if gh_last.get("uploaded_at") else None
        backup_proc = supervisor_state("agentrs_backup")
        if gh_age_h is None or gh_age_h > 6.25:
            warnings.append("no verified GitHub checkpoint in the last 6 h" if gh_age_h else "no verified GitHub checkpoint yet")
        if backup_proc != "RUNNING" and st.get("phase") not in ("done",):
            warnings.append(f"GitHub backup service is {backup_proc}")
        # ---- safety rails
        if trainer == "RUNNING" and tr and hb_age > a.stale_min * 60 and st.get("phase") not in ("done",):
            if restarts < a.max_trainer_restarts:
                restarts += 1
                event(f"trainer heartbeat stale ({hb_age / 60:.0f} min): restarting trainer ({restarts}/{a.max_trainer_restarts})")
                sh(["supervisorctl", "restart", "agentrs_trainer"], timeout=300)
            else:
                warnings.append("trainer heartbeat stale and restart budget exhausted")
        if disk_pct > a.disk_stop_pct and not (run / "STOP").exists():
            (run / "STOP").write_text(f"disk {disk_pct:.1f}%")
            event(f"disk at {disk_pct:.1f}%: STOP requested")
        # spending rail: only once the trainer has applied a pending NEW_SESSION (otherwise the previous
        # session's start time would be billed against this session's cap), and confirmed on two reads
        session_pending = (run / "NEW_SESSION").exists() or (st.get("sessions") and not st.get("session_started_at"))
        over = bool(max_spend and dph and spent >= max_spend and not session_pending)
        spend_strikes = spend_strikes + 1 if over else 0
        if over and spend_strikes >= 2 and not (run / "STOP").exists():
            (run / "STOP").write_text(f"spend ${spent:.2f}")
            event(f"spending cap reached (${spent:.2f}): STOP requested")
        if deadline and t > deadline + a.grace_min * 60 and trainer == "RUNNING":
            event("hard deadline passed: stopping trainer")
            sh(["supervisorctl", "stop", "agentrs_trainer"], timeout=600)
        ckpt = (st.get("latest_checkpoint") or {})
        ck_age = t - ckpt.get("time", t)
        if st.get("phase") == "rl" and ck_age > 45 * 60:
            warnings.append(f"latest checkpoint is {ck_age / 60:.0f} min old")
        if g and g.get("vram_used_gb", 0) > 0.97 * g.get("vram_total_gb", 1):
            warnings.append("VRAM nearly full")
        if mem.get("limit_gb") and mem["used_gb"] > 0.9 * mem["limit_gb"]:
            warnings.append("container memory above 90% of its cgroup limit")
        evals = st.get("evals") or []
        val = [e for e in evals if e.get("split") == "val"]
        c = st.get("counters") or {}
        by_dom = {d: {"attempts": v["attempts"], "successes": v["successes"],
                      "rate": round(v["successes"] / max(1, v["attempts"]), 3)} for d, v in (c.get("by_domain") or {}).items()}
        status = {
            "time": t, "run_start": started, "elapsed_h": round((t - started) / 3600, 3),
            "deadline": deadline, "remaining_h": round((deadline - t) / 3600, 3) if deadline else None,
            "phase": st.get("phase"), "stage": tr.get("stage"), "trainer_process": trainer,
            "trainer_heartbeat_age_s": round(hb_age), "gpu": g, "container_memory": mem,
            "disk": {"used_pct": round(disk_pct, 1), "free_gb": round(du.free / 2**30, 1)},
            "task_attempts": c.get("attempts"), "verified_completions": c.get("successes"),
            "training_success_by_domain": by_dom, "self_play_cycles": c.get("cycles"),
            "policy_version": st.get("version"), "policy_weight_updates": c.get("policy_updates"),
            "bc_updates": c.get("bc_updates"), "world_model_updates": c.get("wm_updates"),
            "critic_updates": c.get("critic_updates"),
            "latest_validation": {k: val[-1].get(k) for k in ("tag", "cycle", "train_families", "heldout_families",
                                                              "world_model", "critic")} if val else None,
            "baselines": {e["tag"] + "/" + e["split"]: e["summary"]["overall"] for e in evals
                          if e["tag"] in ("A_pretrained", "B_after_bc", "C_final", "C_best")},
            "best_checkpoint": st.get("best"), "promoted_checkpoint": st.get("promoted"),
            "latest_local_checkpoint": ckpt,
            "latest_verified_remote_backup": ({k: gh_last.get(k) for k in ("tag", "url", "kind", "checkpoint", "version", "cycle",
                                                                          "verified", "verify_method", "tar_bytes")}
                                              | {"age_h": round(gh_age_h, 2)} if gh_last else "none yet"),
            "github_backup": {"process": backup_proc, "uploads": len(ghb.get("uploads", [])),
                              "failures": len(ghb.get("failures", [])), "in_progress": ghb.get("in_progress"),
                              "last_failure": (ghb.get("failures") or [None])[-1]},
            "task_factory": tr.get("factory"), "prior_sessions_spend_usd": st.get("prior_spend_usd"),
            "estimated_spend_usd": round(spent, 2), "price_per_hour_usd": dph, "spend_cap_usd": max_spend,
            "hackernews_bluesky_observer": {"process": supervisor_state("agentrs_live"),
                                            **{k: live.get(k) for k in ("uptime_s", "posts", "observations", "updates",
                                                                        "optimizer_steps", "checkpoint")},
                                            "scores": (live.get("scores_all_time") or {}).get("overall")},
            "github_observer": {"process": supervisor_state("agentrs_github"), **ghs},
            "errors": (st.get("errors") or [])[-5:], "warnings": warnings, "monitor_events": events[-10:],
            "trainer_restarts_by_monitor": restarts, "trainer_process_restarts": st.get("restarts"),
        }
        tmp = run / "latest_status.json.tmp"
        tmp.write_text(json.dumps(status, indent=1, default=str))
        os.replace(tmp, run / "latest_status.json")
        md = [f"# AGENTRS overnight status — {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(t))}", "",
              f"- phase **{status['phase']}** / stage `{status['stage']}`; trainer {trainer}; elapsed {status['elapsed_h']} h, "
              f"remaining {status['remaining_h']} h",
              f"- GPU {g.get('util_pct')}% util, {g.get('vram_used_gb', 0):.1f}/{g.get('vram_total_gb', 0):.1f} GB VRAM; "
              f"container RAM {mem.get('used_gb', 0):.1f}/{mem.get('limit_gb') or 0:.0f} GB; disk {disk_pct:.1f}%",
              f"- attempts {status['task_attempts']}, verified completions {status['verified_completions']}, cycles "
              f"{status['self_play_cycles']}, policy updates {status['policy_weight_updates']}, WM {status['world_model_updates']}, "
              f"critic {status['critic_updates']}",
              f"- by domain: {json.dumps(by_dom)}",
              f"- latest validation: {json.dumps(status['latest_validation'], default=str)[:600]}",
              f"- best: {status['best_checkpoint']}; promoted: {status['promoted_checkpoint']}",
              f"- spend this session ≈ ${status['estimated_spend_usd']} (cap ${max_spend}; earlier sessions "
              f"${st.get('prior_spend_usd', 0):.2f})",
              f"- GitHub checkpoint backup: {json.dumps(status['latest_verified_remote_backup'], default=str)[:400]}; "
              f"service {backup_proc}, {status['github_backup']['uploads']} uploads, {status['github_backup']['failures']} failures",
              f"- task factory: {json.dumps(tr.get('factory'), default=str)[:300]}",
              f"- HN/Bluesky observer: {json.dumps(status['hackernews_bluesky_observer'], default=str)[:600]}",
              f"- GitHub observer: {json.dumps(status['github_observer'], default=str)[:400]}",
              f"- warnings: {warnings}; errors: {[e.get('error', '')[:120] for e in status['errors']]}"]
        (run / "latest_status.md").write_text("\n".join(md) + "\n")
        if t - last_hist > 600:
            with open(run / "status_history.jsonl", "a") as f:
                f.write(json.dumps({k: status[k] for k in ("time", "phase", "task_attempts", "verified_completions",
                                                           "policy_weight_updates", "latest_validation", "gpu",
                                                           "estimated_spend_usd")}, default=str) + "\n")
            last_hist = t
        # ---- end of run: trainer finished (or stopped after the deadline) -> stop observers, exit
        if st.get("phase") == "done" or (deadline and t > deadline + a.grace_min * 60 and trainer != "RUNNING"):
            for prog in ("agentrs_live", "agentrs_github"):
                if supervisor_state(prog) == "RUNNING":
                    event(f"run finished: stopping {prog}")
                    sh(["supervisorctl", "stop", prog], timeout=120)
            event("run finished; monitor exiting")
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
