"""cam_live.runtime.dashboard — operator monitoring interface (spec 15.3).

A FastAPI application that only *reads* the run directory and writes operator commands to the control channel
(approvals go to the authorization service first when the operator's handle on it is given). It is token-protected
and binds to the loopback interface only.

The interface supervises; it does not steer. There is no input through which a way of pursuing the objective
could be given: the only free-text inputs are the audit notes of stop / approve / reject, which never reach the
learner, and a subsequent objective can only be assigned by choosing a task file, which passes task validation.
"""
from __future__ import annotations

import hmac
import os
import secrets
import shutil
from typing import Any, Dict, List, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from ..contracts import PolicyEngine, RunFiles, WallClock
from .control import OperatorControl, pid_alive, read_pid
from .journal import behavioural_change, prediction_stats
from .logs import LOG_FILES, RunReader, read_json
from .tasks import TaskValidationError, default_tasks_dir, load_task
from .verifier import HashChainLog

LOOPBACK = ("127.0.0.1", "localhost", "::1")
TOKEN_FILE = "dashboard.token"
COOKIE = "cam_live_token"


def load_or_create_token(directory: str) -> str:
    """The dashboard token (file mode 0600 in ``directory``)."""
    os.makedirs(directory, exist_ok=True)
    p = os.path.join(directory, TOKEN_FILE)
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        with open(p) as f:
            return f.read().strip()
    tok = secrets.token_urlsafe(32)
    with os.fdopen(fd, "w") as f:
        f.write(tok)
    return tok


def _num_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """mean / min / max / n of every numeric field across a list of flat dicts."""
    acc: Dict[str, List[float]] = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                acc.setdefault(k, []).append(float(v))
    return {k: {"n": len(v), "mean": sum(v) / len(v), "min": min(v), "max": max(v), "last": v[-1]}
            for k, v in acc.items()}


class RunView:
    """Everything the dashboard shows, computed from the files of one run directory."""

    def __init__(self, run_dir: str):
        self.run_dir = os.path.abspath(run_dir)
        self.r = RunReader(self.run_dir)

    def status(self) -> Dict[str, Any]:
        st = self.r.state()
        task = st.get("task", {})
        lv = st.get("last_verified") or {}
        vo = lv.get("outcome") or {}
        return {
            "run_id": st.get("run_id", ""), "status": st.get("status", "no state"),
            "status_reason": st.get("status_reason", ""), "awaiting_operator": st.get("awaiting_operator", False),
            "objective": {"id": task.get("objective_id", ""), "text": task.get("objective", ""),
                          "mode": task.get("mode", ""), "family": task.get("family", ""),
                          "queued": [t["task"]["objective_id"] for t in st.get("next_tasks", [])]},
            "verified": {"status": vo.get("status"), "value": vo.get("value"), "target": vo.get("target"),
                         "uncertainty": vo.get("uncertainty"), "excluded": vo.get("excluded"),
                         "notes": vo.get("notes"), "secondary": vo.get("secondary"), "evaluated_at": vo.get("evaluated_at"),
                         "record": lv.get("rid"), "refused": st.get("refused_verified", 0)},
            "activity": st.get("activity", {}), "wake": st.get("wake", {}),
            "resources": st.get("limits", {}), "counters": st.get("counters", {}),
            "lifetime": st.get("lifetime", {}),
            "checkpoint": (st.get("committed") or {}).get("ckpt"), "exports": st.get("exports", [])[-5:],
            "pending_outcomes": len(((st.get("live") or {}).get("pending") or {}).get("items", [])),
            "pending_approvals": sum(1 for a in (st.get("approvals") or {}).values() if a.get("status") == "pending"),
            "updated_at": st.get("updated_at"),
        }

    def experiments(self, n: int = 20) -> List[Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for e in self.r.records(RunFiles.EXPERIMENTS, types=["experiment"]):
            d = e["data"]
            latest[d.get("experiment_id", "")] = {
                "record": e["rid"], "ts": e["ts"], "experiment_id": d.get("experiment_id"),
                "question": d.get("question"), "hypothesis": d.get("hypothesis"), "status": d.get("status"),
                "intervention": d.get("intervention"), "assignment": d.get("assignment"),
                "predicted": [{"mean": p["dist"].get("mean"), "std": p["dist"].get("std"), "metric": p["query"].get("metric")}
                              for p in d.get("predicted") or []],
                "actual": [{"value": o.get("value"), "status": o.get("status"), "verified": o.get("verified")}
                           for o in d.get("actual") or []],
                "supported": d.get("supported"), "cost": d.get("cost")}
        return list(latest.values())[-n:]

    def predictions(self, n: int = 50) -> Dict[str, Any]:
        res = self.r.records(RunFiles.PREDICTIONS, types=["resolution"])
        rows = [{"record": x["rid"], "ts": x["ts"], "prediction_id": x["data"]["prediction_id"],
                 "metric": (x["data"].get("outcome") or {}).get("metric"),
                 "predicted_mean": x["data"].get("predicted_mean"), "predicted_std": x["data"].get("predicted_std"),
                 "actual": (x["data"].get("outcome") or {}).get("value"),
                 "verified": (x["data"].get("outcome") or {}).get("verified")} for x in res[-n:]]
        registered = sum(1 for _ in self.r.records(RunFiles.PREDICTIONS, types=["prediction"]))
        return {"resolved": rows, "n_resolved": len(res), "n_registered": registered}

    def calibration(self) -> Dict[str, Any]:
        ps = prediction_stats(self.r.records(RunFiles.PREDICTIONS, types=["resolution"]))
        for k in ("rids", "early_rids", "late_rids"):
            ps.pop(k, None)
        ps["note"] = ("accuracy of prospectively registered predictions against resolved outcomes; coverage uses "
                      "the predictive mean and standard deviation only")
        return ps

    def rewards(self, n: int = 200) -> Dict[str, Any]:
        ev = self.r.records(RunFiles.EVENTS, types=["internal_rewards"], tail=n)
        return {"n_reports": len(ev), "last": ev[-1]["data"] if ev else None,
                "stats": _num_stats([e["data"] for e in ev]),
                "note": "as reported by the learner; the environment and the runtime produce no reward"}

    def learning(self, n: int = 20) -> Dict[str, Any]:
        ups = self.r.records(RunFiles.LEARNING)
        rows = [{"record": u["rid"], "ts": u["ts"], **{k: u["data"].get(k) for k in (
            "update_id", "trigger", "timescale", "losses", "param_groups", "grad_norms", "weight_delta",
            "prediction_change", "n_experiences", "checkpoint_id", "model_version", "notes")},
            "behavioural_change": behavioural_change(u["data"])} for u in ups[-n:]]
        return {"n_updates": len(ups), "n_with_behavioural_change": sum(1 for u in ups if behavioural_change(u["data"])),
                "recent": rows}

    def meta(self, n: int = 20) -> Dict[str, Any]:
        ms = self.r.records(RunFiles.META)
        skipped: Dict[str, int] = {}
        for m in ms:
            if not m["data"].get("applied"):
                k = m["data"].get("skipped_reason") or "unspecified"
                skipped[k] = skipped.get(k, 0) + 1
        last = ms[-1]["data"] if ms else {}
        return {"n_updates": len(ms), "n_applied": sum(1 for m in ms if m["data"].get("applied")),
                "skipped_reasons": skipped, "bootstrap_mix": last.get("bootstrap_mix"),
                "reward_version": last.get("reward_version_after"),
                "recent": [{"record": m["rid"], "ts": m["ts"], **m["data"]} for m in ms[-n:]]}

    def approvals(self) -> Dict[str, Any]:
        st = self.r.state()
        rows = [{k: a.get(k) for k in ("approval_id", "status", "operation", "preview", "reasons", "queued_at",
                                       "resolved_at", "result_status")}
                for a in (st.get("approvals") or {}).values()]
        return {"pending": [a for a in rows if a["status"] == "pending"],
                "resolved": [a for a in rows if a["status"] != "pending"][-20:]}

    def obstacles(self, n: int = 20) -> List[Dict[str, Any]]:
        return [{"record": o["rid"], **o["data"]["obstacle"]} for o in self.r.obstacles()[-n:]]

    def health(self) -> Dict[str, Any]:
        st = self.r.state()
        pid = read_pid(self.run_dir)
        du = shutil.disk_usage(self.run_dir)
        out: Dict[str, Any] = {
            "process": {"pid": pid, "alive": pid_alive(pid)},
            "state_file_age_s": None, "disk_free_gb": round(du.free / 2**30, 2),
            "errors_recent": (st.get("errors") or [])[-5:],
            "log_bytes": {n: os.path.getsize(self.r.path(n)) for n in LOG_FILES if os.path.exists(self.r.path(n))},
            "restarts": len(self.r.records(RunFiles.EVENTS, types=["run_resumed"])),
        }
        try:
            out["state_file_age_s"] = round(WallClock().now() - os.path.getmtime(self.r.path(RunFiles.STATE)), 1)
        except OSError:
            pass
        vlog = self.r.path(RunFiles.VERIFIER)
        if os.path.exists(vlog):
            try:
                head = read_json(os.path.join(self.run_dir, "verifier", "head.json"))
                ok, n, why = HashChainLog.verify_chain(vlog, head)
                out["verifier_log"] = {"intact": ok, "entries": n, "problem": why}
            except OSError as e:
                out["verifier_log"] = {"intact": None, "problem": f"not readable by this user: {e.strerror}"}
        return out

    def log_tail(self, name: str, n: int = 100, before_seq: Optional[int] = None) -> List[Dict[str, Any]]:
        if name not in LOG_FILES:
            raise KeyError(name)
        rows = self.r.records(name)
        if before_seq is not None:
            rows = [x for x in rows if x.get("seq", 0) < before_seq]
        return rows[-max(1, min(n, 1000)):]


def create_app(run_dir: str, token: str, policy: Optional[PolicyEngine] = None, tasks_dir: Optional[str] = None,
               operator: str = "operator") -> FastAPI:
    if not token or len(token) < 16:
        raise ValueError("a dashboard token of at least 16 characters is required")
    view = RunView(run_dir)
    ctl = OperatorControl(view.run_dir, policy=policy, by=operator)
    tasks_dir = os.path.abspath(tasks_dir or default_tasks_dir())
    app = FastAPI(title="CAM Live", docs_url=None, redoc_url=None, openapi_url=None)

    def auth(request: Request) -> None:
        h = request.headers.get("authorization", "")
        given = h[7:] if h.lower().startswith("bearer ") else (request.query_params.get("token")
                                                               or request.cookies.get(COOKIE) or "")
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="missing or wrong token")

    guard = [Depends(auth)]

    @app.get("/", response_class=HTMLResponse, dependencies=guard)
    def page() -> HTMLResponse:
        resp = HTMLResponse(PAGE)
        resp.set_cookie(COOKIE, token, httponly=True, samesite="strict")
        return resp

    @app.get("/api/status", dependencies=guard)
    def status() -> Dict[str, Any]:
        return view.status()

    @app.get("/api/experiments", dependencies=guard)
    def experiments(n: int = 20) -> List[Dict[str, Any]]:
        return view.experiments(n)

    @app.get("/api/predictions", dependencies=guard)
    def predictions(n: int = 50) -> Dict[str, Any]:
        return view.predictions(n)

    @app.get("/api/calibration", dependencies=guard)
    def calibration() -> Dict[str, Any]:
        return view.calibration()

    @app.get("/api/rewards", dependencies=guard)
    def rewards() -> Dict[str, Any]:
        return view.rewards()

    @app.get("/api/learning", dependencies=guard)
    def learning(n: int = 20) -> Dict[str, Any]:
        return view.learning(n)

    @app.get("/api/meta", dependencies=guard)
    def meta(n: int = 20) -> Dict[str, Any]:
        return view.meta(n)

    @app.get("/api/approvals", dependencies=guard)
    def approvals() -> Dict[str, Any]:
        return view.approvals()

    @app.get("/api/obstacles", dependencies=guard)
    def obstacles(n: int = 20) -> List[Dict[str, Any]]:
        return view.obstacles(n)

    @app.get("/api/health", dependencies=guard)
    def health() -> Dict[str, Any]:
        return view.health()

    @app.get("/api/logs/{name}", dependencies=guard)
    def logs(name: str, n: int = 100, before_seq: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            return view.log_tail(name, n, before_seq)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown log; choose from {sorted(LOG_FILES)}")

    @app.get("/api/journal", response_class=PlainTextResponse, dependencies=guard)
    def journal() -> str:
        try:
            with open(view.r.path(RunFiles.JOURNAL)) as f:
                return f.read()
        except FileNotFoundError:
            return ""

    @app.get("/api/tasks", dependencies=guard)
    def tasks() -> List[str]:
        return sorted(f for f in os.listdir(tasks_dir) if f.endswith((".yaml", ".yml"))) if os.path.isdir(tasks_dir) else []

    @app.get("/api/control/result/{command_id}", dependencies=guard)
    def control_result(command_id: str) -> Dict[str, Any]:
        return {"command_id": command_id, "result": ctl.result(command_id)}

    @app.post("/api/control/pause", dependencies=guard)
    def pause(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        return {"command_id": ctl.pause(str(body.get("reason", ""))[:200]), "queued": True}

    @app.post("/api/control/resume", dependencies=guard)
    def resume() -> Dict[str, Any]:
        return {"command_id": ctl.resume(), "queued": True}

    @app.post("/api/control/stop", dependencies=guard)
    def stop(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        return {"command_id": ctl.stop(str(body.get("reason", ""))[:200], emergency=bool(body.get("emergency"))),
                "queued": True}

    def _known(approval_id: str) -> None:
        if approval_id not in (view.r.state().get("approvals") or {}):
            raise HTTPException(status_code=404, detail="unknown approval")

    @app.post("/api/approvals/{approval_id}/approve", dependencies=guard)
    def approve(approval_id: str, body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        _known(approval_id)
        return {"command_id": ctl.approve(approval_id, str(body.get("note", ""))[:200]), "queued": True}

    @app.post("/api/approvals/{approval_id}/reject", dependencies=guard)
    def reject(approval_id: str, body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        _known(approval_id)
        return {"command_id": ctl.reject(approval_id, str(body.get("note", ""))[:200]), "queued": True}

    @app.post("/api/control/export-checkpoint", dependencies=guard)
    def export_checkpoint() -> Dict[str, Any]:
        return {"command_id": ctl.export_checkpoint(), "queued": True}     # destination: <run_dir>/exports/

    @app.post("/api/control/assign-objective", dependencies=guard)
    def assign(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        name = os.path.basename(str(body.get("task", "")))
        path = os.path.join(tasks_dir, name)
        if not name or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="task file not found in the tasks directory")
        try:
            load_task(path, for_run=True)
            return {"command_id": ctl.assign_objective(path), "queued": True}
        except TaskValidationError as e:
            raise HTTPException(status_code=422, detail=e.problems)

    return app


def serve(run_dir: str, token: str, host: str = "127.0.0.1", port: int = 8787, policy: Optional[PolicyEngine] = None,
          tasks_dir: Optional[str] = None) -> None:
    if host not in LOOPBACK:
        raise ValueError(f"the dashboard binds to the loopback interface only (got {host!r}); reach it through an "
                         "SSH tunnel or an authenticating reverse proxy")
    import uvicorn
    uvicorn.run(create_app(run_dir, token, policy, tasks_dir), host=host, port=port, log_level="warning")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CAM Live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font:14px/1.45 system-ui,sans-serif;margin:0;background:#f5f6f8;color:#1c2330}
header{background:#1c2330;color:#fff;padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header b{font-size:16px} main{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px;padding:12px}
section{background:#fff;border:1px solid #d9dde4;border-radius:6px;padding:10px 12px;overflow:auto;max-height:420px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;margin:0 0 6px;color:#52607a}
table{border-collapse:collapse;width:100%} td,th{text-align:left;padding:2px 8px 2px 0;vertical-align:top;border-bottom:1px solid #eef0f4}
pre{white-space:pre-wrap;word-break:break-word;margin:0;font:12px/1.4 ui-monospace,monospace}
button{font:inherit;padding:4px 10px;border:1px solid #8a94a6;border-radius:4px;background:#fff;cursor:pointer}
button.warn{border-color:#b3261e;color:#b3261e} input,select{font:inherit;padding:3px 6px}
.pill{padding:1px 8px;border-radius:10px;background:#e4e8ef} .bar{height:8px;background:#e4e8ef;border-radius:4px}
.bar i{display:block;height:8px;background:#3d6fd9;border-radius:4px} .muted{color:#6b7689}
</style></head><body>
<header><b>CAM Live</b><span id="run" class="muted"></span><span id="st" class="pill"></span>
<span style="flex:1"></span>
<button id="b-pause">Pause</button><button id="b-resume">Resume</button>
<input id="stop-note" title="audit note (not shown to CAM)" aria-label="audit note (not shown to CAM)" size="26" maxlength="200">
<button id="b-stop" class="warn">Stop</button><button id="b-estop" class="warn">Emergency stop</button>
<button id="b-export">Export checkpoint</button></header>
<main>
<section><h2>Objective and verified progress</h2><div id="objective"></div></section>
<section><h2>Current activity and health</h2><div id="activity"></div></section>
<section><h2>Resources against limits</h2><div id="resources"></div></section>
<section><h2>Pending approvals</h2><div id="approvals"></div></section>
<section><h2>Recent experiments</h2><div id="experiments"></div></section>
<section><h2>Predicted and actual outcomes</h2><div id="predictions"></div></section>
<section><h2>Internal reward statistics and calibration</h2><div id="rewards"></div></section>
<section><h2>Recent learning updates</h2><div id="learning"></div></section>
<section><h2>Meta-learning</h2><div id="meta"></div></section>
<section><h2>Obstacles</h2><div id="obstacles"></div></section>
<section><h2>Assign a subsequent objective</h2>
<p class="muted">Choose a task file. Task files state an objective, its verification and its boundaries; files that
contain a way of achieving the objective are rejected.</p>
<select id="task-file"></select> <button id="b-assign">Assign</button><div id="assign-out" class="muted"></div></section>
<section style="grid-column:1/-1;max-height:520px"><h2>Logs
<select id="log-name"><option>events.jsonl</option><option>decisions.jsonl</option><option>experiments.jsonl</option>
<option>predictions.jsonl</option><option>learning_updates.jsonl</option><option>meta_updates.jsonl</option>
<option>obstacles.jsonl</option><option value="journal">journal.md</option></select></h2><pre id="log"></pre></section>
</main>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const get=async p=>{const r=await fetch(p);if(!r.ok)throw new Error(p+' '+r.status);return p.endsWith('journal')?r.text():r.json()};
const post=async(p,b)=>{const r=await fetch(p,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b||{})});
  const j=await r.json();if(!r.ok)alert(JSON.stringify(j.detail));refresh();return j};
const rows=(a,cols)=>'<table><tr>'+cols.map(c=>'<th>'+esc(c[0])+'</th>').join('')+'</tr>'+a.map(x=>'<tr>'+cols.map(c=>'<td>'+esc(c[1](x))+'</td>').join('')+'</tr>').join('')+'</table>';
const f=x=>x==null?'—':(typeof x==='number'?(+x.toPrecision(5)):x);
async function refresh(){try{
 const s=await get('/api/status');$('run').textContent=s.run_id;$('st').textContent=s.status+(s.status_reason?' — '+s.status_reason:'');
 const v=s.verified;
 $('objective').innerHTML='<p><b>'+esc(s.objective.id)+'</b> <span class="pill">'+esc(s.objective.mode)+'</span></p><p>'+esc(s.objective.text||'no objective assigned')+'</p>'+
  '<p>Verified: <b>'+f(v.value)+'</b> of '+f(v.target)+' ('+esc(v.status||'no signed verification yet')+', uncertainty '+f(v.uncertainty)+')</p>'+
  '<p class="muted">'+esc(v.notes||'')+'</p><pre>excluded '+esc(JSON.stringify(v.excluded||{}))+'\\nsecondary '+esc(JSON.stringify((v.secondary||[]).map(q=>[q.name,q.value])))+'</pre>'+
  (s.objective.queued.length?'<p>Queued objectives: '+esc(s.objective.queued.join(', '))+'</p>':'');
 const h=await get('/api/health');
 $('activity').innerHTML='<p><b>'+esc(s.activity.phase||'')+'</b> '+esc(s.activity.detail||'')+'</p><p>Next wake: '+esc(JSON.stringify(s.wake||{}))+'</p>'+
  '<p>Checkpoint: '+esc(s.checkpoint?s.checkpoint.id+' (generation '+s.checkpoint.generation+')':'none')+'</p>'+
  '<p>Pending outcomes: '+s.pending_outcomes+' · process '+(h.process.alive?'alive':'not running')+' (pid '+f(h.process.pid)+') · restarts '+h.restarts+
  ' · disk free '+h.disk_free_gb+' GB · verifier log '+esc(h.verifier_log?(h.verifier_log.intact?'intact':'PROBLEM: '+h.verifier_log.problem):'n/a')+'</p>'+
  '<pre>'+esc(JSON.stringify(h.errors_recent,null,1))+'</pre>';
 $('resources').innerHTML=Object.entries(s.resources).map(([k,r])=>'<p>'+esc(k)+': '+f(r.used)+(r.limit>0?' of '+f(r.limit):' (no limit set)')+
  (r.limit>0?'<span class="bar"><i style="width:'+Math.min(100,100*r.used/r.limit)+'%"></i></span>':'')+'</p>').join('');
 const a=await get('/api/approvals');
 $('approvals').innerHTML=(a.pending.length?'':'<p class="muted">none pending</p>')+a.pending.map(p=>'<div><p><b>'+esc(p.approval_id)+'</b> '+esc(p.operation)+
  ' <span class="muted">'+esc((p.reasons||[]).join('; '))+'</span></p><pre>'+esc(p.preview)+'</pre><p><input id="n-'+esc(p.approval_id)+'" title="audit note (not shown to CAM)" aria-label="audit note (not shown to CAM)" maxlength="200"> '+
  '<button data-ap="'+esc(p.approval_id)+'" data-do="approve">Approve</button> <button class="warn" data-ap="'+esc(p.approval_id)+'" data-do="reject">Reject</button></p></div>').join('')+
  (a.resolved.length?rows(a.resolved,[['id',x=>x.approval_id],['operation',x=>x.operation],['status',x=>x.status]]):'');
 const e=await get('/api/experiments');
 $('experiments').innerHTML=rows(e.reverse(),[['question',x=>x.question],['hypothesis',x=>x.hypothesis],['predicted',x=>(x.predicted||[]).map(p=>f(p.mean)+'±'+f(p.std)).join(', ')],
  ['actual',x=>(x.actual||[]).map(o=>f(o.value)).join(', ')],['supported',x=>f(x.supported)],['status',x=>x.status]]);
 const p=await get('/api/predictions');
 $('predictions').innerHTML='<p>'+p.n_resolved+' resolved of '+p.n_registered+' registered</p>'+rows(p.resolved.reverse(),[['metric',x=>x.metric],['predicted',x=>f(x.predicted_mean)+' ± '+f(x.predicted_std)],['actual',x=>f(x.actual)],['verified',x=>x.verified]]);
 const r=await get('/api/rewards'),c=await get('/api/calibration');
 $('rewards').innerHTML='<p class="muted">'+esc(r.note)+'</p><pre>'+esc(JSON.stringify(r.stats,null,1))+'</pre><p class="muted">'+esc(c.note)+'</p><pre>'+esc(JSON.stringify(c,null,1))+'</pre>';
 const l=await get('/api/learning');
 $('learning').innerHTML='<p>'+l.n_updates+' updates, '+l.n_with_behavioural_change+' with a measured behavioural change</p>'+rows(l.recent.reverse(),[['timescale',x=>x.timescale],['trigger',x=>x.trigger],
  ['groups',x=>(x.param_groups||[]).join(',')],['weight change',x=>JSON.stringify(x.weight_delta)],['prediction change',x=>JSON.stringify(x.prediction_change)],['losses',x=>JSON.stringify(x.losses)]]);
 const m=await get('/api/meta');
 $('meta').innerHTML='<p>'+m.n_applied+' applied of '+m.n_updates+' · bootstrap mix '+f(m.bootstrap_mix)+' · reward version '+esc(f(m.reward_version))+'</p><pre>skipped: '+esc(JSON.stringify(m.skipped_reasons,null,1))+'</pre>'+
  rows(m.recent.reverse(),[['applied',x=>x.applied],['reason',x=>x.skipped_reason],['independent outcomes',x=>x.n_independent_outcomes],['meta-gradient norm',x=>f(x.meta_grad_norm)],['estimator',x=>x.estimator]]);
 const o=await get('/api/obstacles');
 $('obstacles').innerHTML=rows(o.reverse(),[['category',x=>x.category],['failure',x=>x.failure_class],['description',x=>x.description],['what followed',x=>x.outcome],['learning updates',x=>(x.learning_update_ids||[]).length]]);
 const n=$('log-name').value;
 $('log').textContent=n==='journal'?await get('/api/journal'):(await get('/api/logs/'+n+'?n=60')).reverse().map(x=>JSON.stringify(x)).join('\\n');
}catch(err){$('st').textContent='error: '+err.message}}
document.addEventListener('click',ev=>{const t=ev.target;if(t.dataset&&t.dataset.ap)post('/api/approvals/'+t.dataset.ap+'/'+t.dataset.do,{note:$('n-'+t.dataset.ap).value})});
$('b-pause').onclick=()=>post('/api/control/pause');$('b-resume').onclick=()=>post('/api/control/resume');
$('b-stop').onclick=()=>confirm('Stop the run?')&&post('/api/control/stop',{reason:$('stop-note').value});
$('b-estop').onclick=()=>confirm('Emergency stop?')&&post('/api/control/stop',{reason:$('stop-note').value,emergency:true});
$('b-export').onclick=()=>post('/api/control/export-checkpoint');
$('b-assign').onclick=async()=>{const j=await post('/api/control/assign-objective',{task:$('task-file').value});$('assign-out').textContent=JSON.stringify(j)};
$('log-name').onchange=refresh;
get('/api/tasks').then(t=>{$('task-file').innerHTML=t.map(x=>'<option>'+esc(x)+'</option>').join('')});
refresh();setInterval(refresh,5000);
</script></body></html>
"""
