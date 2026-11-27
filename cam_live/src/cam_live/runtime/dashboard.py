"""cam_live.runtime.dashboard — operator monitoring interface (spec 15.3).

A FastAPI application that only *reads* the run directory and writes operator commands to the control channel
(approvals, emergency stops and their reset go to the authorization service first when the operator's handle on it
is given). It binds to the loopback interface only.

Two credentials, both kept in files of mode 0600 owned by the operator:

* the **viewing token** (``dashboard.token``) opens every page and every read endpoint (``Authorization: Bearer``,
  ``?token=`` or the HttpOnly ``cam_live_token`` cookie the page sets);
* the **operator credential** (``dashboard.operator``) is needed in addition for every operator action — approve,
  reject, pause, resume, stop, emergency stop, reset of the emergency stop, export, assigning an objective. It is
  accepted only from the ``X-CAM-Operator`` request header, never from a URL or a cookie, and every action also needs
  the double-submit CSRF token (``cam_live_csrf`` cookie echoed in ``X-CSRF-Token``) and a same-origin request.
  An action arriving over a loopback connection whose socket belongs to one of the service uids (the learner, the
  browser, the sandbox, ...) is refused whatever credentials it carries.

Everything the learner or the environment wrote is untrusted text. The page has no inline script, no inline style
and no event-handler attribute; a strict Content-Security-Policy (``script-src 'self'``) is sent with every response,
and the client script builds the page with ``textContent`` only — no markup from data is ever parsed.

The interface supervises; it does not steer. There is no input through which a way of pursuing the objective
could be given: the only free-text inputs are the audit notes of stop / approve / reject, which never reach the
learner, and a subsequent objective can only be assigned by choosing a task file, which passes task validation.
"""
from __future__ import annotations

import hmac
import ipaddress
import os
import pwd
import secrets
import shutil
import stat
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response

from ..contracts import PolicyEngine, RunFiles, WallClock
from ..harness import SERVICE_USERS
from .control import OperatorControl, pid_alive, read_pid
from .journal import behavioural_change, prediction_stats
from .logs import LOG_FILES, Redactor, RunReader, read_json
from .tasks import TaskValidationError, default_tasks_dir, load_task
from .verifier import HashChainLog

LOOPBACK = ("127.0.0.1", "localhost", "::1")
TOKEN_FILE = "dashboard.token"
OPERATOR_TOKEN_FILE = "dashboard.operator"
COOKIE = "cam_live_token"
CSRF_COOKIE = "cam_live_csrf"
OPERATOR_HEADER = "X-CAM-Operator"
CSRF_HEADER = "X-CSRF-Token"
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'none'; "
       "font-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
SECURITY_HEADERS = {"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff",
                    "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY", "Cache-Control": "no-store",
                    "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Resource-Policy": "same-origin"}
PROC_TCP_TABLES = ("/proc/net/tcp", "/proc/net/tcp6")


# --------------------------------------------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------------------------------------------
def _private_token(path: str) -> str:
    """Create a random token file (0600, O_EXCL) or read an existing one after checking that only this uid owns
    and can read it."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode):
            raise PermissionError(f"{path} is not a regular file")
        if st.st_uid != os.geteuid():
            raise PermissionError(f"{path} is owned by uid {st.st_uid}, not by this user (uid {os.geteuid()})")
        if stat.S_IMODE(st.st_mode) & 0o077:
            raise PermissionError(f"{path} can be read by other users (mode {stat.S_IMODE(st.st_mode):o}); it must "
                                  "be 0600")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as f:
            tok = f.read().strip()
        if len(tok) < 16:
            raise ValueError(f"{path} holds no usable token")
        return tok
    tok = secrets.token_urlsafe(32)
    with os.fdopen(fd, "w") as f:
        f.write(tok)
        f.flush()
        os.fsync(f.fileno())
    return tok


def load_or_create_token(directory: str) -> str:
    """The dashboard viewing token (file ``dashboard.token``, mode 0600 in ``directory``)."""
    return _private_token(os.path.join(directory, TOKEN_FILE))


def load_or_create_operator_token(directory: str) -> str:
    """The operator credential for dashboard actions (file ``dashboard.operator``, mode 0600, owned by this uid;
    a file that other users can read, or that another uid owns, is refused). Always distinct from the viewing
    token."""
    tok = _private_token(os.path.join(directory, OPERATOR_TOKEN_FILE))
    view = os.path.join(directory, TOKEN_FILE)
    if os.path.exists(view) and hmac.compare_digest(tok.encode(), load_or_create_token(directory).encode()):
        raise ValueError(f"{os.path.join(directory, OPERATOR_TOKEN_FILE)} equals the viewing token; the operator "
                         "credential must be distinct")
    return tok


# --------------------------------------------------------------------------------------------------------------
# who owns the connection (loopback TCP): the peer's uid from the kernel's socket tables
# --------------------------------------------------------------------------------------------------------------
def _v6_hex(packed: bytes, port: int) -> str:
    return "".join(f"{struct.unpack('=I', packed[i:i + 4])[0]:08X}" for i in range(0, 16, 4)) + f":{port:04X}"


def _proc_forms(ip: str, port: int) -> List[str]:
    """The spellings of ``ip:port`` in /proc/net/tcp{,6} (IPv4, and IPv4-mapped IPv6 for dual-stack sockets)."""
    a = ipaddress.ip_address(ip)
    if a.version == 4:
        return [f"{struct.unpack('=I', a.packed)[0]:08X}:{port:04X}",
                _v6_hex(ipaddress.IPv6Address("::ffff:" + str(a)).packed, port)]
    out = [_v6_hex(a.packed, port)]
    if a.ipv4_mapped is not None:
        out.append(f"{struct.unpack('=I', a.ipv4_mapped.packed)[0]:08X}:{port:04X}")
    return out


def socket_owner_uid(local_ip: str, local_port: int, remote_ip: str, remote_port: int,
                     tables: Sequence[str] = PROC_TCP_TABLES) -> Optional[int]:
    """uid owning the TCP socket ``local_ip:local_port -> remote_ip:remote_port`` (for a dashboard request: the
    client's side of the connection), from the kernel's socket tables. None when no such socket is listed."""
    loc, rem = set(_proc_forms(local_ip, local_port)), set(_proc_forms(remote_ip, remote_port))
    for table in tables:
        try:
            with open(table) as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if len(fields) > 7 and fields[1].upper() in loc and fields[2].upper() in rem:
                try:
                    return int(fields[7])
                except ValueError:
                    return None
    return None


def service_uids() -> Dict[int, str]:
    """uid -> user name of every CAM Live service user that exists on this machine."""
    out: Dict[int, str] = {}
    for name in SERVICE_USERS.values():
        try:
            out[pwd.getpwnam(name).pw_uid] = name
        except KeyError:
            continue
    return out


def peer_problem(client: Optional[Tuple[str, int]], server: Optional[Tuple[str, int]],
                 refused_uids: Optional[Dict[int, str]] = None,
                 tables: Sequence[str] = PROC_TCP_TABLES) -> str:
    """Why the peer of an operator request may not act ('' = allowed). A request that arrives over a real TCP
    connection must come from a socket whose owner is known and is not a service uid. A request without a socket
    (an in-process test client) has no peer to check."""
    try:
        ipaddress.ip_address(str((server or ("",))[0]))
    except ValueError:
        return ""                                          # no real socket (e.g. the in-process test client)
    try:
        ipaddress.ip_address(str((client or ("",))[0]))
    except ValueError:
        return "the address of this connection is not a network address; operator actions are refused"
    refused = service_uids() if refused_uids is None else refused_uids
    uid = socket_owner_uid(str(client[0]), int(client[1]), str(server[0]), int(server[1]), tables)
    if uid is None:
        return "the user owning this connection could not be determined; operator actions are refused"
    if uid in refused:
        return (f"the connection belongs to service user {refused[uid]} (uid {uid}); service users cannot perform "
                "operator actions")
    return ""


# --------------------------------------------------------------------------------------------------------------
# what the dashboard shows
# --------------------------------------------------------------------------------------------------------------
def _num_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """mean / min / max / n of every numeric field across a list of flat dicts."""
    acc: Dict[str, List[float]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                acc.setdefault(str(k), []).append(float(v))
    return {k: {"n": len(v), "mean": sum(v) / len(v), "min": min(v), "max": max(v), "last": v[-1]}
            for k, v in acc.items()}


def _bounded(n: int, hi: int = 1000) -> int:
    return max(1, min(int(n), hi))


class RunView:
    """Everything the dashboard shows, computed from the files of one run directory."""

    def __init__(self, run_dir: str):
        self.run_dir = os.path.abspath(run_dir)
        self.r = RunReader(self.run_dir)

    def status(self) -> Dict[str, Any]:
        st = self.r.state()
        task = st.get("task", {}) or {}
        lv = st.get("last_verified") or {}
        vo = lv.get("outcome") or {}
        return {
            "run_id": st.get("run_id", ""), "status": st.get("status", "no state"),
            "status_reason": st.get("status_reason", ""), "awaiting_operator": st.get("awaiting_operator", False),
            "objective": {"id": task.get("objective_id", ""), "text": task.get("objective", ""),
                          "mode": task.get("mode", ""), "family": task.get("family", ""),
                          "queued": [((t.get("task") or {}).get("objective_id", "")) for t in st.get("next_tasks", [])
                                     if isinstance(t, dict)]},
            # only the runtime's record of a signature-checked verifier outcome is shown as verified progress
            "verified": {"status": vo.get("status"), "value": vo.get("value"), "target": vo.get("target"),
                         "uncertainty": vo.get("uncertainty"), "excluded": vo.get("excluded"),
                         "notes": vo.get("notes"), "secondary": vo.get("secondary"), "evaluated_at": vo.get("evaluated_at"),
                         "record": lv.get("rid"), "refused": st.get("refused_verified", 0)},
            "activity": st.get("activity", {}), "wake": st.get("wake", {}),
            "resources": st.get("limits", {}), "counters": st.get("counters", {}),
            "lifetime": st.get("lifetime", {}),
            "checkpoint": (st.get("committed") or {}).get("ckpt"), "exports": st.get("exports", [])[-5:],
            "pending_outcomes": len(((st.get("live") or {}).get("pending") or {}).get("items", [])),
            "pending_approvals": sum(1 for a in (st.get("approvals") or {}).values()
                                     if isinstance(a, dict) and a.get("status") == "pending"),
            "updated_at": st.get("updated_at"),
            # INT3B: the shared-GPU lease (holding / waiting), an ended objective's settling, accepted risks
            "gpu": st.get("gpu"), "settling": st.get("settling"),
            "accepted_risks": (self._run_meta().get("accepted_risks") or []),
        }

    def _run_meta(self) -> Dict[str, Any]:
        d = read_json(os.path.join(self.run_dir, "run.json"), {})
        return d if isinstance(d, dict) else {}

    def experiments(self, n: int = 20) -> List[Dict[str, Any]]:
        """Registered experiments. ``actual`` lists only outcomes the runtime itself measured for the experiment
        (``experiment_outcome`` records); whatever the learner's own record says about outcomes is never shown as
        an outcome, and its ``supported`` verdict is labelled as the learner's own assessment."""
        latest: Dict[str, Dict[str, Any]] = {}
        measured: Dict[str, List[Dict[str, Any]]] = {}
        for e in self.r.records(RunFiles.EXPERIMENTS, types=["experiment", "experiment_outcome"]):
            d = e.get("data") or {}
            if not isinstance(d, dict):
                continue
            if e["type"] == "experiment_outcome":
                o = d.get("outcome") or {}
                if isinstance(o, dict):
                    measured.setdefault(str(d.get("experiment_id", "")), []).append(
                        {"record": e["rid"], "decision_id": d.get("decision_id"), "value": o.get("value"),
                         "status": o.get("status"), "verified": o.get("verified") is True,
                         "outcome_record": o.get("outcome_rid")})
                continue
            preds = d.get("predicted") if isinstance(d.get("predicted"), list) else []
            latest[str(d.get("experiment_id", ""))] = {
                "record": e["rid"], "ts": e["ts"], "experiment_id": d.get("experiment_id"),
                "question": d.get("question"), "hypothesis": d.get("hypothesis"), "status": d.get("status"),
                "intervention": d.get("intervention"), "assignment": d.get("assignment"),
                "predicted": [{"mean": (p.get("dist") or {}).get("mean"), "std": (p.get("dist") or {}).get("std"),
                               "metric": (p.get("query") or {}).get("metric")} for p in preds if isinstance(p, dict)],
                "supported_according_to_learner": d.get("supported"), "cost": d.get("cost")}
        rows = list(latest.values())[-_bounded(n):]
        for row in rows:
            row["actual"] = measured.get(str(row["experiment_id"]), [])
        return rows

    def predictions(self, n: int = 50) -> Dict[str, Any]:
        res = self.r.records(RunFiles.PREDICTIONS, types=["resolution"])
        rows = [{"record": x["rid"], "ts": x["ts"], "prediction_id": x["data"].get("prediction_id"),
                 "metric": (x["data"].get("outcome") or {}).get("metric"),
                 "predicted_mean": x["data"].get("predicted_mean"), "predicted_std": x["data"].get("predicted_std"),
                 "actual": (x["data"].get("outcome") or {}).get("value"),
                 "verified": (x["data"].get("outcome") or {}).get("verified") is True} for x in res[-_bounded(n):]]
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
        ev = self.r.records(RunFiles.EVENTS, types=["internal_rewards"], tail=_bounded(n, 5000))
        return {"n_reports": len(ev), "last": ev[-1]["data"] if ev else None,
                "stats": _num_stats([e["data"] for e in ev]),
                "note": "as reported by the learner; the environment and the runtime produce no reward"}

    def learning(self, n: int = 20) -> Dict[str, Any]:
        ups = self.r.records(RunFiles.LEARNING)
        rows = [{"record": u["rid"], "ts": u["ts"], **{k: u["data"].get(k) for k in (
            "update_id", "trigger", "timescale", "losses", "param_groups", "grad_norms", "weight_delta",
            "prediction_change", "n_experiences", "checkpoint_id", "model_version", "notes")},
            "behavioural_change": behavioural_change(u["data"])} for u in ups[-_bounded(n):]]
        return {"n_updates": len(ups), "n_with_behavioural_change": sum(1 for u in ups if behavioural_change(u["data"])),
                "recent": rows}

    def meta(self, n: int = 20) -> Dict[str, Any]:
        ms = self.r.records(RunFiles.META)
        skipped: Dict[str, int] = {}
        for m in ms:
            if not m["data"].get("applied"):
                k = str(m["data"].get("skipped_reason") or "unspecified")
                skipped[k] = skipped.get(k, 0) + 1
        last = ms[-1]["data"] if ms else {}
        return {"n_updates": len(ms), "n_applied": sum(1 for m in ms if m["data"].get("applied") is True),
                "skipped_reasons": skipped, "bootstrap_mix": last.get("bootstrap_mix"),
                "reward_version": last.get("reward_version_after"),
                "recent": [{"record": m["rid"], "ts": m["ts"], **m["data"]} for m in ms[-_bounded(n):]]}

    def approvals(self) -> Dict[str, Any]:
        st = self.r.state()
        rows = [{k: a.get(k) for k in ("approval_id", "status", "operation", "preview", "reasons", "queued_at",
                                       "resolved_at", "result_status")}
                for a in (st.get("approvals") or {}).values() if isinstance(a, dict)]
        return {"pending": [a for a in rows if a["status"] == "pending"],
                "resolved": [a for a in rows if a["status"] != "pending"][-20:]}

    def obstacles(self, n: int = 20) -> List[Dict[str, Any]]:
        return [{"record": o["rid"], **o["data"]["obstacle"]} for o in self.r.obstacles()[-_bounded(n):]]

    def health(self) -> Dict[str, Any]:
        st = self.r.state()
        pid = read_pid(self.run_dir)
        du = shutil.disk_usage(self.run_dir)
        out: Dict[str, Any] = {
            "process": {"pid": pid, "alive": pid_alive(pid)},
            "state_file_age_s": None, "disk_free_gb": round(du.free / 2**30, 2),
            # defence in depth: error text is redacted again before it is served (the runtime redacts it too)
            "errors_recent": Redactor()(list(st.get("errors") or [])[-5:]),
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
        return rows[-_bounded(n):]


# --------------------------------------------------------------------------------------------------------------
# the application
# --------------------------------------------------------------------------------------------------------------
def create_app(run_dir: str, token: str, policy: Optional[PolicyEngine] = None, tasks_dir: Optional[str] = None,
               operator: str = "operator", operator_token: Optional[str] = None) -> FastAPI:
    """``token`` opens the views; ``operator_token`` (distinct from it) is needed in addition for every operator
    action. Without ``operator_token`` the dashboard is view-only."""
    if not token or len(token) < 16:
        raise ValueError("a dashboard token of at least 16 characters is required")
    if operator_token is not None:
        if len(operator_token) < 16:
            raise ValueError("the operator credential must have at least 16 characters")
        if hmac.compare_digest(operator_token.encode(), token.encode()):
            raise ValueError("the operator credential must be distinct from the viewing token")
    own = service_uids().get(os.geteuid())
    actions_off = ("operator actions are disabled: no operator credential is configured for this dashboard"
                   if operator_token is None else
                   f"operator actions are disabled: this dashboard runs as service user {own}" if own else "")
    view = RunView(run_dir)
    ctl = OperatorControl(view.run_dir, policy=policy, by=operator)
    tasks_dir = os.path.abspath(tasks_dir or default_tasks_dir())
    app = FastAPI(title="CAM Live", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            resp.headers[k] = v
        return resp

    def auth(request: Request) -> None:
        h = request.headers.get("authorization", "")
        given = h[7:] if h.lower().startswith("bearer ") else (request.query_params.get("token")
                                                               or request.cookies.get(COOKIE) or "")
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="missing or wrong token")

    def operator_auth(request: Request) -> None:
        """Every operator action: operator credential (header only) + CSRF double-submit + same origin + a peer
        that is not a service user. Runs after the viewing-token check."""
        if actions_off:
            raise HTTPException(status_code=403, detail=actions_off)
        given = request.headers.get(OPERATOR_HEADER, "")
        if not given or not hmac.compare_digest(given.encode(), str(operator_token).encode()):
            raise HTTPException(status_code=401, detail=f"missing or wrong operator credential ({OPERATOR_HEADER})")
        c, hdr = request.cookies.get(CSRF_COOKIE, ""), request.headers.get(CSRF_HEADER, "")
        if not c or not hdr or not hmac.compare_digest(c.encode(), hdr.encode()):
            raise HTTPException(status_code=403, detail="missing or mismatched CSRF token")
        site = request.headers.get("sec-fetch-site", "").lower()
        if site and site not in ("same-origin", "none"):
            raise HTTPException(status_code=403, detail=f"cross-site request refused (Sec-Fetch-Site: {site})")
        origin = request.headers.get("origin")
        if origin is not None:
            o = urlsplit(origin)
            if origin == "null" or o.netloc != request.headers.get("host", "") or o.scheme not in ("http", "https"):
                raise HTTPException(status_code=403, detail="cross-origin request refused")
        why = peer_problem(request.client and (request.client.host, request.client.port), request.scope.get("server"))
        if why:
            raise HTTPException(status_code=403, detail=why)

    guard = [Depends(auth)]
    act = [Depends(auth), Depends(operator_auth)]

    @app.get("/", response_class=HTMLResponse, dependencies=guard)
    def page() -> HTMLResponse:
        resp = HTMLResponse(PAGE)
        resp.set_cookie(COOKIE, token, httponly=True, samesite="strict")
        resp.set_cookie(CSRF_COOKIE, secrets.token_urlsafe(32), httponly=False, samesite="strict")
        return resp

    @app.get("/app.js", dependencies=guard)
    def app_js() -> Response:
        return Response(APP_JS, media_type="text/javascript; charset=utf-8")

    @app.get("/app.css", dependencies=guard)
    def app_css() -> Response:
        return Response(APP_CSS, media_type="text/css; charset=utf-8")

    @app.get("/api/operator", dependencies=guard)
    def operator_info() -> Dict[str, Any]:
        return {"actions_enabled": not actions_off, "reason": actions_off, "header": OPERATOR_HEADER,
                "emergency_stop_reset": policy is not None and callable(getattr(policy, "reset_stop", None))}

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

    # ---- operator actions ------------------------------------------------------------------------------------
    @app.post("/api/control/pause", dependencies=act)
    def pause(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        return {"command_id": ctl.pause(str(body.get("reason", ""))[:200]), "queued": True}

    @app.post("/api/control/resume", dependencies=act)
    def resume() -> Dict[str, Any]:
        return {"command_id": ctl.resume(), "queued": True}

    @app.post("/api/control/stop", dependencies=act)
    def stop(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        return {"command_id": ctl.stop(str(body.get("reason", ""))[:200], emergency=body.get("emergency") is True),
                "queued": True}

    @app.post("/api/control/reset-stop", dependencies=act)
    def reset_stop() -> Dict[str, Any]:
        fn = getattr(policy, "reset_stop", None) if policy is not None else None
        if not callable(fn):
            raise HTTPException(status_code=409, detail="no handle on the authorization service: reset the emergency "
                                                        "stop with the operator tools of the authorization service")
        return {"reset": bool(fn(operator)), "by": operator}

    # Approval and task-file routes answer 404 for an unknown approval / a file outside the tasks directory to any
    # viewer (who can list both anyway); everything that acts is checked by operator_auth first.
    def _known(approval_id: str) -> None:
        if approval_id not in (view.r.state().get("approvals") or {}):
            raise HTTPException(status_code=404, detail="unknown approval")

    @app.post("/api/approvals/{approval_id}/approve", dependencies=guard)
    def approve(approval_id: str, request: Request, body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        _known(approval_id)
        operator_auth(request)
        return {"command_id": ctl.approve(approval_id, str(body.get("note", ""))[:200]), "queued": True}

    @app.post("/api/approvals/{approval_id}/reject", dependencies=guard)
    def reject(approval_id: str, request: Request, body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        _known(approval_id)
        operator_auth(request)
        return {"command_id": ctl.reject(approval_id, str(body.get("note", ""))[:200]), "queued": True}

    @app.post("/api/control/export-checkpoint", dependencies=act)
    def export_checkpoint() -> Dict[str, Any]:
        return {"command_id": ctl.export_checkpoint(), "queued": True}     # destination: <run_dir>/exports/

    @app.post("/api/control/assign-objective", dependencies=guard)
    def assign(request: Request, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        name = os.path.basename(str(body.get("task", "")))
        path = os.path.join(tasks_dir, name)
        if not name or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="task file not found in the tasks directory")
        operator_auth(request)
        try:
            load_task(path, for_run=True)
            return {"command_id": ctl.assign_objective(path), "queued": True}
        except TaskValidationError as e:
            raise HTTPException(status_code=422, detail=e.problems)

    return app


def serve(run_dir: str, token: str, host: str = "127.0.0.1", port: int = 8787, policy: Optional[PolicyEngine] = None,
          tasks_dir: Optional[str] = None, operator_token: Optional[str] = None) -> None:
    if host not in LOOPBACK:
        raise ValueError(f"the dashboard binds to the loopback interface only (got {host!r}); reach it through an "
                         "SSH tunnel or an authenticating reverse proxy")
    import uvicorn
    # proxy headers off: the peer address (and so the uid check of operator actions) is the socket's, never a
    # client-supplied X-Forwarded-For
    uvicorn.run(create_app(run_dir, token, policy, tasks_dir, operator_token=operator_token), host=host, port=port,
                log_level="warning", proxy_headers=False, server_header=False)


# --------------------------------------------------------------------------------------------------------------
# page, script and style (static; no data is ever interpolated into them)
# --------------------------------------------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CAM Live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<link rel="stylesheet" href="/app.css">
<script src="/app.js" defer></script>
</head><body>
<header><b>CAM Live</b><span id="run" class="muted"></span><span id="st" class="pill"></span>
<span class="grow"></span>
<input id="op-cred" type="password" autocomplete="off" title="operator credential (needed for every action; kept in this page only)" aria-label="operator credential (needed for every action; kept in this page only)" size="18">
<button id="b-pause">Pause</button><button id="b-resume">Resume</button>
<input id="stop-note" title="audit note (not shown to CAM)" aria-label="audit note (not shown to CAM)" size="26" maxlength="200">
<button id="b-stop" class="warn">Stop</button><button id="b-estop" class="warn">Emergency stop</button>
<button id="b-reset" class="warn">Reset emergency stop</button>
<button id="b-export">Export checkpoint</button><span id="op-out" class="muted"></span></header>
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
<section class="wide"><h2>Logs
<select id="log-name"><option>events.jsonl</option><option>decisions.jsonl</option><option>experiments.jsonl</option>
<option>predictions.jsonl</option><option>learning_updates.jsonl</option><option>meta_updates.jsonl</option>
<option>obstacles.jsonl</option><option value="journal">journal.md</option></select></h2><pre id="log"></pre></section>
</main>
</body></html>
"""

APP_CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:0;background:#f5f6f8;color:#1c2330}
header{background:#1c2330;color:#fff;padding:10px 18px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
header b{font-size:16px} .grow{flex:1}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px;padding:12px}
section{background:#fff;border:1px solid #d9dde4;border-radius:6px;padding:10px 12px;overflow:auto;max-height:420px}
section.wide{grid-column:1/-1;max-height:520px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;margin:0 0 6px;color:#52607a}
table{border-collapse:collapse;width:100%} td,th{text-align:left;padding:2px 8px 2px 0;vertical-align:top;border-bottom:1px solid #eef0f4}
pre{white-space:pre-wrap;word-break:break-word;margin:0;font:12px/1.4 ui-monospace,monospace}
button{font:inherit;padding:4px 10px;border:1px solid #8a94a6;border-radius:4px;background:#fff;cursor:pointer}
button.warn{border-color:#b3261e;color:#b3261e} input,select{font:inherit;padding:3px 6px}
.pill{padding:1px 8px;border-radius:10px;background:#e4e8ef;color:#1c2330} .muted{color:#6b7689}
meter{width:100%;height:10px;display:block}
"""

# The client builds every element with createElement and fills it with textContent: nothing from the server is
# ever parsed as markup. Values are rendered as text whatever their JSON type.
APP_JS = r"""'use strict';
(function () {
  const $ = id => document.getElementById(id);
  let operatorCredential = '';

  function txt(x) {
    if (x === null || x === undefined) return '—';
    if (typeof x === 'number') return isFinite(x) ? String(+x.toPrecision(5)) : String(x);
    if (typeof x === 'string') return x;
    if (typeof x === 'boolean') return String(x);
    try { return JSON.stringify(x); } catch (e) { return String(x); }
  }
  function list(x) { return Array.isArray(x) ? x : (x === null || x === undefined ? [] : [x]); }
  function obj(x) { return (x !== null && typeof x === 'object' && !Array.isArray(x)) ? x : {}; }
  function num(x) { return typeof x === 'number' && isFinite(x); }
  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = txt(text);
    if (cls) e.className = cls;
    return e;
  }
  function line(tag, parts, cls) {           // parts: strings, or [text, 'b'|'pill'|'muted']
    const p = el(tag, undefined, cls);
    for (const part of parts) {
      if (Array.isArray(part)) {
        const s = el(part[1] === 'b' ? 'b' : 'span', part[0], part[1] === 'b' ? '' : part[1]);
        p.appendChild(s);
      } else {
        p.appendChild(document.createTextNode(txt(part)));
      }
    }
    return p;
  }
  function fill(node, children) {
    node.replaceChildren(...children.filter(c => c));
  }
  function table(rows, cols) {
    const t = el('table');
    const head = el('tr');
    for (const c of cols) head.appendChild(el('th', c[0]));
    t.appendChild(head);
    for (const r of list(rows)) {
      const tr = el('tr');
      for (const c of cols) {
        let v;
        try { v = c[1](obj(r)); } catch (e) { v = '?'; }
        tr.appendChild(el('td', v));
      }
      t.appendChild(tr);
    }
    return t;
  }
  function csrf() {
    for (const c of document.cookie.split(';')) {
      const kv = c.trim();
      if (kv.startsWith('cam_live_csrf=')) return decodeURIComponent(kv.slice('cam_live_csrf='.length));
    }
    return '';
  }
  function note(s) { $('op-out').textContent = s; }
  async function get(p) {
    const r = await fetch(p, {credentials: 'same-origin'});
    if (!r.ok) throw new Error(p + ' ' + r.status);
    return p.endsWith('journal') ? r.text() : r.json();
  }
  async function post(p, body) {
    if (!operatorCredential) { note('enter the operator credential first'); return null; }
    const r = await fetch(p, {method: 'POST', credentials: 'same-origin', headers: {
      'content-type': 'application/json', 'X-CAM-Operator': operatorCredential, 'X-CSRF-Token': csrf()},
      body: JSON.stringify(body || {})});
    let j = {};
    try { j = await r.json(); } catch (e) { j = {}; }
    note(r.ok ? 'accepted ' + txt(j.command_id || j.reset || '') : 'refused (' + r.status + '): ' + txt(j.detail));
    refresh();
    return j;
  }

  async function refresh() {
    try {
      const s = obj(await get('/api/status'));
      $('run').textContent = txt(s.run_id);
      $('st').textContent = txt(s.status) + (s.status_reason ? ' — ' + txt(s.status_reason) : '');
      const o = obj(s.objective), v = obj(s.verified);
      const sec = list(v.secondary).map(q => [obj(q).name, obj(q).value]);
      fill($('objective'), [
        line('p', [[o.id, 'b'], ' ', [o.mode, 'pill']]),
        el('p', o.text || 'no objective assigned'),
        line('p', ['Verified: ', [v.value, 'b'], ' of ', v.target, ' (', v.status || 'no signed verification yet',
                   ', uncertainty ', v.uncertainty, ')']),
        el('p', v.notes || '', 'muted'),
        el('pre', 'excluded ' + txt(v.excluded || {}) + '\nsecondary ' + txt(sec)),
        list(o.queued).length ? el('p', 'Queued objectives: ' + list(o.queued).map(txt).join(', ')) : null]);
      const h = obj(await get('/api/health'));
      const act = obj(s.activity), ck = obj(s.checkpoint), pr = obj(h.process), vl = h.verifier_log;
      fill($('activity'), [
        line('p', [[act.phase || '', 'b'], ' ', act.detail || '']),
        el('p', 'Next wake: ' + txt(s.wake || {})),
        el('p', 'Checkpoint: ' + (s.checkpoint ? txt(ck.id) + ' (generation ' + txt(ck.generation) + ')' : 'none')),
        el('p', 'Pending outcomes: ' + txt(s.pending_outcomes) + ' · process ' + (pr.alive === true ? 'alive' : 'not running') +
           ' (pid ' + txt(pr.pid) + ') · restarts ' + txt(h.restarts) + ' · disk free ' + txt(h.disk_free_gb) +
           ' GB · verifier log ' + (vl ? (obj(vl).intact === true ? 'intact' : 'PROBLEM: ' + txt(obj(vl).problem)) : 'n/a')),
        s.gpu ? el('p', 'Shared GPU lock: ' + (obj(s.gpu).held ? 'held ' + txt(obj(s.gpu).held_for_s) + ' s' : 'not held') +
           (obj(s.gpu).waiting_for_s != null ? ' · WAITING ' + txt(obj(s.gpu).waiting_for_s) + ' s' : '') +
           ' · holds <= ' + txt(obj(s.gpu).hold_s) + ' s · waited ' + txt(obj(s.gpu).waited_s_total) + ' s in total') : null,
        s.settling ? el('p', 'Settling the ended objective: ' + txt(s.settling)) : null,
        list(s.accepted_risks).length ? el('p', 'ACCEPTED RISKS: ' + list(s.accepted_risks).map(r => txt(obj(r).check) + ' — ' +
           txt(obj(r).finding) + ' (' + txt(obj(r).accepted_by) + ')').join('; '), 'b') : null,
        el('pre', txt(list(h.errors_recent)))]);
      fill($('resources'), Object.entries(obj(s.resources)).map(([k, r0]) => {
        const r = obj(r0), p = el('p', k + ': ' + txt(r.used) + (num(r.limit) && r.limit > 0 ? ' of ' + txt(r.limit) : ' (no limit set)'));
        if (num(r.limit) && r.limit > 0 && num(r.used)) {
          const m = document.createElement('meter');
          m.min = 0; m.max = r.limit; m.value = Math.min(r.used, r.limit);
          p.appendChild(m);
        }
        return p;
      }));
      const a = obj(await get('/api/approvals'));
      const pend = list(a.pending).map(p0 => {
        const p = obj(p0), box = el('div');
        box.appendChild(line('p', [[p.approval_id, 'b'], ' ', p.operation, ' ', [list(p.reasons).map(txt).join('; '), 'muted']]));
        box.appendChild(el('pre', p.preview));
        const row = el('p'), input = document.createElement('input');
        input.maxLength = 200; input.title = 'audit note (not shown to CAM)';
        input.setAttribute('aria-label', 'audit note (not shown to CAM)');
        const ok = el('button', 'Approve'), no = el('button', 'Reject', 'warn');
        const id = encodeURIComponent(txt(p.approval_id));
        ok.addEventListener('click', () => post('/api/approvals/' + id + '/approve', {note: input.value}));
        no.addEventListener('click', () => post('/api/approvals/' + id + '/reject', {note: input.value}));
        row.append(input, ' ', ok, ' ', no);
        box.appendChild(row);
        return box;
      });
      fill($('approvals'), [pend.length ? null : el('p', 'none pending', 'muted'), ...pend,
        list(a.resolved).length ? table(a.resolved, [['id', x => x.approval_id], ['operation', x => x.operation],
                                                     ['status', x => x.status]]) : null]);
      const e = list(await get('/api/experiments')).slice().reverse();
      fill($('experiments'), [table(e, [['question', x => x.question], ['hypothesis', x => x.hypothesis],
        ['predicted', x => list(x.predicted).map(p => txt(obj(p).mean) + '±' + txt(obj(p).std)).join(', ')],
        ['measured by the runtime', x => list(x.actual).map(m => txt(obj(m).value) + (obj(m).verified === true ? ' (verifier)' : '')).join(', ')],
        ["learner's own assessment", x => x.supported_according_to_learner], ['status', x => x.status]])]);
      const p = obj(await get('/api/predictions'));
      fill($('predictions'), [el('p', txt(p.n_resolved) + ' resolved of ' + txt(p.n_registered) + ' registered'),
        table(list(p.resolved).slice().reverse(), [['metric', x => x.metric],
          ['predicted', x => txt(x.predicted_mean) + ' ± ' + txt(x.predicted_std)], ['actual', x => x.actual],
          ['verified', x => x.verified === true]])]);
      const r = obj(await get('/api/rewards')), c = obj(await get('/api/calibration'));
      fill($('rewards'), [el('p', r.note, 'muted'), el('pre', JSON.stringify(r.stats, null, 1)), el('p', c.note, 'muted'),
        el('pre', JSON.stringify(c, null, 1))]);
      const l = obj(await get('/api/learning'));
      fill($('learning'), [el('p', txt(l.n_updates) + ' updates, ' + txt(l.n_with_behavioural_change) + ' with a measured behavioural change'),
        table(list(l.recent).slice().reverse(), [['timescale', x => x.timescale], ['trigger', x => x.trigger],
          ['groups', x => list(x.param_groups).map(txt).join(',')], ['weight change', x => JSON.stringify(x.weight_delta)],
          ['prediction change', x => JSON.stringify(x.prediction_change)], ['losses', x => JSON.stringify(x.losses)]])]);
      const m = obj(await get('/api/meta'));
      fill($('meta'), [el('p', txt(m.n_applied) + ' applied of ' + txt(m.n_updates) + ' · bootstrap mix ' + txt(m.bootstrap_mix) +
                          ' · reward version ' + txt(m.reward_version)),
        el('pre', 'skipped: ' + JSON.stringify(m.skipped_reasons, null, 1)),
        table(list(m.recent).slice().reverse(), [['applied', x => x.applied], ['reason', x => x.skipped_reason],
          ['independent outcomes', x => x.n_independent_outcomes], ['meta-gradient norm', x => x.meta_grad_norm],
          ['estimator', x => x.estimator]])]);
      const ob = list(await get('/api/obstacles')).slice().reverse();
      fill($('obstacles'), [table(ob, [['category', x => x.category], ['failure', x => x.failure_class],
        ['description', x => x.description], ['what followed', x => x.outcome],
        ['learning updates', x => list(x.learning_update_ids).length]])]);
      const n = $('log-name').value;
      $('log').textContent = n === 'journal' ? txt(await get('/api/journal')) :
        list(await get('/api/logs/' + encodeURIComponent(n) + '?n=60')).slice().reverse().map(x => JSON.stringify(x)).join('\n');
    } catch (err) {
      $('st').textContent = 'error: ' + txt(err && err.message);
    }
  }

  function start() {
    if (location.search) history.replaceState(null, '', location.pathname);    // no token left in the address bar
    $('op-cred').addEventListener('change', ev => {
      operatorCredential = ev.target.value;
      ev.target.value = '';
      note(operatorCredential ? 'operator credential set for this page' : '');
    });
    $('b-pause').addEventListener('click', () => post('/api/control/pause', {reason: $('stop-note').value}));
    $('b-resume').addEventListener('click', () => post('/api/control/resume'));
    $('b-stop').addEventListener('click', () => { if (confirm('Stop the run?')) post('/api/control/stop', {reason: $('stop-note').value}); });
    $('b-estop').addEventListener('click', () => { if (confirm('Emergency stop?')) post('/api/control/stop', {reason: $('stop-note').value, emergency: true}); });
    $('b-reset').addEventListener('click', () => { if (confirm('Reset the emergency stop at the authorization service?')) post('/api/control/reset-stop'); });
    $('b-export').addEventListener('click', () => post('/api/control/export-checkpoint'));
    $('b-assign').addEventListener('click', async () => {
      const j = await post('/api/control/assign-objective', {task: $('task-file').value});
      $('assign-out').textContent = txt(j);
    });
    $('log-name').addEventListener('change', refresh);
    get('/api/tasks').then(t => {
      const sel = $('task-file');
      sel.replaceChildren();
      for (const x of list(t)) sel.add(new Option(txt(x), txt(x)));
    }).catch(() => {});
    refresh();
    setInterval(refresh, 5000);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
"""
