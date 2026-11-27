#!/usr/bin/env python3
"""Staging rehearsal of the planning phase (OP-11 prerequisite, INT3B). STAGING ONLY: nothing here contacts
discord.com or any other external service.

    planning_rehearsal.py prepare --root R [--config C] [--hold-s 840]   run directory, staging token, policy, config
    planning_rehearsal.py serve   --root R            staging Discord (127.0.0.1:18471) + neutral web (127.0.0.1:18472)
    planning_rehearsal.py holds   --root R --decisions N [--max-holds H]
                                                     `cam-live run` (first hold) / `cam-live resume` (later holds),
                                                     one learner process per bounded hold of the shared GPU lock
    planning_rehearsal.py check   --root R [--out F]  what the run recorded (counts; verbatim excerpts only)

The task is ``tasks/staging/discord_100_planning_staging.yaml`` (the operator's objective verbatim). The operator's
side of the rehearsal is exactly what an operator does: start the staging services, enter a (staging) credential,
register the local services, start / resume the runtime. This tool never answers an approval or an operator request
and never writes anything into the run that CAM reads: it adds no niche, strategy, plan, example or hint anywhere.
The neutral web serves generic pages without a topic.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

TASK = os.path.join(PROJECT, "tasks", "staging", "discord_100_planning_staging.yaml")
LIVE_TASK = os.path.join(PROJECT, "tasks", "discord_100_planning.yaml")
DISCORD_PORT, WEB_PORT = 18471, 18472
HOST = "127.0.0.1"
TOKEN_FILE = "staging_discord.token"
SERVE_FILE = "serve.json"
DISCORD_LOG = "staging_discord_requests.jsonl"
WEB_LOG = "staging_web_requests.jsonl"
POLICY_FILE = "policy_planning_staging.yaml"
CONFIG_FILE = "config.yaml"
LOOPBACK_OWNED = '["127.0.0.1", "localhost", "[::1]", "channel:staging:*", "account:staging:*"]'
NOT_OWNED = '["channel:staging:*", "account:staging:*"]'


def run_dir(root: str) -> str:
    return os.path.join(os.path.abspath(root), "run")


# ----------------------------------------------------------------------------------------------------------------
# neutral web: generic pages, no topic
# ----------------------------------------------------------------------------------------------------------------
_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head><body>
<h1>{title}</h1>{body}
<p><a href="/">Index</a> | <a href="/page/1">Page 1</a> | <a href="/page/2">Page 2</a> | <a href="/page/3">Page 3</a> |
<a href="/search">Search</a> | <a href="/form">Form</a></p></body></html>"""
_NOTE = ("<p>This is a local staging server that stands in for the public web during a staging rehearsal. "
         "It serves a few neutral test pages and contains no information about any topic.</p>")


def web_page(path: str, query: Dict[str, List[str]], posted: Optional[Dict[str, List[str]]] = None) -> str:
    if path in ("", "/"):
        return _PAGE.format(title="Staging web", body=_NOTE)
    if path.startswith("/page/"):
        n = path.rsplit("/", 1)[-1]
        rows = "".join(f"<tr><td>{i}</td><td>{i * 10}</td></tr>" for i in range(1, 4))
        return _PAGE.format(title=f"Test page {html.escape(n)}",
                            body=_NOTE + f"<p>Neutral test page {html.escape(n)}.</p>"
                            "<table><tr><th>row</th><th>value</th></tr>" + rows + "</table>")
    if path == "/search":
        q = (query.get("q") or [""])[0]
        res = "<p>No results: the staging web has no indexed content.</p>" if q else ""
        return _PAGE.format(title="Search", body=_NOTE + '<form action="/search" method="get"><label>Query '
                            '<input name="q" type="text"></label><button type="submit">Search</button></form>' + res)
    if path == "/form":
        got = "<p>Form received.</p>" if posted is not None else ""
        return _PAGE.format(title="Form", body=_NOTE + '<form action="/form" method="post"><label>Text '
                            '<input name="text" type="text"></label><button type="submit">Send</button></form>' + got)
    return ""


class _Log:
    def __init__(self, path: str):
        self.path, self.lock = path, threading.Lock()

    def write(self, rec: Dict[str, Any]) -> None:
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def serve_web(log: _Log, port: int = WEB_PORT) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def _reply(self, posted: Optional[Dict[str, List[str]]] = None) -> None:
            u = urlsplit(self.path)
            body = web_page(u.path, parse_qs(u.query), posted)
            log.write({"ts": time.time(), "method": self.command, "path": u.path, "query_keys": sorted(parse_qs(u.query)),
                       "status": 200 if body else 404})
            raw = (body or "<!doctype html><title>Not found</title><p>Not found.</p>").encode()
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            self._reply()

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            self._reply(parse_qs(self.rfile.read(n).decode(errors="replace")) if n else {})

        def log_message(self, *a: Any) -> None:
            return

    srv = ThreadingHTTPServer((HOST, port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True, name="staging-web").start()
    return srv


def cmd_serve(a: argparse.Namespace) -> int:
    from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer
    root = os.path.abspath(a.root)
    with open(os.path.join(root, TOKEN_FILE)) as f:
        token = f.read().strip()
    state = StagingDiscord(token=token, guild_name="staging guild")
    dlog = _Log(os.path.join(root, DISCORD_LOG))
    inner = state.handle

    def logged(method: str, url: str, headers: Dict[str, str], body: bytes) -> Any:
        out = inner(method, url, headers, body)
        dlog.write({"ts": time.time(), "method": method, "path": urlsplit(url).path, "status": out[0]})
        return out
    state.handle = logged                       # every request the staging Discord receives, method + path only
    dsrv = StagingDiscordServer(state, HOST, DISCORD_PORT)
    wsrv = serve_web(_Log(os.path.join(root, WEB_LOG)))
    info = {"pid": os.getpid(), "discord_base_url": dsrv.base_url, "guild_id": state.guild_id,
            "application_id": state.application["id"], "web": f"http://{HOST}:{WEB_PORT}/", "started": time.time()}
    with open(os.path.join(root, SERVE_FILE), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.wait(5.0):
        with open(os.path.join(root, "staging_discord_state.json"), "w") as f:
            json.dump({"ts": time.time(), "members": len(state.members), "channels": len(state.channels),
                       "messages": sum(len(v) for v in state.messages.values()), "invites": len(state.invites),
                       "roles": len(state.roles), "events": len(state.events), "audit_log": len(state.audit_log)}, f)
    dsrv.close() if hasattr(dsrv, "close") else None
    wsrv.shutdown()
    return 0


# ----------------------------------------------------------------------------------------------------------------
# prepare: run directory, staging credential, local services, policy, configuration
# ----------------------------------------------------------------------------------------------------------------
def staging_policy(dest: str) -> str:
    """The run policy with loopback removed from the staging mode's owned / pre-authorized destinations: the neutral
    web is then a non-owned site (navigation allowed, input queued for approval), as the public web is for the live
    planning task; Discord writes and publications need approval because nothing pre-authorizes the guild."""
    from cam_live.harness.auth.schema import run_policy_path
    src = open(run_policy_path()).read()
    for key in ("owned", "preauthorized"):
        old = f"    {key}: {LOOPBACK_OWNED}"
        if src.count(old) != 1:
            raise RuntimeError(f"the run policy's staging {key} line was not found exactly once")
        src = src.replace(old, f"    {key}: {NOT_OWNED}     # planning rehearsal: loopback is not owned")
    with open(dest, "w") as f:
        f.write("# Generated by tools/planning_rehearsal.py from " + run_policy_path() + "\n" + src)
    return dest


def cmd_prepare(a: argparse.Namespace) -> int:
    import yaml
    from cam_live.harness import credentials as cred
    from cam_live.harness.auth.policy import PolicyService
    from cam_live.runtime.tasks import load_task
    root = os.path.abspath(a.root)
    R = run_dir(root)
    os.makedirs(R, exist_ok=True)
    task = load_task(TASK, for_run=True)
    live = load_task(LIVE_TASK, for_run=False)
    if task.objective != live.objective:
        raise SystemExit("the staging task's objective differs from the live planning task's")
    tok_path = os.path.join(root, TOKEN_FILE)
    if not os.path.exists(tok_path):
        fd = os.open(tok_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, ("staging-" + secrets.token_hex(32)).encode())
        os.close(fd)
    with open(tok_path) as f:
        token = f.read().strip()
    cred.store(R, "discord_bot_token", token, dest=[f"{HOST}:{DISCORD_PORT}"], note="staging rehearsal token")
    policy = staging_policy(os.path.join(root, POLICY_FILE))
    ps = PolicyService(R, policy)
    for port, what in ((WEB_PORT, "neutral staging web (stand-in for the public web)"),
                       (DISCORD_PORT, "staging Discord service")):
        ps.register_local_service(f"{HOST}:{port}", "operator", what)
    with open(os.path.join(PROJECT, "configs", "cam_live.yaml")) as f:
        cfg = yaml.safe_load(f)
    if a.config:
        with open(a.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg["home"] = os.path.join(root, "home")
    rt = dict(cfg.get("runtime") or {})
    rt.update({"policy_file": policy, "gpu_lock": "auto", "gpu_hold_s": float(a.hold_s)})
    cfg["runtime"] = rt
    with open(os.path.join(root, CONFIG_FILE), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    out = {"run_dir": R, "task": TASK, "task_sha256": hashlib.sha256(open(TASK, "rb").read()).hexdigest(),
           "config": os.path.join(root, CONFIG_FILE), "policy": policy, "local_services": ps.local_services(),
           "credential": "discord_bot_token (staging token, bound to the staging Discord address)"}
    print(json.dumps(out, indent=1))
    return 0


# ----------------------------------------------------------------------------------------------------------------
# holds: one learner process per bounded hold of the GPU lock
# ----------------------------------------------------------------------------------------------------------------
def _state(R: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(R, "state.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    try:                                        # a zombie child of this process is not alive
        with open(f"/proc/{int(pid)}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return False


def cmd_holds(a: argparse.Namespace) -> int:
    root = os.path.abspath(a.root)
    R = run_dir(root)
    cfg = os.path.join(root, CONFIG_FILE)
    env = dict(os.environ, CAM_LIVE_HOME=os.path.join(root, "home"),
               PYTHONPATH=os.path.join(PROJECT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cli = [sys.executable, "-m", "cam_live.runtime.cli"]
    holds: List[Dict[str, Any]] = []
    for h in range(int(a.max_holds)):
        st = _state(R)
        if st and (st.get("status") in ("completed", "failed", "exhausted", "stopped")
                   or (st.get("counters") or {}).get("decisions", 0) >= a.decisions):
            break
        first = not st
        argv = (cli + ["run", "--task", TASK, "--run-dir", R, "--config", cfg, "--inproc", "--no-doctor", "--json",
                       "--start-timeout", "2400"] if first else cli + ["resume", "--run-dir", R, "--config", cfg,
                                                                      "--json", "--timeout", "2400"])
        t0 = time.time()
        p = subprocess.run(argv, env=env, capture_output=True, text=True, cwd=os.path.dirname(PROJECT))
        rec: Dict[str, Any] = {"hold": h + 1, "command": argv[3], "exit": p.returncode, "started": t0,
                               "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-1500:]}
        try:
            with open(os.path.join(R, "runtime.pid")) as f:
                pid = int(f.read().strip())
        except (OSError, ValueError):
            pid = None
        while _alive(pid):
            time.sleep(5.0)
        st = _state(R)
        rec.update(seconds=time.time() - t0, status=st.get("status"), reason=str(st.get("status_reason", ""))[:300],
                   decisions=(st.get("counters") or {}).get("decisions"))
        holds.append(rec)
        print(json.dumps({k: rec[k] for k in ("hold", "command", "exit", "seconds", "status", "decisions", "reason")}),
              flush=True)
        with open(os.path.join(root, "holds.json"), "w") as f:
            json.dump(holds, f, indent=1)
        if p.returncode not in (0,) and not st:
            break
        if (st.get("counters") or {}).get("decisions", 0) >= a.decisions:
            break
    return 0


# ----------------------------------------------------------------------------------------------------------------
# check: what the run recorded
# ----------------------------------------------------------------------------------------------------------------
def _jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


_DISCORD_READ = ("discord.get_", "discord.list_", "discord.fetch_", "discord.metrics", "discord.stats")


def check(root: str) -> Dict[str, Any]:
    from cam_live.runtime.tasks import load_task
    root = os.path.abspath(root)
    R = run_dir(root)
    verbatim = load_task(LIVE_TASK, for_run=False).objective
    ev = _jsonl(os.path.join(R, "events.jsonl"))
    dec = _jsonl(os.path.join(R, "decisions.jsonl"))
    preds = _jsonl(os.path.join(R, "predictions.jsonl"))
    exps = _jsonl(os.path.join(R, "experiments.jsonl"))
    learn = _jsonl(os.path.join(R, "learning_updates.jsonl"))
    meta = _jsonl(os.path.join(R, "meta_updates.jsonl"))
    started = [e for e in ev if e["type"] == "run_started"]
    with open(os.path.join(R, "task.json")) as f:
        tj = json.load(f)
    chosen = [d["data"]["decision"]["chosen"] for d in dec]
    results = [e["data"] for e in ev if e["type"] == "action_result"]
    by_kind: Dict[str, int] = {}
    for r in results:
        by_kind[f"{r.get('kind')}:{r.get('status')}"] = by_kind.get(f"{r.get('kind')}:{r.get('status')}", 0) + 1
    ops: Dict[str, int] = {}
    prov: Dict[str, int] = {}
    for c in chosen:
        ops[c["operation"]] = ops.get(c["operation"], 0) + 1
        for p in str(c.get("provenance", "")).split("+"):
            prov[p] = prov.get(p, 0) + 1
    res_by_action = {r.get("action_id"): r for r in results}
    discord_writes = [c for c in chosen if c["operation"].startswith("discord.")
                      and not c["operation"].startswith(_DISCORD_READ)]
    publications = [c for c in chosen if c["operation"].startswith(("publish.", "publishing."))]
    gated = []
    for c in discord_writes + publications:
        r = res_by_action.get(c["action_id"], {})
        gated.append({"operation": c["operation"], "kind": r.get("kind"), "status": r.get("status"),
                      "approval_id": r.get("approval_id", "")})
    executed_without_approval = [g for g in gated if g["kind"] == "execute" and not g["approval_id"]
                                 and g["status"] == "ok"]
    dlog = _jsonl(os.path.join(root, DISCORD_LOG))
    writes_at_service = [r for r in dlog if r.get("method") in ("POST", "PATCH", "PUT", "DELETE")]
    reqs: List[Dict[str, Any]] = []
    for d in (os.path.join(R, "exec"), R):
        reqs = _jsonl(os.path.join(d, "operator_requests.jsonl"))
        if reqs:
            break
    latest: Dict[str, Dict[str, Any]] = {}
    for r in reqs:
        latest[r["request_id"]] = dict(latest.get(r["request_id"], {}), **r)
    notes = _jsonl(os.path.join(R, "exec", "workspace", "notes.jsonl"))
    obs = [e["data"]["observation"] for e in ev if e["type"] == "observation"]
    discord_obs = [o for o in obs if str((o.get("provenance") or {}).get("source", "")).startswith("discord:")]
    store: Dict[str, Any] = {}
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{os.path.join(R, 'learner', 'experience.sqlite')}?mode=ro", uri=True)
        store["experience_records"] = con.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        store["items"] = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        # experience tokens built from what the staging Discord returned: the observations the runtime logged with
        # provenance discord:staging, found in the learner's store by their observation id
        ids = [o.get("obs_id") for o in discord_obs if o.get("obs_id")]
        store["discord_observations_in_store"] = sum(
            con.execute("SELECT COUNT(*) FROM items WHERE ref = ?", (i,)).fetchone()[0] for i in ids)
        disc_actions = [c["action_id"] for c in chosen if c["operation"].startswith("discord.")]
        store["experience_records_of_discord_actions"] = sum(
            con.execute("SELECT COUNT(*) FROM experiences WHERE action_id = ?", (a,)).fetchone()[0]
            for a in disc_actions)
        store["outcomes"] = con.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        store["links"] = con.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        con.close()
    except Exception as e:  # noqa: BLE001
        store["error"] = f"{type(e).__name__}: {e}"
    pend_appr = [a for a in (_state(R).get("approvals") or {}).values()]
    return {
        "run_dir": R, "checked_at": time.time(),
        "objective_verbatim": {
            "run_started": bool(started) and started[0]["data"]["task"]["objective"] == verbatim,
            "task_json_learner_view": tj["learner_view"]["objective"] == verbatim,
            "every_decision_record": bool(dec) and all(d["data"]["objective"] == verbatim for d in dec)},
        "decisions": len(dec), "chosen_operations": ops, "chosen_provenance": prov, "result_kinds": by_kind,
        "notes": {"count": len(notes), "tags": sorted({t for n in notes for t in n.get("tags", [])})},
        "operator_requests": {"count": len(latest), "with_justification": sum(1 for r in latest.values()
                                                                              if str(r.get("justification", "")).strip()),
                              "statuses": sorted({r.get("status", "") for r in latest.values()})},
        "predictions_registered": sum(1 for p in preds if p.get("type") == "prediction"),
        "predictions_resolved": sum(1 for p in preds if p.get("type") == "resolution"),
        "experiment_records": len(exps),
        "approvals": {"rows": len(pend_appr), "statuses": sorted({a.get("status", "") for a in pend_appr})},
        "discord_writes_and_publications_chosen": len(gated), "gated": gated,
        "executed_without_approval": executed_without_approval,
        "write_requests_received_by_staging_discord": len(writes_at_service),
        "staging_discord_requests": len(dlog),
        "discord_observations_logged": len(discord_obs),
        "experience_store": store,
        "learning_updates": {"count": len(learn), "by_timescale": _count(u["data"].get("timescale") for u in learn),
                             "nonzero_weight_delta": sum(1 for u in learn
                                                         if any(float(v or 0) > 0 for v in
                                                                (u["data"].get("weight_delta") or {}).values()))},
        "meta_updates": {"count": len(meta), "applied": sum(1 for m in meta if m["data"].get("applied"))},
        "final_status": _state(R).get("status"), "final_reason": _state(R).get("status_reason"),
    }


def _count(xs: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def cmd_check(a: argparse.Namespace) -> int:
    rep = check(a.root)
    txt = json.dumps(rep, indent=1, sort_keys=True, default=str)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    print(txt)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="planning_rehearsal.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--config", default="", help="YAML merged over configs/cam_live.yaml (e.g. a debug model)")
    p.add_argument("--hold-s", type=float, default=840.0)
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("serve")
    p.add_argument("--root", required=True)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("holds")
    p.add_argument("--root", required=True)
    p.add_argument("--decisions", type=int, default=25)
    p.add_argument("--max-holds", type=int, default=6)
    p.set_defaults(fn=cmd_holds)
    p = sub.add_parser("check")
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="")
    p.set_defaults(fn=cmd_check)
    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
