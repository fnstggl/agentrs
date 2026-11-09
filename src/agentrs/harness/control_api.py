"""Control API (FastAPI) and operator dashboard.

Every route except ``/health`` and the provider callbacks (OAuth redirect, signed webhooks) requires
the operator token (``Authorization: Bearer`` or the dashboard's cookie). The dashboard is
server-rendered HTML with controls to pause agents, revoke accounts, terminate browser sessions,
disable connectors, decide approvals, connect credential *references*, and trigger the emergency stop.
Secret values are never accepted in request bodies that are displayed, and never rendered.
"""

from __future__ import annotations

import hmac
import html
import json
import time
from typing import Any, Optional

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from sqlalchemy import func, select

from agentrs.harness.app import Harness
from agentrs.harness.db import ActionLog, Approval, Experience, Job, PlatformAccount, ProvisioningRequest, Task


def create_control_app(h: Harness, operator_token: str, extras: Optional[dict[str, Any]] = None) -> FastAPI:
    app = FastAPI(title="agentrs control plane")
    extras = extras or {}

    def operator(request: Request, authorization: str = Header("")) -> str:
        tok = authorization.removeprefix("Bearer ").strip() or request.cookies.get("agentrs_operator", "")
        if not tok or not hmac.compare_digest(tok, operator_token):
            raise HTTPException(401, "operator token required")
        return "operator"

    # ------------------------------------------------------------------ health / metrics
    @app.get("/health")
    def health():
        with h.db.session() as s:
            s.execute(select(func.count()).select_from(Task))
        return {"ok": True, "emergency_stop": bool(h.policy.stopped()), "t": time.time()}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(_=Depends(operator)):
        h.metrics.agents_active.set(sum(a.status == "active" for a in h.agents.all()))
        for k, v in h.jobs.counts().items():
            kind, st = k.rsplit(":", 1)
            h.metrics.jobs.labels(kind, st).set(v)
        h.metrics.followups_pending.set(sum(v for k, v in h.outcomes.scheduler.stats().items()
                                            if isinstance(v, int) and k.endswith(":pending")))
        return PlainTextResponse(h.metrics.render().decode())

    # ------------------------------------------------------------------ agents / accounts / connectors
    @app.get("/api/agents")
    def agents(_=Depends(operator)):
        return [{"id": a.id, "role": a.role, "status": a.status, "checkpoint": a.model_checkpoint,
                 "max_tier": a.max_tier, "budget": [a.actions_used, a.action_budget], "last_action_at": a.last_action_at}
                for a in h.agents.all()]

    @app.post("/api/agents/{agent_id}/{op}")
    def agent_op(agent_id: str, op: str, _=Depends(operator)):
        st = {"pause": "paused", "resume": "active", "stop": "stopped"}.get(op)
        if st is None:
            raise HTTPException(400, "pause | resume | stop")
        h.agents.set_status(agent_id, st)
        return {"agent": agent_id, "status": st}

    @app.get("/api/accounts")
    def accounts(_=Depends(operator)):
        return [{"id": a.id, "provider": a.provider, "external_id": a.external_id, "type": a.account_type,
                 "status": a.status, "bot_disclosure": a.bot_disclosure, "capabilities": a.capabilities,
                 "scopes": a.oauth_scopes, "credentials": sorted((a.credential_refs or {}).keys()),
                 "training_data": a.training_data_permission, "allowed_agents": a.allowed_agents,
                 "last_auth_at": a.last_auth_at, "last_activity_at": a.last_activity_at} for a in h.accounts.list()]

    @app.post("/api/accounts/{account_id}/revoke")
    def revoke(account_id: str, reason: str = Form("operator revocation"), _=Depends(operator)):
        h.accounts.revoke(account_id, reason)
        h.gateway.drop_connector(account_id)
        pool = extras.get("browser_pool")
        n = pool.revoke_account(account_id) if pool else 0
        return {"revoked": account_id, "browser_sessions_closed": n}

    @app.post("/api/accounts/{account_id}/suspend")
    def suspend(account_id: str, reason: str = Form("operator suspension"), _=Depends(operator)):
        h.accounts.suspend(account_id, reason)
        h.gateway.drop_connector(account_id)
        return {"suspended": account_id}

    @app.post("/api/connectors/{provider}/{op}")
    def connector_op(provider: str, op: str, _=Depends(operator)):
        if op == "disable":
            h.registry.disable(provider)
        elif op == "enable":
            h.registry.enable_override(provider)  # re-enables only what the config file enables
        else:
            raise HTTPException(400, "disable | enable")
        return {"provider": provider, "enabled": h.registry.enabled(provider)}

    @app.post("/api/emergency_stop")
    def estop(active: bool = Form(True), reason: str = Form(""), who: str = Depends(operator)):
        h.policy.emergency_stop(active, who, reason)
        return {"emergency_stop": active}

    # ------------------------------------------------------------------ provisioning / approvals
    @app.get("/api/provisioning")
    def provisioning(_=Depends(operator)):
        with h.db.session() as s:
            return [{"id": r.id, "provider": r.provider, "capabilities": r.required_capabilities, "purpose": r.purpose,
                     "agent": r.requesting_agent, "state": r.state, "steps": r.human_steps, "reason": r.decision_reason}
                    for r in s.scalars(select(ProvisioningRequest).order_by(ProvisioningRequest.created_at.desc()))]

    @app.post("/api/provisioning/{rid}/deny")
    def prov_deny(rid: str, reason: str = Form(...), who: str = Depends(operator)):
        h.provisioning.deny(rid, reason, who)
        return {"denied": rid}

    @app.post("/api/provisioning/{rid}/connect")
    async def prov_connect(rid: str, body: dict, who: str = Depends(operator)):
        refs = body.get("credential_refs", {})
        if any(not str(v).startswith("secret://") for v in refs.values()):
            raise HTTPException(400, "credential_refs must be secret:// references; put secret values in your secret manager")
        aid = h.provisioning.connect_credentials(rid, who, body["external_id"], refs, body.get("account_type", "bot"),
                                                 body.get("owner_org", ""), body.get("bot_disclosure", "unknown"),
                                                 body.get("oauth_scopes"), body.get("training_data_permission",
                                                                                    "not_authorized"))
        req = await h.provisioning.verify(rid)
        return {"account_id": aid, "state": req.state, "reason": req.decision_reason}

    @app.get("/api/approvals")
    def approvals(_=Depends(operator)):
        with h.db.session() as s:
            return [{"id": a.id, "kind": a.kind, "summary": a.summary, "status": a.status, "created_at": a.created_at,
                     "payload": a.payload.get("summary") or a.payload.get("steps")}
                    for a in s.scalars(select(Approval).order_by(Approval.created_at.desc()).limit(200))]

    @app.post("/api/approvals/{aid}/{op}")
    async def approval_op(aid: str, op: str, note: str = Form(""), who: str = Depends(operator)):
        with h.db.session() as s:
            ap = s.get(Approval, aid)
            if ap is None:
                raise HTTPException(404, "no approval")
            kind = ap.kind
        if kind in ("action", "high_impact"):
            if op == "approve":
                r = await h.gateway.execute_approved(aid, who)
                return {"approval": aid, "result": r.status, "reason": r.reason}
            h.gateway.deny_approval(aid, who, note)
            return {"approval": aid, "result": "denied"}
        if kind == "provisioning":
            with h.db.session() as s:
                rid = s.get(Approval, aid).ref_id
            if op == "approve":
                h.provisioning.approve_existing(rid, who)
            else:
                h.provisioning.deny(rid, note or "denied", who)
            return {"approval": aid, "result": op}
        raise HTTPException(400, f"approvals of kind {kind} are completed on the provider's own interface")

    # ------------------------------------------------------------------ tasks / learning / infra
    @app.get("/api/tasks")
    def tasks(_=Depends(operator)):
        return [{"id": t.id, "goal": t.goal[:160], "env": t.environment, "state": t.state, "agent": t.assigned_agent,
                 "tier": t.authorized_tier} for t in h.tasks.list()][-500:]

    @app.post("/api/tasks")
    def create_task(body: dict, _=Depends(operator)):
        tid = h.tasks.create(task_id=body["id"], source="human", goal=body["goal"], environment=body["environment"],
                             required_capabilities=body.get("required_capabilities", []),
                             allowed_actions=body.get("allowed_actions", []), verifier=body["verifier"],
                             budget=body.get("budget"), time_limit_s=body.get("time_limit_s", 3600),
                             needs_approval=body.get("needs_approval", False),
                             authorized_tier=body.get("authorized_tier", 0))
        return {"task": tid}

    def learning_summary() -> dict:
        with h.db.session() as s:
            n = s.scalar(select(func.count()).select_from(Experience))
            ok = s.scalar(select(func.count()).select_from(Experience).where(Experience.reward == 1.0))
            pend = s.scalar(select(func.count()).select_from(Experience).where(Experience.reward_status == "pending"))
        reg = extras.get("checkpoints")
        cks = [{"id": c.id, "status": c.status, "metrics": c.metrics, "notes": c.notes} for c in reg.list("policy")] if reg else []
        return {"experiences": n, "verified_successes": ok, "pending_outcomes": pend,
                "followups": h.outcomes.scheduler.stats(), "ledger": h.outcomes.ledger.stats(), "checkpoints": cks,
                **(extras.get("learning_stats") or {})}

    @app.get("/api/learning")
    def learning(_=Depends(operator)):
        return learning_summary()

    def infra_summary() -> dict:
        with h.db.session() as s:
            failed = s.scalar(select(func.count()).select_from(Job).where(Job.status == "failed"))
            api_err = s.scalar(select(func.count()).select_from(ActionLog).where(ActionLog.status == "failed"))
        egress = h.gateway.egress_log[-2000:]
        inf = extras.get("inference")
        pool = extras.get("browser_pool")
        return {"jobs": h.jobs.counts(), "failed_jobs": failed, "api_errors": api_err,
                "network_errors": sum(e.get("outcome") == "network_error" for e in egress),
                "egress_denied": sum(e.get("outcome") == "denied" for e in egress),
                "inference": inf.stats if inf else None, "browser_pool": pool.stats() if pool else None,
                "spending_usd": extras.get("spending_usd", 0.0), "emergency_stop": h.policy.stopped()}

    @app.get("/api/infra")
    def infra(_=Depends(operator)):
        return infra_summary()

    # ------------------------------------------------------------------ OAuth + webhooks (provider-facing)
    @app.get("/oauth/callback")
    async def oauth_callback(state: str, code: str):
        cb = extras.get("oauth_complete")
        if cb is None:
            raise HTTPException(503, "OAuth not configured")
        res = await cb(state, code)
        return {"connected": res["account_id"], "scopes": res["scopes"]}

    @app.post("/webhooks/{provider}")
    async def webhook(provider: str, request: Request):
        handler = (extras.get("webhooks") or {}).get(provider)
        if handler is None:
            raise HTTPException(404, "no webhook handler")
        body = await request.body()
        ok, out = await handler(dict(request.headers), body)
        if not ok:
            raise HTTPException(401, "bad signature")
        return out

    # ------------------------------------------------------------------ dashboard
    @app.get("/login", response_class=HTMLResponse)
    def login_page():
        return _page("Operator login", '<form method="post" action="/login"><label>Operator token '
                     '<input type="password" name="token"></label> <button>Log in</button></form>')

    @app.post("/login")
    def login(token: str = Form(...)):
        if not hmac.compare_digest(token, operator_token):
            raise HTTPException(401, "bad token")
        r = RedirectResponse("/dashboard", status_code=303)
        r.set_cookie("agentrs_operator", token, httponly=True, samesite="strict")
        return r

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request):
        try:
            operator(request, "")
        except HTTPException:
            return RedirectResponse("/login", status_code=303)
        with h.db.session() as s:
            recent = list(s.scalars(select(ActionLog).order_by(ActionLog.proposed_at.desc()).limit(25)))
            pend = list(s.scalars(select(Approval).where(Approval.status == "pending").order_by(Approval.created_at)))
            accs = list(s.scalars(select(PlatformAccount)))
        last = {}
        for r in recent:
            last.setdefault(r.agent_id, r)
        rows_agents = "".join(
            f"<tr><td>{_e(a.id)}</td><td>{_e(a.status)}</td><td>{_e(a.model_checkpoint)}</td><td>{a.actions_used}/"
            f"{a.action_budget}</td><td>{_e(last[a.id].provider + ':' + last[a.id].operation + ' → ' + last[a.id].status) if a.id in last else ''}</td>"
            f"<td>{_btn(f'/ui/agents/{a.id}/pause', 'Pause')}{_btn(f'/ui/agents/{a.id}/resume', 'Resume')}</td></tr>"
            for a in h.agents.all())
        rows_acc = "".join(
            f"<tr><td>{_e(a.provider)}</td><td>{_e(a.external_id)}</td><td>{_e(a.status)}</td><td>{_e(a.bot_disclosure)}"
            f"</td><td>{_e(', '.join(a.capabilities or []))}</td><td>{_e(', '.join(a.oauth_scopes or []))}</td>"
            f"<td>{_btn(f'/ui/accounts/{a.id}/revoke', 'Revoke')}</td></tr>" for a in accs)
        rows_ap = "".join(
            f"<tr><td>{_e(a.kind)}</td><td>{_e(a.summary)}</td><td>{_btn(f'/ui/approvals/{a.id}/approve', 'Approve')}"
            f"{_btn(f'/ui/approvals/{a.id}/deny', 'Deny')}</td></tr>" for a in pend)
        rows_prov = "".join(
            f"<tr><td>{_e(r['provider'])}</td><td>{_e(r['agent'])}</td><td>{_e(r['state'])}</td><td>{_e('; '.join(r['steps'] or []))}</td></tr>"
            for r in provisioning("operator") if r["state"] not in ("ACTIVE",))
        rows_conn = "".join(
            f"<tr><td>{_e(p)}</td><td>{'enabled' if h.registry.enabled(p) else 'disabled'}</td><td>"
            f"{_btn(f'/ui/connectors/{p}/disable', 'Disable')}</td></tr>" for p in h.registry.providers)
        rows_act = "".join(
            f"<tr><td>{time.strftime('%H:%M:%S', time.localtime(r.proposed_at))}</td><td>{_e(r.agent_id)}</td>"
            f"<td>{_e(r.provider)}:{_e(r.operation)}</td><td>{r.tier}</td><td>{_e(r.status)}</td><td>{_e(r.reason[:120])}</td></tr>"
            for r in recent)
        L, I = learning_summary(), infra_summary()
        body = (f"<p class=stop>{'EMERGENCY STOP ACTIVE' if I['emergency_stop'] else ''}</p>"
                f"<form method=post action='/ui/emergency_stop'><button class=danger>Emergency stop all agents</button></form>"
                f"<h2>Agents</h2><table><tr><th>agent</th><th>status</th><th>checkpoint</th><th>budget</th><th>last action"
                f"</th><th></th></tr>{rows_agents}</table>"
                f"<h2>Approvals</h2><table>{rows_ap or '<tr><td>none pending</td></tr>'}</table>"
                f"<h2>Provisioning requests</h2><table>{rows_prov or '<tr><td>none open</td></tr>'}</table>"
                f"<h2>Accounts</h2><table><tr><th>provider</th><th>account</th><th>status</th><th>bot disclosure</th>"
                f"<th>capabilities</th><th>scopes</th><th></th></tr>{rows_acc}</table>"
                f"<h2>Connectors</h2><table>{rows_conn}</table>"
                f"<h2>Learning</h2><pre>{_e(json.dumps({k: L[k] for k in ('experiences', 'verified_successes', 'pending_outcomes')}, indent=1))}</pre>"
                f"<pre>{_e(json.dumps(L['checkpoints'][-5:], indent=1, default=str))}</pre>"
                f"<h2>Infrastructure</h2><pre>{_e(json.dumps(I, indent=1, default=str))}</pre>"
                f"<h2>Recent actions</h2><table>{rows_act}</table>")
        return _page("agentrs operator dashboard", body)

    @app.post("/ui/{kind}/{ident}/{op}")
    async def ui_op(kind: str, ident: str, op: str, request: Request):
        who = operator(request, "")
        if kind == "agents":
            agent_op(ident, op, who)
        elif kind == "accounts" and op == "revoke":
            revoke(ident, "revoked from dashboard", who)
        elif kind == "approvals":
            await approval_op(ident, op, "", who)
        elif kind == "connectors":
            connector_op(ident, op, who)
        return RedirectResponse("/dashboard", status_code=303)

    @app.post("/ui/emergency_stop")
    def ui_stop(request: Request):
        who = operator(request, "")
        h.policy.emergency_stop(True, who, "dashboard")
        return RedirectResponse("/dashboard", status_code=303)

    return app


def _e(s: Any) -> str:
    return html.escape(str(s))


def _btn(action: str, label: str) -> str:
    return f"<form method=post action='{_e(action)}' style='display:inline'><button>{_e(label)}</button></form>"


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width'>"
            f"<meta http-equiv=refresh content=15><title>{_e(title)}</title><style>body{{font:14px system-ui;margin:16px}}"
            f"table{{border-collapse:collapse;margin-bottom:12px}}td,th{{border:1px solid #ccc;padding:3px 6px;text-align:left}}"
            f".danger{{background:#b00020;color:#fff}}.stop{{color:#b00020;font-weight:bold}}</style></head>"
            f"<body><h1>{_e(title)}</h1>{body}</body></html>")
