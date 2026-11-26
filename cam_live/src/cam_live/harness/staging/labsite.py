"""cam_live.harness.staging.labsite — ``diagnostics.world.LabWorld`` behind a web UI and JSON pages (workstream EB).

Lets delayed consequences be exercised *through the browser*: an intervention is started by submitting a form, its
adopters arrive later on the world's clock, and the pages / JSON endpoints show whatever is observable at that time.
The site is the external world here, so it applies LabWorld's own rules; authorization of the browser operations
that reach it is the computer environment's job. Hidden ground truth is never served.

:class:`LabSiteSource` is the measurement adapter an operator registers with ``ComputerEnvironment`` so pending
outcome queries created by browser actions are tracked and measured (unverified) by the environment.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List, Optional

from ...contracts import (ActionSpec, AuthorizationDecision, AuthVerdict, Modality, OutcomeObservation, OutcomeQuery,
                          OutcomeStatus, Quantity, ResultStatus, SimClock, new_id)
from ...diagnostics.world import LabConfig, LabWorld
from .server import Request, Response, Site, as_json, esc, page, redirect


class LabSite(Site):
    name = "labsite"
    description = "LabWorld (controlled world with delayed consequences) behind HTML forms and JSON pages"

    def __init__(self, cfg: Optional[LabConfig] = None, clock: Optional[SimClock] = None) -> None:
        super().__init__()
        self.world = LabWorld(cfg or LabConfig(), clock or SimClock())
        self.history: List[Dict[str, Any]] = []
        r = self.route
        r("GET", "/", self.home)
        r("GET", "/act", self.act_form)
        r("POST", "/act", self.act_post)
        r("GET", "/probe", self.probe_form)
        r("POST", "/probe", self.probe_post)
        r("POST", "/unlock", self.unlock_post)
        r("GET", "/history", self.history_page)
        r("GET", "/api/metrics", lambda q: as_json(self.metrics()))
        r("GET", "/api/pending", lambda q: as_json([x.to_dict() for x in self.world.get_pending_outcomes()]))
        r("POST", "/api/measure", self.measure)
        r("GET", "/api/events", self.events)
        r("GET", "/api/history", lambda q: as_json(self.history))
        r("POST", "/control/advance", self.advance)

    # ---- helpers ----------------------------------------------------------------------------------------------
    def _run(self, operation: str, args: Dict[str, Any]) -> Dict[str, Any]:
        w = self.world
        action = ActionSpec(action_id=new_id("labact"), operation=operation, modality=Modality.API, arguments=args)
        res = w.execute(action, AuthorizationDecision(AuthVerdict.ALLOW, action.content_hash()))
        entry = {"action_id": action.action_id, "operation": operation, "arguments": args, "status": res.status,
                 "detail": res.failure_detail, "at": res.started,
                 "text": "; ".join(o.text for o in res.observations),
                 "quantities": [q.to_dict() for o in res.observations for q in o.quantities],
                 "queries": [q.to_dict() for q in res.expected_outcomes]}
        self.history.append(entry)
        return entry

    def metrics(self) -> List[Dict[str, Any]]:
        return [q.to_dict() for q in self.world.get_observable_metrics()]

    def _nav(self) -> str:
        return ("<nav aria-label='Main'><a href='/'>Overview</a><a href='/act'>Run intervention</a>"
                "<a href='/probe'>Probe</a><a href='/history'>History</a></nav>")

    def _channels(self) -> str:
        return "".join(f"<option value='{i}'>{esc(n)}</option>" for i, n in enumerate(self.world.names))

    def _result(self, e: Dict[str, Any]) -> str:
        if e["status"] != ResultStatus.OK:
            return f"<p class='err' role='alert'>Not done: {esc(e['detail'])}.</p>"
        extra = "".join(f"<li>{esc(q['name'])}: {q['value']}</li>" for q in e["quantities"])
        due = "".join(f"<li>Count for <code>{esc(q['metric'])}</code> can be read after "
                      f"{q['horizon_s']:.0f} s.</li>" for q in e["queries"])
        return (f"<p class='ok' role='status'>{esc(e['text'])}. Reference {esc(e['action_id'])}.</p>"
                f"<ul aria-label='Readings'>{extra}</ul><ul aria-label='Later measurements'>{due}</ul>")

    # ---- pages ------------------------------------------------------------------------------------------------
    def home(self, req: Request, note: str = "") -> Response:
        w = self.world
        rows = "".join(f"<tr><th scope='row'>{esc(m['name'])}</th><td>{m['value']:g}"
                       f"{(' ' + esc(m['unit'])) if m['unit'] else ''}</td></tr>" for m in self.metrics())
        head = "".join(f"<th>f{j}</th>" for j in range(w.cfg.n_feat))
        ch = "".join(f"<tr><td>{esc(n)}</td>" + "".join(f"<td>{v}</td>" for v in w.features[i]) + "</tr>"
                     for i, n in enumerate(w.names))
        unlock = ("<form method='post' action='/unlock'><button type='submit'>Prepare last channel</button></form>"
                  if w.cfg.key_door else "")
        body = (f"{self._nav()}<main><h1>Lab world</h1>{note}<table><caption>Current state</caption>{rows}</table>"
                f"<table><caption>Channels</caption><thead><tr><th>Channel</th>{head}</tr></thead>"
                f"<tbody>{ch}</tbody></table>{unlock}</main>")
        return page("Lab world", body)

    def act_form(self, req: Request, note: str = "") -> Response:
        body = (f"{self._nav()}<main><h1>Run intervention</h1>{note}<form method='post' action='/act'>"
                f"<label for='channel'>Channel</label><select id='channel' name='channel'>{self._channels()}</select>"
                f"<label for='effort'>Effort</label><input id='effort' name='effort' type='number' min='1' "
                f"max='{self.world.cfg.max_effort}' value='1'><button type='submit'>Start</button></form></main>")
        return page("Run intervention — Lab world", body)

    def act_post(self, req: Request) -> Response:
        try:
            args = {"channel": int(req.form.get("channel", "")), "effort": int(req.form.get("effort", ""))}
        except ValueError:
            return self.act_form(req, "<p class='err' role='alert'>Channel and effort must be numbers.</p>")
        return self.act_form(req, self._result(self._run("lab.act", args)))

    def probe_form(self, req: Request, note: str = "") -> Response:
        body = (f"{self._nav()}<main><h1>Probe</h1>{note}<form method='post' action='/probe'>"
                f"<label for='channel'>Channel</label><select id='channel' name='channel'>{self._channels()}</select>"
                "<button type='submit'>Measure</button></form></main>")
        return page("Probe — Lab world", body)

    def probe_post(self, req: Request) -> Response:
        try:
            args = {"channel": int(req.form.get("channel", ""))}
        except ValueError:
            return self.probe_form(req, "<p class='err' role='alert'>Channel must be a number.</p>")
        return self.probe_form(req, self._result(self._run("lab.probe", args)))

    def unlock_post(self, req: Request) -> Response:
        return self.home(req, self._result(self._run("lab.unlock", {})))

    def history_page(self, req: Request) -> Response:
        w = self.world
        now = w.clock.now()
        rows = ""
        for e in self.history:
            arrived = sum(1 for t, a, _ in w._arrivals if a == e["action_id"] and t <= now)
            rows += (f"<tr><td>{esc(e['action_id'])}</td><td>{esc(e['operation'])}</td>"
                     f"<td>{esc(json.dumps(e['arguments'], sort_keys=True))}</td><td>{esc(e['status'])}</td>"
                     f"<td>{arrived}</td></tr>")
        body = (f"{self._nav()}<main><h1>History</h1><table><caption>Operations</caption><thead><tr>"
                "<th>Reference</th><th>Operation</th><th>Arguments</th><th>Status</th><th>Adopters so far</th>"
                f"</tr></thead><tbody>{rows}</tbody></table></main>")
        return page("History — Lab world", body)

    # ---- JSON -------------------------------------------------------------------------------------------------
    def measure(self, req: Request) -> Response:
        qs = [OutcomeQuery.from_dict(d) for d in (req.json() or [])]
        return as_json([o.to_dict() for o in self.world.measure(qs)])

    def events(self, req: Request) -> Response:
        evs, cur = self.world.poll_events(req.query.get("cursor", ""))
        return as_json({"events": [e.to_dict() for e in evs], "cursor": cur})

    def advance(self, req: Request) -> Response:
        """Staging clock control (simulated time only): lets tests mature delayed consequences in milliseconds."""
        self.world.clock.advance(float(req.query.get("seconds", "0")))
        return as_json({"now": self.world.clock.now()})

    def state(self) -> Dict[str, Any]:
        return {"history": list(self.history), "metrics": self.metrics(), "hits": dict(self.hits)}


class LabSiteSource:
    """Outcome/metric source for a labsite, read over its JSON pages (plain localhost HTTP, no browser)."""

    def __init__(self, base_url: str, name: str = "labsite", timeout: float = 10.0):
        self.base_url, self.name, self.timeout = base_url.rstrip("/"), name, timeout

    def _get(self, path: str) -> Any:
        with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def _post(self, path: str, obj: Any) -> Any:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def pending(self) -> List[OutcomeQuery]:
        return [OutcomeQuery.from_dict(d) for d in self._get("/api/pending")]

    def metrics(self) -> List[Quantity]:
        return [Quantity.from_dict(d) for d in self._get("/api/metrics")]

    def measure(self, query: OutcomeQuery, now: float) -> Optional[OutcomeObservation]:
        if not query.metric.startswith("lab."):
            return None
        out = self._post("/api/measure", [query.to_dict()])
        o = OutcomeObservation.from_dict(out[0])
        if o.status == OutcomeStatus.UNOBSERVABLE:
            return None
        o.verified, o.source = False, self.name
        return o
