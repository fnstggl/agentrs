"""cam_live.harness.publishing — the authorized publishing adapter (spec 12.4).

Owned by workstream ES.

Owned destinations only: a static website directory the operator owns, and channels of the owner-authorized Discord
server (through :class:`cam_live.harness.discord.DiscordExecutor`).

    publish.draft   -> store content exactly as the agent wrote it (local state; nothing leaves the machine)
    publish.preview -> the exact bytes that would be published and exactly where
    publish.submit  -> the external write, through ``ExecContext.check``:
                         ALLOW           -> published
                         NEEDS_APPROVAL  -> the draft waits in the operator's approval queue; nothing is published
                         DENY            -> the draft is marked rejected; nothing is published
                       submitting a waiting draft again publishes it once the operator has approved it
    publish.update  -> a new revision of a draft or of something already published (needs preview and submit again)
    publish.status  -> state of the drafts, and whether what is live still equals what was published
    publish.get_analytics -> aggregate visit counts of an owned site from its access log, as exact quantities

The adapter adds nothing to the content: no template, no schedule, no wording. It gives no signal about whether
publishing was useful; it reports what happened.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import time as _time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from ..contracts import (ActionResult, ActionSpec, Clock, Cost, ExecContext, Level, Modality,
                         Observation, OutcomeKind, PrimitiveOp, Provenance, Quantity, ResultStatus, ToolDescriptor,
                         Trust, new_id)
from .failures import FailureEvidence, authorization_result, classify, failed_result, failure_of
from .http_api import acting_under, schema_errors

DRAFT, PREVIEWED, PENDING, PUBLISHED, REJECTED = "draft", "previewed", "pending_approval", "published", "rejected"
STATIC_SITE, DISCORD_CHANNEL = "static_site", "discord_channel"


@dataclass
class Destination:
    """An owned place content may be published to."""
    name: str
    kind: str                    # "static_site" | "discord_channel"
    root: str = ""               # static_site: directory the site is served from
    base_url: str = ""           # static_site: public address of the site, for the preview
    access_log: str = ""         # static_site: the web server's access log (combined log format)
    channel_id: str = ""         # discord_channel


_LOG_LINE = re.compile(r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+)[^"]*" (\d{3}) (\S+)(?: "([^"]*)" "([^"]*)")?')
_ASSET = re.compile(r"\.(css|js|png|jpe?g|gif|svg|ico|woff2?|ttf|map|webp|avif|mp4|webm|txt|xml|json)$", re.I)
_AUTOMATED = re.compile(r"bot|crawl|spider|slurp|curl|wget|python|httpx|monitor|headless|scrapy|feed", re.I)

_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "publish.draft": {"type": "object", "additionalProperties": False, "required": ["destination", "content"],
                      "properties": {"destination": {"type": "string", "minLength": 1},
                                     "path": {"type": "string", "maxLength": 240,
                                              "description": "static site: file path relative to the site root"},
                                     "content": {"type": "string", "minLength": 1}}},
    "publish.preview": {"type": "object", "additionalProperties": False, "required": ["draft_id"],
                        "properties": {"draft_id": {"type": "string", "minLength": 1}}},
    "publish.submit": {"type": "object", "additionalProperties": False, "required": ["draft_id"],
                       "properties": {"draft_id": {"type": "string", "minLength": 1},
                                      "destination": {"type": "string", "minLength": 1,
                                                      "description": "the destination address shown by the preview "
                                                                     "(where.address); if given it must be the draft's"},
                                      "content": {"type": "string", "minLength": 1,
                                                  "description": "the exact text to publish (preview: submit_arguments); "
                                                                 "an operator approval then shows and binds this text"},
                                      "content_sha256": {"type": "string", "minLength": 64, "maxLength": 64,
                                                         "description": "SHA-256 of that text (preview: submit_arguments)"}}},
    "publish.update": {"type": "object", "additionalProperties": False, "required": ["draft_id", "content"],
                       "properties": {"draft_id": {"type": "string", "minLength": 1},
                                      "content": {"type": "string", "minLength": 1},
                                      "destination": {"type": "string", "minLength": 1,
                                                      "description": "the draft's destination address (where.address); "
                                                                     "if given it must be the draft's"}}},
    "publish.status": {"type": "object", "additionalProperties": False,
                       "properties": {"draft_id": {"type": "string", "minLength": 1}}},
    "publish.get_analytics": {"type": "object", "additionalProperties": False, "required": ["destination"],
                          "properties": {"destination": {"type": "string", "minLength": 1},
                                         "since_s": {"type": "number", "minimum": 0,
                                                     "description": "only requests at most this many seconds old"}}},
}
_TOOLS = {
    "publish.draft": ("local", "publish.draft", "Store a draft for an owned destination (a file path of an owned "
                      "static site, or a channel of the authorized server). Nothing is published."),
    "publish.preview": ("local", "publish.draft", "Show exactly what a draft would publish and exactly where."),
    "publish.submit": ("external_write", "publish.write", "Publish a previewed draft to its owned destination. Where the "
                       "authorization policy requires it, the draft waits for the operator's approval instead. Submit "
                       "with the submit_arguments of the preview (draft id, destination, exact content and its SHA-256): "
                       "the approval then shows the operator that text and covers only that text; a draft changed "
                       "afterwards needs a new submission and a new approval."),
    "publish.update": ("local", "publish.draft", "Replace the content of a draft or of something already published "
                       "with a new revision, which must be previewed and submitted again."),
    "publish.status": ("none", "publish.read", "Report the state of drafts: draft, previewed, waiting for approval, "
                       "published or rejected, and whether the live content still equals what was published."),
    "publish.get_analytics": ("external_read", "publish.read", "Read aggregate visit counts of an owned static site from "
                          "its access log: requests, page views, distinct visitors and views per page."),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PublishingExecutor:
    """``ModalityExecutor`` (modality ``api``) for the publishing tools."""

    modality = Modality.API

    def __init__(self, state_dir: str, destinations: Sequence[Destination], *, discord: Any = None,
                 clock: Optional[Clock] = None, max_content_bytes: int = 2_000_000, preview_chars: int = 20_000):
        self.state_path = os.path.join(state_dir, "publishing", "drafts.json")
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        self.destinations = {d.name: d for d in destinations}
        self.discord, self.clock = discord, clock
        self.max_content_bytes, self.preview_chars = max_content_bytes, preview_chars
        self._drafts: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as f:
                self._drafts = json.load(f)

    @classmethod
    def from_resources(cls, resources: Mapping[str, Any], state_dir: str, *, discord: Any = None,
                       clock: Optional[Clock] = None) -> "PublishingExecutor":
        """Destinations from ``TaskSpec.resources``: ``owned_sites: [{name, root, base_url?, access_log?}]`` and
        ``discord: {channels: {<name>: <channel id>}}``."""
        dests = [Destination(name=s["name"], kind=STATIC_SITE, root=s["root"], base_url=s.get("base_url", ""),
                             access_log=s.get("access_log", "")) for s in resources.get("owned_sites") or []]
        for name, cid in ((resources.get("discord") or {}).get("channels") or {}).items():
            dests.append(Destination(name=name, kind=DISCORD_CHANNEL, channel_id=str(cid)))
        return cls(state_dir, dests, discord=discord, clock=clock)

    # ---- state ---------------------------------------------------------------------------------------------
    def _save(self) -> None:
        tmp = f"{self.state_path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._drafts, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)

    def drafts(self) -> List[Dict[str, Any]]:
        return [dict(d) for d in self._drafts.values()]

    def _target(self, dest: Destination, path: str) -> Tuple[str, str]:
        """-> (absolute file path, public URL) of a static-site path; raises ValueError if it leaves the root."""
        if not path or path.startswith(("/", "\\")) or "\\" in path or "\x00" in path:
            raise ValueError("path must be a relative path inside the site")
        parts = path.split("/")
        if any(p in ("", ".", "..") or p.startswith(".") for p in parts):
            raise ValueError("path must not contain empty, hidden or parent segments")
        root = os.path.realpath(dest.root)
        full = os.path.realpath(os.path.join(root, *parts))
        if os.path.commonpath([root, full]) != root or full == root:
            raise ValueError("path leaves the site root")
        url = dest.base_url.rstrip("/") + "/" + path if dest.base_url else ""
        return full, url

    # ---- ModalityExecutor ----------------------------------------------------------------------------------
    def tools(self) -> List[ToolDescriptor]:
        any_dest = bool(self.destinations)
        out = []
        for name, (effect, scope, text) in _TOOLS.items():
            out.append(ToolDescriptor(name=name, modality=Modality.API, description=text, arg_schema=_SCHEMAS[name],
                                      level=Level.PRIMITIVE, auth_scope=scope, side_effects=effect,
                                      est_cost=Cost(seconds=0.2, primitive_ops=1), available=any_dest,
                                      unavailable_reason="" if any_dest else "blocked: no owned publishing destination is configured"))
        return out

    def validate(self, action: ActionSpec) -> List[str]:
        schema = _SCHEMAS.get(action.operation)
        if schema is None:
            return [f"unknown operation {action.operation!r} for the publishing adapter"]
        a = action.arguments
        errs = schema_errors(schema, a)
        if errs:
            return errs
        if "destination" in a and action.operation in ("publish.draft", "publish.get_analytics"):
            dest = self.destinations.get(a["destination"])
            if dest is None:
                return [f"arguments.destination: {a['destination']!r} is not an owned destination "
                        f"(known: {sorted(self.destinations)})"]
            if action.operation == "publish.draft":
                if len(a["content"].encode("utf-8")) > self.max_content_bytes:
                    errs.append("arguments.content: too large")
                if dest.kind == STATIC_SITE:
                    try:
                        self._target(dest, a.get("path", ""))
                    except ValueError as e:
                        errs.append(f"arguments.path: {e}")
                elif len(a["content"]) > 2000:
                    errs.append("arguments.content: a channel message holds at most 2000 characters")
            if action.operation == "publish.get_analytics" and (dest.kind != STATIC_SITE or not dest.access_log):
                errs.append("arguments.destination: no access log is configured for this destination")
        if "draft_id" in a and a["draft_id"] not in self._drafts:
            errs.append(f"arguments.draft_id: no draft {a['draft_id']!r}")
        elif action.operation == "publish.update":
            d = self._drafts[a["draft_id"]]
            if self.destinations[d["destination"]].kind == DISCORD_CHANNEL and len(a["content"]) > 2000:
                errs.append("arguments.content: a channel message holds at most 2000 characters")
        return errs

    def close(self) -> None:
        return None

    def execute(self, action: ActionSpec, ctx: ExecContext) -> ActionResult:
        clock, started = ctx.clock, ctx.clock.now()
        errs = self.validate(action)
        if errs:
            return failed_result(action, classify(FailureEvidence(arg_errors=errs, modality=Modality.API)), clock, started)
        handler = {"publish.draft": self._draft, "publish.preview": self._preview, "publish.submit": self._submit,
                   "publish.update": self._update, "publish.status": self._status,
                   "publish.get_analytics": self._analytics}[action.operation]
        return handler(action, ctx, started)

    # ---- helpers -------------------------------------------------------------------------------------------
    def _ok(self, action: ActionSpec, ctx: ExecContext, started: float, text: str, data: Dict[str, Any],
            quantities: Optional[List[Quantity]] = None, ops: Optional[List[PrimitiveOp]] = None,
            source: str = "publishing", uri: str = "", sha: str = "", trust: str = Trust.SELF) -> ActionResult:
        now = ctx.clock.now()
        obs = Observation(obs_id=new_id("obs"), ts=now, kind="tool_result", text=text, quantities=quantities or [],
                          provenance=Provenance(source=source, uri=uri, retrieved_at=now, trust=trust, content_sha256=sha),
                          parent_action_id=action.action_id, data=data)
        for op in ops or []:
            op.ended, op.obs_after = op.ended or now, op.obs_after or obs.obs_id
        return ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=started, ended=now,
                            observations=[obs], ops=list(ops or []), verified=True,
                            cost=Cost(seconds=now - started, primitive_ops=len(ops or [])),
                            artifacts={"draft_id": data["draft_id"]} if "draft_id" in data else {})

    def _where(self, d: Dict[str, Any]) -> Dict[str, Any]:
        dest = self.destinations[d["destination"]]
        if dest.kind == STATIC_SITE:
            full, url = self._target(dest, d["path"])
            existing = None
            if os.path.isfile(full):
                with open(full, "rb") as f:
                    existing = hashlib.sha256(f.read()).hexdigest()
            return {"kind": STATIC_SITE, "destination": dest.name, "location": full, "url": url,
                    "address": url or f"file:{full}", "replaces_sha256": existing}
        pub = d.get("published") or {}
        return {"kind": DISCORD_CHANNEL, "destination": dest.name, "location": f"discord channel {dest.channel_id}",
                "address": f"channel:discord:{dest.channel_id}", "channel_id": dest.channel_id,
                "replaces_message_id": pub.get("message_id")}

    def _public(self, d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: d.get(k) for k in ("draft_id", "destination", "path", "revision", "state", "content_sha256",
                                      "approval_id", "published", "reasons", "created_at", "updated_at")}

    # ---- tools ---------------------------------------------------------------------------------------------
    def _draft(self, action: ActionSpec, ctx: ExecContext, started: float) -> ActionResult:
        a, now = action.arguments, ctx.clock.now()
        dest = self.destinations[a["destination"]]
        d = {"draft_id": new_id("draft"), "destination": dest.name, "path": a.get("path", "") if dest.kind == STATIC_SITE else "",
             "content": a["content"], "content_sha256": _sha(a["content"]), "revision": 1, "state": DRAFT,
             "preview_sha256": "", "approval_id": "", "published": None, "reasons": [], "created_at": now,
             "updated_at": now, "history": [{"ts": now, "event": "drafted", "revision": 1}]}
        self._drafts[d["draft_id"]] = d
        self._save()
        return self._ok(action, ctx, started, f"draft {d['draft_id']} stored for {dest.name} "
                                              f"({len(a['content'])} characters); not published",
                        dict(self._public(d), where=self._where(d)),
                        [Quantity("publish.draft.characters", float(len(a["content"])), kind=OutcomeKind.COUNT)])

    def _preview(self, action: ActionSpec, ctx: ExecContext, started: float) -> ActionResult:
        d = self._drafts[action.arguments["draft_id"]]
        where = self._where(d)
        content = d["content"]
        d["preview_sha256"] = d["content_sha256"]
        if d["state"] == DRAFT:
            d["state"] = PREVIEWED
        d["updated_at"] = ctx.clock.now()
        d["history"].append({"ts": d["updated_at"], "event": "previewed", "revision": d["revision"]})
        self._save()
        shown = content[: self.preview_chars]
        text = (f"preview of draft {d['draft_id']} revision {d['revision']} -> {where['location']}"
                f"{' (' + where['url'] + ')' if where.get('url') else ''}; {len(content.encode('utf-8'))} bytes, "
                f"sha256 {d['content_sha256'][:16]}; exact content follows\n{shown}")
        submit = {"draft_id": d["draft_id"], "destination": where["address"], "content": content,
                  "content_sha256": d["content_sha256"]}
        return self._ok(action, ctx, started, text,
                        dict(self._public(d), where=where, content=shown, content_complete=len(shown) == len(content),
                             bytes=len(content.encode("utf-8")), submit_arguments=submit),
                        [Quantity("publish.preview.bytes", float(len(content.encode("utf-8"))), unit="bytes",
                                  kind=OutcomeKind.COUNT)], sha=d["content_sha256"])

    def _update(self, action: ActionSpec, ctx: ExecContext, started: float) -> ActionResult:
        a, now = action.arguments, ctx.clock.now()
        d = self._drafts[a["draft_id"]]
        errs = ["arguments.content: too large"] if len(a["content"].encode("utf-8")) > self.max_content_bytes else []
        if a.get("destination") is not None and a["destination"] not in (self._where(d)["address"], d["destination"]):
            errs.append(f"arguments.destination: draft {d['draft_id']} goes to {self._where(d)['address']!r}")
        if errs:
            return failed_result(action, classify(FailureEvidence(arg_errors=errs)), ctx.clock, started)
        d.update(content=a["content"], content_sha256=_sha(a["content"]), revision=d["revision"] + 1, state=DRAFT,
                 preview_sha256="", approval_id="", reasons=[], updated_at=now)
        d["history"].append({"ts": now, "event": "updated", "revision": d["revision"]})
        self._save()
        return self._ok(action, ctx, started, f"draft {d['draft_id']} is now revision {d['revision']}; it must be "
                                              f"previewed and submitted again", dict(self._public(d), where=self._where(d)))

    def _status(self, action: ActionSpec, ctx: ExecContext, started: float) -> ActionResult:
        want = action.arguments.get("draft_id")
        rows = []
        for d in self._drafts.values():
            if want and d["draft_id"] != want:
                continue
            row = dict(self._public(d), where=self._where(d))
            pub = d.get("published")
            if pub and row["where"]["kind"] == STATIC_SITE:
                row["live_matches_published"] = row["where"].get("replaces_sha256") == pub["content_sha256"]
            rows.append(row)
        counts = {s: sum(1 for r in rows if r["state"] == s) for s in (DRAFT, PREVIEWED, PENDING, PUBLISHED, REJECTED)}
        text = "; ".join(f"{r['draft_id']} r{r['revision']} [{r['state']}] -> {r['where']['location']}" for r in rows) \
            or "there are no drafts"
        return self._ok(action, ctx, started, text[:4000], {"drafts": rows, "counts": counts},
                        [Quantity(f"publish.drafts.{s}", float(n), kind=OutcomeKind.COUNT) for s, n in counts.items()])

    def _submit(self, action: ActionSpec, ctx: ExecContext, started: float) -> ActionResult:
        clock = ctx.clock
        d = self._drafts[action.arguments["draft_id"]]
        if d["state"] == PUBLISHED and (d.get("published") or {}).get("content_sha256") == d["content_sha256"]:
            return self._ok(action, ctx, started, f"draft {d['draft_id']} revision {d['revision']} is already published",
                            dict(self._public(d), where=self._where(d), already_published=True))
        if d["preview_sha256"] != d["content_sha256"]:
            rep = classify(FailureEvidence(arg_errors=[f"draft {d['draft_id']} revision {d['revision']} has not been "
                                                       f"previewed; publish.preview shows what would be published"]))
            return failed_result(action, rep, clock, started)
        dest, where = self.destinations[d["destination"]], self._where(d)
        stated = action.arguments.get("destination")
        if stated is not None and stated not in (where["address"], dest.name):
            rep = classify(FailureEvidence(arg_errors=[f"arguments.destination: draft {d['draft_id']} goes to "
                                                       f"{where['address']!r}, not {stated!r}"]))
            return failed_result(action, rep, clock, started)
        bound = self._content_binding(action, ctx, d)
        if bound:
            return failed_result(action, classify(FailureEvidence(arg_errors=[bound])), clock, started)

        def refused(res: ActionResult) -> ActionResult:
            now = clock.now()
            if res.status == ResultStatus.NEEDS_APPROVAL:
                d.update(state=PENDING, approval_id=res.artifacts.get("approval_id", ""), updated_at=now)
                if d["approval_id"]:            # this approval is for this exact content
                    d.setdefault("approvals", {})[d["approval_id"]] = d["content_sha256"]
                d["history"].append({"ts": now, "event": "waiting_for_approval", "revision": d["revision"],
                                     "approval_id": d["approval_id"]})
            else:
                d.update(state=REJECTED, reasons=[res.failure_detail], updated_at=now)
                d["history"].append({"ts": now, "event": "refused", "revision": d["revision"], "detail": res.failure_detail[:300]})
            self._save()
            res.artifacts["draft_id"] = d["draft_id"]
            res.observations.append(Observation(
                obs_id=new_id("obs"), ts=now, kind="tool_result",
                text=f"draft {d['draft_id']} revision {d['revision']} was not published: {d['state']}",
                provenance=Provenance(source="publishing", retrieved_at=now, trust=Trust.SELF),
                parent_action_id=action.action_id, data=dict(self._public(d), where=where)))
            return res

        if dest.kind == DISCORD_CHANNEL:
            if self.discord is None:
                why = "blocked: no Discord adapter is configured for channel destinations"
                rep = classify(FailureEvidence(exception=PermissionError(why), detail=why, modality=Modality.API))
                return failed_result(action, rep, clock, started, status=ResultStatus.DENIED)
            prior = (d.get("published") or {}).get("message_id")
            sub = ActionSpec(action_id=f"{action.action_id}.{d['draft_id']}.r{d['revision']}",
                             operation="discord.edit_message" if prior else "discord.send_message", modality=Modality.API,
                             target=dest.channel_id, auth_scope="discord.write", timeout_s=action.timeout_s,
                             parent_decision_id=action.parent_decision_id, experiment_id=action.experiment_id,
                             arguments=dict({"channel_id": dest.channel_id, "content": d["content"]},
                                            **({"guild_id": self.discord.guild_id} if getattr(self.discord, "guild_id", "") else {}),
                                            **({"message_id": prior} if prior else {})))
            with acting_under(ctx, action):               # the Discord primitive is what the policy judges
                res = self.discord.execute(sub, ctx)
            res.action_id = action.action_id
            judged = res.status == ResultStatus.NEEDS_APPROVAL or "authorization_denied" in failure_of(res).get("signals", [])
            if judged:
                return refused(res)
            if res.status != ResultStatus.OK:             # a fault, not a decision about the content: the draft stays
                return res
            msg = res.observations[0].data["message"]
            published = {"revision": d["revision"], "content_sha256": d["content_sha256"], "ts": clock.now(),
                         "location": where["location"], "message_id": msg["id"],
                         "environment": res.observations[0].data.get("environment", "")}
            ops, uri = res.ops, f"discord:{dest.channel_id}/{msg['id']}"
        else:
            op = PrimitiveOp(op="publish.write", modality=Modality.API, started=clock.now(),
                             arguments={"destination": where["address"], "destination_name": dest.name,
                                        "kind": dest.kind, "location": where["location"],
                                        "url": where.get("url", ""), "content_sha256": d["content_sha256"],
                                        "bytes": len(d["content"].encode("utf-8")), "content": d["content"][:65536],
                                        "draft_id": d["draft_id"], "revision": d["revision"]})
            res0 = authorization_result(action, ctx.check(op, action), clock, started, ops=[op])
            if res0 is not None:
                return refused(res0)
            full = where["location"]
            os.makedirs(os.path.dirname(full), exist_ok=True)
            tmp = f"{full}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(d["content"])
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, full)
            with open(full, "rb") as f:                   # inspect the resulting state; do not assume
                written = hashlib.sha256(f.read()).hexdigest()
            if written != d["content_sha256"]:
                rep = classify(FailureEvidence(verified=False, executed_ok=True, modality=Modality.API,
                                               detail="the published file does not equal the previewed content"))
                return failed_result(action, rep, clock, started, ops=[op])
            published = {"revision": d["revision"], "content_sha256": d["content_sha256"], "ts": clock.now(),
                         "location": full, "url": where.get("url", "")}
            ops, uri = [op], where.get("url") or full
        now = clock.now()
        d.update(state=PUBLISHED, published=published, approval_id="", reasons=[], updated_at=now)
        d["history"].append({"ts": now, "event": "published", "revision": d["revision"], "location": published["location"]})
        self._save()
        return self._ok(action, ctx, started, f"draft {d['draft_id']} revision {d['revision']} published to "
                                              f"{published['location']}", dict(self._public(d), where=self._where(d)),
                        [Quantity("publish.published.bytes", float(len(d["content"].encode("utf-8"))), unit="bytes",
                                  kind=OutcomeKind.COUNT)], ops=ops, uri=uri, sha=d["content_sha256"])

    @staticmethod
    def _content_binding(action: ActionSpec, ctx: ExecContext, d: Dict[str, Any]) -> str:
        """Why this submission may not publish the draft's current content ("" = it may). A submission that names
        the content (``content`` / ``content_sha256``) publishes only that content; one executed under an operator
        approval must name it, so that the approval the operator gave covers exactly the text that is published."""
        a = action.arguments
        changed = (f"draft {d['draft_id']} is now revision {d['revision']} and no longer has the content this "
                   f"submission names; preview it and submit again (an approval given for the earlier text does not "
                   f"cover the new one)")
        if "content" in a and a["content"] != d["content"]:
            return changed
        if "content_sha256" in a and a["content_sha256"] != d["content_sha256"]:
            return changed
        auth = getattr(ctx, "authorization", None)
        approval = str(getattr(auth, "approval_id", "") or "")
        if approval:
            if approval in (d.get("approvals") or {}) and d["approvals"][approval] != d["content_sha256"]:
                return changed
            if "content" not in a and "content_sha256" not in a:
                return ("an approved publication must name the exact content it was approved for: submit with the "
                        "submit_arguments of publish.preview, which the operator then sees and approves")
        return ""

    # ---- aggregate analytics -------------------------------------------------------------------------------
    def _analytics(self, action: ActionSpec, ctx: ExecContext, started: float) -> ActionResult:
        clock, a = ctx.clock, action.arguments
        dest = self.destinations[a["destination"]]
        op = PrimitiveOp(op="publish.get_analytics", modality=Modality.API, started=clock.now(),
                         arguments={"destination_name": dest.name, "path": dest.access_log, "aggregate_only": True})
        res0 = authorization_result(action, ctx.check(op, action), clock, started, ops=[op])
        if res0 is not None:
            return res0
        try:
            with open(dest.access_log, "rb") as f:
                raw = f.read()
        except OSError as e:
            rep = classify(FailureEvidence(target_missing=True, modality=Modality.API,
                                           detail=f"the access log could not be read: {type(e).__name__}"))
            return failed_result(action, rep, clock, started, ops=[op])
        stats = aggregate_access_log(raw.decode("utf-8", errors="replace"), now=clock.now(), since_s=a.get("since_s"),
                                     own_host=urlsplit(dest.base_url).netloc)
        pre = f"site.{re.sub(r'[^A-Za-z0-9_]', '_', dest.name)}"
        exposure = max(1.0, stats["window_s"])
        q = [Quantity(f"{pre}.requests", float(stats["requests"]), kind=OutcomeKind.COUNT, exposure=exposure, unit="requests"),
             Quantity(f"{pre}.page_views", float(stats["page_views"]), kind=OutcomeKind.COUNT, exposure=exposure, unit="views"),
             Quantity(f"{pre}.visitors", float(stats["visitors"]), kind=OutcomeKind.COUNT, exposure=exposure, unit="clients"),
             Quantity(f"{pre}.page_views.declared_automated", float(stats["automated_page_views"]), kind=OutcomeKind.COUNT,
                      exposure=exposure),
             Quantity(f"{pre}.errors", float(stats["errors"]), kind=OutcomeKind.COUNT, exposure=exposure),
             Quantity(f"{pre}.window_s", float(stats["window_s"]), unit="s")]
        for path, n in list(stats["by_path"].items())[:24]:
            q.append(Quantity(f"{pre}.views.{re.sub(r'[^A-Za-z0-9_]', '_', path.strip('/')) or 'index'}", float(n),
                              kind=OutcomeKind.COUNT, exposure=exposure))
        text = (f"access log of {dest.name}: {stats['requests']} requests, {stats['page_views']} page views by "
                f"{stats['visitors']} distinct clients ({stats['automated_page_views']} views by self-declared "
                f"automated clients, counted separately) over {stats['window_s']:.0f} s")
        return self._ok(action, ctx, started, text, {"destination": dest.name, "analytics": stats}, q, ops=[op],
                        source="publishing.analytics", uri=dest.access_log, sha=hashlib.sha256(raw).hexdigest(),
                        trust=Trust.UNTRUSTED)


def _log_time(stamp: str) -> Optional[float]:
    try:
        return calendar.timegm(_time.strptime(stamp[:20], "%d/%b/%Y:%H:%M:%S")) - _tz(stamp[21:])
    except (ValueError, IndexError):
        return None


def _tz(offset: str) -> float:
    try:
        sign = -1.0 if offset.strip().startswith("-") else 1.0
        digits = offset.strip().lstrip("+-")
        return sign * (int(digits[:2]) * 3600 + int(digits[2:4]) * 60)
    except (ValueError, IndexError):
        return 0.0


def aggregate_access_log(text: str, *, now: float, since_s: Optional[float] = None, own_host: str = "") -> Dict[str, Any]:
    """Aggregate a web server access log (common / combined log format). Only counts leave this function: client
    addresses are used to count distinct clients and are neither returned nor stored.

    A page view is a successful GET of something that is not a static asset. Views by clients whose user agent
    declares automation are counted separately and are not included in ``page_views`` or ``visitors``.
    """
    requests = views = auto_views = errors = unparsed = 0
    clients, by_path, referrers = set(), {}, {}
    t_min = t_max = None
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _LOG_LINE.match(line)
        if m is None:
            unparsed += 1
            continue
        ip, stamp, method, target, status, _size, referer, agent = m.groups()
        ts = _log_time(stamp)
        if since_s is not None and (ts is None or ts < now - since_s):
            continue
        if ts is not None:
            t_min, t_max = (ts if t_min is None else min(t_min, ts)), (ts if t_max is None else max(t_max, ts))
        requests += 1
        code = int(status)
        if code >= 400:
            errors += 1
        path = urlsplit(target).path or "/"
        if method != "GET" or not (200 <= code < 300 or code == 304) or _ASSET.search(path):
            continue
        if agent and _AUTOMATED.search(agent):
            auto_views += 1
            continue
        views += 1
        clients.add(hashlib.sha256(ip.encode()).digest()[:8])
        by_path[path] = by_path.get(path, 0) + 1
        host = urlsplit(referer).netloc if referer and referer != "-" else ""
        if host and host != own_host:
            referrers[host] = referrers.get(host, 0) + 1
    window = 0.0
    if since_s is not None:
        window = float(since_s)
    elif t_min is not None and t_max is not None:
        window = float(t_max - t_min)
    return {"requests": requests, "page_views": views, "visitors": len(clients), "automated_page_views": auto_views,
            "errors": errors, "lines_unparsed": unparsed, "window_s": window,
            "by_path": dict(sorted(by_path.items(), key=lambda kv: (-kv[1], kv[0]))),
            "referrer_hosts": dict(sorted(referrers.items(), key=lambda kv: (-kv[1], kv[0]))[:20]),
            "first_ts": t_min, "last_ts": t_max}
