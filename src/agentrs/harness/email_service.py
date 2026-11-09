"""Managed email for agents (operator-provided domain and mailbox).

* Backends: ``SmtpImapBackend`` (any provider offering SMTP + IMAP, password or OAuth2 XOAUTH2 via
  secret references) and ``LocalMailServer`` (an ``aiosmtpd`` SMTP server on localhost with a local
  mailbox — the development / test simulator).
* ``EmailConnector`` exposes three operations through the tool gateway: ``read_messages`` (tier 0),
  ``send_transactional`` (tier 2: only to the operator's own domains, e.g. reports / notifications)
  and ``send_external`` (tier 3: anyone else → approval queue by default). Recipients are classified
  by the harness, not by the model, and a transactional send to an outside address is refused.
* Every message gets a Message-ID; replies are matched to threads (In-Reply-To / References) and to
  the originating task; delivery failures (SMTP errors, bounces from the mailer daemon) are recorded;
  replies are tracked over time through ``fetch_outcome`` like any other delayed outcome.
* Inbound account-service mail (verification links, codes) is surfaced to the operator as a human
  verification step. The harness never follows verification links or reads codes to complete a
  provider's verification on its own.
"""

from __future__ import annotations

import email
import email.utils
import re
import smtplib
import ssl
import threading
import time
import uuid
from email.message import EmailMessage as MIMEMessage
from typing import Any, ClassVar, Optional

from agentrs.harness.connectors.base import ConnectorError, PlatformConnector
from agentrs.harness.types import OperationSpec, Tier

VERIFICATION = re.compile(r"(?i)(verify your|verification code|confirm your (email|account)|one-time (code|password)|"
                          r"security code|reset your password)")


class LocalMailServer:
    """SMTP server on 127.0.0.1 that stores everything it receives (development / tests)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        from aiosmtpd.controller import Controller

        self.messages: list[dict] = []
        self._lock = threading.Lock()
        outer = self

        class Handler:
            async def handle_DATA(self, server, session, envelope):  # noqa: N802 - aiosmtpd API
                msg = email.message_from_bytes(envelope.content)
                with outer._lock:
                    outer.messages.append({"from": envelope.mail_from, "rcpt": list(envelope.rcpt_tos), "msg": msg,
                                           "raw": envelope.content, "t": time.time()})
                if any(r.endswith("@bounce.invalid") for r in envelope.rcpt_tos):
                    return "550 mailbox unavailable"
                return "250 OK"

        import socket

        if port == 0:
            with socket.socket() as so:
                so.bind((host, 0))
                port = so.getsockname()[1]
        self.host, self.port = host, port
        self.controller = Controller(Handler(), hostname=host, port=port)
        self.controller.start()

    def stop(self) -> None:
        self.controller.stop()

    def inbox(self, address: str) -> list[dict]:
        with self._lock:
            return [m for m in self.messages if address in m["rcpt"]]

    def deliver(self, msg: MIMEMessage, to: str) -> None:
        """Simulate an inbound message (a reply from a person, a provider notification, a bounce)."""
        with smtplib.SMTP(self.host, self.port) as s:
            s.send_message(msg, to_addrs=[to])


class SmtpImapBackend:
    def __init__(self, smtp_host: str, smtp_port: int, imap_host: Optional[str], address: str, username_ref: str,
                 password_ref: str, resolve, starttls: bool = True, imap_port: int = 993, local: Optional[LocalMailServer] = None):
        self.smtp_host, self.smtp_port, self.imap_host, self.imap_port = smtp_host, smtp_port, imap_host, imap_port
        self.address, self.username_ref, self.password_ref, self.resolve = address, username_ref, password_ref, resolve
        self.starttls = starttls
        self.local = local  # when set, inbound mail is read from the local server instead of IMAP

    def send(self, msg: MIMEMessage, recipients: list[str]) -> dict:
        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20) as s:
            if self.starttls:
                s.starttls(context=ssl.create_default_context())
                s.login(self.resolve(self.username_ref), self.resolve(self.password_ref))
            refused = s.send_message(msg, to_addrs=recipients)
        return {"refused": {k: list(v) for k, v in (refused or {}).items()}}

    def fetch(self, since: float) -> list[MIMEMessage]:
        if self.local is not None:
            return [m["msg"] for m in self.local.inbox(self.address) if m["t"] >= since]
        import imaplib

        out = []
        with imaplib.IMAP4_SSL(self.imap_host, self.imap_port) as im:
            im.login(self.resolve(self.username_ref), self.resolve(self.password_ref))
            im.select("INBOX", readonly=True)
            day = time.strftime("%d-%b-%Y", time.gmtime(since))
            _, ids = im.search(None, f'(SINCE "{day}")')
            for i in (ids[0] or b"").split():
                _, data = im.fetch(i, "(RFC822)")
                out.append(email.message_from_bytes(data[0][1]))
        return out


class EmailConnector(PlatformConnector):
    provider = "email"
    db: ClassVar[Any] = None
    backend_factory: ClassVar[Any] = None  # fn(ctx) -> backend
    internal_domains: ClassVar[list[str]] = []
    operations = {
        "read_messages": OperationSpec(name="read_messages", tier=Tier.OBSERVE, mutates=False, optional_params=["since"]),
        "send_transactional": OperationSpec(name="send_transactional", tier=Tier.REVERSIBLE, mutates=True,
                                            params=["to", "subject", "body"], optional_params=["in_reply_to"],
                                            content_param="body", reversible=False),
        "send_external": OperationSpec(name="send_external", tier=Tier.EXTERNAL, mutates=True,
                                       params=["to", "subject", "body"], optional_params=["in_reply_to"],
                                       content_param="body", reversible=False),
    }

    def __init__(self, ctx):
        super().__init__(ctx)
        self.backend = self.backend_factory(ctx)
        self.address = ctx.account["external_id"]
        self.task_id: Optional[str] = None

    @classmethod
    def classify(cls, recipients: list[str]) -> str:
        doms = [r.rsplit("@", 1)[-1].lower() for r in recipients]
        return "transactional" if all(d in cls.internal_domains for d in doms) else "external"

    async def validate_credentials(self) -> dict[str, Any]:
        self.backend.fetch(time.time() - 60)
        return {"ok": True, "address": self.address}

    def _store(self, **row) -> None:
        from agentrs.harness.db import EmailMessage

        with self.db.session() as s:
            s.add(EmailMessage(account_id=self.ctx.account["id"], **row))

    async def _send(self, to, subject, body, category: str, in_reply_to: Optional[str]) -> dict:
        rcpts = [to] if isinstance(to, str) else list(to)
        if self.classify(rcpts) != category:
            raise ConnectorError(f"recipients are {self.classify(rcpts)}, not {category}", 403)
        domain = self.address.split("@")[-1]
        mid = f"<{uuid.uuid4().hex}@{domain}>"
        msg = MIMEMessage()
        msg["From"], msg["To"], msg["Subject"], msg["Message-ID"] = self.address, ", ".join(rcpts), subject, mid
        msg["Date"] = email.utils.formatdate()
        msg["X-Agentrs-Agent"] = "automated agent (operator: see account record)"  # disclosure header
        thread = mid
        if in_reply_to:
            msg["In-Reply-To"] = msg["References"] = in_reply_to
            thread = self._thread_of(in_reply_to) or in_reply_to
        msg.set_content(body)
        try:
            res = self.backend.send(msg, rcpts)
            status, err = ("failed", f"refused: {res['refused']}") if res.get("refused") else ("sent", "")
        except smtplib.SMTPException as e:
            status, err = "failed", f"{type(e).__name__}: {e}"
        self._store(direction="out", category=category, thread_id=thread, message_id=mid, in_reply_to=in_reply_to,
                    task_id=self.task_id, sender=self.address, recipients=rcpts, subject=subject, body=body,
                    status=status, error=err)
        if status != "sent":
            raise ConnectorError(f"delivery failed: {err}", 502)
        return {"object_id": mid, "thread_id": thread}

    def _thread_of(self, message_id: str) -> Optional[str]:
        from agentrs.harness.db import EmailMessage
        from sqlalchemy import select

        with self.db.session() as s:
            m = s.scalars(select(EmailMessage).where(EmailMessage.message_id == message_id)).first()
            return m.thread_id if m else None

    async def op_send_transactional(self, to, subject, body, in_reply_to=None) -> dict:
        return await self._send(to, subject, body, "transactional", in_reply_to)

    async def op_send_external(self, to, subject, body, in_reply_to=None) -> dict:
        return await self._send(to, subject, body, "external", in_reply_to)

    def ingest(self, since: float = 0.0) -> list[dict]:
        """Pull new inbound mail, thread it, flag bounces and verification mail (for the operator)."""
        from agentrs.harness.db import EmailMessage
        from sqlalchemy import select

        out = []
        for m in self.backend.fetch(since):
            mid = m.get("Message-ID") or f"<{uuid.uuid4().hex}@inbound>"
            with self.db.session() as s:
                if s.scalars(select(EmailMessage).where(EmailMessage.message_id == mid)).first():
                    continue
            body = m.get_payload(decode=True).decode(errors="replace") if not m.is_multipart() else str(m.get_payload(0))
            irt = m.get("In-Reply-To")
            bounce = (m.get("From", "").lower().startswith("mailer-daemon") or "delivery status notification"
                      in (m.get("Subject") or "").lower())
            category = "notification" if (bounce or VERIFICATION.search((m.get("Subject") or "") + body)) else "external"
            thread = self._thread_of(irt) if irt else None
            self._store(direction="in", category=category, thread_id=thread or mid, message_id=mid, in_reply_to=irt,
                        task_id=None, sender=m.get("From", ""), recipients=[self.address],
                        subject=m.get("Subject", ""), body=body[:20000], status="received",
                        error="bounce" if bounce else "")
            if bounce and irt:
                with self.db.session() as s:
                    orig = s.scalars(select(EmailMessage).where(EmailMessage.message_id == irt)).first()
                    if orig is not None:
                        orig.status, orig.error = "failed", "bounced"
            out.append({"message_id": mid, "thread_id": thread or mid, "category": category,
                        "needs_human": category == "notification" and not bounce})
        return out

    async def op_read_messages(self, since=0.0) -> dict:
        new = self.ingest(float(since))
        return {"new": new}

    async def fetch_outcome(self, object_ref: str) -> dict[str, Any]:
        from agentrs.harness.db import EmailMessage
        from sqlalchemy import select

        self.ingest(time.time() - 30 * 86400)
        with self.db.session() as s:
            orig = s.scalars(select(EmailMessage).where(EmailMessage.message_id == object_ref)).first()
            if orig is None:
                raise ConnectorError("unknown message", 404)
            replies = list(s.scalars(select(EmailMessage).where(EmailMessage.thread_id == orig.thread_id,
                                                                EmailMessage.direction == "in",
                                                                EmailMessage.category == "external")))
            state = "removed" if orig.status == "failed" else "visible"
        first = min((r.created_at for r in replies), default=None)
        occ = {"first_response": bool(replies), "removed": state == "removed"}
        return {"counts": {"responses": len(replies)}, "state": state, "occurred": occ,
                "occurred_at": {"first_response": first} if first else {}, "raw": {"status": orig.status}}
