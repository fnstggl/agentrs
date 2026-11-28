"""G2-runtime probe: run-record redaction (LX-20) on shapes G-rt's p07 did not plant, written through the runtime's own
RunLog, then every byte of the run directory searched for the planted values.

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_redaction.py

Fake credentials are assembled at run time (no token-shaped literal in this file).
"""
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import Report, scratch  # noqa: E402

from cam_live.contracts import SimClock  # noqa: E402
from cam_live.runtime.logs import Redactor, RunLog, load_salt  # noqa: E402

R = Report("probe_redaction")
run = os.path.join(scratch("red"), "run")
os.makedirs(run, exist_ok=True)
clock = SimClock()
perms = {"store_message_content": False, "store_member_identifiers": False}
red = Redactor.from_config({}, load_salt(run), perms)
log = RunLog(run, clock, provenance="staging", durable=False, redactor=red)


def b64(s):
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


jwt = ".".join([b64('{"alg":"HS256","typ":"JWT"}'), b64('{"sub":"operator","scope":"admin"}'),
                b64("x" * 32) + "Q"])
basic = base64.b64encode(b"operator:" + ("p" + "w" * 10).encode()).decode()
oauth = "ya29." + "A0ARrdaM" + "k" * 40
hexsecret = "9f" * 20
member_int = 1_100_000_000_000_000_123
planted = {
    "JWT in page text": ("observation", {"text": f"localStorage session = {jwt}"}, jwt),
    "JWT as a bare value under an unlisted key": ("observation", {"data": {"value": jwt}}, jwt),
    "HTTP Basic credentials in a header dump": ("observation", {"text": f"Authorization: Basic {basic}"}, basic),
    "OAuth access token (ya29.*) in text": ("observation", {"text": f"got {oauth} from the redirect"}, oauth),
    "40-hex secret under key 'value'": ("observation", {"data": {"name": "api", "value": hexsecret}}, hexsecret),
    "member id as an integer under an unlisted key (owner_id)": ("observation", {"data": {"owner_id": member_int}},
                                                                  str(member_int)),
    "US phone number 555-123-4567": ("observation", {"text": "call me at 555-123-4567 tonight"}, "555-123-4567"),
    "US phone number (555) 123-4567": ("observation", {"text": "phone (555) 123-4567"}, "123-4567"),
    "phone number written without separators +15551234567": ("observation", {"text": "sms +15551234567"},
                                                             "5551234567"),
    "IPv6 address of a member's connection": ("observation", {"text": "from 2001:db8:85a3::8a2e:370:7334"},
                                              "2001:db8:85a3::8a2e:370:7334"),
}
for label, (kind, data, _value) in planted.items():
    log.event("g2_planted", {"label": label, **data})
log.close()

blob = b""
for root, _d, files in os.walk(run):
    for fn in files:
        with open(os.path.join(root, fn), "rb") as f:
            blob += f.read()
for label, (_k, _data, value) in planted.items():
    R.check(f"{label}: not in any file of the run directory", value.encode() not in blob)

# over-redaction of operator-facing text (should stay readable: no secret in it)
benign = {"stop reason the runtime writes for a revoked authorization":
              "stop condition authorization_revoked: access for this task has been revoked by the operator",
          "a version number": "client library 2.10.3.1",
          "prose with 'token:'": "the token: GUILD_MEMBERS intent is missing"}
for label, text in benign.items():
    out = red.text(text)
    R.check(f"over-redaction — {label}: text without a secret is left readable", out == text, {"in": text, "out": out})
sys.exit(R.done())
