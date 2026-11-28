"""Shared helpers for the AU tests."""
import base64
import hashlib
import io
import json
import os
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cam_live import contracts as C
from cam_live.harness.auth import liveauth

from ea_helpers import ALLOW, DENY, NEEDS, act, install_operator_key, live_record, live_task  # noqa: F401

PROFILE = "routine-reversible-1"
GUILD = "123456789012345678"
HOSTING = "https://deploy.host.example/*"
INTEGRATION = "https://api.integration.example/*"


def standing_record(t, now, **over):
    """A signed-record body with CAM's designated infrastructure pre-authorized (what ``authorize_live.py
    --standing`` writes): its Discord server, a hosting API, an integration; public web reading."""
    owned = [f"account:discord:{GUILD}", "channel:discord:*", HOSTING, INTEGRATION]
    rec = live_record(t, now, owned=owned, read=["https://*", "pypi.org", "files.pythonhosted.org"],
                      preauthorized=list(owned),
                      standing={"profile": PROFILE, "code_network": ["pypi.org", "files.pythonhosted.org"]})
    rec.update(over)
    return rec


def sign_register(svc, tmp, rec):
    priv = install_operator_key(svc, tmp)
    signed = liveauth.sign_live_authorization(rec, priv)
    svc.register_live_authorization(signed, "operator")
    return signed


def decide(svc, t, op, modality=None, **arguments):
    return svc.authorize(act(op, modality, arguments=arguments), t, {})


def approve_and_retry(svc, t, a):
    d = svc.authorize(a, t, {})
    assert d.verdict == NEEDS and d.approval_id, (d.verdict, d.reasons)
    svc.resolve_approval(d.approval_id, True, "operator")
    d2 = svc.authorize(a, t, {})
    assert d2.verdict == ALLOW, d2.reasons
    return d, d2


# ---------------------------------------------------------------------------------------------- a local package index
def build_wheel(name, version, module_source):
    """A minimal pure-Python wheel (PEP 427) built in memory."""
    dist = f"{name}-{version}.dist-info"
    files = {f"{name}/__init__.py": module_source.encode(),
             f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
             f"{dist}/WHEEL": b"Wheel-Version: 1.0\nGenerator: au-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    record = []
    for path, data in files.items():
        dig = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={dig},{len(data)}")
    record.append(f"{dist}/RECORD,,")
    files[f"{dist}/RECORD"] = ("\n".join(record) + "\n").encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path, data in files.items():
            z.writestr(path, data)
    return f"{name}-{version}-py3-none-any.whl", buf.getvalue()


class PackageIndex:
    """A PEP 503 simple index on 127.0.0.1 serving the given wheels; records every request."""

    def __init__(self, wheels):
        self.wheels = dict(wheels)                     # filename -> bytes
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                outer.requests.append(self.path)
                body, ctype = b"not found", "text/plain"
                status = 404
                parts = [p for p in self.path.split("?")[0].split("/") if p]
                if parts[:1] == ["simple"] and len(parts) == 2:
                    proj = parts[1].lower().replace("_", "-")
                    links = [f'<a href="/files/{fn}#sha256={hashlib.sha256(data).hexdigest()}">{fn}</a>'
                             for fn, data in outer.wheels.items() if fn.split("-")[0].lower().replace("_", "-") == proj]
                    if links:
                        status, ctype = 200, "text/html"
                        body = ("<!DOCTYPE html><html><body>" + "".join(links) + "</body></html>").encode()
                elif parts[:1] == ["files"] and len(parts) == 2 and parts[1] in outer.wheels:
                    status, ctype, body = 200, "application/octet-stream", outer.wheels[parts[1]]
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.host = f"127.0.0.1:{self.port}"
        self.url = f"http://{self.host}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def files_containing(root, needle):
    """Every file under ``root`` whose bytes contain ``needle`` (also base64 / hex / JSON-escaped forms)."""
    forms = {needle.encode(), base64.b64encode(needle.encode()), needle.encode().hex().encode(),
             json.dumps(needle)[1:-1].encode()}
    hits = []
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            if any(x in data for x in forms):
                hits.append(os.path.relpath(p, root))
    return sorted(hits)
