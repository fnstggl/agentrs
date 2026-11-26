"""Unauthorised network access probes (Gate W.8, SP-16): v5 worlds under the process-level network guard, and the
OS-level jails of the subprocess tools (FFmpeg/ffprobe/SoX and Chromium) that the Python-level guard cannot see.

A local TCP listener (a canary on loopback) records every connection; a jailed tool or browser that reaches it, or
anything that resolves a name or connects from the world process, is a network escape."""
from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import threading
import time

from ..framework import HELD, INTEGRITY_BREAK, Attacker, Evaluator, NotApplicable, Operator, Verdict
from .base import V5Probe, b64

HOSTILE = ("http://example.org/", "https://169.254.169.254/latest/meta-data/", "http://127.0.0.1:11111/capabilities",
           "file:///etc/passwd")


class Canary:
    """A loopback TCP listener counting connections (the address an escaping tool or page would hit)."""

    def __init__(self):
        self.s = socket.socket()
        self.s.bind(("127.0.0.1", 0))
        self.s.listen(16)
        self.s.settimeout(0.2)
        self.port = self.s.getsockname()[1]
        self.hits = 0
        self._stop = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while not self._stop:
            try:
                c, _ = self.s.accept()
                self.hits += 1
                c.close()
            except OSError:
                pass

    def close(self):
        self._stop = True
        self.t.join(1)
        self.s.close()


class _Guarded(V5Probe):
    target, owner = "world network isolation", "O"
    gates = ("W.8",)

    def run_guarded(self, fn):
        from ...realworld import NetworkGuard
        with NetworkGuard() as g:
            try:
                fn()
            finally:
                self.attempts = list(g.attempts)

    def verdict(self, ev: Evaluator) -> Verdict:
        evd = {"network_attempts": self.attempts[:20], "network_escapes": len(self.attempts),
               "calls": getattr(self, "calls", [])[:20]}
        if self.attempts:
            return Verdict(INTEGRITY_BREAK, f"the world process tried to reach the network: {self.attempts[:3]}", evd,
                           severity="critical")
        return Verdict(HELD, "no name resolution, connection or datagram left the world process", evd)


class PopulationWorldOffline(_Guarded):
    name = "v5.network.population_world_is_offline"
    attack_class = "network/egress"
    world = {"v5": "population"}
    description = ("Under a guard recording every resolve/connect/sendto of the world process, a population world "
                   "runs 6 simulated hours while an agent asks `net.fetch` for an external site, the cloud metadata "
                   "address and the instance's management API. Checked: no network attempt at all.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("netprobe", 10_000)

    def attack(self, a: Attacker) -> None:
        def go():
            self.calls = []
            for url in HOSTILE:
                r = a.act(self.att, "net.fetch", url=url)
                self.calls.append([url, r.ok, (r.error or {}).get("code")])
            a.wait(6 * 3600)
        self.run_guarded(go)


class SocialWorldOffline(_Guarded):
    name = "v5.network.social_studio_with_hostile_urls_is_offline"
    attack_class = "network/egress"
    world = {"v5": "social", "extra": ["media"]}
    description = ("Under the guard, an agent writes studio pages and HTML with external images, stylesheets, "
                   "iframes to file:// and scripts calling fetch(), renders and previews them, and publishes posts "
                   "linking the hostile URLs. Checked: no network attempt from the world process.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("netsocial", 100_000)

    def attack(self, a: Attacker) -> None:
        html = ('<img src="http://example.org/x.png"><iframe src="file:///etc/passwd"></iframe>'
                '<link rel=stylesheet href="https://example.org/a.css"><script>fetch("http://169.254.169.254/")</script>')

        def go():
            self.calls = []
            for tool, args in (("studio.page", {"title": "t", "html": html, "script": "fetch('http://example.org/')"}),
                               ("studio.write", {"format": "html", "title": "t", "html": html}),
                               ("media.render_html", {"html": html}),
                               ("posts.publish", {"text": "see " + " ".join(HOSTILE)})):
                r = a.act(self.att, tool, **args)
                self.calls.append([tool, r.ok, (r.error or {}).get("code")])
                ref = (r.result or {}).get("ref") if r.ok and isinstance(r.result, dict) else None
                if ref and tool.startswith("studio"):
                    p = a.act(self.att, "studio.preview", ref=ref)
                    self.calls.append(["studio.preview", p.ok, (p.error or {}).get("code")])
        self.run_guarded(go)


class JailedMediaToolsHaveNoNetwork(V5Probe):
    name = "v5.network.jailed_media_tools_cannot_reach_network_or_host_files"
    attack_class = "network/subprocess_tools"
    gates = ("W.8", "W.1")
    target, owner = "media tools (sandbox/isolation)", "O"
    world = {"v5": "substrate"}
    description = ("Defence in depth below FFmpeg's own protocol and format whitelists: FFmpeg is started through "
                   "the jail with a permissive whitelist (as if a whitelist bug let a crafted playlist through) and "
                   "asked to open a loopback canary over HTTP and TCP, the cloud metadata address, and /etc/passwd "
                   "and the crucible source tree; SoX is asked to read /etc/passwd. Checked: every attempt fails, the "
                   "canary sees no connection, and an ordinary decode still works.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("ffjail", 10)
        from ...sandbox import isolation
        ok, why = isolation.available()
        if not ok:
            raise NotApplicable(f"the jail cannot be applied here: {why}")

    def attack(self, a: Attacker) -> None:
        from ...sandbox import isolation
        import wave
        import struct
        can = Canary()
        d = tempfile.mkdtemp(prefix="rt-ffjail-")
        with wave.open(os.path.join(d, "a.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(struct.pack("<800h", *([0] * 800)))
        open(os.path.join(d, "list.m3u8"), "w").write(f"#EXTM3U\n#EXTINF:1,\nhttp://127.0.0.1:{can.port}/a.ts\n"
                                                      "#EXTINF:1,\nfile:///etc/passwd\n")
        wl = ["-protocol_whitelist", "file,http,https,tcp,tls,crypto,data,pipe"]
        cases = {"http_canary": ["ffmpeg", "-y", *wl, "-i", f"http://127.0.0.1:{can.port}/x", "o.wav"],
                 "tcp_canary": ["ffmpeg", "-y", *wl, "-i", f"tcp://127.0.0.1:{can.port}", "o2.wav"],
                 "metadata": ["ffmpeg", "-y", *wl, "-i", "http://169.254.169.254/latest", "o3.wav"],
                 "hls_playlist": ["ffmpeg", "-y", *wl, "-i", "list.m3u8", "o4.wav"],
                 "read_passwd": ["ffmpeg", "-y", "-f", "data", "-i", "/etc/passwd", "-map", "0", "-c", "copy",
                                 "-f", "data", "o5.bin"],
                 "read_source": ["ffmpeg", "-y", "-f", "data", "-i", os.path.abspath(__file__), "-map", "0", "-c",
                                 "copy", "-f", "data", "o6.bin"],
                 "sox_passwd": ["sox", "-t", "raw", "-r", "8000", "-e", "signed", "-b", "16", "-c", "1", "/etc/passwd",
                                "o7.wav"],
                 "ordinary_decode": ["ffprobe", "-v", "error", "-show_format", "a.wav"]}
        self.res = {}
        for k, argv in cases.items():
            cmd, cfg = isolation.tool_command(argv, d)
            try:
                p = subprocess.run(cmd, cwd=d, capture_output=True, timeout=30)
                self.res[k] = {"rc": p.returncode, "out": [f for f in ("o5.bin", "o6.bin", "o7.wav", "o.wav")
                                                           if os.path.exists(os.path.join(d, f))]}
            finally:
                os.unlink(cfg)
        time.sleep(0.3)
        self.hits = can.hits
        can.close()
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    def verdict(self, ev: Evaluator) -> Verdict:
        esc = [k for k, v in self.res.items() if k != "ordinary_decode" and v["rc"] == 0]
        evd = {"results": self.res, "canary_connections": self.hits, "network_escapes": self.hits + len(
            [k for k in esc if k in ("http_canary", "tcp_canary", "metadata", "hls_playlist")]),
               "sandbox_escapes": len([k for k in esc if k.startswith(("read", "sox"))])}
        if self.hits or esc:
            return Verdict(INTEGRITY_BREAK, f"a jailed media tool escaped: {esc}, canary hits {self.hits}", evd,
                           severity="critical")
        if self.res["ordinary_decode"]["rc"] != 0:
            raise NotApplicable("the jailed ordinary decode failed; the jail is misconfigured")
        return Verdict(HELD, f"{len(self.res) - 1} network and host-file attempts of jailed FFmpeg/SoX failed; the "
                             "canary saw no connection; ordinary decoding works", evd)


PAGE = """<html><body><h1 id=s>start</h1><script>
(async () => {
  const out = {};
  const port = %d;
  try { const r = await fetch('http://127.0.0.1:' + port + '/x'); out.fetch = r.status; } catch (e) { out.fetch = 'blocked'; }
  try { await new Promise((res, rej) => { const w = new WebSocket('ws://127.0.0.1:' + port + '/'); w.onopen = () => res(1);
        w.onerror = () => rej(1); setTimeout(() => rej(1), 1500); }); out.ws = 'open'; } catch (e) { out.ws = 'blocked'; }
  try { const pc = new RTCPeerConnection({iceServers: [{urls: 'stun:127.0.0.1:' + port}]}); pc.createDataChannel('x');
        await pc.setLocalDescription(await pc.createOffer()); await new Promise(r => setTimeout(r, 1500));
        out.ice = pc.localDescription.sdp.split('\\n').filter(l => l.startsWith('a=candidate')).length; } catch (e) { out.ice = 'blocked'; }
  try { const r = await fetch('file:///etc/passwd'); out.file = (await r.text()).slice(0, 10); } catch (e) { out.file = 'blocked'; }
  document.getElementById('s').textContent = JSON.stringify(out);
})();
</script></body></html>"""


class JailedBrowserHasNoNetwork(V5Probe):
    name = "v5.network.agent_javascript_in_jailed_browser_cannot_exfiltrate"
    attack_class = "network/browser"
    gates = ("W.8", "W.1")
    target, owner = "browser (sandbox/isolation, web.browser)", "O"
    world = {"v5": "substrate"}
    description = ("Agent-authored JavaScript runs in the jailed Chromium in both modes: the media renderer's (no "
                   "sockets but AF_UNIX) and the agent browser's (TCP only to the world's web-app port). It tries "
                   "fetch() and a WebSocket to a loopback canary on another port, WebRTC ICE (UDP) and reading "
                   "file:///etc/passwd. Checked: the canary sees no connection and every attempt fails.")
    slow = True

    def setup(self, op: Operator) -> None:
        self.att = op.join("jsjail", 10)
        from ...sandbox import isolation
        ok, why = isolation.available()
        if not ok:
            raise NotApplicable(f"the jail cannot be applied here: {why}")
        try:
            import playwright  # noqa: F401
        except ImportError:
            raise NotApplicable("Playwright is not installed")

    def attack(self, a: Attacker) -> None:
        from playwright.sync_api import sync_playwright
        from ...sandbox import isolation
        can = Canary()
        app = Canary()                       # stands in for the world's web app (the one allowed port)
        self.res = {}
        try:
            with sync_playwright() as pw:
                for mode, ports in (("renderer", ()), ("agent_browser", (app.port,))):
                    exe, kw = isolation.browser_executable(f"rt-{mode}", tcp_ports=ports)
                    b = pw.chromium.launch(headless=True, executable_path=exe, args=["--disable-gpu"], **kw)
                    try:
                        page = b.new_page(java_script_enabled=True)
                        page.set_content(PAGE % can.port)
                        page.wait_for_function("document.getElementById('s').textContent !== 'start'", timeout=15000)
                        self.res[mode] = page.inner_text("#s")
                    finally:
                        b.close()
        finally:
            time.sleep(0.3)
            self.hits = can.hits
            can.close()
            app.close()

    def verdict(self, ev: Evaluator) -> Verdict:
        import json as _j
        bad = []
        for mode, txt in self.res.items():
            try:
                o = _j.loads(txt)
            except ValueError:
                o = {"raw": txt}
            self.res[mode] = o
            for k in ("fetch", "ws", "file"):
                if o.get(k) not in ("blocked", None):
                    bad.append(f"{mode}:{k}")
            if isinstance(o.get("ice"), int) and o["ice"] > 0:
                bad.append(f"{mode}:ice")
        evd = {"results": self.res, "canary_connections": self.hits, "network_escapes": self.hits + len(bad)}
        if self.hits or bad:
            return Verdict(INTEGRITY_BREAK, f"agent JavaScript reached the network or host files: {bad}, canary "
                           f"{self.hits}", evd, severity="critical")
        return Verdict(HELD, "fetch, WebSocket, WebRTC and file:// all failed in both browser modes; the canary saw "
                             "no connection", evd)


def probes() -> list:
    return [PopulationWorldOffline(), SocialWorldOffline(), JailedMediaToolsHaveNoNetwork(),
            JailedBrowserHasNoNetwork()]
