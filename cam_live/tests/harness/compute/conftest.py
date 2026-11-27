"""Fixtures for the EC tests (sandbox, background jobs, visual executor, grounding)."""
import os
import shutil
import threading
import uuid

import pytest

from cam_live import contracts as c
from cam_live.harness import sandbox as sbx

_OPT_IN = ("browser", "privileged", "network")


def pytest_collection_modifyitems(config, items):
    """browser / privileged / network tests run only when named with -m (gpu / slow: see the root conftest)."""
    expr = config.getoption("-m") or ""
    for item in items:
        for mark in _OPT_IN:
            if mark in item.keywords and mark not in expr:
                item.add_marker(pytest.mark.skip(reason=f"run with -m {mark}"))
        if "privileged" in item.keywords and os.geteuid() != 0:
            item.add_marker(pytest.mark.skip(reason="needs root"))


class Policy:
    """A recording stand-in for PolicyEngine.check_primitive (the authorization service belongs to EA)."""

    def __init__(self, rule=None):
        self.rule = rule or (lambda op, action: c.AuthVerdict.ALLOW)
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, op, action):
        verdict = self.rule(op, action)
        with self._lock:
            self.calls.append((op.op, dict(op.arguments), verdict))
        return c.AuthorizationDecision(verdict=verdict, reasons=[] if verdict == c.AuthVerdict.ALLOW else ["test policy"])

    def ops(self, name=None):
        return [x for x in self.calls if name is None or x[0] == name]


@pytest.fixture(scope="session")
def sbx_base():
    base = f"/var/tmp/cam-live-sbx-test-{uuid.uuid4().hex[:10]}"
    os.makedirs(base, exist_ok=True)
    os.chmod(base, 0o711)
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def sbx_cfg(sbx_base):
    def make(**kw):
        kw.setdefault("allow_same_uid", os.geteuid() != 0)      # non-root hosts: test the weaker mode, reported
        return sbx.SandboxConfig(base_dir=sbx_base, **kw)
    return make


@pytest.fixture
def sandbox(sbx_base, sbx_cfg):
    made = []

    def make(**kw):
        s = sbx.Sandbox(os.path.join(sbx_base, "ws", uuid.uuid4().hex[:10]), sbx_cfg(**kw))
        if not s.available:
            pytest.skip(f"sandbox unavailable on this host: {s.unavailable_reason}")
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


@pytest.fixture
def make_ctx(tmp_path):
    def make(policy=None, run_dir=None, clock=None, **kw):
        sink = []
        ctx = c.ExecContext(clock=clock or c.WallClock(), check=policy if policy is not None else Policy(),
                            obs_sink=sink.append, run_dir=str(run_dir or tmp_path / "run"),
                            cancel=kw.pop("cancel", threading.Event()), **kw)
        os.makedirs(ctx.run_dir, exist_ok=True)
        ctx.sink = sink
        return ctx
    return make


def action(op, modality=c.Modality.CODE, timeout_s=60.0, verify=None, **arguments):
    return c.ActionSpec(action_id=c.new_id("act"), operation=op, modality=modality, arguments=arguments,
                        timeout_s=timeout_s, verify=verify or c.VerificationCondition())


@pytest.fixture
def act():
    return action


@pytest.fixture
def policy_cls():
    return Policy


def session_processes(sid_or_uid_filter=None):
    """Live (non-zombie) processes whose command line contains ``sid_or_uid_filter``."""
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
            with open(f"/proc/{name}/stat", "rb") as f:
                state = f.read().rsplit(b")", 1)[1].split()[0]
        except OSError:
            continue
        if state != b"Z" and sid_or_uid_filter in cmd:
            out.append((int(name), cmd))
    return out


@pytest.fixture
def procs_matching():
    return session_processes


# ---------------------------------------------------------------------------------------------------------- screens
class PageScreen:
    """A Playwright-page-backed ``ScreenProvider`` (the real browser screen belongs to workstream EB)."""

    def __init__(self, page):
        self.page = page

    def screenshot(self) -> bytes:
        return self.page.screenshot(type="png", full_page=False)

    def size(self):
        v = self.page.viewport_size
        return int(v["width"]), int(v["height"])

    def click(self, x, y) -> None:
        self.page.mouse.click(x, y)

    def double_click(self, x, y) -> None:
        self.page.mouse.dblclick(x, y)

    def drag(self, x1, y1, x2, y2) -> None:
        self.page.mouse.move(x1, y1)
        self.page.mouse.down()
        self.page.mouse.move(x2, y2, steps=8)
        self.page.mouse.up()

    def scroll(self, dx, dy) -> None:
        self.page.mouse.wheel(dx, dy)

    def type(self, text) -> None:
        self.page.keyboard.type(text)

    def key(self, combo) -> None:
        self.page.keyboard.press(combo)

    def url(self) -> str:
        return self.page.url


@pytest.fixture(scope="session")
def chromium():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chromium cannot be started here: {e}")
        yield browser
        browser.close()


@pytest.fixture
def open_page(chromium):
    contexts = []

    def make(html, width=1280, height=800):
        context = chromium.new_context(viewport={"width": width, "height": height}, device_scale_factor=1)
        contexts.append(context)
        page = context.new_page()
        page.set_content(html, wait_until="load")
        return page
    yield make
    for context in contexts:
        context.close()


class FrameScreen:
    """A tiny software screen (numpy frame buffer) for CPU-only tests: one button that toggles colour, a text line
    that grows when keys arrive, a square that follows a drag, and a pattern that moves with the wheel."""

    def __init__(self, width=320, height=240):
        import numpy as np
        self.np = np
        self.w, self.h = width, height
        self.on = False
        self.text = ""
        self.keys = []
        self.square = (200, 150)
        self.offset = 0
        self.clicks = []
        self.double_clicks = []
        self.button = (40, 40, 120, 50)                  # x, y, w, h

    def _frame(self):
        np = self.np
        f = np.full((self.h, self.w, 3), 235, dtype=np.uint8)
        rows = (np.arange(self.h) + self.offset) % 40 < 2
        f[rows, 300:min(316, self.w)] = (90, 90, 90)
        x, y, w, h = self.button
        f[y:y + h, x:x + w] = (30, 160, 60) if self.on else (40, 100, 220)
        f[120:126, 40:40 + min(8 * len(self.text), 240)] = (20, 20, 20)
        sx, sy = self.square
        f[sy:sy + 20, sx:sx + 20] = (200, 60, 60)
        return f

    def screenshot(self) -> bytes:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(self._frame()).save(buf, "PNG")
        return buf.getvalue()

    def size(self):
        return self.w, self.h

    def _hit(self, x, y):
        bx, by, bw, bh = self.button
        return bx <= x < bx + bw and by <= y < by + bh

    def click(self, x, y) -> None:
        self.clicks.append((x, y))
        if self._hit(x, y):
            self.on = not self.on

    def double_click(self, x, y) -> None:
        self.double_clicks.append((x, y))

    def drag(self, x1, y1, x2, y2) -> None:
        sx, sy = self.square
        if sx <= x1 < sx + 20 and sy <= y1 < sy + 20:
            self.square = (int(x2) - 10, int(y2) - 10)

    def scroll(self, dx, dy) -> None:
        self.offset += int(dy)

    def type(self, text) -> None:
        self.text += text

    def key(self, combo) -> None:
        self.keys.append(combo)
        if combo == "Backspace":
            self.text = self.text[:-1]


@pytest.fixture
def frame_screen():
    return FrameScreen()


@pytest.fixture
def page_screen_cls():
    return PageScreen
