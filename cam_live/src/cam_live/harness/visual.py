"""cam_live.harness.visual — screenshot-based computer use: the ``visual`` execution modality (workstream EC).

Observations are screenshots (viewport, full page, crops). Actions are mouse and keyboard primitives addressed by
pixel coordinates of *one particular screenshot*. Before any primitive is sent to the screen the executor checks,
in this order:

1. the action names the **current** screenshot observation of that screen (anything older is ``stale``);
2. the coordinates lie inside that screenshot, and the screen still has that size;
3. the screen still looks the way the observation shows it where it matters — the patch around the target point
   for pointer actions, the whole frame for keyboard and wheel actions (otherwise ``stale``: the fresh screenshot
   becomes the current observation and is returned so the caller can decide again);
4. ``ExecContext.check`` allows the primitive (a denial is final: nothing is sent, nothing is retried).

After the primitive the screen is observed again; the result carries the measured change and the outcome of
``ActionSpec.verify``. Success is never assumed: ``status == ok`` means the gesture was delivered, ``verified``
says whether the expected effect is visible.

A **grounded** action (``visual.click_target``) asks a :class:`contracts.VisualGrounder` where an intended target
is. The grounder's answer is a proposal — a point and a confidence, nothing else is read from it — which goes
through exactly the same validation and authorization before the click, and through the same re-observation after.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import inspect
import io
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple, runtime_checkable

from .. import contracts as c

__all__ = ["ScreenProvider", "VisualConfig", "VisualExecutor", "frame_change"]


@runtime_checkable
class ScreenProvider(Protocol):
    """What the visual executor needs from a screen (a browser tab, a desktop): pixels in, gestures out."""

    def screenshot(self) -> bytes: ...                       # PNG of what is visible now
    def size(self) -> Tuple[int, int]: ...                   # (width, height) in screenshot pixels
    def click(self, x: float, y: float) -> None: ...
    def double_click(self, x: float, y: float) -> None: ...
    def drag(self, x1: float, y1: float, x2: float, y2: float) -> None: ...
    def scroll(self, dx: float, dy: float) -> None: ...
    def type(self, text: str) -> None: ...
    def key(self, combo: str) -> None: ...


@dataclass
class VisualConfig:
    max_stored: int = 24                 # screenshots kept for crops, staleness checks and effect measurement
    patch_px: int = 36                   # half-size of the square around a target point compared for staleness
    patch_tolerance: float = 4.0         # mean absolute difference (0..255) above which the patch has changed
    frame_tolerance: float = 0.02        # fraction of changed pixels above which the whole frame has changed
    pixel_threshold: int = 24            # per-pixel summed RGB difference that counts as "changed"
    min_effect_fraction: float = 0.0002  # changed fraction below which an action had no observable effect
    settle_s: float = 0.12               # wait before the verifying screenshot
    settle_max_s: float = 1.2            # keep re-observing until two consecutive frames agree, at most this long
    min_confidence: float = 0.0          # grounder proposals below this are rejected
    grounder_order: Tuple[str, ...] = ("goclick-base", "mai-ui-2b")
    max_ground_attempts: int = 2
    gpu_lock_timeout_s: float = 120.0    # wait for the shared GPU lock before a grounder that needs it
    release_heavy_after_use: bool = True
    full_page_max_screens: int = 12
    max_text_len: int = 4000
    save_images: bool = True


# ==============================================================================================================
# Pixels
# ==============================================================================================================
def _decode(png: bytes):
    import numpy as np
    from PIL import Image
    img = Image.open(io.BytesIO(png))
    img.load()
    return np.asarray(img.convert("RGB"))


def _encode(arr) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def frame_change(before, after, threshold: int = 24) -> Dict[str, Any]:
    """Measured difference of two frames: fraction of changed pixels and the bounding box of the change."""
    import numpy as np
    if before.shape != after.shape:
        return {"changed_fraction": 1.0, "bbox": [0, 0, int(after.shape[1]), int(after.shape[0])], "resized": True}
    diff = np.abs(before.astype(np.int16) - after.astype(np.int16)).sum(-1) > threshold
    n = int(diff.sum())
    if n == 0:
        return {"changed_fraction": 0.0, "bbox": None, "resized": False}
    ys, xs = np.nonzero(diff)
    return {"changed_fraction": n / diff.size, "resized": False,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]}


def _patch_diff(a, b, x: float, y: float, r: int) -> float:
    import numpy as np
    if a.shape != b.shape:
        return 255.0
    h, w = a.shape[:2]
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    return float(np.abs(a[y0:y1, x0:x1].astype(np.int16) - b[y0:y1, x0:x1].astype(np.int16)).mean())


@dataclass
class _Shot:
    obs_id: str
    screen: str
    mode: str                         # "viewport" | "full_page" | "crop"
    png: bytes
    arr: Any
    ts: float
    sha256: str
    path: str = ""
    origin: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.arr.shape[1])

    @property
    def height(self) -> int:
        return int(self.arr.shape[0])


class _Reject(Exception):
    def __init__(self, failure: str, detail: str, status: str = c.ResultStatus.FAILED, obs: Optional[c.Observation] = None):
        super().__init__(detail)
        self.failure, self.detail, self.status, self.obs = failure, detail, status, obs


@contextlib.contextmanager
def _shared_gpu_lock(timeout_s: float) -> Iterator[bool]:
    """The lock every tenant of this GPU takes for work above 4 GB (``cam_live.config.GPU_LOCK_PATH``)."""
    from ..config import GPU_LOCK_PATH
    os.makedirs(os.path.dirname(GPU_LOCK_PATH), exist_ok=True)
    f = open(GPU_LOCK_PATH, "a+")
    got = False
    end = time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
                break
            except BlockingIOError:
                if time.monotonic() >= end:
                    break
                time.sleep(1.0)
        yield got
    finally:
        if got:
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


# ==============================================================================================================
# Executor
# ==============================================================================================================
_NUM = {"type": "number"}
_SCREEN = {"type": "string"}
_OBS = {"type": "string", "minLength": 1}
_POINTER_OPS = ("visual.click", "visual.double_click")
_KEY_RE = re.compile(r"[A-Za-z0-9+_\-]{1,64}\Z")


class VisualExecutor:
    """``contracts.ModalityExecutor`` for modality ``visual``.

    ``screen`` is one :class:`ScreenProvider`; ``screen_factory(name, action, ctx)`` returns the provider for a
    named screen (for example a browser session bound to this action's authorization context). ``grounders`` maps
    names to ``VisualGrounder`` objects; names missing there are created by ``grounder_factory(name)`` on first
    use (default: :func:`cam_live.harness.grounding.load_grounder`, a separate process per model).
    """

    modality = c.Modality.VISUAL

    def __init__(self, screen: Optional[ScreenProvider] = None, *,
                 screen_factory: Optional[Callable[[str, c.ActionSpec, c.ExecContext], ScreenProvider]] = None,
                 grounders: Optional[Dict[str, c.VisualGrounder]] = None,
                 grounder_factory: Optional[Callable[[str], c.VisualGrounder]] = None,
                 cfg: Optional[VisualConfig] = None):
        self.cfg = cfg or VisualConfig()
        self._screen, self._factory = screen, screen_factory
        self._grounders: Dict[str, c.VisualGrounder] = dict(grounders or {})
        self._grounder_factory = grounder_factory
        self._shots: "OrderedDict[str, _Shot]" = OrderedDict()
        self._current: Dict[str, str] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------------------------------ contract
    def capabilities(self) -> Dict[str, Any]:
        return {"modality": self.modality, "available": self._screen is not None or self._factory is not None,
                "observations": ["viewport", "full_page", "crop"],
                "primitives": ["click", "double_click", "drag", "scroll", "type", "key"],
                "validation": ["current observation id", "screen bounds", "target patch unchanged",
                               "ExecContext.check per primitive", "re-observation after every action"],
                "grounders": {"configured": sorted(self._grounders), "order": list(self.cfg.grounder_order)},
                "desktop": "any ScreenProvider; this host has no display server, only browser screens exist"}

    def tools(self) -> List[c.ToolDescriptor]:
        ok = self._screen is not None or self._factory is not None
        why = "" if ok else "no screen is attached to the visual executor"

        def tool(name: str, desc: str, props: Dict[str, Any], required: List[str], side: str = "external_write",
                 seconds: float = 0.4) -> c.ToolDescriptor:
            return c.ToolDescriptor(
                name=name, modality=self.modality, description=desc, level=c.Level.PRIMITIVE,
                arg_schema={"type": "object", "properties": dict(props, screen=_SCREEN), "required": required,
                            "additionalProperties": False},
                auth_scope="visual", side_effects=side, est_cost=c.Cost(seconds=seconds, primitive_ops=1),
                available=ok, unavailable_reason=why)

        at = "pixel coordinates of the screenshot named by observation_id, which must be the current one"
        return [
            tool("visual.screenshot", "Take a screenshot of the screen: the visible viewport, or the whole page "
                 "stitched from top to bottom (mode='full_page'). Returns an image observation with its id.",
                 {"mode": {"type": "string", "enum": ["viewport", "full_page"]}}, [], side="external_read"),
            tool("visual.crop", "Cut a rectangle out of an earlier screenshot observation and return it as an image.",
                 {"observation_id": _OBS, "x": _NUM, "y": _NUM, "width": _NUM, "height": _NUM},
                 ["observation_id", "x", "y", "width", "height"], side="none", seconds=0.01),
            tool("visual.click", f"Click the left mouse button at {at}.",
                 {"observation_id": _OBS, "x": _NUM, "y": _NUM}, ["observation_id", "x", "y"]),
            tool("visual.double_click", f"Double-click the left mouse button at {at}.",
                 {"observation_id": _OBS, "x": _NUM, "y": _NUM}, ["observation_id", "x", "y"]),
            tool("visual.drag", f"Press the left mouse button at (x1, y1), move to (x2, y2) and release; {at}.",
                 {"observation_id": _OBS, "x1": _NUM, "y1": _NUM, "x2": _NUM, "y2": _NUM},
                 ["observation_id", "x1", "y1", "x2", "y2"]),
            tool("visual.scroll", "Turn the mouse wheel by (dx, dy) pixels, optionally with the pointer at (x, y).",
                 {"observation_id": _OBS, "dx": _NUM, "dy": _NUM, "x": _NUM, "y": _NUM}, ["observation_id", "dy"]),
            tool("visual.type", "Type text on the keyboard into whatever has the input focus.",
                 {"observation_id": _OBS, "text": {"type": "string"}}, ["observation_id", "text"]),
            tool("visual.key", "Press a key or a key combination such as 'Enter' or 'Control+A'.",
                 {"observation_id": _OBS, "combo": {"type": "string", "minLength": 1}}, ["observation_id", "combo"]),
            tool("visual.ground", "Ask a visual-grounding model where a described target is on the current screen. "
                 "Returns proposed coordinates and a confidence; nothing is clicked.",
                 {"target": {"type": "string", "minLength": 1}, "grounder": {"type": "string"}}, ["target"],
                 side="external_read", seconds=1.0),
            tool("visual.click_target", "Locate a described target with a visual-grounding model, check the proposed "
                 "point, click it, and look at the screen again to see what changed.",
                 {"target": {"type": "string", "minLength": 1}, "grounder": {"type": "string"},
                  "action": {"type": "string", "enum": ["click", "double_click"]},
                  "expect_change": {"type": "boolean"}}, ["target"], seconds=1.5),
        ]

    def validate(self, action: c.ActionSpec) -> List[str]:
        schema = next((t.arg_schema for t in self.tools() if t.name == action.operation), None)
        if schema is None:
            return [f"unknown operation {action.operation!r} for modality {self.modality}"]
        a = action.arguments
        if not isinstance(a, dict):
            return ["arguments must be an object"]
        errs = [f"missing argument {k!r}" for k in schema["required"] if k not in a]
        for k, v in a.items():
            spec = schema["properties"].get(k)
            if spec is None:
                errs.append(f"unknown argument {k!r}")
                continue
            t = spec["type"]
            if t == "number":
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    errs.append(f"{k}: expected a finite number")
            elif t == "boolean":
                if not isinstance(v, bool):
                    errs.append(f"{k}: expected boolean")
            elif not isinstance(v, str):
                errs.append(f"{k}: expected string")
            elif len(v) < spec.get("minLength", 0):
                errs.append(f"{k}: must not be empty")
            elif "enum" in spec and v not in spec["enum"]:
                errs.append(f"{k}: must be one of {spec['enum']}")
        if not errs:
            if action.operation == "visual.type" and len(a["text"]) > self.cfg.max_text_len:
                errs.append(f"text is longer than {self.cfg.max_text_len} characters")
            if action.operation == "visual.key" and not _KEY_RE.match(a["combo"]):
                errs.append("combo: not a key combination")
            if action.operation == "visual.crop" and (a["width"] < 1 or a["height"] < 1):
                errs.append("crop: width and height must be at least 1")
        return errs

    def close(self) -> None:
        with self._lock:
            grounders, self._grounders = self._grounders, {}
            self._shots.clear()
            self._current.clear()
        for g in grounders.values():
            closer = getattr(g, "close", None)
            if callable(closer):
                closer()

    # ------------------------------------------------------------------------------------------ images
    def image(self, obs_id: str) -> bytes:
        """PNG bytes of a stored screenshot observation (for perception code in the same process)."""
        return self._shots[obs_id].png

    def current_observation_id(self, screen: str = "default") -> str:
        return self._current.get(screen, "")

    def _screen_for(self, action: c.ActionSpec, ctx: c.ExecContext) -> Tuple[str, ScreenProvider]:
        name = str(action.arguments.get("screen", "default"))
        if self._factory is not None:
            return name, self._factory(name, action, ctx)
        if self._screen is None:
            raise _Reject(c.FailureClass.INVALID_ARGUMENTS, "no screen is attached to the visual executor")
        return name, self._screen

    def _store(self, shot: _Shot, ctx: c.ExecContext) -> None:
        if self.cfg.save_images and getattr(ctx, "run_dir", ""):
            d = os.path.join(ctx.run_dir, "blobs", "visual")
            os.makedirs(d, exist_ok=True)
            shot.path = os.path.join(d, f"{shot.obs_id}.png")
            with open(shot.path, "wb") as f:
                f.write(shot.png)
        self._shots[shot.obs_id] = shot
        keep = set(self._current.values()) | {shot.obs_id}
        while len(self._shots) > self.cfg.max_stored:
            victim = next((k for k in self._shots if k not in keep), None)
            if victim is None:
                break
            del self._shots[victim]

    def _observe(self, name: str, png: bytes, arr: Any, ctx: c.ExecContext, action: c.ActionSpec,
                 mode: str = "viewport", origin: Optional[Dict[str, Any]] = None,
                 extra: Optional[Dict[str, Any]] = None) -> c.Observation:
        now = ctx.clock.now()
        shot = _Shot(obs_id=c.new_id("obs"), screen=name, mode=mode, png=png, arr=arr, ts=now,
                     sha256=hashlib.sha256(png).hexdigest(), origin=origin or {})
        self._store(shot, ctx)
        if mode == "viewport":
            self._current[name] = shot.obs_id
        data = {"mode": mode, "screen": name, "width": shot.width, "height": shot.height, "sha256": shot.sha256,
                "actionable": mode == "viewport", "current": mode == "viewport"}
        data.update(origin or {})
        data.update(extra or {})
        obs = c.Observation(
            obs_id=shot.obs_id, ts=now, kind="image",
            text=f"{mode} screenshot of screen {name!r}, {shot.width}x{shot.height} pixels",
            quantities=[c.Quantity("visual.width", float(shot.width), "px"),
                        c.Quantity("visual.height", float(shot.height), "px")],
            refs={"image": shot.path or f"memory:{shot.obs_id}"},
            provenance=c.Provenance(source="visual", uri=f"screen:{name}", retrieved_at=now, trust=c.Trust.UNTRUSTED,
                                    content_sha256=shot.sha256),
            parent_action_id=action.action_id, data=data)
        sink = getattr(ctx, "obs_sink", None)
        if callable(sink):
            sink(obs)
        return obs

    def _grab(self, screen: ScreenProvider) -> Tuple[bytes, Any]:
        png = screen.screenshot()
        return png, _decode(png)

    def _settled(self, screen: ScreenProvider) -> Tuple[bytes, Any]:
        """Screenshot after the screen has stopped changing (two equal consecutive frames), within a time limit."""
        time.sleep(self.cfg.settle_s)
        png, arr = self._grab(screen)
        end = time.monotonic() + self.cfg.settle_max_s
        while time.monotonic() < end:
            time.sleep(self.cfg.settle_s)
            png2, arr2 = self._grab(screen)
            same = arr2.shape == arr.shape and frame_change(arr, arr2, self.cfg.pixel_threshold)["changed_fraction"] == 0.0
            png, arr = png2, arr2
            if same:
                break
        return png, arr

    # ------------------------------------------------------------------------------------------ validation
    def _current_shot(self, name: str, obs_id: str) -> _Shot:
        cur = self._current.get(name, "")
        if obs_id != cur:
            known = self._shots.get(obs_id)
            if known is not None and known.mode != "viewport":
                raise _Reject(c.FailureClass.INVALID_ARGUMENTS,
                              f"observation {obs_id} is a {known.mode} image; actions need a viewport screenshot")
            raise _Reject(c.FailureClass.STALE_OBSERVATION,
                          f"observation {obs_id!r} is not the current screenshot of screen {name!r}"
                          + (f" (current: {cur})" if cur else " (none taken yet)"))
        return self._shots[cur]

    @staticmethod
    def _in_bounds(shot: _Shot, x: float, y: float, what: str = "point") -> None:
        if not (0 <= x < shot.width and 0 <= y < shot.height):
            raise _Reject(c.FailureClass.INVALID_ARGUMENTS,
                          f"{what} ({x:g}, {y:g}) is outside the {shot.width}x{shot.height} screenshot")

    def _fresh(self, name: str, screen: ScreenProvider, shot: _Shot, ctx: c.ExecContext, action: c.ActionSpec,
               points: List[Tuple[float, float]]) -> Any:
        """The screen as it is now, if it still matches ``shot`` where the action looks; otherwise reject as stale
        and make the fresh frame the current observation."""
        png, arr = self._grab(screen)
        size = tuple(int(v) for v in screen.size())
        reason = ""
        if arr.shape != shot.arr.shape or size != (shot.width, shot.height):
            reason = f"the screen is now {size[0]}x{size[1]}, the observation is {shot.width}x{shot.height}"
        elif points:
            worst = max(_patch_diff(shot.arr, arr, x, y, self.cfg.patch_px) for x, y in points)
            if worst > self.cfg.patch_tolerance:
                reason = f"the screen changed at the target since the observation (patch difference {worst:.1f})"
        else:
            frac = frame_change(shot.arr, arr, self.cfg.pixel_threshold)["changed_fraction"]
            if frac > self.cfg.frame_tolerance:
                reason = f"the screen changed since the observation ({frac:.1%} of the pixels)"
        if reason:
            obs = self._observe(name, png, arr, ctx, action, extra={"replaces": shot.obs_id})
            raise _Reject(c.FailureClass.STALE_OBSERVATION, reason, obs=obs)
        return arr

    def _check(self, ctx: c.ExecContext, op: c.PrimitiveOp, action: c.ActionSpec) -> None:
        chk = getattr(ctx, "check", None)
        if not callable(chk):
            verdict, reason = c.AuthVerdict.DENY, "no authorization callback in the execution context"
        else:
            try:
                d = chk(op, action)
                verdict = getattr(d, "verdict", c.AuthVerdict.DENY)
                reason = "; ".join(getattr(d, "reasons", None) or [])
            except Exception as e:  # noqa: BLE001 - an unreachable policy service means no permission
                verdict, reason = c.AuthVerdict.DENY, f"authorization check failed: {type(e).__name__}: {e}"
        if verdict == c.AuthVerdict.ALLOW:
            return
        status = c.ResultStatus.NEEDS_APPROVAL if verdict == c.AuthVerdict.NEEDS_APPROVAL else c.ResultStatus.DENIED
        op.status, op.failure_class, op.note = status, c.FailureClass.MISSING_PERMISSION, reason or "denied by policy"
        raise _Reject(c.FailureClass.MISSING_PERMISSION, op.note, status=status)

    def _verify(self, action: c.ActionSpec, screen: ScreenProvider, before: Any, after: Any) -> Tuple[Optional[bool], str]:
        """Evaluate ``ActionSpec.verify`` on the frames around the action. (None, why) when it cannot be judged
        from pixels."""
        v = action.verify
        if v.kind == "none":
            return None, ""
        spec = v.spec or {}
        if v.kind == "element_state":
            if "color_at" in spec:
                x, y = (int(t) for t in spec["color_at"])
                if not (0 <= x < after.shape[1] and 0 <= y < after.shape[0]):
                    return False, "colour probe outside the screen"
                want = [int(t) for t in spec.get("rgb", (0, 0, 0))]
                got = [int(t) for t in after[y, x]]
                return sum(abs(g - w) for g, w in zip(got, want)) <= int(spec.get("tolerance", 60)), f"pixel {got}"
            if "changed" in spec:
                b, a = before, after
                if "region" in spec and b.shape == a.shape:
                    x, y, w, h = (int(t) for t in spec["region"])
                    b, a = b[y:y + h, x:x + w], a[y:y + h, x:x + w]
                frac = frame_change(b, a, self.cfg.pixel_threshold)["changed_fraction"]
                did = frac >= float(spec.get("min_fraction", self.cfg.min_effect_fraction))
                return did == bool(spec["changed"]), f"changed fraction {frac:.5f}"
            return None, "element_state needs 'changed' or 'color_at' for the visual modality"
        if v.kind == "url_matches":
            getter = getattr(screen, "url", None)
            if callable(getter):
                url = str(getter())
                return re.search(str(spec.get("pattern", "")), url) is not None, f"url {url[:200]}"
            return None, "this screen has no address"
        return None, f"verification kind {v.kind!r} cannot be judged from pixels"

    # ------------------------------------------------------------------------------------------ results
    def _result(self, action: c.ActionSpec, ctx: c.ExecContext, t0: float, m0: float, status: str,
                failure: str = c.FailureClass.NONE, detail: str = "", obs: Optional[List[c.Observation]] = None,
                ops: Optional[List[c.PrimitiveOp]] = None, verified: Optional[bool] = None,
                compute_s: float = 0.0, artifacts: Optional[Dict[str, str]] = None) -> c.ActionResult:
        ops = ops or []
        return c.ActionResult(action_id=action.action_id, status=status, failure_class=failure, failure_detail=detail,
                              started=t0, ended=ctx.clock.now(), observations=obs or [], ops=ops,
                              cost=c.Cost(seconds=time.monotonic() - m0, compute_s=compute_s, primitive_ops=len(ops)),
                              verified=verified, artifacts=artifacts or {})

    # ------------------------------------------------------------------------------------------ execute
    def execute(self, action: c.ActionSpec, ctx: c.ExecContext) -> c.ActionResult:
        t0, m0 = ctx.clock.now(), time.monotonic()
        errs = self.validate(action)
        if errs:
            return self._result(action, ctx, t0, m0, c.ResultStatus.FAILED, c.FailureClass.INVALID_ARGUMENTS,
                                "; ".join(errs))
        cancel = getattr(ctx, "cancel", None)
        if cancel is not None and cancel.is_set():
            return self._result(action, ctx, t0, m0, c.ResultStatus.CANCELLED, detail="cancelled before start")
        deadline = getattr(ctx, "deadline", 0.0) or 0.0
        if deadline and ctx.clock.now() >= deadline:
            return self._result(action, ctx, t0, m0, c.ResultStatus.TIMEOUT, c.FailureClass.RESOURCE_LIMIT,
                                "the execution deadline had already passed")
        ops: List[c.PrimitiveOp] = []
        with self._lock:
            try:
                name, screen = self._screen_for(action, ctx)
                op = action.operation
                if op == "visual.screenshot":
                    return self._do_screenshot(action, ctx, t0, m0, name, screen, ops)
                if op == "visual.crop":
                    return self._do_crop(action, ctx, t0, m0, name, ops)
                if op == "visual.ground":
                    return self._do_ground(action, ctx, t0, m0, name, screen, ops)
                if op == "visual.click_target":
                    return self._do_click_target(action, ctx, t0, m0, name, screen, ops)
                return self._do_primitive(action, ctx, t0, m0, name, screen, ops)
            except _Reject as r:
                return self._result(action, ctx, t0, m0, r.status, r.failure, r.detail,
                                    obs=[r.obs] if r.obs is not None else [], ops=ops)

    def _op(self, ctx: c.ExecContext, name: str, **arguments: Any) -> c.PrimitiveOp:
        return c.PrimitiveOp(op=name, modality=self.modality, arguments=arguments, started=ctx.clock.now())

    # ---- observations ----------------------------------------------------------------------------------------
    def _do_screenshot(self, action, ctx, t0, m0, name, screen, ops) -> c.ActionResult:
        mode = action.arguments.get("mode", "viewport")
        op = self._op(ctx, "visual.screenshot", mode=mode, screen=name)
        ops.append(op)
        self._check(ctx, op, action)
        png, arr = self._grab(screen)
        if mode == "viewport":
            obs = self._observe(name, png, arr, ctx, action)
            op.obs_after, op.ended = obs.obs_id, ctx.clock.now()
            return self._result(action, ctx, t0, m0, c.ResultStatus.OK, obs=[obs], ops=ops,
                                artifacts={"image": obs.refs["image"]})
        page, info = self._full_page(screen, arr)
        out = [self._observe(name, _encode(page), page, ctx, action, mode="full_page", extra=info)]
        png2, arr2 = self._grab(screen)                      # where the viewport is after stitching
        cur = self._shots.get(self._current.get(name, ""))
        if cur is None or cur.arr.shape != arr2.shape or \
                frame_change(cur.arr, arr2, self.cfg.pixel_threshold)["changed_fraction"] > 0:
            out.append(self._observe(name, png2, arr2, ctx, action))
        op.obs_after, op.ended = out[0].obs_id, ctx.clock.now()
        return self._result(action, ctx, t0, m0, c.ResultStatus.OK, obs=out, ops=ops,
                            artifacts={"image": out[0].refs["image"]})

    def _full_page(self, screen: ScreenProvider, first: Any) -> Tuple[Any, Dict[str, Any]]:
        """The whole scrollable page as one image. Uses the provider's own full-page capture when it has one,
        otherwise scrolls half a screen at a time and stitches (the scroll position is restored). The last, partial
        scroll is recognised by an exact match of the overlap of consecutive frames (ignoring the top and bottom
        15 % of the screen, where fixed headers and footers live). Limits: content that repeats exactly with a
        period shorter than half a screen comes out too short, and a featureless overlap at the page end is taken
        as a full step (the image is then padded at the bottom)."""
        import numpy as np
        own = getattr(screen, "screenshot_full_page", None)
        if callable(own):
            return _decode(own()), {"stitched": False, "screens": 1}
        h, w = first.shape[:2]
        step = max(1, h // 2)                            # half screens: consecutive frames always overlap

        def signature(arr: Any) -> Any:                  # per row: mean colour of 16 column blocks
            cols = (w // 16) * 16
            return arr[:, :cols].reshape(h, 16, cols // 16, 3).mean(2).reshape(h, 48).astype(np.float32)

        frames, offsets, sigs = [first], [0], [signature(first)]
        for _ in range(2 * self.cfg.full_page_max_screens - 1):
            screen.scroll(0, step)
            time.sleep(self.cfg.settle_s)
            _, arr = self._grab(screen)
            prev = frames[-1]
            if arr.shape != prev.shape or frame_change(prev, arr, self.cfg.pixel_threshold)["changed_fraction"] == 0.0:
                break
            sa, sb = sigs[-1], signature(arr)
            shift = step                                 # unless a smaller shift makes the overlap agree (page end)
            m = int(0.15 * h)                            # rows that a fixed header or footer may occupy
            for cand in range(1, step):
                top, bottom = sb[m:h - cand - m], sa[m + cand:h - m]
                if float(np.abs(top[1:] - top[:-1]).max()) < 2.0:
                    break                                # a featureless overlap says nothing: keep the full step
                if float(np.abs(bottom - top).mean(1).max()) < 2.0:      # every compared row agrees
                    shift = cand
                    break
            frames.append(arr)
            sigs.append(sb)
            offsets.append(offsets[-1] + shift)
        page = np.zeros((offsets[-1] + h, first.shape[1], 3), dtype=first.dtype)
        for arr, off in zip(frames, offsets):
            page[off:off + h] = arr
        if offsets[-1]:
            screen.scroll(0, -offsets[-1])
            time.sleep(self.cfg.settle_s)
        return page, {"stitched": True, "screens": len(frames), "scrolled_px": int(offsets[-1])}

    def _do_crop(self, action, ctx, t0, m0, name, ops) -> c.ActionResult:
        a = action.arguments
        shot = self._shots.get(a["observation_id"])
        if shot is None:
            raise _Reject(c.FailureClass.STALE_OBSERVATION, f"screenshot {a['observation_id']!r} is no longer kept")
        x, y, w, h = int(a["x"]), int(a["y"]), int(a["width"]), int(a["height"])
        if x < 0 or y < 0 or x + w > shot.width or y + h > shot.height:
            raise _Reject(c.FailureClass.INVALID_ARGUMENTS,
                          f"crop {x},{y} {w}x{h} is outside the {shot.width}x{shot.height} image")
        arr = shot.arr[y:y + h, x:x + w].copy()
        obs = self._observe(shot.screen, _encode(arr), arr, ctx, action, mode="crop",
                            origin={"source_observation": shot.obs_id, "offset": [x, y]})
        ops.append(self._op(ctx, "visual.crop", observation_id=shot.obs_id, x=x, y=y, width=w, height=h))
        ops[-1].ended = ctx.clock.now()
        return self._result(action, ctx, t0, m0, c.ResultStatus.OK, obs=[obs], ops=ops,
                            artifacts={"image": obs.refs["image"]})

    # ---- primitives ------------------------------------------------------------------------------------------
    def _deliver(self, screen: ScreenProvider, op: str, a: Dict[str, Any]) -> None:
        if op == "visual.click":
            screen.click(a["x"], a["y"])
        elif op == "visual.double_click":
            screen.double_click(a["x"], a["y"])
        elif op == "visual.drag":
            screen.drag(a["x1"], a["y1"], a["x2"], a["y2"])
        elif op == "visual.scroll":
            dx, dy = a.get("dx", 0), a["dy"]
            if "x" in a and "y" in a and {"x", "y"} <= set(inspect.signature(screen.scroll).parameters):
                screen.scroll(dx, dy, x=a["x"], y=a["y"])
            else:
                screen.scroll(dx, dy)
        elif op == "visual.type":
            screen.type(a["text"])
        elif op == "visual.key":
            screen.key(a["combo"])
        else:
            raise _Reject(c.FailureClass.INVALID_ARGUMENTS, f"unknown operation {op!r}")

    def _act(self, action: c.ActionSpec, ctx: c.ExecContext, name: str, screen: ScreenProvider, opname: str,
             a: Dict[str, Any], shot: _Shot, ops: List[c.PrimitiveOp], extra: Optional[Dict[str, Any]] = None
             ) -> Tuple[c.Observation, Dict[str, Any], Optional[bool], str]:
        """Validate, authorize, deliver one primitive and re-observe. Returns (observation after, measured change,
        verified, verification note)."""
        points: List[Tuple[float, float]] = []
        if opname in _POINTER_OPS:
            self._in_bounds(shot, a["x"], a["y"])
            points = [(a["x"], a["y"])]
        elif opname == "visual.drag":
            self._in_bounds(shot, a["x1"], a["y1"], "start point")
            self._in_bounds(shot, a["x2"], a["y2"], "end point")
            points = [(a["x1"], a["y1"]), (a["x2"], a["y2"])]
        elif opname == "visual.scroll" and "x" in a and "y" in a:
            self._in_bounds(shot, a["x"], a["y"])
        before = self._fresh(name, screen, shot, ctx, action, points)
        logged = {k: v for k, v in a.items() if k != "text"}
        if "text" in a:
            logged.update(chars=len(a["text"]), sha256=hashlib.sha256(a["text"].encode()).hexdigest())
        op = self._op(ctx, opname, screen=name, **logged, **(extra or {}))
        op.obs_before = shot.obs_id
        ops.append(op)
        self._check(ctx, op, action)
        self._deliver(screen, opname, a)
        png, after = self._settled(screen)
        change = frame_change(before, after, self.cfg.pixel_threshold)
        change["effect_observed"] = change["changed_fraction"] >= self.cfg.min_effect_fraction
        verified, note = self._verify(action, screen, before, after)
        obs = self._observe(name, png, after, ctx, action, extra={
            "after": opname, "previous_observation": shot.obs_id, "change": change, "verified": verified,
            "verification": note})
        obs.quantities.append(c.Quantity("visual.changed_fraction", float(change["changed_fraction"])))
        op.obs_after, op.ended = obs.obs_id, ctx.clock.now()
        return obs, change, verified, note

    def _do_primitive(self, action, ctx, t0, m0, name, screen, ops) -> c.ActionResult:
        a = action.arguments
        shot = self._current_shot(name, a["observation_id"])
        obs, change, verified, note = self._act(action, ctx, name, screen, action.operation, a, shot, ops)
        return self._result(action, ctx, t0, m0, c.ResultStatus.OK, obs=[obs], ops=ops, verified=verified,
                            detail="" if verified is not False else f"expected effect not observed ({note})")

    # ---- grounding -------------------------------------------------------------------------------------------
    def _grounder(self, name: str) -> c.VisualGrounder:
        g = self._grounders.get(name)
        if g is None:
            factory = self._grounder_factory
            if factory is None:
                from .grounding import load_grounder as factory        # one process per model, clean environment
            try:
                g = factory(name)
            except Exception as e:  # noqa: BLE001 - an unknown or unavailable grounder is a failed proposal
                raise _Reject(c.FailureClass.INVALID_ARGUMENTS, f"grounder {name!r} is not available: {e}")
            self._grounders[name] = g
        return g

    def _propose(self, ctx: c.ExecContext, gname: str, shot: _Shot, target: str, ops: List[c.PrimitiveOp]
                 ) -> Tuple[Optional[Dict[str, Any]], str, float]:
        """One grounder's proposal for ``target`` on ``shot``, validated. (proposal | None, why not, seconds)."""
        op = self._op(ctx, "visual.ground", grounder=gname, target=target[:200], observation_id=shot.obs_id)
        op.obs_before = shot.obs_id
        ops.append(op)

        def bad(why: str, cls: str = c.FailureClass.VISUAL_GROUNDING) -> Tuple[None, str, float]:
            op.status, op.failure_class, op.note, op.ended = c.ResultStatus.FAILED, cls, why, ctx.clock.now()
            return None, why, time.monotonic() - m

        m = time.monotonic()
        g = self._grounder(gname)
        heavy = bool(getattr(g, "heavy", False))
        try:
            if heavy:
                with _shared_gpu_lock(self.cfg.gpu_lock_timeout_s) as got:
                    if not got:
                        return bad("the shared GPU lock was not free in time", c.FailureClass.RESOURCE_LIMIT)
                    try:
                        raw = g.ground(shot.png, target)
                    finally:
                        if self.cfg.release_heavy_after_use and callable(getattr(g, "close", None)):
                            g.close()                         # give the memory back before the lock is released
            else:
                raw = g.ground(shot.png, target)
        except Exception as e:  # noqa: BLE001 - whatever the model or its process did, it is a failed proposal
            return bad(f"{type(e).__name__}: {e}"[:300])
        dt = time.monotonic() - m
        # the proposal is data: only a point and a confidence are read, and both are checked here
        try:
            x, y = raw["x"], raw["y"]
            conf = float(raw.get("confidence", 0.0))
        except (TypeError, KeyError, ValueError):
            return bad("the grounder returned no point")
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)) \
                or not (math.isfinite(x) and math.isfinite(y) and math.isfinite(conf)):
            return bad("the grounder returned a point that is not a number")
        if not (0 <= x < shot.width and 0 <= y < shot.height):
            return bad(f"the proposed point ({x:g}, {y:g}) is outside the {shot.width}x{shot.height} screenshot")
        if conf < self.cfg.min_confidence:
            return bad(f"confidence {conf:.3f} is below the required {self.cfg.min_confidence:.3f}")
        prop = {"x": int(round(x)), "y": int(round(y)), "confidence": conf, "grounder": gname,
                "revision": str(getattr(g, "revision", "")), "observation_id": shot.obs_id, "seconds": dt}
        op.arguments.update(x=prop["x"], y=prop["y"], confidence=conf)
        op.ended = ctx.clock.now()
        return prop, "", dt

    def _order(self, action: c.ActionSpec) -> List[str]:
        asked = action.arguments.get("grounder")
        if asked:
            return [asked]
        return [g for g in self.cfg.grounder_order] or sorted(self._grounders)

    def _do_ground(self, action, ctx, t0, m0, name, screen, ops) -> c.ActionResult:
        png, arr = self._grab(screen)
        view = self._observe(name, png, arr, ctx, action)
        prop, why, dt = self._propose(ctx, self._order(action)[0], self._shots[view.obs_id], action.arguments["target"], ops)
        if prop is None:
            cls = ops[-1].failure_class
            return self._result(action, ctx, t0, m0, c.ResultStatus.FAILED, cls, why, obs=[view], ops=ops, compute_s=dt)
        now = ctx.clock.now()
        text = (f"proposed point ({prop['x']}, {prop['y']}) for the described target on observation {view.obs_id} "
                f"(grounder {prop['grounder']}, confidence {prop['confidence']:.3f}); nothing was clicked")
        obs = c.Observation(obs_id=c.new_id("obs"), ts=now, kind="tool_result", text=text,
                            quantities=[c.Quantity("visual.ground.confidence", prop["confidence"]),
                                        c.Quantity("visual.ground.seconds", dt, "s")],
                            provenance=c.Provenance(source=f"grounder:{prop['grounder']}", uri=f"screen:{name}",
                                                    retrieved_at=now, trust=c.Trust.UNTRUSTED),
                            parent_action_id=action.action_id, data=dict(prop, proposal_only=True))
        sink = getattr(ctx, "obs_sink", None)
        if callable(sink):
            sink(obs)
        return self._result(action, ctx, t0, m0, c.ResultStatus.OK, obs=[view, obs], ops=ops, compute_s=dt)

    def _do_click_target(self, action, ctx, t0, m0, name, screen, ops) -> c.ActionResult:
        a = action.arguments
        opname = "visual." + a.get("action", "click")
        expect = bool(a.get("expect_change", True))
        order = self._order(action)[:max(1, self.cfg.max_ground_attempts)]
        observations: List[c.Observation] = []
        compute = 0.0
        attempts: List[Dict[str, Any]] = []
        last_cls, last_why = c.FailureClass.VISUAL_GROUNDING, "no grounder is configured"
        for gname in order:
            png, arr = self._grab(screen)                    # ground on the screen as it is now
            view = self._observe(name, png, arr, ctx, action)
            observations.append(view)
            shot = self._shots[view.obs_id]
            prop, why, dt = self._propose(ctx, gname, shot, a["target"], ops)
            compute += dt
            if prop is None:
                attempts.append({"grounder": gname, "proposal": None, "reason": why})
                last_cls, last_why = ops[-1].failure_class, why
                continue
            point = {"x": prop["x"], "y": prop["y"]}
            obs, change, verified, note = self._act(action, ctx, name, screen, opname, point, shot, ops, extra={
                "target": a["target"][:200], "grounder": gname, "confidence": prop["confidence"]})
            obs.data["grounding"] = prop
            observations.append(obs)
            attempts.append({"grounder": gname, "proposal": prop, "change": change, "verified": verified})
            obs.data["attempts"] = attempts
            if verified is False:
                # something was clicked and the expected effect is not there: report it, do not click again
                return self._result(action, ctx, t0, m0, c.ResultStatus.FAILED, c.FailureClass.VISUAL_GROUNDING,
                                    f"clicked ({prop['x']}, {prop['y']}) but the expected effect was not observed ({note})",
                                    obs=observations, ops=ops, verified=False, compute_s=compute)
            if expect and not change["effect_observed"]:
                ops[-1].note = "no observable effect"
                last_cls, last_why = c.FailureClass.VISUAL_GROUNDING, \
                    f"the {a.get('action', 'click')} at ({prop['x']}, {prop['y']}) proposed by {gname} had no observable effect"
                continue                                     # nothing changed: another grounder may try
            return self._result(action, ctx, t0, m0, c.ResultStatus.OK, obs=observations, ops=ops, verified=verified,
                                compute_s=compute)
        return self._result(action, ctx, t0, m0, c.ResultStatus.FAILED, last_cls, last_why, obs=observations, ops=ops,
                            verified=False if action.verify.kind != "none" else None, compute_s=compute)
