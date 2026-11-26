"""Shared machinery of the visual grounders (workstream EC).

A grounder turns (screenshot, description of a target) into a proposed point. It never acts: the visual executor
validates the proposal against the current observation and the authorization layer before anything is clicked.

Rules implemented here for every neural grounder:

* weights come from a local Hugging Face snapshot pinned by commit hash; nothing is downloaded at load time and
  ``trust_remote_code`` is never used (only model classes that ship inside ``transformers``, safetensors weights);
* the model is frozen (``eval()``, ``requires_grad_(False)``) and loaded lazily on first use;
* it is not loaded in a process that holds credentials (:func:`credential_holder`); the runtime path is the
  subprocess in :mod:`cam_live.harness.grounding.worker`, which starts with an empty environment.
"""
from __future__ import annotations

import io
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

from .. import SERVICE_USERS, uid_of, users_exist


class GroundingError(RuntimeError):
    """The grounder produced no usable point for this request."""


class GrounderUnavailable(RuntimeError):
    """The grounder cannot be used on this host (missing snapshot, no GPU, licence not accepted, ...)."""


class GrounderLoadRefused(RuntimeError):
    """Loading model weights was refused in this process."""


def hf_home() -> str:
    return os.environ.get("HF_HOME") or "/workspace/.hf_home"


def snapshot_dir(repo: str, revision: str) -> str:
    """Local directory of exactly ``repo@revision``. No network access; raises when it is not present."""
    path = os.path.join(hf_home(), "hub", "models--" + repo.replace("/", "--"), "snapshots", revision)
    if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "config.json")):
        raise GrounderUnavailable(f"{repo}@{revision} is not in the local cache ({path}); download it first with "
                                  f"HF_HOME={hf_home()} HF_HUB_OFFLINE=0 huggingface-cli download {repo} --revision {revision}")
    return path


def credential_holder() -> str:
    """Why this process must not load third-party model weights ('' when it may)."""
    if os.environ.get("CAM_LIVE_HOLDS_CREDENTIALS"):
        return "the process is marked CAM_LIVE_HOLDS_CREDENTIALS"
    if users_exist() and os.geteuid() != 0 and os.geteuid() == uid_of("auth"):
        return f"the process runs as the credential service user {SERVICE_USERS['auth']}"
    return ""


def decode_png(image_png: bytes):
    """PNG bytes -> RGB ``PIL.Image`` (fully decoded, bounded size)."""
    from PIL import Image
    if not isinstance(image_png, (bytes, bytearray)) or len(image_png) < 16:
        raise GroundingError("screenshot is not PNG data")
    try:
        img = Image.open(io.BytesIO(bytes(image_png)))
        img.load()
    except Exception as e:  # noqa: BLE001 - any decoder failure is the same outcome
        raise GroundingError(f"cannot decode screenshot: {type(e).__name__}") from e
    if img.width * img.height > 64_000_000:
        raise GroundingError("screenshot too large")
    return img.convert("RGB")


def point_result(x: float, y: float, width: int, height: int, confidence: float, name: str, revision: str,
                 raw: str = "", **extra: Any) -> Dict[str, Any]:
    """The contract's result shape. Coordinates are rounded; values off the image are a grounding error."""
    if not (math.isfinite(x) and math.isfinite(y)):
        raise GroundingError("the model returned a non-finite coordinate")
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < width and 0 <= yi < height):
        raise GroundingError(f"the model pointed outside the image: ({xi}, {yi}) for {width}x{height}")
    conf = float(confidence) if math.isfinite(float(confidence)) else 0.0
    return {"x": xi, "y": yi, "confidence": max(0.0, min(1.0, conf)), "grounder": name, "revision": revision,
            "raw": raw[:400], **extra}


class NeuralGrounder:
    """Base of the model-backed grounders: pinned snapshot, guarded lazy load, frozen weights, measurements."""

    name = ""
    repo = ""
    revision = ""
    license = ""
    architecture = ""
    heavy = False                      # True: needs more than 4 GB of GPU memory -> take the shared GPU lock
    noncommercial = False              # True: the licence forbids commercial use
    baseline = False

    def __init__(self, device: str = "auto", allow_inprocess: bool = False, accept_noncommercial: bool = False):
        self._device_req = device
        self._allow_inprocess = allow_inprocess or bool(os.environ.get("CAM_LIVE_GROUNDER_WORKER"))
        self._accept_nc = accept_noncommercial
        self._model: Any = None
        self.device = ""
        self.load_s = 0.0
        self.gpu_mem_mb = 0.0          # allocated after load
        self.gpu_peak_mb = 0.0         # peak allocated during inference
        self.params = 0
        self.calls = 0
        self.total_latency_s = 0.0

    # ------------------------------------------------------------------------------------------ loading
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def available(self) -> Tuple[bool, str]:
        try:
            snapshot_dir(self.repo, self.revision)
        except GrounderUnavailable as e:
            return False, str(e)
        if self.noncommercial and not self._accept_nc:
            return False, f"licence {self.license} forbids commercial use; pass accept_noncommercial=True for evaluation"
        return True, ""

    def _pick_device(self) -> str:
        import torch
        if self._device_req != "auto":
            return self._device_req
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self, snapshot: str, device: str) -> Any:
        raise GrounderUnavailable("abstract grounder")

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        why = credential_holder()
        if why:
            raise GrounderLoadRefused(f"{self.name}: not loading model weights here: {why}")
        if not self._allow_inprocess:
            raise GrounderLoadRefused(
                f"{self.name}: in-process loading must be requested explicitly (allow_inprocess=True) by a process "
                f"that holds no credentials; the runtime uses grounding.load_grounder(..., isolated=True)")
        ok, why = self.available()
        if not ok:
            raise GrounderUnavailable(why)
        import torch
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        device = self._pick_device()
        t0 = time.monotonic()
        base = torch.cuda.memory_allocated() if device.startswith("cuda") else 0
        model = self._load(snapshot_dir(self.repo, self.revision), device)
        model.eval()
        model.requires_grad_(False)                     # frozen: the grounder is never trained here
        self.params = sum(p.numel() for p in model.parameters())
        self._model, self.device = model, device
        self.load_s = time.monotonic() - t0
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            self.gpu_mem_mb = (torch.cuda.memory_allocated() - base) / 2**20

    # ------------------------------------------------------------------------------------------ inference
    def _ground(self, image: Any, instruction: str) -> Dict[str, Any]:
        raise GroundingError("abstract grounder")

    def ground(self, image_png: bytes, instruction: str) -> Dict[str, Any]:
        if not isinstance(instruction, str) or not instruction.strip():
            raise GroundingError("empty target description")
        image = decode_png(image_png)
        self.ensure_loaded()
        import torch
        cuda = self.device.startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        with torch.inference_mode():
            out = self._ground(image, instruction.strip()[:1000])
        if cuda:
            torch.cuda.synchronize()
            self.gpu_peak_mb = max(self.gpu_peak_mb, torch.cuda.max_memory_allocated() / 2**20)
        dt = time.monotonic() - t0
        self.calls += 1
        self.total_latency_s += dt
        out["latency_s"] = dt
        return out

    def info(self) -> Dict[str, Any]:
        ok, why = self.available()
        return {"name": self.name, "repo": self.repo, "revision": self.revision, "license": self.license,
                "architecture": self.architecture, "noncommercial": self.noncommercial, "baseline": self.baseline,
                "needs_gpu_lock": self.heavy, "trust_remote_code": False, "frozen": True,
                "available": ok, "unavailable_reason": why, "loaded": self.loaded, "device": self.device,
                "params": self.params, "load_s": self.load_s, "gpu_mem_mb": self.gpu_mem_mb,
                "gpu_peak_mb": self.gpu_peak_mb, "calls": self.calls,
                "mean_latency_s": self.total_latency_s / self.calls if self.calls else None}

    def close(self) -> None:
        """Drop the model and give its GPU memory back."""
        if self._model is None:
            return
        self._model = None
        self._drop()
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _drop(self) -> None:
        pass
