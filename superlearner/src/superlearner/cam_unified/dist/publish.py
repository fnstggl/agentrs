"""Atomic, versioned, weights-only checkpoint publication and actor-side refresh (spec 14.3, manifest F02).

Layout of a checkpoint root::

    ckpt/
      v00000000/weights.pt   torch.save({"format", "version", "config", "state_dict"}) — tensors + plain config only
      v00000000/meta.json    {"version", "sha256", "bytes", "n_tensors", "n_params", "rel_change", "validated",
                              "validation", "t", "meta": {...}}
      LATEST                 {"version", "dir", "sha256", "t"} — only ever points at a validated version
      .tmp-v.../             a version being written (never referenced)

Publication protocol (`CheckpointPublisher.publish`): write ``weights.pt`` and ``meta.json`` into a hidden temporary
directory, ``fsync`` both, ``rename`` the directory to ``v<version>``, ``fsync`` the root; then **validate** the
published file — reload it with ``torch.load(weights_only=True)``, check the sha256, the key set, shapes and that every
floating tensor is finite, and run the optional ``validate_fn(state_dict, meta) -> (ok, info)`` (e.g. a held-out
probe) — and only then replace ``LATEST`` atomically (write ``LATEST.tmp`` → fsync → ``os.replace`` → fsync root).
Versions are strictly increasing; a failed validation leaves the version on disk marked ``validated: false`` and
``LATEST`` unchanged. Old versions beyond ``keep_last`` are pruned (never the one ``LATEST`` names).

**Restart.** A new publisher on an existing root treats every version directory *above* LATEST as the leftover of a
publication that never completed (a crash between the directory rename and the LATEST update, or a rejected
validation): it is renamed to ``.unpublished-v<N>-<t>`` (kept for audit, never referenced, never counted as a
version), and ``last_version`` becomes LATEST's version — so a resumed learner, which continues from LATEST,
publishes its next version without a monotonicity conflict. Readers never see those directories (they only follow
LATEST).

`PolicyClient` (actor side) polls ``LATEST``, loads a newer version with ``weights_only=True`` after verifying its
sha256, and hands it to the policy **only at safe boundaries** (trajectory boundaries by default; decision boundaries
when ``refresh_at="decision"``). It reports the version the actor currently runs (stamped on every EventRecord as
``model_version``) and ``drift`` = ‖θ_now − θ_state‖ / ‖θ_state‖, the exact relative parameter change between the
weights now loaded and the weights under which the actor's latent state was built (computed on the actor from both
state dicts, so skipped versions are accounted for). The actor uses it to decide whether its latent state must be
reconstructed from raw episode history under the new weights instead of being reused (spec 14.3: no blind reuse of
stale latent state). ``drift_mode="meta_sum"`` avoids the reference copy for very large models: it sums the published
per-version absolute changes ``abs_change`` = ‖θ_v − θ_prev‖ along the chain of validated versions adopted since the
latent state was built and divides by that state's weight norm — an upper bound on the exact drift by the triangle
inequality, and infinite (forcing a rebuild) if an intermediate version's metadata was pruned.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import time
from typing import Callable, Optional

import torch

from ..observe.accounting import atomic_write_json

WEIGHTS_FORMAT = "cam-unified-weights/1"


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _vdir(version: int) -> str:
    return f"v{version:08d}"


def state_norm(sd: dict) -> float:
    return math.sqrt(sum(float(v.double().pow(2).sum()) for v in sd.values() if v.is_floating_point()))


def param_change(new: dict, old: Optional[dict]) -> tuple:
    """(‖θ_new − θ_old‖₂, ‖θ_old‖₂) over shared floating tensors of identical shape ((None, None) without old)."""
    if old is None:
        return None, None
    num = den = 0.0
    for k, v in new.items():
        o = old.get(k)
        if o is None or not v.is_floating_point() or o.shape != v.shape:
            continue
        num += float((v.double() - o.double()).pow(2).sum())
        den += float(o.double().pow(2).sum())
    return math.sqrt(num), math.sqrt(den)


def relative_change(new: dict, old: Optional[dict]) -> Optional[float]:
    """‖θ_new − θ_old‖₂ / max(‖θ_old‖₂, 1e-12) over shared floating tensors of identical shape (None without old)."""
    a, n = param_change(new, old)
    return None if a is None else a / max(n, 1e-12)


class PublishRejected(RuntimeError):
    pass


class CheckpointPublisher:
    """Learner-side publisher. Single writer per root."""

    def __init__(self, root: str, keep_last: int = 5, fsync: bool = True,
                 validate_fn: Optional[Callable[[dict, dict], tuple]] = None):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.keep_last = keep_last
        self.fsync = fsync
        self.validate_fn = validate_fn
        self._prev_state: Optional[dict] = None
        for fn in os.listdir(self.root):          # leftovers of a crash mid-publication are never referenced
            if fn.startswith(".tmp-"):
                shutil.rmtree(os.path.join(self.root, fn), ignore_errors=True)
        lt = self.latest()
        lv = int(lt["version"]) if lt is not None else -1
        self.unpublished: list = []
        for v in self.versions():                 # never became LATEST: crash before the pointer update, or rejected
            if v > lv:
                dst = os.path.join(self.root, f".unpublished-{_vdir(v)}-{int(time.time() * 1e6)}")
                os.rename(os.path.join(self.root, _vdir(v)), dst)
                self.unpublished.append(v)
        if self.unpublished and self.fsync:
            _fsync_dir(self.root)
        self.last_version = lv
        if lt is not None:
            try:
                self._prev_state = load_weights(os.path.join(self.root, lt["dir"], "weights.pt"))["state_dict"]
            except (OSError, RuntimeError):
                self._prev_state = None

    # ------------------------------------------------------------ views
    def versions(self) -> list:
        out = []
        for fn in os.listdir(self.root):
            if fn.startswith("v") and fn[1:].isdigit() and os.path.isdir(os.path.join(self.root, fn)):
                out.append(int(fn[1:]))
        return sorted(out)

    def latest(self) -> Optional[dict]:
        return read_latest(self.root)

    def meta(self, version: int) -> dict:
        with open(os.path.join(self.root, _vdir(version), "meta.json")) as fh:
            return json.load(fh)

    # ------------------------------------------------------------ publishing
    def publish(self, state_dict: dict, version: int, meta: Optional[dict] = None,
                config: Optional[dict] = None) -> dict:
        """Publish ``state_dict`` (name → tensor) as ``version``. Returns the version's meta (``validated`` tells
        whether LATEST now points at it). Raises PublishRejected for a non-monotone version."""
        version = int(version)
        if version <= self.last_version:
            raise PublishRejected(f"version {version} <= last published {self.last_version} (versions are monotone)")
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"state_dict[{k!r}] is not a tensor (weights-only checkpoints hold tensors only)")
        sd = {k: v.detach().to("cpu").contiguous().clone() for k, v in state_dict.items()}
        payload = {"format": WEIGHTS_FORMAT, "version": version, "config": json.loads(json.dumps(config or {})),
                   "state_dict": sd}
        bio = io.BytesIO()
        torch.save(payload, bio)
        data = bio.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        absc, prev_norm = param_change(sd, self._prev_state)
        rel = None if absc is None else absc / max(prev_norm, 1e-12)
        m = {"version": version, "sha256": digest, "bytes": len(data), "n_tensors": len(sd),
             "n_params": int(sum(v.numel() for v in sd.values())), "rel_change": rel, "abs_change": absc,
             "norm": state_norm(sd),
             "prev_version": self.latest()["version"] if self.latest() else None, "t": time.time(),
             "validated": False, "validation": None, "meta": meta or {}}
        tmp = os.path.join(self.root, f".tmp-{_vdir(version)}-{os.getpid()}")
        os.makedirs(tmp, exist_ok=False)
        with open(os.path.join(tmp, "weights.pt"), "wb") as fh:
            fh.write(data)
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())
        atomic_write_json(os.path.join(tmp, "meta.json"), m, fsync=self.fsync)
        if self.fsync:
            _fsync_dir(tmp)
        final = os.path.join(self.root, _vdir(version))
        os.rename(tmp, final)
        if self.fsync:
            _fsync_dir(self.root)
        self.last_version = version

        ok, info = self._validate(final, digest, sd, m)
        m["validated"], m["validation"] = bool(ok), info
        atomic_write_json(os.path.join(final, "meta.json"), m, fsync=self.fsync)
        if ok:
            ptr = {"version": version, "dir": _vdir(version), "sha256": digest, "t": time.time(), "rel_change": rel}
            atomic_write_json(os.path.join(self.root, "LATEST"), ptr, fsync=self.fsync)
            if self.fsync:
                _fsync_dir(self.root)
            self._prev_state = sd
            self._prune(version)
        return m

    def _validate(self, final: str, digest: str, sd: dict, m: dict) -> tuple:
        path = os.path.join(final, "weights.pt")
        with open(path, "rb") as fh:
            data = fh.read()
        if hashlib.sha256(data).hexdigest() != digest:
            return False, {"reason": "sha256 mismatch after write"}
        try:
            back = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        except Exception as e:                       # noqa: BLE001 — any load failure invalidates the version
            return False, {"reason": f"weights_only load failed: {type(e).__name__}: {e}"}
        bsd = back.get("state_dict", {})
        if set(bsd) != set(sd):
            return False, {"reason": "key set changed on reload"}
        for k, v in bsd.items():
            if v.shape != sd[k].shape or v.dtype != sd[k].dtype:
                return False, {"reason": f"tensor {k} changed shape/dtype on reload"}
            if v.is_floating_point() and not bool(torch.isfinite(v).all()):
                return False, {"reason": f"non-finite values in {k}"}
        info = {"reason": "ok", "checks": ["sha256", "weights_only_load", "keys", "shapes", "finite"]}
        if self.validate_fn is not None:
            ok, extra = self.validate_fn(bsd, m)
            info["validate_fn"] = extra
            if not ok:
                info["reason"] = "validate_fn rejected"
                return False, info
            info["checks"].append("validate_fn")
        return True, info

    def _prune(self, keep_version: int) -> None:
        if self.keep_last is None or self.keep_last <= 0:
            return
        vs = [v for v in self.versions() if v != keep_version]
        excess = len(vs) + 1 - self.keep_last
        for v in vs[:max(0, excess)]:
            shutil.rmtree(os.path.join(self.root, _vdir(v)), ignore_errors=True)


def read_latest(root: str) -> Optional[dict]:
    """The LATEST pointer (always a complete JSON document thanks to os.replace), or None before the first
    validated publication."""
    try:
        with open(os.path.join(root, "LATEST")) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def load_weights(path: str, expect_sha256: Optional[str] = None) -> dict:
    """Load a published weights file with ``weights_only=True`` (no pickled code can run), verifying the digest."""
    with open(path, "rb") as fh:
        data = fh.read()
    if expect_sha256 is not None and hashlib.sha256(data).hexdigest() != expect_sha256:
        raise RuntimeError(f"sha256 mismatch for {path}")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if payload.get("format") != WEIGHTS_FORMAT:
        raise RuntimeError(f"{path}: not a {WEIGHTS_FORMAT} file")
    return payload


class PolicyClient:
    """Actor-side view of a checkpoint root.

    refresh_at: "trajectory" (default) — new weights are adopted only at trajectory (episode) boundaries;
    "decision" — also between decisions (the actor then decides whether to rebuild latent state, see `drift`).
    """

    def __init__(self, root: str, refresh_at: str = "trajectory", actor_id: str = "", drift_mode: str = "exact"):
        if refresh_at not in ("trajectory", "decision"):
            raise ValueError(refresh_at)
        if drift_mode not in ("exact", "meta_sum"):
            raise ValueError(drift_mode)
        self.root = os.path.abspath(root)
        self.refresh_at = refresh_at
        self.actor_id = actor_id
        self.drift_mode = drift_mode
        self.version = -1
        self.state_version = -1       # weights version under which the latent state was (re)built
        self.drift = 0.0              # relative parameter change since the latent state was (re)built
        self.history: list = []       # [(version, t_adopted, boundary)]
        self.load_s = 0.0
        self._loaded: Optional[dict] = None     # state dict currently loaded
        self._state_ref: Optional[dict] = None  # state dict the latent state was built under ("exact" mode)
        self._state_norm = 0.0                  # ‖θ‖ of that state ("meta_sum" mode)
        self._abs_since_state = 0.0             # summed published ‖Δθ‖ since then ("meta_sum" mode)

    def newer(self) -> Optional[dict]:
        lt = read_latest(self.root)
        if lt is None or int(lt["version"]) <= self.version:
            return None
        return lt

    def fetch(self, ptr: dict) -> tuple:
        """-> (state_dict, meta) of the version a LATEST pointer names; retries once if it was pruned meanwhile."""
        for _ in range(3):
            try:
                path = os.path.join(self.root, ptr["dir"], "weights.pt")
                payload = load_weights(path, ptr.get("sha256"))
                with open(os.path.join(self.root, ptr["dir"], "meta.json")) as fh:
                    meta = json.load(fh)
                return payload["state_dict"], meta
            except FileNotFoundError:
                ptr = read_latest(self.root) or ptr
        raise FileNotFoundError(f"checkpoint {ptr} vanished")

    def maybe_refresh(self, policy, boundary: str) -> bool:
        """Adopt the newest validated version if ``boundary`` is a safe point under this client's policy.
        ``policy.load_weights(state_dict, version)`` must swap the weights. Returns True when weights changed."""
        if boundary not in ("trajectory", "decision"):
            raise ValueError(boundary)
        if boundary == "decision" and self.refresh_at != "decision":
            return False
        ptr = self.newer()
        if ptr is None:
            return False
        t0 = time.perf_counter()
        sd, meta = self.fetch(ptr)
        new_v = int(meta["version"])
        if new_v <= self.version:
            return False
        policy.load_weights(sd, new_v)
        self.load_s += time.perf_counter() - t0
        old_v = self.version
        self.version = new_v
        self._loaded = sd
        if self.drift_mode == "exact":
            ref = self._state_ref
            self.drift = float(relative_change(sd, ref)) if ref is not None else 0.0
        else:
            self._abs_since_state += self._meta_sum(old_v, new_v)
            self.drift = self._abs_since_state / max(self._state_norm, 1e-12) if self._state_norm else math.inf
        self.history.append((self.version, time.time(), boundary))
        return True

    def _meta_sum(self, old_v: int, new_v: int) -> float:
        """Sum of the published ``abs_change`` along the chain of validated versions new_v -> prev_version -> ...
        -> old_v (each relative to its predecessor). inf when a link is missing (pruned metadata, or no predecessor),
        so the change cannot be bounded and the caller rebuilds."""
        tot, cur = 0.0, new_v
        while cur != old_v:
            try:
                with open(os.path.join(self.root, _vdir(cur), "meta.json")) as fh:
                    m = json.load(fh)
            except (OSError, json.JSONDecodeError):
                return math.inf
            if not m.get("validated") or m.get("prev_version") is None or m.get("abs_change") is None:
                return math.inf
            tot += float(m["abs_change"])
            cur = int(m["prev_version"])
            if cur < old_v:
                return math.inf
        return tot

    def wait_for_version(self, policy, min_version: int = 0, timeout_s: float = 120.0, poll_s: float = 0.05) -> int:
        """Block until a validated version >= ``min_version`` exists and load it (actor start-up)."""
        t0 = time.monotonic()
        while True:
            lt = read_latest(self.root)
            if lt is not None and int(lt["version"]) >= min_version:
                self.maybe_refresh(policy, "trajectory")
                self.state_rebuilt()
                return self.version
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"no validated checkpoint >= {min_version} in {self.root} after {timeout_s}s")
            time.sleep(poll_s)

    def state_rebuilt(self) -> None:
        """Call after the latent state was reset or reconstructed under the current weights."""
        self.drift = 0.0
        self.state_version = self.version
        self._state_ref = self._loaded if self.drift_mode == "exact" else None
        self._abs_since_state = 0.0
        if self.drift_mode == "meta_sum" and self._loaded is not None:
            self._state_norm = state_norm(self._loaded)
