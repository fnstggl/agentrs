"""The native Python API: `WorldServer`.

A `WorldServer` hosts any number of persistent worlds on one durable store (``memory://``, ``sqlite:///path``,
``postgresql://...``) and mediates every agent interaction with them:

* **Operator operations** (in-process research code is the operator): create/load/list/close worlds, admit actors
  (`join_world`), move the clock (`advance_time`), snapshot / fork / restore, read the full event log, evaluate.
* **Actor operations** take an actor credential - an `ActorIdentity` or its bearer token string - and are scoped
  to what that actor may see and do: `observe`, `available_tools`, `execute_action`, filtered `replay_events`.

Tokens are random secrets issued at join (or by `issue_token`); only their SHA-256 hashes are stored, in the
store's blob table, so a fresh `WorldServer` on the same store authenticates the same tokens. Per-actor
observation cursors ("sessions") are persisted the same way, so an observation after a restart contains exactly
the events the actor has not seen yet.

Worlds never reset. `restore_world` does not rewrite history: it forks the snapshot into a new world id and points
the world's *name* (an alias) at the fork; the superseded world stays in the store with status ``superseded``.

All public methods are thread-safe (one re-entrant lock per server); the HTTP server relies on this.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import pkgutil
import re
import secrets
import threading
import time as _time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from ..core.errors import ProtocolError
from ..core.kernel import MODULES, ActionResult, Event, World, WorldSpec
from ..storage.base import Store
from . import render as R

API_VERSION = "crucible-api/1"

__all__ = ["API_VERSION", "WorldServer", "ActorIdentity", "Observation", "ActionResult", "EvaluationResult",
           "ApiError", "NotFoundError", "AuthError", "ForbiddenError", "ConflictError", "open_store",
           "ensure_modules"]

_BLOB = "crucible-api"
_MODULE_PACKAGES = ("services", "economy", "sandbox", "web", "worlds", "actors", "goals", "evaluation", "bench")
_HIDDEN_HOSTS = ("admin",)  # operator-only tools are never listed to in-world actors


# ====================================================================== errors
class ApiError(Exception):
    """An API-level failure (bad request, unknown world, bad credentials). In-world failures are never raised:
    they come back as ``ok=False`` action results."""
    code = "api_error"
    status = 400

    def __init__(self, message: str, code: str | None = None, status: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status:
            self.status = status

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


class NotFoundError(ApiError):
    code, status = "not_found", 404


class AuthError(ApiError):
    code, status = "unauthenticated", 401


class ForbiddenError(ApiError):
    code, status = "forbidden", 403


class ConflictError(ApiError):
    code, status = "conflict", 409


# ====================================================================== public types
@dataclass(frozen=True)
class ActorIdentity:
    """Credential of one actor in one world. `token` is the bearer secret; keep it private."""
    world_id: str
    principal: str
    token: str = field(repr=False)
    account: str | None = None

    def to_dict(self, include_token: bool = True) -> dict:
        d = {"world_id": self.world_id, "principal": self.principal, "account": self.account}
        if include_token:
            d["token"] = self.token
        return d


@dataclass
class Observation:
    """What one actor perceives at one moment.

    `events` are the actor-visible events since its previous observation (compact dicts: seq, t, type, actor,
    payload), at most `max_events` of them; `dropped` counts newer-than-cursor visible events not included
    (oldest are dropped first) and `unscanned` counts log entries skipped without inspection (cursor too far
    behind). `summary` holds per-module summaries (`World.observe`). `tools` is included when the tool list
    changed since the previous observation (always on the first one) or when requested."""
    world_id: str
    principal: str
    time: int
    seq: int
    events: list[dict]
    dropped: int
    summary: dict
    last_result: dict | None
    tools_changed: bool
    tools: list[dict] | None = None
    new_tools: list[str] = field(default_factory=list)
    removed_tools: list[str] = field(default_factory=list)
    unscanned: int = 0
    observed: int = 1
    text: str = ""
    api: str = API_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    world_id: str
    time: int
    seq: int
    ok: bool
    invariant_violations: list[str]
    metrics: dict
    source: str  # "hook" (world.hooks["evaluate"]) or "builtin"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Session:
    cursor: int
    tool_names: list[str] | None = None
    last_result: dict | None = None
    observed: int = 0


# ====================================================================== helpers
def open_store(url: str) -> Store:
    from ..storage.postgres import open_store as _open
    return _open(url)


def _registers(path: str, name: str) -> bool:
    """Does the source file register a Crucible module called `name`? Checked on the text, without importing:
    some files in these packages are standalone scripts that must never be imported (e.g. sandbox/boot.py)."""
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return False
    return "register_module" in src and re.search(rf"""^\s+name\s*=\s*["']{re.escape(name)}["']""", src,
                                                  re.M) is not None


def ensure_modules(names) -> None:
    """Import the Python files that register the given Crucible module names (modules register on import).
    Only files under ``crucible.<services|economy|...>`` whose source declares ``@register_module`` with a
    matching ``name = "..."`` are imported; other modules must be imported by the caller."""
    missing = [n for n in names if n not in MODULES]
    if not missing:
        return
    for pkg in _MODULE_PACKAGES:
        try:
            p = importlib.import_module(f"crucible.{pkg}")
        except ImportError:
            continue
        infos = sorted(pkgutil.iter_modules(getattr(p, "__path__", [])), key=lambda i: i.name)
        for n in list(missing):
            # the file named like the module first, then any other file of the package that registers it
            for info in sorted(infos, key=lambda i: i.name != n):
                if info.ispkg:
                    continue
                path = os.path.join(info.module_finder.path, info.name + ".py")
                if not _registers(path, n):
                    continue
                try:
                    importlib.import_module(f"crucible.{pkg}.{info.name}")
                except ImportError:
                    continue
                if n in MODULES:
                    missing.remove(n)
                    break
        if not missing:
            return


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def compact_event(ev: Event) -> dict:
    return {"seq": ev.seq, "t": ev.t, "type": ev.type, "actor": ev.actor, "payload": ev.payload}


# ====================================================================== server
class WorldServer:
    """Hosts persistent worlds on one store. See the module docstring and docs/AGENT_API.md.

    Parameters: `store_url` (or an already-open `store`); `autoflush` writes events to the store after every
    mutating call; `lease_ttl_s` makes this server the single writer of every world it loads (``None`` disables
    leases); `max_events` bounds the events per observation; `max_scan` bounds how far back an observation looks.
    """

    def __init__(self, store_url: str = "memory://", *, store: Store | None = None, strict: bool = False,
                 autoflush: bool = True, lease_ttl_s: float | None = 600.0, max_events: int = 50,
                 max_scan: int = 200_000, persist_sessions: bool = True,
                 renderer: Callable[[Observation], str] | None = None, hide_own_commands: bool = True):
        self.store_url = store_url if store is None else f"<{getattr(store, 'kind', 'store')}>"
        self.store = store if store is not None else open_store(store_url)
        self.strict = strict
        self.autoflush = autoflush
        self.lease_ttl_s = lease_ttl_s
        self.max_events = max_events
        self.max_scan = max_scan
        self.persist_sessions = persist_sessions
        self.renderer = renderer or R.render_observation
        self.hide_own_commands = hide_own_commands
        self.owner = f"api-{secrets.token_hex(6)}"
        self._lock = threading.RLock()
        self._worlds: dict[str, World] = {}
        self._ephemeral: dict[str, Store] = {}
        self._tokens: dict[str, dict[str, dict]] = {}
        self._sessions: dict[tuple[str, str], _Session] = {}
        self._leases: dict[str, float] = {}
        self._aliases: dict[str, str] = self._load_json("aliases") or {}
        self._closed = False

    # ------------------------------------------------------------------ blob helpers
    def _store_for(self, world_id: str) -> Store:
        return self._ephemeral.get(world_id, self.store)

    def _load_json(self, key: str, store: Store | None = None) -> Any:
        try:
            b = (store or self.store).get_blob(f"{_BLOB}/{key}")
        except NotImplementedError:
            return None
        return json.loads(b.decode()) if b else None

    def _save_json(self, key: str, value: Any, store: Store | None = None) -> None:
        try:
            (store or self.store).put_blob(f"{_BLOB}/{key}", json.dumps(value, sort_keys=True).encode())
        except NotImplementedError:
            pass

    # ------------------------------------------------------------------ world registry
    def resolve(self, world_id: str) -> str:
        """World name or id -> current world id (follows `restore_world` aliases)."""
        seen = 0
        while world_id in self._aliases and seen < 32:
            world_id = self._aliases[world_id]
            seen += 1
        return world_id

    def _acquire(self, world_id: str) -> None:
        if self.lease_ttl_s is None or world_id in self._ephemeral:
            return
        if not self.store.acquire(world_id, self.owner, self.lease_ttl_s):
            raise ConflictError(f"world {world_id} is owned by another writer (lease held)")
        self._leases[world_id] = _time.monotonic()

    def _renew(self, world_id: str) -> None:
        t = self._leases.get(world_id)
        if t is not None and _time.monotonic() - t > self.lease_ttl_s / 3:
            self._acquire(world_id)

    def world(self, world_id: str) -> World:
        """The live kernel `World` (loaded on demand). Operator access: read state, register hooks."""
        with self._lock:
            wid = self.resolve(world_id)
            w = self._worlds.get(wid)
            if w is not None:
                return w
            store = self._store_for(wid)
            info = store.world_info(wid)
            if info is None:
                raise NotFoundError(f"no world {world_id!r}")
            ensure_modules([m["name"] for m in info["spec"]["modules"]])
            self._acquire(wid)
            try:
                w = World.load(store, wid, strict=self.strict)
            except Exception:
                self._release(wid)
                raise
            self._worlds[wid] = w
            return w

    def _release(self, wid: str) -> None:
        if self._leases.pop(wid, None) is not None:
            try:
                self.store.release(wid, self.owner)
            except Exception:  # noqa: BLE001
                pass

    def _flush(self, w: World) -> None:
        if self.autoflush:
            w.flush()
        self._renew(w.world_id)

    def _unique_id(self, base: str) -> str:
        if self._store_for(base).world_info(base) is None and base not in self._worlds:
            return base
        n = 2
        while self.store.world_info(f"{base}-{n}") is not None or f"{base}-{n}" in self._worlds:
            n += 1
        return f"{base}-{n}"

    @staticmethod
    def _spec_from(config: Any, seed: int | None) -> WorldSpec:
        if isinstance(config, WorldSpec):
            spec = config
            if seed is not None and seed != spec.seed:
                spec = WorldSpec(spec.family, int(seed), spec.modules, spec.params, spec.kernel)
            return spec
        if isinstance(config, Mapping) and "modules" in config:
            d = dict(config)
            d.setdefault("family", "custom")
            if seed is not None:
                d["seed"] = int(seed)
            d.setdefault("seed", 0)
            return WorldSpec.from_dict(d)
        try:
            from ..worldgen import generate  # type: ignore[attr-defined]
        except ImportError as e:
            raise ApiError("config is not a WorldSpec (no 'modules') and crucible.worldgen.generate is "
                           f"unavailable: {e}", code="worldgen_unavailable") from e
        spec = generate(config, 0 if seed is None else int(seed))
        if isinstance(spec, Mapping):
            spec = WorldSpec.from_dict(spec)
        if not isinstance(spec, WorldSpec):
            raise ApiError(f"worldgen.generate returned {type(spec).__name__}, expected WorldSpec")
        return spec

    # ------------------------------------------------------------------ operator: lifecycle
    def create_world(self, config: Any, seed: int | None = None, *, world_id: str | None = None) -> str:
        """Create and persist a new world. `config` is a `WorldSpec`, a WorldSpec dict (has ``"modules"``), or a
        world-generator config passed to ``crucible.worldgen.generate(config, seed)``. Returns the world id."""
        with self._lock:
            spec = self._spec_from(config, seed)
            ensure_modules([m["name"] for m in spec.modules])
            if world_id is not None:
                if self.store.world_info(world_id) is not None or world_id in self._aliases:
                    raise ConflictError(f"world {world_id} exists")
                wid = world_id
            else:
                wid = self._unique_id(f"w-{spec.hash()[:12]}")
            try:
                w = World.create(spec, wid, self.store, strict=self.strict)
            except ProtocolError as e:
                raise ApiError(str(e), code="bad_spec") from e
            self._worlds[wid] = w
            self._acquire(wid)
            return wid

    def load_world(self, world_id: str) -> dict:
        """Load a persisted world into memory (snapshot + replay, hash chain verified). Returns its info."""
        with self._lock:
            self.world(world_id)
            return self.world_info(world_id)

    def world_info(self, world_id: str) -> dict:
        with self._lock:
            wid = self.resolve(world_id)
            return self._info_exact(wid, world_id if world_id != wid else None)

    def _info_exact(self, wid: str, name: str | None = None) -> dict:
        info = self._store_for(wid).world_info(wid)
        if info is None:
            raise NotFoundError(f"no world {wid!r}")
        w = self._worlds.get(wid)
        out = {"world_id": wid, "name": name, "family": info["spec"]["family"], "seed": info["spec"]["seed"],
               "modules": [m["name"] for m in info["spec"]["modules"]], "parent": info.get("parent"),
               "status": info.get("status"), "live": w is not None, "ephemeral": wid in self._ephemeral}
        if w is not None:
            out.update(time=w.now, seq=w.seq, hash=w.last_hash)
        return out

    def list_worlds(self) -> list[dict]:
        """Every world in the store (plus live ephemeral forks), then the name aliases created by restores."""
        with self._lock:
            ids = [i["id"] for i in self.store.list_worlds()] + sorted(self._ephemeral)
            return [self._info_exact(i) for i in ids] + [
                {"world_id": self.resolve(a), "name": a, "alias": True} for a in sorted(self._aliases)]

    def close_world(self, world_id: str) -> None:
        """Flush and unload a world (it stays in the store). An ephemeral fork is discarded entirely."""
        with self._lock:
            self._close_exact(self.resolve(world_id))

    def _close_exact(self, wid: str) -> None:
        w = self._worlds.pop(wid, None)
        if w is not None:
            w.close()
        for k in [k for k in self._sessions if k[0] == wid]:
            if w is not None and wid not in self._ephemeral:
                self._save_session(*k)
            del self._sessions[k]
        self._release(wid)
        self._tokens.pop(wid, None)
        self._ephemeral.pop(wid, None)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for wid in list(self._worlds):
                self._close_exact(wid)
            self._closed = True
            self.store.close()

    def __enter__(self) -> "WorldServer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ tokens & actors
    def _tokens_of(self, wid: str) -> dict[str, dict]:
        t = self._tokens.get(wid)
        if t is None:
            t = self._load_json(f"tokens/{wid}", self._store_for(wid)) or {}
            self._tokens[wid] = t
        return t

    def _save_tokens(self, wid: str) -> None:
        self._save_json(f"tokens/{wid}", self._tokens_of(wid), self._store_for(wid))

    def issue_token(self, world_id: str, principal: str, *, account: str | None = None) -> ActorIdentity:
        """Operator: issue a (new) bearer token for an existing principal (e.g. one created by the generator)."""
        with self._lock:
            w = self.world(world_id)
            p = w.principal(principal)
            if p is None or p["kind"] not in ("person", "service"):
                raise NotFoundError(f"no actor principal {principal!r} in {w.world_id}")
            token = "cru_" + secrets.token_urlsafe(24)
            self._tokens_of(w.world_id)[_hash_token(token)] = {"principal": principal, "issued_seq": w.seq}
            self._save_tokens(w.world_id)
            if account is None and "bank" in w.modules:
                try:
                    account = w.modules["bank"].primary_account(w.state, principal)
                except Exception:  # noqa: BLE001
                    account = None
            return ActorIdentity(w.world_id, principal, token, account)

    def revoke_token(self, world_id: str, token: str) -> bool:
        with self._lock:
            wid = self.resolve(world_id)
            ok = self._tokens_of(wid).pop(_hash_token(token), None) is not None
            self._save_tokens(wid)
            return ok

    def authenticate(self, world_id: str, actor: ActorIdentity | str) -> ActorIdentity:
        """Token (or `ActorIdentity`) -> verified identity in the *current* world for that name or id."""
        with self._lock:
            token = actor.token if isinstance(actor, ActorIdentity) else actor
            if not isinstance(token, str) or not token:
                raise AuthError("missing actor token")
            w = self.world(world_id)
            rec = self._tokens_of(w.world_id).get(_hash_token(token))
            if rec is None:
                raise AuthError("invalid token for this world")
            p = w.principal(rec["principal"])
            if p is None:
                raise AuthError(f"principal {rec['principal']} does not exist in world {w.world_id}")
            account = actor.account if isinstance(actor, ActorIdentity) else None
            return ActorIdentity(w.world_id, rec["principal"], token, account)

    def join_world(self, world_id: str, handle: str, kind: str = "agent", endowment: int = 0, *,
                   name: str | None = None, password: str | None = None, controller: str = "",
                   info: dict | None = None) -> ActorIdentity:
        """Operator: admit a new actor via the recorded ``admin.join`` command (identity + bank account +
        endowment in cents) and issue its token. The actor's first observation starts just before its join."""
        with self._lock:
            w = self.world(world_id)
            if "admin" not in w.modules:
                raise ApiError(f"world {w.world_id} has no 'admin' module; cannot admit actors", code="no_admin")
            seq0 = w.seq
            args: dict[str, Any] = {"handle": handle, "kind": kind, "endowment": int(endowment),
                                    "controller": controller}
            if name:
                args["name"] = name
            if password:
                args["password"] = password
            if info:
                args["info"] = info
            from ..core.errors import CrucibleError
            try:
                out = w.system_call("admin.join", args)
            except CrucibleError as e:
                raise ApiError(f"join failed: {e.message}", code=e.code,
                               status=409 if e.code == "conflict" else 400) from e
            self._flush(w)
            ident = self.issue_token(w.world_id, out["principal"], account=out.get("account"))
            self._sessions[(w.world_id, ident.principal)] = _Session(cursor=seq0)
            self._save_session(w.world_id, ident.principal)
            return ident

    # ------------------------------------------------------------------ sessions
    def _session(self, w: World, principal: str) -> _Session:
        key = (w.world_id, principal)
        s = self._sessions.get(key)
        if s is None:
            d = self._load_json(f"session/{w.world_id}/{principal}", self._store_for(w.world_id)) \
                if self.persist_sessions else None
            if d:
                s = _Session(cursor=int(d["cursor"]), tool_names=d.get("tool_names"),
                             last_result=d.get("last_result"), observed=int(d.get("observed", 0)))
            else:
                par = (self._store_for(w.world_id).world_info(w.world_id) or {}).get("parent")
                s = _Session(cursor=min(par["seq"] if par else w.seq, w.seq))
            self._sessions[key] = s
        return s

    def _save_session(self, wid: str, principal: str) -> None:
        if not self.persist_sessions:
            return
        s = self._sessions[(wid, principal)]
        self._save_json(f"session/{wid}/{principal}", asdict(s), self._store_for(wid))

    def _tools(self, w: World, principal: str) -> list[dict]:
        return [t for t in w.available_tools(principal) if t["name"].split(".", 1)[0] not in _HIDDEN_HOSTS]

    # ------------------------------------------------------------------ actor operations
    def available_tools(self, world_id: str, actor: ActorIdentity | str) -> list[dict]:
        """Tool specs (MCP-shaped: name, description, inputSchema, annotations) the actor can call now."""
        with self._lock:
            ident = self.authenticate(world_id, actor)
            return self._tools(self.world(ident.world_id), ident.principal)

    def observe(self, world_id: str, actor: ActorIdentity | str, *, max_events: int | None = None,
                include_tools: str = "changed", peek: bool = False) -> Observation:
        """New visible events since this actor's previous observation, module summaries, the last action result
        and the tool list when it changed. `include_tools`: "changed" | "always" | "never". `peek=True` does not
        advance the cursor."""
        with self._lock:
            ident = self.authenticate(world_id, actor)
            w = self.world(ident.world_id)
            p = ident.principal
            s = self._session(w, p)
            limit = self.max_events if max_events is None else max_events
            start = max(s.cursor, w.seq - self.max_scan)
            unscanned = max(0, start - s.cursor)
            ps = w.principals_of(p)
            evs = []
            for e in w.events_range(start, w.seq):
                if not w.visible(e, ps):
                    continue
                if self.hide_own_commands and e.type in ("cmd", "cmd.rejected") and e.actor == p:
                    continue
                evs.append(e)
            dropped = max(0, len(evs) - limit)
            evs = evs[len(evs) - limit:] if limit > 0 else []
            tools = self._tools(w, p)
            names = [t["name"] for t in tools]
            changed = s.tool_names is None or names != s.tool_names
            prev = set(s.tool_names or [])
            new_tools = [n for n in names if n not in prev] if s.tool_names is not None else []
            removed = sorted(prev - set(names))
            show = include_tools == "always" or (include_tools == "changed" and changed)
            summary = {k: v for k, v in w.observe(p).items() if k not in ("time", "principal")}
            obs = Observation(world_id=w.world_id, principal=p, time=w.now, seq=w.seq,
                              events=[compact_event(e) for e in evs], dropped=dropped, summary=summary,
                              last_result=s.last_result, tools_changed=changed, tools=tools if show else None,
                              new_tools=new_tools, removed_tools=removed, unscanned=unscanned,
                              observed=s.observed + 1)
            obs.text = self.renderer(obs)
            if not peek:
                s.cursor = w.seq
                s.tool_names = names
                s.observed += 1
                self._save_session(w.world_id, p)
            return obs

    def execute_action(self, world_id: str, actor: ActorIdentity | str, action: Any) -> dict:
        """Execute ``{"tool": str, "args": {...}[, "idempotency_key": str]}`` (or model text containing one, see
        `render.parse_action`) as the actor. Returns the ActionResult dict (ok, t, tool, cost_s, result|error).
        Unparseable text is not sent to the world; it returns ``ok=False`` with error code ``parse_error``."""
        with self._lock:
            ident = self.authenticate(world_id, actor)
            w = self.world(ident.world_id)
            s = self._session(w, ident.principal)
            act = R.parse_action(action) if not isinstance(action, Mapping) else R.normalize_action(action)
            if "error" in act:
                d = {"ok": False, "t": w.now, "tool": "", "cost_s": 0, "error": act["error"]}
            else:
                d = w.execute(ident.principal, act).to_dict()
                self._flush(w)
            s.last_result = d
            return d

    # ------------------------------------------------------------------ operator: time, snapshots, forks
    def advance_time(self, world_id: str, until: int | None = None, *, dt: int | None = None,
                     max_jobs: int | None = None) -> dict:
        """Run scheduled processes up to `until` (absolute) or now+`dt` seconds; returns time, runs, seq."""
        with self._lock:
            w = self.world(world_id)
            if until is None:
                if dt is None:
                    raise ApiError("advance_time needs `until` or `dt`")
                until = w.now + int(dt)
            if int(until) < w.now:
                raise ApiError(f"time cannot go backwards ({until} < {w.now})")
            runs = w.advance_time(int(until), max_jobs=max_jobs)
            self._flush(w)
            return {"world_id": w.world_id, "time": w.now, "runs": runs, "seq": w.seq}

    def snapshot_world(self, world_id: str) -> str:
        """Persist an immutable snapshot of the world's current state; returns the snapshot id."""
        with self._lock:
            w = self.world(world_id)
            sid = w.snapshot()
            self._renew(w.world_id)
            return sid

    def _find_snapshot(self, snapshot_id: str) -> tuple[dict, bytes, Store]:
        for st in [self.store, *self._ephemeral.values()]:
            try:
                meta, blob = st.get_snapshot(snapshot_id)
                return meta, blob, st
            except Exception:  # noqa: BLE001 - KeyError (memory/sqlite) or backend-specific
                continue
        raise NotFoundError(f"no snapshot {snapshot_id!r}")

    def fork_world(self, snapshot_id: str, new_world_id: str | None = None, *, ephemeral: bool = False,
                   inherit_tokens: bool = True) -> str:
        """Create an isolated world from a snapshot (copy-on-snapshot; history shared up to the snapshot).
        `ephemeral=True` keeps the fork in a private in-memory store that disappears on `close_world` (episodic
        training). With `inherit_tokens`, actors of the source world keep their tokens in the fork."""
        with self._lock:
            meta, blob, _ = self._find_snapshot(snapshot_id)
            src = meta["world"]
            if new_world_id is None:
                n = 1
                base = src if len(src) <= 48 else f"w-{secrets.token_hex(5)}"
                while (self.store.world_info(f"{base}.f{n}") is not None or f"{base}.f{n}" in self._worlds
                       or f"{base}.f{n}" in self._ephemeral):
                    n += 1
                new_world_id = f"{base}.f{n}"
            elif (self.store.world_info(new_world_id) is not None or new_world_id in self._worlds
                  or new_world_id in self._aliases):
                raise ConflictError(f"world {new_world_id} exists")
            import pickle
            import zlib
            spec = WorldSpec.from_dict(pickle.loads(zlib.decompress(blob))["spec"])
            ensure_modules([m["name"] for m in spec.modules])
            from ..storage.memory import MemoryStore
            store = MemoryStore() if ephemeral else self.store
            w = World.from_blob(blob, world_id=new_world_id, store=store, fork=True, strict=self.strict)
            if ephemeral:
                self._ephemeral[new_world_id] = store
            self._worlds[new_world_id] = w
            self._acquire(new_world_id)
            if inherit_tokens:
                toks = dict(self._tokens_of(src))
                for wid in {self.resolve(src)} - {src}:
                    toks.update(self._tokens_of(wid))
                self._tokens[new_world_id] = toks
                self._save_tokens(new_world_id)
            return new_world_id

    def restore_world(self, snapshot_id: str, name: str | None = None) -> str:
        """Make the world called `name` continue from a snapshot, without rewriting history: fork the snapshot
        into a new world id and point the name at it (the old world is kept, status ``superseded``). Actors keep
        their tokens. `name` defaults to the name of the snapshot's world. Returns the new world id."""
        with self._lock:
            meta, _, _ = self._find_snapshot(snapshot_id)
            src = meta["world"]
            if name is None:
                if src in self._aliases:
                    name = src
                else:
                    name = next((a for a, t in sorted(self._aliases.items()) if t == src), src)
            current = self.resolve(name)
            n = 1
            while self.store.world_info(f"{name}.r{n}") is not None:
                n += 1
            new = self.fork_world(snapshot_id, f"{name}.r{n}")
            toks = self._tokens_of(new)
            for k, v in self._tokens_of(current).items():
                toks.setdefault(k, v)
            self._save_tokens(new)
            self._aliases[name] = new
            self._save_json("aliases", self._aliases)
            if self.store.world_info(current) is not None and current != new:
                self.store.set_status(current, "superseded")
                if current in self._worlds:
                    self._close_exact(current)
            return new

    # ------------------------------------------------------------------ operator: history & evaluation
    def _read_lineage(self, wid: str, from_seq: int, to_seq: int | None) -> list[dict]:
        store = self._store_for(wid)
        info = store.world_info(wid)
        if info is None:
            return []
        par = info.get("parent")
        out: list[dict] = []
        if par and from_seq < par["seq"]:
            out = self._read_lineage(par["world"], from_seq, par["seq"] if to_seq is None else min(to_seq,
                                                                                                    par["seq"]))
            from_seq = par["seq"]
        if to_seq is None or to_seq > from_seq:
            out.extend(store.read_events(wid, from_seq, to_seq))
        return out

    def replay_events(self, world_id: str, from_seq: int = 0, to_seq: int | None = None,
                      actor: ActorIdentity | str | None = None) -> list[dict]:
        """Event dicts with from_seq < seq <= to_seq, following fork lineage into parent worlds. With `actor`,
        only events that actor may see; without, the full log (operator)."""
        with self._lock:
            wid = self.resolve(world_id)
            w = self._worlds.get(wid)
            if w is not None:
                w.flush()
            elif self._store_for(wid).world_info(wid) is None:
                raise NotFoundError(f"no world {world_id!r}")
            evs = self._read_lineage(wid, int(from_seq), None if to_seq is None else int(to_seq))
            if actor is None:
                return evs
            ident = self.authenticate(wid, actor)
            w = self.world(wid)
            ps = w.principals_of(ident.principal)
            return [d for d in evs if w.visible(Event.from_dict(d), ps)]

    def evaluate_world(self, world_id: str, evaluator_context: Any = None) -> EvaluationResult:
        """Delegates to ``world.hooks["evaluate"](world, evaluator_context)`` if registered (it may return an
        `EvaluationResult` or a metrics dict); otherwise invariant checks and basic statistics."""
        with self._lock:
            w = self.world(world_id)
            viol = w.check_invariants()
            hook = w.hooks.get("evaluate")
            if hook is not None:
                out = hook(w, evaluator_context)
                if isinstance(out, EvaluationResult):
                    return out
                return EvaluationResult(w.world_id, w.now, w.seq, not viol, viol, dict(out or {}), "hook")
            kinds: dict[str, int] = {}
            for _, p in w.state.scan("principals"):
                kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
            metrics = {"principals": kinds, "events": w.seq, "stats": dict(w.stats), "digest": w.digest(),
                       "tables": {t: w.state.count(t) for t in sorted(w.state.tables)}}
            return EvaluationResult(w.world_id, w.now, w.seq, not viol, viol, metrics, "builtin")
