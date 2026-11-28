"""cam_live.nn.persist — the experiential core's persistent latent state, carried across decisions (PS, amendment
item 3; spec 1.3 "recurrent or persistent latent state", spec 2.1 ``b_t = F_theta(H_t, m_t)``).

Before PS every caller passed ``state=None`` to the core (KF-E04): each belief was computed from the learned initial
slots and the re-encoded context window (at most ``max_context`` items of the *current* objective). Nothing the agent
experienced outside that window — earlier in a long objective or in an earlier objective — reached a belief except
through the slowly changing weights and the per-kind fast posteriors (which are reset at every objective).

**Definition.** Positions ``p = 0, 1, 2, ...`` are the agent's committed decisions in order over its whole lifetime
(objectives included). For each position the chain stores (durably, ``<run_dir>/learner/core_state.sqlite``):

* ``W_p``: the item refs of the context window the decision was made on (= ``ExperienceRecord.context_item_refs``);
* ``key(W_p)`` / ``key(P_p)``: hashes of the refs of that window and of the post-result window
  (``post_item_refs``) exactly as the tokenizer sees them (items present in the store, newest ``max_context``);
* the anchor ``Sbar_p``: the core state the decision was made with (slots rounded to float16, newest-event time).

The state of position ``q`` is recomputed with the *current* weights from the anchor ``K`` positions back::

    S_q = Phi(W_{q-1}, Phi(W_{q-2}, ... Phi(W_{q-K}, Sbar_{q-K}) ...)),   Phi(W, S) = ExperientialCore(tok(W), S).state

(``Sbar_0`` is the fresh state; positions below ``K`` start there). A belief about position ``q`` — at decision time
and whenever a stored record of it is re-encoded for learning — is ``b = F(W, S_q)`` for the record's own window
(context window: ``q``; post window: ``q + 1``). Hence:

* decision-time and replayed beliefs are the same function of the same inputs (no train/test mismatch; tested);
* gradients of every loss flow through the ``K`` recomputed transitions into everything that writes, carries and
  reads the state (truncated back-propagation through time over decisions, with stored anchors that are refreshed
  ``K`` steps by the current weights before use, as in R2D2's stored-state burn-in). The anchor is a constant: credit
  for information that must survive more than ``K`` decisions reaches the write only through the carried state's
  usefulness at intermediate positions (which are themselves trained);
* anchors are inputs, never targets: an anchor written by older weights is refreshed by ``K`` transitions of the
  current ones; staleness beyond that is a measured limitation (``status()["anchor_age"]``).

**Initial state, reset, export.** The carried state is experience-dependent state, like the fast beliefs: a
weights-only export (``credit.checkpoint.export_learner_weights``, spec 15.3) carries no chain, so a model rebuilt from
it (``load_exported_model``) or a new agent loaded from it starts from the *initial chain state* (nothing committed:
every belief from the fresh state). :meth:`PersistentStateChain.initial` evaluates the current model at that initial
state without changing anything (what an export computes); :meth:`PersistentStateChain.reset` starts a new segment:
the next decision is made from the fresh state and no recomputation crosses the boundary (earlier records keep their
own states).

The chain is opt-in per model: ``CamLiveModel.core_forward(tokens, None)`` resolves ``None`` through an attached chain
(row by row: a stored context window -> its position's state, a stored post window -> the next position's state,
any other window inside a :meth:`PersistentStateChain.live` block -> the next uncommitted position's state, anything
else -> the fresh state). Without an attached chain, or with ``enabled = False``, the model computes exactly what it
computed before PS.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .api import CoreState, TokenBatch

#: version of the chain's on-disk / checkpoint format
PERSIST_VERSION = 1
SECTION = "persistent_state"
FP16_MAX = 6.0e4


@dataclass
class PersistSettings:
    """``cfg.section("persistent_state")`` (all optional)."""
    enabled: bool = False           # off by default: PS's evidence shows no measurable benefit yet (docs/PS.md)
    unroll: int = 2                 # K: transitions recomputed (with gradient) from the stored anchor
    anchor_cache: int = 2048        # anchors kept in memory (the database holds all of them)
    legacy_rho: float = 0.0         # retention given to a checkpoint written before PS (0 = compute as before)

    @classmethod
    def from_cfg(cls, cfg: Any, **over: Any) -> "PersistSettings":
        sec = dict(cfg.section(SECTION)) if hasattr(cfg, "section") else {}
        sec.update(over)
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(sec) - known)
        if unknown:
            raise KeyError(f"unknown keys in the {SECTION!r} config section: {unknown}")
        s = cls(**sec)
        if s.unroll < 1:
            raise ValueError("persistent_state.unroll must be >= 1 (with 0 no transition is ever applied: every state "
                             "would be the fresh one; switch the chain off with enabled: false instead)")
        return s


def _rows(tok: TokenBatch, rows: Sequence[int]) -> TokenBatch:
    """The given rows of a token batch, trimmed to their longest real context (right padding)."""
    ii = torch.tensor(list(rows), device=tok.x.device, dtype=torch.long)
    pad = tok.pad_mask.index_select(0, ii)
    T = max(1, int(pad.sum(1).max()))
    return TokenBatch(x=tok.x.index_select(0, ii)[:, :T], pad_mask=pad[:, :T], time=tok.time.index_select(0, ii)[:, :T],
                      kind_ids=tok.kind_ids.index_select(0, ii)[:, :T], values=tok.values.index_select(0, ii)[:, :T],
                      value_mask=tok.value_mask.index_select(0, ii)[:, :T],
                      refs=[tok.refs[r][:T] for r in rows] if tok.refs else [])


def window_key(refs: Sequence[str]) -> str:
    """Hash of a window's item refs in order (as the tokenizer sees them)."""
    h = hashlib.sha256()
    for r in refs:
        h.update(r.encode("utf-8", "surrogatepass"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


class PersistentStateChain:
    """See the module docstring. ``path`` = "" keeps the chain in memory (tests, diagnostics)."""

    def __init__(self, model: Any, store: Any, settings: Optional[PersistSettings] = None, *, path: str = "",
                 attach: bool = True):
        self.model = model
        self.store = store
        self.settings = settings or PersistSettings()
        self.path = path
        self.counters: Counter = Counter()
        self._lock = threading.RLock()
        self._live_depth = 0
        self._live_cache: Optional[Tuple[int, CoreState]] = None
        self._last_live: Optional[Tuple[int, CoreState]] = None
        self._anchors: "OrderedDict[int, Tuple[Tensor, float]]" = OrderedDict()
        self._refs: "OrderedDict[int, List[str]]" = OrderedDict()
        self.restore_report: Dict[str, Any] = {}
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.db = sqlite3.connect(path or ":memory:", check_same_thread=False, isolation_level=None)
        if path:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS positions (pos INTEGER PRIMARY KEY, decision_id TEXT UNIQUE NOT "
                        "NULL, objective_id TEXT, ts REAL, key_pre TEXT, key_post TEXT, refs TEXT, anchor BLOB, "
                        "anchor_time REAL, k INTEGER, d INTEGER, written_at REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
        self._initial_depth = 0
        row = self.db.execute("SELECT v FROM meta WHERE k='boundaries'").fetchone()
        self._boundaries: List[int] = sorted(int(x) for x in json.loads(row[0])) if row else []
        self._pre: Dict[str, int] = {}
        self._post: Dict[str, int] = {}
        self._by_decision: Dict[str, int] = {}
        for pos, did, kp, kq in self.db.execute("SELECT pos, decision_id, key_pre, key_post FROM positions ORDER BY pos"):
            self._index(pos, did, kp, kq)
        self.n = len(self._by_decision)
        if self.n and max(self._by_decision.values()) != self.n - 1:
            raise RuntimeError(f"{path}: positions are not contiguous")
        if attach:
            self.attach()

    # ---- wiring -------------------------------------------------------------------------------------------------
    def attach(self) -> None:
        self.model.persistent = self

    def detach(self) -> None:
        if getattr(self.model, "persistent", None) is self:
            self.model.persistent = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled)

    def _index(self, pos: int, did: str, key_pre: str, key_post: str) -> None:
        self._by_decision[did] = pos
        if key_pre:
            self._pre.setdefault(key_pre, pos)
        if key_post:
            self._post[key_post] = pos

    def _core(self) -> Any:
        return self.model.exp.core

    # ---- decision time ------------------------------------------------------------------------------------------
    @contextlib.contextmanager
    def live(self) -> Iterator["PersistentStateChain"]:
        """Decision-time block: a window that is not a stored one gets the state of the next uncommitted position
        (computed once per outermost block, without gradient)."""
        with self._lock:
            self._live_depth += 1
        try:
            yield self
        finally:
            with self._lock:
                self._live_depth -= 1
                if self._live_depth == 0:
                    self._live_cache = None

    @contextlib.contextmanager
    def initial(self) -> Iterator["PersistentStateChain"]:
        """Evaluate at the chain's initial state: inside the block every ``core_forward(tokens, None)`` reads the
        fresh state, as a model rebuilt from a weights-only export does. Nothing is changed."""
        with self._lock:
            self._initial_depth += 1
        try:
            yield self
        finally:
            with self._lock:
                self._initial_depth -= 1

    def reset(self, reason: str = "") -> int:
        """Start a new segment at the next position: the next decision is made from the fresh state and no state is
        recomputed across the boundary; stored records before it keep their own states. Returns the boundary."""
        with self._lock:
            if self.n not in self._boundaries:
                self._boundaries.append(self.n)
                self.db.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('boundaries', ?)",
                                (json.dumps(self._boundaries),))
            self._live_cache = self._last_live = None
            self.counters["resets"] += 1
            return self.n

    def _segment_start(self, q: int) -> int:
        b = 0
        for x in self._boundaries:
            if x <= q:
                b = max(b, x)
        return b

    @property
    def is_live(self) -> bool:
        return self._live_depth > 0

    def live_state(self) -> CoreState:
        """The state the next decision is made with (position ``n``), without gradient."""
        with self._lock:
            if self._live_cache is not None and self._live_cache[0] == self.n:
                return self._live_cache[1]
            with torch.no_grad():
                st = self.state_for([self.n])
            st = CoreState(st.slots.detach(), st.last_time, st.step)
            self._last_live = (self.n, st)
            if self.is_live:
                self._live_cache = (self.n, st)
            return st

    def commit(self, decision_id: str, *, context_refs: Sequence[str], post_refs: Sequence[str],
               objective_id: str = "", ts: float = 0.0) -> int:
        """Record a decision as the next position (idempotent per decision id: a later delivery only updates the
        post window, e.g. the executed result of a decision that waited for approval). The anchor is the state the
        decision was made with."""
        with self._lock:
            kp = self._key_of(context_refs)
            kq = self._key_of(post_refs) if post_refs else ""
            if decision_id in self._by_decision:
                pos = self._by_decision[decision_id]
                if kq:
                    self.db.execute("UPDATE positions SET key_post=? WHERE pos=?", (kq, pos))
                    self._post[kq] = pos
                self.counters["commit_repeated"] += 1
                return pos
            pos = self.n
            if self._last_live is not None and self._last_live[0] == pos:
                st = self._last_live[1]
            else:
                with torch.no_grad():
                    st = self.state_for([pos])
                self.counters["commit_recomputed_anchor"] += 1
            slots = st.slots[0].detach().to("cpu", torch.float32)
            if not torch.isfinite(slots).all():
                raise FloatingPointError("persistent state is not finite")
            blob = slots.clamp(-FP16_MAX, FP16_MAX).to(torch.float16).numpy().tobytes()
            lt = float(st.last_time[0])
            refs = [r for r in context_refs]
            self.db.execute("INSERT INTO positions (pos, decision_id, objective_id, ts, key_pre, key_post, refs, anchor, "
                            "anchor_time, k, d, written_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (pos, decision_id, objective_id, float(ts), kp, kq, json.dumps(refs), blob,
                             None if math.isnan(lt) else lt, int(slots.shape[0]), int(slots.shape[1]), time.time()))
            self._index(pos, decision_id, kp, kq)
            self.n = pos + 1
            self._remember_anchor(pos, slots.to(torch.float16).to(torch.float32), lt)
            self._live_cache = None
            self.counters["commits"] += 1
            return pos

    # ---- resolution (called by CamLiveModel.core_forward) -------------------------------------------------------
    def _key_of(self, refs: Sequence[str]) -> str:
        """Key of a window as the tokenizer will see it: the refs the store holds, newest ``max_context``."""
        refs = list(refs)
        have = [it.ref for it in self._items(refs)] if self.store is not None else refs
        mc = int(self.model.core_cfg.max_context)
        return window_key(have[-mc:] if len(have) > mc else have)

    def _items(self, refs: Sequence[str]) -> List[Any]:
        try:
            return self.store.items(list(refs), missing_ok=True)       # SqliteExperienceStore: counted, no warning
        except TypeError:                                              # the plain ExperienceStore protocol
            return self.store.items(list(refs))

    def row_positions(self, tokens: TokenBatch) -> List[Optional[int]]:
        n_real = tokens.pad_mask.sum(1).tolist()
        out: List[Optional[int]] = []
        for b, row in enumerate(tokens.refs or [[] for _ in n_real]):
            key = window_key(row[:int(n_real[b])])
            if key in self._pre:
                out.append(self._pre[key])
                self.counters["rows_context"] += 1
            elif key in self._post:
                out.append(self._post[key] + 1)
                self.counters["rows_post"] += 1
            elif self.is_live:
                out.append(-1)                                     # the next uncommitted position
                self.counters["rows_live"] += 1
            else:
                out.append(None)
                self.counters["rows_fresh"] += 1
        return out

    def resolve(self, tokens: TokenBatch) -> Optional[CoreState]:
        """The core state for each row of ``tokens`` (None = every row fresh)."""
        if not self.enabled or self._initial_depth > 0:
            return None
        with self._lock:
            pos = self.row_positions(tokens)
            if all(p is None for p in pos):
                return None
            B = len(pos)
            core = self._core()
            dev = core.slot_init.device
            need = sorted({p for p in pos if p is not None and p >= 0})
            states: Dict[int, Tuple[Tensor, float]] = {}
            if need:
                st = self.state_for(need)
                for i, p in enumerate(need):
                    states[p] = (st.slots[i], float(st.last_time[i]))
            if any(p == -1 for p in pos):
                ls = self.live_state()
                states[-1] = (ls.slots[0], float(ls.last_time[0]))
            fresh = core.init_state(1, dev)
            rows, times = [], []
            for p in pos:
                if p is None:
                    rows.append(fresh.slots[0])
                    times.append(float("nan"))
                else:
                    s, t = states[p]
                    rows.append(s)
                    times.append(t)
            return CoreState(slots=torch.stack(rows, 0), last_time=torch.tensor(times, dtype=torch.float64, device=dev),
                             step=1)

    # ---- the recomputed chain -------------------------------------------------------------------------------------
    def state_for(self, positions: Sequence[int]) -> CoreState:
        """S_q for each q in ``positions`` (0 <= q <= n), recomputed from the anchor K positions back with the current
        weights; differentiable when gradients are enabled (the anchor is a constant)."""
        core = self._core()
        dev = core.slot_init.device
        K = int(self.settings.unroll)
        qs = [int(q) for q in positions]
        if any(q < 0 or q > self.n for q in qs):
            raise IndexError(f"positions out of range 0..{self.n}: {qs}")
        starts = [max(self._segment_start(q), q - K) for q in qs]
        slots, times = [], []
        fresh = core.init_state(1, dev).slots[0]       # a fresh anchor is the *current* learned initial state
        for a in starts:
            s, t = (fresh, float("nan")) if a >= self.n else self._anchor(a)    # a >= n: nothing committed yet
            slots.append(fresh if math.isnan(t) else s.to(dev))
            times.append(t)
        S = torch.stack(slots, 0)
        T = torch.tensor(times, dtype=torch.float64, device=dev)
        n_win = [q - a for q, a in zip(qs, starts)]
        L = max(n_win) if n_win else 0
        steps = []                                       # per step: rows taking part and the window each reads
        for j in range(L):
            idx = [i for i, m in enumerate(n_win) if j >= L - m]
            steps.append((idx, [starts[i] + (j - (L - n_win[i])) for i in idx]))
        uniq = sorted({w for _, wins in steps for w in wins})
        if uniq:                                         # every window of every step tokenised in one call
            row_of = {w: k for k, w in enumerate(uniq)}
            allt = self.model.tokenize([self._items(self._window_refs(w)) for w in uniq])
        for idx, wins in steps:
            tok = _rows(allt, [row_of[w] for w in wins])
            ii = torch.tensor(idx, device=dev, dtype=torch.long)
            out = core(tok, CoreState(S.index_select(0, ii), T.index_select(0, ii), 1))
            S = S.index_copy(0, ii, out.state.slots)
            T = T.index_copy(0, ii, out.state.last_time)
            self.counters["refresh_windows"] += len(idx)
        return CoreState(slots=S, last_time=T, step=1)

    def _window_refs(self, pos: int) -> List[str]:
        r = self._refs.get(pos)
        if r is None:
            row = self.db.execute("SELECT refs FROM positions WHERE pos=?", (pos,)).fetchone()
            r = json.loads(row[0])
            self._refs[pos] = r
            while len(self._refs) > max(16, self.settings.anchor_cache):
                self._refs.popitem(last=False)
        return r

    def _anchor(self, pos: int) -> Tuple[Tensor, float]:
        hit = self._anchors.get(pos)
        if hit is not None:
            self._anchors.move_to_end(pos)
            return hit
        row = self.db.execute("SELECT anchor, anchor_time, k, d FROM positions WHERE pos=?", (pos,)).fetchone()
        if row is None:
            raise KeyError(f"no anchor at position {pos}")
        import numpy as np
        arr = np.frombuffer(row[0], dtype=np.float16).reshape(int(row[2]), int(row[3]))
        s = torch.from_numpy(arr.astype(np.float32))
        t = float("nan") if row[1] is None else float(row[1])
        self._remember_anchor(pos, s, t)
        return s, t

    def _remember_anchor(self, pos: int, s: Tensor, t: float) -> None:
        self._anchors[pos] = (s, t)
        self._anchors.move_to_end(pos)
        while len(self._anchors) > max(4, self.settings.anchor_cache):
            self._anchors.popitem(last=False)

    # ---- persistence ---------------------------------------------------------------------------------------------
    def rho(self) -> List[float]:
        return [float(x) for x in self._core().retention().detach().cpu()]

    def state_dict(self) -> Dict[str, Any]:
        """What the agent's checkpoint records (the anchors themselves are durable in the database)."""
        last = ""
        if self.n:
            last = self.db.execute("SELECT decision_id FROM positions WHERE pos=?", (self.n - 1,)).fetchone()[0]
        return {"version": PERSIST_VERSION, "n_positions": self.n, "last_decision_id": last, "path": self.path,
                "boundaries": list(self._boundaries),
                "settings": asdict(self.settings), "counters": dict(self.counters)}

    def load_state_dict(self, d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Restore from an agent checkpoint. ``None`` = a checkpoint written before PS: its weights were trained on
        fresh-state beliefs only, so the retention is set to ``settings.legacy_rho`` (0: the restored model computes
        exactly what it computed before) and learning opens it where carrying the state lowers the losses."""
        rep: Dict[str, Any] = {"positions_in_database": self.n}
        if not d:
            with torch.no_grad():
                self._core().persist_rho.fill_(float(self.settings.legacy_rho))
            rep.update(legacy_checkpoint=True, rho_initialised=float(self.settings.legacy_rho))
        else:
            if int(d.get("version", 0)) != PERSIST_VERSION:
                raise ValueError(f"persistent-state format {d.get('version')} is not {PERSIST_VERSION}")
            n_ck = int(d.get("n_positions", 0))
            rep.update(legacy_checkpoint=False, positions_at_checkpoint=n_ck)
            if self.n < n_ck:
                rep["missing_positions"] = n_ck - self.n            # database older than the checkpoint
            elif self.n > n_ck:
                rep["positions_after_checkpoint"] = self.n - n_ck   # journalled after it (kept, like the store)
            if n_ck and self.n >= n_ck:
                did = self.db.execute("SELECT decision_id FROM positions WHERE pos=?", (n_ck - 1,)).fetchone()[0]
                rep["consistent"] = did == d.get("last_decision_id")
            self.counters.update({f"restored_{k}": v for k, v in (d.get("counters") or {}).items()
                                  if k in ("commits",)})
        rep["rho"] = self.rho()
        self._live_cache = self._last_live = None
        self.restore_report = rep
        return rep

    def status(self) -> Dict[str, Any]:
        rho = self.rho()
        out = {"enabled": self.enabled, "positions": self.n, "unroll": int(self.settings.unroll),
               "rho_mean": sum(rho) / max(1, len(rho)), "rho_min": min(rho) if rho else None,
               "rho_max": max(rho) if rho else None, "counters": dict(self.counters)}
        if self.restore_report:
            out["restore"] = dict(self.restore_report)
        return out

    def close(self) -> None:
        self.detach()
        try:
            self.db.close()
        except Exception:  # noqa: BLE001
            pass
