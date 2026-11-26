"""The independent evaluator: a separate process that owns worlds, partitions, the sealed salt and the verifiers
(MASTER_SPEC Parts 14.3, 14.4, 15.2, 15.7; U.7, T.7, SP-14.4, SP-15.7).

    ev = Evaluator.start(sealed_dir=...)            # spawns `python -I` with its own registry and salt
    t = ev.open_episode("t1:rapid:mk_elastic", 1_000_003, purpose="dev")        # validation item -> ticket
    env = ev.participant_env(t)                     # what a learner may hold: observe/execute/advance/tools only
    run_episode(env, learner, learn=False, rng=..., verify=False)
    rec = ev.result(t)                              # verified, normalised outcome (evaluator channel only)
    ev.preregister({...metrics...}); ev.open_sealed("final", ack=SEALED_ACK, purpose=..., prereg=h)

Trust model. The evaluator process is started with ``-I`` (no user site, no PYTHONPATH, no current directory on the
path), an environment holding only what it needs, and two socket channels: the *evaluator* channel (kept by the
harness) and the *participant* channel (the only thing a learner gets). Messages are newline-delimited JSON - never
pickle - so no learner byte is ever executed by the evaluator; a malformed message is refused and logged. The
participant channel serves `PARTICIPANT_OPS` on episodes named by unguessable tickets; everything else (results,
registry, sealed sets, proposals) exists only on the evaluator channel. The evaluator builds its registry from the
code, so relabelling a family in the learner's process changes nothing here (its digest is reported). The sealed
salt is read only by this process from a directory named only in its environment (`CRUCIBLE_V5_SEALED_DIR`); the
parent never reads it. Residual: same uid and same filesystem (no uid jail): code with arbitrary file access in the
learner's process could look for the salt file on disk; it never receives its location or contents.

Partitions. Episodes of every partition can be opened only here. Results are tagged with their partition and with
the requester's declared purpose; `results(view=...)` serves only the partitions a view allows (the curriculum's view
is train + validation; sealed and evaluation results are refused and the attempt is counted). Sealed sets have an
access budget; a request beyond it is refused, logged and marks the set contaminated. Confirmation runs
(evaluation and sealed partitions) require a preregistration hash registered before the first confirmation
episode; development runs (validation partition) do not, and a confirmation item can never be opened as a
development item.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from typing import Any, Mapping

PARTICIPANT_OPS = ("agents", "observe", "execute", "advance", "tools", "observe_since", "done", "propose")
EVALUATOR_OPS = ("ping", "registry", "open_episode", "result", "results", "declare_sealed", "open_sealed",
                 "sealed_status", "preregister", "log", "propose", "fresh_check", "close")
SEALED_ACK = "I am running the final sealed evaluation and will not use it for tuning"
CONFIRMATION_PARTITIONS = ("evaluation", "sealed")
VIEWS = {"curriculum": ("train", "validation"), "development": ("train", "validation"),
         "confirmation": ("train", "validation", "evaluation", "sealed")}
MAX_LINE = 1 << 20


class EvaluatorError(RuntimeError):
    pass


# ====================================================================== the server (child process)
class _Server:
    def __init__(self, sealed_dir: str | None):
        from .distribution import build_registry
        self.reg = build_registry()
        self.sealed_dir = sealed_dir
        self.salt = self._salt() if sealed_dir else None
        self.episodes: dict[str, dict] = {}
        self.records: dict[str, dict] = {}
        self.log: list[dict] = []
        self.sealed: dict[str, dict] = {}
        self.prereg: dict[str, dict] = {}
        self.refused: list[dict] = []
        self.curriculum_sealed_served = 0
        self.dev_items: set[tuple[str, int]] = set()
        self.conf_items: set[tuple[str, int]] = set()
        self.accepted: list[str] = []
        self.base_digest = self.reg.digest

    def _salt(self) -> bytes:
        os.makedirs(self.sealed_dir, mode=0o700, exist_ok=True)
        p = os.path.join(self.sealed_dir, "v5_sealed_salt")
        if not os.path.exists(p):
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(fd, secrets.token_bytes(32))
            os.close(fd)
        return open(p, "rb").read()

    def _sealed_seed(self, name: str, gen: int, entry: str, k: int) -> int:
        from .distribution import SEALED_SEED0
        d = hmac.new(self.salt, f"{name}|{gen}|{entry}|{k}".encode(), hashlib.sha256).digest()
        return SEALED_SEED0 + int.from_bytes(d[:8], "big") % SEALED_SEED0

    def _refuse(self, side: str, op: str, why: str) -> dict:
        self.refused.append({"side": side, "op": op, "why": why[:200], "t": time.time()})
        return {"ok": False, "error": why}

    # ---------------------------------------------------------------- evaluator ops
    def ev_ping(self) -> dict:
        return {"pid": os.getpid(), "isolated": bool(sys.flags.isolated), "registry": self.reg.version,
                "salt_loaded": self.salt is not None}

    def ev_registry(self) -> dict:
        evals = sorted(w["id"] for w in self.reg.worlds() if w["family_partition"] != "train")
        return {"version": self.reg.version, "digest": self.reg.digest, "base_digest": self.base_digest,
                "counts": self.reg.kind_counts(), "tiers": {str(k): v for k, v in self.reg.tiers().items()},
                "evaluation_families": evals, "accepted_proposals": list(self.accepted)}

    def ev_open_episode(self, entry: str, seed: int, purpose: str = "dev", difficulty: dict | None = None,
                        prereg: str | None = None, _sealed_alias: str | None = None) -> dict:
        from .episodes import references, world
        part = self.reg.partition_of(entry, int(seed))
        key = (entry, int(seed))
        if part == "sealed" and _sealed_alias is None:
            return self._refuse("evaluator", "open_episode", "sealed items are opened only through open_sealed")
        if part in CONFIRMATION_PARTITIONS:
            if prereg not in self.prereg:
                return self._refuse("evaluator", "open_episode",
                                    f"{part} items are confirmation items: register the prespecified metrics first")
            if key in self.dev_items:
                return self._refuse("evaluator", "open_episode", "this item was used in a development run")
            self.conf_items.add(key)
        else:
            if key in self.conf_items:
                return self._refuse("evaluator", "open_episode", "this item belongs to a confirmation run")
            self.dev_items.add(key)
        env, refs = world(self.reg, entry, int(seed), difficulty)
        t = secrets.token_hex(16)
        self.episodes[t] = {"env": env, "refs": refs, "entry": entry, "seed": int(seed), "partition": part,
                            "purpose": purpose, "traj": [], "interactions": 0, "prereg": prereg,
                            "alias": _sealed_alias, "opened": time.time()}
        self.log.append({"op": "open", "ticket": t[:8], "entry": entry if part != "sealed" else _sealed_alias,
                         "partition": part, "purpose": purpose})
        return {"ok": True, "ticket": t, "partition": part}

    def ev_result(self, ticket: str, view: str = "confirmation") -> dict:
        from .episodes import normalised
        ep = self.episodes.get(ticket)
        if ep is None:
            return self._refuse("evaluator", "result", "unknown ticket")
        if ep["partition"] not in VIEWS.get(view, ()):
            if view == "curriculum":
                self.log.append({"op": "refused_result", "view": view, "partition": ep["partition"]})
            return self._refuse("evaluator", "result", f"view {view!r} may not read {ep['partition']} results")
        traj = ep["traj"]
        H = len(traj)
        span = max(ep["refs"]["upper"] - ep["refs"]["wait"], 1e-9)
        q = max(1, H // 4)
        rec = {"partition": ep["partition"], "purpose": ep["purpose"], "ticks": H,
               "value": traj[-1] if traj else None, "interactions": ep["interactions"],
               "score": normalised(traj[-1], ep["refs"]) if traj else None,
               "early_rate": (traj[q - 1] / q) * H / span if H else None,
               "late_rate": ((traj[-1] - traj[-q - 1]) / q) * H / span if H > q else None,
               "verifier": "ExperienceEnv.verify (evaluator process)", "prereg": ep["prereg"]}
        if ep["partition"] == "sealed":
            rec["item"] = ep["alias"]
        else:
            rec["entry"], rec["seed"] = ep["entry"], ep["seed"]
        if view == "curriculum" and ep["partition"] in ("sealed", "evaluation"):
            self.curriculum_sealed_served += 1
        self.records[ticket] = rec
        return {"ok": True, "result": rec}

    def ev_results(self, view: str) -> dict:
        allowed = VIEWS.get(view)
        if allowed is None:
            return self._refuse("evaluator", "results", f"unknown view {view!r}")
        out = [r for r in self.records.values() if r["partition"] in allowed]
        return {"ok": True, "results": out}

    def ev_declare_sealed(self, name: str, entries: list[str], per_entry: int, budget: int) -> dict:
        if self.salt is None:
            return self._refuse("evaluator", "declare_sealed", "no sealed directory configured")
        if name in self.sealed:
            return self._refuse("evaluator", "declare_sealed", f"sealed set {name} exists")
        for e in entries:
            if e not in self.reg:
                return self._refuse("evaluator", "declare_sealed", f"unknown entry {e}")
        self.sealed[name] = {"entries": list(entries), "per_entry": int(per_entry), "budget": int(budget),
                             "accesses": 0, "generation": 0, "contaminated": False, "refused": 0}
        return {"ok": True, "items": len(entries) * int(per_entry)}

    def ev_open_sealed(self, name: str, ack: str, purpose: str, prereg: str | None = None) -> dict:
        s = self.sealed.get(name)
        if s is None:
            return self._refuse("evaluator", "open_sealed", "unknown sealed set")
        if ack != SEALED_ACK:
            return self._refuse("evaluator", "open_sealed", "wrong acknowledgement")
        if prereg not in self.prereg:
            return self._refuse("evaluator", "open_sealed", "no preregistration for this confirmation run")
        if s["contaminated"] or s["accesses"] >= s["budget"]:
            s["refused"] += 1
            s["contaminated"] = True
            self.log.append({"op": "sealed_refused", "set": name, "purpose": purpose})
            return self._refuse("evaluator", "open_sealed", "access budget exhausted: the set is contaminated")
        s["accesses"] += 1
        self.log.append({"op": "sealed_open", "set": name, "access": s["accesses"], "budget": s["budget"],
                         "purpose": purpose, "prereg": prereg})
        tickets = []
        for e in s["entries"]:
            for k in range(s["per_entry"]):
                alias = f"sealed/{name}/g{s['generation']}/{e}/{k}"
                r = self.ev_open_episode(e, self._sealed_seed(name, s["generation"], e, k), purpose=purpose,
                                         prereg=prereg, _sealed_alias=alias)
                if not r.get("ok"):
                    return r
                tickets.append({"ticket": r["ticket"], "item": alias})
        return {"ok": True, "tickets": tickets, "access": s["accesses"], "budget": s["budget"]}

    def ev_sealed_status(self) -> dict:
        return {"ok": True, "sets": {k: {x: v[x] for x in ("budget", "accesses", "contaminated", "refused",
                                                           "generation")} for k, v in self.sealed.items()},
                "curriculum_sealed_served": self.curriculum_sealed_served}

    def ev_preregister(self, record: dict) -> dict:
        h = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        if h in self.prereg:
            return {"ok": True, "prereg": h, "registered_at": self.prereg[h]["t"]}
        self.prereg[h] = {"record": record, "t": time.time(), "episodes_before": len(self.episodes)}
        return {"ok": True, "prereg": h, "registered_at": self.prereg[h]["t"]}

    def ev_log(self) -> dict:
        return {"ok": True, "log": list(self.log), "refused": list(self.refused)}

    def ev_propose(self, structure: Any, proposer: str = "harness") -> dict:
        r = self.pa("propose", structure)
        return {"ok": True, "verdict": r["value"]}

    def ev_fresh_check(self) -> dict:
        return {"ok": True, "modules": sorted(m for m in sys.modules if m.startswith("crucible."))[:5]}

    # ---------------------------------------------------------------- participant ops
    def pa(self, op: str, ticket: Any, *args) -> dict:
        if op == "propose":                            # a world proposal: public verdict only (Part 14.4)
            from .proposal import validate_proposal
            v = validate_proposal(self.reg, ticket, proposer="participant", accepted=self.accepted)
            if v["accepted"]:
                self.accepted.append(v["structure_id"])
                self.reg = self.reg.extend([v.pop("_entry")])
            v.pop("_entry", None)
            return {"ok": True, "value": v}
        ep = self.episodes.get(ticket) if isinstance(ticket, str) else None
        if ep is None:
            return self._refuse("participant", op, "unknown ticket")
        env = ep["env"]
        if op == "agents":
            return {"ok": True, "value": env.agents()}
        if op == "observe":
            return {"ok": True, "value": env.observe(str(args[0]))}
        if op == "tools":
            return {"ok": True, "value": env.tools(str(args[0]))}
        if op == "observe_since":
            evs, cur = env.observe_since(str(args[0]), int(args[1]))
            return {"ok": True, "value": [evs, cur]}
        if op == "execute":
            if not isinstance(args[1], dict):
                return self._refuse("participant", op, "an action is a JSON object")
            before = env.changes
            r = env.execute(str(args[0]), dict(args[1]))
            ep["interactions"] += int(env.changes > before)
            return {"ok": True, "value": r}
        if op == "advance":
            from .episodes import verified_value
            env.advance(int(args[0]))
            ep["traj"].append(verified_value(env.verify()))
            return {"ok": True, "value": None}
        if op == "done":
            return {"ok": True, "value": bool(getattr(env, "done", False))}
        return self._refuse("participant", op, "not a participant operation")


def _serve(p_fd: int, e_fd: int, sealed_dir: str | None) -> None:
    import selectors
    srv = _Server(sealed_dir)
    socks = {"participant": socket.socket(fileno=p_fd), "evaluator": socket.socket(fileno=e_fd)}
    files = {k: (s.makefile("rb"), s.makefile("wb")) for k, s in socks.items()}
    sel = selectors.DefaultSelector()
    for k, s in socks.items():
        sel.register(s, selectors.EVENT_READ, k)
    for k in socks:
        _send(files[k][1], {"ok": True, "ready": True, "pid": os.getpid()})
    alive = set(socks)
    while "evaluator" in alive:
        for key, _ in sel.select():
            side = key.data
            rf, wf = files[side]
            line = rf.readline(MAX_LINE + 1)
            if not line:
                sel.unregister(socks[side])
                alive.discard(side)
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
                if not isinstance(msg, dict) or not isinstance(msg.get("op"), str):
                    raise ValueError("a request is a JSON object with an 'op'")
            except (ValueError, UnicodeDecodeError) as e:
                _send(wf, srv._refuse(side, "?", f"malformed request ({type(e).__name__})"))
                continue
            op, args, kw = msg["op"], list(msg.get("args") or []), dict(msg.get("kw") or {})
            try:
                if side == "participant":
                    if op not in PARTICIPANT_OPS:
                        out = srv._refuse(side, op, f"{op!r} is not available to participants")
                    else:
                        out = srv.pa(op, *args)
                else:
                    if op not in EVALUATOR_OPS:
                        out = srv._refuse(side, op, f"unknown evaluator operation {op!r}")
                    elif op == "close":
                        _send(wf, {"ok": True})
                        return
                    else:
                        out = getattr(srv, f"ev_{op}")(*args, **{k: v for k, v in kw.items() if not k.startswith("_")})
                        if "ok" not in out:
                            out = {"ok": True, **out}
            except Exception as e:  # noqa: BLE001 - an error goes back to the caller, the server keeps running
                out = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            _send(wf, out)


def _send(wf, obj) -> None:
    wf.write((json.dumps(obj, default=_json_default) + "\n").encode())
    wf.flush()


def _json_default(x):
    import numpy as np
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON: {type(x).__name__}")


# ====================================================================== the client side
class Channel:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.rf, self.wf = sock.makefile("rb"), sock.makefile("wb")

    def call(self, op: str, *args, **kw) -> dict:
        _send(self.wf, {"op": op, "args": list(args), "kw": kw})
        line = self.rf.readline(MAX_LINE * 64)
        if not line:
            raise EvaluatorError("the evaluator process closed the channel")
        return json.loads(line)

    def raw(self, data: bytes) -> dict:
        """Send raw bytes (for probes of what a participant may not do)."""
        self.wf.write(data)
        self.wf.flush()
        return json.loads(self.rf.readline(MAX_LINE * 64))


class ParticipantEnv:
    """The learner's view of one evaluator-held episode: the participant protocol and nothing else."""
    reward_mode = "observation_only"

    def __init__(self, chan: Channel, ticket: str):
        self._chan, self._t = chan, ticket
        self.changes = 0

    def _v(self, op, *a):
        r = self._chan.call(op, self._t, *a)
        if not r.get("ok"):
            raise EvaluatorError(r.get("error"))
        return r["value"]

    def agents(self):
        return self._v("agents")

    def observe(self, agent):
        return self._v("observe", agent)

    def tools(self, agent):
        return self._v("tools", agent)

    def execute(self, agent, action):
        return self._v("execute", agent, action)

    def advance(self, seconds):
        return self._v("advance", int(seconds))

    def observe_since(self, agent, cursor):
        return tuple(self._v("observe_since", agent, int(cursor)))

    def reward(self, agent):
        return 0.0


class Evaluator:
    """Harness-side handle of the evaluator process (holds the evaluator channel)."""

    def __init__(self, proc, ev: Channel, part: Channel, part_sock_for_child: socket.socket | None = None):
        self.proc, self.ev, self.part = proc, ev, part
        self.pid = proc.pid

    @classmethod
    def start(cls, sealed_dir: str | None = None, timeout: float = 120.0) -> "Evaluator":
        p_parent, p_child = socket.socketpair()
        e_parent, e_child = socket.socketpair()
        env = isolated_env()
        if sealed_dir:
            env["CRUCIBLE_V5_SEALED_DIR"] = sealed_dir
        boot = ("import sys,os; sys.path.insert(0, sys.argv[1]); "
                "from crucible.research.v5.evaluator import _serve; "
                "_serve(int(sys.argv[2]), int(sys.argv[3]), os.environ.get('CRUCIBLE_V5_SEALED_DIR'))")
        proc = subprocess.Popen([sys.executable, "-I", "-c", boot, src_root(), str(p_child.fileno()),
                                 str(e_child.fileno())], pass_fds=(p_child.fileno(), e_child.fileno()), env=env,
                                cwd=env["HOME"], stdin=subprocess.DEVNULL)
        p_child.close()
        e_child.close()
        e_parent.settimeout(timeout)
        p_parent.settimeout(timeout)
        ev, part = Channel(e_parent), Channel(p_parent)
        for ch in (ev, part):
            line = ch.rf.readline()
            if not line or not json.loads(line).get("ready"):
                proc.kill()
                raise EvaluatorError("the evaluator process did not start")
        e_parent.settimeout(None)
        p_parent.settimeout(None)
        return cls(proc, ev, part)

    def call(self, op, *args, **kw) -> dict:
        r = self.ev.call(op, *args, **kw)
        return r

    def must(self, op, *args, **kw) -> dict:
        r = self.ev.call(op, *args, **kw)
        if not r.get("ok"):
            raise EvaluatorError(f"{op}: {r.get('error')}")
        return r

    def open_episode(self, entry, seed, purpose="dev", difficulty=None, prereg=None) -> str:
        return self.must("open_episode", entry, int(seed), purpose=purpose, difficulty=difficulty,
                         prereg=prereg)["ticket"]

    def participant_env(self, ticket: str) -> ParticipantEnv:
        return ParticipantEnv(self.part, ticket)

    def result(self, ticket: str, view: str = "confirmation") -> dict:
        return self.must("result", ticket, view=view)["result"]

    def preregister(self, record: dict) -> str:
        return self.must("preregister", record)["prereg"]

    def close(self) -> None:
        try:
            self.ev.call("close")
        except (EvaluatorError, OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def src_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def isolated_env(home: str | None = None) -> dict:
    """A minimal environment for a fresh child: a private empty HOME and TMPDIR, one thread, nothing else of the
    parent's (no PYTHONPATH, no tokens, no CRUCIBLE_* paths)."""
    import tempfile
    base = tempfile.mkdtemp(prefix="crucible-v5-child-")
    home = home or os.path.join(base, "home")
    os.makedirs(home, exist_ok=True)
    tmp = os.path.join(base, "tmp")
    os.makedirs(tmp, exist_ok=True)
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": home, "TMPDIR": tmp, "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
            "CRUCIBLE_CACHE": os.path.join(base, "cache"), "LANG": "C.UTF-8"}


def run_remote_episode(ev: Evaluator, ticket: str, learner, rng, *, learn: bool = False,
                       budget: int | None = None) -> dict:
    """Drive `learner` through the participant channel of an evaluator-held episode; the measured record comes
    from the evaluator channel."""
    from .episodes import run_episode
    env = ev.participant_env(ticket)
    run_episode(env, learner, learn=learn, rng=rng, budget=budget, verify=False)
    return ev.result(ticket)
