"""Sealed evaluation: the final test sets, their access log and their contamination rule.

A sealed set is a list of validated tasks on **test-split** mechanism families whose seeds nobody can know in
advance: each seed is derived from a secret salt that lives outside the repository (`~/.crucible/sealed_salt`,
created on first use with mode 600; `CRUCIBLE_SEALED_DIR` moves the whole directory) together with the set's name,
its generation and the task's position. The salt is never returned, printed or logged by anything in this module.

Rules, all enforced here:

* A set is declared once (`create_set`) with its families, its size and its **budget**: how many times it may be
  handed out. Only families of the test split are accepted.
* `open_set(name, ack=SEALED_ACK, purpose=...)` hands out the tasks. It requires the exact acknowledgement
  string, a stated purpose, and it appends one line to the access log (`sealed_access.jsonl`, next to the salt)
  with the purpose, the caller and the set's running counter - before anything is returned.
* One access is one evaluation session: evaluate everything you declared in it. A request beyond the budget is
  refused, logged, and marks the set **contaminated**; every later request is refused until `rotate` creates a
  new generation (new seeds from the same salt; the rotation is logged with its reason). Results of the old
  generation stay tagged with the generation they came from.
* Results produced through `evaluate` carry `"sealed": {...}` (set, generation, access number, budget, purpose)
  and do not contain the task seeds: tasks appear under the aliases `sealed/<set>/g<generation>/<family>/<k>`.
  A result file with seeds in it would unseal the set for everybody who reads the repository.
* Learner code and world code cannot reach the accessor: no module under `crucible` other than this one imports
  it (checked by a test that scans imports), and at run time `open_set` refuses when any frame on the call stack
  belongs to a `crucible` module outside `ALLOWED_CALLERS` or to a method of a `Learner`.

The 3.0 locked test (`evaluation.harness.locked_test_goals`) uses the same acknowledgement pattern; this module
adds the secret seeds, the budget and the generations.
"""
from __future__ import annotations

import fcntl
import getpass
import hashlib
import json
import os
import secrets
import sys
import time
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

SEALED_ACK = "I am running the final sealed evaluation and will not use it for tuning"
SALT_FILE, LOG_FILE, REGISTRY_FILE = "sealed_salt", "sealed_access.jsonl", "sealed_sets.json"
SEALED_SEED0, SEALED_SEED_SPAN = 2_000_000_000, 2_000_000_000        # far from the training and validation ranges
ALLOWED_CALLERS = ("crucible.research.sealed", "crucible.cli", "crucible.__main__")
MAX_CANDIDATES = 4            # per requested task: how many seeds are tried before the family is reported short


class SealedSetError(PermissionError):
    """The accessor refused: wrong acknowledgement, unknown set, exhausted budget or contaminated set."""


def sealed_dir() -> str:
    return os.environ.get("CRUCIBLE_SEALED_DIR") or os.path.join(os.path.expanduser("~"), ".crucible")


def _path(name: str) -> str:
    return os.path.join(sealed_dir(), name)


@contextmanager
def _locked():
    os.makedirs(sealed_dir(), mode=0o700, exist_ok=True)
    fd = os.open(_path(".sealed.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _salt() -> bytes:
    """The secret. Created on first use (32 random bytes, mode 600). Never leaves this module."""
    p = _path(SALT_FILE)
    os.makedirs(sealed_dir(), mode=0o700, exist_ok=True)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w") as fh:
            fh.write(secrets.token_hex(32))
        _log({"event": "salt_created"})
    with open(p) as fh:
        s = fh.read().strip()
    if len(s) < 32:
        raise SealedSetError("the sealed salt file is damaged")
    return bytes.fromhex(s)


def _seed(salt: bytes, name: str, generation: int, family: str, k: int) -> int:
    h = hashlib.blake2b(f"{name}|{int(generation)}|{family}|{int(k)}".encode(), key=salt[:64], digest_size=8)
    return SEALED_SEED0 + int.from_bytes(h.digest(), "big") % SEALED_SEED_SPAN


def _log(entry: dict) -> None:
    os.makedirs(sealed_dir(), mode=0o700, exist_ok=True)
    try:
        user = getpass.getuser()
    except Exception:                                    # noqa: BLE001
        user = str(os.getuid())
    line = json.dumps(dict({"t": time.time(), "pid": os.getpid(), "user": user}, **entry), sort_keys=True)
    fd = os.open(_path(LOG_FILE), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "a") as fh:
        fh.write(line + "\n")


def access_log(name: str | None = None) -> list[dict]:
    """Every logged event (creation, access, refusal, denial, rotation), oldest first."""
    try:
        with open(_path(LOG_FILE)) as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
    except FileNotFoundError:
        return []
    return [r for r in rows if name is None or r.get("set") == name]


def _registry() -> dict:
    try:
        with open(_path(REGISTRY_FILE)) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def _write_registry(reg: dict) -> None:
    tmp = _path(REGISTRY_FILE) + f".{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(reg, fh, indent=1, sort_keys=True)
    os.replace(tmp, _path(REGISTRY_FILE))


def _public(name: str, s: Mapping) -> dict:
    return {"set": name, "families": list(s["families"]), "n_per_family": int(s["n_per_family"]),
            "budget": int(s["budget"]), "generation": int(s["generation"]), "accesses": int(s["accesses"]),
            "remaining": max(0, int(s["budget"]) - int(s["accesses"])), "contaminated": bool(s["contaminated"]),
            "created": s["created"], "note": s.get("note", ""), "history": list(s.get("history", []))}


# ====================================================================== who is asking
def _caller() -> str:
    f = sys._getframe(1)
    while f is not None and f.f_globals.get("__name__") == __name__:
        f = f.f_back
    if f is None:
        return "unknown"
    return f"{f.f_globals.get('__name__', '?')}:{f.f_code.co_name}:{f.f_lineno}"


def _forbidden_frame() -> str | None:
    """The first frame on the stack from which a sealed set must not be reachable, or None."""
    from .learner import Learner
    f = sys._getframe(1)
    while f is not None:
        mod = str(f.f_globals.get("__name__", ""))
        if (mod == "crucible" or mod.startswith("crucible.")) and mod not in ALLOWED_CALLERS:
            return f"{mod}:{f.f_code.co_name}"
        me = f.f_locals.get("self")
        if isinstance(me, Learner):
            return f"{type(me).__module__}.{type(me).__name__}.{f.f_code.co_name}"
        f = f.f_back
    return None


def _guard(name: str, what: str) -> None:
    bad = _forbidden_frame()
    if bad is not None:
        _log({"event": "denied", "set": name, "what": what, "caller": bad})
        raise SealedSetError(f"sealed sets are not reachable from learner or world code ({bad})")


# ====================================================================== declaring and rotating sets
def create_set(name: str, families: Sequence[str] | None = None, n_per_family: int = 8, budget: int = 1,
               note: str = "") -> dict:
    """Declare a sealed set: `n_per_family` tasks on each of `families` (default: every one-seat family of the
    test split), which may be handed out `budget` times. Only test-split families are accepted. Returns the
    public description (no seeds)."""
    from .tasks import MECH_FAMILIES, families_by
    _guard(name, "create")
    if not name or any(c in name for c in "/|\\ \t\n"):
        raise ValueError("a sealed set needs a simple name")
    fams = list(families) if families is not None else families_by("test", seats=1)
    for f in fams:
        if f not in MECH_FAMILIES:
            raise ValueError(f"unknown mechanism family {f!r}")
        if MECH_FAMILIES[f].split != "test":
            raise SealedSetError(f"family {f!r} belongs to the {MECH_FAMILIES[f].split} split: sealed sets are "
                                 "restricted to test-split families")
    if not fams or int(n_per_family) < 1 or int(budget) < 1:
        raise ValueError("a sealed set needs families, at least one task per family and a budget of at least 1")
    _salt()
    with _locked():
        reg = _registry()
        if name in reg:
            raise SealedSetError(f"sealed set {name!r} already exists; use rotate() for a new generation")
        reg[name] = {"families": fams, "n_per_family": int(n_per_family), "budget": int(budget), "generation": 1,
                     "accesses": 0, "contaminated": False, "created": time.time(), "note": note, "history": []}
        _write_registry(reg)
        _log({"event": "create", "set": name, "generation": 1, "budget": int(budget), "families": fams,
              "n_per_family": int(n_per_family), "caller": _caller()})
        return _public(name, reg[name])


def describe(name: str | None = None) -> dict:
    """Public status of one set, or of all sets: families, size, budget, generation, accesses, contamination."""
    reg = _registry()
    if name is None:
        return {k: _public(k, v) for k, v in sorted(reg.items())}
    if name not in reg:
        raise SealedSetError(f"no sealed set named {name!r}")
    return _public(name, reg[name])


def rotate(name: str, ack: str, reason: str, budget: int | None = None) -> dict:
    """Retire the current generation of a set and start a new one: new seeds from the salt, counter back to
    zero, contamination cleared. The rotation and its reason are logged and kept in the set's history."""
    _guard(name, "rotate")
    if ack != SEALED_ACK:
        raise SealedSetError("rotating a sealed set requires the exact acknowledgement string")
    if not reason or not str(reason).strip():
        raise SealedSetError("a rotation needs a stated reason")
    with _locked():
        reg = _registry()
        if name not in reg:
            raise SealedSetError(f"no sealed set named {name!r}")
        s = reg[name]
        s["history"].append({"generation": s["generation"], "accesses": s["accesses"], "budget": s["budget"],
                             "contaminated": s["contaminated"], "retired": time.time(), "reason": str(reason)})
        s["generation"] += 1
        s["accesses"], s["contaminated"] = 0, False
        if budget is not None:
            s["budget"] = int(budget)
        _write_registry(reg)
        _log({"event": "rotate", "set": name, "generation": s["generation"], "budget": s["budget"],
              "reason": str(reason), "caller": _caller()})
        return _public(name, s)


# ====================================================================== the tasks
def _cache_file(name: str, generation: int) -> str:
    """Where the built tasks of a generation are kept. The file name carries the task and mechanism-contract
    versions: when a task's definition changes (as it did for `task/2`), sealed tasks are rebuilt like any other."""
    from ..core.canonical import digest
    from ..mechanisms.base import CONTRACT_VERSION
    from .tasks import TASK_VERSION
    v = digest([TASK_VERSION, CONTRACT_VERSION], 4)
    return os.path.join(sealed_dir(), "sealed_tasks", f"{name}.g{int(generation)}.{v}.json")


def _build_one(args) -> dict:
    family, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .tasks import make_task
    try:
        return {"ok": True, "task": make_task(family, seed, validate=True).to_dict()}
    except Exception as e:                               # noqa: BLE001 - reported to the caller, never swallowed
        return {"ok": False, "family": family, "error": f"{type(e).__name__}: {e}"}


def _tasks(name: str, s: Mapping, workers: int = 1) -> list:
    """The tasks of the current generation, built on first need and kept (mode 600) next to the salt."""
    from .tasks import Task
    path = _cache_file(name, s["generation"])
    if os.path.exists(path):
        with open(path) as fh:
            return [Task.from_dict(d) for d in json.load(fh)]
    salt = _salt()
    n = int(s["n_per_family"])
    out: list[dict] = []
    for fam in s["families"]:
        got: list[dict] = []
        k = 0
        while len(got) < n and k < MAX_CANDIDATES * n:
            want = [(fam, _seed(salt, name, s["generation"], fam, j)) for j in range(k, k + n - len(got))]
            k += len(want)
            if workers > 1 and len(want) > 1:
                import multiprocessing as mp
                with mp.get_context("spawn").Pool(min(int(workers), len(want))) as pool:
                    res = pool.map(_build_one, want, chunksize=1)
            else:
                res = [_build_one(w) for w in want]
            got += [r["task"] for r in res if r["ok"]]
        if len(got) < n:
            raise SealedSetError(f"family {fam!r}: only {len(got)} of {n} sealed tasks could be validated")
        out += got[:n]
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path + ".tmp", os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(out, fh)
    os.replace(path + ".tmp", path)
    return [Task.from_dict(d) for d in out]


def prepare(name: str, workers: int = 1) -> dict:
    """Build (validate) the tasks of the current generation ahead of time. Not an access: nothing is handed out;
    returns only the public description and the number of tasks ready."""
    _guard(name, "prepare")
    reg = _registry()
    if name not in reg:
        raise SealedSetError(f"no sealed set named {name!r}")
    n = len(_tasks(name, reg[name], workers))
    _log({"event": "prepare", "set": name, "generation": reg[name]["generation"], "n_tasks": n, "caller": _caller()})
    return dict(_public(name, reg[name]), n_tasks=n)


class SealedSet:
    """What one access hands out: the tasks, the tag that results must carry, and the aliases under which the
    tasks may be named in anything that is written down."""

    def __init__(self, tasks: list, tag: dict):
        self.tasks = list(tasks)
        self.tag = dict(tag)
        count: dict[str, int] = {}
        self._alias: dict[str, str] = {}
        for t in self.tasks:
            k = count.get(t.family, 0)
            count[t.family] = k + 1
            self._alias[t.id] = f"sealed/{tag['set']}/g{tag['generation']}/{t.family}/{k}"

    def alias(self, task_id: str) -> str:
        return self._alias[task_id]

    def redact(self, obj: Any) -> Any:
        """A deep copy of a result with every task id replaced by its alias and every task seed removed."""
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in ("seed", "mech_seed", "world_seed") and isinstance(v, int) and not isinstance(v, bool) \
                        and "task" in obj:
                    out[k] = None
                else:
                    out[k] = self.redact(v)
            return out
        if isinstance(obj, (list, tuple)):
            return [self.redact(v) for v in obj]
        if isinstance(obj, str):
            if obj in self._alias:
                return self._alias[obj]
            for tid, al in self._alias.items():
                if tid in obj:
                    obj = obj.replace(tid, al)
            return obj
        return obj

    def __len__(self) -> int:
        return len(self.tasks)

    def __repr__(self) -> str:
        return (f"SealedSet({self.tag['set']!r}, generation {self.tag['generation']}, {len(self.tasks)} tasks, "
                f"access {self.tag['access']} of {self.tag['budget']})")


def open_set(name: str, ack: str, purpose: str, caller: str | None = None, workers: int = 1) -> SealedSet:
    """Hand out a sealed set for one evaluation session. Requires the exact acknowledgement string and a stated
    purpose. Logged before anything is returned; refused (and the set marked contaminated) beyond the budget."""
    _guard(name, "open")
    who = caller or _caller()
    if ack != SEALED_ACK:
        _log({"event": "refused", "set": name, "why": "acknowledgement", "caller": who, "purpose": str(purpose)})
        raise SealedSetError("sealed test access requires the exact acknowledgement string")
    if not purpose or not str(purpose).strip():
        _log({"event": "refused", "set": name, "why": "no purpose", "caller": who})
        raise SealedSetError("sealed test access requires a stated purpose")
    with _locked():
        reg = _registry()
        if name not in reg:
            _log({"event": "refused", "set": name, "why": "unknown set", "caller": who, "purpose": str(purpose)})
            raise SealedSetError(f"no sealed set named {name!r}")
        s = reg[name]
        if s["contaminated"] or s["accesses"] >= s["budget"]:
            first = not s["contaminated"]
            s["contaminated"] = True
            _write_registry(reg)
            _log({"event": "refused", "set": name, "generation": s["generation"], "count": s["accesses"],
                  "budget": s["budget"], "why": "budget exhausted" if first else "contaminated",
                  "marked_contaminated": first, "caller": who, "purpose": str(purpose)})
            raise SealedSetError(
                f"sealed set {name!r} (generation {s['generation']}) has been handed out {s['accesses']} times, its "
                f"declared budget is {s['budget']}: it is now marked contaminated. Create a new generation with "
                "rotate() and report results of this one as such.")
        s["accesses"] += 1
        _write_registry(reg)
        tag = {"sealed": True, "set": name, "generation": int(s["generation"]), "access": int(s["accesses"]),
               "budget": int(s["budget"]), "purpose": str(purpose), "caller": who, "t": time.time()}
        _log({"event": "access", "set": name, "generation": tag["generation"], "count": tag["access"],
              "budget": tag["budget"], "purpose": tag["purpose"], "caller": who})
    tasks = _tasks(name, s, workers)
    from .tasks import TASK_VERSION
    tag["n_tasks"] = len(tasks)
    tag["task_version"] = TASK_VERSION
    tag["families"] = list(s["families"])
    return SealedSet(tasks, tag)


def evaluate(subjects: Mapping[str, Sequence[Any]], name: str, ack: str, purpose: str, *,
             reference: str | None = None, caller: str | None = None, workers: int = 1, **kw) -> dict:
    """One sealed evaluation session: open the set once and evaluate every learner in `subjects`
    (`{name: [learner per seed]}`, as they are: load their trained checkpoints first) frozen on its tasks, paired
    and corrected (`protocols.evaluate_many`). The result is tagged as sealed and carries no task seeds."""
    from . import protocols as P
    ss = open_set(name, ack, purpose, caller=caller or _caller(), workers=workers)
    res = P.evaluate_many(subjects, ss.tasks, reference=reference, **kw)
    res = ss.redact(res)
    res["protocol"] = "sealed"
    res["sealed"] = dict(ss.tag)
    for per in res["per_learner"].values():
        for r in per.get("records", []):
            r["sealed"] = True
    return res
