"""cam_live.eval.openended — open-ended objective families with independent verifiers (Protocol v3, docs/S/PROTOCOLS.md).

The environment defines primitives and authorization boundaries only: the general-purpose computer environment
(``harness.env.ComputerEnvironment``) with the internal, code (sandbox: write / run programs) and api (HTTP)
executors, authorized by the real ``harness.auth.PolicyService`` in staging mode. No solution, demonstration,
strategy template or task-specific decision rule is given; the objective is a natural-language statement.

Families
``SW``    building software: implement a function described in words in ``solution.py`` of the sandbox
          workspace. The independent :class:`HiddenTestVerifier` runs hidden tests (inputs drawn by the verifier from
          its own seed, expected outputs from a hidden reference the agent never sees) in a separate process with a
          time limit; value = pass fraction; achieved at >= ``SW_PASS``.
``MECH``  discovering a mechanism: an owned local HTTP service answers ``GET /measure?x=..&y=..`` with the output
          of a hidden function; write ``model.py`` with ``predict(x, y)``; :class:`HeldOutVerifier` scores it on
          held-out points the service never answered (R^2, achieved at >= ``MECH_R2``).

Both verifiers sign their verdicts and measurements (HMAC, key held by the verifier), so the agent can check
signatures but not produce them; ``measure`` answers the agent's questions about the task's success metric.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from ..contracts import (GoalStatus, Limits, OutcomeObservation, OutcomeQuery, OutcomeStatus, Quantity, TaskSpec,
                         VerifiedOutcome)

SW_PASS = 0.9
MECH_R2 = 0.9
SOLUTION, MODEL = "solution.py", "model.py"


# ================================================================================================ SW problems
def _rle(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        n = j - i
        out.append(f"{n}{s[i]}" if n > 1 else s[i])
        i = j
    return "".join(out)


def _rpn(expr: str) -> int:
    st: List[int] = []
    for t in expr.split():
        if t in ("+", "-", "*", "max"):
            b, a = st.pop(), st.pop()
            st.append(a + b if t == "+" else a - b if t == "-" else a * b if t == "*" else max(a, b))
        else:
            st.append(int(t))
    return st[-1]


def _merge(iv: List[List[int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for a, b in sorted([list(x) for x in iv]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _topk(text: str, k: int) -> List[str]:
    import re
    words = re.findall(r"[a-z]+", text.lower())
    cnt: Dict[str, int] = {}
    for w in words:
        cnt[w] = cnt.get(w, 0) + 1
    return [w for w, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def _longest_unique(s: str) -> int:
    best, start, seen = 0, 0, {}
    for i, ch in enumerate(s):
        if ch in seen and seen[ch] >= start:
            start = seen[ch] + 1
        seen[ch] = i
        best = max(best, i - start + 1)
    return best


def _to_base(n: int, b: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    neg, n, out = n < 0, abs(n), []
    while n:
        out.append(digits[n % b])
        n //= b
    return ("-" if neg else "") + "".join(reversed(out))


def _gen_rle(r: random.Random) -> Tuple:
    return ("".join(r.choice("aab") * r.randint(1, 4) for _ in range(r.randint(0, 6))),)


def _gen_rpn(r: random.Random) -> Tuple:
    toks, depth = [str(r.randint(-9, 9))], 1
    for _ in range(r.randint(0, 5)):
        toks.append(str(r.randint(-9, 9)))
        toks.append(r.choice(["+", "-", "*", "max"]))
    return (" ".join(toks),)


def _gen_merge(r: random.Random) -> Tuple:
    iv = []
    for _ in range(r.randint(0, 6)):
        a = r.randint(0, 20)
        iv.append([a, a + r.randint(0, 6)])
    return (iv,)


def _gen_topk(r: random.Random) -> Tuple:
    words = ["alpha", "beta", "Gamma", "delta", "beta", "ALPHA", "eps", "zeta"]
    text = " ".join(r.choice(words) + r.choice(["", ",", ".", "!"]) for _ in range(r.randint(1, 15)))
    return (text, r.randint(1, 4))


def _gen_unique(r: random.Random) -> Tuple:
    return ("".join(r.choice("abcdea") for _ in range(r.randint(0, 14))),)


def _gen_base(r: random.Random) -> Tuple:
    return (r.randint(-5000, 5000), r.randint(2, 36))


#: class -> (statement, reference, input generator). Statements specify behaviour only.
SW_PROBLEMS: Dict[str, Tuple[str, Callable, Callable]] = {
    "rle": ("Write a Python file solution.py in your workspace that defines solve(s: str) -> str. It must replace "
            "every maximal run of n > 1 equal characters c by the decimal number n followed by c; characters that "
            "are not repeated stay as they are. Example of the format: a run 'xxx' becomes '3x'.", _rle, _gen_rle),
    "rpn": ("Write a Python file solution.py in your workspace that defines solve(expr: str) -> int, evaluating an "
            "expression in reverse Polish notation: tokens separated by single spaces, integer operands, and the "
            "binary operators +, -, * and max (max returns the larger of its two operands).", _rpn, _gen_rpn),
    "merge": ("Write a Python file solution.py in your workspace that defines solve(intervals: list) -> list: the "
              "input is a list of [start, end] integer pairs (start <= end); return the union as a list of "
              "non-overlapping [start, end] pairs sorted by start, merging pairs that overlap or touch at an "
              "endpoint.", _merge, _gen_merge),
    "topk": ("Write a Python file solution.py in your workspace that defines solve(text: str, k: int) -> list: "
             "words are maximal runs of ASCII letters, compared case-insensitively and returned in lower case; "
             "return the k most frequent words, most frequent first, ties broken alphabetically.", _topk, _gen_topk),
    "unique": ("Write a Python file solution.py in your workspace that defines solve(s: str) -> int: the length "
               "of the longest contiguous substring of s in which no character occurs twice.", _longest_unique,
               _gen_unique),
    "base": ("Write a Python file solution.py in your workspace that defines solve(n: int, b: int) -> str: the "
             "representation of the integer n in base b (2 <= b <= 36) with digits 0-9 then a-z, a leading '-' for "
             "negative numbers and '0' for zero.", _to_base, _gen_base),
}
SW_TRAIN = ("rle", "rpn", "merge", "topk")
SW_HELDOUT = ("unique", "base")

_RUNNER = r"""
import json, sys, importlib.util, signal
spec = importlib.util.spec_from_file_location("solution", sys.argv[1])
cases = json.load(open(sys.argv[2]))
out = []
try:
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
except BaseException as e:
    print(json.dumps({"import_error": repr(e)[:300]})); sys.exit(0)
def _alarm(*a): raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
for args in cases:
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        r = m.solve(*args) if hasattr(m, "solve") else m.predict(*args)
        signal.setitimer(signal.ITIMER_REAL, 0)
        out.append(["ok", r])
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        out.append(["error", repr(e)[:120]])
print(json.dumps({"results": out}, default=str))
"""


def run_hidden(path: str, cases: Sequence[Tuple], timeout_s: float = 30.0) -> Dict[str, Any]:
    """Run the agent's file on ``cases`` in a separate, isolated Python process (``-I``, temp cwd, no env)."""
    if not os.path.exists(path):
        return {"missing": True, "results": []}
    work = tempfile.mkdtemp(prefix="camlive_verifier_")
    try:
        sol = os.path.join(work, "candidate.py")
        shutil.copy2(path, sol)
        with open(os.path.join(work, "cases.json"), "w") as f:
            json.dump([list(c) for c in cases], f)
        with open(os.path.join(work, "runner.py"), "w") as f:
            f.write(_RUNNER)
        p = subprocess.run([sys.executable, "-I", os.path.join(work, "runner.py"), sol,
                            os.path.join(work, "cases.json")], cwd=work, capture_output=True, text=True,
                           timeout=timeout_s, env={"PATH": "/usr/bin:/bin"})
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"crash": p.stderr[-300:], "results": []}
    except subprocess.TimeoutExpired:
        return {"timeout": True, "results": []}
    finally:
        shutil.rmtree(work, ignore_errors=True)


class _Signer:
    def __init__(self, secret: bytes):
        self._secret = secret

    def _sign(self, o: Any) -> str:
        d = o.to_dict()
        d.pop("signature", None)
        return hmac.new(self._secret, json.dumps(d, sort_keys=True, default=str).encode(), hashlib.sha256).hexdigest()

    def check_signature(self, outcome: VerifiedOutcome) -> bool:
        return (isinstance(outcome, VerifiedOutcome) and bool(outcome.signature)
                and hmac.compare_digest(outcome.signature, self._sign(outcome)))

    def check_measurement(self, obs: OutcomeObservation) -> bool:
        return (isinstance(obs, OutcomeObservation) and bool(obs.signature) and obs.verified
                and hmac.compare_digest(obs.signature, self._sign(obs)))


class HiddenTestVerifier(_Signer):
    """Independent verifier of an SW objective (``contracts.OutcomeVerifier``)."""

    def __init__(self, workspace: str, problem: str, seed: int, clock: Any, *, n_tests: int = 60,
                 secret: bytes = b"sw-verifier-key"):
        super().__init__(secret)
        self.workspace, self.problem, self.clock = workspace, problem, clock
        _, ref, gen = SW_PROBLEMS[problem]
        r = random.Random(f"{problem}:{seed}:hidden")
        self.cases = [gen(r) for _ in range(n_tests)]
        self.expected = [json.loads(json.dumps(ref(*c))) for c in self.cases]
        self.history: List[Dict[str, Any]] = []

    def score(self) -> Dict[str, Any]:
        res = run_hidden(os.path.join(self.workspace, SOLUTION), self.cases)
        got = res.get("results", [])
        ok = sum(1 for g, e in zip(got, self.expected) if g[0] == "ok" and json.loads(json.dumps(g[1])) == e)
        frac = ok / len(self.expected)
        info = {"pass_fraction": frac, "n": len(self.expected), "missing": bool(res.get("missing")),
                "import_error": res.get("import_error"), "errors": sum(1 for g in got if g[0] == "error")}
        self.history.append({"t": float(self.clock.now()), **info})
        return info

    def verify(self, task: TaskSpec) -> VerifiedOutcome:
        s = self.score()
        target = float(task.success.get("target", SW_PASS))
        now = float(self.clock.now())
        status = GoalStatus.ACHIEVED if s["pass_fraction"] >= target else GoalStatus.PENDING
        o = VerifiedOutcome(objective_id=task.objective_id, status=status, value=float(s["pass_fraction"]),
                            target=target, evaluated_at=now, notes=f"hidden tests: {s['n']}",
                            secondary=[Quantity("sw.errors", float(s["errors"]))])
        o.signature = self._sign(o)
        return o

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        out = []
        s = None
        for q in queries:
            if q.metric != "sw.pass_fraction":
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.UNOBSERVABLE, source="verifier:sw"))
                continue
            s = s or self.score()
            o = OutcomeObservation(q.query_id, OutcomeStatus.OBSERVED, float(s["pass_fraction"]),
                                   observed_at=float(self.clock.now()), verified=True, source="verifier:sw",
                                   dedup_key=f"sw:{q.query_id}")
            o.signature = self._sign(o)
            out.append(o)
        return out


def sw_task(problem: str, objective_id: str, *, max_decisions: int, duration_s: float = 6 * 3600.0) -> TaskSpec:
    return TaskSpec(objective_id=objective_id, objective=SW_PROBLEMS[problem][0] + " It is checked by independent "
                    "hidden tests.", success={"verifier": "hidden_tests", "metric": "sw.pass_fraction",
                                              "target": SW_PASS},
                    family=f"openended/sw/{problem}", interfaces=["code", "internal"],
                    resources={"owned": [], "preauthorized": [], "read": []},
                    limits=Limits(duration_s=duration_s, money=1.0, compute_s=0.0,
                                  max_external_actions=10 * max_decisions, max_decisions=int(max_decisions)),
                    data_permissions={"aggregate_metrics": True}, mode="staging")


# ================================================================================================ MECH
def mechanism(seed: int) -> Callable[[float, float], float]:
    """A hidden smooth mechanism of two inputs; the family differs per seed (no form is ever disclosed)."""
    r = random.Random(f"mech:{seed}")
    kind = r.choice(["poly", "trig", "ratio"])
    a, b, c = (round(r.uniform(0.5, 3.0), 2) for _ in range(3))
    if kind == "poly":
        return lambda x, y: a * x * x - b * x * y + c * y
    if kind == "trig":
        return lambda x, y: a * math.sin(b * x) + c * y
    return lambda x, y: a * x / (1.0 + b * y * y) + c


class MechanismService:
    """Owned local HTTP service: ``GET /measure?x=&y=`` (x, y in [-3, 3]) returns the hidden mechanism's value."""

    def __init__(self, seed: int):
        self.f = mechanism(seed)
        self.queries: List[Tuple[float, float]] = []
        svc = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                u = urlparse(self.path)
                if u.path != "/measure":
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'{"error": "unknown path; use /measure?x=&y="}')
                    return
                q = parse_qs(u.query)
                try:
                    x, y = float(q["x"][0]), float(q["y"][0])
                    if not (-3 <= x <= 3 and -3 <= y <= 3):
                        raise ValueError("x and y must be in [-3, 3]")
                except (KeyError, ValueError) as e:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
                    return
                svc.queries.append((x, y))
                body = json.dumps({"x": x, "y": y, "value": round(svc.f(x, y), 6)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a: Any) -> None:
                pass
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class HeldOutVerifier(_Signer):
    """Independent verifier of a MECH objective: R^2 of ``model.predict`` on held-out points."""

    def __init__(self, workspace: str, service: MechanismService, seed: int, clock: Any, *, n: int = 80,
                 secret: bytes = b"mech-verifier-key"):
        super().__init__(secret)
        self.workspace, self.svc, self.clock = workspace, service, clock
        r = random.Random(f"mech:{seed}:heldout")
        self.cases = [(round(r.uniform(-3, 3), 3), round(r.uniform(-3, 3), 3)) for _ in range(n)]
        self.truth = [service.f(x, y) for x, y in self.cases]
        self.history: List[Dict[str, Any]] = []

    def score(self) -> Dict[str, Any]:
        res = run_hidden(os.path.join(self.workspace, MODEL), self.cases)
        got = res.get("results", [])
        preds = []
        for g in got:
            try:
                preds.append(float(g[1]) if g[0] == "ok" else float("nan"))
            except (TypeError, ValueError):
                preds.append(float("nan"))
        mu = sum(self.truth) / len(self.truth)
        sst = sum((t - mu) ** 2 for t in self.truth)
        if len(preds) == len(self.truth) and all(math.isfinite(p) for p in preds):
            r2 = 1.0 - sum((p - t) ** 2 for p, t in zip(preds, self.truth)) / sst
        else:
            r2 = float("-inf") if preds else None
        val = 0.0 if r2 is None or not math.isfinite(r2) else max(0.0, min(1.0, r2))
        info = {"r2": r2 if (r2 is not None and math.isfinite(r2)) else None, "value": val,
                "missing": bool(res.get("missing")), "n_service_queries": len(self.svc.queries)}
        self.history.append({"t": float(self.clock.now()), **info})
        return info

    def verify(self, task: TaskSpec) -> VerifiedOutcome:
        s = self.score()
        target = float(task.success.get("target", MECH_R2))
        status = GoalStatus.ACHIEVED if s["value"] >= target else GoalStatus.PENDING
        o = VerifiedOutcome(objective_id=task.objective_id, status=status, value=float(s["value"]), target=target,
                            evaluated_at=float(self.clock.now()), notes="held-out R^2 (clipped to [0, 1])")
        o.signature = self._sign(o)
        return o

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        out, s = [], None
        for q in queries:
            if q.metric != "mech.r2":
                out.append(OutcomeObservation(q.query_id, OutcomeStatus.UNOBSERVABLE, source="verifier:mech"))
                continue
            s = s or self.score()
            o = OutcomeObservation(q.query_id, OutcomeStatus.OBSERVED, float(s["value"]),
                                   observed_at=float(self.clock.now()), verified=True, source="verifier:mech",
                                   dedup_key=f"mech:{q.query_id}")
            o.signature = self._sign(o)
            out.append(o)
        return out


def mech_task(service: MechanismService, objective_id: str, *, max_decisions: int,
              duration_s: float = 6 * 3600.0) -> TaskSpec:
    host = service.url.replace("http://", "")
    return TaskSpec(objective_id=objective_id,
                    objective=(f"An unknown mechanism maps two numbers x and y in [-3, 3] to a value; the service "
                               f"{service.url}/measure?x=..&y=.. returns the value for one pair. Write a Python file "
                               f"model.py in your workspace that defines predict(x: float, y: float) -> float. It is "
                               f"scored on pairs the service was never asked about (R^2 >= {MECH_R2})."),
                    success={"verifier": "held_out", "metric": "mech.r2", "target": MECH_R2},
                    family="openended/mech", interfaces=["code", "api", "internal"],
                    resources={"owned": [host, service.url], "preauthorized": [host], "read": [host]},
                    limits=Limits(duration_s=duration_s, money=1.0, compute_s=0.0,
                                  max_external_actions=10 * max_decisions, max_decisions=int(max_decisions)),
                    data_permissions={"aggregate_metrics": True}, mode="staging")


# ================================================================================================ GOAL-FREE (X5)
GOAL_FREE_TEXT = ("No objective is assigned for this period and nothing is checked. The code sandbox and its tools "
                  "are available.")


def goal_free_task(objective_id: str, *, max_decisions: int, duration_s: float = 6 * 3600.0) -> TaskSpec:
    """Protocol v3.4 X5 / spec 11.4: a period with no success criterion on the same computer (``success`` empty:
    the runtime verifies nothing and the agent's goal-free mode chooses its own learning goals)."""
    return TaskSpec(objective_id=objective_id, objective=GOAL_FREE_TEXT, success={}, family="openended/goal_free",
                    interfaces=["code", "internal"], resources={"owned": [], "preauthorized": [], "read": []},
                    limits=Limits(duration_s=duration_s, money=1.0, compute_s=0.0,
                                  max_external_actions=10 * max_decisions, max_decisions=int(max_decisions)),
                    data_permissions={"aggregate_metrics": True}, mode="staging")


class NoCriterionVerifier(_Signer):
    """Verifier of a goal-free period: there is nothing to check, so every verdict is ``pending`` with value 0
    (signed like the others) and every measurement is unobservable."""

    def __init__(self, clock: Any, *, secret: bytes = b"goal-free-verifier-key"):
        super().__init__(secret)
        self.clock = clock
        self.history: List[Dict[str, Any]] = []

    def verify(self, task: TaskSpec) -> VerifiedOutcome:
        o = VerifiedOutcome(objective_id=task.objective_id, status=GoalStatus.PENDING, value=0.0, target=0.0,
                            evaluated_at=float(self.clock.now()), notes="goal-free period: no success criterion")
        o.signature = self._sign(o)
        return o

    def measure(self, queries: Sequence[OutcomeQuery]) -> List[OutcomeObservation]:
        return [OutcomeObservation(q.query_id, OutcomeStatus.UNOBSERVABLE, source="verifier:goal_free")
                for q in queries]


# ================================================================================================ environment
#: operation prefixes of executors registered by the harness that none of these families authorizes (no Discord
#: or publishing resources exist here): hidden from discovery so that they are not offered as actions
UNUSED_PREFIXES = ("discord.", "publish.")


class _ToolFilter:
    """The computer environment with the tools of unauthorized external services hidden from ``discover_tools``."""

    def __init__(self, env: Any):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def discover_tools(self) -> List[Any]:
        return [t for t in self._env.discover_tools() if not t.name.startswith(UNUSED_PREFIXES)]


class OpenEndedComputer:
    """One persistent computer (sandbox workspace, HTTP client, authorization service) across the objectives
    given to it, like the persistent computer of spec 12A.1. The workspace is emptied between objectives (the
    previous objective's files are not a cached solution for the next)."""

    def __init__(self, root: str, clock: Any, *, modalities: Sequence[str] = ("internal", "code", "api")):
        self.root, self.clock = os.path.abspath(root), clock
        self.modalities = tuple(modalities)
        os.makedirs(self.root, exist_ok=True)
        self.workspace = os.path.join(self.root, "sandbox")
        os.makedirs(self.workspace, exist_ok=True)
        self.env: Any = None
        self.policy: Any = None
        self.service: Optional[MechanismService] = None

    def _clear_workspace(self) -> None:
        for name in os.listdir(self.workspace):
            p = os.path.join(self.workspace, name)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)

    def open(self, family: str, problem: str, seed: int, objective_id: str, *, max_decisions: int
             ) -> Tuple[Any, Any, Any, TaskSpec]:
        from ..harness.auth import PolicyService
        from ..harness.env import ComputerEnvironment
        from ..harness.factories import all_executors
        self._clear_workspace()
        if self.service is not None:
            self.service.close()
            self.service = None
        if family == "SW":
            task = sw_task(problem, objective_id, max_decisions=max_decisions)
            verifier: Any = HiddenTestVerifier(self.workspace, problem, seed, self.clock)
        elif family == "MECH":
            self.service = MechanismService(seed)
            task = mech_task(self.service, objective_id, max_decisions=max_decisions)
            verifier = HeldOutVerifier(self.workspace, self.service, seed, self.clock)
        elif family == "GOALFREE":
            task = goal_free_task(objective_id, max_decisions=max_decisions)
            verifier = NoCriterionVerifier(self.clock)
        else:
            raise KeyError(family)
        if self.policy is None:
            self.policy = PolicyService(os.path.join(self.root, "auth_state"), clock=self.clock)
        self.policy.register_task(task, "operator")
        if self.env is None:
            exec_dir = os.path.join(self.root, "exec")
            self.env = ComputerEnvironment(exec_dir, self.policy, clock=self.clock,
                                           secrets=self.policy.secrets_handle(), provenance="staging")
            excl = tuple(m for m in ("internal", "structured_browser", "code", "visual", "api", "skill")
                         if m not in self.modalities)
            for ex in all_executors(exec_dir, self.clock, env=self.env, policy=self.policy,
                                    secrets=self.policy.secrets_handle(), task=task,
                                    browser_enabled="structured_browser" in self.modalities,
                                    sandbox_workspace=self.workspace, exclude=excl):
                self.env.register(ex)
        return _ToolFilter(self.env), self.policy, verifier, task

    def close(self) -> None:
        if self.service is not None:
            self.service.close()
        if self.env is not None:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                pass
