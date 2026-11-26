"""A small, safe expression language for laws of nature and for scientists' models (world side).

Hidden laws in the `sci` module and the models that participants publish are written in the same language: Python
expression syntax over named variables, numeric constants, + - * / **, unary minus, comparisons (which give 1.0 or
0.0), `a if c else b`, and the functions exp, log, sqrt, abs, min, max, tanh, sigmoid, step, clip and pow. Nothing
else is accepted (no attribute access, no names other than the declared variables, no calls of anything else), so an
expression can be evaluated inside the kernel without running participant code.

`compile_expr(text, variables)` validates and returns an `Expr`; `Expr(env)` evaluates it on scalars or NumPy arrays
(the same expression also runs as plain Python with `PRELUDE` defining the functions, which is how a participant's
service would embed a model).

`LAW_FAMILIES` are seeded generators of hidden laws, each a structurally different functional form:

* `sat_lin`       A * x / (K + x) * (b + c * z)                 saturating response times a linear one
* `thresh_quad`   H * step(z - theta) + a + b * x - c * x ** 2  activation threshold plus diminishing returns
* `peak`          A * exp(-((x - mu) / w) ** 2) * (1 + g * z) + b  an optimum (non-monotonic) with a modifier
* `power_exp`     A * x ** p * exp(-k / z)                       power law times an activation (Arrhenius-like) term

A generator returns the law text (hidden), the variables it actually uses, and a reasonable "practice" point.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

MAX_LEN = 2000
MAX_NODES = 300
FUNCS = ("exp", "log", "sqrt", "abs", "min", "max", "tanh", "sigmoid", "step", "clip", "pow")
BOUND = 1e12

PRELUDE = ("import math\n"
           "def exp(x):\n    return math.exp(min(700.0, x))\n"
           "def log(x):\n    return math.log(x) if x > 0 else -1e12\n"
           "def sqrt(x):\n    return math.sqrt(x) if x > 0 else 0.0\n"
           "def tanh(x):\n    return math.tanh(x)\n"
           "def sigmoid(x):\n    return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, x))))\n"
           "def step(x):\n    return 1.0 if x > 0 else 0.0\n"
           "def clip(x, lo, hi):\n    return lo if x < lo else hi if x > hi else x\n"
           "def pow(a, b):\n    return math.pow(a, b) if a > 0 else 0.0\n")


class LawError(ValueError):
    pass


def _f_exp(x):
    return np.exp(np.minimum(700.0, x))


def _f_log(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), -BOUND)


def _f_sqrt(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.maximum(x, 0.0))


def _f_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def _f_step(x):
    return np.where(np.asarray(x, dtype=float) > 0, 1.0, 0.0)


def _f_pow(a, b):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, np.power(np.where(a > 0, a, 1.0), b), 0.0)


_IMPL: dict[str, tuple[Callable, tuple[int, ...]]] = {
    "exp": (_f_exp, (1,)), "log": (_f_log, (1,)), "sqrt": (_f_sqrt, (1,)), "abs": (np.abs, (1,)),
    "min": (np.minimum, (2,)), "max": (np.maximum, (2,)), "tanh": (np.tanh, (1,)), "sigmoid": (_f_sigmoid, (1,)),
    "step": (_f_step, (1,)), "clip": (np.clip, (3,)), "pow": (_f_pow, (2,)),
}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: None, ast.Pow: None}
_CMP = {ast.Gt: np.greater, ast.Lt: np.less, ast.GtE: np.greater_equal, ast.LtE: np.less_equal}


def _div(a, b):
    b = np.asarray(b, dtype=float)
    safe = np.where(np.abs(b) < 1e-12, np.where(b < 0, -1e-12, 1e-12), b)
    return np.divide(a, safe)


def _powop(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    neg = a < 0
    # a negative base is allowed only with an integer exponent (x ** 2); otherwise the value is undefined -> 0
    intb = np.equal(np.mod(b, 1.0), 0.0)
    base = np.where(neg & ~intb, 0.0, a)
    with np.errstate(all="ignore"):
        out = np.power(np.where(np.abs(base) > 1e6, np.sign(base) * 1e6, base), np.clip(b, -50.0, 50.0))
    return out


@dataclass(frozen=True)
class Expr:
    text: str
    variables: tuple[str, ...]        # the declared variables (the expression may use a subset)
    used: tuple[str, ...]             # the variables it actually uses
    _fn: Callable[[dict], Any]
    n_nodes: int
    n_constants: int

    def __call__(self, env: dict) -> np.ndarray:
        with np.errstate(all="ignore"):
            v = np.asarray(self._fn(env), dtype=float)
        v = np.where(np.isfinite(v), v, np.nan)
        return np.clip(v, -BOUND, BOUND)

    def at(self, points: Sequence[dict]) -> np.ndarray:
        """Evaluate at a list of {variable: value} points (missing variables are 0)."""
        env = {k: np.array([float(p.get(k, 0.0)) for p in points], dtype=float) for k in self.variables}
        out = self(env)
        return np.broadcast_to(out, (len(points),)).astype(float) if np.ndim(out) == 0 else out


def compile_expr(text: str, variables: Sequence[str]) -> Expr:
    """Validate `text` and compile it into an evaluator over `variables`. Raises LawError with a reason."""
    if not isinstance(text, str) or not text.strip():
        raise LawError("empty expression")
    if len(text) > MAX_LEN:
        raise LawError(f"expression longer than {MAX_LEN} characters")
    variables = tuple(variables)
    for v in variables:
        if v in FUNCS or not v.isidentifier():
            raise LawError(f"bad variable name {v!r}")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise LawError(f"syntax error: {e.msg}") from None
    count = {"nodes": 0, "consts": 0}
    used: set[str] = set()

    def build(node) -> Callable[[dict], Any]:
        count["nodes"] += 1
        if count["nodes"] > MAX_NODES:
            raise LawError(f"expression has more than {MAX_NODES} nodes")
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise LawError("only numeric constants are allowed")
            if not math.isfinite(float(node.value)):
                raise LawError("constants must be finite")
            count["consts"] += 1
            c = float(node.value)
            return lambda env, _c=c: _c
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise LawError(f"unknown name {node.id!r} (variables: {', '.join(variables)})")
            used.add(node.id)
            return lambda env, _k=node.id: env[_k]
        if isinstance(node, ast.UnaryOp):
            f = build(node.operand)
            if isinstance(node.op, ast.USub):
                return lambda env, _f=f: np.negative(_f(env))
            if isinstance(node.op, ast.UAdd):
                return f
            raise LawError("only unary + and - are allowed")
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN:
                raise LawError("only + - * / ** are allowed")
            a, b = build(node.left), build(node.right)
            if isinstance(node.op, ast.Div):
                return lambda env, _a=a, _b=b: _div(_a(env), _b(env))
            if isinstance(node.op, ast.Pow):
                return lambda env, _a=a, _b=b: _powop(_a(env), _b(env))
            op = _BIN[type(node.op)]
            return lambda env, _a=a, _b=b, _op=op: _op(_a(env), _b(env))
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in _CMP:
                raise LawError("only single comparisons with < > <= >= are allowed")
            a, b = build(node.left), build(node.comparators[0])
            op = _CMP[type(node.ops[0])]
            return lambda env, _a=a, _b=b, _op=op: np.where(_op(_a(env), _b(env)), 1.0, 0.0)
        if isinstance(node, ast.IfExp):
            c, a, b = build(node.test), build(node.body), build(node.orelse)
            return lambda env, _c=c, _a=a, _b=b: np.where(np.asarray(_c(env)) != 0, _a(env), _b(env))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _IMPL or node.keywords:
                raise LawError(f"only the functions {', '.join(FUNCS)} may be called")
            fn, arities = _IMPL[node.func.id]
            if len(node.args) not in arities:
                raise LawError(f"{node.func.id} takes {arities[0]} argument(s)")
            args = [build(a) for a in node.args]
            return lambda env, _fn=fn, _args=args: _fn(*[g(env) for g in _args])
        raise LawError(f"{type(node).__name__} is not allowed in an expression")

    fn = build(tree)
    return Expr(text.strip(), variables, tuple(v for v in variables if v in used), fn, count["nodes"], count["consts"])


def fmt(x: float) -> str:
    """A constant as text that round-trips (used when generating law and model texts)."""
    return repr(float(round(x, 6)))


# ====================================================================== hidden-law families (world generation)
def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def law_sat_lin(rng, xs: Sequence[str]) -> dict:
    x, z = xs[0], xs[1]
    A, K, b, c = _u(rng, 8, 14), _u(rng, 1.5, 4.0), _u(rng, 0.6, 1.2), _u(rng, 0.15, 0.4)
    return {"law": f"{fmt(A)} * {x} / ({fmt(K)} + {x}) * ({fmt(b)} + {fmt(c)} * {z})", "uses": [x, z],
            "form": "sat_lin"}


def law_thresh_quad(rng, xs: Sequence[str]) -> dict:
    x, z = xs[0], xs[1]
    H, th = _u(rng, 3.0, 6.0), _u(rng, 1.8, 3.6)
    a, b, c = _u(rng, 2.0, 4.0), _u(rng, 1.2, 2.2), _u(rng, 0.06, 0.12)
    return {"law": f"{fmt(H)} * step({z} - {fmt(th)}) + {fmt(a)} + {fmt(b)} * {x} - {fmt(c)} * {x} ** 2",
            "uses": [x, z], "form": "thresh_quad"}


def law_peak(rng, xs: Sequence[str]) -> dict:
    x, z = xs[0], xs[1]
    A, mu, w, g, b = _u(rng, 8, 14), _u(rng, 3.0, 5.0), _u(rng, 1.6, 2.6), _u(rng, 0.1, 0.25), _u(rng, 1.0, 3.0)
    return {"law": f"{fmt(A)} * exp(-(({x} - {fmt(mu)}) / {fmt(w)}) ** 2) * (1 + {fmt(g)} * {z}) + {fmt(b)}",
            "uses": [x, z], "form": "peak"}


def law_power_exp(rng, xs: Sequence[str]) -> dict:
    x, z = xs[0], xs[1]
    A, p, k = _u(rng, 4.0, 7.0), _u(rng, 0.35, 0.65), _u(rng, 1.0, 3.0)
    return {"law": f"{fmt(A)} * {x} ** {fmt(p)} * exp(-{fmt(k)} / ({z} + 0.5))", "uses": [x, z], "form": "power_exp"}


LAW_FAMILIES: dict[str, Callable[[Any, Sequence[str]], dict]] = {
    "sat_lin": law_sat_lin, "thresh_quad": law_thresh_quad, "peak": law_peak, "power_exp": law_power_exp}

VAR_POOL = ("temper", "dose", "flux", "load", "spin", "moist", "charge", "feed", "tilt", "brine", "glow", "pulse")


def draw_variables(rng, n: int) -> list[str]:
    """n distinct neutral variable names (seeded)."""
    idx = rng.permutation(len(VAR_POOL))[:n]
    return [VAR_POOL[int(i)] for i in idx]
