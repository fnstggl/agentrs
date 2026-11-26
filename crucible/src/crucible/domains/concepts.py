"""Executable concepts for the education ecosystem (MASTER_SPEC 9.10).

A concept is a kind of problem the world can generate, parse, solve and grade: `gen(rng, level)` draws a problem,
`text(p)` renders it, `parse(text)` reads a problem back from a lesson line, `solve(p)` is the correct answer,
`buggy(p)` the answer under the concept's typical misconception, `level(p)` the difficulty (1-3) read off the
problem's structure, and `grade(answer, p)` compares a stated answer with the solution after normalisation.

Grading is form-aware: where the answer format asks for lowest terms (fractions, equation solutions), an equal but
unsimplified fraction is wrong, and a quotient with a remainder at least the divisor is wrong. A misconception only
bites where it yields a wrong answer (`misconception(p)` is None when the buggy procedure happens to give the right
one), so a learner whose misconception was strengthened by wrong content can never score better because of it
(the grader-validity property audit I S.5 asked for; tests/test_v5_domains_education.py checks it for every concept
and level).

Nothing here is a score anyone assigns: a worked problem in a lesson counts only if its problem is written in one of
the curriculum's notations, and it is correct only if the stated answer equals the solver's. How a learner reads a
document (`parse_lesson`) is a reading model, not a file format: it finds worked problems and exercises in ordinary
writing (`12 + 7 = 19`, `What is 12 + 7? 19`, `Q: ... A: ...`, `Exercise: simplify 6/8 (answer: 3/4)`, numbered
lists, `=>`), and a document that explains in prose only teaches nothing (time spent viewing is not learning).
`PREREQS` is the prerequisite graph of the whole library; a world uses a subset closed under prerequisites
(`closure`).
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

INT = r"-?\d{1,6}"


def _norm_int(s: str):
    s = str(s).strip().replace(",", "")
    return int(s) if re.fullmatch(r"-?\d{1,12}", s) else None


def _norm_frac(s: str):
    s = str(s).strip().replace(" ", "")
    m = re.fullmatch(r"(-?\d{1,9})(?:/(\d{1,9}))?", s)
    if not m or (m.group(2) is not None and int(m.group(2)) == 0):
        return None
    return Fraction(int(m.group(1)), int(m.group(2) or 1))


def _norm_frac_lowest(s: str):
    """A fraction answer whose format asks for lowest terms: the value only when written in lowest terms with a
    positive denominator (an unsimplified answer such as 12/18 for 2/3 is a different, wrong answer)."""
    s = str(s).strip().replace(" ", "")
    m = re.fullmatch(r"(-?\d{1,9})(?:/(\d{1,9}))?", s)
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2) or 1)
    if den == 0:
        return None
    if math.gcd(num, den) != 1 and not (num == 0 and den == 1):
        return ("unsimplified", num, den)
    return Fraction(num, den)


def _norm_dec(s: str):
    s = str(s).strip().replace(",", "")
    if not re.fullmatch(r"-?\d{1,9}(\.\d{1,6})?", s):
        return None
    return round(float(s), 2)


def _fmt_frac(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _fmt_dec(x: float) -> str:
    s = f"{round(x, 2):.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _digits(n: int) -> list[int]:
    return [int(c) for c in str(abs(n))]


def _no_carry_add(a: int, b: int) -> int:
    da, db = _digits(a)[::-1], _digits(b)[::-1]
    out = [(da[i] if i < len(da) else 0) + (db[i] if i < len(db) else 0) for i in range(max(len(da), len(db)))]
    return int("".join(str(d % 10) for d in out[::-1]) or "0")


def _small_from_large(a: int, b: int) -> int:
    da, db = _digits(a)[::-1], _digits(b)[::-1]
    out = [abs((da[i] if i < len(da) else 0) - (db[i] if i < len(db) else 0)) for i in range(max(len(da), len(db)))]
    return int("".join(str(d) for d in out[::-1]) or "0")


@dataclass(frozen=True)
class Concept:
    name: str
    title: str
    doc: str
    fmt: str                                # how problems are written (public)
    answer_fmt: str
    pattern: str
    params: Callable                        # regex groups -> params tuple or None
    gen: Callable                           # (rng, level) -> params
    text: Callable                          # params -> problem text
    solve: Callable                         # params -> canonical answer
    buggy: Callable                         # params -> answer under the misconception
    level: Callable                         # params -> 1..3
    norm: Callable                          # answer text -> comparable value or None
    alt: tuple = ()                         # other notations of the same problem (regexes with the same groups)

    def parse(self, text: str):
        t = str(text).strip().rstrip("?.!:").strip()
        for pat in (self.pattern,) + tuple(self.alt):
            m = re.fullmatch(pat, t, flags=re.IGNORECASE)
            if not m:
                continue
            try:
                return self.params(m.groups())
            except (ValueError, ZeroDivisionError, OverflowError):
                return None
        return None

    def grade(self, answer: str, p) -> bool:
        return _grade(self.name, str(answer), tuple(p))

    def misconception(self, p) -> str | None:
        """The answer under the concept's misconception, or None where the buggy procedure gives the right answer
        (the misconception does not show on this problem)."""
        return _misconception(self.name, tuple(p))

    def key(self, p) -> str:
        return f"{self.name}:{','.join(str(x) for x in p)}"


@functools.lru_cache(maxsize=200_000)
def _solution_norm(name: str, p: tuple):
    con = LIBRARY[name]
    return con.norm(con.solve(p))


@functools.lru_cache(maxsize=400_000)
def _grade(name: str, answer: str, p: tuple) -> bool:
    a = LIBRARY[name].norm(answer)
    return a is not None and a == _solution_norm(name, p)


@functools.lru_cache(maxsize=200_000)
def _misconception(name: str, p: tuple) -> str | None:
    b = LIBRARY[name].buggy(p)
    return None if _grade(name, str(b), p) else b


def _r(rng, lo, hi) -> int:
    return int(rng.integers(lo, hi + 1))


# ---------------------------------------------------------------------------------------------- the library
def _add_gen(rng, L):
    if L == 1:
        return (_r(rng, 1, 19), _r(rng, 1, 19))
    if L == 2:
        return (_r(rng, 20, 999), _r(rng, 20, 999))
    return (_r(rng, 1000, 99999), _r(rng, 100, 9999)) if rng.random() < 0.5 else (_r(rng, -999, -1), _r(rng, 1, 999))


def _add_level(p):
    a, b = p
    return 3 if a < 0 or b < 0 or max(a, b) >= 1000 else (2 if max(a, b) >= 20 else 1)


def _sub_gen(rng, L):
    if L == 1:
        a = _r(rng, 2, 19)
        return (a, _r(rng, 1, a))
    if L == 2:
        a = _r(rng, 100, 999)
        return (a, _r(rng, 10, a))
    a = _r(rng, 10, 500)
    return (a, a + _r(rng, 1, 500))


def _sub_level(p):
    a, b = p
    return 3 if a < b else (2 if max(a, b) >= 20 else 1)


def _mul_gen(rng, L):
    if L == 1:
        return (_r(rng, 2, 9), _r(rng, 2, 9))
    if L == 2:
        return (_r(rng, 11, 99), _r(rng, 2, 9))
    return (_r(rng, 11, 99), _r(rng, 11, 99))


def _mul_level(p):
    a, b = abs(p[0]), abs(p[1])
    return 1 if max(a, b) <= 9 else (2 if min(a, b) <= 9 else 3)


def _mul_bug(p):
    a, b = p
    if max(abs(a), abs(b)) <= 9:
        return str(a + b)
    big, small = (a, b) if abs(a) >= abs(b) else (b, a)
    if abs(small) <= 9:                     # forgets to multiply the ones digit: 57 * 8 -> 50 * 8 + 7
        return str((big // 10) * 10 * small + big % 10)
    return str(sum(big * d for d in _digits(small)))   # partial products not shifted


def _div_gen(rng, L):
    if L == 1:
        b = _r(rng, 2, 9)
        return (b * _r(rng, 2, 11), b)
    if L == 2:
        b = _r(rng, 3, 12)
        return (b * _r(rng, 5, 60) + _r(rng, 1, b - 1), b)
    b = _r(rng, 7, 40)
    return (b * _r(rng, 60, 400) + _r(rng, 1, b - 1), b)


def _div_solve(p):
    q, r = divmod(p[0], p[1])
    return f"{q}" if r == 0 else f"{q} r {r}"


def _div_norm(s):
    m = re.fullmatch(r"\s*(\d{1,9})(?:\s*(?:r|remainder|rem)\s*(\d{1,9}))?\s*", str(s), flags=re.IGNORECASE)
    return None if not m else (int(m.group(1)), int(m.group(2) or 0))


def _div_level(p):
    a, b = p
    return 1 if a % b == 0 and a < 100 else (2 if a < 1000 else 3)


def _gcd_gen(rng, L):
    hi = {1: 6, 2: 15, 3: 40}[L]
    g = _r(rng, 2, hi)
    while True:
        x, y = _r(rng, 1, hi), _r(rng, 1, hi)
        if math.gcd(x, y) == 1 and x != y:
            return (g * x, g * y)


def _size_level(p, a=30, b=300):
    m = max(abs(x) for x in p)
    return 1 if m < a else (2 if m < b else 3)


def _frac_simplify_gen(rng, L):
    g = {1: lambda: _r(rng, 2, 3), 2: lambda: _r(rng, 4, 9), 3: lambda: _r(rng, 10, 36)}[L]()
    while True:
        x, y = _r(rng, 1, 12), _r(rng, 2, 13)
        if math.gcd(x, y) == 1 and x < y:
            return (g * x, g * y)


def _frac_simplify_bug(p):
    """Divides by 2 once (whatever the common factor), or not at all when a term is odd."""
    a, b = p
    return f"{a // 2}/{b // 2}" if a % 2 == 0 and b % 2 == 0 and b // 2 else f"{a}/{b}"


def _div_bug(p):
    """Drops the remainder; on an exact division stops one step early and leaves the divisor as the remainder."""
    q, r = divmod(p[0], p[1])
    return f"{q - 1} r {p[1]}" if r == 0 else str(q)


def _frac_add_gen(rng, L):
    if L == 1:
        d = _r(rng, 3, 12)
        return (_r(rng, 1, d - 1), d, _r(rng, 1, d - 1), d)
    if L == 2:
        d = _r(rng, 2, 6)
        k = _r(rng, 2, 4)
        return (_r(rng, 1, d - 1), d, _r(rng, 1, d * k - 1), d * k)
    while True:
        b, d = _r(rng, 2, 11), _r(rng, 2, 11)
        if math.gcd(b, d) == 1:
            return (_r(rng, 1, b - 1), b, _r(rng, 1, d - 1), d)


def _frac_add_level(p):
    a, b, c, d = p
    return 1 if b == d else (2 if b % d == 0 or d % b == 0 else 3)


def _frac_mul_gen(rng, L):
    while True:
        a, b, c, d = _r(rng, 1, 9), _r(rng, 2, 9 if L < 3 else 15), _r(rng, 1, 9), _r(rng, 2, 9 if L < 3 else 15)
        g = math.gcd(a * c, b * d)
        if (L == 1 and g == 1) or (L >= 2 and g > 1):
            return (a, b, c, d)


def _frac_mul_level(p):
    a, b, c, d = p
    return 1 if math.gcd(a * c, b * d) == 1 else (2 if max(b, d) <= 9 else 3)


def _pct_gen(rng, L):
    if L == 1:
        p = int(rng.choice([10, 25, 50]))
        return (p, 20 * _r(rng, 1, 25))
    if L == 2:
        while True:
            p, n = _r(rng, 1, 99), _r(rng, 2, 400)
            if (p * n) % 100 == 0 and p not in (10, 25, 50):
                return (p, n)
    while True:
        p, n = _r(rng, 1, 99), _r(rng, 2, 400)
        if (p * n) % 100 != 0:
            return (p, n)


def _pct_level(p):
    pc, n = p
    return 1 if pc in (10, 25, 50) and (pc * n) % 100 == 0 else (2 if (pc * n) % 100 == 0 else 3)


def _lin_gen(rng, L):
    a = 1 if L == 1 else (_r(rng, 2, 9) if L == 2 else -_r(rng, 2, 9))
    x, b = _r(rng, -9 if L == 3 else 1, 20), _r(rng, 1, 30)
    sign = "+" if rng.random() < 0.5 else "-"
    c = a * x + (b if sign == "+" else -b)
    return (a, sign, b, c)


def _lin_params(g):
    return (int(g[0]) if g[0] not in ("", "-") else (-1 if g[0] == "-" else 1), g[1], int(g[2]), int(g[3]))


def _lin_text(p):
    a, s, b, c = p
    lead = "" if a == 1 else ("-" if a == -1 else str(a))
    return f"{lead}x {s} {b} = {c}"


def _lin_x(p, flip=False):
    a, s, b, c = p
    bb = b if s == "+" else -b
    if flip:
        bb = -bb
    return _fmt_frac(Fraction(c - bb, a))


def _lin_level(p):
    a, s, b, c = p
    return 1 if a == 1 else (2 if a > 0 and c >= 0 else 3)


def _mean_gen(rng, L):
    n = {1: _r(rng, 2, 3), 2: _r(rng, 4, 6), 3: _r(rng, 7, 9)}[L]
    lo = -20 if L == 3 else 0
    return tuple(_r(rng, lo, 30 if L == 1 else 99) for _ in range(n))


def _mean_params(g):
    xs = tuple(int(x) for x in re.split(r"\s*,\s*", g[0].strip()))
    if not 2 <= len(xs) <= 10:
        raise ValueError("2-10 numbers")
    return xs


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _mean_level(p):
    return 3 if len(p) >= 7 or min(p) < 0 else (2 if len(p) >= 4 else 1)


def _bin_gen(rng, L):
    return (_r(rng, *{1: (2, 15), 2: (16, 255), 3: (256, 65535)}[L]),)


def _bin_norm(s):
    s = str(s).strip()
    return s.lstrip("0") or "0" if re.fullmatch(r"[01]{1,40}", s) else None


def _pow_gen(rng, L):
    m = _r(rng, 3, 13 if L < 3 else 97)
    return (_r(rng, 2, 9 if L < 3 else 50), _r(rng, 2, 3) if L == 1 else (_r(rng, 4, 10) if L == 2 else _r(rng, 11, 60)), m)


def _pow_level(p):
    a, e, m = p
    return 1 if e <= 3 else (2 if e <= 10 else 3)


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _pair(g):
    return (int(g[0]), int(g[1]))


def _frac_pair(g):
    a, b, c, d = (int(x) for x in g)
    if b == 0 or d == 0:
        raise ZeroDivisionError
    return (a, b, c, d)


LIBRARY: dict[str, Concept] = {}


def _c(**kw):
    LIBRARY[kw["name"]] = Concept(**kw)


_c(name="add", title="Addition of whole numbers", doc="Add two whole numbers (carrying; negative numbers at level 3).",
   fmt="a + b", answer_fmt="the sum, e.g. 85", pattern=rf"({INT})\s*\+\s*({INT})", params=_pair, gen=_add_gen,
   alt=(rf"(?:what is|compute|calculate|find)?\s*({INT})\s+plus\s+({INT})", rf"(?:the )?sum of ({INT}) and ({INT})"),
   text=lambda p: f"{p[0]} + {p[1]}", solve=lambda p: str(p[0] + p[1]),
   buggy=lambda p: str(_no_carry_add(p[0], p[1]) if min(p) >= 0 else abs(p[0]) + abs(p[1])),
   level=_add_level, norm=_norm_int)
_c(name="sub", title="Subtraction of whole numbers", doc="Subtract (borrowing; negative results at level 3).",
   fmt="a - b", answer_fmt="the difference, e.g. 17", pattern=r"(\d{1,6})\s*-\s*(\d{1,6})", params=_pair,
   alt=(r"(\d{1,6})\s+minus\s+(\d{1,6})",),
   gen=_sub_gen, text=lambda p: f"{p[0]} - {p[1]}", solve=lambda p: str(p[0] - p[1]),
   buggy=lambda p: str(_small_from_large(p[0], p[1]) if p[0] >= p[1] else p[1] - p[0]),
   level=_sub_level, norm=_norm_int)
_c(name="mul", title="Multiplication", doc="Multiply whole numbers (times tables, then long multiplication).",
   fmt="a * b", answer_fmt="the product, e.g. 56", pattern=r"(-?\d{1,4})\s*[*×]\s*(-?\d{1,4})", params=_pair,
   alt=(r"(-?\d{1,4})\s+(?:times|multiplied by)\s+(-?\d{1,4})", r"(-?\d{1,4})\s*x\s*(-?\d{1,4})"),
   gen=_mul_gen, text=lambda p: f"{p[0]} * {p[1]}", solve=lambda p: str(p[0] * p[1]), buggy=_mul_bug,
   level=_mul_level, norm=_norm_int)
_c(name="div", title="Division with remainder", doc="Divide whole numbers; write the remainder as 'q r r'.",
   fmt="a div b", answer_fmt="quotient and remainder, e.g. 7 r 2 (or just 7 when exact)",
   pattern=r"(\d{1,6})\s*(?:div|÷)\s*(\d{1,4})", params=lambda g: (int(g[0]), int(g[1])) if int(g[1]) else 1 / 0,
   alt=(r"(\d{1,6})\s+divided by\s+(\d{1,4})",),
   gen=_div_gen, text=lambda p: f"{p[0]} div {p[1]}", solve=_div_solve, buggy=_div_bug,
   level=_div_level, norm=_div_norm)
_c(name="gcd", title="Greatest common divisor", doc="The largest whole number dividing both numbers.",
   fmt="gcd(a, b)", answer_fmt="a whole number, e.g. 6", pattern=r"gcd\(\s*(\d{1,5})\s*,\s*(\d{1,5})\s*\)",
   alt=(r"(?:the )?(?:gcd|greatest common divisor|highest common factor) of (\d{1,5}) and (\d{1,5})",),
   params=_pair, gen=_gcd_gen, text=lambda p: f"gcd({p[0]}, {p[1]})", solve=lambda p: str(math.gcd(*p)),
   buggy=lambda p: str(min(p)), level=_size_level, norm=_norm_int)
_c(name="lcm", title="Least common multiple", doc="The smallest positive number both numbers divide.",
   fmt="lcm(a, b)", answer_fmt="a whole number, e.g. 12", pattern=r"lcm\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)",
   alt=(r"(?:the )?(?:lcm|least common multiple|lowest common multiple) of (\d{1,4}) and (\d{1,4})",),
   params=lambda g: _pair(g) if min(_pair(g)) > 0 else 1 / 0, gen=_gcd_gen, text=lambda p: f"lcm({p[0]}, {p[1]})",
   solve=lambda p: str(_lcm(*p)), buggy=lambda p: str(p[0] * p[1]), level=_size_level, norm=_norm_int)
_c(name="frac_simplify", title="Simplifying fractions", doc="Write a fraction in lowest terms.",
   fmt="simplify a/b", answer_fmt="a fraction in lowest terms, e.g. 2/3", pattern=r"simplify\s+(\d{1,4})\s*/\s*(\d{1,4})",
   alt=(r"(\d{1,4})\s*/\s*(\d{1,4}) in (?:its )?lowest terms", r"reduce\s+(\d{1,4})\s*/\s*(\d{1,4})"),
   params=lambda g: _pair(g) if int(g[1]) else 1 / 0, gen=_frac_simplify_gen, text=lambda p: f"simplify {p[0]}/{p[1]}",
   solve=lambda p: _fmt_frac(Fraction(*p)), buggy=_frac_simplify_bug,
   level=lambda p: 1 if math.gcd(*p) <= 3 else (2 if math.gcd(*p) <= 9 else 3), norm=_norm_frac_lowest)
_c(name="frac_add", title="Adding fractions", doc="Add two fractions; give the result in lowest terms.",
   fmt="a/b + c/d", answer_fmt="a fraction in lowest terms, e.g. 5/6",
   pattern=r"(\d{1,3})\s*/\s*(\d{1,3})\s*\+\s*(\d{1,3})\s*/\s*(\d{1,3})", params=_frac_pair, gen=_frac_add_gen,
   text=lambda p: f"{p[0]}/{p[1]} + {p[2]}/{p[3]}", solve=lambda p: _fmt_frac(Fraction(p[0], p[1]) + Fraction(p[2], p[3])),
   buggy=lambda p: _fmt_frac(Fraction(p[0] + p[2], p[1] + p[3])), level=_frac_add_level, norm=_norm_frac_lowest)
_c(name="frac_mul", title="Multiplying fractions", doc="Multiply two fractions; give the result in lowest terms.",
   fmt="a/b * c/d", answer_fmt="a fraction in lowest terms, e.g. 3/8",
   pattern=r"(\d{1,3})\s*/\s*(\d{1,3})\s*[*×]\s*(\d{1,3})\s*/\s*(\d{1,3})", params=_frac_pair,
   gen=_frac_mul_gen, text=lambda p: f"{p[0]}/{p[1]} * {p[2]}/{p[3]}",
   solve=lambda p: _fmt_frac(Fraction(p[0], p[1]) * Fraction(p[2], p[3])),
   buggy=lambda p: _fmt_frac(Fraction(p[0] * p[3], p[1] * p[2])) if p[2] else "0", level=_frac_mul_level,
   norm=_norm_frac_lowest)
_c(name="percent", title="Percentages", doc="Compute a percentage of a number.", fmt="p% of n",
   answer_fmt="a number (up to 2 decimals), e.g. 12.5", pattern=r"(\d{1,3})\s*%\s*of\s*(\d{1,6})", params=_pair,
   alt=(r"(\d{1,3})\s+percent of\s+(\d{1,6})",),
   gen=_pct_gen, text=lambda p: f"{p[0]}% of {p[1]}", solve=lambda p: _fmt_dec(p[0] * p[1] / 100),
   buggy=lambda p: _fmt_dec(p[0] * p[1] / 10), level=_pct_level, norm=_norm_dec)
_c(name="lin_eq", title="Linear equations", doc="Solve a*x + b = c for x.", fmt="ax + b = c (or ax - b = c)",
   answer_fmt="x as a whole number or fraction, e.g. -3", pattern=r"(-?\d{0,3})x\s*([+-])\s*(\d{1,4})\s*=\s*(-?\d{1,5})",
   params=lambda g: _lin_params(g) if _lin_params(g)[0] != 0 else 1 / 0, gen=_lin_gen, text=_lin_text,
   alt=(r"solve\s+(-?\d{0,3})x\s*([+-])\s*(\d{1,4})\s*=\s*(-?\d{1,5})(?:\s+for x)?",),
   solve=_lin_x, buggy=lambda p: _lin_x(p, flip=True), level=_lin_level, norm=_norm_frac_lowest)
_c(name="mean", title="Averages (mean)", doc="The mean of a list of numbers.", fmt="mean of a, b, c",
   answer_fmt="a number (up to 2 decimals), e.g. 4.5", pattern=r"mean of\s+(-?\d{1,4}(?:\s*,\s*-?\d{1,4}){1,9})",
   alt=(r"(?:the )?(?:average|arithmetic mean) of\s+(-?\d{1,4}(?:\s*,\s*-?\d{1,4}){1,9})",),
   params=_mean_params, gen=_mean_gen, text=lambda p: "mean of " + ", ".join(str(x) for x in p),
   solve=lambda p: _fmt_dec(sum(p) / len(p)), buggy=lambda p: _fmt_dec(_median(p)), level=_mean_level, norm=_norm_dec)
_c(name="binary", title="Binary numbers", doc="Write a whole number in base 2.", fmt="n in binary",
   answer_fmt="binary digits, e.g. 1101", pattern=r"(\d{1,6})\s+in binary", params=lambda g: (int(g[0]),),
   alt=(r"(?:write|convert)\s+(\d{1,6})\s+(?:in|to) (?:binary|base 2)",),
   gen=_bin_gen, text=lambda p: f"{p[0]} in binary", solve=lambda p: bin(p[0])[2:],
   buggy=lambda p: bin(p[0])[2:][::-1].lstrip("0") or "0", level=lambda p: _size_level(p, 16, 256), norm=_bin_norm)
_c(name="power_mod", title="Modular powers", doc="Compute a^e modulo m.", fmt="a^e mod m",
   answer_fmt="a whole number, e.g. 4", pattern=r"(\d{1,3})\s*\^\s*(\d{1,3})\s*mod\s*(\d{1,3})",
   params=lambda g: (int(g[0]), int(g[1]), int(g[2])) if int(g[2]) > 1 else 1 / 0, gen=_pow_gen,
   text=lambda p: f"{p[0]}^{p[1]} mod {p[2]}", solve=lambda p: str(pow(p[0], p[1], p[2])),
   buggy=lambda p: str((p[0] * p[1]) % p[2]), level=_pow_level, norm=_norm_int)

PREREQS: dict[str, tuple[str, ...]] = {
    "add": (), "sub": ("add",), "mul": ("add",), "div": ("mul", "sub"), "gcd": ("div",), "lcm": ("gcd", "mul"),
    "frac_simplify": ("gcd",), "frac_add": ("frac_simplify", "lcm"), "frac_mul": ("frac_simplify", "mul"),
    "percent": ("frac_mul",), "lin_eq": ("mul", "sub", "div"), "mean": ("add", "div"), "binary": ("div",),
    "power_mod": ("mul", "div"),
}
# parse order: more specific formats first (a fraction sum must never be read as an integer sum)
PARSE_ORDER = ("frac_add", "frac_mul", "frac_simplify", "lin_eq", "percent", "power_mod", "gcd", "lcm", "mean",
               "binary", "div", "mul", "sub", "add")


def closure(names) -> list[str]:
    out: set[str] = set()
    todo = list(names)
    while todo:
        c = todo.pop()
        if c not in out:
            out.add(c)
            todo.extend(PREREQS[c])
    return [c for c in PREREQS if c in out]


def depth(name: str, names=None) -> int:
    pre = [p for p in PREREQS[name] if names is None or p in names]
    return 0 if not pre else 1 + max(depth(p, names) for p in pre)


def generate(name: str, rng, level: int, tries: int = 30) -> tuple:
    """A problem of `name` whose structural level is `level` (the generator's draw is re-drawn until it is)."""
    con = LIBRARY[name]
    p = con.gen(rng, level)
    for _ in range(tries):
        if con.level(p) == level:
            return p
        p = con.gen(rng, level)
    return p


def identify(problem: str, curriculum) -> tuple[str, tuple] | None:
    for c in PARSE_ORDER:
        if c in curriculum:
            p = LIBRARY[c].parse(problem)
            if p is not None:
                return c, p
    return None


# ---------------------------------------------------------------------------------------------- the reading model
_PREFIX = re.compile(
    r"^\s*(?:[-*\u2022]\s*|\(?\d{1,3}[.)]\s*)?"
    r"(?:(?P<kw>worked example|for example|example|e\.g\.|practice problem|practice|exercise|try this|try it|try|"
    r"question|quiz|problem|your turn|test yourself|check yourself|q)\s*\d{0,3}\s*[:.)\-]\s*)?", re.IGNORECASE)
_PRACTICE = {"practice", "practice problem", "exercise", "try this", "try it", "try", "question", "quiz", "problem",
             "your turn", "test yourself", "check yourself", "q"}
_ASK = re.compile(r"^(?:what is|what's|compute|calculate|find|work out|evaluate|determine|solve for x in)\s+",
                  re.IGNORECASE)
_SPLITS = (
    re.compile(r"^(?P<p>.+?)\s*(?:=>|->|\u2192)\s*(?P<a>.+)$"),
    re.compile(r"^(?P<p>.+?)\s*\(\s*(?:answer|ans|solution)\s*[:=]?\s*(?P<a>[^()]+?)\s*\)\s*\.?$", re.IGNORECASE),
    re.compile(r"^(?P<p>.+?)[\s,;]+(?:answer|ans|solution|a)\s*[:=]\s*(?P<a>.+)$", re.IGNORECASE),
    re.compile(r"^(?P<p>.+?)[,;:]\s*x\s*=\s*(?P<a>-?[\d/ ]+)$", re.IGNORECASE),
    re.compile(r"^(?P<p>.+?)\?\s*(?P<a>\S.*)$"),
    re.compile(r"^(?P<p>.+?)\s+(?:is|equals|gives|makes)\s+(?P<a>\S.*)$", re.IGNORECASE),
    re.compile(r"^(?P<p>[^=]+?)\s*=\s*(?P<a>[^=]+)$"),
)
_ANSWER_LINE = re.compile(r"^\s*(?:a|answer|ans|solution)\s*[:=.)]\s*(?P<a>.+?)\s*$", re.IGNORECASE)
NEXT = re.compile(r"\b(?:next|continue(?: with)?|then read|go on to|follow[- ]?up|part \d+)\b[^\n]{0,80}?"
                  r"(doc_[0-9a-z_]+)", re.IGNORECASE)


def _clean_answer(a: str) -> str:
    a = str(a).strip().strip("`*_\"'").strip()
    a = re.sub(r"^(?:x\s*=\s*|it is\s+|it's\s+|the answer is\s+)", "", a, flags=re.IGNORECASE)
    return a.rstrip(".!;,").strip()[:40]


def _problem(text: str, curriculum):
    t = _ASK.sub("", str(text).strip().strip("`*_").strip()).strip()
    return identify(t, curriculum)


def read_line(line: str, curriculum):
    """One line as a learner reads it: (kind, concept, params, stated answer or None) for a worked problem or an
    exercise written in a curriculum notation, else None."""
    m = _PREFIX.match(line)
    kw = (m.group("kw") or "").lower() if m else ""
    rest = line[m.end():] if m else line
    rest = rest.strip()
    if not rest:
        return None
    kind = "practice" if kw in _PRACTICE else "example"
    for sp in _SPLITS:
        mm = sp.match(rest)
        if not mm:
            continue
        hit = _problem(mm.group("p"), curriculum)
        if hit is not None:
            ans = _clean_answer(mm.group("a"))
            if ans:
                return kind, hit[0], hit[1], ans
    hit = _problem(rest.rstrip("?"), curriculum)
    if hit is not None:
        return ("practice" if rest.endswith("?") or kw else kind), hit[0], hit[1], None
    return None


def parse_lesson(body: str, curriculum, max_lines: int = 60) -> dict:
    """The learning items a learner reads in a document: worked examples and exercises, in order, each with its
    concept, level and whether its stated answer is correct (evaluated by the world's solver), plus a link to a
    next document. An exercise whose answer follows on the next line (`Q: ...` / `A: ...`) is read as one item; a
    problem without any stated answer teaches nothing and is skipped."""
    items = []
    nxt = None
    lines = str(body).splitlines()
    k = 0
    while k < len(lines):
        line = lines[k]
        k += 1
        if nxt is None:
            n = NEXT.search(line)
            if n:
                nxt = n.group(1)
        if len(items) >= max_lines or len(line) > 400:
            continue
        got = read_line(line, curriculum)
        if got is None:
            continue
        kind, c, p, ans = got
        if ans is None and k < len(lines):
            am = _ANSWER_LINE.match(lines[k])
            if am:
                ans = _clean_answer(am.group("a"))
                kind = "practice"
                k += 1
        if not ans:
            continue
        con = LIBRARY[c]
        items.append({"kind": kind, "concept": c, "level": int(con.level(p)), "key": con.key(p), "params": list(p),
                      "stated": ans, "correct": bool(con.grade(ans, p))})
    return {"items": items, "next": nxt}


# ---------------------------------------------------------------------------------------------- writing styles
STYLES = ("arrow", "equals", "qa", "question", "numbered", "answer_paren")


def write_item(kind: str, con: "Concept", p, answer: str, style: str, n: int = 1) -> list[str]:
    """One worked problem or exercise written in a style (how authors - population teachers, references - write)."""
    t = con.text(p)
    lab = "Example" if kind == "example" else "Practice"
    if style == "arrow":
        return [f"{lab}: {t} => {answer}"]
    if style == "equals" and "=" not in t:
        return [f"{'' if kind == 'example' else 'Try: '}{t} = {answer}"]
    if style == "qa":
        return [f"Q: What is {t}?", f"A: {answer}"]
    if style == "question":
        return [f"{'' if kind == 'example' else 'Exercise: '}What is {t}? {answer}"]
    if style == "numbered":
        return [f"{n}. {'' if kind == 'example' else 'Exercise: '}{t} (answer: {answer})"]
    return [f"{'Example' if kind == 'example' else 'Exercise'} {n}: {t} (answer: {answer})"]
