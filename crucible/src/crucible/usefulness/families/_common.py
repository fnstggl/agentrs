"""Shared vocabulary and small helpers for the built-in category families.

Nothing here decides a hidden rule: every family draws its own rule from its own seeded generator. Strings that
may end up in a table (and therefore travel through the `transform` wire formats when categories are composed)
use only lowercase letters, digits, spaces and the characters `-().@` - never a wire delimiter.
"""
from __future__ import annotations

import re

import numpy as np

WORDS = ("amber", "anchor", "basin", "beacon", "birch", "cable", "canyon", "cedar", "cinder", "cobalt", "copper",
         "coral", "delta", "ember", "falcon", "fennel", "garnet", "glacier", "harbor", "hazel", "indigo", "iris",
         "jasper", "juniper", "kelp", "lagoon", "lantern", "lichen", "maple", "marble", "meadow", "nectar", "nickel",
         "oasis", "onyx", "orchid", "pebble", "pepper", "quartz", "quill", "raven", "reef", "saffron", "sierra",
         "spruce", "summit", "thistle", "timber", "tundra", "umber", "velvet", "violet", "walnut", "willow", "yarrow",
         "zephyr", "zinc", "atlas", "bramble", "cirrus")
FIRST = ("ada", "ben", "cora", "dan", "elsa", "finn", "gwen", "hugo", "ines", "jack", "kira", "liam", "mona", "nils",
         "olga", "pete", "quin", "rosa", "sven", "tess", "ugo", "vera", "wade", "yara")
LAST = ("abbott", "barros", "castell", "dimitri", "eklund", "ferrand", "gallo", "hartmann", "iversen", "jovanic",
        "kessler", "lindqvist", "moreau", "navarro", "okafor", "pellegri", "quintero", "rasmus", "sorensen",
        "takeda", "ulrich", "vasquez", "weiland", "zamora")
CITIES = ("alder bay", "brightwater", "cold harbor", "dunmore", "eastfield", "fallow creek", "grayling", "highmoor",
          "ironbridge", "juniper falls", "kestrel point", "lowmarsh")
TABLE_STRING = re.compile(r"[a-z0-9 ().@+-]*\Z")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


def pick(rng: np.random.Generator, seq, k: int | None = None):
    """Seeded choice from a sequence (without replacement when k is given), returned as plain Python values."""
    if k is None:
        return seq[int(rng.integers(0, len(seq)))]
    return [seq[int(i)] for i in rng.choice(len(seq), size=int(k), replace=False)]


def r3(x: float) -> float:
    return round(float(x), 3)


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def check_table(table) -> list[str]:
    """Why a table could not travel through a wire format (empty list: it can)."""
    errs = []
    if not isinstance(table, list) or not table:
        return ["a table is a non-empty list of records"]
    for i, r in enumerate(table):
        if not isinstance(r, dict):
            errs.append(f"record {i} is not an object")
            continue
        for k, v in r.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", k):
                errs.append(f"record {i}: bad field name {k!r}")
            if isinstance(v, bool) or v is None or is_int(v):
                continue
            if isinstance(v, float):
                if v != round(v, 3) or not is_num(v):
                    errs.append(f"record {i}.{k}: floats must have at most 3 decimals")
            elif isinstance(v, str):
                if not TABLE_STRING.match(v) or v != v.strip():
                    errs.append(f"record {i}.{k}: string {v!r} uses characters outside the table alphabet")
            else:
                errs.append(f"record {i}.{k}: nested values are not allowed in a table")
    return errs


class Stream:
    """Cheap scalar draws for instance generators.

    A request generator makes hundreds of small draws; one numpy call per draw would dominate the cost of
    simulating demand. A stream takes uniforms from the caller's generator in blocks and hands them out one by
    one, so an instance is still a deterministic function of the generator's state."""
    __slots__ = ("rng", "buf", "i", "n")

    def __init__(self, rng: np.random.Generator, block: int = 256):
        self.rng, self.n = rng, int(block)
        self.buf = rng.random(self.n).tolist()
        self.i = 0

    def u(self) -> float:
        i = self.i
        if i >= self.n:
            self.buf = self.rng.random(self.n).tolist()
            i = 0
        self.i = i + 1
        return self.buf[i]

    # the methods below repeat the body of `u` on purpose: they are the inner loop of every request generator
    def below(self, k: int) -> int:
        """Integer in [0, k)."""
        i = self.i
        if i >= self.n:
            self.buf = self.rng.random(self.n).tolist()
            i = 0
        self.i = i + 1
        return int(self.buf[i] * k)

    def between(self, lo: int, hi: int) -> int:
        """Integer in [lo, hi] (inclusive)."""
        i = self.i
        if i >= self.n:
            self.buf = self.rng.random(self.n).tolist()
            i = 0
        self.i = i + 1
        return lo + int(self.buf[i] * (hi - lo + 1))

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.u()

    def chance(self, p: float) -> bool:
        i = self.i
        if i >= self.n:
            self.buf = self.rng.random(self.n).tolist()
            i = 0
        self.i = i + 1
        return self.buf[i] < p

    def pick(self, seq):
        i = self.i
        if i >= self.n:
            self.buf = self.rng.random(self.n).tolist()
            i = 0
        self.i = i + 1
        return seq[int(self.buf[i] * len(seq))]

    def sample(self, seq, k: int) -> list:
        """k distinct elements, in random order."""
        pool = list(seq)
        k = min(int(k), len(pool))
        for j in range(k):
            m = j + self.below(len(pool) - j)
            pool[j], pool[m] = pool[m], pool[j]
        return pool[:k]

    def shuffled(self, seq) -> list:
        return self.sample(seq, len(seq))

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        import math
        return mu + sigma * math.sqrt(-2.0 * math.log(max(1e-12, self.u()))) * math.cos(6.283185307179586 * self.u())

    def exponential(self, mean: float) -> float:
        import math
        return -mean * math.log(max(1e-12, 1.0 - self.u()))
