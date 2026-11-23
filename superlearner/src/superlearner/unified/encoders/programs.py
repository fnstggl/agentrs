"""An executable program language over a world's levers.

A program is a token sequence ``BOS statement+ EOS``. Statements are ``SET lever value``, ``ADD lever value`` and
``WAIT duration``. Programs compose and run for several ticks, so they express pulses, ramps, staged changes and
experiments that are not enumerated anywhere. The language is bounded: it does not write arbitrary code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

PAD, BOS, EOS, SET, ADD, WAIT = 0, 1, 2, 3, 4, 5


@dataclass(frozen=True)
class Language:
    n_levers: int = 12
    n_values: int = 21
    durations: tuple = (1, 2, 3, 4, 6, 8, 12, 16)
    max_statements: int = 6
    max_ticks: int = 24

    @property
    def lever0(self) -> int:
        return 6

    @property
    def value0(self) -> int:
        return self.lever0 + self.n_levers

    @property
    def dur0(self) -> int:
        return self.value0 + self.n_values

    @property
    def vocab(self) -> int:
        return self.dur0 + len(self.durations)

    @property
    def max_len(self) -> int:
        return 2 + 3 * self.max_statements

    @property
    def n_features(self) -> int:
        return 2 * self.n_levers + 8

    @property
    def move_index(self) -> int:
        """Feature holding the mean absolute change a program makes to the available levers."""
        return 2 * self.n_levers + 6

    def value(self, token: int) -> float:
        return -1.0 + 2.0 * (token - self.value0) / (self.n_values - 1)

    def value_token(self, x: float) -> int:
        k = int(round((float(np.clip(x, -1, 1)) + 1.0) * 0.5 * (self.n_values - 1)))
        return self.value0 + k

    def duration(self, token: int) -> int:
        return int(self.durations[token - self.dur0])

    def dur_token(self, ticks: int) -> int:
        return self.dur0 + self.durations.index(ticks)


DEFAULT = Language()


def encode(statements: list[tuple], lang: Language = DEFAULT) -> np.ndarray:
    """``[("set", lever, x), ("add", lever, dx), ("wait", ticks)]`` to a padded token array."""
    if not 1 <= len(statements) <= lang.max_statements:
        raise ValueError("program must have between 1 and max_statements statements")
    out = [BOS]
    for st in statements:
        if st[0] == "wait":
            out += [WAIT, lang.dur_token(int(st[1]))]
        elif st[0] in ("set", "add"):
            out += [SET if st[0] == "set" else ADD, lang.lever0 + int(st[1]), lang.value_token(st[2])]
        else:
            raise ValueError(f"unknown statement {st[0]}")
    out.append(EOS)
    arr = np.zeros(lang.max_len, dtype=np.int64)
    arr[:len(out)] = out
    return arr


def parse(tokens, lang: Language = DEFAULT) -> list[tuple] | None:
    """Strict parser: returns the statements, or None for any malformed program."""
    t = [int(x) for x in tokens]
    if not t or t[0] != BOS:
        return None
    i, out = 1, []
    while i < len(t):
        c = t[i]
        if c == EOS:
            if not out or any(x != PAD for x in t[i + 1:]):
                return None
            return out
        if len(out) >= lang.max_statements:
            return None
        if c in (SET, ADD):
            if i + 2 >= len(t):
                return None
            lv, val = t[i + 1], t[i + 2]
            if not lang.lever0 <= lv < lang.value0 or not lang.value0 <= val < lang.dur0:
                return None
            out.append(("set" if c == SET else "add", lv - lang.lever0, lang.value(val)))
            i += 3
        elif c == WAIT:
            if i + 1 >= len(t) or not lang.dur0 <= t[i + 1] < lang.vocab:
                return None
            out.append(("wait", lang.duration(t[i + 1])))
            i += 2
        else:
            return None
    return None


def execute(statements, settings: np.ndarray, lever_mask: np.ndarray, lang: Language = DEFAULT) -> np.ndarray | None:
    """Compile to a per-tick schedule ``[ticks, n_levers]``. None if the program touches an unavailable lever or
    exceeds the duration limit. A program that does not end in WAIT applies its changes for one tick."""
    if statements is None:
        return None
    cur = np.asarray(settings, dtype=np.float32).copy()
    ticks: list[np.ndarray] = []
    for st in statements:
        if st[0] == "wait":
            ticks.extend(cur.copy() for _ in range(st[1]))
        else:
            if lever_mask[st[1]] <= 0.5:
                return None
            cur[st[1]] = st[2] if st[0] == "set" else float(np.clip(cur[st[1]] + st[2], -1, 1))
    if not statements or statements[-1][0] != "wait":
        ticks.append(cur.copy())
    if len(ticks) > lang.max_ticks:
        return None
    return np.stack(ticks)


def features(statements, settings: np.ndarray, lever_mask: np.ndarray, lang: Language = DEFAULT,
             evidence: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Execution-relevant structure of a program in its current context: per-lever change and final setting, then
    log-duration, statement count, time-varying flag and validity. Returns ``(features, ticks)``; ticks 0 if invalid."""
    f = np.zeros(lang.n_features, dtype=np.float32)
    sched = execute(statements, settings, lever_mask, lang)
    if sched is None:
        return f, 0
    final = sched[-1]
    n = lang.n_levers
    f[:n] = final - settings
    f[n:2 * n] = final
    f[2 * n] = np.log1p(len(sched)) / np.log1p(lang.max_ticks)
    f[2 * n + 1] = len(statements) / lang.max_statements
    f[2 * n + 2] = float(np.abs(sched - sched[0]).max() > 1e-6)
    f[2 * n + 3] = 1.0
    change = final - settings
    if evidence is not None:     # first-order effect of this program under the agent's own measured slopes
        f[2 * n + 4] = float(np.clip(5 * (change * evidence[:, 0]).sum(), -5, 5)) / 5
        f[2 * n + 5] = float(np.clip(5 * (change * evidence[:, 1]).sum(), -5, 5)) / 5
    active = lever_mask > 0.5
    f[2 * n + 6] = float(np.abs(change[active]).mean()) if active.any() else 0.0
    f[2 * n + 7] = float(np.abs(change).max())
    return f, len(sched)


class Grammar:
    """Token-level validity masks, so a generator's every sample parses."""
    ST_FIRST, ST_NEXT, ST_LEVER, ST_VALUE, ST_DUR, ST_DONE = 0, 1, 2, 3, 4, 5

    def __init__(self, lang: Language = DEFAULT, device="cpu"):
        self.lang = lang
        v = lang.vocab
        m = torch.zeros(6, v, dtype=torch.bool, device=device)
        m[self.ST_FIRST, [SET, ADD, WAIT]] = True
        m[self.ST_NEXT, [SET, ADD, WAIT, EOS]] = True
        m[self.ST_LEVER, lang.lever0:lang.value0] = True
        m[self.ST_VALUE, lang.value0:lang.dur0] = True
        m[self.ST_DUR, lang.dur0:v] = True
        m[self.ST_DONE, PAD] = True
        self.base = m
        self.device = device

    def to(self, device) -> "Grammar":
        return Grammar(self.lang, device)

    def step(self, state: torch.Tensor, n_stmts: torch.Tensor, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the grammar state after ``token``."""
        new, n = state.clone(), n_stmts.clone()
        start = (state == self.ST_FIRST) | (state == self.ST_NEXT)
        new[start & ((token == SET) | (token == ADD))] = self.ST_LEVER
        new[start & (token == WAIT)] = self.ST_DUR
        new[start & (token == EOS)] = self.ST_DONE
        new[state == self.ST_LEVER] = self.ST_VALUE
        ended = (state == self.ST_VALUE) | (state == self.ST_DUR)
        new[ended] = self.ST_NEXT
        n = n + ended.long()
        return new, n

    def mask(self, state: torch.Tensor, n_stmts: torch.Tensor, lever_mask: torch.Tensor) -> torch.Tensor:
        """``[B, vocab]`` allowed tokens, given ``lever_mask`` ``[B, n_levers]``."""
        out = self.base[state].clone()
        lang = self.lang
        lever_rows = state == self.ST_LEVER
        if lever_rows.any():
            out[lever_rows, lang.lever0:lang.value0] = lever_mask[lever_rows] > 0.5
        full = (state == self.ST_NEXT) & (n_stmts >= lang.max_statements)
        if full.any():
            out[full] = False
            out[full, EOS] = True
        return out


def structured_candidates(rng: np.random.Generator, lever_mask: np.ndarray, k: int, lang: Language = DEFAULT) -> list[np.ndarray]:
    """A fixed, untrained proposal distribution: hold, one- and two-lever changes, staged probes. It has no access to
    anything but the public lever mask, and it is the control against which a trainable generator is compared."""
    levers = np.flatnonzero(lever_mask > 0.5)
    out = [encode([("wait", 1)], lang)]
    if len(levers) == 0:
        return out * k
    while len(out) < k:
        r = rng.random()
        dur = int(rng.choice([1, 2, 4]))
        a = int(rng.choice(levers))
        if r < 0.35:
            st = [("add", a, float(rng.choice([-0.4, -0.2, -0.1, 0.1, 0.2, 0.4]))), ("wait", dur)]
        elif r < 0.6:
            st = [("set", a, float(rng.uniform(-1, 1))), ("wait", dur)]
        elif r < 0.8 and len(levers) > 1:
            b = int(rng.choice(levers[levers != a]))
            st = [("add", a, float(rng.choice([-0.3, -0.1, 0.1, 0.3]))),
                  ("add", b, float(rng.choice([-0.3, -0.1, 0.1, 0.3]))), ("wait", dur)]
        elif r < 0.9:
            dx = float(rng.choice([-0.4, 0.4]))
            st = [("add", a, dx), ("wait", 2), ("add", a, -dx), ("wait", 2)]
        else:
            st = [("wait", int(rng.choice([2, 4, 8])))]
        out.append(encode(st, lang))
    return out
