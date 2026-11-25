"""Token-level constraint automata for schema-valid argument strings (used by grammar.py).

Two constraints, both exact, both re-playable from the generated token ids alone (so the learner re-applies the
identical masks when it rescores an action):

* `JsonStringTables` - per-token tables over the whole vocabulary for generating the *body of a JSON string
  literal* (the opening quote is forced by the walker; the hole ends with the token that carries the closing quote).
  States: NORMAL (0) and ESC (1, right after a backslash). Allowed escapes: \\" \\\\ \\/ \\b \\f \\n \\r \\t
  (\\uXXXX is not generated: every character can be written directly because only tokens that decode to complete
  UTF-8 are allowed; control characters other than the five escapable ones cannot be produced - documented
  limitation). A token is allowed only if it never leaves the string except through a final closing quote, never
  contains a raw control character, and the length bounds (decoded code points, exactly Python `len`) and the
  remaining token budget can still be met. `mask()` is vectorised over the vocabulary on the GPU.

* `RegexDFA` - `pattern` strings (Crucible validates with `re.fullmatch`) are generated character by character over
  single-character printable-ASCII tokens (no quote / backslash). The pattern is parsed with Python's own regex
  parser (`re._parser`) into a Thompson NFA and determinised eagerly; for every DFA state the set of achievable
  completion lengths is precomputed as a bitset, so a character is allowed only if an accepting completion that
  satisfies minLength/maxLength and the token budget remains reachable. Unsupported regex features (back-references,
  look-around, case-insensitive flags, anchors inside the pattern) raise `UnsupportedSchema`; such a tool is then
  excluded from the action generator's availability mask (logged), never generated invalidly.
"""
from __future__ import annotations

import re
from typing import Optional

import torch

try:                                    # Python >= 3.11
    from re import _parser as sre_parse, _constants as sre_c
except ImportError:                     # pragma: no cover
    import sre_parse                    # type: ignore
    import sre_constants as sre_c       # type: ignore

NORMAL, ESC = 0, 1
_ESCAPES = set('"\\/bfnrt')
PATTERN_ALPHABET = [chr(c) for c in range(0x20, 0x7F) if chr(c) not in '"\\']


class UnsupportedSchema(ValueError):
    """The schema (or a pattern inside it) uses a feature the constrained decoder cannot guarantee."""


# ============================================================================ token texts
def _bytes_to_unicode() -> dict:
    """GPT-2 byte-level BPE alphabet (bytes -> printable unicode chars), as used by the Qwen tokenizers."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def token_texts(tokenizer, vocab_size: int) -> list:
    """Decoded text of every token id in [0, vocab_size): str when the token's bytes are complete UTF-8, else None
    (partial multi-byte pieces, special/added tokens, padding rows beyond the tokenizer)."""
    dec = {v: k for k, v in _bytes_to_unicode().items()}
    n_tok = len(tokenizer)
    pieces = tokenizer.convert_ids_to_tokens(list(range(min(n_tok, vocab_size))))
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    added = set()
    try:
        added = set(tokenizer.get_added_vocab().values())
    except Exception:  # noqa: BLE001
        pass
    out: list = [None] * vocab_size
    for i, p in enumerate(pieces):
        if p is None or i in special or i in added:
            continue
        try:
            b = bytes(dec[c] for c in p)
            out[i] = b.decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            out[i] = None
    return out


# ============================================================================ JSON string body
def simulate_json_string(text: str, state: int):
    """Feed one token's text into the JSON-string-body automaton from `state`.
    Returns (valid, closes, end_state, n_chars); a closing quote is valid only as the token's last character."""
    n = 0
    s = state
    last = len(text) - 1
    for k, ch in enumerate(text):
        if s == NORMAL:
            if ch == '"':
                return (k == last), True, NORMAL, n
            if ch == "\\":
                s = ESC
            elif ord(ch) < 0x20:
                return False, False, s, n
            else:
                n += 1
        else:
            if ch in _ESCAPES:
                n += 1
                s = NORMAL
            else:
                return False, False, s, n
    return (len(text) > 0), False, s, n


class JsonStringTables:
    """Vocabulary tables for the JSON-string-body automaton. Built once per tokenizer (about 1-2 s for 152k tokens)."""

    def __init__(self, texts: list, device):
        V = len(texts)
        valid = torch.zeros(2, V, dtype=torch.bool)
        closes = torch.zeros(2, V, dtype=torch.bool)
        end = torch.zeros(2, V, dtype=torch.long)
        nch = torch.zeros(2, V, dtype=torch.long)
        for i, t in enumerate(texts):
            if not t:
                continue
            for s in (NORMAL, ESC):
                ok, cl, e, n = simulate_json_string(t, s)
                if ok:
                    valid[s, i] = True
                    closes[s, i] = cl
                    end[s, i] = e
                    nch[s, i] = n
        self.V = V
        self.texts = texts
        self.valid, self.closes = valid.to(device), closes.to(device)
        self.end = end.to(torch.int8).to(device)
        self.nch = nch.to(torch.int32).to(device)
        # host copies for the per-token automaton transitions (no device synchronisation in the decode loop)
        self.valid_h = valid.numpy()
        self.closes_h = closes.numpy()
        self.end_h = end.to(torch.int8).numpy()
        self.nch_h = nch.to(torch.int32).numpy()
        single = {}
        for i, t in enumerate(texts):
            if t is not None and len(t) == 1 and t not in single:
                single[t] = i
        self.char_id = single                                      # one-character token ids
        self.quote_id = single['"']

    def mask(self, state: int, count: int, remaining: int, min_len: int, max_len: int) -> torch.Tensor:
        """Allowed next tokens [V] bool for a hole in `state` with `count` decoded chars so far and `remaining` tokens
        of budget (including this one)."""
        s = state
        v, cl, e, n = self.valid[s], self.closes[s], self.end[s], self.nch[s]
        c1 = count + n
        close_ok = v & cl & (c1 >= min_len) & (c1 <= max_len)
        need0 = (min_len - c1).clamp(min=0)
        need1 = (min_len - c1 - 1).clamp(min=0)
        r1 = remaining - 1
        cont0 = (e == NORMAL) & (c1 <= max_len) & (need0 + 1 <= r1)
        cont1 = (e == ESC) & (c1 + 1 <= max_len) & (need1 + 2 <= r1)
        return close_ok | (v & ~cl & (cont0 | cont1))

    def advance(self, state: int, count: int, token: int):
        """(new_state, new_count, closed) after appending `token` (which must be allowed)."""
        s = state
        return int(self.end_h[s, token]), count + int(self.nch_h[s, token]), bool(self.closes_h[s, token])

    @staticmethod
    def feasible(min_len: int, max_len: int, budget: int) -> bool:
        return min_len <= max_len and min_len + 1 <= budget


# ============================================================================ regex DFA
class _NFA:
    def __init__(self):
        self.eps: list = []      # state -> list of states
        self.trans: list = []    # state -> list of (frozenset chars, state)

    def new(self) -> int:
        self.eps.append([])
        self.trans.append([])
        if len(self.eps) > 20000:
            raise UnsupportedSchema("pattern expands to too many NFA states")
        return len(self.eps) - 1


_ALPHA = frozenset(PATTERN_ALPHABET)
_DIG = frozenset("0123456789")
_WORD = frozenset(c for c in PATTERN_ALPHABET if c.isalnum() or c == "_")
_SPACE = frozenset(" ")


def _category(cat) -> frozenset:
    name = str(cat)
    table = {"CATEGORY_DIGIT": _DIG, "CATEGORY_NOT_DIGIT": _ALPHA - _DIG, "CATEGORY_WORD": _WORD,
             "CATEGORY_NOT_WORD": _ALPHA - _WORD, "CATEGORY_SPACE": _SPACE, "CATEGORY_NOT_SPACE": _ALPHA - _SPACE}
    if name not in table:
        raise UnsupportedSchema(f"regex category {name}")
    return table[name]


def _charset(op, av) -> frozenset:
    n = str(op)
    if n == "LITERAL":
        return frozenset([chr(av)]) & _ALPHA
    if n == "NOT_LITERAL":
        return _ALPHA - {chr(av)}
    if n == "ANY":
        return _ALPHA
    if n == "IN":
        neg = False
        acc: set = set()
        for iop, iav in av:
            nm = str(iop)
            if nm == "NEGATE":
                neg = True
            elif nm == "LITERAL":
                acc.add(chr(iav))
            elif nm == "RANGE":
                acc.update(chr(c) for c in range(iav[0], iav[1] + 1))
            elif nm == "CATEGORY":
                acc.update(_category(iav))
            else:
                raise UnsupportedSchema(f"regex set item {nm}")
        s = frozenset(acc) & _ALPHA
        return (_ALPHA - s) if neg else s
    raise UnsupportedSchema(f"regex op {n}")


def _build(nfa: _NFA, sub, start: int, top: bool) -> int:
    """Thompson construction of a parsed sub-pattern from `start`; returns the end state."""
    items = list(sub)
    cur = start
    for k, (op, av) in enumerate(items):
        n = str(op)
        if n in ("LITERAL", "NOT_LITERAL", "ANY", "IN"):
            nxt = nfa.new()
            cs = _charset(op, av)
            if cs:
                nfa.trans[cur].append((cs, nxt))
            cur = nxt
        elif n in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"):
            lo, hi, body = av
            unbounded = hi == sre_c.MAXREPEAT
            for _ in range(lo):
                cur = _build(nfa, body, cur, False)
            if unbounded:
                loop = nfa.new()
                nfa.eps[cur].append(loop)
                end_body = _build(nfa, body, loop, False)
                nfa.eps[end_body].append(loop)
                out = nfa.new()
                nfa.eps[loop].append(out)
                cur = out
            else:
                out = nfa.new()
                for _ in range(hi - lo):
                    nfa.eps[cur].append(out)
                    cur = _build(nfa, body, cur, False)
                nfa.eps[cur].append(out)
                cur = out
        elif n in ("SUBPATTERN", "ATOMIC_GROUP"):
            if n == "SUBPATTERN":
                _g, add_flags, del_flags, body = av
                if add_flags or del_flags:
                    raise UnsupportedSchema("regex inline flags")
            else:
                body = av
            cur = _build(nfa, body, cur, False)
        elif n == "BRANCH":
            _, alts = av
            out = nfa.new()
            for alt in alts:
                s0 = nfa.new()
                nfa.eps[cur].append(s0)
                e0 = _build(nfa, alt, s0, False)
                nfa.eps[e0].append(out)
            cur = out
        elif n == "AT":
            an = str(av)
            at_start = top and k == 0 and an in ("AT_BEGINNING", "AT_BEGINNING_STRING")
            at_end = top and k == len(items) - 1 and an in ("AT_END", "AT_END_STRING")
            if not (at_start or at_end):
                raise UnsupportedSchema(f"regex anchor {an} inside the pattern")
        else:
            raise UnsupportedSchema(f"regex feature {n}")
    return cur


class RegexDFA:
    """Deterministic automaton for `re.fullmatch(pattern, s)` over PATTERN_ALPHABET with completion-length bitsets."""

    def __init__(self, pattern: str, max_len_cap: int = 1536, max_states: int = 4096):
        try:
            parsed = sre_parse.parse(pattern, 0)
        except re.error as e:
            raise UnsupportedSchema(f"bad pattern {pattern!r}: {e}") from e
        if parsed.state.flags & (re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE):
            raise UnsupportedSchema("regex flags")
        nfa = _NFA()
        s0 = nfa.new()
        acc = _build(nfa, parsed, s0, True)

        def closure(states) -> frozenset:
            stack, seen = list(states), set(states)
            while stack:
                s = stack.pop()
                for t in nfa.eps[s]:
                    if t not in seen:
                        seen.add(t)
                        stack.append(t)
            return frozenset(seen)

        start = closure([s0])
        index = {start: 0}
        sets = [start]
        trans: list = []
        i = 0
        while i < len(sets):
            cur = sets[i]
            by_char: dict = {}
            for c in PATTERN_ALPHABET:
                tgt = set()
                for s in cur:
                    for cs, t in nfa.trans[s]:
                        if c in cs:
                            tgt.add(t)
                if tgt:
                    nxt = closure(tgt)
                    if nxt not in index:
                        if len(sets) >= max_states:
                            raise UnsupportedSchema("pattern DFA too large")
                        index[nxt] = len(sets)
                        sets.append(nxt)
                    by_char[c] = index[nxt]
            trans.append(by_char)
            i += 1
        self.pattern = pattern
        self.trans = trans
        self.accepting = [acc in s for s in sets]
        self.n_states = len(sets)
        self.cap = int(max_len_cap)
        # reach[S]: bit l set <=> an accepting completion of exactly l more characters exists from S (l <= cap)
        full = (1 << (self.cap + 1)) - 1
        succ = [sorted(set(t.values())) for t in trans]
        reach = [1 if a else 0 for a in self.accepting]
        for _ in range(self.cap):
            changed = False
            new = []
            for s in range(self.n_states):
                r = reach[s]
                for t in succ[s]:
                    r |= (reach[t] << 1)
                r &= full
                changed |= r != reach[s]
                new.append(r)
            reach = new
            if not changed:
                break
        self.reach = reach

    def feasible_from(self, state: int, lo: int, hi: int) -> bool:
        """Is there an accepting completion of l characters with lo <= l <= hi?"""
        lo = max(lo, 0)
        hi = min(hi, self.cap)
        if hi < lo:
            return False
        window = (self.reach[state] >> lo) & ((1 << (hi - lo + 1)) - 1)
        return window != 0

    def allowed(self, state: int, count: int, remaining: int, min_len: int, max_len: int):
        """(list of (char, next_state) allowed now, closing allowed) for `count` chars so far and `remaining` tokens
        (including this one)."""
        out = []
        for c, t in self.trans[state].items():
            lo = min_len - count - 1
            hi = min(max_len - count - 1, remaining - 2)
            if self.feasible_from(t, lo, hi):
                out.append((c, t))
        close = self.accepting[state] and min_len <= count <= max_len
        return out, close

    def matches(self, s: str) -> bool:
        st = 0
        for c in s:
            if c not in self.trans[st]:
                return False
            st = self.trans[st][c]
        return self.accepting[st]


class PatternMasks:
    """Per-DFA-state vocabulary masks over single-character tokens (+ the closing quote), cached per state."""

    def __init__(self, dfa: RegexDFA, tables: JsonStringTables):
        missing = [c for c in PATTERN_ALPHABET if c not in tables.char_id]
        if missing:
            raise UnsupportedSchema(f"tokenizer lacks single-character tokens for {missing[:5]}")
        self.dfa = dfa
        self.tables = tables

    def mask(self, state: int, count: int, remaining: int, min_len: int, max_len: int) -> torch.Tensor:
        allowed, close = self.dfa.allowed(state, count, remaining, min_len, max_len)
        m = torch.zeros(self.tables.V, dtype=torch.bool, device=self.tables.valid.device)
        ids = [self.tables.char_id[c] for c, _ in allowed]
        if close:
            ids.append(self.tables.quote_id)
        if ids:
            m[torch.tensor(ids, device=m.device)] = True
        return m

    def advance(self, state: int, count: int, token: int):
        """(new_state, new_count, closed)."""
        if token == self.tables.quote_id:
            return state, count, True
        c = self.tables.texts[token]
        return self.dfa.trans[state][c], count + 1, False


_DFA_CACHE: dict = {}


def compile_pattern(pattern: str, cap: int = 1536) -> RegexDFA:
    key = (pattern, cap)
    if key not in _DFA_CACHE:
        _DFA_CACHE[key] = RegexDFA(pattern, cap)
    return _DFA_CACHE[key]


_TABLE_CACHE: dict = {}


def tables_for(tokenizer, vocab_size: int, device, name: Optional[str] = None) -> JsonStringTables:
    """Process-wide cache of JsonStringTables per (tokenizer name, vocab size, device)."""
    key = (name or getattr(tokenizer, "name_or_path", "tok"), vocab_size, str(device))
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = JsonStringTables(token_texts(tokenizer, vocab_size), device)
    return _TABLE_CACHE[key]
