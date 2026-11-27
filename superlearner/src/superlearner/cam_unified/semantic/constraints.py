"""Token-level constraint automata for schema-valid argument strings (used by grammar.py).

Two constraints, both exact, both re-playable from the generated token ids alone (so the learner re-applies the
identical masks when it rescores an action):

* `JsonStringTables` - per-token tables over the whole vocabulary for generating the *body of a JSON string
  literal* (the opening quote is forced by the walker; the hole ends with the token that carries the closing quote).
  States: NORMAL (0) and ESC (1, right after a backslash). Allowed escapes: \\" \\\\ \\/ \\b \\f \\n \\r \\t
  (\\uXXXX is not generated: every character can be written directly because only tokens that decode to complete
  UTF-8 are allowed; control characters other than the five escapable ones cannot be produced - documented
  limitation). A token is allowed only if it never contains a raw control character, the length bounds (decoded code
  points, exactly Python `len`) and the remaining token budget can still be met, and it leaves the string only
  through its first unescaped quote, after which it may carry a *suffix* of further characters. `mask()` is
  vectorised over the vocabulary on the GPU.

  Boundary-aware closing (STRING_PROTOCOL 4). A byte-level BPE tokenizer writes the end of a JSON string mostly with
  merged tokens - '",' '"}' '"],' '":' carry 51-99% of Qwen3's closing probability on JSON arguments - so a closing
  quote is allowed inside a token when the text after it (the suffix) is a prefix of the forced JSON text that
  follows the hole (`follow`): the token then also writes those forced characters and the decoder consumes them from
  the next forced segment (grammar.HoleFiller re-tokenises the rest). The plain closing quote (suffix "") is always a
  closer. Which tokens close, how many body characters they add and their suffix are fixed per (state, token), so the
  allowed set of a hole is a pure function of (state, count, budget, bounds, follow text) and replays exactly. Before
  protocol 4 a quote was valid only as a token's last character (`simulate_json_string`), which removed the merged
  closers: the constrained closing probability fell to 4e-5..0.22 (audit 2026-11-26, constraints.py:93), strings
  rarely ended by the LM's own choice and often ended in a stray comma.

* `RegexDFA` - `pattern` strings (Crucible validates with `re.fullmatch`) are generated character by character over
  single-character printable-ASCII tokens (no quote / backslash). The pattern is parsed with Python's own regex
  parser (`re._parser`) into a Thompson NFA and determinised eagerly; for every DFA state the set of achievable
  completion lengths is precomputed as a bitset, so a character is allowed only if an accepting completion that
  satisfies minLength/maxLength and the token budget remains reachable. Unsupported regex features (back-references,
  look-around, case-insensitive flags, anchors inside the pattern) raise `UnsupportedSchema`; such a tool is then
  excluded from the action generator's availability mask (logged), never generated invalidly. A pattern string
  closes with the plain quote token or with a quote-led merged token ('",' '"}') whose suffix the follow text allows.
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
STRING_PROTOCOL = 4          # 4: boundary-aware closing (merged closing tokens consume forced characters)
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


def split_close(text: str, state: int):
    """Feed one token's text from `state` up to its first unescaped quote.
    Returns (closes, n_chars, suffix): closes = the text up to an unescaped quote is a valid string body (characters
    and escapes), n_chars = decoded body characters before that quote, suffix = the token text after it (None when
    the token does not close a string from this state)."""
    n = 0
    s = state
    for k, ch in enumerate(text):
        if s == NORMAL:
            if ch == '"':
                return True, n, text[k + 1:]
            if ch == "\\":
                s = ESC
            elif ord(ch) < 0x20:
                return False, n, None
            else:
                n += 1
        else:
            if ch in _ESCAPES:
                n += 1
                s = NORMAL
            else:
                return False, n, None
    return False, n, None


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
        # merged closers (protocol 4): tokens whose first unescaped quote is followed by more text. Per state:
        # token -> (body chars before the quote, suffix) and suffix -> token ids. Their nch entry holds the body chars
        # (they are never valid as continuing tokens, so the entry is otherwise unused).
        self.merged: list = [{}, {}]
        self.by_suffix: list = [{}, {}]
        for i, t in enumerate(texts):
            if not t or '"' not in t:
                continue
            for s in (NORMAL, ESC):
                cl, n, suf = split_close(t, s)
                if cl and suf:
                    self.merged[s][i] = (n, suf)
                    self.by_suffix[s].setdefault(suf, []).append(i)
                    nch[s, i] = n
        self.max_suffix = max([len(k) for d in self.by_suffix for k in d] + [0])
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
        # quote-led merged closers ('",' '"}'): the merged closers of pattern holes (no body characters in the token)
        self.quote_led = {i: suf for i, (n, suf) in self.merged[NORMAL].items() if texts[i].startswith('"')}
        self._follow_cache: dict = {}

    # ---------------------------------------------------------------- boundary-aware closing (protocol 4)
    def follow_ids(self, state: int, follow: str, quote_led: bool = False) -> list:
        """Merged closing tokens allowed before `follow` (the forced text after the hole): their suffix is a non-empty
        prefix of it. quote_led: only tokens that start with the quote (the closers of pattern holes)."""
        out = []
        for j in range(1, min(len(follow or ""), self.max_suffix) + 1):
            for t in self.by_suffix[state].get(follow[:j], ()):
                if not quote_led or t in self.quote_led:
                    out.append(t)
        return out

    def follow_mask(self, state: int, follow: str, quote_led: bool = False) -> torch.Tensor:
        """`follow_ids` as a [V] bool device mask, cached per (state, first max_suffix chars of follow, quote_led)."""
        key = (int(state), (follow or "")[: self.max_suffix], bool(quote_led))
        m = self._follow_cache.get(key)
        if m is None:
            m = torch.zeros(self.V, dtype=torch.bool)
            ids = self.follow_ids(key[0], key[1], key[2])
            if ids:
                m[torch.tensor(ids, dtype=torch.long)] = True
            m = m.to(self.valid.device)
            if len(self._follow_cache) > 4096:
                self._follow_cache.clear()
            self._follow_cache[key] = m
        return m

    def consumed(self, state: int, token: int) -> int:
        """Characters of the following forced text a closing token writes (its suffix length; 0 for a plain close)."""
        e = self.merged[state].get(int(token))
        return len(e[1]) if e is not None else 0

    def mask(self, state: int, count: int, remaining: int, min_len: int, max_len: int,
             follow: str = "") -> torch.Tensor:
        """Allowed next tokens [V] bool for a hole in `state` with `count` decoded chars so far and `remaining` tokens
        of budget (including this one). `follow` = the forced text after the hole: merged closing tokens whose suffix
        is a prefix of it are closers too (protocol 4); with follow "" only the plain closing quote closes."""
        s = state
        v, cl, e, n = self.valid[s], self.closes[s], self.end[s], self.nch[s]
        c1 = count + n
        closer = v & cl
        if follow:
            closer = closer | self.follow_mask(s, follow)
        close_ok = closer & (c1 >= min_len) & (c1 <= max_len)
        need0 = (min_len - c1).clamp(min=0)
        need1 = (min_len - c1 - 1).clamp(min=0)
        r1 = remaining - 1
        cont0 = (e == NORMAL) & (c1 <= max_len) & (need0 + 1 <= r1)
        cont1 = (e == ESC) & (c1 + 1 <= max_len) & (need1 + 2 <= r1)
        return close_ok | (v & ~cl & (cont0 | cont1))

    def advance(self, state: int, count: int, token: int):
        """(new_state, new_count, closed) after appending `token` (which must be allowed). A merged closer closes the
        string and adds its body characters; its suffix length is `consumed(state, token)`."""
        s = state
        e = self.merged[s].get(token)
        if e is not None:
            return NORMAL, count + e[0], True
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

    def mask(self, state: int, count: int, remaining: int, min_len: int, max_len: int,
             follow: str = "") -> torch.Tensor:
        allowed, close = self.dfa.allowed(state, count, remaining, min_len, max_len)
        m = torch.zeros(self.tables.V, dtype=torch.bool, device=self.tables.valid.device)
        ids = [self.tables.char_id[c] for c, _ in allowed]
        if close:
            ids.append(self.tables.quote_id)
            if follow:                                   # quote-led merged closers ('",') before the forced text
                ids += self.tables.follow_ids(NORMAL, follow, quote_led=True)
        if ids:
            m[torch.tensor(ids, device=m.device)] = True
        return m

    def advance(self, state: int, count: int, token: int):
        """(new_state, new_count, closed)."""
        if token == self.tables.quote_id or token in self.tables.quote_led:
            return state, count, True
        c = self.tables.texts[token]
        return self.dfa.trans[state][c], count + 1, False

    def consumed(self, token: int) -> int:
        """Forced characters a closing token writes after its quote (0 for the plain quote token)."""
        suf = self.tables.quote_led.get(int(token))
        return len(suf) if suf else 0


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


# ============================================================================ JSON values (LM-written untyped arguments)
J_NONE, J_OBJ, J_ARR = 0, 1, 2
_WS = " \t\n\r"
_WS_STAY = ("VAL", "VAL_OR_CLOSE", "KEY_OR_CLOSE", "KEY", "COLON", "AFTER")
_NUM_DONE = ("N0", "NI", "NF", "NX1", "NX2")             # complete numbers: a non-number character ends them
_NUM = ("NM", "ND", "NE", "NS") + _NUM_DONE
_LIT = {"t": "rue", "f": "alse", "n": "ull"}
J_LEX = tuple(dict.fromkeys(["VAL", "VAL_OR_CLOSE", "KEY_OR_CLOSE", "KEY", "COLON", "AFTER", "DONE", "SK0", "SK1",
                             "SV0", "SV1", *_NUM] + ["L:" + s[i:] for s in _LIT.values() for i in range(len(s))]))
J_INIT = ("VAL", ())                                       # a JSON hole starts at a top-level value
JSON_FOLLOW_FIRST = ",}]"      # the forced text after a JSON hole starts with one of these (grammar.serialize)


def _jtops(lex: str) -> tuple:
    if lex == "VAL_OR_CLOSE":
        return (J_ARR,)
    if lex in ("KEY_OR_CLOSE", "KEY", "COLON", "SK0", "SK1"):
        return (J_OBJ,)
    if lex == "AFTER":
        return (J_OBJ, J_ARR)
    if lex == "DONE":
        return (J_NONE,)
    return (J_NONE, J_OBJ, J_ARR)


def json_need(lex: str, top: int) -> int:
    """Tokens that always suffice to bring the value from `lex` to a point where the current container (or, at the
    top level, the value) can close, writing one character per token ("0", '"', ":" ...). The total completion need of
    a state is json_need + the number of open containers (one closing bracket each)."""
    if lex == "DONE":
        return 0
    if lex.startswith("L:"):
        return len(lex) - 2
    if top == J_NONE:                                      # top level: a complete number still needs its terminator
        return {"VAL": 2, "SV0": 1, "SV1": 2, "NM": 2, "ND": 2, "NE": 2, "NS": 2}.get(lex, 1)
    return {"VAL": 1, "KEY": 4, "SK0": 3, "SK1": 4, "COLON": 2, "SV0": 1, "SV1": 2, "NM": 1, "ND": 1, "NE": 1,
            "NS": 1}.get(lex, 0)


def json_sim(text: str, lex: str, known: list, bottom: bool, reveal: tuple = ()):
    """Feed `text` into the JSON value automaton (lexical state `lex`, container stack `known` with its top last).
    `bottom`: `known` is the whole stack; otherwise a pop that empties `known` reads the next container type from
    `reveal` (J_NONE = none below). Returns (status, lex, known, bottom, n_reveal, rel, maxrel, pos): status "ok"
    (consumed, value open), "done" (the top-level value completed; text[pos:] is the suffix), "bad", or "more" (the
    token pops below what is known and `reveal` is exhausted); rel / maxrel = net / maximal depth change.

    Grammar: RFC 8259 values; strings as in the JSON-string-body automaton (escapes \\" \\\\ \\/ \\b \\f \\n \\r \\t,
    no \\uXXXX, no raw control characters); numbers -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]{1,2})? (at most two
    exponent digits); whitespace between tokens inside containers, none before or after the top-level value."""
    known = list(known)
    rel = maxrel = nrev = 0
    i, n = 0, len(text)
    while i < n:
        if lex == "DONE":
            return "done", lex, known, bottom, nrev, rel, maxrel, i
        ch = text[i]
        top = known[-1] if known else J_NONE
        value_end = False
        pop = None
        if lex == "SV0" or lex == "SK0":
            if ch == '"':
                if lex == "SK0":
                    lex = "COLON"
                else:
                    value_end = True
            elif ch == "\\":
                lex = lex[:2] + "1"
            elif ord(ch) < 0x20:
                return "bad", lex, known, bottom, nrev, rel, maxrel, i
        elif lex == "SV1" or lex == "SK1":
            if ch not in _ESCAPES:
                return "bad", lex, known, bottom, nrev, rel, maxrel, i
            lex = lex[:2] + "0"
        elif lex[0] == "L":
            if ch != lex[2]:
                return "bad", lex, known, bottom, nrev, rel, maxrel, i
            if len(lex) == 3:
                value_end = True
            else:
                lex = "L:" + lex[3:]
        elif lex in _NUM:
            nxt = None
            if lex == "NM":
                nxt = "N0" if ch == "0" else ("NI" if "1" <= ch <= "9" else None)
                if nxt is None:
                    return "bad", lex, known, bottom, nrev, rel, maxrel, i
            elif lex in ("ND", "NE", "NS"):
                if "0" <= ch <= "9":
                    nxt = "NF" if lex == "ND" else "NX1"
                elif lex == "NE" and ch in "+-":
                    nxt = "NS"
                else:
                    return "bad", lex, known, bottom, nrev, rel, maxrel, i
            elif "0" <= ch <= "9":
                if lex == "N0" or lex == "NX2":
                    return "bad", lex, known, bottom, nrev, rel, maxrel, i
                nxt = {"NI": "NI", "NF": "NF", "NX1": "NX2"}[lex]
            elif ch == "." and lex in ("N0", "NI"):
                nxt = "ND"
            elif ch in "eE" and lex in ("N0", "NI", "NF"):
                nxt = "NE"
            if nxt is None:                                # the number ends here; re-read ch after the value
                lex = "DONE" if (bottom and not known) else "AFTER"
                continue
            lex = nxt
        else:
            if ch in _WS and lex in _WS_STAY and top != J_NONE:
                i += 1
                continue
            if lex == "AFTER":
                if ch == ",":
                    lex = "KEY" if top == J_OBJ else "VAL"
                elif ch == "}" and top == J_OBJ or ch == "]" and top == J_ARR:
                    pop = True
                else:
                    return "bad", lex, known, bottom, nrev, rel, maxrel, i
            elif lex == "COLON":
                if ch != ":":
                    return "bad", lex, known, bottom, nrev, rel, maxrel, i
                lex = "VAL"
            elif lex in ("KEY", "KEY_OR_CLOSE"):
                if ch == '"':
                    lex = "SK0"
                elif ch == "}" and lex == "KEY_OR_CLOSE":
                    pop = True
                else:
                    return "bad", lex, known, bottom, nrev, rel, maxrel, i
            elif lex in ("VAL", "VAL_OR_CLOSE"):
                if ch == "]" and lex == "VAL_OR_CLOSE":
                    pop = True
                elif ch == "{" or ch == "[":
                    known.append(J_OBJ if ch == "{" else J_ARR)
                    rel += 1
                    maxrel = max(maxrel, rel)
                    lex = "KEY_OR_CLOSE" if ch == "{" else "VAL_OR_CLOSE"
                elif ch == '"':
                    lex = "SV0"
                elif ch == "-":
                    lex = "NM"
                elif ch == "0":
                    lex = "N0"
                elif "1" <= ch <= "9":
                    lex = "NI"
                elif ch in _LIT:
                    lex = "L:" + _LIT[ch]
                else:
                    return "bad", lex, known, bottom, nrev, rel, maxrel, i
            else:
                return "bad", lex, known, bottom, nrev, rel, maxrel, i
        if pop:
            known.pop()
            rel -= 1
            if not known and not bottom:
                if nrev >= len(reveal):
                    return "more", lex, known, bottom, nrev, rel, maxrel, i
                r = reveal[nrev]
                nrev += 1
                if r == J_NONE:
                    bottom = True
                else:
                    known.append(r)
            value_end = True
        if value_end:
            lex = "DONE" if (bottom and not known) else "AFTER"
        i += 1
    return ("done" if lex == "DONE" else "ok"), lex, known, bottom, nrev, rel, maxrel, n


def json_reveal(stack: tuple, k: int) -> Optional[tuple]:
    """Container types a token that pops k times uncovers below the top of `stack` (J_NONE = the bottom)."""
    d = len(stack)
    out = []
    for i in range(1, k + 1):
        j = d - 1 - i
        if j < -1:
            return None
        out.append(stack[j] if j >= 0 else J_NONE)
    return tuple(out)


class JsonValueTables:
    """Exact token masks for a *JSON value hole*: the decoder writes one complete JSON value (any type, nested to
    `max_depth`) as text, e.g. the untyped `input` of code.run. A pushdown automaton over (lexical state, container
    stack); the allowed set of a step is a pure function of (state, remaining budget, follow text), so a recorded value
    replays exactly from its token ids.

    Vectorisation: per extended state E = (lexical state, top container type) every token's effect is precomputed -
    `cost` [E, V] = depth change + json_need of the state it ends in (tokens that do not pop below the top), `pops`
    (tokens that pop below the known top, keyed by the container types they uncover) and `closers` (tokens that
    complete the top-level value; their text after the value - the suffix - must be a prefix of the forced JSON text
    after the hole, consumed like a merged string closer). A token is allowed iff it is valid for the actual stack,
    the completion need after it fits the remaining budget (one token per open container plus json_need) and the
    nesting stays <= max_depth. String interiors reuse the JsonStringTables rows; the rest is simulated per token for
    the first characters each state accepts (about 2 s for 152k tokens, once per process). Closers are kept when
    their suffix is empty or starts with one of JSON_FOLLOW_FIRST: the forced text after a JSON hole (", " / "}" /
    "]") always does (HoleFiller.program checks it), so no closer that could match is dropped."""

    BIG = 30000

    def __init__(self, st: JsonStringTables, max_depth: int = 16, max_reveal: int = 3):
        self.st = st
        self.texts = st.texts
        self.V = V = st.V
        self.max_depth = int(max_depth)
        self.max_reveal = int(max_reveal)
        self.E = [(lex, top) for lex in J_LEX if lex != "DONE" for top in _jtops(lex)]
        self.eidx = {e: i for i, e in enumerate(self.E)}
        nE = len(self.E)
        cost = torch.full((nE, V), self.BIG, dtype=torch.int16)
        maxrel = torch.zeros((nE, V), dtype=torch.int8)
        pops: list = [dict() for _ in range(nE)]
        self.closers: list = [[] for _ in range(nE)]           # (token, reveal, suffix, maxrel)
        texts = self.texts
        by_first: dict = {}
        stripped: dict = {}            # whitespace-led tokens: first char after the run -> text after it -> ids
        for t, x in enumerate(texts):
            if not x:
                continue
            by_first.setdefault(x[0], []).append(t)
            if x[0] in _WS:
                rest = x.lstrip(_WS)
                stripped.setdefault(rest[:1], {}).setdefault(rest, []).append(t)
        stripped.setdefault("", {})

        def put(e, t, outs, lead=0):
            for r, (status, lex, known, bottom, _, rel, mr, pos) in outs:
                if status == "done":
                    suf = texts[t][pos + lead:]
                    if not suf or suf[0] in JSON_FOLLOW_FIRST:
                        self.closers[e].append((t, r, suf, mr))
                elif status == "ok":
                    c = rel + json_need(lex, known[-1] if known else J_NONE)
                    if r:
                        pops[e].setdefault(r, []).append((t, c, mr))
                    else:
                        cost[e, t] = c
                        maxrel[e, t] = mr

        for e, (lex, top) in enumerate(self.E):
            if lex[0] == "S":
                self._string_rows(e, lex, top, cost, maxrel, pops, put)
                continue
            firsts = [c for c in by_first if c not in _WS and self._outcomes(c, lex, top)]
            for c in firsts:
                for t in by_first[c]:
                    put(e, t, self._outcomes(texts[t], lex, top))
            ws_lex = lex if lex in _WS_STAY and top != J_NONE else ("AFTER" if lex in _NUM_DONE and top != J_NONE
                                                                    else None)
            if ws_lex is None:                                 # (a top-level value is never followed by whitespace:
                continue                                       #  the follow text of a JSON hole starts with , } ])
            for c in [""] + [c for c in stripped if c and self._outcomes(c, ws_lex, top)]:
                for rest, toks in stripped[c].items():
                    outs = self._outcomes(rest, ws_lex, top)
                    for t in toks:
                        put(e, t, outs, lead=len(texts[t]) - len(rest))
        self.cost_h, self.maxrel_h = cost.numpy(), maxrel.numpy()        # host copies (single-token checks)
        self.pops_h = [{r: {a: (c, m) for a, c, m in v} for r, v in d.items()} for d in pops]
        self.cost = cost.to(st.valid.device)
        self.maxrel = maxrel.to(st.valid.device)
        dev = st.valid.device
        self.pops = [{r: (torch.tensor([a for a, _, _ in v], dtype=torch.long, device=dev),
                          torch.tensor([c for _, c, _ in v], dtype=torch.int32, device=dev),
                          torch.tensor([m for _, _, m in v], dtype=torch.int32, device=dev))
                      for r, v in d.items()} for d in pops]
        self.max_suffix = max([len(s) for cl in self.closers for _, _, s, _ in cl] + [0])
        self._close_cache: dict = {}

    def _outcomes(self, text: str, lex: str, top: int) -> list:
        """[(reveal, json_sim result)] of every valid way `text` can be read from (lex, top), by the uncovered
        container types (depth-first over at most max_reveal reveals)."""
        out, todo = [], [()]
        known = [top] if top != J_NONE else []
        while todo:
            r = todo.pop()
            res = json_sim(text, lex, known, top == J_NONE, r)
            if res[0] == "more":
                if len(r) < self.max_reveal:
                    todo += [r + (x,) for x in (J_NONE, J_OBJ, J_ARR)]
            elif res[0] != "bad":
                out.append((r, res))
        return out

    def _string_rows(self, e, lex, top, cost, maxrel, pops, put):
        st = self.st
        s = NORMAL if lex[2] == "0" else ESC
        key = lex[1] == "K"
        stay = st.valid[s].cpu() & ~st.closes[s].cpu()
        end = st.end[s].cpu().long()
        n0, n1 = json_need(lex[:2] + "0", top), json_need(lex[:2] + "1", top)
        cost[e] = torch.where(stay, torch.where(end == NORMAL, n0, n1), self.BIG).to(torch.int16)
        after = "COLON" if key else ("DONE" if top == J_NONE else "AFTER")
        plain = (st.valid[s].cpu() & st.closes[s].cpu()).nonzero().flatten().tolist()
        for t in plain:                                       # the quote is the token's last character
            if after == "DONE":
                self.closers[e].append((t, (), "", 0))
            else:
                cost[e, t] = json_need(after, top)
        for t, (_, suf) in st.merged[s].items():               # the quote is followed by more text
            if after == "DONE":
                if suf[0] in JSON_FOLLOW_FIRST:
                    self.closers[e].append((t, (), suf, 0))
            else:
                put(e, t, self._outcomes(suf, after, top), lead=len(self.texts[t]) - len(suf))

    # ---------------------------------------------------------------- runtime
    @staticmethod
    def top(state) -> int:
        return state[1][-1] if state[1] else J_NONE

    def need(self, state) -> int:
        return json_need(state[0], self.top(state)) + len(state[1])

    def _closers(self, e: int, stack: tuple, follow: str) -> tuple:
        """(closing token ids as a device tensor or None, the same as a set) for extended state e, stack, follow."""
        d = len(stack)
        rv = json_reveal(stack, min(d, self.max_reveal))
        key = (e, rv, d, (follow or "")[: self.max_suffix])
        hit = self._close_cache.get(key)
        if hit is None:
            lst = [t for t, r, suf, mr in self.closers[e]
                   if (not r or r == rv[: len(r)]) and (follow or "").startswith(suf) and d + mr <= self.max_depth]
            hit = (torch.tensor(lst, dtype=torch.long, device=self.cost.device) if lst else None, set(lst))
            if len(self._close_cache) > 8192:
                self._close_cache.clear()
            self._close_cache[key] = hit
        return hit

    def allowed(self, state, remaining: int, follow: str, token: int) -> tuple:
        """(allowed, closes) of one token - the same conditions as `masks`, evaluated on the host."""
        e = self.eidx[(state[0], self.top(state))]
        stack, d = state[1], len(state[1])
        if token in self._closers(e, stack, follow)[1]:
            return True, True
        if int(self.cost_h[e, token]) + d <= remaining - 1 and int(self.maxrel_h[e, token]) + d <= self.max_depth:
            return True, False
        for k in range(1, min(d, self.max_reveal) + 1):
            ent = self.pops_h[e].get(json_reveal(stack, k))
            if ent is not None and token in ent:
                c, mr = ent[token]
                if c + d <= remaining - 1 and mr + d <= self.max_depth:
                    return True, False
        return False, False

    def masks(self, rows: list, chunk: int = 64) -> tuple:
        """rows: [(state, remaining tokens incl. this one, follow text)] -> (allowed [P, V], closing [P, V])."""
        dev = self.cost.device
        outs = []
        for c0 in range(0, len(rows), chunk):                 # bounded [chunk, V] int16 gathers
            sub = rows[c0: c0 + chunk]
            e = torch.tensor([self.eidx[(st[0], self.top(st))] for st, _, _ in sub], device=dev)
            d = [len(st[1]) for st, _, _ in sub]
            lim = torch.tensor([rem - 1 - dd for (_, rem, _), dd in zip(sub, d)], device=dev, dtype=torch.int16)
            dep = torch.tensor([self.max_depth - dd for dd in d], device=dev, dtype=torch.int8)
            outs.append((self.cost[e] <= lim.unsqueeze(1)) & (self.maxrel[e] <= dep.unsqueeze(1)))
        out = torch.cat(outs) if len(outs) > 1 else outs[0]
        close = torch.zeros_like(out)
        for i, (st, rem, follow) in enumerate(rows):
            ei = self.eidx[(st[0], self.top(st))]
            stack, dd = st[1], len(st[1])
            for k in range(1, min(dd, self.max_reveal) + 1):
                ent = self.pops[ei].get(json_reveal(stack, k))
                if ent is not None:
                    ids, cs, ms = ent
                    ok = (cs + dd <= rem - 1) & (ms + dd <= self.max_depth)
                    out[i, ids[ok]] = True
            if rem >= 1:
                cl = self._closers(ei, stack, follow)[0]
                if cl is not None:
                    close[i, cl] = True
        return out | close, close

    def advance(self, state, token: int):
        """(new state, closed, consumed suffix length) after appending `token` (which must be allowed)."""
        text = self.texts[token]
        status, lex, known, _, _, _, _, pos = json_sim(text or "", state[0], list(state[1]), True)
        if status not in ("ok", "done") or not text:
            raise UnsupportedSchema(f"token {token!r} is not valid in JSON state {state!r}")
        if status == "done":
            return ("DONE", ()), True, len(text) - pos
        return (lex, tuple(known)), False, 0


_JTABLE_CACHE: dict = {}


def json_tables_for(st: JsonStringTables, max_depth: int = 16) -> JsonValueTables:
    """Process-wide cache of JsonValueTables per (string tables, max_depth)."""
    key = (id(st), int(max_depth))
    if key not in _JTABLE_CACHE:
        _JTABLE_CACHE[key] = JsonValueTables(st, max_depth)
    return _JTABLE_CACHE[key]
