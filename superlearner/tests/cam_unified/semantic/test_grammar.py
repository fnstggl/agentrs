"""Constraint automata and the schema walker (Crucible's strict JSON-schema subset)."""
from __future__ import annotations

import json
import random
import re

import pytest
import torch

from superlearner.cam_unified.interfaces import ToolSchema
from superlearner.cam_unified.semantic.constraints import (ESC, NORMAL, PATTERN_ALPHABET, RegexDFA, UnsupportedSchema,
                                                           simulate_json_string, tables_for)
from superlearner.cam_unified.semantic.grammar import (GrammarOptions, NumberGrid, ToolGrammar, drive_walks,
                                                       encode_value, holes_of, materialize, serialize)

CRUCIBLE_PATTERNS = ["[a-z0-9][a-z0-9-]{1,40}", "[a-z][a-z0-9_]{0,23}", "[a-z][a-z0-9-]{2,30}",
                     "[a-z][a-z0-9_-]{0,39}", "[a-z0-9][a-z0-9:._-]{0,30}", "[a-z0-9][a-z0-9._-]{0,60}"]


def _validate(schema, value):
    from crucible.core.schema import validate
    return validate(schema, value)


@pytest.mark.parametrize("pat", CRUCIBLE_PATTERNS + [r"(ab|cd)+x?", r"\d{2,3}-\w+", "a|bc|", r"[^a-c]{1,3}", ".{0,2}z"])
def test_regex_dfa_agrees_with_python_re(pat):
    d = RegexDFA(pat, 64)
    rng = random.Random(0)
    alpha = "abcdxz0123-_:.Az !"
    for _ in range(400):                                 # random strings: DFA acceptance == re.fullmatch
        s = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 8)))
        assert d.matches(s) == bool(re.fullmatch(pat, s)), (pat, s)
    for _ in range(200):                                 # constrained random generation always matches
        s, st = "", 0
        while True:
            al, close = d.allowed(st, len(s), 40 - len(s), 0, 64)
            if close and (not al or rng.random() < 0.25):
                break
            assert al, (pat, s)
            c, st = rng.choice(al)
            s += c
        assert re.fullmatch(pat, s), (pat, s)


def test_regex_dfa_respects_length_bounds_and_budget():
    d = RegexDFA("[a-z]+", 100)
    rng = random.Random(1)
    for _ in range(100):
        s, st = "", 0
        lo, hi = 3, 7
        while True:
            al, close = d.allowed(st, len(s), 9 - len(s), lo, hi)
            if close and (not al or rng.random() < 0.3):
                break
            c, st = rng.choice(al)
            s += c
        assert lo <= len(s) <= hi
    for bad in [r"(a)\1", r"(?=a)a", r"(?i)abc", r"a^b"]:
        with pytest.raises(UnsupportedSchema):
            RegexDFA(bad)


def test_json_string_automaton():
    assert simulate_json_string('abc"', NORMAL) == (True, True, NORMAL, 3)
    assert simulate_json_string('a"b', NORMAL)[0] is False          # cannot continue after the closing quote
    assert simulate_json_string('\\n', NORMAL) == (True, False, NORMAL, 1)
    assert simulate_json_string('\\', NORMAL) == (True, False, ESC, 0)
    assert simulate_json_string('u1234', ESC)[0] is False            # \\u escapes are not generated
    assert simulate_json_string('"x', ESC) == (True, False, NORMAL, 2)
    assert simulate_json_string('a\x01', NORMAL)[0] is False         # raw control characters are invalid


def test_json_string_masks_generate_valid_bounded_strings(enc):
    T = tables_for(enc.tok, enc.vocab_size, enc.device, enc.model_name)
    g = torch.Generator(device=enc.device).manual_seed(0)
    for min_len, max_len, cap in [(0, 10, 12), (5, 40, 30), (1, 200000, 25), (0, 0, 3)]:
        for _ in range(12):
            s, c, toks = NORMAL, 0, []
            while True:
                m = T.mask(s, c, cap - len(toks), min_len, max_len)
                assert bool(m.any())
                logits = torch.randn(T.V, device=enc.device, generator=g)
                t = int(torch.multinomial(torch.softmax(logits.masked_fill(~m, -torch.inf), 0), 1, generator=g))
                toks.append(t)
                s, c, closed = T.advance(s, c, t)
                if closed:
                    break
            text = "".join(T.texts[t] for t in toks)
            value = json.loads('"' + text)
            assert min_len <= len(value) <= max_len and len(toks) <= cap


def test_number_grid_is_inside_bounds_and_exact_for_small_integer_ranges():
    opts = GrammarOptions()
    g = NumberGrid(1, 50, True, opts)
    assert g.exact and g.n == 50
    assert [g.value(i) for i in range(50)] == list(range(1, 51))
    assert all(g.index(v) == v - 1 for v in range(1, 51))
    big = NumberGrid(1, 10**9, True, opts)
    assert not big.exact and big.n == opts.num_base ** opts.num_levels
    vals = [big.value(i) for i in range(big.n)]
    assert min(vals) == 1 and max(vals) == 10**9 and all(isinstance(v, int) for v in vals)
    f = NumberGrid(0.0, 1.0, False, opts)
    vs = [f.value(i) for i in range(0, f.n, 97)]
    assert all(0.0 <= v <= 1.0 for v in vs) and f.value(0) == 0.0 and f.value(f.n - 1) == 1.0
    for level in range(opts.num_levels):                    # digit masks keep the index inside the grid
        assert any(g.digit_mask(level, 0))


def _random_fill(template, rng):
    for h in holes_of(template):
        if h.kind == "pattern":
            s, st = "", 0
            while True:
                al, close = h.dfa.allowed(st, len(s), 60 - len(s), h.min_len, h.max_len)
                if close and (not al or rng.random() < 0.3):
                    break
                c, st = rng.choice(al)
                s += c
            h.value = s
        else:
            n = rng.randint(h.min_len, min(h.max_len, h.min_len + 6))
            h.value = "".join(rng.choice("ab c\"\\\n€") for _ in range(n))
    return materialize(template)


def test_every_ow_tiny_schema_compiles_and_random_walks_are_valid(ow_tiny_tools):
    opts = GrammarOptions()
    assert len(ow_tiny_tools) >= 140
    rng = random.Random(0)
    n_args = 0
    for t in ow_tiny_tools:
        g = ToolGrammar(t, opts)
        assert g.generable, (t.name, g.error)
        for _ in range(25):
            tpl = drive_walks([g.root], opts,
                              lambda rows, decs: [rng.choice([i for i, m in enumerate(d.mask) if m]) for d in decs])[0]
            args = _random_fill(tpl, rng)
            assert _validate(t.parameters, args) == [], (t.name, args)
            # the serialised segments are valid JSON once the holes are filled (free-map keys repeating a declared
            # property or an earlier key are dropped by materialize, first occurrence wins)
            text = "".join(s if isinstance(s, str) else json.dumps(s.value, ensure_ascii=False)[1:]
                           for s in serialize(tpl))
            stats = {}
            materialize(tpl, stats)
            if not stats.get("dropped_keys"):
                assert json.loads(text) == args
            n_args += 1
    assert n_args == 25 * len(ow_tiny_tools)


def test_encode_value_inverts_the_walk(ow_tiny_tools):
    opts = GrammarOptions()
    rng = random.Random(3)
    for t in ow_tiny_tools[::3]:
        g = ToolGrammar(t, opts)
        rec = []

        def decide(rows, decs):
            cs = [rng.choice([i for i, m in enumerate(d.mask) if m]) for d in decs]
            rec.extend(cs)
            return cs
        tpl = drive_walks([g.root], opts, decide)[0]
        stats = {}
        _random_fill(tpl, rng)
        args = materialize(tpl, stats)
        if stats.get("dropped_keys"):
            continue
        choices, holes, flags = [], [], {}
        encode_value(g.root, args, opts, choices, holes, flags)
        assert not flags.get("approx"), t.name             # generated values always lie on the grid
        it = iter(choices)
        tpl2 = drive_walks([g.root], opts, lambda rows, decs: [next(it) for _ in decs])[0]
        assert next(it, None) is None
        for h, (_, v) in zip(holes_of(tpl2), holes):
            h.value = v
        assert materialize(tpl2) == args, t.name


def test_unsupported_schemas_are_reported_not_generated():
    opts = GrammarOptions()
    bad = [ToolSchema("x", "", {"type": "object", "properties": {"a": {"type": "string", "pattern": r"(a)\1"}},
                                "required": ["a"]}),
           ToolSchema("y", "", {"type": "object", "properties": {"a": {"type": "array", "minItems": 50}},
                                "required": ["a"]}),
           ToolSchema("z", "", {"type": "object", "properties": {"a": {"type": "integer", "minimum": 5,
                                                                         "maximum": 1}}, "required": ["a"]})]
    for t in bad:
        g = ToolGrammar(t, opts)
        assert not g.generable and g.error
    # an unsupported OPTIONAL property is dropped (logged on the node); the tool stays generable
    ok = ToolGrammar(ToolSchema("w", "", {"type": "object", "properties": {"a": {"type": "string", "pattern": r"(a)\1"},
                                                                            "b": {"type": "boolean"}}}), opts)
    assert ok.generable and ok.root.dropped == ["a"]
