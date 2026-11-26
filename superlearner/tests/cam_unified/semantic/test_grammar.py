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
from superlearner.cam_unified.semantic.grammar import (DecimalSpec, GrammarOptions, NumberGrid, ToolGrammar,
                                                       drive_walks, encode_value, holes_of, materialize,
                                                       number_candidates, serialize)

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


def test_number_grid_is_exact_for_small_integer_ranges():
    opts = GrammarOptions()
    g = NumberGrid(1, 50, opts)
    assert g.n == 50
    assert [g.value(i) for i in range(50)] == list(range(1, 51))
    assert all(g.index(v) == v - 1 for v in range(1, 51))
    assert not NumberGrid.fits(1, 10 ** 9, opts)
    for level in range(opts.num_levels):                    # digit masks keep the index inside the grid
        assert any(g.digit_mask(level, 0))


def _random_decimal(spec, rng):
    k = rng.choice([i for i, m in enumerate(spec.k_mask()) if m]) if spec.F else 0
    s = rng.choice([i for i, m in enumerate(spec.sign_mask(k)) if m])
    L = 1 + rng.choice([i for i, m in enumerate(spec.len_mask(k, s)) if m])
    n = 0
    for pos in range(L):
        d = rng.choice([i for i, m in enumerate(spec.digit_mask(k, s, L, pos, n)) if m])
        n = n * 10 + d
    return (k, s, n), spec.value(k, s, n)


@pytest.mark.parametrize("lo,hi,integer", [(1, 10 ** 12, True), (-10 ** 9, 10 ** 9, True), (0.1, 2.5, False),
                                           (-3.25, 1e3, False), (0.0, 1.0, False), (1, 1e12, False), (7, 7, True),
                                           (-1e15, 1e15, False)])
def test_decimal_digits_stay_in_bounds_and_have_one_path_per_value(lo, hi, integer):
    opts = GrammarOptions()
    spec = DecimalSpec(lo, hi, integer, opts)
    rng = random.Random(0)
    for _ in range(400):                                       # random admissible walks: in bounds, unique encoding
        path, v = _random_decimal(spec, rng)
        assert lo <= v <= hi and isinstance(v, int if integer else float)
        k, s, n, approx = spec.encode(v)
        assert (k, s, n) == path and not approx


def test_exact_amounts_are_representable_in_wide_ranges():
    """Exact cents and timestamps over wide ranges (the old 4096-point symlog grid could not write them)."""
    opts = GrammarOptions()
    cents = ToolGrammar(ToolSchema("pay", "", {"type": "object", "properties": {
        "amount": {"type": "integer", "minimum": 1, "maximum": 10 ** 12},
        "price": {"type": "number", "minimum": 0},
        "at": {"type": "integer", "minimum": 0, "maximum": 10 ** 9}}, "required": ["amount", "price", "at"]}), opts)
    rng = random.Random(4)
    vals = [1, 2, 10 ** 12, 10 ** 12 - 1, 123456789012, 4999, 1000001] + [rng.randint(1, 10 ** 12) for _ in range(40)]
    prices = [0.0, 0.01, 12.5, 1999.99, 3.141593, 1e9, 123456.789] + [round(rng.uniform(0, 1e6), 2) for _ in range(40)]
    for a, p in zip(vals, prices):
        args = {"amount": a, "price": p, "at": rng.randint(0, 10 ** 9)}
        choices, holes, flags = [], [], {}
        encode_value(cents.root, args, opts, choices, holes, flags)
        assert not flags.get("approx")
        it = iter(choices)
        tpl = drive_walks([cents.root], opts, lambda rows, decs: [next(it) for _ in decs])[0]
        assert next(it, None) is None
        out = materialize(tpl)
        assert out == args and type(out["amount"]) is int and _validate(cents.tool.parameters, out) == []
    # more than float_digits fractional digits round (flagged, never silent)
    flags = {}
    encode_value(cents.root, {"amount": 5, "price": 0.1234567, "at": 0}, opts, [], [], flags)
    assert flags["approx"] == 1


def test_numbers_copy_exactly_from_evidence():
    opts = GrammarOptions()
    g = ToolGrammar(ToolSchema("bank.transfer", "", {"type": "object", "properties": {
        "amount": {"type": "integer", "minimum": 1, "maximum": 10 ** 12},
        "rate": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["amount", "rate"]}), opts)
    cands = number_candidates([([(0.035, "interest_rate = 0.035 fraction")],
                                "invoice inv_9 from acct_7: amount 1,234,567 cents, due day 12; ref 88")], 24)
    texts = [c.text for c in cands]
    assert texts[0] == "0.035" and "1,234,567" in texts and "12" in texts and "88" in texts
    assert "9" not in texts and "7" not in texts                 # digits inside identifiers are not numbers
    seen_copy = set()
    rng = random.Random(1)
    for _ in range(60):
        rec = []

        def decide(rows, decs):
            out = []
            for d in decs:
                ok = [i for i, m in enumerate(d.mask) if m]
                c = rng.choice(ok)
                if d.kind == "numsrc":
                    assert ok[0] == 0 and all(d.values[i - 1].admissible(d.node) for i in ok[1:])
                    seen_copy.add((d.node.path, c))
                out.append(c)
            rec.extend(out)
            return out
        tpl = drive_walks([g.root], opts, decide, [cands])[0]
        args = materialize(tpl)
        assert _validate(g.tool.parameters, args) == []
        choices, flags = [], {}
        encode_value(g.root, args, opts, choices, [], flags, cands)
        it = iter(choices)                                        # the encoding replays to the same arguments
        assert materialize(drive_walks([g.root], opts, lambda r, d: [next(it) for _ in d], [cands])[0]) == args
    assert any(p == ".amount" and c > 0 for p, c in seen_copy) and any(p == ".rate" and c > 0 for p, c in seen_copy)
    flags = {}
    choices = []
    encode_value(g.root, {"amount": 1234567, "rate": 0.035}, opts, choices, [], flags, cands)
    assert flags["copied"] == 2                                   # demonstrations copy when the number is shown
    it = iter(choices)
    assert materialize(drive_walks([g.root], opts, lambda r, d: [next(it) for _ in d], [cands])[0]) == \
        {"amount": 1234567, "rate": 0.035}


_JSON_VALUES = [None, True, 0, -12, 3.25, "a\"b", [], {}, [1, {"k": [None, "x"]}], {"rows": [{"src": "b1", "n": 2}]}]


def _random_fill(template, rng):
    for h in holes_of(template):
        if h.kind == "json":                                  # JSON-value hole (untyped schema): any JSON value
            h.value = rng.choice(_JSON_VALUES)
            h.tokens = [0]                                    # marks the hole filled (a JSON null is a value)
        elif h.kind == "pattern":
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
            text = "".join(s if isinstance(s, str) else
                           json.dumps(s.value, ensure_ascii=False)[0 if s.kind == "json" else 1:]
                           for s in serialize(tpl))
            stats = {}
            materialize(tpl, stats)
            if not stats.get("dropped_keys"):
                assert json.loads(text) == args
            n_args += 1
    assert n_args == 25 * len(ow_tiny_tools)


def _numeric_nodes(node, out):
    if node.kind in ("integer", "number"):
        out.append(node)
    for _, c, _ in node.props:
        _numeric_nodes(c, out)
    for c in [node.extra, node.item] + list(node.options.values()):
        if c is not None:
            _numeric_nodes(c, out)
    return out


def test_every_ow_tiny_numeric_slot_writes_exact_values_and_reductions_are_listed(ow_tiny_tools):
    """All numeric slots of the real catalogue write exact values at their bounds and at random points inside (money
    in cents, timestamps, budgets); every capability reduction is listed per tool. Untyped values written by the
    typed heads (json_holes=False) keep their numeric slots; with JSON-value holes the decoder writes them."""
    opts = GrammarOptions(json_holes=False)
    rng = random.Random(7)
    n_slots, kinds = 0, {}
    for t in ow_tiny_tools:
        g = ToolGrammar(t, opts)
        for r in g.reductions:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        for node in _numeric_nodes(g.root, []):
            spec = node.num
            n_slots += 1
            lo, hi = spec.lo, spec.hi
            if node.kind == "integer":
                vals = [lo, hi, (lo + hi) // 2] + [rng.randint(lo, hi) for _ in range(10)]
            else:
                vals = [float(lo), float(hi)] + [round(rng.uniform(max(lo, -1e6), min(hi, 1e6)), 2) for _ in range(10)]
            for v in vals:
                ch, fl = [], {}
                encode_value(node, v, opts, ch, [], fl)
                it = iter(ch)
                gen = drive_walks([node], opts, lambda r, d: [next(it) for _ in d])[0]
                assert gen == v and not fl.get("approx"), (t.name, node.path, v, gen)
    assert n_slots >= 90
    # the caps the verifier listed are surfaced: arrays capped below maxItems, free maps, defaulted bounds
    red = {(t.name, r["path"], r["kind"]) for t in ow_tiny_tools for r in ToolGrammar(t, opts).reductions}
    for name, path in [("mail.send", ".to"), ("ads.create_campaign", ".keywords"), ("cal.create", ".attendees"),
                       ("contracts.propose", ".clauses"), ("issues.create", ".labels")]:
        assert (name, path, "array_cap") in red
    assert ("cal.create", ".start", "default_bound") in red and ("bank.invoice", ".due_in", "default_bound") in red
    assert kinds.get("map_cap", 0) >= 1 and kinds.get("string_cap", 0) >= 1
    print("ow_tiny numeric slots:", n_slots, "reductions:", kinds)


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
