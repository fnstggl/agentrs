"""JSON-value holes: untyped arguments (schema {} / no type, e.g. code.run `input`) are written by the decoder as one
JSON value under the exact pushdown automaton constraints.JsonValueTables (grammar.GrammarOptions.json_holes)."""
from __future__ import annotations

import json
import random

import pytest
import torch

from superlearner.cam_unified.interfaces import ToolSchema
from superlearner.cam_unified.semantic.constraints import (J_INIT, J_NONE, J_OBJ, JsonStringTables, JsonValueTables,
                                                           json_sim, token_texts)
from superlearner.cam_unified.semantic.grammar import (GrammarError, GrammarOptions, HoleFiller, ToolGrammar,
                                                       drive_walks, encode_value, holes_of, materialize, serialize)

CODE_RUN = ToolSchema("code.run", "Execute a function from a repository in the sandbox.", {
    "type": "object", "properties": {
        "repo": {"type": "string", "maxLength": 81}, "target": {"type": "string", "maxLength": 160},
        "input": {"description": "JSON input passed to the function"}, "ref": {"type": "string", "maxLength": 40},
        "timeout_s": {"type": "number", "minimum": 0.1, "maximum": 30}},
    "additionalProperties": False, "required": ["repo", "target"]})
# shaped like the ow_tiny ExampleInductor code.run demonstrations: 5 levels, > 8 items, > 4 keys
DEMO_INPUT = {"rows": [{"src": "bas1", "data": [["lekcin", "y", "'garnet'"], ["nehcil", "b", "-0x1cb2"],
                                                ["radec", "n", "1892.4%"], ["tunlaw", "z", "2014-05-18"]]},
                       {"src": "del7", "data": "f_Nickel|v=bramble velvet^f_Lichen|c=1440449000m^f_Cedar|f=25.872"}]
              + [{"src": f"s{i}", "data": [[i, -i * 1.5, None, True, "é\"\\\n"]]} for i in range(9)],
              "k1": 1, "k2": 2.5e-3, "k3": [], "k4": {}, "k5": False}


@pytest.fixture(scope="module")
def qtok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen3-0.6B tokenizer not cached: {e}")


@pytest.fixture(scope="module")
def jt(qtok):
    st = JsonStringTables(token_texts(qtok, 151936), "cpu")
    return JsonValueTables(st), JsonValueTables(st, max_depth=3)


def _random_value(rng, d=0):
    k = rng.random()
    if d < 5 and k < 0.25:
        return {rng.choice(["a", "src", "data", "x y", "é", 'q"']): _random_value(rng, d + 1)
                for _ in range(rng.randint(0, 4))}
    if d < 5 and k < 0.5:
        return [_random_value(rng, d + 1) for _ in range(rng.randint(0, 5))]
    return rng.choice([0, -12, 3.25, 1e-5, -0.5e+12, 12345678901234, True, False, None, "abc", 'q"uo\\te\n', "",
                       "f_Nickel|v=bramble", "2014-05-18"])


def test_the_automaton_accepts_json_and_rejects_everything_else():
    def complete(text):                     # one complete value, then the forced '}' as the suffix
        st = json_sim(text + "}", "VAL", [], True)
        return st[0] == "done" and (text + "}")[st[-1]:] == "}"
    for good in ['{"a": [1, 2.5e-3, true, null], "b": {}}', "[]", '"x\\n"', "-0.25", "1e+12", "false", "[ 1 ,\n2 ]"]:
        assert complete(good), good
        json.loads(good)
    for bad in ["01", "1e123", "[1,]", '{"a" 1}', "{,}", "tru e", '"a\x01"', '"\\u0041"', " 1", "[1]]", "{1: 2}",
                "+1", ".5", "1.", "nul l", "[1 2]", "1 "]:
        assert not complete(bad), bad
    # a top-level number ends at the follow text's first character (consumed like a merged string closer)
    st = json_sim("42}", "VAL", [], True)
    assert st[0] == "done" and "42}"[st[-1]:] == "}"


def test_natural_tokenisations_of_random_values_replay_under_the_masks(qtok, jt):
    J, _ = jt
    rng = random.Random(0)
    n_merged = 0
    for _ in range(200):
        v = _random_value(rng)
        text = json.dumps(v, ensure_ascii=False)
        follow = rng.choice(["}", ', "ref": "', "]", ", "])
        state, out, closed = J_INIT, [], False
        for u, t in enumerate(qtok(text + follow, add_special_tokens=False)["input_ids"]):
            m, c = J.masks([(state, 1536 - u, follow)])
            assert bool(m[0, t]), (text, follow, u, J.texts[t], state)
            assert J.allowed(state, 1536 - u, follow, t) == (True, bool(c[0, t]))     # host check == device mask
            state, closed, cons = J.advance(state, t)
            out.append(t)
            if closed:
                body = "".join(J.texts[x] for x in out)
                assert body[: len(body) - cons] == text and follow.startswith(body[len(body) - cons:])
                n_merged += cons > 0
                break
        assert closed, text
    assert n_merged >= 50                       # '}}' '"],' ... carry the value's end into the forced text


def test_sampling_under_the_masks_always_completes_within_budget_and_depth(qtok, jt):
    """Random logits (an adversarial 'LM'): every sampled value parses, closes inside its budget and nesting cap."""
    J, J3 = jt
    g = torch.Generator().manual_seed(0)
    for i in range(60):
        tab = J3 if i % 2 else J
        cap = [3, 6, 12, 40][i % 4]
        follow = ["}", ", ", "]"][i % 3]
        state, text, closed = J_INIT, "", False
        for u in range(cap):
            m, c = tab.masks([(state, cap - u, follow)])
            assert bool(m.any()), (i, state, cap - u)
            logits = torch.randn(m.shape[1], generator=g) * 3
            t = int(torch.multinomial(torch.softmax(logits.masked_fill(~m[0], -torch.inf), -1), 1, generator=g))
            state, closed, cons = tab.advance(state, t)
            text += tab.texts[t]
            if closed:
                assert bool(c[0, t])
                break
            assert tab.need(state) <= cap - u - 1
        assert closed, (i, text)
        body = text[: len(text) - cons]
        v = json.loads(body)
        depth = lambda x: 1 + max([depth(y) for y in (x.values() if isinstance(x, dict) else x)] + [0]) \
            if isinstance(x, (dict, list)) else 0
        assert depth(v) <= tab.max_depth and follow.startswith(text[len(text) - cons:])


def test_untyped_arguments_compile_to_json_holes_and_encode_demonstrations():
    opts = GrammarOptions(max_string_tokens=1536, max_action_tokens=3072)
    g = ToolGrammar(CODE_RUN, opts)
    assert g.generable and any(r["kind"] == "json_hole" and r["path"] == ".input" for r in g.reductions)
    choices, holes, flags = [], [], {}
    args = {"repo": "zovo-15-labs/zovo-15-decode", "target": "service:solve", "input": DEMO_INPUT}
    encode_value(g.root, args, opts, choices, holes, flags)
    assert [k for k, _ in holes] == ["str", "str", "json"] and holes[2][1] == DEMO_INPUT
    # the typed heads cannot write it at any untyped depth (the reason for JSON holes)
    for depth in (2, 4):
        typed_opts = GrammarOptions(json_holes=False, untyped_depth=depth)
        with pytest.raises(GrammarError):
            encode_value(ToolGrammar(CODE_RUN, typed_opts).root, args, typed_opts, [], [], {})
    with pytest.raises(GrammarError):
        encode_value(g.root, {**args, "input": float("nan")}, opts, [], [], {})
    it = iter(choices)
    tpl = drive_walks([g.root], opts, lambda rows, decs: [next(it) for _ in decs])[0]
    segs = serialize(tpl)
    j = next(i for i, s in enumerate(segs) if not isinstance(s, str) and s.kind == "json")
    assert segs[j - 1].endswith('"input": ') and segs[j + 1].startswith("}")     # no quotes around the value


@pytest.fixture(scope="module")
def filler(enc):
    from superlearner.cam_unified.semantic.constraints import tables_for
    opts = GrammarOptions(max_string_tokens=1536, max_action_tokens=3072, length_prior=False)
    f = HoleFiller(enc, tables_for(enc.tok, enc.vocab_size, enc.device, enc.model_name), opts)
    f.merged_acting = False
    return f, opts


def _code_run_program(f, opts, value):
    g = ToolGrammar(CODE_RUN, opts)
    args = {"repo": "zovo-15-labs/zovo-15-decode", "target": "service:solve", "input": value}
    choices, holes = [], []
    encode_value(g.root, args, opts, choices, holes, {})
    it = iter(choices)
    tpl = drive_walks([g.root], opts, lambda rows, decs: [next(it) for _ in decs])[0]
    return tpl, f.program(tpl), holes, args


def _ctx(f, n):
    prompt = ("Write the JSON arguments for one call of the tool below. Reply with the JSON object only.\nTool: code.run"
              "\nDescription: Execute a function from a repository in the sandbox.\nArguments schema: "
              + json.dumps(CODE_RUN.parameters))
    return [f.context_ids(prompt)[0] for _ in range(n)]


def test_demonstrated_json_values_tokenise_and_replay_exactly(enc, filler):
    f, opts = filler
    vals = [DEMO_INPUT, 42, -0.5, "text", None, [], {"a": [1, {"b": None}]}]
    progs, tpls = [], []
    for v in vals:
        tpl, p, holes, args = _code_run_program(f, opts, v)
        for k, (h, (kind, x)) in enumerate(zip(p.holes, holes)):
            h.tokens = f.tokenize_hole(kind, x, h, p.cap[k])
        progs.append(p)
        tpls.append((tpl, args))
    prefix = torch.zeros(len(vals), 0, enc.sem_dim, device=enc.device)
    ctx = _ctx(f, len(vals))
    lp1, _, nt = f.score(progs, prefix, ctx, 1.0, exact_rows=True)
    lp2, _, _ = f.score(progs[::-1], prefix, ctx, 1.0, exact_rows=True)
    assert torch.equal(lp1, lp2.flip(0)) and bool(torch.isfinite(lp1).all())
    for (tpl, args), p in zip(tpls, progs):
        assert materialize(tpl) == args                                   # the parsed value, nulls included
    jh = progs[0].holes[2]
    assert jh.kind == "json" and len(jh.tokens) > 300 and int(nt[0]) > 300
    assert progs[1].holes[2].consumed == 1                                # 42 ends at the forced '}'
    print(f"\ncode.run demo input: {len(jh.tokens)} tokens, logp/token {float(lp1[0]) / int(nt[0]):.3f}")


def test_sampled_json_values_are_valid_and_rescore_bit_for_bit(enc):
    """The untrained decoder (tempered sampling, a short cap) writes code.run inputs that always parse and validate."""
    from superlearner.cam_unified.semantic.constraints import tables_for
    opts = GrammarOptions(max_string_tokens=64, max_action_tokens=256, length_prior=False)
    f = HoleFiller(enc, tables_for(enc.tok, enc.vocab_size, enc.device, enc.model_name), opts)
    f.merged_acting = False
    R = 4
    progs, tpls = [], []
    for _ in range(R):
        tpl, p, _, _ = _code_run_program(f, opts, {"x": 1})       # the template: repo, target, input present
        progs.append(p)
        tpls.append(tpl)
    prefix = torch.zeros(R, 0, enc.sem_dim, device=enc.device)
    ctx = _ctx(f, R)
    g = torch.Generator(device=enc.device).manual_seed(0)
    f.sample(progs, prefix, ctx, 1.0, False, g)
    from crucible.core.schema import validate
    for tpl, p in zip(tpls, progs):
        args = materialize(tpl)
        assert validate(CODE_RUN.parameters, args) == []
        assert p.holes[2].kind == "json" and p.holes[2].tokens
    s1 = f.score(progs, prefix, ctx, 1.0, exact_rows=True)[0]
    s2 = f.score(progs, prefix, ctx, 1.0, exact_rows=True)[0]
    assert torch.equal(s1, s2) and bool(torch.isfinite(s1).all())
