"""String protocol 4: boundary-aware closing of JSON string holes (constraints.split_close / JsonStringTables merged
closers, grammar.HoleFiller consumption of forced characters, exact replay, refusal of other-protocol records)."""
from __future__ import annotations

import json

import pytest
import torch

from superlearner.cam_unified.interfaces import ToolSchema
from superlearner.cam_unified.semantic.constraints import (ESC, NORMAL, STRING_PROTOCOL, JsonStringTables,
                                                           simulate_json_string, split_close, token_texts)
from superlearner.cam_unified.semantic.grammar import (GrammarError, GrammarOptions, HoleFiller, ToolGrammar,
                                                       drive_walks, materialize)

SCHEMA = {"type": "object", "properties": {
    "handle": {"type": "string", "maxLength": 40, "description": "short unique handle"},
    "name": {"type": "string", "maxLength": 80},
    "tags": {"type": "array", "items": {"type": "string", "maxLength": 20}, "maxItems": 3},
    "note": {"type": "string", "maxLength": 200}},
    "required": ["handle", "name", "tags", "note"], "additionalProperties": False}


@pytest.fixture(scope="module")
def qtok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen3-0.6B tokenizer not cached: {e}")


@pytest.fixture(scope="module")
def tables(qtok):
    return JsonStringTables(token_texts(qtok, 151936), "cpu")


def _id(qtok, text):
    ids = qtok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, (text, ids)
    return ids[0]


def test_split_close_finds_the_first_unescaped_quote():
    assert split_close('",', NORMAL) == (True, 0, ",")
    assert split_close('abc"}', NORMAL) == (True, 3, "}")
    assert split_close('abc"', NORMAL) == (True, 3, "")
    assert split_close('\\""', NORMAL) == (True, 1, "")           # escaped quote, then the closing one
    assert split_close('"x', ESC) == (False, 2, None)             # \" inside the body never closes
    assert split_close('a\x01"', NORMAL)[0] is False               # raw control characters are invalid
    assert split_close("abc", NORMAL)[0] is False
    # the end-of-token automaton is unchanged: a quote followed by text is not a plain close
    assert simulate_json_string('",', NORMAL)[0] is False


def test_merged_closers_are_allowed_only_before_matching_forced_text(qtok, tables):
    T = tables
    comma, brace, quote = _id(qtok, '",'), _id(qtok, '"}'), _id(qtok, '"')
    assert T.merged[NORMAL][comma] == (0, ",") and T.merged[NORMAL][brace] == (0, "}")
    assert comma in T.quote_led and brace in T.quote_led
    m_next = T.mask(NORMAL, 3, 20, 0, 100, follow=', "name": "')
    m_last = T.mask(NORMAL, 3, 20, 0, 100, follow="}")
    m_old = T.mask(NORMAL, 3, 20, 0, 100)
    assert bool(m_next[comma]) and not bool(m_next[brace])
    assert bool(m_last[brace]) and not bool(m_last[comma])
    assert not bool(m_old[comma]) and not bool(m_old[brace])           # follow "" = the pre-v4 closing set
    assert bool(m_next[quote]) and bool(m_last[quote]) and bool(m_old[quote])   # the plain quote always closes
    assert T.advance(NORMAL, 3, comma) == (NORMAL, 3, True) and T.consumed(NORMAL, comma) == 1
    assert T.consumed(NORMAL, quote) == 0
    # length bounds apply to the body characters before the quote of a merged closer
    dot_comma = next(t for t, (n, suf) in T.merged[NORMAL].items() if T.texts[t] == '.",')
    assert bool(T.mask(NORMAL, 4, 20, 5, 100, follow=", ")[dot_comma])      # 4 + 1 = 5 >= min_len
    assert not bool(T.mask(NORMAL, 3, 20, 5, 100, follow=", ")[dot_comma])  # 3 + 1 < 5
    assert not bool(T.mask(NORMAL, 10, 20, 0, 10, follow=", ")[dot_comma])  # 10 + 1 > max_len


def test_merged_closers_cover_most_of_the_natural_closing_vocabulary(tables):
    T = tables
    ids = T.follow_ids(NORMAL, ', "subject": "')
    assert len(ids) >= 20 and all(T.texts[i].split('"', 1)[1] in (",", ", ") or T.texts[i].endswith('",')
                                  for i in ids)
    assert T.max_suffix >= 2 and STRING_PROTOCOL == 4


# ---------------------------------------------------------------------------------------------- with the decoder
def _program(filler, opts, choices):
    g = ToolGrammar(ToolSchema("org.create", "Create an organisation.", SCHEMA), opts)
    it = iter(choices)
    tpl = drive_walks([g.root], opts, lambda rows, decs: [next(it) for _ in decs])[0]
    return tpl, filler.program(tpl)


@pytest.fixture(scope="module")
def filler(enc):
    from superlearner.cam_unified.semantic.constraints import tables_for
    opts = GrammarOptions(max_string_tokens=48, max_action_tokens=192, length_prior=False)
    f = HoleFiller(enc, tables_for(enc.tok, enc.vocab_size, enc.device, enc.model_name), opts)
    f.merged_acting = False
    return f, opts


def _ctx(f, n):
    prompt = ("Write the JSON arguments for one call of the tool below. Reply with the JSON object only.\nTool: "
              "org.create\nDescription: Create an organisation for a bakery.\nArguments schema: " + json.dumps(SCHEMA))
    return [f.context_ids(prompt)[0] for _ in range(n)]


def _decoded_text(f, p):
    """The exact text the decoder fed / produced after the stub: forced segments (as re-tokenised after merged
    closers) and hole tokens."""
    T = f.tables
    out = "".join(T.texts[t] for t in p.forced[0])
    for k, h in enumerate(p.holes):
        out += "".join(T.texts[t] for t in h.tokens)
        if k + 1 < len(p.holes):
            out += "".join(T.texts[t] for t in f.segment(p, k + 1))
    return out + p.forced_text[-1][p.holes[-1].consumed:]


def test_sampled_strings_close_naturally_and_the_token_stream_is_the_json(enc, filler):
    f, opts = filler
    R = 4
    tpls, progs = zip(*[_program(f, opts, [2]) for _ in range(R)])      # tags: 2 items
    prefix = torch.zeros(R, 0, enc.sem_dim, device=enc.device)
    ctx = _ctx(f, R)
    g = torch.Generator(device=enc.device).manual_seed(0)
    f.sample(list(progs), prefix, ctx, 1.0, True, g)                     # greedy
    merged = own = n = 0
    for tpl, p in zip(tpls, progs):
        args = materialize(tpl)
        assert json.loads(_decoded_text(f, p)) == args                    # consumed characters are the forced JSON
        for h in p.holes:
            n += 1
            merged += h.consumed > 0
            own += bool(h.own_close)
            assert not h.value.rstrip().endswith(",")
    assert merged >= n // 2 and own == n
    # exact replay of the sampled records: bit-identical, also row-wise in another batch composition
    s1 = f.score(list(progs), prefix, ctx, 1.0, exact_rows=True)[0]
    s2 = f.score(list(progs[::-1]), prefix, ctx, 1.0, exact_rows=True)[0]
    assert torch.equal(s1, s2.flip(0)) and bool(torch.isfinite(s1).all())


def test_replay_checks_the_recorded_consumption_and_protocol(enc, filler):
    f, opts = filler
    _, p = _program(f, opts, [1])
    prefix = torch.zeros(1, 0, enc.sem_dim, device=enc.device)
    ctx = _ctx(f, 1)
    f.sample([p], prefix, ctx, 1.0, True)
    rec = [list(h.tokens) for h in p.holes]
    consumed = p.consumed()
    _, q = _program(f, opts, [1])                                        # a fresh program replays the record
    for h, tk, c in zip(q.holes, rec, consumed):
        h.tokens, h.expect_consumed = list(tk), c
    a = f.score([p], prefix, ctx, 1.0)[0]
    b = f.score([q], prefix, ctx, 1.0)[0]
    assert torch.equal(a, b) and q.consumed() == consumed
    k = next(i for i, c in enumerate(consumed) if c > 0)
    q.holes[k].expect_consumed = consumed[k] + 1
    with pytest.raises(GrammarError):
        f.score([q], prefix, ctx, 1.0)
    # the prompt hash covers the string protocol: a protocol-3 record never matches a protocol-4 prompt
    h4 = f.prompt_hash(ctx[0])
    f.protocol = 3
    try:
        assert f.prompt_hash(ctx[0]) != h4
        _, p3 = _program(f, opts, [1])
        assert all(h.follow == "" for h in p3.holes)                     # protocol 3: no merged closers
    finally:
        f.protocol = STRING_PROTOCOL


def test_demonstrated_strings_use_the_tokenizers_own_closing_tokens(enc, filler):
    f, opts = filler
    tpl, p = _program(f, opts, [2])
    vals = ["bakery-labs", "Bakery Labs", "bread", "cake", "We bake bread."]
    for k, (h, v) in enumerate(zip(p.holes, vals)):
        h.tokens = f.tokenize_hole("str", v, h, p.cap[k])
    lp = f.score([p], torch.zeros(1, 0, enc.sem_dim, device=enc.device), _ctx(f, 1), 1.0)[0]
    assert bool(torch.isfinite(lp).all())
    assert [h.value for h in p.holes] == vals
    assert sum(h.consumed > 0 for h in p.holes) >= 3                     # '",' / '"],' / '"}' closers
    natural = enc.tok('"' + json.dumps(dict(zip(["handle", "name"], vals[:2])))[11:], add_special_tokens=False)
    assert natural["input_ids"]                                          # (tokenizer sanity)


def test_pattern_holes_close_with_quote_led_merged_tokens(enc, qtok):
    from superlearner.cam_unified.semantic.constraints import tables_for
    sch = {"type": "object", "properties": {"id": {"type": "string", "pattern": "[a-z]{2,6}"},
                                            "x": {"type": "integer", "minimum": 0, "maximum": 3}},
           "required": ["id", "x"]}
    opts = GrammarOptions(max_string_tokens=16, max_action_tokens=64, length_prior=False)
    f = HoleFiller(enc, tables_for(enc.tok, enc.vocab_size, enc.device, enc.model_name), opts)
    g = ToolGrammar(ToolSchema("p", "", sch), opts)
    tpl = drive_walks([g.root], opts, lambda rows, decs: [1 for _ in decs])[0]
    p = f.program(tpl)
    h = p.holes[0]
    assert h.kind == "pattern" and h.follow.startswith(", ")
    h.tokens = f.tokenize_hole("pattern", "abc", h, p.cap[0])
    comma = _id(qtok, '",')
    assert h.tokens[-1] == comma                                         # the natural '",' closer
    lp = f.score([p], torch.zeros(1, 0, enc.sem_dim, device=enc.device), _ctx(f, 1), 1.0)[0]
    assert bool(torch.isfinite(lp).all()) and h.value == "abc" and h.consumed == 1


def test_ow_tiny_string_slots_close_by_the_lm_without_trailing_commas(gen, tok, cfg, ow_tiny_tools):
    """The real ow_tiny catalogue through the ActionGenerator (untrained, no length prior): strings end by the LM's
    own closing choice, never with a stray comma (pre-v4: 41/80 actions ended a string with ',')."""
    from sem_helpers import make_context
    from superlearner.cam_unified.interfaces import EntityRegistry
    from superlearner.cam_unified.testing import sample_observations
    names = [t.name for t in ow_tiny_tools if gen.grammar(t).generable and gen.grammar(t).may_hole][:16]
    B = len(names)
    tool_lists = [ow_tiny_tools] * B
    obs_lists = [sample_observations(4, seed=b) for b in range(B)]
    R, goal, belief = make_context(cfg, B, seed=5)
    obs = tok.observations(obs_lists, [EntityRegistry(cfg.max_entities, seed=b) for b in range(B)], [0.0] * B)
    tools = tok.tools(tool_lists)
    lists = gen.tool_lists_with_builtins(tool_lists)
    mask = torch.zeros(B, max(len(l) for l in lists), dtype=torch.bool, device=gen.device)
    for b, n in enumerate(names):
        mask[b, [t.name for t in lists[b]].index(n)] = True
    captured = []
    orig = gen.filler.sample
    saved = (gen.length_head, gen.filler.length_head)
    gen.filler.sample = lambda progs, *a, **k: (captured.append(progs), orig(progs, *a, **k))[1]
    gen.length_head = gen.filler.length_head = None
    try:
        gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, greedy=True, tool_mask=mask, rescore=False)
    finally:
        gen.filler.sample = orig
        gen.length_head, gen.filler.length_head = saved
    holes = [h for progs in captured for p in progs for h in p.holes]
    commas = sum(h.value.rstrip().endswith(",") for h in holes)
    own = sum(bool(h.own_close) for h in holes)
    print(f"\now_tiny string slots: {len(holes)}, own close {own}, trailing commas {commas}, merged closes "
          f"{sum(h.consumed > 0 for h in holes)}; e.g. {[h.value[:30] for h in holes[:6]]}")
    assert len(holes) >= 16
    assert commas == 0
    assert own >= 0.8 * len(holes)
