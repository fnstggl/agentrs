"""Argument-length control of the untrained decoder (workstream A2; spec 9.2, 22).

Schema-derived per-field token caps (grammar.string_class / string_token_cap) and the learned stop hazard
(grammar.LengthHead + stop_mixture_log_softmax): a free string's expected length at init is short, the LM can still
close whenever it wants, the reported log-probability includes the stop decisions, every parameter of the head is
trained by the exact log-probabilities, and long code stays possible (and learnable)."""
from __future__ import annotations

import math

import pytest
import torch

from superlearner.cam_unified.interfaces import EntityRegistry, ToolSchema
from superlearner.cam_unified.semantic.actions import ActionGenerator
from superlearner.cam_unified.semantic.grammar import (STRING_CLASSES, GrammarOptions, LengthHead, ToolGrammar,
                                                       holes_of, stop_mixture_log_softmax, string_class)
from superlearner.cam_unified.testing import sample_observations

from sem_helpers import make_context


def _validate(schema, value):
    from crucible.core.schema import validate
    return validate(schema, value)


def _string_nodes(node, out):
    if node.kind == "string":
        out.append(node)
    for _, c, _ in node.props:
        _string_nodes(c, out)
    if node.extra is not None:
        _string_nodes(node.extra, out)
    if node.item is not None:
        _string_nodes(node.item, out)
    for o in node.options.values():
        _string_nodes(o, out)
    return out


def test_stop_mixture_is_a_distribution_that_never_lowers_the_lm_closing_probability():
    torch.manual_seed(0)
    P, V = 6, 50
    logits = torch.randn(P, V) * 3
    allowed = torch.rand(P, V) < 0.7
    close = torch.zeros(P, V, dtype=torch.bool)
    close[:, :5] = True
    close[4] = False                       # row 4 cannot close (minLength not reached)
    allowed[5] = False
    allowed[5, :3] = True                  # row 5 must close (only closing tokens allowed)
    allowed[:, 0] = True
    hz = torch.tensor([-8.0, -2.0, 0.0, 3.0, 1.0, 1.0], requires_grad=True)
    log_lam = torch.nn.functional.logsigmoid(hz)
    lp = stop_mixture_log_softmax(logits, allowed, close, 1.0, log_lam)
    lm = stop_mixture_log_softmax(logits, allowed, close, 1.0, None)
    assert torch.allclose(torch.logsumexp(lp, -1), torch.zeros(P), atol=1e-5)          # a distribution per row
    cl = close & allowed
    p_stop = lp.masked_fill(~cl, -torch.inf).logsumexp(-1).exp()
    p_lm = lm.masked_fill(~cl, -torch.inf).logsumexp(-1).exp()
    lam = torch.sigmoid(hz.detach())
    for r in range(4):                                   # mixing rows: P(stop) = lam + (1 - lam) P_LM(close)
        assert abs(float(p_stop[r]) - float(lam[r] + (1 - lam[r]) * p_lm[r])) < 1e-5
        assert float(p_stop[r]) >= float(p_lm[r]) - 1e-7
        # continuing tokens keep the LM's relative probabilities (scaled by 1 - lam)
        cont = allowed[r] & ~cl[r]
        assert torch.allclose(lp[r][cont], lm[r][cont] + torch.log1p(-lam[r]), atol=1e-5)
    assert torch.allclose(lp[4], lm[4]) and torch.allclose(lp[5], lm[5])             # no mixing possible
    # lam -> 0 recovers the pure LM
    lp0 = stop_mixture_log_softmax(logits, allowed, close, 1.0, torch.full((P,), -30.0))
    assert torch.allclose(lp0[allowed], lm[allowed], atol=1e-6)
    # gradients are finite everywhere (also for the rows that cannot close)
    lp[allowed].sum().backward()
    assert torch.isfinite(hz.grad).all() and float(hz.grad[4]) == 0.0 and float(hz.grad[5]) == 0.0


def test_length_prior_expected_lengths_are_short_and_trainable():
    head = LengthHead(16, 32)
    e = {c: head.expected_length(0.0, c) for c in STRING_CLASSES}
    # an LM that never closes is stopped by the prior after about 28 / 104 / 441 tokens (instead of the hard caps)
    assert 20 < e["short"] < 36 and 90 < e["text"] < 120 and 380 < e["code"] < 520, e
    # the per-token stop probability at the start of a string is tiny (<= ~0.25%) in every class
    lam0 = head.log_hazard(torch.zeros(3), torch.arange(3), torch.zeros(3)).exp()
    assert float(lam0.max()) < 3e-3
    # the prior keeps long code possible: the survival of a never-closing 1,068-token code string is a few percent
    n = torch.arange(1068, dtype=torch.float32)
    c = torch.full((1068,), STRING_CLASSES.index("code"))
    surv = float(torch.log1p(-head.log_hazard(torch.zeros(1068), c, n).exp()).sum().exp())
    assert 0.005 < surv < 0.1, surv
    # the parameters are real parameters: raising the location lengthens the expected string
    with torch.no_grad():
        head.log_L[STRING_CLASSES.index("text")] += math.log(3.0)
    assert head.expected_length(0.0, "text") > 2.5 * e["text"]
    assert all(p.requires_grad for p in head.parameters())


def test_schema_derived_caps_on_the_ow_tiny_catalogue(ow_tiny_tools):
    """Per-field caps come from the schema: maxLength (2*maxLength + 1 tokens), format hints, the field's class
    (code / file fields keep cfg.max_arg_tokens); patterns keep their DFA caps; every cut is a listed reduction."""
    opts = GrammarOptions()
    by = {}
    for t in ow_tiny_tools:
        g = ToolGrammar(t, opts)
        assert g.generable, (t.name, g.error)
        for n in _string_nodes(g.root, []):
            by[(t.name, n.path)] = n
    files = by[("code.commit", ".files.*")]
    assert files.str_class == "code" and files.tok_cap == opts.max_string_tokens == 1536
    body = by[("forum.post", ".body")]
    assert body.str_class == "text" and body.tok_cap == opts.text_string_tokens
    acct = by[("bank.transfer", ".from_account")]                       # maxLength 40 -> 81 tokens, not cut
    assert acct.str_class == "short" and acct.tok_cap == 81
    title = by[("forum.post", ".title")]                                # maxLength 200 -> short cap 96 (listed)
    assert title.str_class == "short" and title.tok_cap == opts.short_string_tokens
    red = {(r["path"], r["kind"]) for r in ToolGrammar(next(t for t in ow_tiny_tools if t.name == "forum.post"),
                                                       opts).reductions}
    assert (".title", "string_cap") in red and (".body", "string_cap") in red
    red_acct = {(r["path"], r["kind"]) for r in ToolGrammar(next(t for t in ow_tiny_tools
                                                                if t.name == "bank.transfer"), opts).reductions}
    assert (".from_account", "string_cap") not in red_acct
    classes = {}
    for n in by.values():
        if n.dfa is None:
            classes[n.str_class] = classes.get(n.str_class, 0) + 1
    assert classes.get("code", 0) >= 1 and classes.get("text", 0) >= 20 and classes.get("short", 0) >= 150, classes
    # format hints and an unknown long field with a randomised name
    s = {"type": "object", "properties": {"when": {"type": "string", "format": "date-time"},
                                          "zq_9": {"type": "string", "maxLength": 50000},
                                          "src": {"type": "string"}}, "required": ["when", "zq_9", "src"]}
    g = ToolGrammar(ToolSchema("x.y", "", s), opts)
    nodes = {n.path: n for n in _string_nodes(g.root, [])}
    assert nodes[".when"].tok_cap == 48 and nodes[".zq_9"].str_class == "text" and nodes[".src"].str_class == "code"
    assert string_class(".a.b.files.*", "", {"maxLength": 10}) == "code"


@pytest.fixture(scope="module")
def free_gen(enc, tok, cfg):
    """A generator with production-like caps (cfg.max_arg_tokens for code, the default short/text caps)."""
    torch.manual_seed(1)
    return ActionGenerator(cfg, enc, tok, opts=GrammarOptions(max_string_tokens=cfg.max_arg_tokens,
                                                              max_action_tokens=2 * cfg.max_arg_tokens))


def _rows(tok, cfg, tool, B, seed):
    obs = sample_observations(4, seed=seed)
    R, goal, belief = make_context(cfg, B, seed=seed)
    T = tok.tools([[tool]] * B)
    O = tok.observations([obs] * B, [EntityRegistry(cfg.max_entities, seed=b) for b in range(B)], [0.0] * B)
    return R, goal, belief, T, O, [[tool]] * B, [obs] * B


def test_untrained_free_strings_are_short_and_the_logp_includes_the_stop_decisions(free_gen, tok, cfg):
    """A body (maxLength 20,000) and a code file at init: decoded lengths stay near the prior's scale instead of the
    1,536-token cap; the reported logp is the learner-path value including the stop factors (bit-equal to
    log_prob), and every LengthHead parameter receives a gradient from it."""
    tool = ToolSchema("notes.write", "Write a note and attach a small python file.",
                      {"type": "object", "properties": {"body": {"type": "string", "maxLength": 20000},
                                                        "files": {"type": "object", "additionalProperties":
                                                                  {"type": "string", "maxLength": 65536}},
                                                        "title": {"type": "string", "maxLength": 120}},
                       "required": ["body", "title"], "additionalProperties": False})
    B = 8
    args = _rows(tok, cfg, tool, B, seed=3)
    g = torch.Generator(device=free_gen.device).manual_seed(0)
    only = torch.zeros(B, 3, dtype=torch.bool)
    only[:, 0] = True                                     # the tool itself (not the built-in wait / noop)
    acts = free_gen.sample(*args, generator=g, tool_mask=only)
    lens = {"body": [], "code": [], "title": []}
    for a in acts:
        assert not _validate(tool.parameters, a.args)
        rec = a.logp_parts["decode"]
        assert a.tool == "notes.write" and rec["len_prior"] is True and a.logp_parts["rescored"]
        lens["body"].append(len(rec["holes"][0]))
    total = [a.logp_parts["n_tokens"] for a in acts]
    assert max(total) < 1536 and sum(total) / B < 400, total
    assert sum(lens["body"]) / B < 200, lens
    # learner path with gradients: identical value, includes the stop decisions
    lp, _ = free_gen.log_prob(*args, acts)
    assert all(float(lp[b]) == acts[b].logp for b in range(B))
    free_gen.zero_grad(set_to_none=True)
    (-lp.sum()).backward()
    head = free_gen.length_head
    assert head.log_L.grad is not None and float(head.log_L.grad.abs().sum()) > 0
    assert head.log_w.grad is not None and float(head.log_w.grad.abs().sum()) > 0
    assert head.ff[-1].weight.grad is not None and float(head.ff[-1].weight.grad.abs().sum()) > 0
    free_gen.zero_grad(set_to_none=True)


def test_long_code_stays_possible_and_behaviour_cloning_lengthens_the_prior(free_gen, tok, cfg, enc):
    """A ~1,000-token code file (the size of a real code.commit) is a valid, scorable action; a few BC steps on it
    lengthen the code class's prior (expected length) and raise the demo log-probability (the prior is no hard cut)."""
    tool = ToolSchema("code.put", "Commit one source file.",
                      {"type": "object", "properties": {"files": {"type": "object", "additionalProperties":
                                                                  {"type": "string", "maxLength": 65536}}},
                       "required": ["files"], "additionalProperties": False})
    code = "\n".join(f"def f{i}(x):\n    y = x * {i} + {i * 7 % 13}\n    return y - {i}\n" for i in range(38))
    demo_args = {"files": {"main.py": code}}
    from superlearner.cam_unified.interfaces import Action
    demo = Action(tool="code.put", args=demo_args)
    with torch.no_grad():                                 # fixed inputs: only the length head is trained here
        args = _rows(tok, cfg, tool, 1, seed=5)
    head = free_gen.length_head
    saved = {k: v.detach().clone() for k, v in head.state_dict().items()}
    opt = torch.optim.Adam(list(head.parameters()), lr=0.02)
    E0 = head.expected_length(0.0, "code")
    lp0 = None
    try:
        for step in range(10):
            lp, _ = free_gen.log_prob(*args, [demo])
            if lp0 is None:
                lp0 = float(lp[0])
                n_tok = free_gen.last_compute["tokens"]
                assert 900 < n_tok <= cfg.max_arg_tokens, n_tok
            assert math.isfinite(float(lp[0]))
            opt.zero_grad()
            (-lp.sum()).backward()
            opt.step()
        lp1 = float(free_gen.log_prob(*args, [demo])[0][0])
        E1 = head.expected_length(0.0, "code")
    finally:                                              # restore the shared module-scoped generator's prior
        head.load_state_dict(saved)
    assert E1 > 1.1 * E0 and lp1 > lp0 + 1.0, (E0, E1, lp0, lp1)
    print(f"long code: {n_tok} tokens, demo logp {lp0:.1f} -> {lp1:.1f}, E[len | never closes] {E0:.0f} -> {E1:.0f}")
