"""ActionGenerator: schema validity on the real Crucible catalogue, exact log-prob replay, gradient routing,
unseen tools, built-ins, latent influence, behaviour cloning, provenance."""
from __future__ import annotations

import gc
import json
import math
import time

import pytest
import torch

from sem_helpers import make_context
from superlearner.cam_unified.interfaces import Action, ActionSource, EntityRegistry, Observation, ObsKind, ToolSchema
from superlearner.cam_unified.semantic.actions import ActionGenerator
from superlearner.cam_unified.semantic.grammar import GrammarError, GrammarOptions
from superlearner.cam_unified.testing import sample_observations, sample_tools


def _validate(schema, value):
    from crucible.core.schema import validate
    return validate(schema, value)


def _inputs(tok, cfg, tools_per_row, obs_per_row, seed=0, requires_grad=False):
    B = len(tools_per_row)
    R, goal, belief = make_context(cfg, B, seed=seed, requires_grad=requires_grad)
    regs = [EntityRegistry(cfg.max_entities, seed=seed + b) for b in range(B)]
    obs = tok.observations(obs_per_row, regs, [0.0] * B)
    tools = tok.tools(tools_per_row)
    return R, goal, belief, tools, obs


def _force(gen, tool_lists, names):
    """tool_mask [B, T'] allowing exactly one named tool per row (over the tool list incl. built-ins)."""
    lists = gen.tool_lists_with_builtins(tool_lists)
    T = max(len(l) for l in lists)
    m = torch.zeros(len(lists), T, dtype=torch.bool, device=gen.device)
    for b, (lst, n) in enumerate(zip(lists, names)):
        m[b, [t.name for t in lst].index(n)] = True
    return m


def test_schema_valid_generation_for_every_ow_tiny_tool(enc, tok, cfg, ow_tiny_tools):
    """Every tool of a real Crucible ow_tiny catalogue (WorldServer.available_tools) gets a schema-valid call from
    the full decoder (pointer -> evidence -> typed heads -> Qwen strings): 100% by construction."""
    g2 = ActionGenerator(cfg, enc, tok, opts=GrammarOptions(max_string_tokens=12, max_action_tokens=48,
                                                             max_key_tokens=8, max_array_items=3, max_map_keys=2))
    names = [t.name for t in ow_tiny_tools]
    obs_row = sample_observations(5, seed=2)
    gen_rng = torch.Generator(device=enc.device).manual_seed(0)
    valid, n, t0, used_qwen = 0, 0, time.time(), 0
    chunk = 16
    for s in range(0, len(names), chunk):
        rows = names[s: s + chunk]
        tool_lists = [ow_tiny_tools] * len(rows)
        R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, [obs_row] * len(rows), seed=s)
        acts = g2.sample(R, goal, belief, tools, obs, tool_lists, [obs_row] * len(rows),
                         tool_mask=_force(g2, tool_lists, rows), generator=gen_rng)
        used_qwen += g2.last_compute["hole_rows"]
        for name, a in zip(rows, acts):
            assert a.tool == name and a.source == ActionSource.CAM_DECODER
            schema = next(t for t in ow_tiny_tools if t.name == name).parameters
            errs = _validate(schema, a.args)
            valid += not errs
            n += 1
            assert not errs, (name, a.args, errs)
            assert a.logp is not None and a.logp <= 0 and "decode" in a.logp_parts
            # sample() replayed every record through the learner path (rescore) and took its logp from there
            assert a.logp_parts["rescored"] and math.isfinite(a.logp)
            assert a.logp_parts["decode"]["reduced"] == len(g2.reduced.get(name, []))
    assert n == len(ow_tiny_tools) >= 140 and valid == n
    assert used_qwen > 50                                   # most tools have string slots filled by Qwen
    # capability reductions are surfaced per tool and counted
    assert any(r["kind"] == "array_cap" and r["path"] == ".to" for r in g2.reduced["mail.send"])
    assert g2.stats["reduced_tools"] > 0 and g2.reduction_counts["array_cap"] >= 6
    print(f"ow_tiny: {valid}/{n} schema-valid, {used_qwen} with Qwen strings, {time.time() - t0:.1f}s")


@pytest.fixture(scope="module")
def fp32_stack(cfg, cuda):
    from superlearner.cam_unified.semantic.encoder import QwenSemantic
    from superlearner.cam_unified.semantic.tokenizer import ObservationTokenizer
    e32 = QwenSemantic(cfg, test=True, dtype=torch.float32)
    t32 = ObservationTokenizer(cfg, e32, autocast=False)
    torch.manual_seed(1)
    g32 = ActionGenerator(cfg, e32, t32, autocast=False,
                          opts=GrammarOptions(max_string_tokens=24, max_action_tokens=64, max_key_tokens=8))
    yield e32, t32, g32
    del g32, t32, e32
    gc.collect()
    torch.cuda.empty_cache()


def test_log_prob_of_sampled_actions_matches_sampling_logp_fp32(fp32_stack, cfg):
    e32, t32, g32 = fp32_stack
    tool_lists = [sample_tools(5), sample_tools(5, randomize_names=True), sample_tools(4), sample_tools(5)]
    obs_lists = [sample_observations(6, seed=b) for b in range(4)]
    R, goal, belief, tools, obs = _inputs(t32, cfg, tool_lists, obs_lists, seed=5)
    g = torch.Generator(device=e32.device).manual_seed(3)
    seen_tokens = 0
    for temperature, greedy, forced in [(1.0, False, None), (0.7, False, ["search_web", "tool", "run_python",
                                                                          "hire"]), (1.0, True, None)]:
        mask = None
        if forced is not None:
            names = [forced[0], tool_lists[1][3].name, forced[2], forced[3]]
            mask = _force(g32, tool_lists, names)
        # rescore=False: compare the incremental (KV-cache) sampling computation itself with the learner path
        acts = g32.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, temperature=temperature, greedy=greedy,
                          generator=g, tool_mask=mask, rescore=False)
        parts, _ = g32.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, acts)
        lp = sum(parts.values())
        for b, a in enumerate(acts):
            inc = a.logp_parts["sample_logp"]
            assert abs(float(lp[b]) - inc) < 2e-3 + 2e-4 * a.logp_parts["n_tokens"], (a.tool, float(lp[b]), inc)
            pre = ""
            if greedy:                     # deterministic behaviour: logp 0, the policy's value kept separately
                assert a.logp == 0.0 and a.logp_parts["greedy"] and a.logp_parts["policy_logp"] == inc
                assert not ActionGenerator.importance_weight_usable(a)
                pre = "policy_"            # behaviour parts are 0 (mu = 1); the policy parts are policy_<part>
            else:
                assert a.logp == inc
            assert abs(float(parts["tool"][b]) - a.logp_parts[pre + "tool"]) < 1e-4
            assert abs(float(parts["evidence"][b]) - a.logp_parts[pre + "evidence"]) < 1e-4
            assert abs(float(parts["heads"][b]) - a.logp_parts[pre + "args_heads"]) < 1e-4
            seen_tokens += a.logp_parts["n_tokens"]
    assert seen_tokens > 10                                   # Qwen string tokens were part of the check


def _free_string_rows(tok, cfg, B, seed):
    """B rows forced to free-vocabulary string tools (search_web / send_message) with varied observations."""
    tool_lists = [sample_tools(5) for _ in range(B)]
    obs_lists = [sample_observations(3 + b % 5, seed=seed + b) for b in range(B)]
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=seed, requires_grad=True)
    names = ["search_web" if b % 2 else "send_message" for b in range(B)]
    return tool_lists, obs_lists, R, goal, belief, tools, obs, names


def test_default_bf16_behaviour_logp_is_the_learner_logp_on_free_strings(gen, tok, cfg):
    """The default sample() path (bf16, incremental KV decoding of free-vocabulary JSON strings) returns the logp the
    learner recomputes - bit-for-bit, in another minibatch composition and with gradients enabled - so PPO ratios at
    the behaviour parameters are exactly 1. The incremental value differs (kept only as a diagnostic)."""
    B = 16
    tool_lists, obs_lists, R, goal, belief, tools, obs, names = _free_string_rows(tok, cfg, B, seed=40)
    acts = gen.sample(R.detach(), goal, belief, tools, obs, tool_lists, obs_lists,
                      tool_mask=_force(gen, tool_lists, names),
                      generator=torch.Generator(device=gen.device).manual_seed(3))
    n_tok = [a.logp_parts["n_tokens"] for a in acts]
    assert sum(n_tok) > 4 * B and max(n_tok) > 20                     # real free strings of varied length
    lp, _ = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, acts)       # grad-enabled learner path
    assert lp.requires_grad
    assert all(float(lp[b]) == a.logp for b, a in enumerate(acts)), [(float(lp[b]), a.logp) for b, a in
                                                                     enumerate(acts)]
    perm = list(reversed(range(B)))[: B // 2]                         # another minibatch: rows reversed, half of them
    sub = lambda ts: ts.select(perm)
    lp2, _ = gen.log_prob(R[perm], sub(goal), sub(belief), sub(tools), sub(obs), [tool_lists[i] for i in perm],
                          [obs_lists[i] for i in perm], [acts[i] for i in perm])
    assert all(float(lp2[j]) == acts[i].logp for j, i in enumerate(perm))
    drift = sorted(abs(a.logp - a.logp_parts["sample_logp"]) for a in acts)
    assert all(ActionGenerator.importance_weight_usable(a) for a in acts)
    batched, _ = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, acts, exact_rows=False)
    bdrift = sorted(abs(float(batched[b]) - a.logp) for b, a in enumerate(acts))
    print(f"\nincremental-vs-learner drift (nats): median {drift[B // 2]:.3f} max {drift[-1]:.3f}; "
          f"batched-vs-row-wise: median {bdrift[B // 2]:.4f} max {bdrift[-1]:.4f}; tokens {sum(n_tok)}")


def test_exact_rows_is_bitwise_batch_invariant_and_none_rows_score_zero(gen, tok, cfg):
    tool_lists = [sample_tools(5), sample_tools(5), sample_tools(5, randomize_names=True, seed=3)[:2] + sample_tools(5)[4:]]
    obs_lists = [sample_observations(4, seed=7), sample_observations(6, seed=8), sample_observations(2, seed=9)]
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=9)
    mask = _force(gen, tool_lists, ["send_message", "set_price", "hire"])
    acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, tool_mask=mask,
                      generator=torch.Generator(device=gen.device).manual_seed(0))
    lp, ent = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, acts)
    assert [float(v) for v in lp] == [a.logp for a in acts] and (ent > 0).all()
    lp2, en2 = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, [acts[0], None, acts[2]])
    assert float(lp2[1]) == 0.0 and float(en2[1]) == 0.0
    assert float(lp2[0]) == acts[0].logp and float(lp2[2]) == acts[2].logp


def test_temperature_override_and_greedy_behaviour_logp(gen, tok, cfg):
    tool_lists = [sample_tools(5)] * 2
    obs_lists = [sample_observations(4, seed=1)] * 2
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=14)
    mask = _force(gen, tool_lists, ["hire", "send_message"])
    acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, temperature=0.7, tool_mask=mask,
                      generator=torch.Generator(device=gen.device).manual_seed(6))
    lp_rec, _ = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, acts)
    assert [float(v) for v in lp_rec] == [a.logp for a in acts]
    lp_t1, _ = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, acts, temperature=1.0)
    lp_t07, _ = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, acts, temperature=0.7)
    assert torch.equal(lp_t07, lp_rec) and (lp_t1 - lp_rec).abs().min() > 1e-3   # target policy at tau = 1
    g_acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, greedy=True, tool_mask=mask)
    lp_g, _ = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, g_acts)
    parts_g, _ = gen.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, g_acts)
    for b, a in enumerate(g_acts):
        assert a.logp == 0.0 and a.logp_parts["greedy"] and a.logp_parts["policy_logp"] == float(lp_g[b])
        assert not ActionGenerator.importance_weight_usable(a)
        # contract v3 #12: every behaviour part of a deterministic action is 0 - "tool" is always filled and is the
        # intervention's assignment log-probability (exp = 1 for the causal ledger); policy values under policy_*
        for k in ("tool", "evidence", "args", "args_heads", "args_tokens"):
            assert a.logp_parts[k] == 0.0 and isinstance(a.logp_parts["policy_" + k], float)
        assert a.logp_parts["policy_tool"] == float(parts_g["tool"][b])
        assert a.logp_parts["policy_args"] == a.logp_parts["policy_args_heads"] + a.logp_parts["policy_args_tokens"]
    # sampled (stochastic) actions always carry the tool part as their assignment log-probability (here the tool
    # pointer was restricted to one tool, so the assignment probability is exactly 1)
    assert all(isinstance(a.logp_parts["tool"], float) and "policy_tool" not in a.logp_parts for a in acts)
    assert all(a.logp_parts["tool"] == 0.0 for a in acts)


def test_replay_refuses_a_different_tool_list_or_evidence_candidates(gen, tok, cfg):
    """A record replays only against the tool list and evidence candidates it was sampled with - including records
    whose evidence decision was 'stop at once' (evidence = []) and tools without string holes."""
    adjust = ToolSchema("adjust_budget", "Set the daily budget in cents.", {"type": "object", "properties": {
        "cents": {"type": "integer", "minimum": 1, "maximum": 10 ** 9}}, "required": ["cents"],
        "additionalProperties": False})
    B = 12
    tool_lists = [sample_tools(3) + [adjust] for _ in range(B)]
    obs_lists = [sample_observations(7, seed=b) for b in range(B)]
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=15)
    names = ["adjust_budget" if b % 2 else "search_web" for b in range(B)]
    acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, tool_mask=_force(gen, tool_lists, names),
                      generator=torch.Generator(device=gen.device).manual_seed(11))
    recs = [a.logp_parts["decode"] for a in acts]
    assert all(r["ev"] and r["cands"] for r in recs)                 # evidence evaluated for strings and numbers
    empty = [b for b, r in enumerate(recs) if r["evidence"] == []]
    no_hole = [b for b, r in enumerate(recs) if not r["holes"]]
    assert empty and no_hole, (empty, no_hole)
    one = lambda b, o=None, ol=None, tl=None, tt=None: gen.log_prob(
        R[b:b + 1], goal.select([b]), belief.select([b]), tt if tt is not None else tools.select([b]),
        o if o is not None else obs.select([b]), tl or [tool_lists[b]], ol or [obs_lists[b]], [acts[b]])
    for b in sorted(set(empty) | set(no_hole)):
        assert float(one(b)[0][0]) == acts[b].logp                   # the same inputs replay exactly
        with pytest.raises(GrammarError):                             # observation tokens missing
            gen.log_prob(R[b:b + 1], goal.select([b]), belief.select([b]), tools.select([b]), None,
                         [tool_lists[b]], [obs_lists[b]], [acts[b]])
        changed = list(obs_lists[b])                                   # one candidate's content changed
        changed[0] = Observation(ObsKind.TEXT, changed[0].content + " (edited)", t=changed[0].t,
                                 source=changed[0].source, entity=changed[0].entity,
                                 entity_type=changed[0].entity_type)
        o2 = tok.observations([changed], [EntityRegistry(cfg.max_entities, seed=b)], [0.0])
        with pytest.raises(GrammarError):
            one(b, o=o2, ol=[changed])
    # an unrestricted action (here 'wait') replayed against another tool list
    tl3 = [sample_tools(3)]
    R1, g1, b1, t1, o1 = _inputs(tok, cfg, tl3, [[]], seed=16)
    w = gen.sample(R1, g1, b1, t1, o1, tl3, [[]], generator=torch.Generator(device=gen.device).manual_seed(2))[0]
    assert w.logp_parts["decode"]["allowed"] is None
    tl5 = [sample_tools(5)]
    with pytest.raises(GrammarError):
        gen.log_prob(R1, g1, b1, tok.tools(tl5), o1, tl5, [[]], [w])
    assert float(gen.log_prob(R1, g1, b1, t1, o1, tl3, [[]], [w])[0][0]) == w.logp


def test_long_prompts_truncate_evidence_first_and_are_counted(gen, tok, cfg):
    filler = gen.filler
    schema_piece = 'Arguments schema: {"type":"object","properties":{"q":{"type":"string"}}}\n'
    pieces = [("Write the JSON arguments.\nTool: t\n", 0), ("Description: " + "word " * 300 + "\n", 2),
              (schema_piece, 3), ("Evidence selected for this decision:\n", 0),
              ("[1] " + "alpha " * 900 + "\n", 1), ("[2] short evidence\n", 1)]
    n_desc = len(gen.enc.tok(pieces[1][0], add_special_tokens=False)["input_ids"])
    ids, trunc = filler.fit_pieces(pieces, budget=n_desc + 200)
    text = gen.enc.tok.decode(ids)
    assert len(ids) <= n_desc + 200 and trunc > 0
    assert "word " * 300 in text and schema_piece.strip() in text and "short evidence" in text   # evidence cut first
    assert "tokens omitted" in text
    ids2, trunc2 = filler.fit_pieces(pieces, budget=150)              # then the description; the schema survives
    assert len(ids2) <= 150 and trunc2 > trunc and schema_piece.strip() in gen.enc.tok.decode(ids2)
    # end to end: a 5000-word description is shortened, the record and the stats count it
    huge = ToolSchema("annotate", "Annotate. " + "very " * 5000, {"type": "object", "properties": {
        "note": {"type": "string", "maxLength": 30}}, "required": ["note"], "additionalProperties": False})
    tl = [[huge]]
    R, goal, belief, tools, obs = _inputs(tok, cfg, tl, [sample_observations(3, seed=0)], seed=17)
    before = gen.stats["prompt_truncated_tokens"]
    a = gen.sample(R, goal, belief, tools, obs, tl, [sample_observations(3, seed=0)],
                   tool_mask=_force(gen, tl, ["annotate"]), generator=torch.Generator(device=gen.device).manual_seed(0))[0]
    rec = a.logp_parts["decode"]
    assert a.tool == "annotate" and a.logp_parts["n_tokens"] > 0
    assert rec["prompt_truncated"] > 0 and gen.stats["prompt_truncated_tokens"] - before == rec["prompt_truncated"]
    assert gen.last_compute["prompt_truncated_tokens"] == rec["prompt_truncated"]
    assert float(gen.log_prob(R, goal, belief, tools, obs, tl, [sample_observations(3, seed=0)], [a])[0][0]) == a.logp


def test_exact_amount_is_copied_from_the_selected_evidence(gen, tok, cfg):
    """After declared-condition behaviour cloning on one demonstration, the policy selects the invoice as evidence
    and copies its exact amount (1,234,567 cents) - a value no digit grid would hit by chance."""
    from superlearner.cam_unified.semantic.objectives import behaviour_cloning_loss
    pay = ToolSchema("bank.pay", "Pay an amount in cents from your account.", {"type": "object", "properties": {
        "amount": {"type": "integer", "minimum": 1, "maximum": 10 ** 12}}, "required": ["amount"],
        "additionalProperties": False})
    obs_l = [Observation(ObsKind.TEXT, "Weather: 21 degrees, wind 12 km/h.", t=0.0, source="news"),
             Observation(ObsKind.TEXT, "Invoice inv_42 from acct_7: amount 1,234,567 cents, due in 3 days.", t=0.5,
                         source="mail"),
             Observation(ObsKind.NUMERIC, value=88000.0, unit="cents", t=0.2, source="bank.balance",
                         meta={"metric": "balance"})]
    tool_lists, obs_lists = [[pay]], [obs_l]
    demo = [Action("bank.pay", {"amount": 1234567}, source=ActionSource.BASELINE)]
    declared = {"demo-init:example_inductor"}
    gen_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    try:
        with torch.no_grad():                                         # only the generator is trained here
            R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=18)
        n0 = gen.stats["demo_numbers_copied"]
        params = [p for n, p in gen.named_parameters() if p.requires_grad and not n.startswith("lora")]
        opt = torch.optim.Adam(params, lr=3e-3)
        for _ in range(25):
            loss, info = behaviour_cloning_loss(gen, R, goal, belief, tools, obs, tool_lists, obs_lists, demo,
                                                "demo-init:example_inductor", declared, evidence=[[1]])
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert gen.stats["demo_numbers_copied"] - n0 == 25           # the demo number is encoded as a copy
        a = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, greedy=True)[0]
        assert a.args == {"amount": 1234567} and a.logp_parts["n_copied"] == 1
        assert a.logp_parts["decode"]["evidence"] == [1]
    finally:
        gen.load_state_dict(gen_state)
        gen.zero_grad(set_to_none=True)


def test_gradients_reach_R_bridge_heads_pointers_lora_tokenizer_not_frozen_qwen(gen, tok, enc, cfg):
    tool_lists = [sample_tools(5), sample_tools(5)]
    obs_lists = [sample_observations(5, seed=1), sample_observations(5, seed=2)]
    gen.zero_grad(set_to_none=True)
    tok.zero_grad(set_to_none=True)
    for p in enc.peft.parameters():
        p.grad = None
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=3, requires_grad=True)
    mask = _force(gen, tool_lists, ["send_message", "set_price"])          # strings + typed numbers
    with torch.no_grad():
        acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, tool_mask=mask,
                          generator=torch.Generator(device=gen.device).manual_seed(1))
    assert acts[0].logp_parts["n_tokens"] > 0 and acts[1].logp_parts["n_decisions"] > 0
    lp, ent = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, acts)
    (lp.sum() + 0.01 * ent.sum()).backward()
    nz = lambda ps: sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
    assert R.grad is not None and float(R.grad.abs().sum()) > 0
    assert nz(gen.bridge.parameters()) > 0
    assert nz(gen.heads.parameters()) > 0
    assert nz([gen.tool_query, *gen.tool_q.parameters(), *gen.tool_k.parameters()]) > 0
    assert nz([gen.ev_query, *gen.ev_q.parameters(), *gen.ev_k.parameters()]) > 0
    lora_b = [p for n, p in enc.peft.named_parameters() if "lora_B" in n]
    assert nz(lora_b) > 0                                  # LoRA (decoder path) is trained by the action loss
    assert nz([tok.q_tool, *tok.pool.parameters()]) > 0     # through TOOL / observation tokens
    frozen = [p for n, p in enc.peft.named_parameters() if "lora_" not in n]
    assert all(p.grad is None for p in frozen)
    gen.zero_grad(set_to_none=True)
    tok.zero_grad(set_to_none=True)
    for p in enc.peft.parameters():
        p.grad = None


def test_unseen_and_renamed_tools_need_no_new_parameters(gen, tok, cfg):
    n0 = sum(p.numel() for p in gen.parameters()) + sum(p.numel() for p in tok.parameters())
    keys0 = set(gen.state_dict()) | set(tok.state_dict())
    novel = ToolSchema("zq_frobnicate", "Rotate the quantum widget by an angle and label it.",
                       {"type": "object", "properties": {"angle": {"type": "number", "minimum": 0, "maximum": 360},
                                                         "label": {"type": "string", "maxLength": 20},
                                                         "mode": {"type": "string", "enum": ["fast", "safe"]},
                                                         "tags": {"type": "array", "items": {"type": "string",
                                                                                            "pattern": "[a-z]{2,5}"},
                                                                  "maxItems": 3}},
                        "required": ["angle", "mode"], "additionalProperties": False})
    tool_lists = [sample_tools(5, randomize_names=True, seed=11) + [novel], sample_tools(5, randomize_names=True,
                                                                                         seed=12) + [novel]]
    obs_lists = [sample_observations(3, seed=4)] * 2
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=4)
    mask = _force(gen, tool_lists, ["zq_frobnicate", tool_lists[1][1].name])
    acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, tool_mask=mask,
                      generator=torch.Generator(device=gen.device).manual_seed(2))
    for a, lst in zip(acts, tool_lists):
        schema = next(t for t in lst if t.name == a.tool).parameters
        assert _validate(schema, a.args) == []
    assert acts[0].tool == "zq_frobnicate" and acts[0].args["mode"] in ("fast", "safe")
    n1 = sum(p.numel() for p in gen.parameters()) + sum(p.numel() for p in tok.parameters())
    assert n1 == n0 and (set(gen.state_dict()) | set(tok.state_dict())) == keys0


def test_builtin_wait_and_noop_are_ordinary_tools(gen, tok, cfg):
    tool_lists = [sample_tools(3), sample_tools(3)]
    obs_lists = [[], []]
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=6)
    acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, tool_mask=_force(gen, tool_lists,
                                                                                         ["wait", "noop"]))
    w, n = acts
    assert w.tool == "wait" and 1 <= w.args["seconds"] <= cfg.max_wait_s and isinstance(w.args["seconds"], int)
    assert abs(w.expected_duration - w.args["seconds"] / 86400.0) < 1e-12
    assert n.tool == "noop" and n.args == {}
    lists = gen.tool_lists_with_builtins(tool_lists)
    assert [t.name for t in lists[0]][-2:] == ["wait", "noop"]


def test_latent_state_changes_the_action_distribution(gen, tok, cfg):
    tool_lists = [sample_tools(5)]
    obs_lists = [sample_observations(4, seed=3)]
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=10)
    act = Action("send_message", {"to": "cust_7", "text": "We can deliver 40 units by Friday."})
    p1, _ = gen.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, [act])
    p2, _ = gen.log_prob_parts(R * 0 + torch.randn_like(R), goal, belief, tools, obs, tool_lists, obs_lists, [act])
    assert abs(float(p1["tool"][0] - p2["tool"][0])) > 1e-4        # pointer depends on R
    assert abs(float(p1["tokens"][0] - p2["tokens"][0])) > 1e-3    # Qwen strings depend on R through the prefix
    P1 = gen.bridge(R, goal, belief)
    assert P1.shape == (1, cfg.n_prefix, gen.enc.sem_dim)
    rms = P1.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, gen.enc.embedding_rms), rtol=1e-3)


def test_behaviour_cloning_raises_demo_logprob(gen, tok, enc, cfg):
    tool_lists = [sample_tools(5), sample_tools(5)]
    obs_lists = [sample_observations(5, seed=0), sample_observations(5, seed=1)]
    demos = [Action("set_price", {"product": "prod_1", "price": 12.5}, source=ActionSource.BASELINE),
             Action("send_message", {"to": "cust_7", "text": "Yes, 40 units by Friday."}, source=ActionSource.BASELINE)]
    declared = {"demo-init:example_inductor"}
    from superlearner.cam_unified.semantic.objectives import behaviour_cloning_loss
    gen_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    tok_state = {k: v.detach().clone() for k, v in tok.state_dict().items()}
    lora_state = enc.trainable_state_dict()
    try:
        params = [p for p in list(gen.parameters()) + list(tok.parameters()) if p.requires_grad]
        opt = torch.optim.Adam(params, lr=2e-3)
        R, goal, belief, _, _ = _inputs(tok, cfg, tool_lists, obs_lists, seed=2)
        tt = tok.tools(tool_lists)
        for bad in ("", "demo-init:test", "demo-init:builder_transcripts", "bc:example_inductor"):
            with pytest.raises(ValueError):                    # undeclared / privileged / malformed conditions
                behaviour_cloning_loss(gen, R, goal, belief, tt, None, tool_lists, obs_lists, demos, bad,
                                       declared | {"demo-init:builder_transcripts"})
        scripted = [Action(a.tool, a.args, source=ActionSource.SCRIPTED) for a in demos]
        with pytest.raises(ValueError):                        # builder (scripted) transcripts are never cloned
            behaviour_cloning_loss(gen, R, goal, belief, tt, None, tool_lists, obs_lists, scripted,
                                   "demo-init:example_inductor", declared)
        first = None
        ns0 = gen.demo_number_stats
        for step in range(12):
            tools = tok.tools(tool_lists)
            loss, info = behaviour_cloning_loss(gen, R, goal, belief, tools, None, tool_lists, obs_lists, demos,
                                                "demo-init:example_inductor", declared)
            first = float(loss) if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
            enc.mark_updated()
        assert info["n"] == 2 and float(loss) < first - 1.0, (first, float(loss))
        ns = gen.demo_number_stats                                # 12.5 is written exactly: never rounded
        assert ns["numbers"] - ns0["numbers"] == 12 and ns["approx"] == ns0["approx"]
    finally:
        gen.load_state_dict(gen_state)
        tok.load_state_dict(tok_state)
        enc.load_trainable_state_dict(lora_state)
        gen.zero_grad(set_to_none=True)
        tok.zero_grad(set_to_none=True)


def test_candidates_and_selection_provenance(gen, tok, cfg):
    tool_lists = [sample_tools(5)]
    obs_lists = [sample_observations(3, seed=5)]
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=12)
    cands = gen.sample_candidates(R, goal, belief, tools, obs, tool_lists, obs_lists, n=3,
                                  generator=torch.Generator(device=gen.device).manual_seed(4))
    assert len(cands) == 1 and len(cands[0]) == 3
    assert all(c.source == ActionSource.CAM_DECODER and c.logp is not None for c in cands[0])
    chosen = ActionGenerator.select_candidate(cands[0], [0.2, 0.5, 0.3], 1, source=ActionSource.CAM_PLANNER,
                                              scores=[1.0, 2.0, 1.5])
    assert chosen.source == ActionSource.CAM_PLANNER
    assert chosen.proposal["n"] == 3 and chosen.proposal["chosen"] == 1 and chosen.proposal["probs"] == [0.2, 0.5, 0.3]
    assert chosen.proposal["proposal_logp"] == [c.logp for c in cands[0]]
    assert chosen.logp == cands[0][1].logp and chosen.logp_parts["decode"] == cands[0][1].logp_parts["decode"]
    rt = Action.from_dict(json.loads(chosen.to_json()))       # survives the record boundary
    assert rt.source == ActionSource.CAM_PLANNER and rt.logp_parts["decode"] == chosen.logp_parts["decode"]


def test_evidence_is_chosen_by_cam_and_shown_verbatim(gen, tok, cfg):
    tool_lists = [sample_tools(5)] * 3
    obs_lists = [sample_observations(7, seed=b) for b in range(3)]
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=13)
    acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists,
                      tool_mask=_force(gen, tool_lists, ["search_web"] * 3),
                      generator=torch.Generator(device=gen.device).manual_seed(5))
    n_ev = 0
    for b, a in enumerate(acts):
        rec = a.logp_parts["decode"]
        ev = rec["evidence"]
        assert len(ev) <= cfg.max_evidence and len(set(ev)) == len(ev)
        assert all(0 <= i < len(obs_lists[b]) for i in ev)
        pieces = gen._prompt(tool_lists[b][0], [obs_lists[b][i] for i in ev])
        prompt = "".join(t for t, _ in pieces)
        for i in ev:
            if obs_lists[b][i].content:
                assert obs_lists[b][i].content[:30] in prompt
        ids, trunc = gen.filler.context_ids(pieces)
        assert trunc == 0 and rec["prompt"] == gen.filler.prompt_hash(ids) and rec["prompt_truncated"] == 0
        n_ev += len(ev)
        # log_prob with the observation tokens missing cannot silently rescore a different decision
        if ev:
            with pytest.raises(Exception):
                gen.log_prob(R[b:b + 1], goal.select([b]), belief.select([b]), tools.select([b]), None,
                             [tool_lists[b]], [obs_lists[b]], [a])
    assert n_ev > 0
