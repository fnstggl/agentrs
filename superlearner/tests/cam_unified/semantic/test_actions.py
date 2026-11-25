"""ActionGenerator: schema validity on the real Crucible catalogue, exact log-prob replay, gradient routing,
unseen tools, built-ins, latent influence, behaviour cloning, provenance."""
from __future__ import annotations

import gc
import json
import time

import pytest
import torch

from sem_helpers import make_context
from superlearner.cam_unified.interfaces import Action, ActionSource, EntityRegistry, ToolSchema
from superlearner.cam_unified.semantic.actions import ActionGenerator
from superlearner.cam_unified.semantic.grammar import GrammarOptions
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
        # every record replays exactly through the learner path
        lp, _ = g2.log_prob(R, goal, belief, tools, obs, tool_lists, [obs_row] * len(rows), acts)
        assert torch.isfinite(lp).all()
    assert n == len(ow_tiny_tools) >= 140 and valid == n
    assert used_qwen > 50                                   # most tools have string slots filled by Qwen
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
        acts = g32.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, temperature=temperature, greedy=greedy,
                          generator=g, tool_mask=mask)
        parts, _ = g32.log_prob_parts(R, goal, belief, tools, obs, tool_lists, obs_lists, acts)
        lp = sum(parts.values())
        for b, a in enumerate(acts):
            assert abs(float(lp[b]) - a.logp) < 2e-3 + 2e-4 * a.logp_parts["n_tokens"], (a.tool, float(lp[b]), a.logp)
            assert abs(float(parts["tool"][b]) - a.logp_parts["tool"]) < 1e-4
            assert abs(float(parts["evidence"][b]) - a.logp_parts["evidence"]) < 1e-4
            assert abs(float(parts["heads"][b]) - a.logp_parts["args_heads"]) < 1e-4
            seen_tokens += a.logp_parts["n_tokens"]
    assert seen_tokens > 10                                   # Qwen string tokens were part of the check


def test_rescore_makes_behaviour_logp_identical_to_learner_path(gen, tok, cfg):
    tool_lists = [sample_tools(5), sample_tools(5)]
    obs_lists = [sample_observations(4, seed=7), sample_observations(4, seed=8)]
    R, goal, belief, tools, obs = _inputs(tok, cfg, tool_lists, obs_lists, seed=9)
    mask = _force(gen, tool_lists, ["send_message", "set_price"])
    acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, tool_mask=mask, rescore=True,
                      exact_rows=True, generator=torch.Generator(device=gen.device).manual_seed(0))
    lp, ent = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, acts, exact_rows=True)
    for b, a in enumerate(acts):
        assert abs(float(lp[b]) - a.logp) < 1e-4
    assert (ent > 0).all()
    # row-wise scoring is batch-invariant: another minibatch composition gives the same numbers; None rows score 0
    lp2, en2 = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, [acts[0], None], exact_rows=True)
    assert float(lp2[1]) == 0.0 and float(en2[1]) == 0.0 and abs(float(lp2[0]) - acts[0].logp) < 1e-3


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
    demos = [Action("set_price", {"product": "prod_1", "price": 12.5}, source=ActionSource.SCRIPTED),
             Action("send_message", {"to": "cust_7", "text": "Yes, 40 units by Friday."}, source=ActionSource.SCRIPTED)]
    from superlearner.cam_unified.semantic.objectives import behaviour_cloning_loss
    gen_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    tok_state = {k: v.detach().clone() for k, v in tok.state_dict().items()}
    lora_state = enc.trainable_state_dict()
    try:
        params = [p for p in list(gen.parameters()) + list(tok.parameters()) if p.requires_grad]
        opt = torch.optim.Adam(params, lr=2e-3)
        R, goal, belief, _, _ = _inputs(tok, cfg, tool_lists, obs_lists, seed=2)
        with pytest.raises(ValueError):
            behaviour_cloning_loss(gen, R, goal, belief, tok.tools(tool_lists), None, tool_lists, obs_lists, demos, "")
        first = None
        for step in range(12):
            tools = tok.tools(tool_lists)
            loss, info = behaviour_cloning_loss(gen, R, goal, belief, tools, None, tool_lists, obs_lists, demos,
                                                "demo-init:test")
            first = float(loss) if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
            enc.mark_updated()
        assert info["n"] == 2 and float(loss) < first - 1.0, (first, float(loss))
    finally:
        gen.load_state_dict(gen_state)
        tok.load_state_dict(tok_state)
        enc.load_trainable_state_dict(lora_state)


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
        prompt = gen._prompt(tool_lists[b][0], [obs_lists[b][i] for i in ev])
        for i in ev:
            if obs_lists[b][i].content:
                assert obs_lists[b][i].content[:30] in prompt
        assert rec["prompt"] == gen.filler.prompt_hash(gen.filler.context_ids(prompt))
        n_ev += len(ev)
        # log_prob with the observation tokens missing cannot silently rescore a different decision
        if ev:
            with pytest.raises(Exception):
                gen.log_prob(R[b:b + 1], goal.select([b]), belief.select([b]), tools.select([b]), None,
                             [tool_lists[b]], [obs_lists[b]], [a])
    assert n_ev > 0
