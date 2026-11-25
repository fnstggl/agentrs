"""Semantic learning objectives (spec 2.3): each loss is computed on real components and decreases when trained."""
from __future__ import annotations

import pytest
import torch

from sem_helpers import make_context
from superlearner.cam_unified.interfaces import Action, EntityRegistry, Observation, ObsKind, TokenKind
from superlearner.cam_unified.semantic.objectives import (CodeExecutionObjective, EntityGroundingObjective,
                                                          MaskedSemanticObjective, ToolConsequenceObjective,
                                                          evidence_retrieval_loss, mention_labels, slot_mean_keys)
from superlearner.cam_unified.testing import DoubleCore, sample_observations, sample_tools


def _train(loss_fn, params, steps, lr=3e-3):
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    for _ in range(steps):
        loss, info = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    return hist, info


def _texts_obs(n, t0=0.0):
    topics = ["The bakery sells bread at 300 cents.", "Server latency rose to 900 ms.", "A customer asked for a refund.",
              "The repository has 3 failing tests.", "Rain is expected tomorrow in Lyon.", "Hiring two engineers.",
              "The invoice was paid by acct_9.", "Deploy failed with exit code 2.", "Users love the new search.",
              "Inventory of widgets is low.", "The forum thread discusses pricing.", "GPU prices fell this week."]
    return [Observation(ObsKind.TEXT, topics[i % len(topics)] + f" (note {i})", t=t0 + i, source="feed")
            for i in range(n)]


def test_masked_semantic_prediction_trains_core_and_predictor(tok, cfg, cuda):
    obs_lists = [_texts_obs(6), _texts_obs(6, t0=1.0)[::-1], _texts_obs(5, t0=2.0)]
    with torch.no_grad():
        ts = tok.observations(obs_lists, None, [10.0, 10.0, 10.0])
    torch.manual_seed(0)
    core = DoubleCore(cfg).to(cuda)
    # the reference double has no time encoding; workstream B's backbone encodes TokenSet.time - emulate that here
    freqs = torch.exp(torch.linspace(-2.0, 2.0, 8, device=cuda))
    tenc = torch.nn.Linear(16, cfg.d_model).to(cuda)

    def core_fn(ctx):
        ang = ctx.time.unsqueeze(-1) * freqs
        x = ctx.x + tenc(torch.cat([torch.sin(ang), torch.cos(ang)], -1))
        return core(core.init_state(ctx.B), type(ctx)(x, ctx.mask, ctx.kind, ctx.time, ctx.entity, ctx.refs))[1]

    obj = MaskedSemanticObjective(cfg.d_model, mask_frac=0.25).to(cuda)
    g = torch.Generator(device=cuda).manual_seed(0)
    gi = obj.group_index(ts)
    assert gi[0, :cfg.sem_tokens_per_obs].tolist() == list(range(cfg.sem_tokens_per_obs))
    hist, info = _train(lambda: obj(core_fn, ts, generator=g),
                        list(core.parameters()) + list(obj.parameters()) + list(tenc.parameters()), 80)
    assert info["n_masked"] > 0
    assert sum(hist[-5:]) / 5 < sum(hist[:5]) / 5 - 0.5, hist[::10]


def test_tool_consequence_prediction_learns_action_to_result(tok, cfg, cuda):
    tok.register_tools(sample_tools(5))
    prices = [3.0, 7.5, 12.0, 20.0, 45.0, 90.0, 150.0, 400.0]
    acts = [Action("set_price", {"product": "prod_1", "price": p}) for p in prices]
    results = [[Observation(ObsKind.TOOL_RESULT, f'{{"ok": true, "price": {p}, "demand_change": "{"up" if p < 20 else "down"}"}}',
                            source="set_price")] for p in prices]
    with torch.no_grad():
        A = tok.action(acts)
        Rt = tok.observations(results, None, [0.0] * len(prices))
    state = torch.zeros(len(prices), cfg.d_model, device=cuda)
    obj = ToolConsequenceObjective(cfg.d_model, n_heads=cfg.n_heads).to(cuda)
    hist, info = _train(lambda: obj(state, A, Rt), obj.parameters(), 80)
    assert hist[-1] < hist[0] - 1.0 and info["acc"] >= 0.75, (hist[::10], info)


def test_entity_grounding_labels_and_contrastive_learning(tok, cfg, cuda):
    ents = ["zx_12", "qv_77", "mm_03", "kp_41", "rt_90"]
    desc = ["a bakery in Lyon", "a GPU cloud provider", "a translation service", "a payments startup",
            "a university lab"]
    bound = [Observation(ObsKind.TEXT, f"{e} is {d}.", entity=e, entity_type="organisation", t=0.0, source="registry")
             for e, d in zip(ents, desc)]
    mentions = [Observation(ObsKind.TEXT, f"An order placed with {e} ({d}) arrived late.", t=1.0, source="mail")
                for e, d in zip(ents, desc)]
    reg = EntityRegistry(cfg.max_entities, seed=4)
    obs_list = bound + mentions
    with torch.no_grad():
        ts = tok.observations([obs_list], [reg], [1.0])
    labels = mention_labels(ts, [obs_list], [reg])
    by_obs = {}
    from superlearner.cam_unified.semantic.tokenizer import parse_obs_ref
    for j, pos in labels[0]:
        by_obs.setdefault(parse_obs_ref(ts.refs[0][j]), pos)
    for k, e in enumerate(ents):
        assert by_obs[k] == {reg.slot[e]} and by_obs[len(ents) + k] == {reg.slot[e]}
    # entity keys from the bound observations only (mentions are not bound: entity slot -1)
    bound_only = ts.select([0])
    keys, present = slot_mean_keys(bound_only, cfg.max_entities)
    assert int(present.sum()) == len(ents)
    mention_rows = [(j, pos) for j, pos in labels[0] if parse_obs_ref(ts.refs[0][j]) >= len(ents)]
    obj = EntityGroundingObjective(cfg.d_model).to(cuda)
    hist, info = _train(lambda: obj(ts, keys, present, [mention_rows]), obj.parameters(), 60)
    assert info["n"] == len(ents) * cfg.sem_tokens_per_obs
    assert hist[-1] < hist[0] - 0.5 and info["acc"] >= 0.8, (hist[::10], info)


def test_code_execution_matching_on_real_sandbox_runs(tok, cfg, cuda):
    from superlearner.cam_unified.semantic.sandbox import CodeSandbox, crucible_sandbox_capabilities
    if not crucible_sandbox_capabilities().get("available"):
        pytest.skip("sandbox unavailable")
    sbx = CodeSandbox()
    progs = ["print(2 + 2)", "print('hello ' * 2)", "print([i * i for i in range(5)])", "x = 1 / 0",
             "import math\nprint(round(math.pi, 3))", "print(sorted('banana'))", "raise ValueError('bad input')",
             "print(sum(range(100)))"]
    results = [sbx.run(code=p).observations for p in progs]
    assert sum(r[0].kind == ObsKind.ERROR for r in results) == 2
    tok.register_tools([sample_tools(3)[2]])
    with torch.no_grad():
        C = tok.action([Action("run_python", {"code": p}) for p in progs])
        E = tok.observations(results, None, [0.0] * len(progs))
    obj = CodeExecutionObjective(cfg.d_model).to(cuda)
    hist, info = _train(lambda: obj(C, E), obj.parameters(), 80)
    assert hist[-1] < hist[0] - 1.0 and info["acc"] >= 0.75, (hist[::10], info)


def test_evidence_retrieval_supervision_trains_the_pointer(gen, tok, cfg):
    tool_lists = [sample_tools(5)] * 2
    obs_lists = [sample_observations(7, seed=0), sample_observations(7, seed=1)]
    R, goal, belief = make_context(cfg, 2, seed=30)
    with torch.no_grad():
        obs = tok.observations(obs_lists, None, [0.0, 0.0])
        tools = tok.tools(tool_lists)
    labels = [[3], [5, 0]]                                  # the error log; the message + the text
    state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    try:
        ev_params = [gen.ev_query, *gen.ev_tool.parameters(), *gen.ev_reader.parameters(), *gen.ev_q.parameters(),
                     *gen.ev_k.parameters(), *gen.ev_stop.parameters(), *gen.ev_update.parameters()]
        loss_fn = lambda: evidence_retrieval_loss(gen, R, goal, belief, tools, obs, tool_lists, obs_lists,
                                                  ["search_web", "send_message"], labels)
        hist, info = _train(loss_fn, ev_params, 40, lr=5e-3)
        assert hist[-1] < hist[0] - 1.0, hist[::8]
        # after training, greedy sampling picks the labelled evidence for the labelled tool
        mask = torch.zeros(2, len(gen.tool_lists_with_builtins(tool_lists)[0]), dtype=torch.bool, device=gen.device)
        mask[0, 0] = True
        mask[1, 3] = True
        acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, greedy=True, tool_mask=mask)
        assert acts[0].logp_parts["decode"]["evidence"] == [3]
        assert sorted(acts[1].logp_parts["decode"]["evidence"]) == [0, 5]
    finally:
        gen.load_state_dict(state)
