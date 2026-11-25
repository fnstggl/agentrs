"""Semantic learning objectives (spec 2.3): each loss is computed on real components and decreases when trained."""
from __future__ import annotations

import random

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


SUBJECTS = ["The bakery", "A GPU cloud", "The forum", "Our repository", "The mayor", "A payments startup", "The lab",
            "Customer seg_A", "The warehouse", "A translator", "The reactor", "The ad campaign"]
FACTS = ["raised prices by 3 percent.", "reported a failing deploy.", "asked for a refund.", "hired two engineers.",
         "lost its biggest client.", "doubled its weekly sales.", "moved to a new office.", "published a bug report."]


def _pool_tokens(tok, n):
    """Q pooled tokens of n distinct texts (encoded once): [n, Q, D]."""
    texts = [f"{SUBJECTS[i % len(SUBJECTS)]} {FACTS[(i // len(SUBJECTS) + i) % len(FACTS)]} (case {i})"
             for i in range(n)]
    with torch.no_grad():
        ts = tok.observations([[Observation(ObsKind.TEXT, t, t=0.0, source="feed")] for t in texts], None, [0.0] * n)
    return ts.x.detach()


def _pair_batch(pool, B, informative, g, cfg):
    """Rows of 3 observation pairs; the two members of a pair share a time stamp. informative: the partner has the
    same content; otherwise the partner is an unrelated text (identical structure, no information)."""
    from superlearner.cam_unified.interfaces import TokenSet
    n, Q, D = pool.shape
    xs, times, refs = [], [], []
    for _ in range(B):
        idx = torch.randperm(n, generator=g)[:6].tolist()
        members = [(idx[k], idx[k] if informative else idx[k + 3]) for k in range(3)]
        tvals = (-torch.randperm(6, generator=g)[:3].float() - 1.0).tolist()       # 3 distinct days ago
        seq = []
        for k, (a, b2) in enumerate(members):
            seq += [(a, tvals[k]), (b2, tvals[k])]
        order = torch.randperm(6, generator=g).tolist()
        seq = [seq[o] for o in order]
        xs.append(torch.cat([pool[i] for i, _ in seq], 0))
        times.append(torch.tensor([t for _, t in seq]).repeat_interleave(Q))
        refs.append([f"obs:{j}" for j in range(6) for _ in range(Q)])
    x = torch.stack(xs)
    dev = x.device
    N = x.shape[1]
    return TokenSet(x, torch.ones(B, N, dtype=torch.bool, device=dev),
                    torch.full((B, N), int(TokenKind.OBS_SEM), dtype=torch.long, device=dev),
                    torch.stack(times).to(dev), torch.full((B, N), -1, dtype=torch.long, device=dev), refs)


def test_masked_observation_prediction_uses_context_not_siblings(tok, cfg, cuda):
    """Whole observations are masked, so the target cannot be read off its own sibling tokens: with an uninformative
    context the right observation is identified at chance level; when another observation carries the content, the
    objective learns to retrieve it."""
    pool = _pool_tokens(tok, 48)
    Q = cfg.sem_tokens_per_obs
    results = {}
    for informative in (True, False):
        torch.manual_seed(0)
        core = DoubleCore(cfg).to(cuda)
        freqs = torch.exp(torch.linspace(-1.0, 1.5, 8, device=cuda))
        tenc = torch.nn.Linear(16, cfg.d_model).to(cuda)

        def core_fn(ctx):                          # the double has no time encoding; B's backbone encodes it
            ang = ctx.time.unsqueeze(-1) * freqs
            x = ctx.x + tenc(torch.cat([torch.sin(ang), torch.cos(ang)], -1))
            return core(core.init_state(ctx.B), type(ctx)(x, ctx.mask, ctx.kind, ctx.time, ctx.entity, ctx.refs))[1]

        obj = MaskedSemanticObjective(cfg.d_model, mask_frac=0.0).to(cuda)       # exactly one observation per row
        g = torch.Generator().manual_seed(1)
        opt = torch.optim.Adam(list(core.parameters()) + list(obj.parameters()) + list(tenc.parameters()), lr=3e-3)
        for _ in range(350):
            loss, info = obj(core_fn, _pair_batch(pool, 16, informative, g, cfg))
            assert info["n_masked_obs"] == 16 and info["n_masked"] == 16 * Q      # all Q tokens of each masked obs
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            ev = [obj(core_fn, _pair_batch(pool, 16, informative, g, cfg)) for _ in range(8)]
        results[informative] = (sum(float(l) for l, _ in ev) / 8, sum(i["obs_acc"] for _, i in ev) / 8)
    (l_inf, acc_inf), (l_un, acc_un) = results[True], results[False]
    print(f"\nmasked-observation prediction: informative loss {l_inf:.2f} obs_acc {acc_inf:.2f}; "
          f"uninformative loss {l_un:.2f} obs_acc {acc_un:.2f} (chance {1 / 16:.2f})")
    assert acc_un < 0.25 and acc_inf > 0.6 and l_inf < l_un - 1.0
    gi = MaskedSemanticObjective.group_index(_pair_batch(pool, 1, True, torch.Generator().manual_seed(0), cfg))
    assert gi[0, :Q].tolist() == list(range(Q))


WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima",
         "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
         "xray", "yankee", "zulu", "apple", "banana", "cherry", "grape", "lemon", "mango"]


def test_tool_consequence_prediction_generalises_to_held_out_actions(tok, cfg, cuda):
    """Trained on 24 (action, result) pairs, the predictor ranks the right result first for 8 held-out actions it
    never saw (chance 1/8) - not a memorised table."""
    words = list(WORDS)
    random.Random(0).shuffle(words)

    def make(ws):
        acts = [Action("send_message", {"to": "cust_7", "text": f"Please confirm the {w} order."}) for w in ws]
        res = [[Observation(ObsKind.TOOL_RESULT, f'{{"ok": true, "delivered": "Please confirm the {w} order.", '
                                                 f'"thread": "{w}-thread"}}', source="send_message")] for w in ws]
        with torch.no_grad():
            return (tok.action(acts, tool_lists=[sample_tools(5)] * len(ws)),
                    tok.observations(res, None, [0.0] * len(ws)))

    (A, Rt), (A_te, R_te) = make(words[:24]), make(words[24:])
    zeros = lambda n: torch.zeros(n, cfg.d_model, device=cuda)
    before, held = [], []
    for seed in range(3):                                   # mean over initialisations (single runs vary 0.375-0.875)
        torch.manual_seed(seed)
        obj = ToolConsequenceObjective(cfg.d_model, n_heads=cfg.n_heads).to(cuda)
        with torch.no_grad():
            before.append(obj(zeros(8), A_te, R_te)[1]["acc"])
        hist, info = _train(lambda: obj(zeros(24), A, Rt), obj.parameters(), 150)
        assert hist[-1] < hist[0] - 1.0 and info["acc"] == 1.0, (hist[::15], info)
        with torch.no_grad():
            held.append(obj(zeros(8), A_te, R_te)[1]["acc"])
    mb, mh = sum(before) / 3, sum(held) / 3
    print(f"\ntool consequence held-out top-1: before {before} after {held} (chance 0.125)")
    assert mh >= 0.5 and mh >= mb + 0.25, (before, held)


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


def test_code_execution_matching_generalises_to_held_out_programs(tok, cfg, cuda):
    """Code <-> real sandbox execution matching: trained on 24 programs of four kinds (string output, raised
    ValueError, sorted characters, KeyError), it matches 8 held-out programs to their executions."""
    from superlearner.cam_unified.semantic.sandbox import CodeSandbox, crucible_sandbox_capabilities
    if not crucible_sandbox_capabilities().get("available"):
        pytest.skip("sandbox unavailable")
    sbx = CodeSandbox()
    words = list(WORDS)
    random.Random(0).shuffle(words)
    progs = []
    for i, w in enumerate(words):
        progs.append([f"print('{w}' * 2)", f"raise ValueError('{w} is not allowed')", f"print(sorted('{w}'))",
                      f"d = {{'{w}': 1}}\nprint(d['{w[::-1]}'])"][i % 4])
    results = [sbx.run(code=p).observations for p in progs]
    assert sum(r[0].kind == ObsKind.ERROR for r in results) == 16          # ValueError and KeyError programs
    with torch.no_grad():
        C = tok.action([Action("run_python", {"code": p}) for p in progs], tool_lists=[sample_tools(3)] * len(progs))
        E = tok.observations(results, None, [0.0] * len(progs))
    tr, te = list(range(24)), list(range(24, 32))
    before, held = [], []
    for seed in range(3):                                   # mean over initialisations
        torch.manual_seed(seed)
        obj = CodeExecutionObjective(cfg.d_model).to(cuda)
        with torch.no_grad():
            before.append(obj(C.select(te), E.select(te))[1]["acc"])
        hist, info = _train(lambda: obj(C.select(tr), E.select(tr)), obj.parameters(), 150)
        assert hist[-1] < hist[0] - 1.0 and info["acc"] >= 0.9, (hist[::15], info)
        with torch.no_grad():
            held.append(obj(C.select(te), E.select(te))[1]["acc"])
    mb, mh = sum(before) / 3, sum(held) / 3
    print(f"\ncode<->execution held-out top-1: before {before} after {held} (chance 0.125)")
    assert mh >= 0.5 and mh >= mb + 0.25, (before, held)


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
