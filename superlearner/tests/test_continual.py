"""Continual learning in the weights: lifetime runner, update rule, backbone, plasticity instruments, the PIT."""
import copy
import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch

from superlearner.algo import base as algo_base
from superlearner.algo.continual import ContinualPPO, anchor_penalty, param_group
from superlearner.cam.agent import AgentConfig, CAMAgent, ModelChooser
from superlearner.cam.batch import collate, evaluate
from superlearner.cam.model import CAMController, EmbStore, ModelConfig
from superlearner.cam.worldmodel import joint_loss
from superlearner.cli import load_config
from superlearner.continual import plasticity as PL
from superlearner.continual.lifetime import Lifetime, derive
from superlearner.llm.server import EmbedOnly
from superlearner.train.demos import TeacherChooser
from superlearner.world.core import CoreConfig, SessionCore
from superlearner.world.tasks import task

SMALL = {"d": 64, "layers": 2, "heads": 4, "ff": 128, "bag_dim": 32, "hid": 64}
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "..", "configs")


def _episodes(family="found", n=4, teacher=True, start="scratch", seed0=40):
    torch.manual_seed(0)
    model = CAMController(ModelConfig(**SMALL))
    model.eval()
    store = EmbStore()
    ag = CAMAgent(model, store, EmbedOnly(), AgentConfig(use_llm=False, use_memory=False, use_tools=False))
    eps = []
    for i in range(n):
        core = SessionCore(task(family, "tiny", seed0 + i, start=start), CoreConfig(max_decisions=10))
        saved = ag.model
        if teacher:
            ag.model = None
        eps.append(ag.run(core, TeacherChooser("auto", i) if teacher else ModelChooser(seed=i), f"e{i}"))
        ag.model = saved
    return model, store, eps


def _algo(model, **over):
    cfg = {"lr": 1e-3, "epochs": 1, "minibatch_eps": 4, "critic_warmup": 0, "detach_critic": True, "sil_steps": 1, "sil_batch": 2,
           "wm": {"enabled": True, "replay_steps": 1, "replay_batch": 2}, "observe": {"steps": 6, "lr": 1e-3}}
    cfg.update(over)
    return algo_base.make("continual", model, cfg, "cpu")


# ------------------------------------------------------------------ the update rule
def test_every_controller_parameter_belongs_to_a_group_and_groups_can_be_frozen():
    model = CAMController(ModelConfig(**SMALL))
    a = _algo(model)
    sizes = a.group_sizes()
    assert sum(v for k, v in sizes.items() if k != "wm") == sum(p.numel() for p in model.parameters())
    assert {param_group(n) for n, _ in model.named_parameters()} == {"input", "encoder", "recurrent", "policy", "critic"}
    b = _algo(CAMController(ModelConfig(**SMALL)), group_lr={"encoder": 0.0, "input": 0.0})
    assert "encoder" not in b.group_sizes() and "input" not in b.group_sizes()
    assert not any(p.requires_grad for n, p in b.model.named_parameters() if param_group(n) == "encoder")


def test_observation_changes_the_weights_and_trains_the_world_model_online():
    model, store, eps = _episodes("found", 4)
    a = _algo(model)
    before, wm_before = PL.state_of(model), copy.deepcopy(a.wm.state_dict())
    hist = a.observe(eps, store, 12)
    assert hist[-1]["bc_nll"] < hist[0]["bc_nll"]                      # imitation of what it was shown
    assert hist[-1]["wm_loss"] < hist[0]["wm_loss"]                    # ... and its model of consequences, in the same steps
    ch = PL.param_change(model, before)
    assert ch["all"]["l2"] > 0 and ch["policy"]["relative"] > 0 and ch["encoder"]["relative"] > 0
    assert any((wm_before[k] != v).any() for k, v in a.wm.state_dict().items())
    assert len(a.buffer) == 4 and all(e["summary"].get("demo") for e in a.buffer)   # shown trajectories are rehearsable


def test_world_model_gradient_reaches_the_shared_representation_only_when_asked():
    model, store, eps = _episodes("found", 2)
    a = _algo(model)
    b = collate([e["decisions"] for e in eps], store, "cpu")
    for repr_scale, expect in ((0.0, False), (0.5, True)):
        model.zero_grad(set_to_none=True)
        model.train()
        out = evaluate(model, b, store)
        loss, info = joint_loss(a.wm, out, b.t, 0.99, repr_scale)
        loss.backward()
        got = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.encoder.parameters())
        assert got is expect and "wm_ok_acc" in info


def test_reservoir_keeps_every_arm_and_naive_learners_drop_old_arms_at_a_phase_boundary():
    model, store, e1 = _episodes("found", 3)
    _, _, e2 = _episodes("fund", 3)
    for keep, expect in ((True, 2), (False, 1)):
        a = _algo(CAMController(ModelConfig(**SMALL)), rehearsal={"keep": keep, "per_arm": 2})
        a._reservoir(copy.deepcopy(e1))
        a._reservoir(copy.deepcopy(e2))
        arms = {tuple(e["summary"]["arm"]) for e in a.buffer}
        assert len(arms) == 2 and len(a.buffer) == 4               # two arms, at most two episodes of each
        a.begin_phase(1, [("fund", 0, "scratch")])
        assert len({tuple(e["summary"]["arm"]) for e in a.buffer}) == expect


def test_consolidation_anchor_is_zero_at_the_anchor_and_grows_with_distance_and_importance():
    model, store, eps = _episodes("found", 4)
    for kind in ("l2sp", "ewc"):
        m = copy.deepcopy(model)
        a = _algo(m, anchor={"kind": kind, "coef": 10.0, "fisher_episodes": 4, "top_frac": 1.0})
        a._reservoir(copy.deepcopy(eps))
        info = a.end_phase(store)
        assert info["anchor"] == kind
        p0, _ = a.penalty()
        assert float(p0.detach()) == 0.0
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01)
        p1, i1 = a.penalty()
        assert float(p1.detach()) > 0 and i1["anchor_pen"] > 0
    sel = _algo(copy.deepcopy(model), anchor={"kind": "ewc", "coef": 10.0, "fisher_episodes": 4, "top_frac": 0.05})
    sel._reservoir(copy.deepcopy(eps))
    info = sel.end_phase(store)
    assert info["fisher_nonzero"] <= 0.06                              # selective: only the most important weights are held
    ps = [torch.zeros(3), torch.ones(2)]
    assert float(anchor_penalty([p + 1 for p in ps], ps, None)) == pytest.approx(1.0)


def test_exploratory_episodes_never_enter_the_policy_gradient_term():
    model, store, eps = _episodes("found", 4, teacher=False)
    a = _algo(model)
    for e in eps[2:]:
        e["summary"]["explore"] = True
    seen = {}
    orig = a._update_on_policy
    a._update_on_policy = lambda own, st, demos=None: (seen.setdefault("n", len(own)), orig(own, st, demos))[1]
    m = a.update(eps, store)
    assert seen["n"] == 2 and len(a.experience) == 4 and m["wm_replay_steps"] == 1


def test_update_rule_state_survives_a_save_and_load():
    model, store, eps = _episodes("found", 4)
    a = _algo(model, anchor={"kind": "ewc", "coef": 5.0, "fisher_episodes": 4}, l2init=0.1)
    a.observe(eps, store, 4)
    a.begin_phase(0, [("found", 0, "scratch")])
    a.end_phase(store)
    s, bufs = copy.deepcopy(a.state_dict()), a.buffers_state()
    m2 = CAMController(ModelConfig(**SMALL))
    m2.load_state_dict(model.state_dict())
    b = _algo(m2, anchor={"kind": "ewc", "coef": 5.0, "fisher_episodes": 4}, l2init=0.1)
    b.load_state_dict(s)
    b.load_buffers_state(bufs)
    assert b.n_updates == a.n_updates and len(b.buffer) == len(a.buffer) and b.fisher is not None and b.anchors is not None
    assert all(torch.equal(x, y) for x, y in zip(a.anchors, b.anchors))
    sa, sb = a.opt.state_dict()["state"], b.opt.state_dict()["state"]
    assert len(sa) == len(sb) and all(torch.equal(sa[k]["exp_avg"], sb[k]["exp_avg"]) for k in sa)   # Adam's moments
    assert all(torch.equal(x, y) for x, y in zip(a.wm.state_dict().values(), b.wm.state_dict().values()))


# ------------------------------------------------------------------ instruments
def test_plasticity_instruments_are_finite_and_deterministic():
    model, store, eps = _episodes("found", 3)
    d1, d2 = PL.diagnostics(model, eps, store), PL.diagnostics(model, eps, store)
    assert d1 == d2 and 0.0 <= d1["dormant_fraction"] <= 1.0 and 1.0 <= d1["state_effective_rank"] <= 64
    assert set(d1["weight_norm"]) == {"input", "encoder", "recurrent", "policy", "critic"}
    f1, f2 = PL.probe_fit(model, eps, store, steps=6), PL.probe_fit(model, eps, store, steps=6)
    assert f1 == f2 and f1["loss_last"] < f1["loss_first"]             # a healthy network can fit new targets
    dig = PL.weights_digest(model)
    with torch.no_grad():
        next(model.parameters()).add_(1e-3)
    assert PL.weights_digest(model) != dig


def test_retention_matrix_measures():
    rows = [{"lists": {"A": {"success": 1.0}, "B": {"success": 0.25}}}, {"lists": {"A": {"success": 0.5}, "B": {"success": 0.75}}}]
    d = derive(rows, [{"name": "A"}, {"name": "B"}])
    assert d["acquisition"] == [1.0, 0.5] and d["zero_shot_before_training"] == [0.0, 0.25]
    assert d["forgetting"] == [0.5, 0.0] and d["mean_forgetting"] == 0.5 and d["backward_transfer"] == -0.5
    assert d["mean_final_seen"] == pytest.approx(0.625)


def test_recurrent_state_starts_from_zero_in_every_episode():
    model, store, _ = _episodes("found", 1)
    ag = CAMAgent(model, store, EmbedOnly(), AgentConfig(use_llm=False, use_memory=False, use_tools=False))
    ag.run(SessionCore(task("found", "tiny", 71), CoreConfig(max_decisions=4)), ModelChooser(seed=0), "a")
    assert float(ag.state[0].abs().sum()) > 0                          # the state moved during the episode
    ag.begin(SessionCore(task("found", "tiny", 72), CoreConfig(max_decisions=4)), "b")
    assert float(ag.state[0].abs().sum()) == 0 and float(ag.state[1].abs().sum()) == 0


# ------------------------------------------------------------------ lifetime
@pytest.fixture(scope="module")
def smoke_cfg():
    return load_config(os.path.join(CFG, "life_smoke.yaml"))


def test_a_lifetime_killed_and_restarted_ends_with_the_same_weights(smoke_cfg, tmp_path):
    cfg = copy.deepcopy(smoke_cfg)
    cfg["lifetime"]["phases"] = cfg["lifetime"]["phases"][:2]
    cfg["lifetime"]["eval"]["sampled"] = False
    a = Lifetime(cfg, str(tmp_path / "whole"), verbose=False)
    while a.step():
        pass
    da, ra = PL.weights_digest(a.tr.model), a.x["rows"]
    a.tr.close()
    b = Lifetime(cfg, str(tmp_path / "cut"), verbose=False)
    for _ in range(3):                                                 # begin + observe, practice, close of the first phase
        b.step()
    b.tr.close()
    del b
    c = Lifetime(cfg, str(tmp_path / "cut"), verbose=False)            # a new object: everything comes from the file
    assert c.x["phase"] == 1 and c.x["restarts"] == 1 and len(c.x["rows"]) == 1
    assert c.tr.algo.n_updates > 0 and len(c.tr.algo.buffer) > 0 and c.tr.algo.anchors is not None
    while c.step():
        pass
    dc, rc = PL.weights_digest(c.tr.model), c.x["rows"]
    res = json.load(open(tmp_path / "cut" / "results.json"))
    c.tr.close()
    assert dc == da                                                    # bit-identical weights after the restart
    assert [r["lists"] for r in ra] == [{k: dict(v, wall_s=x["lists"][k]["wall_s"]) for k, v in r["lists"].items()}
                                        for r, x in zip(rc, ra)]
    assert res["finished"] and res["restarts"] == 1 and len(res["derived"]["matrix"]) == 2
    assert da != a.x["digest_init"]                                    # and learning did change them
    row = ra[-1]
    assert row["weights"]["since_init"]["all"]["l2"] > 0 and row["world_model"] and "probe_fit" in row["plasticity"]


def test_frozen_core_condition_never_changes_weights_after_the_first_phase(smoke_cfg, tmp_path):
    cfg = copy.deepcopy(smoke_cfg)
    cfg["lifetime"].update(phases=cfg["lifetime"]["phases"][:2], memory=True, freeze_after=0)
    cfg["lifetime"]["eval"]["sampled"] = False
    life = Lifetime(cfg, str(tmp_path / "frozen"), verbose=False)
    while life.step():
        pass
    r0, r1 = life.x["rows"]
    life.tr.close()
    assert r0["weights"]["digest"] == r1["weights"]["digest"] and r1["frozen"] and not r0["frozen"]
    assert r1["memory_items"] > r0["memory_items"]                     # what it "learned" later is written, not in weights
    assert "lists_memory_off" in r1


# ------------------------------------------------------------------ trainable language backbone
def _tiny_backbone(mode):
    tf = pytest.importorskip("transformers")
    from superlearner.cam.backbone import TextBackbone
    try:
        tok = tf.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
        conf = tf.AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("the Qwen3-0.6B tokenizer is not in the local cache")
    conf.hidden_size, conf.intermediate_size, conf.num_hidden_layers = 32, 64, 5
    conf.num_attention_heads, conf.num_key_value_heads, conf.head_dim = 2, 2, 16
    torch.manual_seed(0)
    lm = tf.AutoModelForCausalLM.from_config(conf).model.float()
    return TextBackbone("tiny-test-double", mode, "cpu", embed_tokens=24, chunk_tokens=60, lm=lm, tok=tok)


TEXTS = ["Open a checking account.", "Deploy a repository as a service.", "insufficient funds", "Search the web of this world.",
         "Publish a listing on the market for a service you operate."]


def test_backbone_gradient_cache_equals_direct_backpropagation():
    bb = _tiny_backbone("full")
    assert bb.n_blocks == 2 and bb.n_params()["trainable"] == bb.n_params()["total"]
    ps = [p for p in bb.parameters() if p.requires_grad]
    w = torch.randn(len(TEXTS), 512)
    E = bb.embed(TEXTS).requires_grad_(True)
    (E * w).sum().backward()
    bb.backprop(TEXTS, E.grad)                                         # several chunks (chunk_tokens is small)
    g1 = torch.cat([p.grad.reshape(-1) for p in ps]).clone()
    for p in ps:
        p.grad = None
    bb.chunk_tokens = 10**9
    (bb._finish(bb._encode(TEXTS)) * w).sum().backward()
    g2 = torch.cat([p.grad.reshape(-1) for p in ps])
    assert len(bb.chunks(TEXTS)) == 1                                  # (padding differs between the two, so not bitwise)
    assert float(torch.nn.functional.cosine_similarity(g1, g2, dim=0)) > 0.9999
    assert float((g1 - g2).norm() / g2.norm()) < 1e-2


def test_backbone_modes_start_from_the_frozen_function_and_differ_in_what_can_learn():
    none, lora, top = _tiny_backbone("none"), _tiny_backbone("lora"), _tiny_backbone("top:1")
    e0 = none.embed(TEXTS)
    assert torch.allclose(e0, lora.embed(TEXTS), atol=1e-6) and torch.allclose(e0, top.embed(TEXTS), atol=1e-6)
    assert none.n_params()["trainable"] == 0 and not none.trainable
    assert 0 < lora.n_params()["trainable"] < top.n_params()["trainable"] < top.n_params()["total"]
    assert torch.allclose(e0.norm(dim=-1), torch.ones(len(TEXTS)), atol=1e-5)


def test_a_learning_step_moves_the_backbone_and_every_stored_text_vector():
    from superlearner.cam.backbone import LiveTable
    model, store, eps = _episodes("found", 3)
    bb = _tiny_backbone("full")
    texts = {}
    for key in list(store.key2row):                                    # give every stored key a text the backbone can read
        if key:
            texts[key] = f"text number {key % 97} about accounts and services"
            store.mat[store.key2row[key]] = bb.embed_np([texts[key]])[0]
    live = LiveTable(bb, store, texts)
    model.live = live
    a = _algo(model, backbone_lr=1e-3)
    a.attach_backbone(bb)
    assert a.group_sizes()["backbone"] == bb.n_params()["trainable"]
    w0 = {k: v.clone() for k, v in bb.trainable_state().items()}
    table0 = store.mat[: store.n].copy()
    hist = a.observe(eps, store, 3)
    assert live.n_flush == 3 and hist[-1]["bc_nll"] == hist[-1]["bc_nll"]
    moved = sum(float((v - w0[k]).abs().sum()) for k, v in bb.trainable_state().items())
    assert moved > 0                                                   # the language backbone's own weights changed
    v0 = bb.version
    n = live.refresh_store()
    assert n == len(texts) and bb.version == v0 + 1 and float(np.abs(store.mat[: store.n] - table0).sum()) > 0
    state = bb.trainable_state()
    bb2 = _tiny_backbone("full")
    bb2.load_trainable_state(state)                                    # and they persist through a file-shaped round trip
    assert torch.allclose(bb2.embed(TEXTS), bb.embed(TEXTS), atol=1e-6)


# ------------------------------------------------------------------ the Parametric Internalization Test
def test_pit_exports_tensors_only_and_evaluates_them_in_a_new_process():
    from superlearner.continual.internalize import run_route
    cfg = load_config(os.path.join(CFG, "pit_smoke.yaml"))
    cfg["pit"]["base_phase"].update(observe_steps=8)
    cfg["pit"]["routes"] = {"observe": {"demos_per_arm": 2, "observe_steps": 8, "iters": 0}}
    cfg["pit"]["eval"].update(n=2, n_unseen=1, n_related=1, n_familiar=1, handle_control="")
    out = tempfile.mkdtemp(dir="/var/tmp")
    try:
        r = run_route(cfg, out, 0, "fund", "observe", workers=0)
        w = torch.load(os.path.join(out, "s0", "fund__observe", "weights.pt"), map_location="cpu", weights_only=False)
        assert set(w) == {"model", "wm", "backbone", "digest"} and all(torch.is_tensor(v) for v in w["model"].values())
        assert r["weights_changed"] and r["weights_before"]["digest"] != r["weights_after"]["digest"]
        assert r["parameter_change"]["all"]["l2"] > 0
        st = r["fresh_process"]["state_at_start"]
        assert st == {"memory_items": 0, "tools": 0, "embedding_rows": 0, "replay_buffer": 0, "texts": 0}
        assert r["fresh_process"]["pid"] != os.getpid()
        assert set(r["fresh_process"]["before"]) == {"unfamiliar", "unfamiliar_unseen_worlds", "related_untrained", "familiar"}
        assert len(r["fresh_process"]["after"]["unfamiliar"]["per_task"]) == 2
    finally:
        shutil.rmtree(out, ignore_errors=True)
