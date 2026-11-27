"""Atomic checkpoints: bit-exact round trip, crash safety, tolerant loading, weights-only export, promotion and
rollback (CL-05, CL-06). Plain nn.Modules here; the real model is wired in by the continual learner."""
import json
import os
import random
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest
import torch
from torch import nn

from cam_live.contracts import SimClock
from cam_live.credit import checkpoint as C
from cam_live.credit.replay import SqliteExperienceStore

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class Net(nn.Module):
    """Small stand-in with the same grouping interface as the real model (``param_groups`` / ``arch_signature``)."""

    def __init__(self, extra_head: bool = False, d: int = 16):
        super().__init__()
        self.core = nn.Sequential(nn.Linear(8, d), nn.Tanh(), nn.Linear(d, d))
        self.policy = nn.Linear(d, 4)
        self.reward = nn.Linear(d, 1)
        self.register_buffer("norm_mean", torch.zeros(d))
        self.register_buffer("scale", torch.tensor(2.5))            # zero-dimensional persistent buffer
        self.frozen = nn.Linear(4, 4)
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        if extra_head:
            self.new_head = nn.Linear(d, 3)

    def forward(self, x):
        h = self.core(x) - self.norm_mean
        return self.policy(h), self.reward(h)

    def param_groups(self):
        heads = list(self.policy.parameters()) + (list(self.new_head.parameters()) if hasattr(self, "new_head") else [])
        return {"backbone_frozen": list(self.frozen.parameters()), "experiential": list(self.core.parameters()),
                "heads": heads, "reward": list(self.reward.parameters())}

    def arch_signature(self):
        return {"d": self.policy.in_features, "heads": sorted(n for n, _ in self.named_children())}


def _optims(net):
    g = net.param_groups()
    return {"main": torch.optim.AdamW([{"params": g["experiential"], "lr": 1e-2}, {"params": g["heads"], "lr": 3e-2}],
                                      betas=(0.8, 0.95), weight_decay=0.01),
            "reward": torch.optim.Adam(g["reward"], lr=5e-3)}


def _steps(net, opts, n, gen):
    for _ in range(n):
        x = torch.randn(32, 8, generator=gen)
        pol, rew = net(x)
        loss = pol.pow(2).mean() + (rew - 1.0).pow(2).mean()
        for o in opts.values():
            o.zero_grad()
        loss.backward()
        for o in opts.values():
            o.step()


def _trained(seed=0, steps=5, **kw):
    torch.manual_seed(seed)
    net = Net(**kw)
    opts = _optims(net)
    gen = torch.Generator().manual_seed(seed + 1)
    _steps(net, opts, steps, gen)
    net.norm_mean.add_(0.25)
    return net, opts, gen


def _full_state(net, opts, gen, **kw):
    return C.capture_state(
        net, opts, generators={"data": gen},
        fast_state={"mu": torch.randn(3, 5, dtype=torch.float64), "Sigma": torch.eye(5, dtype=torch.float64) * 0.37,
                    "slots": torch.randn(1, 4, 16), "n_obs": 17},
        lifetime_context=torch.randn(2, 16), current_task={"objective_id": "obj-7", "objective": "an objective text"},
        schedules={"frequent_updates": 12, "adapter_every": 8, "bootstrap_mix": 0.75, "lr": {"main": [1e-2, 3e-2]}},
        auth_config_hash="authhash-abc", model_version="m-12", reward_version="r-3", extra={"note": "x"}, **kw)


def _same(a, b):
    assert a.keys() == b.keys()
    for g in a:
        assert a[g].keys() == b[g].keys(), g
        for n in a[g]:
            assert a[g][n].dtype == b[g][n].dtype and torch.equal(a[g][n], b[g][n]), (g, n)


def test_round_trip_is_bit_exact_and_training_continues_identically(tmp_path):
    net, opts, gen = _trained()
    random.seed(5), np.random.seed(6)
    state = _full_state(net, opts, gen)
    assert set(state.weights) == {"experiential", "heads", "reward", "buffers"}          # no frozen backbone
    assert not any(n.startswith("frozen") for d in state.weights.values() for n in d)
    info = C.save_checkpoint(str(tmp_path), state, notes="first", clock=SimClock(123.0))
    files = sorted(os.path.relpath(os.path.join(d, f), info.path) for d, _, fs in os.walk(info.path) for f in fs)
    assert files == ["MANIFEST.json", "fast_state.pt", "lifetime.pt", "optim/main.pt", "optim/reward.pt", "rng.pt",
                     "state.json", "weights/buffers.safetensors", "weights/experiential.safetensors",
                     "weights/heads.safetensors", "weights/reward.safetensors"]
    assert C.verify_checkpoint(info.path) == [] and info.created_at == 123.0
    assert info.group_hashes == {g: C.tensor_hash(d) for g, d in state.weights.items()}
    # what the original process does next
    expected_draws = (random.random(), float(np.random.rand()), torch.rand(3), torch.rand(2, generator=gen))
    _steps(net, opts, 4, gen)
    # a fresh process state: different init, different RNG positions
    torch.manual_seed(999), random.seed(999), np.random.seed(999)
    net2 = Net()
    opts2, gen2 = _optims(net2), torch.Generator().manual_seed(4242)
    loaded = C.load_checkpoint(str(tmp_path))
    _same(loaded.weights, state.weights)
    rep = C.restore_state(loaded, net2, opts2, strict=True, generators={"data": gen2}, expected_auth_hash="authhash-abc")
    assert rep["weights"].clean and rep["arch_signature_differences"] == [] and rep["auth_config_changed"] is False
    assert rep["optimizers"]["main"]["fresh"] == [] and rep["optimizers"]["main"]["hyper"] == ["restored"]
    assert opts2["main"].param_groups[1]["lr"] == 3e-2 and opts2["main"].param_groups[0]["betas"] == (0.8, 0.95)
    for k in ("mu", "Sigma", "slots"):
        assert loaded.fast_state[k].dtype == state.fast_state[k].dtype and torch.equal(loaded.fast_state[k], state.fast_state[k])
    assert loaded.fast_state["n_obs"] == 17 and torch.equal(loaded.lifetime_context, state.lifetime_context)
    assert loaded.current_task == state.current_task and loaded.schedules == state.schedules
    assert (loaded.auth_config_hash, loaded.model_version, loaded.reward_version) == ("authhash-abc", "m-12", "r-3")
    assert loaded.arch_signature == net.arch_signature() and loaded.extra["checkpoint_id"] == info.checkpoint_id
    got_draws = (random.random(), float(np.random.rand()), torch.rand(3), torch.rand(2, generator=gen2))
    assert got_draws[0] == expected_draws[0] and got_draws[1] == expected_draws[1]
    assert torch.equal(got_draws[2], expected_draws[2]) and torch.equal(got_draws[3], expected_draws[3])
    _steps(net2, opts2, 4, gen2)                       # optimizer moments and RNG were restored exactly
    for (n1, p1), (n2, p2) in zip(net.named_parameters(), net2.named_parameters()):
        if p1.requires_grad:            # the frozen backbone is not part of a checkpoint (it is a pinned snapshot)
            assert torch.equal(p1, p2), n1
    assert C.restore_state(loaded, expected_auth_hash="other", rng=False)["auth_config_changed"] is True


CRASHER = """
import os, sys, torch
sys.path.insert(0, {src!r})
from cam_live.credit import checkpoint as C
def state(v):
    return C.CheckpointState(weights={{"experiential": {{"w": torch.full((64, 64), float(v))}}, "heads": {{"h": torch.full((8,), float(v))}}}},
                             fast_state={{"mu": torch.full((4,), float(v), dtype=torch.float64)}}, schedules={{"step": v}},
                             model_version="m%d" % v)
C.save_checkpoint({root!r}, state(1))
def hook(point):
    if point == {point!r}:
        os._exit(9)
C.FAULT_HOOK = hook
C.save_checkpoint({root!r}, state(2))
print("FINISHED")
"""


@pytest.mark.parametrize("point,expect", [("weights", 1), ("files", 1), ("before_rename", 1), ("after_rename", 2),
                                          ("after_latest", 2)])
def test_crash_at_every_step_of_a_save_leaves_a_loadable_checkpoint(tmp_path, point, expect):
    root = str(tmp_path / "ckpt")
    out = subprocess.run([sys.executable, "-c", CRASHER.format(src=SRC, root=root, point=point)], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 9 and "FINISHED" not in out.stdout, out.stderr
    debris = [n for n in os.listdir(root) if n.startswith(".tmp-")]
    assert bool(debris) == (expect == 1)                         # an unfinished save leaves only a temp directory
    # loading works immediately, even before any clean-up, and gives a complete state
    st = C.load_checkpoint(root)
    assert st.schedules == {"step": expect} and st.model_version == f"m{expect}"
    assert float(st.weights["experiential"]["w"].unique()) == float(expect) and float(st.fast_state["mu"][0]) == float(expect)
    rep = C.recover(root)
    assert not [n for n in os.listdir(root) if n.startswith(".tmp-")] and rep["corrupt"] == []
    infos = C.list_checkpoints(root)
    assert [i.seq for i in infos] == list(range(1, expect + 1))      # the previous checkpoint is always kept
    assert open(os.path.join(root, "LATEST")).read() == infos[-1].checkpoint_id == rep["latest"]
    # and saving continues with the next sequence number
    nxt = C.save_checkpoint(root, st)
    assert nxt.seq == expect + 1


LOOPER = """
import sys, torch
sys.path.insert(0, {src!r})
from cam_live.credit import checkpoint as C
mgr = C.CheckpointManager({root!r}, keep_last=3)
k = 0
while True:
    k += 1
    st = C.CheckpointState(weights={{"experiential": {{"w": torch.full((1200, 1200), float(k))}}}}, schedules={{"step": k}})
    info = mgr.save(st)
    print(k, info.checkpoint_id, flush=True)
"""


@pytest.mark.parametrize("delay", [0.0, 0.004, 0.011, 0.023])
def test_kill_9_during_save_leaves_a_loadable_checkpoint(tmp_path, delay):
    root = str(tmp_path / "ckpt")
    proc = subprocess.Popen([sys.executable, "-c", LOOPER.format(src=SRC, root=root)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    acked = {}
    try:
        while len(acked) < 3:
            line = proc.stdout.readline()
            assert line, proc.stderr.read()
            k, cid = line.split()
            acked[int(k)] = cid
        time.sleep(delay)                                   # somewhere inside the next save
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
    last = max(acked)
    st = C.load_checkpoint(root)                            # no recovery step needed to load
    step = st.schedules["step"]
    assert step >= last and float(st.weights["experiential"]["w"].unique()) == float(step)
    mgr = C.CheckpointManager(root, keep_last=3)            # recovery on open
    assert not [n for n in os.listdir(root) if n.startswith((".tmp-", ".trash-"))]
    infos = mgr.list()
    assert len(infos) >= 2 and all(C.verify_checkpoint(i.path) == [] for i in infos)
    assert mgr.latest().checkpoint_id == open(os.path.join(root, "LATEST")).read()
    assert acked[last] in {i.checkpoint_id for i in infos}


def test_damaged_newest_checkpoint_falls_back_to_the_previous_one(tmp_path):
    root = str(tmp_path)
    mk = lambda v: C.CheckpointState(weights={"heads": {"w": torch.full((300,), float(v))}}, schedules={"step": v})
    a, b = C.save_checkpoint(root, mk(1)), C.save_checkpoint(root, mk(2))
    wf = os.path.join(b.path, "weights", "heads.safetensors")
    raw = bytearray(open(wf, "rb").read())
    raw[-40] ^= 0xFF                                           # silent bit rot in the newest weights
    open(wf, "wb").write(bytes(raw))
    assert C.verify_checkpoint(b.path) == ["weights/heads.safetensors: sha256 mismatch"]
    assert C.latest_valid(root).checkpoint_id == a.checkpoint_id
    assert C.load_checkpoint(root).schedules == {"step": 1}
    with pytest.raises(C.CheckpointError):
        C.load_checkpoint(root, b.checkpoint_id)
    rep = C.recover(root)
    assert rep["latest"] == a.checkpoint_id and rep["corrupt"][0]["checkpoint_id"] == b.checkpoint_id
    with pytest.raises(C.CheckpointError):
        C.load_checkpoint(str(tmp_path / "empty"))


def test_tolerant_loading_reports_added_removed_and_reshaped_modules(tmp_path):
    net, opts, gen = _trained()
    C.save_checkpoint(str(tmp_path), _full_state(net, opts, gen))
    loaded = C.load_checkpoint(str(tmp_path))
    # a model that gained a head since the checkpoint was written
    torch.manual_seed(3)
    bigger = Net(extra_head=True)
    init_new = {n: p.detach().clone() for n, p in bigger.new_head.named_parameters()}
    with pytest.raises(C.CheckpointMismatch) as err:
        C.load_weights(bigger, loaded.weights, strict=True)
    assert err.value.report.missing == ["new_head.bias", "new_head.weight"]
    with pytest.warns(UserWarning, match="tolerant checkpoint load"):
        rep = C.load_weights(bigger, loaded.weights)
    assert rep.missing == ["new_head.bias", "new_head.weight"] and rep.unexpected == [] and rep.mismatched == []
    assert len(rep.loaded) == 10                              # 4 core + 2 policy + 2 reward + 2 buffers
    for n, p in net.named_parameters():
        if p.requires_grad:
            assert torch.equal(dict(bigger.named_parameters())[n], p), n
    assert torch.equal(bigger.norm_mean, net.norm_mean)
    for n, p in bigger.new_head.named_parameters():
        assert torch.equal(p, init_new[n])                    # the new head keeps its initialisation
    # optimizer state follows parameter names: old parameters keep their moments, the new head starts fresh
    opts_b = _optims(bigger)
    with pytest.warns(UserWarning, match="tolerant optimizer load"):
        orep = C.load_optimizer_state(opts_b["main"], bigger, loaded.optimizers["main"])
    assert orep["fresh"] == ["new_head.weight", "new_head.bias"] and len(orep["restored"]) == 6 and orep["dropped"] == []
    for p_new, p_old in zip(bigger.core.parameters(), net.core.parameters()):
        assert torch.equal(opts_b["main"].state[p_new]["exp_avg_sq"], opts["main"].state[p_old]["exp_avg_sq"])
    _steps(bigger, opts_b, 2, gen)                            # and training simply continues
    with pytest.warns(UserWarning):
        full = C.restore_state(loaded, bigger, _optims(bigger), rng=False)
    assert full["arch_signature_differences"] == ["heads"] and full["weights"].missing == ["new_head.bias", "new_head.weight"]
    # a module that was removed, and one whose shape changed
    state_b = C.capture_state(bigger, opts_b)
    with pytest.warns(UserWarning):
        rep2 = C.load_weights(Net(), state_b.weights)
    assert rep2.unexpected == ["new_head.weight", "new_head.bias"] and rep2.missing == []
    with pytest.warns(UserWarning):
        rep3 = C.load_weights(Net(d=24), loaded.weights)
    assert ("policy.weight", (4, 16), (4, 24)) in rep3.mismatched and "core.0.weight" in [m[0] for m in rep3.mismatched]
    with pytest.warns(UserWarning):
        orep3 = C.load_optimizer_state(_optims(Net(d=24))["main"], Net(d=24), loaded.optimizers["main"])
    assert orep3["restored"] == []


FRESH = """
import sys, json, torch
sys.path.insert(0, {src!r}); sys.path.insert(0, {tests!r})
from test_checkpoint import Net
from cam_live.credit import checkpoint as C
torch.manual_seed(31337)
net = Net()
weights, cfg = C.load_weights_only({path!r})
rep = C.load_weights(net, weights, strict=True)
x = torch.linspace(-1, 1, 64).view(8, 8)
pol, rew = net(x)
print(json.dumps({{"pol": pol.flatten().tolist(), "rew": rew.flatten().tolist(), "sig": cfg["arch_signature"]}}))
"""


def test_weights_only_export_contains_only_parameters_and_fixed_configuration(tmp_path):
    net, opts, gen = _trained()
    state = _full_state(net, opts, gen)
    state.fast_state["mu"] = torch.full((3, 5), 1234.56789, dtype=torch.float64)       # recognisable bytes
    state.lifetime_context = torch.full((2, 16), -8765.4321)
    C.save_checkpoint(str(tmp_path / "ck"), state)
    out = str(tmp_path / "export")
    cfg = C.export_weights_only(C.load_checkpoint(str(tmp_path / "ck")), out, config={"scale": "debug"})
    assert sorted(os.listdir(out)) == ["config.json", "weights.safetensors"]
    expected = {f"{g}::{n}" for g, d in state.weights.items() for n in d}
    assert set(C.export_keys(out)) == expected and len(expected) == 10
    assert set(cfg) == {"format", "arch_signature", "model_version", "reward_version", "config", "group_hashes",
                        "n_tensors", "weights_sha256"}
    blob = open(os.path.join(out, "weights.safetensors"), "rb").read() + open(os.path.join(out, "config.json"), "rb").read()
    moment = opts["main"].state[next(net.core.parameters())]["exp_avg_sq"]
    for secret in (state.fast_state["mu"].numpy().tobytes()[:16], state.lifetime_context.numpy().tobytes()[:16],
                   moment.numpy().tobytes()[:32], b"an objective text", b"obj-7", b"authhash", b"exp_avg",
                   b"frequent_updates", b"draws"):
        assert secret not in blob
    assert next(net.core.parameters()).detach().numpy().tobytes()[:32] in blob        # the parameters are there
    # a fresh process with nothing but the export reproduces the network exactly
    res = subprocess.run([sys.executable, "-W", "ignore", "-c",
                          FRESH.format(src=SRC, tests=os.path.dirname(__file__), path=out)],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    pol, rew = net(torch.linspace(-1, 1, 64).view(8, 8))
    assert got["pol"] == pol.flatten().tolist() and got["rew"] == rew.flatten().tolist()
    assert got["sig"] == net.arch_signature()
    raw = bytearray(open(os.path.join(out, "weights.safetensors"), "rb").read())
    raw[-3] ^= 0x01
    open(os.path.join(out, "weights.safetensors"), "wb").write(bytes(raw))
    with pytest.raises(C.CheckpointError):
        C.load_weights_only(out)


def test_promotion_is_gated_by_validation_and_retention_and_rollback_restores_the_promoted_state(tmp_path):
    clock = SimClock(1000.0)
    mgr = C.CheckpointManager(str(tmp_path), keep_last=2, gates=C.PromotionGates(primary="success_rate",
                                                                               max_retention_drop=0.02), clock=clock)
    mk = lambda v: C.CheckpointState(weights={"heads": {"w": torch.full((16,), float(v))}}, schedules={"step": v})
    c1 = mgr.save(mk(1))
    assert mgr.promoted() is None
    with pytest.raises(C.CheckpointError):
        mgr.rollback()
    # internal reward is not evidence; missing metrics are not evidence
    d = mgr.promote(c1.checkpoint_id, validation={"intrinsic_reward": 99.0}, retention={})
    assert not d.promoted and d.ignored == ["intrinsic_reward"]
    assert d.reasons == ["no independent validation metric supplied", "no retention metric supplied"]
    assert not mgr.promote(c1.checkpoint_id, validation={"success_rate": float("nan")}, retention={"old_family": 0.8}).promoted
    assert not mgr.promote("ckpt_999999_deadbeef", validation={"success_rate": 0.5}, retention={"old_family": 0.8}).promoted
    d1 = mgr.promote(c1.checkpoint_id, validation={"success_rate": 0.50, "calibration": -0.20}, retention={"old_family": 0.80})
    assert d1.promoted and mgr.promoted().checkpoint_id == c1.checkpoint_id
    # better on the new objective but it forgot the old one: refused
    c2 = mgr.save(mk(2))
    d2 = mgr.promote(c2.checkpoint_id, validation={"success_rate": 0.70, "calibration": -0.20}, retention={"old_family": 0.60})
    assert not d2.promoted and "forgetting" in d2.reasons[0] and d2.deltas["retention.old_family"] == pytest.approx(-0.2)
    # retained but no better: refused; a retention metric that was not measured: refused
    assert not mgr.promote(c2.checkpoint_id, validation={"success_rate": 0.49, "calibration": -0.2}, retention={"old_family": 0.8}).promoted
    assert not mgr.promote(c2.checkpoint_id, validation={"success_rate": 0.9, "calibration": -0.2}, retention={}).promoted
    c3 = mgr.save(mk(3))
    d3 = mgr.promote(c3.checkpoint_id, validation={"success_rate": 0.66, "calibration": -0.20}, retention={"old_family": 0.79})
    assert d3.promoted and d3.baseline_id == c1.checkpoint_id and mgr.promoted().checkpoint_id == c3.checkpoint_id
    c4, c5 = mgr.save(mk(4)), mgr.save(mk(5))
    assert not mgr.promote(c5.checkpoint_id, validation={"success_rate": 0.3, "calibration": -0.9}, retention={"old_family": 0.1}).promoted
    # retention on disk: the two newest plus every promoted checkpoint (kept for comparison)
    assert [i.seq for i in mgr.list()] == [1, 3, 4, 5]
    info, state = mgr.rollback("validation collapsed")
    assert float(state.weights["heads"]["w"].unique()) == 3.0 and state.schedules == {"step": 3}
    assert info.rollback_of == c5.checkpoint_id and info.parent == c3.checkpoint_id
    # a restart now resumes from the promoted state
    assert mgr.latest().checkpoint_id == info.checkpoint_id and mgr.load().schedules == {"step": 3}
    assert mgr.promoted().checkpoint_id == c3.checkpoint_id
    log = mgr.promotion_log()
    assert [e["event"] for e in log].count("promote") == 9 and log[-1]["event"] == "rollback"
    assert log[-1]["to"] == c3.checkpoint_id and log[-1]["from"] == c5.checkpoint_id
    assert sum(1 for e in log if e.get("promoted")) == 2
    # a new process sees the same promoted checkpoint (rebuilt from the log)
    os.remove(os.path.join(str(tmp_path), "PROMOTED"))
    again = C.CheckpointManager(str(tmp_path), keep_last=2)
    assert again.recovery["promoted"] == c3.checkpoint_id and again.promoted().checkpoint_id == c3.checkpoint_id


def test_replay_indices_and_pending_pointer_travel_with_the_checkpoint(tmp_path):
    from conftest import build_record
    st = SqliteExperienceStore(str(tmp_path / "experience.sqlite"), durable=False, seed=3)
    with st.batch():
        for i in range(80):
            st.append(build_record(i))
    st.sample_ids(8, loss="world")
    net, opts, gen = _trained()
    state = C.capture_state(net, opts, store=st)
    assert state.replay_state["draws"] == 1 and state.pending_pointer["n_pending"] == 80
    assert state.pending_pointer["store"] == st.path and state.pending_pointer["head_seq"] == 80
    C.save_checkpoint(str(tmp_path / "ck"), state)
    after = [st.sample_ids(8, loss="world") for _ in range(3)]
    assert st.state_dict()["draws"] == 4
    loaded = C.load_checkpoint(str(tmp_path / "ck"))
    rep = C.restore_state(loaded, net, opts, store=st, rng=False)
    assert rep["replay"]["records_after_checkpoint"] == 0 and st.state_dict()["draws"] == 1
    assert [st.sample_ids(8, loss="world") for _ in range(3)] == after      # the sampler stream resumes exactly
    # experience written after the checkpoint is kept: it is never rolled back with the weights
    st.append(build_record(500))
    rep = C.restore_state(loaded, net, opts, store=st, rng=False)
    assert rep["replay"]["records_after_checkpoint"] == 1 and st.count() == 81 and st.get("e500") is not None


def test_runtime_named_checkpoint_directory_and_export_from_disk(tmp_path):
    """The runtime chooses checkpoint paths itself (``checkpoints/ckpt_00000007``) and exports from disk."""
    net, opts, gen = _trained()
    state = _full_state(net, opts, gen)
    path = str(tmp_path / "checkpoints" / "ckpt_00000007")
    info = C.save_checkpoint_at(path, state, notes="generation 7")
    assert info.path == path and info.seq == 7 and info.checkpoint_id == "ckpt_00000007"
    assert C.verify_checkpoint(path) == [] and not [n for n in os.listdir(os.path.dirname(path)) if n.startswith(".tmp-")]
    _same(C.load_checkpoint(path).weights, state.weights)
    with pytest.raises(C.CheckpointError):
        C.save_checkpoint_at(path, state)                    # a complete checkpoint is never overwritten
    nxt = C.save_checkpoint_at(str(tmp_path / "checkpoints" / "ckpt_00000008"), state)
    assert [i.seq for i in C.list_checkpoints(str(tmp_path / "checkpoints"))] == [7, 8] and nxt.seq == 8
    cfg = C.export_weights(path, str(tmp_path / "exports" / "e7"))
    weights, cfg2 = C.load_weights_only(str(tmp_path / "exports" / "e7"))
    assert cfg2 == cfg and cfg["n_tensors"] == 10
    _same(weights, state.weights)


def test_real_model_class_round_trips_through_a_checkpoint(tmp_path):
    """``CamLiveModel`` (debug preset, tiny random backbone, CPU): grouped weights, strict load into a fresh
    instance, identical group hashes; a head registered later is reported, not fatal."""
    try:
        from cam_live.config import CamLiveConfig
        from cam_live.nn.model import build_model
        torch.manual_seed(0)
        model = build_model(CamLiveConfig.debug())
    except Exception as e:                                   # Workstream A's model is not importable in this tree
        pytest.skip(f"real model not available: {type(e).__name__}: {e}")
    with torch.no_grad():                                    # the model initialises deterministically from its
        for p in model.parameters():                         # config, so move the trainable weights first
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))
    state = C.capture_state(model, {})
    assert set(state.weights) <= {"semantic_adapters", "experiential", "reward", "heads", "buffers"}
    assert {"semantic_adapters", "experiential"} <= set(state.weights)
    assert {n for d in state.weights.values() for n in d} == set(model.trainable_state())
    assert state.arch_signature == model.arch_signature()
    C.save_checkpoint(str(tmp_path), state)
    torch.manual_seed(1)
    fresh = build_model(CamLiveConfig.debug())
    hashes = lambda m: {g: h for g, h in m.group_hashes().items() if g != "backbone_frozen"}
    assert hashes(fresh) != hashes(model)
    rep = C.restore_state(C.load_checkpoint(str(tmp_path)), fresh, {}, strict=True, rng=False)
    assert rep["weights"].clean and rep["arch_signature_differences"] == [] and hashes(fresh) == hashes(model)
    fresh.register_head("added_later", nn.Linear(fresh.d_model, 3))
    with pytest.warns(UserWarning, match="tolerant checkpoint load"):
        rep2 = C.load_weights(fresh, C.load_checkpoint(str(tmp_path)).weights)
    assert len(rep2.missing) == 2 and all("added_later" in n for n in rep2.missing) and not rep2.unexpected


@pytest.mark.gpu
def test_checkpoint_of_a_gpu_model_is_stored_on_cpu(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    net, opts, gen = _trained()
    net.cuda()
    opts = _optims(net)
    x = torch.randn(16, 8, device="cuda")
    (net(x)[0].pow(2).mean()).backward()
    opts["main"].step()
    state = C.capture_state(net, opts, fast_state={"mu": torch.randn(4, device="cuda")})
    assert all(t.device.type == "cpu" for d in state.weights.values() for t in d.values())
    C.save_checkpoint(str(tmp_path), state)
    loaded = C.load_checkpoint(str(tmp_path))
    assert loaded.fast_state["mu"].device.type == "cpu"
    net2 = Net().cuda()
    rep = C.restore_state(loaded, net2, _optims(net2), strict=True, rng=False)
    assert rep["weights"].clean
    assert all(torch.equal(a, b) for a, b in zip(net.parameters(), net2.parameters()) if a.requires_grad)
    del net, net2
    torch.cuda.empty_cache()
