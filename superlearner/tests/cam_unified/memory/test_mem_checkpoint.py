"""Checkpoint semantics: weights-only files hold parameters + config only and load into fresh objects (also in a
fresh process); full-lifetime checkpoints restore everything; resets; spec 13.7 transfer conditions; crash recovery."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from memhelpers import ToyAgent, light_events, probe_input
from superlearner.cam_unified.memory.archive import EventArchive
from superlearner.cam_unified.memory.checkpoint import (WEIGHTS_METADATA_KEYS, CheckpointOps, config_from_dict,
                                                        inspect_weights, read_weights)
from superlearner.cam_unified.memory.consolidate import Consolidator
from superlearner.cam_unified.memory.errors import CheckpointMismatch
from superlearner.cam_unified.memory.index import MemoryIndex
from superlearner.cam_unified.memory.replay import ReplaySampler
from superlearner.cam_unified.testing import tiny_config

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))


def _cfg():
    return tiny_config(mem_segment=8, mem_segment_tokens=2)


def _lived_agent(tmp_path, seed=0) -> ToyAgent:
    """An agent with a real lifetime: trained weights, optimiser moments, scheduler, fast/latent state, RNG use,
    archive + memory index + replay + consolidator attached."""
    ag = ToyAgent(_cfg(), seed=seed)
    ag.archive = EventArchive(str(tmp_path / "archive"), chunk_rows=200)
    ag.archive.append(light_events(450, episode_len=50))
    ag.index = MemoryIndex(str(tmp_path / "index"), ag.config.mem_key_dim, 2, ag.config.d_model, device="cpu")
    ag.model.retriever.attach_index(ag.index)
    ag.consolidator = Consolidator(ag.archive, ag.index, ag.model.retriever)
    ag.consolidator.step()
    ag.consolidator.ensure_current()
    ag.replay = ReplaySampler(ag.archive, seed=seed)
    ag.replay.sync()
    ag.replay.sample(4, 8)
    ag.train_steps(5, seed=seed)
    torch.rand(3, generator=ag.rng["torch"])
    return ag


def _run(script: str, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, HERE]), OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script), *args], env=env, capture_output=True,
                          text=True, timeout=600)


def test_weights_only_file_contains_only_parameters_buffers_and_config(tmp_path):
    ag = _lived_agent(tmp_path)
    ops = CheckpointOps(ag)
    path = str(tmp_path / "w.safetensors")
    ops.save_weights(path)
    info = inspect_weights(path)
    sd = ag.model.state_dict()
    assert set(info["tensors"]) == set(sd)                  # exactly the parameters and buffers
    for k, (dtype, shape) in info["tensors"].items():
        assert tuple(sd[k].shape) == shape
    assert set(info["metadata"]) <= WEIGHTS_METADATA_KEYS
    assert json.loads(info["metadata"]["config"]) == json.loads(json.dumps(ag.config.to_dict(), default=list))
    n_param_bytes = sum(t.numel() * t.element_size() for t in sd.values())
    assert n_param_bytes <= info["bytes"] < n_param_bytes + 64_000   # no room for anything else
    raw = open(path, "rb").read()
    header = json.loads(raw[8:8 + int.from_bytes(raw[:8], "little")])
    meta = header.pop("__metadata__")
    assert set(meta) <= WEIGHTS_METADATA_KEYS and set(header) == set(sd)   # the raw file agrees with the reader
    for name in header:
        for forbidden in ("exp_avg", "step", "replay", "rng", "fast", "latent", "archive", "cursor", "counter",
                          "optim", "sched", "context"):
            assert forbidden not in name.lower(), name
    opt_keys = {k for st in ag.optimizers["core"].state.values() for k in st}
    assert opt_keys and not any(k in info["tensors"] for k in opt_keys)
    sd2, cfg_d, meta = read_weights(path)
    assert all(torch.equal(sd2[k], sd[k].cpu()) for k in sd)
    assert config_from_dict(cfg_d) == ag.config
    ag.archive.close()


_FRESH = """
import json, sys, torch
from memhelpers import build_agent, probe_input
from superlearner.cam_unified.memory.checkpoint import CheckpointOps, read_weights
path, out = sys.argv[1], sys.argv[2]
_, cfg_d, _ = read_weights(path)
ag = build_agent(cfg_d, seed=1234)              # a fresh object from the file's configuration only
before = [p.detach().clone() for p in ag.model.parameters()]
ops = CheckpointOps(ag)
ops.load_weights_only(path)
ag.model.eval()
with torch.no_grad():
    y = ag.model(probe_input(ag.config)).tolist()
fresh = build_agent(cfg_d, seed=1234)
st = {
  "y": y,
  "changed": any(not torch.equal(a, b) for a, b in zip(before, ag.model.parameters())),
  "opt_state": sum(len(s) for s in ag.optimizers["core"].state.values()),
  "lr": ag.optimizers["core"].param_groups[0]["lr"],
  "sched_epoch": ag.schedulers["core"].last_epoch,
  "archive": ag.archive is None and ag.index is None and ag.replay is None and ag.consolidator is None,
  "retriever_index": ag.model.retriever.index is None,
  "version": ag.version, "counters": ag.counters,
  "fast_equal_fresh": torch.equal(ag.fast_state.particles, fresh.fast_state.particles)
                      and torch.equal(ag.fast_state.steps, fresh.fast_state.steps) and ag.fast_state.channels == fresh.fast_state.channels,
  "latent_zero": float(ag.latent_state.abs().sum()),
  "running": ag.model.running.tolist()[:3],
}
json.dump(st, open(out, "w"))
"""


def test_load_weights_only_in_a_fresh_subprocess(tmp_path):
    ag = _lived_agent(tmp_path)
    ag.version = 7
    path = str(tmp_path / "w.safetensors")
    CheckpointOps(ag).save_weights(path)
    ag.model.eval()
    with torch.no_grad():
        want = ag.model(probe_input(ag.config))
    r = _run(_FRESH, path, str(tmp_path / "out.json"))
    assert r.returncode == 0, r.stderr[-3000:]
    st = json.load(open(tmp_path / "out.json"))
    torch.testing.assert_close(torch.tensor(st["y"]), want, rtol=1e-5, atol=1e-6)
    assert st["changed"]
    assert st["opt_state"] == 0 and st["lr"] == 1e-3 and st["sched_epoch"] == 0     # no optimiser leakage
    assert st["archive"] and st["retriever_index"]                                   # no experience access
    assert st["version"] == 0 and st["counters"] == {}
    assert st["fast_equal_fresh"] and st["latent_zero"] == 0.0                       # no adaptive / context state
    assert st["running"] == ag.model.running.tolist()[:3]                            # buffers are weights
    ag.archive.close()


def test_load_weights_only_resets_a_used_object_in_process(tmp_path):
    src = _lived_agent(tmp_path / "src")
    path = str(tmp_path / "w.safetensors")
    CheckpointOps(src).save_weights(path)
    dst = _lived_agent(tmp_path / "dst", seed=5)              # an object with its own lifetime
    CheckpointOps(dst).load_weights_only(path)
    assert all(torch.equal(a, b) for a, b in zip(src.model.state_dict().values(), dst.model.state_dict().values()))
    assert sum(len(s) for s in dst.optimizers["core"].state.values()) == 0
    assert dst.archive is None and dst.model.retriever.index is None and dst.version == 0
    assert float(dst.latent_state.abs().sum()) == 0.0 and dst.fast_state.channels == [{}, {}]
    src.archive.close()


def test_full_lifetime_roundtrip_restores_everything(tmp_path):
    ag = _lived_agent(tmp_path)
    ops = CheckpointOps(ag)
    ck = str(tmp_path / "life")
    ops.save_full_lifetime(ck)
    assert sorted(os.listdir(ck)) == ["lifetime.json", "optim.pt", "state.pt", "weights.safetensors"]
    # what the original produces next
    nxt_torch = torch.rand(4, generator=ag.rng["torch"])
    nxt_np = ag.rng["np"].random(3)
    nxt_global = torch.rand(2)
    nxt_replay = ag.replay.sample_anchors(10)
    exp_avg = {k: v["exp_avg"].clone() for k, v in zip(range(99), ag.optimizers["core"].state.values())}
    # restore into a fresh object (experience handles are reopened from the checkpoint's pointers)
    ag.archive.close()
    ag.replay.close()
    ag.index.close()
    fresh = ToyAgent(_cfg(), seed=77)
    torch.manual_seed(5)
    r = CheckpointOps(fresh).load_full_lifetime(ck)
    assert r["restored"] == ["weights", "optimizer", "fast_state", "rng", "counters", "experience"]
    assert all(torch.equal(a, b) for a, b in zip(ag.model.state_dict().values(), fresh.model.state_dict().values()))
    got = {k: v["exp_avg"] for k, v in zip(range(99), fresh.optimizers["core"].state.values())}
    assert exp_avg.keys() == got.keys() and all(torch.equal(exp_avg[k], got[k]) for k in got)
    assert fresh.schedulers["core"].last_epoch == ag.schedulers["core"].last_epoch == 5
    assert fresh.optimizers["core"].param_groups[0]["lr"] == ag.optimizers["core"].param_groups[0]["lr"]
    assert torch.equal(torch.rand(4, generator=fresh.rng["torch"]), nxt_torch)
    assert (fresh.rng["np"].random(3) == nxt_np).all() and torch.equal(torch.rand(2), nxt_global)
    assert fresh.version == 5 and fresh.counters == {"steps": 5}
    assert torch.equal(fresh.fast_state.particles, ag.fast_state.particles)
    assert fresh.fast_state.channels == ag.fast_state.channels and torch.equal(fresh.latent_state, ag.latent_state)
    assert fresh.archive.count() == 450 and fresh.archive.root == str(tmp_path / "archive")
    assert fresh.model.retriever.index is fresh.index and fresh.index.count() > 0
    assert fresh.replay.sample_anchors(10) == nxt_replay
    q = torch.nn.functional.normalize(torch.randn(1, fresh.config.mem_key_dim), dim=-1)
    assert len(fresh.model.retriever.retrieve(q, 3)[1][0]) == 3
    fresh.archive.close()


def test_transfer_conditions_restore_exactly_their_parts(tmp_path):
    ag = _lived_agent(tmp_path)
    ck = str(tmp_path / "life")
    CheckpointOps(ag).save_full_lifetime(ck)
    w = {k: v.clone() for k, v in ag.model.state_dict().items()}
    ag.archive.close()

    def fresh():
        return ToyAgent(_cfg(), seed=11)

    def same_w(a):
        return all(torch.equal(a.model.state_dict()[k], w[k]) for k in w)

    def moments(a):
        return sum(len(s) for s in a.optimizers["core"].state.values())
    A = fresh()
    CheckpointOps(A).load_transfer_condition(None, "A")
    assert not same_w(A) and moments(A) == 0 and A.archive is None
    B = fresh()
    CheckpointOps(B).load_transfer_condition(ck, "B")
    assert same_w(B) and moments(B) == 0 and B.archive is None and B.replay is None and B.version == 0
    C = fresh()
    CheckpointOps(C).load_transfer_condition(ck, "C")
    assert same_w(C) and moments(C) > 0 and C.archive is None and C.replay is None
    D = fresh()
    CheckpointOps(D).load_transfer_condition(ck, "D")
    assert same_w(D) and moments(D) > 0 and D.archive is not None and D.replay is not None
    assert D.version == 0 and float(D.latent_state.abs().sum()) == 0.0   # history, not context, is restored
    D.archive.close()
    with pytest.raises(ValueError):
        CheckpointOps(fresh()).load_transfer_condition(ck, "E")


def test_architecture_mismatch_and_tampering_are_refused(tmp_path):
    ag = ToyAgent(_cfg())
    path = str(tmp_path / "w.safetensors")
    CheckpointOps(ag).save_weights(path)
    other = ToyAgent(tiny_config(mem_segment=8, mem_segment_tokens=2, d_model=32, n_heads=4))
    with pytest.raises(CheckpointMismatch, match="architecture"):
        CheckpointOps(other).load_weights_only(path)
    ok = ToyAgent(tiny_config(mem_segment=8, mem_segment_tokens=2, lr=5e-4))  # training hyperparameters may differ
    CheckpointOps(ok).load_weights_only(path)
    raw = bytearray(open(path, "rb").read())
    raw[-3] ^= 0xFF                                          # flip bits in the last tensor
    open(tmp_path / "bad.safetensors", "wb").write(bytes(raw))
    with pytest.raises(CheckpointMismatch, match="digest"):
        CheckpointOps(ToyAgent(_cfg())).load_weights_only(str(tmp_path / "bad.safetensors"))


def test_excluded_prefixes_are_not_written(tmp_path):
    ag = ToyAgent(_cfg())
    ag.weights_exclude = ("retriever.target_",)              # e.g. a frozen pretrained base reloaded by name
    path = str(tmp_path / "w.safetensors")
    CheckpointOps(ag).save_weights(path)
    names = inspect_weights(path)["tensors"]
    assert names and not any(k.startswith("retriever.target_") for k in names)
    b = ToyAgent(_cfg(), seed=3)
    b.weights_exclude = ag.weights_exclude
    CheckpointOps(b).load_weights_only(path)
    assert torch.equal(b.model.head.weight, ag.model.head.weight)


_LIFE = """
import os, sys, torch
from memhelpers import ToyAgent, light_events
from superlearner.cam_unified.memory.archive import EventArchive
from superlearner.cam_unified.memory.checkpoint import CheckpointOps
from superlearner.cam_unified.memory.replay import ReplaySampler
from superlearner.cam_unified.testing import tiny_config
root = sys.argv[1]
ag = ToyAgent(tiny_config(mem_segment=8, mem_segment_tokens=2), seed=0)
ag.archive = EventArchive(os.path.join(root, "archive"), chunk_rows=100, autoflush=True)
ag.replay = ReplaySampler(ag.archive, seed=0)
ops = CheckpointOps(ag)
for epoch in range(6):
    ag.archive.append(light_events(150, start=150 * epoch))
    ag.replay.sync()
    ag.train_steps(2, seed=epoch)
    if epoch == 3:
        ops.save_full_lifetime(os.path.join(root, "life"))
        ops.save_full_lifetime(os.path.join(root, "life"))   # overwrite atomically with identical state
        torch.save({k: v.clone() for k, v in ag.model.state_dict().items()}, os.path.join(root, "expect.pt"))
        torch.save({"draws": ag.replay.sample_anchors(5), "rand": torch.rand(3, generator=ag.rng["torch"])},
                   os.path.join(root, "expect_next.pt"))
    if epoch == 5:
        import shutil
        real = shutil.rmtree
        def boom(p, *a, **k):
            if ".prev-" in str(p):
                os._exit(9)                        # crash after the new checkpoint replaced the old one
            return real(p, *a, **k)
        shutil.rmtree = boom
        ops.save_full_lifetime(os.path.join(root, "life"))
"""


def test_full_lifetime_recovery_after_a_crash(tmp_path):
    root = str(tmp_path)
    r = _run(_LIFE, root)
    assert r.returncode == 9, r.stderr[-3000:]
    # the crash hit after the new checkpoint was renamed into place; its predecessor is still present beside it
    assert os.path.isdir(os.path.join(root, "life"))
    ag = ToyAgent(tiny_config(mem_segment=8, mem_segment_tokens=2), seed=42)
    ops = CheckpointOps(ag)
    res = ops.load_full_lifetime(os.path.join(root, "life"))
    assert res["version"] == 12 and ag.counters == {"steps": 12}
    assert ag.archive.count() == 900                       # every acknowledged event survived the crash
    assert ag.replay.cursor == 899
    ag.archive.append(light_events(10, start=900))         # and the lifetime continues
    ag.train_steps(1)
    assert ag.version == 13
    ag.archive.close()
    # the epoch-3 checkpoint state is also recoverable exactly when the newer one is removed
    import shutil
    prevs = [p for p in os.listdir(root) if p.startswith("life.prev-")]
    assert len(prevs) == 1
    shutil.rmtree(os.path.join(root, "life"))
    old = ToyAgent(tiny_config(mem_segment=8, mem_segment_tokens=2), seed=42)
    res = CheckpointOps(old).load_full_lifetime(os.path.join(root, "life"))
    assert res["version"] == 8
    want = torch.load(os.path.join(root, "expect.pt"))
    assert all(torch.equal(old.model.state_dict()[k], v) for k, v in want.items())
    nxt = torch.load(os.path.join(root, "expect_next.pt"))
    assert old.replay.cursor == 599                        # the sampler is back at the checkpoint's view ...
    assert old.replay.sample_anchors(5) == nxt["draws"]    # ... and draws exactly what it drew then
    assert torch.equal(torch.rand(3, generator=old.rng["torch"]), nxt["rand"])
    assert old.replay.sync() == 310                        # later experience (incl. the 10 above) re-enters
    old.archive.close()
