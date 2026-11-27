"""W9 checkpoint repair (plan 2026-11-26 §3 W9; audit 'archive' CheckpointOps + 'h' Exp3 findings):

* weight files carry arch_hash / code_commit; an explicit `allow_new` / `allow_dropped` / `allow_config` migration
  is the only way to load a file whose layout differs from the model's (the length_head failure mode of the audit);
* load_weights_only on a USED agent resets everything a lifetime accumulates: fresh buffer, working state
  (trackers, options, A_{t-1}, last decision time), pending observations, learner diagnostics, trainer counters /
  stage / history, version and counters, and the RNGs (re-seeded deterministically).

Unit tests: the agent-level tests use the integration doubles (allowed for unit tests; the real-stack check with
Qwen3-0.6B is experiments/ckpt_e2e.py, evidence under docs/evidence/dom/W9/)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from memhelpers import ToyAgent, probe_input
from superlearner.cam_unified.memory.checkpoint import (WEIGHTS_METADATA_KEYS, CheckpointOps, arch_hash_of,
                                                        code_version, inspect_weights, model_arch_hash,
                                                        read_weights)
from superlearner.cam_unified.memory.errors import CheckpointMismatch
from superlearner.cam_unified.testing import tiny_config


def _cfg(**kw):
    return tiny_config(mem_segment=8, mem_segment_tokens=2, **kw)


class _Extra(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.ceil_logit = nn.Linear(D, 4)
        self.register_buffer("scale", torch.ones(4))


def _agent_with_extra(cfg, seed=0):
    ag = ToyAgent(cfg, seed=seed)
    ag.model.length_head = _Extra(cfg.d_model)        # a module added after the checkpoint was written
    ag.optimizers = ag.make_optimizers()
    ag.schedulers = ag.make_schedulers()
    return ag


def test_weight_files_record_arch_hash_and_code_commit(tmp_path):
    ag = ToyAgent(_cfg())
    path = str(tmp_path / "w.safetensors")
    info = CheckpointOps(ag).save_weights(path)
    meta = inspect_weights(path)["metadata"]
    assert set(meta) <= WEIGHTS_METADATA_KEYS and {"arch_hash", "code_commit", "code_dirty"} <= set(meta)
    assert meta["arch_hash"] == model_arch_hash(ag.model) == info["arch_hash"]
    sd, _, _ = read_weights(path)
    assert arch_hash_of(sd) == meta["arch_hash"]
    cv = code_version()
    assert meta["code_commit"] == (cv["commit"] or "") and len(meta["code_commit"]) in (0, 40)
    assert json.loads(meta["code_dirty"]) in (True, False, None)
    # the hash is a function of the layout only: other values, same hash; one more tensor, another hash
    ag2 = ToyAgent(_cfg(), seed=9)
    assert model_arch_hash(ag2.model) == meta["arch_hash"]
    assert model_arch_hash(_agent_with_extra(_cfg()).model) != meta["arch_hash"]


def test_added_module_is_refused_without_and_migrated_with_allow_new(tmp_path):
    src = ToyAgent(_cfg(), seed=1)
    path = str(tmp_path / "w.safetensors")
    CheckpointOps(src).save_weights(path)
    dst = _agent_with_extra(_cfg(), seed=2)
    init_extra = {k: v.clone() for k, v in dst.model.length_head.state_dict().items()}
    before = {k: v.clone() for k, v in dst.model.state_dict().items()}
    with pytest.raises(CheckpointMismatch, match="missing.*length_head"):
        CheckpointOps(dst).load_weights_only(path)
    assert all(torch.equal(before[k], v) for k, v in dst.model.state_dict().items())     # untouched on refusal
    with pytest.raises(CheckpointMismatch):
        CheckpointOps(dst).load_weights_only(path, allow_new=("head.",))                  # the wrong prefix
    rep = CheckpointOps(dst).load_weights_only(path, allow_new=("length_head.",))
    assert rep["migrated"] == ["length_head.ceil_logit.bias", "length_head.ceil_logit.weight", "length_head.scale"]
    assert rep["dropped"] == [] and rep["arch_hash"] != rep["model_arch_hash"]
    for k, v in dst.model.length_head.state_dict().items():
        assert torch.equal(v, init_extra[k])                                                # kept at its init
    for k, v in src.model.state_dict().items():
        assert torch.equal(dst.model.state_dict()[k], v)                                    # everything else loaded
    x = probe_input(src.config)
    with torch.no_grad():
        assert torch.equal(src.model(x), dst.model(x))


def test_removed_module_shape_change_and_config_change_need_explicit_permission(tmp_path):
    src = _agent_with_extra(_cfg(), seed=1)
    path = str(tmp_path / "w.safetensors")
    CheckpointOps(src).save_weights(path)
    dst = ToyAgent(_cfg(), seed=2)
    with pytest.raises(CheckpointMismatch, match="unexpected.*length_head"):
        CheckpointOps(dst).load_weights_only(path)
    rep = CheckpointOps(dst).load_weights(path, allow_dropped=("length_head.",))
    assert len(rep["dropped"]) == 3 and rep["migrated"] == []
    # a shape change is never a migration
    wide = ToyAgent(_cfg(), seed=3)
    wide.model.head = nn.Linear(wide.config.d_model, 5)
    p2 = str(tmp_path / "wide.safetensors")
    CheckpointOps(wide).save_weights(p2)
    with pytest.raises(CheckpointMismatch, match="shapes differ"):
        CheckpointOps(ToyAgent(_cfg())).load_weights(p2, allow_new=("head.",), allow_dropped=("head.",))
    # architecture configuration differences: refused unless the field is named
    other = ToyAgent(_cfg(n_options=6), seed=4)
    p3 = str(tmp_path / "opt.safetensors")
    CheckpointOps(other).save_weights(p3)
    with pytest.raises(CheckpointMismatch, match="architecture configuration"):
        CheckpointOps(ToyAgent(_cfg())).load_weights(p3)
    rep = CheckpointOps(ToyAgent(_cfg())).load_weights(p3, allow_config=("n_options",))
    assert rep["config_changes"] == {"n_options": [6, 4]} or rep["config_changes"] == {"n_options": (6, 4)}


def test_dom_fields_missing_from_old_files_compare_as_their_defaults(tmp_path):
    """A pre-DOM file (no n_intent / state_gate in its config) loads into a default-shaped model, and is refused by
    a DOM-shaped one unless the change is allowed."""
    src = ToyAgent(_cfg(), seed=1)
    path = str(tmp_path / "w.safetensors")
    CheckpointOps(src).save_weights(path)
    from safetensors.torch import load_file, save_file
    meta = inspect_weights(path)["metadata"]
    cfg = json.loads(meta["config"])
    for k in ("n_intent", "state_gate", "d_satellite", "value_channel"):
        cfg.pop(k)
    meta["config"] = json.dumps(cfg, sort_keys=True)
    old = str(tmp_path / "old.safetensors")
    save_file(load_file(path), old, metadata=meta)
    CheckpointOps(ToyAgent(_cfg())).load_weights(old)
    with pytest.raises(CheckpointMismatch, match="n_intent"):
        CheckpointOps(ToyAgent(_cfg(n_intent=16))).load_weights(old)
    CheckpointOps(ToyAgent(_cfg(n_intent=16))).load_weights(old, allow_config=("n_intent",))


def test_tampered_layout_hash_is_refused(tmp_path):
    ag = ToyAgent(_cfg())
    path = str(tmp_path / "w.safetensors")
    CheckpointOps(ag).save_weights(path)
    from safetensors.torch import load_file, save_file
    meta = inspect_weights(path)["metadata"]
    meta["arch_hash"] = "0" * 64
    bad = str(tmp_path / "bad.safetensors")
    save_file(load_file(path), bad, metadata=meta)
    with pytest.raises(CheckpointMismatch, match="arch_hash"):
        read_weights(bad)


# ---------------------------------------------------------------------------------------------- agent-level reset
def _stack(seed=0, real_trainer=True):
    from superlearner.cam_unified import integration_doubles as dbl
    from superlearner.cam_unified.agent import CamUnifiedAgent
    from superlearner.cam_unified.learn.trainer import CombinedTrainer
    from superlearner.cam_unified.model import build_model
    torch.manual_seed(seed)
    cfg = tiny_config(imag_branches=3, imag_samples=2, imag_horizons=(2,), mem_top_k=4)
    model = build_model(cfg, components=dbl.double_components(cfg))
    core_ids = {id(p) for p in model.core.parameters()}
    trainer = CombinedTrainer(cfg, core=model.core, heads=[p for p in model.parameters()
                                                          if p.requires_grad and id(p) not in core_ids],
                              stage="continual", amp=False, grad_probe_every=1) if real_trainer \
        else dbl.DoubleTrainer(model, cfg)
    archive = dbl.DoubleArchive()
    agent = CamUnifiedAgent(model, archive=archive, recorder=dbl.ListTraceRecorder(),
                            replay=dbl.DoubleReplay(archive, seed=seed), trainer=trainer, index=model.retriever,
                            seq_len=4, n_seqs=2, seed=seed, loss_cap=10.0)
    env = dbl.ToyShopEnv(B=2, seed=seed)
    return agent, env, dbl.ToyShopEvaluator(env)


def test_load_weights_only_resets_the_whole_lifetime_of_a_used_agent(tmp_path):
    from superlearner.cam_unified.runtime import run_steps
    ag, env, ev = _stack(seed=0)
    ops = CheckpointOps(ag)                              # constructed with the freshly built agent (as build does)
    run_steps(ag, env, 7, evaluator=ev, update_every=3)
    ob = env.observe()
    acts = ag.act(ob, env.available_tools())              # a decision whose consequences are still pending
    res = env.execute(acts)
    from superlearner.cam_unified.interfaces import ObservationBatch
    adv = env.advance_time([r.duration + 0.01 for r in res])
    ag.observe(res, [0.0, 0.0], ObservationBatch(ob.env_ids, [list(r.observations) + list(o) for r, o in
                                                               zip(res, adv.observations)], adv.t, adv.done))
    t = ag.trainer
    assert t.n_updates >= 2 and len(t.history) >= 2 and ag.fresh and ag.loss_mag and ag.update_log
    assert any(a is not None for a in ag.state.last_actions) and ag._post and ag.n_decisions > 0
    src_events = {r.event_id for r in ag.archive.rows} | {e for st in ag.fresh for e in (st.event_ids or [])}
    assert src_events
    trackers_before = [id(x) for x in ag.state.trackers]
    path = str(tmp_path / "w.safetensors")
    ops.save_weights(path)
    n_arch = ag.archive.count()
    src_archive = ag.archive
    info = ops.load_weights_only(path, seed=123)
    assert {"experience_access", "fresh_buffer", "working_state", "trainer", "learner_diagnostics",
            "rng(agent+global)", "pending_observations"} <= set(info["reset"]["reset"])
    assert ag.fresh == [] and ag._last_fresh == {} and ag._post == {} and ag._pending is None
    assert ag.loss_mag == {} and ag.update_log == [] and ag.memory_sync_log == [] and ag.n_decisions == 0
    assert t.n_updates == 0 and len(t.history) == 0 and t.stage.name == "continual" and t.n_skipped == 0
    assert all(a is None for a in ag.state.last_actions) and all(float(x) == -1.0 for x in ag.state.last_now)
    assert all(int(o) == -1 for o in ag.state.option) and \
        not set(trackers_before) & {id(x) for x in ag.state.trackers}
    assert ag.archive is None and ag.replay is None and ag.version == 0
    assert src_archive.count() == n_arch            # the pending observations were dropped, not flushed into it
    # the RNGs are a function of the seed only
    ref = _stack(seed=5)[0]
    CheckpointOps(ref).load_weights_only(path, seed=123)
    assert ag.py_rng.getstate() == ref.py_rng.getstate()
    assert torch.equal(ag.torch_rng.get_state(), ref.torch_rng.get_state())
    # the next update consumes only decisions of the new lifetime (its own, freshly attached archive + replay)
    from superlearner.cam_unified import integration_doubles as dbl
    fresh_archive = dbl.DoubleArchive()
    ag.archive, ag.replay = fresh_archive, dbl.DoubleReplay(fresh_archive, seed=0)
    rep = run_steps(ag, env, 3, evaluator=ev, update_every=3)
    ups = [u for u in rep.updates if u.get("updated")]
    assert ups, rep.updates
    new_ids = {r.event_id for r in fresh_archive.rows}
    for u in ups:
        assert set(u["consumed_events"]) <= new_ids and not set(u["consumed_events"]) & src_events
        assert set(u["replay_events"]) <= new_ids
    assert t.n_updates == len(ups)


def test_two_processes_loading_the_same_file_with_the_same_seed_act_identically(tmp_path):
    """Process-level determinism of load_weights_only(seed=...) on the doubles stack (the real-stack version is the
    GPU end-to-end check)."""
    from superlearner.cam_unified.runtime import run_steps
    ag, env, ev = _stack(seed=0)
    run_steps(ag, env, 4, evaluator=ev, update_every=2)
    path = str(tmp_path / "w.safetensors")
    CheckpointOps(ag).save_weights(path)
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "..", "..", "src"))
    script = f"""
import json, sys, torch
sys.path[:0] = [{src!r}, {here!r}]
from test_mem_checkpoint_migration import _stack
from superlearner.cam_unified.memory.checkpoint import CheckpointOps
from superlearner.cam_unified.runtime import run_steps
ag, env, ev = _stack(seed=int(sys.argv[2]))
run_steps(ag, env, int(sys.argv[3]), evaluator=ev, update_every=2)        # a different prior lifetime per process
CheckpointOps(ag).load_weights_only(sys.argv[1], seed=77)
ag.archive = None
env2 = type(env)(B=2, seed=11)
ag.reset(env2.observe().env_ids)
rep = run_steps(ag, env2, 5, evaluator=None)
print(json.dumps([s["tools"] for s in rep.steps]))
"""
    outs = []
    for seed, n in ((1, 2), (2, 5)):
        r = subprocess.run([sys.executable, "-c", script, path, str(seed), str(n)], capture_output=True, text=True,
                           timeout=600, env=dict(os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=""))
        assert r.returncode == 0, r.stderr[-3000:]
        outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]
