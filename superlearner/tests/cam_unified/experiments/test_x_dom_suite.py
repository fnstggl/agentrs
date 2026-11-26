"""W9 experiment drivers (experiments/dom_suite.py, meta_suite.py): mechanics on CPU with the integration doubles
(unit level; the real-stack smoke runs with Qwen3-0.6B are the evidence, docs/evidence/dom/W9/).

* the driver's loop is the canonical one (runtime.run_lifetime when present, else a delegation to runtime.run_steps
  with no evaluator): no environment reward reaches the agent and labels go to the trainer-side store only;
* Condition G's wrapper probes swapped latent states on snapshots without changing what the agent executes;
* split guard, pre-registration guard, frozen-group learning rates, decoder-conditioning switch;
* meta_suite writes an explicit BLOCKED provenance row for a condition whose prerequisites are absent.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from superlearner.cam_unified.experiments import dom_suite as DS
from superlearner.cam_unified.experiments import meta_suite as MS


def _stack(seed=0, archive_dir=None):
    from superlearner.cam_unified import integration_doubles as dbl
    from superlearner.cam_unified.agent import CamUnifiedAgent
    from superlearner.cam_unified.memory.archive import EventArchive
    from superlearner.cam_unified.model import build_model
    from superlearner.cam_unified.testing import tiny_config
    torch.manual_seed(seed)
    cfg = tiny_config(imag_branches=3, imag_samples=2, imag_horizons=(2,), mem_top_k=4, reward_mode="observation_only")
    model = build_model(cfg, components=dbl.double_components(cfg))
    arc = EventArchive(archive_dir) if archive_dir else dbl.DoubleArchive()
    ag = CamUnifiedAgent(model, archive=arc, recorder=dbl.ListTraceRecorder(), replay=None,
                         trainer=dbl.DoubleTrainer(model, cfg), index=model.retriever, seq_len=4, n_seqs=2, seed=seed,
                         greedy=True)
    return ag, dbl.ToyShopEnv(B=3, seed=seed), cfg


class _Labels:
    def __init__(self):
        self.calls = 0

    def take_labels(self):
        from superlearner.cam_unified.envs.labels import LabelRecord
        self.calls += 1
        return [LabelRecord("shop0", self.calls, "success", 0.7, 7.0, None, "unit")] if self.calls == 2 else []


def test_the_driver_loop_delivers_no_environment_reward_and_drains_labels(tmp_path):
    from superlearner.cam_unified.envs.labels import LabelStore
    fn, name = DS.resolve_loop()
    assert name in ("runtime.run_lifetime", "runtime.run_steps (run_lifetime absent at this commit)")
    ag, env, cfg = _stack(archive_dir=str(tmp_path / "arc"))
    store, labels = LabelStore(str(tmp_path / "labels")), _Labels()
    res = fn(ag, env, labels=labels, label_store=store, reward_mode="observation_only", learner=None, days=None,
             max_decisions=9, update_every=0, think_s=None, checkpoint=None, on_step=None)
    st = DS._lifetime_stats(res)
    assert st["decisions"] == 9 and labels.calls >= 3 and len(store) == 1
    scan = DS.integrity_scan(ag.archive, cfg, store)
    assert scan["ok"] and scan["records"] >= 9 and scan["nonzero_rewards"] == 0
    with pytest.raises(ValueError):
        DS._run_lifetime_via_run_steps(ag, env, reward_mode="legacy", max_decisions=3)


def test_latent_swap_wrapper_probes_without_changing_the_executed_decisions(tmp_path):
    from superlearner.cam_unified.runtime import run_steps
    plain, env1, _ = _stack(seed=3)
    rep1 = run_steps(plain, env1, 4, evaluator=None, seeds=[1, 2, 3])
    inner, env2, _ = _stack(seed=3)
    w = DS.LatentSwapAgent(inner, probe_every=1)
    rep2 = run_steps(w, env2, 4, evaluator=None, seeds=[1, 2, 3])
    assert [s["tools"] for s in rep1.steps] == [s["tools"] for s in rep2.steps]
    assert len(w.probes) == 4 * 3                                  # every live row of every probe round
    m = w.metrics()
    assert m["g_probes"] == 12 and 0.0 <= m["g_action_change_rate"] <= 1.0
    assert all(p["donor"] != p["row"] for p in w.probes)
    assert inner.archive.count() == plain.archive.count()          # probes never wrote experience


def test_split_and_preregistration_guards():
    from crucible.rapid.families import families
    val = families("val")[0]
    with pytest.raises(ValueError, match="split guard"):
        DS.families_for("train", [val])
    assert DS.families_for("val", [val]) == [val]
    assert "coop_split_info" not in DS.families_for("train")       # broken multi-agent families are excluded
    ok, why = DS.prereg_committed("superlearner/docs/experiments/U-NOPE-9.md")
    assert not ok and "does not exist" in why
    spec = DS.RunSpec(condition="B", run_dir="/nonexistent", set="confirm", smoke=True)
    if DS.prereg_committed()[0]:
        with pytest.raises(ValueError, match="smoke"):
            DS.run_condition(spec)
    else:
        with pytest.raises(RuntimeError, match="refused"):
            DS.run_condition(spec)


def test_frozen_groups_and_the_decoder_conditioning_switch():
    from superlearner.cam_unified.learn.trainer import CombinedTrainer
    from superlearner.cam_unified.testing import tiny_config
    cfg = tiny_config()
    core, heads, sem = torch.nn.Linear(4, 4), torch.nn.Linear(4, 2), torch.nn.Linear(2, 2)
    tr = CombinedTrainer(cfg, core=core, heads=list(heads.parameters()), semantic=list(sem.parameters()), amp=False)
    frozen = DS._set_group_lr(tr, lambda n: not n.startswith("core/"))
    assert frozen and all(n.startswith("core/") for n in frozen)
    assert all(g["lr"] == 0.0 for g in tr.opt.param_groups if g["name"].startswith("core/"))
    tr.reset_optimizer()                                            # a reset keeps the freeze
    assert all(g["lr"] == 0.0 for g in tr.opt.param_groups if g["name"].startswith("core/"))
    assert any(g["lr"] > 0 for g in tr.opt.param_groups if g["name"].startswith("heads/"))

    class Bridge(torch.nn.Module):
        def forward(self, R, goal=None, belief=None):
            return R * 2

    class AG(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bridge = Bridge()

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_generator = AG()
    m = M()
    r = DS.disable_decoder_conditioning(m)
    assert r["hooks"] == 1 and torch.equal(m.action_generator.bridge(torch.ones(2, 3)), torch.zeros(2, 3))
    for h in r["_handles"]:
        h.remove()
    assert torch.equal(m.action_generator.bridge(torch.ones(2, 3)), 2 * torch.ones(2, 3))


def test_exp_rows_carry_checkpoint_and_code_provenance(tmp_path):
    """Audit 'h' Exp1: runs.jsonl recorded no checkpoint path, digest or commit (provenance only by reproduction)."""
    import argparse
    import hashlib
    from superlearner.cam_unified.experiments import exp1 as E1
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "weights.safetensors").write_bytes(b"weights-bytes")
    a = argparse.Namespace(learner="cam", tag=None, seed=2, set="dev", ckpt=str(ck))
    m = E1.run_meta(a, "decide")
    assert m["ckpt_sha256"] == hashlib.sha256(b"weights-bytes").hexdigest()
    assert m["cmd"] == "decide" and m["run_id"].startswith("decide-cam-s2-") and m["t_wall"] > 0
    assert "code_commit" in m and "legacy" in m["reward_mode"]
    assert E1.run_meta(argparse.Namespace(learner="nig_meta", tag=None, seed=0, set="dev", ckpt=None),
                       "predict")["ckpt_sha256"] is None


def test_meta_suite_records_blocked_conditions_instead_of_running_them(tmp_path):
    pre = MS.prerequisites()
    assert set(pre) == {"rpsi_real", "loop_reward_mode", "goal_free", "meta_step", "credit", "curriculum"}
    blocked = [c for c in MS.CONDITIONS.values() if any(not pre[r][0] for r in c.requires)]
    if not blocked:
        pytest.skip("every prerequisite is met at this commit")
    c = blocked[0]
    row = MS.run_meta_condition(MS.MetaSpec(exp=c.exp, condition=c.cid, run_dir=str(tmp_path)))
    assert row["status"] == "BLOCKED" and row["blocked_by"]
    rows = [json.loads(x) for x in open(tmp_path / "eval" / "provenance.jsonl")]
    assert rows[-1]["condition"] == c.cid and rows[-1]["status"] == "BLOCKED"
    assert {x.exp for x in MS.CONDITIONS.values()} == {"D", "E", "F", "I", "J"}
    assert sum(1 for x in MS.CONDITIONS.values() if x.exp == "D") == 7
    assert not MS.CONDITIONS["D1"].compliant and not MS.CONDITIONS["D2"].compliant
