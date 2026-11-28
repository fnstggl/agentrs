"""Checkpoints as the spec requires (Part 10.3, 15.2, 15.3; INT2):

* the agent promotes checkpoints through ``credit.checkpoint.CheckpointManager`` only on independent evidence —
  held-out validation losses and measured retention on fixed sets of earlier experience — never on internal reward;
* a refused promotion (measured forgetting) rolls the learner back to the last promoted checkpoint; experience is kept;
* ``agent.export_weights`` and ``cam-live export-checkpoint`` produce a weights-only directory that
  ``credit.checkpoint.load_exported_model`` reads in a fresh process, giving the same policy.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys

import torch

from cam_live.agent import CamLiveAgent
from cam_live.contracts import RunFiles, SimClock
from cam_live.runtime.logs import iter_jsonl
from cam_live.validation import HeldOutProbe
from int_helpers import Recorder, agent_config, run_lifetime

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))


def _agent(tmp_path, **kw):
    over = dict(use_backbone=False, planner=False, promote_every=10_000, retention_records=8)
    over.update(kw)
    return CamLiveAgent(agent_config(**over), str(tmp_path / "agent"), clock=SimClock())


def test_promotion_needs_independent_evidence_and_a_refusal_rolls_back(tmp_path):
    agent = _agent(tmp_path)
    clock = agent.clock
    try:
        rec = Recorder(agent)
        lt1 = run_lifetime(rec, clock, str(tmp_path), 31, max_steps=30, max_decisions=14)
        lt2 = run_lifetime(rec, clock, str(tmp_path), 32, max_steps=6, max_decisions=3)
        assert agent.promotions, agent.last.get("promotion_error")
        first = agent.promotions[0]
        assert first["trigger"].startswith(f"objective {lt1.task.objective_id} ended")
        assert first["promoted"], first["reasons"]
        assert first["validation"] and all(k.startswith("heldout/") for k in first["validation"])
        fam = f"{lt1.task.family}@{lt1.task.objective_id}"
        assert any(k.startswith(fam) for k in first["retention"]), first["retention"]
        promoted = agent.ckpts.promoted().checkpoint_id
        assert promoted == first["checkpoint_id"]
        # the promotion reached the runtime's learning log as a checkpoint record
        ups = [e["data"] for e in iter_jsonl(os.path.join(lt2.run_dir, RunFiles.LEARNING))]
        assert any(u["timescale"] == "checkpoint" and u["checkpoint_id"] == promoted for u in ups)
        # damaging change of the weights (as a harmful stretch of learning would make): forgetting is measured
        good = agent.ckpts.load(promoted)
        n_rec = agent.store.count()
        with torch.no_grad():
            g = torch.Generator().manual_seed(0)
            for n, p in agent.model.named_parameters():
                if p.requires_grad and ("predict_policy" in n or "predict_world" in n):
                    p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) * 0.5)
        psi_before = {k: v.clone() for k, v in agent.meta.reward.state_dict().items()}
        pol_before = {k: v.clone() for k, v in agent.meta.policy.state_dict().items()}
        out = agent.promote("test: after a damaging change")
        last = agent.promotions[-1]
        assert not last["promoted"] and any("retention" in r or "dropped" in r for r in last["reasons"]), last
        assert last["rolled_back_to"] == promoted and out and out[0].timescale == "checkpoint"
        assert out[0].diagnostics["rolled_back"] and sum(out[0].weight_delta.values()) > 0
        # LRN-6: R_psi is not judged by the gates and stays; the policy head goes back only when its retention
        # (decision agreement) is a reason of the refusal
        assert "reward" in last["kept"]
        assert all(torch.equal(v, psi_before[k]) for k, v in agent.meta.reward.state_dict().items())
        policy_forgot = any("decision_agreement" in r for r in last["reasons"])
        assert ("policy_head" in last["kept"]) == (not policy_forgot), last
        if not policy_forgot:
            assert all(torch.equal(v, pol_before[k]) for k, v in agent.meta.policy.state_dict().items())
        restored = agent.model.trainable_state()
        for grp, tensors in good.weights.items():
            if grp == "reward":
                continue
            for name, t in tensors.items():
                if name.startswith("heads.reward."):          # R_psi's buffers (normalisers, lifetime) stay too
                    continue
                if name in restored and (policy_forgot or "predict_policy" not in name):
                    assert torch.equal(restored[name], t.to(restored[name].dtype)), (grp, name)
        assert agent.store.count() == n_rec                     # experience is never rolled back
        assert agent.ckpts.get(last["checkpoint_id"]) is not None   # the refused checkpoint is kept for comparison
        # the agent's own reward can never justify a promotion
        info = agent.ckpts.save(agent.ckpts.load(promoted))
        dec = agent.ckpts.promote(info.checkpoint_id, validation={"intrinsic_reward": 9.0},
                                  retention={"internal_reward": 1.0})
        assert not dec.promoted and sorted(dec.ignored) == ["internal_reward", "intrinsic_reward"]
    finally:
        agent.close()


def _child(export_dir):
    env = dict(os.environ, PYTHONPATH=SRC, OMP_NUM_THREADS="2")
    out = subprocess.run([sys.executable, os.path.join(HERE, "int2_export_child.py"), export_dir], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_export_weights_reads_in_a_fresh_process(tmp_path):
    from cam_live.runtime import cli
    agent = _agent(tmp_path)
    try:
        lt = run_lifetime(Recorder(agent), agent.clock, str(tmp_path), 33, max_steps=20, max_decisions=10)
        dest = str(tmp_path / "export")
        agent.export_weights(dest)
        assert sorted(os.listdir(dest)) == ["config.json", "weights.safetensors"]     # nothing but weights + config
        # a weights-only export carries no persistent core-state chain (experience-dependent state, like the fast
        # beliefs; nn.persist): the fresh-process model computes the agent's distribution at the initial chain state
        chain = getattr(agent, "persist", None)
        with (chain.initial() if chain is not None else contextlib.nullcontext()):
            p_ref = HeldOutProbe(agent).distribution().tolist()
        got = _child(dest)
        assert max(abs(a - b) for a, b in zip(p_ref, got["probs"])) < 1e-6, (p_ref, got["probs"])
        hashes = agent.model.group_hashes()
        assert all(got["group_hashes"][g] == hashes[g] for g in ("experiential", "heads", "semantic_adapters", "reward"))
        # cam-live export-checkpoint without a running process: the committed checkpoint, weights-only
        dest2 = str(tmp_path / "export_cli")
        os.remove(os.path.join(lt.run_dir, "runtime.pid"))      # the run's process is gone (this test process wrote it)
        assert cli.main(["export-checkpoint", "--run-dir", lt.run_dir, "--out", dest2, "--json"]) == 0
        meta = json.load(open(dest2 + ".export.json"))
        assert "export_weights_only" in meta["method"], meta["method"]
        got2 = _child(dest2)
        assert len(got2["probs"]) == len(p_ref) and abs(sum(got2["probs"]) - 1.0) < 1e-6
    finally:
        agent.close()
