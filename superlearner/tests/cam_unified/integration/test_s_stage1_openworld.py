"""Stage-1 vertical path with the REAL components on a REAL Crucible OpenWorld (spec Part 21 Stage 1; A22, A13, A20,
A21): Qwen3-0.6B semantic model + A's tokenizer / action generator, B's core / entities / reasoner / beliefs /
Kalman layer, C's outcome head / world model, D's heads and CombinedTrainer, E's archive / artifacts / memory index /
retriever / consolidator / replay / checkpoint ops, F's TraceRecorder and dashboard, driven by G's lifetime harness
over G's process-isolated ow_tiny operator and evaluator. No test doubles anywhere in the assembly.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

from superlearner.cam_unified.testing import tiny_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and the cached Qwen3-0.6B")

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
TINY = dict(n_layers=2, d_model=64, n_heads=4, n_state=8, d_particle=32, n_particles=8, bayes_feat=8,
            stoch_groups=8, stoch_classes=8, mem_key_dim=32, reason_layers=1, max_ctx_tokens=64, n_prefix=4,
            max_entities=32, n_options=4, max_arg_tokens=24, imag_horizons=(2,), mem_segment=8)


def _system_config(**kw):
    from superlearner.cam_unified.stage1.system import SystemConfig
    tiny_config(**TINY)                                   # the same overrides are a valid tiny config
    base = dict(scale="dev", overrides=dict(TINY), sem_model="Qwen/Qwen3-0.6B", imag_candidates=2, imag_samples=1,
                imag_horizon=2, seq_len=4, n_seqs=2, adapt_seq_len=6, fresh_max_steps=16)
    base.update(kw)
    return SystemConfig(**base)


def _rows(p):
    return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else []


@pytest.fixture(scope="module")
def stage1_run(tmp_path_factory):
    from superlearner.cam_unified.stage1.runner import RunSpec, run
    rd = str(tmp_path_factory.mktemp("stage1") / "run")
    out = run(RunSpec(run_dir=rd, task="open_economy", max_decisions=6, update_every=3, checkpoint_every_days=0.02,
                      system=_system_config()))
    return rd, out


def test_real_stack_vertical_path_on_a_real_ow_tiny_world(stage1_run):
    """semantic observation -> latent model -> generated action -> real Crucible execution -> verified outcome ->
    archived experience -> neural update -> later decisions under the updated weights."""
    from superlearner.cam_unified.memory.archive import EventArchive
    from superlearner.cam_unified.interfaces import ActionSource, Provenance
    rd, out = stage1_run
    assert out["error"] is None and out["decisions_total"] == 6 and out["version"] == 2
    meta = json.load(open(os.path.join(rd, "run.json")))
    rep = meta["system"]["build_report"]
    for name in ("tokenizer", "action_generator", "core", "entity_encoder", "reasoner", "beliefs", "bayes",
                 "outcome_head", "world_model", "retriever", "value_head", "option_policy", "exploration_head"):
        assert rep[name].startswith("built from superlearner.cam_unified."), (name, rep[name])
        assert "integration_doubles" not in rep[name] and "testing" not in rep[name], (name, rep[name])
    assert meta["system"]["semantic_model"] == "Qwen/Qwen3-0.6B" and meta["system"]["planner"] == \
        "LatentImaginationPlanner"
    assert meta["task"]["world_family"] == "ow_tiny" and 100_000 <= meta["task"]["world_seed"] < 1_000_000
    arc = EventArchive(os.path.join(rd, "archive"), readonly=True)
    recs = [r for b in arc.scan() for r in b]
    dec = [r for r in recs if r.action is not None]
    assert len(dec) == 6 and all(r.provenance == Provenance.REAL for r in recs)
    assert all(r.action.source in (ActionSource.CAM_PLANNER, ActionSource.CAM_DECODER) for r in dec)
    assert all(r.result is not None and r.observations for r in dec)
    # actions are schema-valid against the world's own tool schemas at decision time (Crucible's validator), decoded
    # by CAM's grammar walk with a replay record and learner-path log-probability
    tr = _rows(os.path.join(rd, "transcript.jsonl"))[1:]
    for r in dec:
        assert r.action.logp_parts.get("decode") and r.action.logp_parts["rescored"]
    rows = _rows(os.path.join(rd, "decisions.jsonl"))
    assert all(x["schema_valid"] is True for x in rows), [x["schema_errors"] for x in rows]
    assert [x["model_version"] for x in rows] == [0, 0, 0, 1, 1, 1]          # later decisions: updated weights
    assert sorted({r.model_version for r in dec}) == [0, 1]
    assert all(x["n_tools"] >= 100 for x in rows)                           # the full OpenWorld tool list
    assert len(tr) == 6 and all(t["result"] is not None for t in tr)
    for r in dec:
        if r.result.ok:
            assert not r.result.error
        else:
            assert r.result.error                                           # a real error code from Crucible
    # neural updates from archived real events: shared-core weights changed, provenance recorded in the archive
    ups = _rows(os.path.join(rd, "learner", "updates.jsonl"))
    assert [u["version"] for u in ups] == [1, 2]
    for u in ups:
        assert {"world", "adapt", "actor", "value"} <= set(u["losses"])
        assert u["core_hash_before"] != u["core_hash_after"] and u["param_delta"]["core"]["l2_delta"] > 0
        assert u["backbone_grad_norm"] and u["backbone_grad_norm"]["world"] > 0
        ids = list(dict.fromkeys(u["replay_event_ids"] + u["consumed_event_ids"]))
        assert ids
        got = arc.get(ids)                                                  # every consumed event id resolves
        assert all(u["update_id"] in r.training_provenance.get("used_in_updates", []) for r in got)
    assert ups[0]["core_hash_after"] == ups[1]["core_hash_before"]
    # the policy at a fixed probe context changed with the weights
    pr = _rows(os.path.join(rd, "probes", "probes.jsonl"))
    assert [p["version"] for p in pr] == [0, 1, 2]
    assert any(abs(a["logp_total"] - b["logp_total"]) > 1e-6 for a, b in zip(pr[0]["rows"], pr[-1]["rows"]))
    arc.close()


def test_every_decision_has_a_trace_and_the_dashboard_renders(stage1_run):
    from superlearner.cam_unified.observe.dashboard import build_dashboard
    from superlearner.cam_unified.observe.trace import load_traces
    from superlearner.cam_unified.stage1.evidence import summarize
    rd, _ = stage1_run
    ts = load_traces(os.path.join(rd, "traces"))
    assert len(ts) == 6
    for t in ts:
        assert t.predicted and t.uncertainty and t.candidates and t.imagined and t.selected and t.objective
        assert t.reasoning["iters"] >= 1 and t.belief_before and t.belief_after and t.outcome
        assert t.context_refs
    assert sum(1 for t in ts if t.update_refs) >= 5                         # the consumed decisions are linked
    html = open(build_dashboard(rd)).read()
    for sec in ("Learning curves", "World-model calibration", "Real versus imagined", "Lifetimes", "Compute"):
        assert sec in html
    s = summarize(rd)
    assert s["decisions"]["consequential_cam"] == 6 and s["updates"]["n_updates"] == 2
    assert s["updates"]["core_hash_changed_every_update"] and s["updates"]["core_hash_chain_consistent"]
    pc = s["updates"]["provenance_check"]
    assert pc["event_ids_checked"] > 0 and pc["resolvable_in_archive"] == pc["event_ids_checked"] == \
        pc["with_update_id_in_used_in_updates"]
    assert s["traces"]["n_live"] == 6 and s["traces"]["fields_filled"]["imagined"] == 6


def _cli(*args, timeout=900):
    env = dict(os.environ, PYTHONPATH=SRC, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2")
    p = subprocess.run([sys.executable, "-m", "superlearner.cam_unified.stage1", *args], env=env,
                       capture_output=True, text=True, timeout=timeout)
    return p


@pytest.mark.slow
def test_kill_mid_lifetime_and_resume_in_new_processes(tmp_path):
    """A21: a process is SIGKILLed mid-lifetime; two NEW processes resume from the last checkpoint + archive
    (copies of the same directory): archive count, model version, optimiser steps and replay state are those of the
    checkpoint, the interrupted process's later events are an abandoned branch excluded from training, and the
    resumed processes make identical next decisions."""
    rd = str(tmp_path / "run")
    over = [f"--set={k}={json.dumps(v)}" for k, v in TINY.items()]
    p = _cli("run", "--run-dir", rd, "--sem-model", "Qwen/Qwen3-0.6B", "--update-every", "3",
             "--checkpoint-every-days", "0.02", "--imag-candidates", "2", "--imag-horizon", "2", "--imag-samples", "1",
             "--seq-len", "4", "--adapt-seq-len", "6", "--fresh-max-steps", "16", "--kill-after", "8", *over)
    assert p.returncode == -9, p.stderr[-3000:]                             # SIGKILL
    killed = json.load(open(os.path.join(rd, "killed.json")))
    latest = open(os.path.join(rd, "lifetime_ckpt", "LATEST")).read().strip()
    life = json.load(open(os.path.join(rd, "lifetime_ckpt", latest, "agent", "lifetime", "lifetime.json")))
    outs = []
    for k in ("a", "b"):
        p = _cli("resume", "--run-dir", str(tmp_path / k), "--copy-from", rd, "--exit-after", "2")
        assert p.returncode == 0, p.stderr[-3000:]
        outs.append(json.loads(p.stdout.strip().splitlines()[-1]))
    for k, o in zip("ab", outs):
        ri = _rows(os.path.join(tmp_path, k, "resume.jsonl"))[-1]
        assert ri["pid"] != killed["pid"] and ri["version"] == life["version"]
        assert ri["archive_count_checkpoint"] == life["archive"]["count"]
        assert ri["archive_count_now"] == killed["archive"]
        assert ri["abandoned_events"] == killed["archive"] - life["archive"]["count"] > 0
        assert ri["replay_cursor"] == life["replay"]["cursor"]
        assert ri["trainer_n_updates"] == life["version"] and ri["optimizer_steps"] == life["version"]
        assert len(ri["abandoned_decision_rows"]) == ri["abandoned_decisions"] == ri["abandoned_events"]
    # the abandoned events are excluded from what replay / consolidation read, and their traces are marked
    from superlearner.cam_unified.observe.trace import load_traces
    ts = load_traces(os.path.join(tmp_path, "a", "traces"))
    ab = {e for e in ri["abandoned_event_ids"]}
    assert {t.event_id for t in ts if (t.outcome or {}).get("abandoned_branch")} == ab
    da = _rows(os.path.join(tmp_path, "a", "decisions.jsonl"))
    db = _rows(os.path.join(tmp_path, "b", "decisions.jsonl"))
    assert len(da) == len(db) == life["counters"]["n_decisions"] + 2
    for x, y in zip(da[-2:], db[-2:]):                                      # the two resumed lifetimes agree
        assert x["step"] == y["step"] and x["tool"] == y["tool"] and x["args"] == y["args"]
        assert abs(x["behavior_logp"] - y["behavior_logp"]) < 1e-3
    # and the resumed decisions are the ones the interrupted process made at those steps (restored weights,
    # optimiser, beliefs, latent state, RNG streams and semantic caches: an exact continuation)
    by_step = {r["step"]: r for r in da}
    for ab in ri["abandoned_decision_rows"]:
        x = by_step[ab["step"]]
        assert x["tool"] == ab["tool"] and x["args"] == ab["args"], (ab["step"], ab["tool"], x["tool"])
        assert abs(x["behavior_logp"] - ab["behavior_logp"]) < 1e-3
