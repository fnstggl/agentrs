"""World model, self-play arena, trainer checkpoints and the CLI."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from superlearner.cam.agent import AgentConfig, CAMAgent, ModelChooser
from superlearner.cam.model import CAMController, EmbStore, ModelConfig
from superlearner.cam.worldmodel import WorldModel, evaluate_model, fit, transitions
from superlearner.llm.server import EmbedOnly
from superlearner.selfplay.arena import Arena, IdleChooser, make_learner, online_update
from superlearner.train.demos import TeacherChooser
from superlearner.train.trainer import Trainer
from superlearner.world.core import CoreConfig, SessionCore
from superlearner.world.tasks import task

SMALL = {"d": 64, "layers": 2, "heads": 4, "ff": 128, "bag_dim": 32, "hid": 64}
HERE = os.path.dirname(os.path.abspath(__file__))


def test_world_model_trains_on_recorded_transitions_and_is_scored_against_trivial_predictors():
    torch.manual_seed(0)
    model = CAMController(ModelConfig(**SMALL))
    model.eval()
    store = EmbStore()
    ag = CAMAgent(model, store, EmbedOnly(), AgentConfig(use_llm=False))
    eps = [ag.run(SessionCore(task("found", "tiny", 40 + i), CoreConfig(max_decisions=10)), ModelChooser(seed=i), f"e{i}") for i in range(8)]
    tr, te = transitions(model, eps[:6], store), transitions(model, eps[6:], store)
    n = sum(len(e["decisions"]) for e in eps[:6])
    assert len(tr["ok"]) == n and tr["z"].shape == (n, 64) and tr["next"].shape == (n, 64)
    assert float(tr["has_next"].sum()) == n - 6                    # the last decision of each episode has no successor
    wm = WorldModel(64, 64)
    hist = fit(wm, tr, epochs=40, lr=3e-3, batch=32)
    assert hist[-1]["ok"] < hist[0]["ok"] and hist[-1]["next"] < hist[0]["next"]
    rep = evaluate_model(wm, te, tr)
    assert rep["n"] == sum(len(e["decisions"]) for e in eps[6:]) and 0.0 <= rep["ok_accuracy"] <= 1.0 and "ok_accuracy_base_rate" in rep and "fclass_accuracy_majority" in rep
    fitrep = evaluate_model(wm, tr, tr)
    assert fitrep["ok_nll"] < fitrep["ok_nll_base_rate"]            # it fits its own training transitions better than the base rate
    bonus = wm.plan_bonus(torch.zeros(1, 64), torch.randn(1, 7, 64))
    assert bonus.shape == (7,) and torch.isfinite(bonus).all()


def test_arena_runs_several_participants_in_one_persistent_world_and_supports_counterfactuals():
    parts = [{"handle": "ada", "goal": "service", "start": "coded", "endowment": 800_000},
             {"handle": "bea", "goal": "service", "start": "scratch", "endowment": 500_000}]

    def run(idle: bool):
        arena = Arena("tiny", 7, parts, round_s=3600, core={"max_decisions": 10**6})
        players = {}
        for actor in sorted(arena.cores):
            ag = CAMAgent(None, EmbStore(), EmbedOnly(), AgentConfig(use_llm=False))
            players[actor] = (ag, IdleChooser() if (idle or actor == "p:bea") else TeacherChooser("auto", 0))
        return arena, arena.run(players, days=3.0, seed=1)
    arena, out = run(idle=False)
    assert set(out["final"]["outcomes"]) == {"p:ada", "p:bea"} and not out["invariants"]
    a = out["final"]["outcomes"]["p:ada"]
    assert a["live_services"] == 1 and a["listed_services"] == 1 and a["requests_served"] > 0     # a real, used service
    assert out["final"]["outcomes"]["p:bea"]["live_services"] == 0
    assert len(out["history"]) >= 3 and out["agents"]["p:ada"]["decisions"] > 3
    _, cf = run(idle=True)                                           # the same world with ada idle
    assert cf["final"]["outcomes"]["p:ada"]["requests_served"] == 0
    need = next(n for n, _ in a["listings"])
    assert out["final"]["market"][need]["sellers"] == cf["final"]["market"][need]["sellers"] + 1   # ada changed the market


def test_learners_in_the_arena_update_their_own_weights_online():
    parts = [{"handle": "ada", "goal": "service", "start": "coded"}, {"handle": "bea", "goal": "service", "start": "coded"}]
    arena = Arena("tiny", 9, parts, round_s=1800)
    llm = EmbedOnly()
    players, algos, before = {}, {}, {}
    for i, actor in enumerate(sorted(arena.cores)):
        ag, algo, ch = make_learner({"model_cfg": SMALL, "model": CAMController(ModelConfig(**SMALL)).state_dict(),
                                     "store_keys": np.zeros(0, np.int64), "store_mat": np.zeros((0, 512), np.float32)},
                                    llm, "ppo", {"lr": 1e-3, "epochs": 1, "minibatch_eps": 1, "target_kl": 0.0}, {"use_llm": False}, seed=i)
        players[actor], algos[actor] = (ag, ch), algo
        before[actor] = torch.cat([p.detach().reshape(-1) for p in ag.model.parameters()]).clone()
    cb = online_update(algos, None)
    out = arena.run(players, days=0.5, seed=0, on_segment=cb, segment=8)
    for actor, (ag, _) in players.items():
        after = torch.cat([p.detach().reshape(-1) for p in ag.model.parameters()])
        assert (after != before[actor]).any() and torch.isfinite(after).all()      # each learner learned in the shared world
        assert len(cb.stats[actor]) >= 1
    assert out["events"] > 0


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("run"))
    cfg = {"name": "t", "seed": 0, "device": "cpu", "workers": 0, "iters": 2, "episodes_per_iter": 6, "eval_every": 2,
           "eval_episodes": 4, "ckpt_every": 1, "llm": {"kind": "none"}, "model": SMALL, "agent": {"use_llm": False},
           "core": {"max_decisions": 8}, "algo": {"name": "ppo", "lr": 3e-4, "epochs": 2, "minibatch_eps": 3, "target_kl": 0.0},
           "curriculum": {"world_families": ["tiny"], "arms": [{"family": "found"}]},
           "demos": {"episodes": 4, "bc_epochs": 2}, "store_episodes_iters": 2}
    tr = Trainer(cfg, d)
    tr.run()
    tr.close()
    return d, cfg


def test_training_run_writes_metrics_checkpoints_trajectories_and_changes_weights(smoke_run):
    d, _ = smoke_run
    rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    rl = [r for r in rows if r.get("phase") == "rl"]
    assert len(rl) == 2 and all(r["weights_changed"] for r in rl) and rl[-1]["param_dist_from_init"] > 0
    assert [r["phase"] for r in rows if r.get("phase") in ("demos", "bc")][:1] == ["demos"]
    for f in ("config.json", "meta.json", "episodes.jsonl", "eval.jsonl", "demos.pkl", "ckpt/bc.pt", "ckpt/latest.pt",
              "ckpt/iter_0002.pt", "replay/iter0001.pkl.gz"):
        assert os.path.exists(os.path.join(d, f)), f
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert set(meta["code"]) == {"learner", "protected", "grounding"} and meta["params"]["trainable"] > 0
    ev = [json.loads(l) for l in open(os.path.join(d, "eval.jsonl"))]
    assert [e["env_decisions"] for e in ev][0] == 0 and ev[-1]["env_decisions"] > 0 and "success_ci" in ev[-1]


def test_resume_continues_from_the_checkpoint_with_the_same_counters_and_weights(smoke_run):
    d, cfg = smoke_run
    ck = torch.load(os.path.join(d, "ckpt", "latest.pt"), map_location="cpu", weights_only=False)
    tr = Trainer(cfg, d, resume=True)
    try:
        assert tr.iter == 2 and tr.env_decisions == ck["env_decisions"] and len(tr.demos) > 0
        assert len(tr.algo.buffer) == ck["replay_n"]            # the replay buffer is part of the resumed state
        assert os.path.exists(os.path.join(d, "ckpt", "latest.pt.replay.pkl.gz"))
        for k, v in tr.model.state_dict().items():
            assert torch.equal(v.cpu(), ck["model"][k])
        assert tr.curriculum.counter == ck["curriculum"]["counter"]
        tr.run(iters=1)
        assert tr.iter == 3
    finally:
        tr.close()
    # resuming again from the iteration-2 checkpoint (as after a crash during iteration 3): iteration 3's records
    # are set aside, so the curve has every iteration once
    import shutil
    shutil.copy(os.path.join(d, "ckpt", "iter_0002.pt"), os.path.join(d, "ckpt", "latest.pt"))
    tr = Trainer(cfg, d, resume=True)
    try:
        rl = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
        assert tr.iter == 2 and [r["iter"] for r in rl if r.get("phase") == "rl"] == [1, 2]
        assert os.path.exists(os.path.join(d, "metrics.jsonl.interrupted"))
    finally:
        tr.close()
    # weights and memory can be restored separately (for the ablations)
    ck = torch.load(os.path.join(d, "ckpt", "latest.pt"), map_location="cpu", weights_only=False)
    tr2 = Trainer(cfg, d + "_abl")
    try:
        w0 = {k: v.clone() for k, v in tr2.model.state_dict().items()}
        tr2.load(os.path.join(d, "ckpt", "latest.pt"), weights=False, memory=True)
        assert all(torch.equal(v, w0[k]) for k, v in tr2.model.state_dict().items()) and len(tr2.memory) == len(ck["memory"]["items"])
        tr2.load(os.path.join(d, "ckpt", "latest.pt"), weights=True, memory=False)
        assert any(not torch.equal(v, w0[k]) for k, v in tr2.model.state_dict().items())
    finally:
        tr2.close()


def test_cli_commands_run(smoke_run, tmp_path):
    d, _ = smoke_run
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "src"), SUPERLEARNER_RUNS=str(tmp_path))

    def cli(*args):
        r = subprocess.run([sys.executable, "-m", "superlearner", *args], capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stderr[-1500:]
        return r.stdout
    out = cli("report", "--experiment", d)
    assert "AULC" in out and "updates changed the weights" in out
    traj = sorted(f for f in os.listdir(os.path.join(d, "traj")) if f.startswith("eval"))[-1]
    out = cli("inspect-trajectory", "--trajectory", os.path.join(d, "traj", traj))
    assert "p(head)=" in out and "reward=" in out
    out = cli("evaluate", "--checkpoint", os.path.join(d, "ckpt", "latest.pt"), "--split", "select", "--episodes", "2",
              "--workers", "0", "--run-dir", str(tmp_path / "ev"))
    assert json.loads(out[out.index("{"):])["episodes"] == 2
    lock = str(tmp_path / "final.json")
    assert "seal" in cli("lock-final", "--out", lock, "--n", "6")
    rep = json.loads(cli("doctor", "--quick"))
    assert rep["checks"]["end_to_end_cpu"]["ok"] and rep["hardware"]["cpu_count"] > 0


def test_job_queue_limits_dependencies_and_restart(tmp_path):
    """The experiment queue: respects the CPU limit, runs dependents after their dependency, blocks dependents of a
    failed job, and on a second invocation skips jobs whose result already exists."""
    import json as _json
    from superlearner.jobs import run_jobs
    d = str(tmp_path)
    mark = lambda n: os.path.join(d, n + ".done")          # noqa: E731
    stamp = "import sys,time; open(sys.argv[1]+'.t0','w').write(str(time.time())); time.sleep(0.6); open(sys.argv[1],'w').write(str(time.time()))"
    spec = {"max_gpu": 1, "max_cpu": 4, "jobs": [
        {"name": "a", "cmd": ["python", "-c", stamp, mark("a")], "cpu": 3, "gpu": 1, "done": mark("a")},
        {"name": "b", "cmd": ["python", "-c", stamp, mark("b")], "cpu": 3, "done": mark("b")},
        {"name": "c", "cmd": ["python", "-c", stamp, mark("c")], "cpu": 1, "after": ["a"], "done": mark("c")},
        {"name": "bad", "cmd": ["python", "-c", "raise SystemExit(3)"], "cpu": 1, "retries": 0},
        {"name": "after_bad", "cmd": ["python", "-c", "print(1)"], "cpu": 1, "after": ["bad"]}]}
    st = run_jobs(spec, log_dir=os.path.join(d, "q"), poll_s=0.1, verbose=False)
    assert [st[n]["state"] for n in ("a", "b", "c", "bad", "after_bad")] == ["done", "done", "done", "failed", "blocked"]
    t = lambda n, suf="": float(open(mark(n) + suf).read())          # noqa: E731
    assert t("b", ".t0") >= t("a")                      # a and b together need 6 CPU > 4: b waited for a
    assert t("c", ".t0") >= t("a")                      # c depends on a
    assert _json.load(open(os.path.join(d, "q", "status.json")))["jobs"]["bad"]["returncode"] == 3
    first = t("a")
    st2 = run_jobs(dict(spec, jobs=spec["jobs"][:3]), log_dir=os.path.join(d, "q"), poll_s=0.1, verbose=False)
    assert all(s["state"] == "done" and s["attempts"] == 0 for s in st2.values()) and t("a") == first
