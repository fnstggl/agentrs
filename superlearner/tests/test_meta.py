"""The Meta-Researcher's machinery: confinement, protected code, generated learning rules, trusted measurement."""
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

from superlearner.meta import proposals as P
from superlearner.meta.compare import compare, pareto, run_measures, verdict
from superlearner.meta.ledger import Ledger
from superlearner.meta.surrogate import Surrogate
from superlearner.meta.workspace import IsolationUnavailable, create_candidate, is_editable, probe_isolation, tree_hashes, verify

ROOT = os.geteuid() == 0


@pytest.mark.skipif(not ROOT, reason="dropping privileges needs root")
def test_confinement_is_checked_from_inside_and_denies_files_network_and_privileges():
    r = probe_isolation(gpu=False, force=True)
    assert r["usable"], r
    inside = r["inside"]
    assert inside["uid"] != 0
    for k in ("read_outside_allowlist", "read_home", "write_outside_workdir", "write_own_source", "tcp_socket", "udp_socket", "setuid_root"):
        assert inside[k].startswith("denied"), (k, inside[k])
    assert inside["unix_socket_to_trusted_service"] == "ALLOWED" and inside["write_workdir"] == "ALLOWED"


def test_only_learner_files_are_candidate_editable_and_protected_changes_are_rejected(tmp_path):
    assert is_editable("algo/rule.py") and is_editable("cam/model.py") and is_editable("train/curriculum.py")
    for bad in ("evalkit/metrics.py", "world/core.py", "world/tasks.py", "meta/compare.py", "train/trainer.py",
                "../outside.py", "/etc/passwd", "algo/data.bin", "llm/base.py"):
        assert not is_editable(bad)
    with pytest.raises(PermissionError):
        create_candidate(str(tmp_path / "evil"), {"evalkit/metrics.py": "print('higher scores')\n"})
    rec = create_candidate(str(tmp_path / "ok"), {"algo/rule.py": P.render_rule(P.BASE_GENOME)})
    assert rec["protected_ok"] and rec["changed"] == ["algo/rule.py"] and verify(rec["path"])[0]
    # tampering with a protected file after creation is detected by hash before anything runs
    with open(os.path.join(rec["pkg"], "world", "core.py"), "a") as f:
        f.write("\n# reward *= 10\n")
    okv, why = verify(rec["path"])
    assert not okv and "world/core.py" in why
    assert tree_hashes(rec["pkg"])["protected"] != rec["protected_hash"]


def _load(src: str, name: str):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


class _B:
    pass


def _numeric_check(mod) -> float:
    cfg = mod.RuleConfig()
    Bn, T, K = 3, 6, 4
    torch.manual_seed(0)
    reward, value = torch.randn(Bn, T) * 0.1, torch.randn(Bn, T) * 0.1
    done, trunc = torch.zeros(Bn, T), torch.zeros(Bn, T)
    done[:, -1] = 1
    valid = torch.ones(Bn, T, dtype=torch.bool)
    adv, ret = mod.advantages(reward, value, done, trunc, valid, cfg)
    logp = (torch.randn(Bn, T) * 0.1 - 2).requires_grad_(True)
    out = {"logp": logp, "ent": torch.rand(Bn, T) + 1, "ent_head": torch.rand(Bn, T),
           "values": torch.randn(Bn, T, K, requires_grad=True), "succ_logit": torch.randn(Bn, T, requires_grad=True)}
    b = _B()
    b.t = {"logp_old": logp.detach() + 0.05 * torch.randn(Bn, T), "valid": valid, "value_old": value,
           "success": torch.tensor([1.0, 0.0, 1.0]), "ok": (torch.rand(Bn, T) > 0.3).float()}
    loss, info = mod.total_loss(out, b, adv, ret, cfg, torch.ones(Bn, K))
    rl = mod.replay_loss(out, b, ret, cfg)
    assert torch.isfinite(rl) and isinstance(mod.REPLAY_STEPS, int) and mod.REPLAY_STEPS >= 0
    (loss + 0.0 * rl).backward()
    assert torch.isfinite(loss) and torch.isfinite(adv).all() and adv.shape == reward.shape
    assert logp.grad.abs().sum() > 0 and out["values"].grad.abs().sum() > 0 and "kl" in info
    return float(loss.detach())


def test_every_fragment_of_the_rule_program_space_renders_to_a_working_learning_rule():
    rs = P.RuleSpace(0)
    n = 0
    for slot, (table, _) in P.SLOT_TABLE.items():
        for name, (_, spec, _) in table.items():
            g = json.loads(json.dumps(P.BASE_GENOME))
            g[slot] = [name, P._sample_params(spec, rs.rng)]
            _numeric_check(_load(P.render_rule(g), f"rule_{slot}_{name}"))
            n += 1
    assert n == sum(len(t) for t, _ in P.SLOT_TABLE.values()) >= 23
    # the baseline genome reproduces the hand-written baseline rule numerically
    from superlearner.algo import rule as R
    assert abs(_numeric_check(_load(P.render_rule(P.BASE_GENOME), "rule_base")) - _numeric_check(R)) < 1e-5


def test_rule_space_proposals_are_distinct_programs_with_hypotheses():
    rs = P.RuleSpace(1)
    props = [rs.propose(P.BASE_GENOME) for _ in range(25)]
    assert len({p["id"] for p in props}) >= 20                       # a space of programs, not a short menu
    for p in props:
        assert p["source"] == "rule_space" and p["hypothesis"] != "no change" and p["lines_changed"] > 0
        assert p["changes"][P.RULE_PATH] != P.render_rule(P.BASE_GENOME) and p["slots"]
        _numeric_check(_load(p["changes"][P.RULE_PATH], "rule_" + p["id"]))


def test_learned_rule_starts_at_the_baseline_and_its_search_moves_the_meta_parameters():
    base = _numeric_check(_load(P.render_rule(P.BASE_GENOME), "rule_base2"))
    assert abs(_numeric_check(_load(P.render_learned(P.phi_init()), "rule_learned0")) - base) < 1e-4
    lr = P.LearnedRule(0)
    props = [lr.propose() for _ in range(4)]
    for p in props:
        _numeric_check(_load(p["changes"][P.RULE_PATH], "rule_" + p["id"]))
    e = [lr.pending[p["id"]] for p in props]
    assert np.allclose(e[0], -e[1]) and np.allclose(e[2], -e[3])        # antithetic pairs
    for p, f in zip(props, (0.30, 0.10, 0.25, 0.20)):
        lr.tell(p["id"], f)
    before = lr.mean.copy()
    step = lr.step()
    assert step["generation"] == 0 and lr.gen == 1 and np.linalg.norm(lr.mean - before) > 0
    assert (lr.mean - before) @ e[0] > 0                               # moved towards the better perturbation


def test_llm_written_functions_are_spliced_only_if_they_parse_and_keep_the_signature():
    src = P.render_rule(P.BASE_GENOME)
    new = "def entropy_loss(ent, valid, cfg: RuleConfig):\n    return -_mean(ent.clamp(max=4.0), valid)\n"
    out = P.splice_function(src, "entropy_loss", new)
    assert "clamp(max=4.0)" in out and out.count("def entropy_loss") == 1
    _numeric_check(_load(out, "rule_spliced"))
    with pytest.raises(ValueError):
        P.splice_function(src, "entropy_loss", "def entropy_loss(ent, valid):\n    return ent.mean()\n")      # arguments changed
    with pytest.raises(ValueError):
        P.splice_function(src, "entropy_loss", "def other(x):\n    return x\n")
    with pytest.raises(SyntaxError):
        P.splice_function(src, "entropy_loss", "def entropy_loss(ent, valid, cfg):\n    return (\n")
    with pytest.raises(ValueError):
        P.splice_function(src, "entropy_loss", "import os\ndef entropy_loss(ent, valid, cfg):\n    return ent.mean()\n")


def test_llm_proposer_records_unusable_replies_and_returns_executable_proposals():
    from superlearner.llm.base import ScriptedLLM
    replies = iter(["I think you should try a different optimiser.",
                    json.dumps({"hypothesis": "bound the entropy term", "expected": "entropy stays lower",
                                "code": "def value_loss(values, ret, value_old, valid, cfg, boot=None):\n"
                                        "    per = (values - ret.unsqueeze(-1)).abs()\n"
                                        "    w = valid.to(values.dtype).unsqueeze(-1).expand_as(per)\n"
                                        "    return (per * w).sum() / w.sum().clamp(min=1.0)\n"})])
    pr = P.LLMProposer(ScriptedLLM(lambda m, v: next(replies)), think=False)
    assert pr.propose({}, "diagnosis", "", function="value_loss") is None and len(pr.failures) == 1
    p = pr.propose({}, "diagnosis", "", function="value_loss")
    assert p["source"] == "llm" and p["slots"] == ["value_loss"] and ".abs()" in p["changes"][P.RULE_PATH]
    _numeric_check(_load(p["changes"][P.RULE_PATH], "rule_llm"))


def test_surrogate_learns_from_outcomes_and_changes_what_it_selects():
    rs = P.RuleSpace(2)
    props = [rs.propose(P.BASE_GENOME) for _ in range(40)]
    s = Surrogate(None, seed=0)
    assert s.fit()["fitted"] is False and len(s.select(props[:5], 2)) == 2       # no data: uniform choice

    def truth(p):          # a hidden regularity: changes to the policy objective help, value-loss changes hurt
        return 0.08 * ("policy_loss" in p["slots"]) - 0.06 * ("value_loss" in p["slots"])
    for p in props[:28]:
        s.add(p, truth(p), True)
    info = s.fit()
    assert info["fitted"] and info["n"] == 28 and s.updates == 1
    mu, sd, pv = s.predict(props[28:])
    y = np.array([truth(p) for p in props[28:]])
    assert np.corrcoef(mu, y)[0, 1] > 0.6                                         # predicts held-out proposals
    top = s.select(props[28:], 3, beta=0.0)
    assert np.mean([truth(props[28 + i]) for i in top]) > y.mean()
    assert s.loo_report()["spearman"] > 0.3
    s2 = Surrogate(None, seed=0)
    s2.load_state_dict(s.state_dict())
    assert np.allclose(s2.predict(props[28:])[0], mu, atol=1e-5)


def test_measures_come_from_trusted_records_and_comparisons_carry_uncertainty():
    recs = []
    for stamp, rate in ((0, 0.2), (2000, 0.4), (4000, 0.8)):
        recs += [{"kind": "eval", "stamp": stamp, "summary": {"success": i < rate * 10, "decisions": 5, "commands": 5, "commands_ok": 4}} for i in range(10)]
    recs += [{"kind": "train", "stamp": 100 * (i + 1), "summary": {"success": i > 25, "decisions": 10, "commands": 10, "commands_ok": 7}} for i in range(40)]
    m = run_measures(recs, 4000)
    assert m["eval_rounds"] == 3 and m["first_eval"] == 0.2 and m["final_eval"] == 0.8
    assert abs(m["aulc_eval"] - ((0.2 + 0.4) / 2 * 0.5 + (0.4 + 0.8) / 2 * 0.5)) < 1e-6
    assert m["train_decisions"] == 400 and abs(m["train_valid_rate"] - 0.7) < 1e-9
    c = compare({1: 0.50, 2: 0.55, 3: 0.52, 4: 0.58}, {1: 0.40, 2: 0.41, 3: 0.45, 4: 0.44})
    assert c["paired"] and c["diff"] > 0.08 and c["ci"][0] > 0 and c["p_improve"] > 0.99 and c["established"]
    assert verdict(c, {"diff": 0.0}) == "improved"
    c1 = compare({1: 0.50}, {1: 0.40})
    assert not c1["established"] and verdict(c1, {"diff": 0.0}) == "inconclusive"       # one seed proves nothing
    c2 = compare({1: 0.30, 2: 0.31, 3: 0.29}, {1: 0.40, 2: 0.41, 3: 0.45})
    assert verdict(c2, {"diff": -0.1}) == "harmful"
    assert verdict(c, {"diff": -0.2}) == "inconclusive"                                 # faster but ends lower
    par = pareto({"a": {"aulc_eval": 0.5, "final_eval": 0.8, "wall_s": 100}, "b": {"aulc_eval": 0.4, "final_eval": 0.7, "wall_s": 120},
                  "c": {"aulc_eval": 0.45, "final_eval": 0.9, "wall_s": 300}})
    assert par == {"a": True, "b": False, "c": True}


def test_env_hub_writes_trusted_results_and_enforces_the_experience_budget(tmp_path):
    from multiprocessing.connection import Client
    from superlearner.world.core import RemoteCore
    from superlearner.world.hub import AUTHKEY, start_hub
    from superlearner.world.tasks import task
    res, sock = str(tmp_path / "trusted.jsonl"), str(tmp_path / "env.sock")
    hub = start_hub(sock, [task("found", "tiny", 91, "select")], cfg={"max_decisions": 4}, results_path=res, budget={"train_decisions": 6})
    try:
        rc = RemoteCore(Client(sock, family="AF_UNIX", authkey=AUTHKEY))
        for i in range(2):
            rc.reset(task("found", "tiny", 500 + i))
            while True:
                rc.wait(60)
                if rc.end_decision()["truncated"]:
                    break
            rc.summary()
        assert rc.budget()["train_decisions_used"] == 8
        with pytest.raises(RuntimeError, match="budget exhausted"):
            rc.reset(task("found", "tiny", 502))
        rc.reset(eval_index=0)                       # evaluation is still served after the training budget is spent
        rc.wait(60)
        rc.end_decision()
        rc.summary()
    finally:
        hub.terminate()
    rows = [json.loads(l) for l in open(res)]
    assert [r["kind"] for r in rows] == ["train", "train", "eval"] and [r["stamp"] for r in rows] == [4, 8, 8]
    assert rows[2]["summary"]["task"].startswith("found:entrenched") or rows[2]["summary"]["task"].startswith("found:tiny")


def test_ledger_keeps_every_experiment_and_summarises_history(tmp_path):
    led = Ledger(str(tmp_path / "ledger.jsonl"))
    led.append({"kind": "experiment", "candidate": "c1", "source": "llm", "target": "algo/rule.py", "decision": "rejected_invalid",
                "stage_reached": "verify", "reject_reason": "self-test failed", "hypothesis": "h1", "comparison": {}})
    led.append({"kind": "experiment", "candidate": "c2", "source": "rule_space", "target": "algo/rule.py", "decision": "inconclusive",
                "stage_reached": "develop", "reject_reason": "", "hypothesis": "h2",
                "comparison": {"aulc_eval": {"diff": 0.012, "ci": [-0.02, 0.04], "n_cand": 3}}})
    assert len(led.experiments()) == 2 and [r["decision"] for r in led.table()] == ["rejected_invalid", "inconclusive"]
    txt = led.summary_text()
    assert "rejected_invalid" in txt and "+0.012" in txt and "self-test failed" in txt


def test_inert_changes_are_told_apart_from_real_ones_by_the_confined_probe(tmp_path):
    """A candidate that only adds an unused variable computes the same update as the champion on fixed data; one
    that changes the discount does not. (Found in the real search: both LLM-written candidates of a round were
    of the first kind, and their measured 'effects' were run-to-run noise.)"""
    import json as _json
    from superlearner.meta.workspace import PKG, create_candidate, probe_isolation, run_isolated
    from superlearner.train.trainer import Trainer
    if os.getuid() != 0 or not probe_isolation(gpu=False).get("usable", False):
        pytest.skip("needs the confinement")
    small = {"d": 64, "layers": 2, "heads": 4, "ff": 128, "bag_dim": 32, "hid": 64}
    cfg = {"name": "t", "seed": 0, "device": "cpu", "workers": 0, "llm": {"kind": "none"}, "model": small, "agent": {"use_llm": False},
           "core": {"max_decisions": 8}, "algo": {"name": "ppo", "epochs": 1, "minibatch_eps": 4, "critic_warmup": 1},
           "curriculum": {"world_families": ["tiny"], "arms": [{"family": "found"}]}, "demos": {"episodes": 4, "bc_epochs": 1}}
    import shutil
    import tempfile
    root = tempfile.mkdtemp(prefix="sl-test-", dir="/var/tmp")      # pytest's tmp_path is not reachable for uid nobody
    os.chmod(root, 0o755)
    request_cleanup = lambda: shutil.rmtree(root, ignore_errors=True)          # noqa: E731
    tmp_path = type(tmp_path)(root)
    demo_dir = str(tmp_path / "demos")
    tr = Trainer(cfg, demo_dir)
    try:
        assert tr.collect_demos(4)["kept"] >= 2
    finally:
        tr.close()
    os.chmod(demo_dir, 0o755)
    os.chmod(os.path.join(demo_dir, "demos.pkl"), 0o644)
    rule = open(os.path.join(PKG, "algo", "rule.py")).read()
    line = "    al, _ = aux_loss(out, batch, cfg)\n"
    assert line in rule and "gamma: float = 0.995" in rule
    trees = {"champion": {}, "inert": {"algo/rule.py": rule.replace(line, line + "    unused = adv.mean() * 0.0\n", 1)},
             "active": {"algo/rule.py": rule.replace("gamma: float = 0.995", "gamma: float = 0.9", 1)}}
    fp = {}
    for name, changes in trees.items():
        dest = str(tmp_path / name)
        os.makedirs(dest)
        create_candidate(dest, changes)
        work = os.path.join(dest, "work")
        os.makedirs(work)
        jp = os.path.join(work, "job.json")
        _json.dump({"stage": "probe", "config": cfg, "workdir": work, "demos_path": os.path.join(demo_dir, "demos.pkl")}, open(jp, "w"))
        res = run_isolated("superlearner.meta.candidate_run", [jp], os.path.join(dest, "src"), work, timeout_s=300, extra_read=(demo_dir,))
        assert res["returncode"] == 0, res.get("stdout", "")[-600:]
        fp[name] = _json.load(open(os.path.join(work, "probe_result.json")))["fingerprint"]
    request_cleanup()
    assert fp["inert"] == fp["champion"] and fp["champion"]["delta_norm"] > 0
    assert fp["active"] != fp["champion"]
