"""Workstream N: the instrument against adversarial learners and the control learners' known mechanics
(DEF-N-2, DEF-N-4, DEF-N-6 regressions; SP-15.8 hooks in `crucible.research.v5.adversarial`)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from crucible.research.v5 import adversarial as A
from crucible.research.v5 import fresh as FR
from crucible.research.v5 import learners as LR
from crucible.research.v5.distribution import build_registry
from crucible.research.v5.episodes import run_episode, score_episode
from crucible.research.v5.sampler import build_env

FAM = "t1:rapid:cx_balancing_loop"


@pytest.fixture(scope="module")
def reg():
    return build_registry()


def _train(L, reg, n=4, reps=2, seed=0):
    rng = np.random.default_rng(seed)
    for s in range(310, 310 + n):
        for _ in range(reps):
            run_episode(build_env(reg[FAM], s, {}, {"mode": "assigned"}), L, learn=True, rng=rng, verify=False)


def test_instrument_worlds_are_observation_only_and_learners_read_only_their_cash(reg):
    """DEF-N-4: the v5 instrument builds its worlds through crucible.experience in its default mode; no reward
    reaches a learner (scalar 0.0) and the control learners' signal is the change of their own cash."""
    env = build_env(reg[FAM], 3, {}, {"mode": "assigned"})
    assert env.reward_mode == "observation_only"
    a = env.agents()[0]
    run_episode(env, LR.make("adapter", 0), learn=False, rng=np.random.default_rng(0))
    assert env.reward(a) == 0.0
    seen = []

    class Spy(LR.Adapter):
        def observe(self, k, s):
            seen.append(s)
            super().observe(k, s)
    env2 = build_env(reg[FAM], 3, {}, {"mode": "assigned"})
    rec = run_episode(env2, Spy(0), learn=False, rng=np.random.default_rng(0))
    assert len(seen) == rec["ticks"] and abs(sum(seen) - rec["observed_signal_sum"]) < 1e-6


def test_fresh_child_is_jailed_and_checks_its_confinement(reg, tmp_path):
    w = LR.make("weights", 1)
    _train(w, reg)
    man = FR.save_checkpoint(w, str(tmp_path / "w"))
    r = FR.weights_only_eval(man, [{"entry": FAM, "seed": 1_000_101}])
    assert r["ok"], r
    assert set(r["checks"]) == set(FR.CHECKS) and r["checks"]["filesystem_confined"] and r["jailed"]
    open_ = FR.weights_only_eval(man, [{"entry": FAM, "seed": 1_000_101}], jail=False)
    assert not open_["ok"] and open_["checks"]["filesystem_confined"] is False


def test_configuration_smuggling_is_refused(reg, tmp_path):
    L = LR.make("adv_config_smuggle", 3)
    _train(L, reg, n=2, reps=1)
    man = FR.save_checkpoint(L, str(tmp_path / "s"))
    assert json.load(open(os.path.join(man["dir"], "arch.json")))["cfg"]["table"]
    r = FR.weights_only_eval(man, [{"entry": FAM, "seed": 310}])
    assert not r["ok"] and r["checks"]["authorised_configuration"] is False and "table" in r["refused"]


def test_file_stash_outside_the_checkpoint_is_unreadable_under_the_jail(reg, tmp_path):
    L = LR.make("adv_file_stash", 9_901)
    try:
        os.unlink(L._stash())
    except OSError:
        pass
    worlds = [{"entry": FAM, "seed": s} for s in (1_000_101, 1_000_102)]
    L.frozen = True                                         # scored as the weights-only child plays: no search
    base = np.mean([score_episode(reg, w["entry"], w["seed"], L, learn=False, rng=np.random.default_rng(0))["score"]
                    for w in worlds])
    L.frozen = False
    _train(L, reg)
    try:
        man = FR.save_checkpoint(L, str(tmp_path / "f"))
        jailed = FR.weights_only_eval(man, worlds)
        unjailed = FR.weights_only_eval(man, worlds, jail=False)
    finally:
        os.unlink(L._stash())
    fj = np.mean([x["score"] for x in jailed["scores"]])
    fu = np.mean([x["score"] for x in unjailed["scores"]])
    assert jailed["ok"] and abs(fj - base) < 1e-9          # the jail credits nothing to the stash
    assert fu - base > A.GAIN_MIN                           # without the jail the stash would have worked


def test_weights_replay_is_memory_not_parameters(reg):
    w = LR.make("weights", 2)
    _train(w, reg, n=2, reps=1)
    assert len(w.memory_state()["replay"]) == 2 and set(w.parameters()) == {"W1", "b1", "W2", "b2"}
    w.clear_memory()
    assert w.memory_state() == {"replay": []}


def test_transfer_learner_credits_its_strategy_from_the_second_episode(reg):
    """Regression: the bandit was credited only at a task's 4th episode, so a battery playing each training world 3
    times left the transfer control untrained."""
    T = LR.make("transfer", 4)
    _train(T, reg, n=3, reps=3)
    assert T.bandit.n.sum() == 3
    T2 = LR.make("transfer", 4)
    _train(T2, reg, n=3, reps=4)
    assert T2.bandit.n.sum() == 3                            # a task counts once, whatever its episode count


def test_generator_has_no_fixed_rule_shortcut():
    """DEF-N-6: fixed public rules (the audit's anti-default rule, quartiles, centre) reach less than SHORTCUT_MAX
    of the oracle's improvement in the median calibrated generated world."""
    r = A.calibration_shortcut(n_structures=8, seeds=(0,))
    assert r["worlds"] >= 6
    assert r["passed"], r["median_share_of_oracle"]


def _wandering_default_and_centre(cfg):
    """(public default of the key lever, hidden centre c0) of a composed wandering-optimum world: c0 enters the two
    logistic fit edges of the key lever as biases -(c0 - w) and -(c0 + w) (`compose._wandering`)."""
    key = cfg["claim"]["lever"]
    kn = next(nd for nd in cfg["nodes"] if nd["id"] == key)
    edges = [nd for nd in cfg["nodes"] if nd.get("type") == "prim" and (nd.get("bias") or {}).get("x") is not None
             and any(t[0] == key for t in (nd.get("ports") or {}).get("x", []))]
    b = sorted(-float(nd["bias"]["x"]) for nd in edges[-2:])
    return float(kn["lever"]["default"]), (b[0] + b[1]) / 2.0, float(kn["lever"]["lo"]), float(kn["lever"]["hi"])


def test_wandering_default_does_not_disclose_the_hidden_centre():
    """DEF-N-8 (verifier probe probe_t4.py): the public default of a wandering-optimum lever was the hidden centre c0
    of the optimum's path (|default - c0| <= 9e-16 in 24 of 24 worlds), so a tracker starting at the default started
    at the hidden optimum and admission's dominance check measured gains over a waiting policy already sitting
    there. The default is now drawn uniformly over the lever's range independently of c0: in 120 composed worlds it
    is nowhere near c0 systematically (share within 0.25 lever units about what independence gives, ~0.05), its
    correlation with c0 is small, and it covers the whole range."""
    from crucible.research.v5 import compose as C
    rng = np.random.default_rng(2026)
    s = C.normalize({"topology": "chain", "motifs": ["saturation"], "combiner": "sum", "temporal": "instant",
                     "channels": [], "n_levers": 1, "drivers": ["wandering_optimum"], "population": {"kind": "none"},
                     "horizon": 60})
    rows = [_wandering_default_and_centre(C._compose_once(rng, s, {})) for _ in range(120)]
    d = np.array([r[0] for r in rows])
    c0 = np.array([r[1] for r in rows])
    lo, hi = rows[0][2], rows[0][3]
    assert np.all((c0 >= 1.9) & (c0 <= 8.1))                 # the fit edges were read correctly
    near = float(np.mean(np.abs(d - c0) < 0.25))
    assert near <= 0.15, near                                # was 1.0 (default == c0)
    assert abs(float(np.corrcoef(d, c0)[0, 1])) <= 0.25
    assert d.min() <= lo + 0.15 * (hi - lo) and d.max() >= hi - 0.15 * (hi - lo)


def test_wandering_reference_reads_no_hidden_parameter():
    """DEF-N-8: the scripted reference of wandering worlds (`genref.play`) acts only through the public view; it
    scans the lever range first because the default carries no information. Its first SCAN actions are the evenly
    spaced scan settings whatever the world's hidden centre is (two worlds with centres on opposite sides)."""
    from crucible.rapid.families import make
    from crucible.rapid.identify import PublicView
    from crucible.research.v5 import compose as C
    from crucible.research.v5 import genref as GR
    s = C.normalize({"topology": "chain", "motifs": ["saturation"], "combiner": "sum", "temporal": "instant",
                     "channels": [], "n_levers": 1, "drivers": ["wandering_optimum"], "population": {"kind": "none"},
                     "horizon": 40})
    firsts = []
    with C.registered(s) as names:
        for sd in (0, 1, 2, 3):
            env = make(names[0], sd)
            seen = []

            class V(PublicView):
                def step(self, a):
                    seen.append(sorted((a or {}).get("values", {}).values()))
                    return super().step(a)
            GR.play(V(env), env)
            firsts.append([round(v[0], 6) for v in seen[:GR.SCAN]])
            assert len(seen) == env.horizon
    lo, hi = 0.0, C.LEVER_HI
    expect = [round(lo + (i + 0.5) / GR.SCAN * (hi - lo), 6) for i in range(GR.SCAN)]
    assert all(f == expect for f in firsts), firsts


def test_control_labels_control_false_labels_and_keep_power():
    """The instrument's own statistics, adversarially: on crossed null data (learner-seed and world effects, no
    learner effect) a label is given in at most 1% of batteries at `instrument.LABEL_LEVEL`, for learner-seed sds
    0.05-0.15 (the two development seeds of the controls put the seed sd of their known effects at 0.00-0.14).
    Power, at a world sd of 0.5 of the differences: effects of 0.4 at seed sd 0.1 and 0.5 at seed sd 0.15 (the
    pretrained control's development effect, its noisiest) are labelled in at least 90% of batteries. Known limit,
    measured here and reported in N.json: with five learner seeds at the family-wise level an effect of 0.3 at seed
    sd >= 0.1 is labelled in only about three quarters of batteries."""
    from crucible.research.v5 import instrument as I
    rng = np.random.default_rng(11)
    T = 600

    def draw(sds):
        return rng.normal(0, sds, (5, 1)) + rng.normal(0, 0.5, (1, 50)) + rng.normal(0, 0.1, (5, 50))

    for sds in (0.05, 0.1, 0.15):
        fp = 0
        for _ in range(T):
            r0 = I._one(draw(sds))
            fp += int(r0["lo"] > 0 and r0["estimate"] >= 0.03)
        assert fp / T <= 0.01, (sds, fp / T)
    for sds, eff in ((0.1, 0.4), (0.15, 0.5)):
        hit = 0
        for _ in range(T):
            r1 = I._one(draw(sds) + eff)
            hit += int(r1["lo"] > 0 and r1["estimate"] >= 0.10)
        assert hit / T >= 0.9, (sds, eff, hit / T)


def test_widened_layers_checkpoint_is_refused(reg, tmp_path):
    """SP-15.8 (security workstream's checkpoint_tampering control): a checkpoint whose network is wider than the
    declared architecture, under a manifest the adversary rehashed, is refused by its shapes; the honest checkpoint
    of the same learner is accepted."""
    w = LR.make("weights", 2)
    _train(w, reg, n=2, reps=1)
    man = FR.save_checkpoint(w, str(tmp_path / "w"))
    spec = [{"entry": FAM, "seed": 1_000_102}]
    assert FR.weights_only_eval(man, spec)["ok"]
    wide = FR.tampered_copy(man, "widened_layers", str(tmp_path / "wide"))
    with np.load(os.path.join(wide["dir"], "params.npz")) as z:
        assert z["W1"].shape[1] == 2 * w.parameters()["W1"].shape[1]
        assert LR.params_hash({k: z[k] for k in z.files}) == wide["params_hash"]     # the manifest matches
    r = FR.weights_only_eval(wide, spec)
    assert not r["ok"] and r["checks"]["authorised_parameters"] is False and "shapes" in r["refused"]


@pytest.mark.parametrize("dtype", ["complex128", "float32", "float16", "int64"])
def test_retyped_checkpoint_is_refused(reg, tmp_path, dtype):
    """DEF-O-12 (verifier probe probe_sp158.py): a 'weights' checkpoint with the declared shapes but another dtype
    (complex128 carries a second real per entry), under a manifest the adversary rehashed, is refused by its dtypes;
    the honest checkpoint of the same learner is accepted."""
    w = LR.make("weights", 2)
    _train(w, reg, n=2, reps=1)
    man = FR.save_checkpoint(w, str(tmp_path / "w"))
    spec = [{"entry": FAM, "seed": 1_000_102}]
    assert FR.weights_only_eval(man, spec)["ok"]
    bad = FR.tampered_copy(man, f"retyped:{dtype}", str(tmp_path / dtype))
    with np.load(os.path.join(bad["dir"], "params.npz")) as z:
        assert all(z[k].dtype == np.dtype(dtype) and z[k].shape == w.parameters()[k].shape for k in z.files)
        assert LR.params_hash({k: z[k] for k in z.files}) == bad["params_hash"]     # the manifest matches
    r = FR.weights_only_eval(bad, spec)
    assert not r["ok"] and r["checks"]["authorised_parameters"] is False and "dtypes" in r["refused"]


def test_evaluator_enforces_the_interaction_budget(reg):
    """SP-15.8 (interaction_budget control): a learner granted k informative interactions that calls its
    participant environment itself gets exactly k executed; the evaluator refuses the rest and reports them."""
    from crucible.research.v5 import evaluator as EV
    ev = EV.Evaluator.start()
    try:
        t = ev.open_episode(FAM, 1_000_103, purpose="dev", budget=2)
        env = ev.participant_env(t)
        a = env.agents()[0]
        tool = next(x for x in env.tools(a) if x["name"] == "rapid.levers")
        names = sorted(tool["inputSchema"]["properties"]["values"]["properties"])
        ok = refused = 0
        for i in range(6):
            try:
                env.execute(a, {"tool": "rapid.levers", "args": {"values": {names[0]: 1.0 + i}}})
                ok += 1
            except EV.EvaluatorError:
                refused += 1
            env.advance(3600)
        res = ev.result(t, view="development")
        assert ok == 2 and refused == 4
        assert res["interactions"] == 2 and res["budget"] == 2 and res["refused_beyond_budget"] == 4
        assert res["budget_exceeded"] is False
        with pytest.raises(EV.EvaluatorError):
            ev.open_episode(FAM, 1_000_104, purpose="dev", budget=-1)
    finally:
        ev.close()


def test_hosted_learner_cannot_reach_the_evaluator(reg, tmp_path):
    """DEF-O-10: the same adversarial prober finds the evaluator when driven in the harness process
    (`evaluator.run_remote_episode`) and nothing when hosted (`hosted.HostedLearner`: jailed child, participant
    operations relayed); the hosted episode is still verified by the evaluator."""
    r = A.evaluator_reach(reg, str(tmp_path))
    assert r["in_process_found"], r
    assert r["hosted_found"] == {} and r["hosted_uid"] != 0 and r["hosted_verified_ticks"], r
    assert r["passed"]


def test_hosted_learner_trains_and_only_relays_participant_operations(reg):
    from crucible.research.v5 import evaluator as EV
    from crucible.research.v5.hosted import HostedError, HostedLearner, run_hosted_episode
    with EV.Evaluator.start() as ev:
        with HostedLearner("crucible.research.v5.learners:make", {"arch": "weights", "seed": 3}) as h:
            d0 = h.call("parameters_digest")
            for i in range(2):
                t = ev.open_episode(FAM, 320 + i, purpose="dev", budget=8)
                rec = run_hosted_episode(ev, t, h, learn=True, seed=i, budget=8)
                assert rec["ticks"] > 0 and rec["interactions"] <= 8 and rec["budget_exceeded"] is False
            assert h.call("parameters_digest") != d0          # it learned, inside the jail
            with pytest.raises(HostedError):
                h.call("load_parameters")                      # only the declared calls
