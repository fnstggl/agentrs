"""End-to-end flow of Workstream B's modules against the LabWorld diagnostic environment (known ground truth):

    discover tools -> schema-walker candidates -> world-model prediction -> prospective registration ->
    randomised experiment -> execution -> delayed outcome -> resolution -> learning from matured outcomes ->
    calibration of the next, prospectively registered predictions.

The only stand-in is ``encode`` below: a fixed random projection of the candidates' *exact* channel features and
effort into a decision latent. It replaces Workstream A's encoders (not yet available in stage 1); everything under
test — heads, likelihoods, masks, registries, selection, analysis — is the real code. LabWorld results are
diagnostic, never evidence about a live task.
"""
import math

import numpy as np
import pytest
import torch

from cam_live.contracts import OutcomeKind, OutcomeQuery, OutcomeStatus, PredictionRecord, RunFiles, new_id
from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabWorld, lab_task
from cam_live.predict import policy as P
from cam_live.predict.candidates import schema_candidates
from cam_live.predict.registry import ExperimentRegistry, PredictionRegistry, RegistryError
from cam_live.predict.world import K_COUNT, KindNormaliser, WorldConfig, WorldHeads, build_targets, world_nll

DM, DAY = 16, 86400.0


class Lab:
    def __init__(self, tmp_path, seed=3):
        self.world = LabWorld(LabConfig(seed=seed, budget=1e9, key_door=False, noisy_tv=False))
        self.clock = self.world.clock
        self.task = lab_task(self.world)
        self.auth = AllowAllPolicy()
        self.tools = [t for t in self.world.discover_tools() if t.name == "lab.act"]
        self.preds = PredictionRegistry(str(tmp_path / RunFiles.PREDICTIONS), self.clock)
        self.exps = ExperimentRegistry(str(tmp_path / RunFiles.EXPERIMENTS), self.preds, self.clock, seed=5)
        feats = {}
        for obs in self.world.observe():                                # exact quantities, not text
            for q in obs.quantities:
                if q.name.startswith("channel."):
                    _, ch, f = q.name.split(".")
                    feats.setdefault(int(ch), {})[f] = q.value
        self.feats = torch.tensor([[feats[c][f"f{j}"] for j in range(3)] for c in sorted(feats)])
        g = torch.Generator().manual_seed(0)
        self.proj = torch.randn(5, DM, generator=g)
        self.e_q = torch.randn(1, 1, DM, generator=g)
        self.heads = WorldHeads(WorldConfig(d_model=DM, hidden=64, n_members=3, n_hazard_bins=4, max_categories=4))
        self.pending = []                                               # (prediction id, query, z, action)

    def encode(self, actions):
        x = torch.stack([torch.cat([self.feats[a.arguments["channel"]],
                                    torch.tensor([a.arguments["effort"] / 3.0, 1.0])]) for a in actions])
        return x @ self.proj                                            # [N, D]

    def candidates(self, tag):
        return schema_candidates(self.tools, per_tool=18, max_per_field=6, id_prefix=tag)

    def predict_register_execute(self, action, experiment_id=""):
        """One decision: predict the delayed consequence, register the prediction, only then act."""
        tool = self.tools[0]
        query = OutcomeQuery(query_id=new_id("q"), metric=f"lab.adopters.from.{action.action_id}",
                             kind=OutcomeKind.COUNT, horizon_s=self.world.cfg.outcome_horizon_s,
                             issued_at=self.clock.now() + tool.est_cost.seconds, exposure=float(action.arguments["effort"]),
                             experiment_id=experiment_id)
        z = self.encode([action])[None]                                 # [1, 1, D]
        with torch.no_grad():
            out = self.heads(z, self.e_q, torch.tensor([[K_COUNT]]), torch.tensor([[query.horizon_s]], dtype=torch.float64),
                             exposure=torch.tensor([[query.exposure]]))
        rec = PredictionRecord(new_id("pred"), query, out.to_distribution(0, 0, 0), made_at=self.clock.now(),
                               action_id=action.action_id)
        self.preds.register(rec)                                        # before the action exists in the world
        result = self.world.execute(action, self.auth.authorize(action, self.task))
        assert result.status == "ok", result.failure_detail
        self.pending.append((rec.prediction_id, query, z[0, 0], action))
        return rec, result

    def mature(self):
        """Measure every due query and resolve its prediction. Returns (query, observation, z) triples."""
        done, still = [], []
        for pid, query, z, action in self.pending:
            (obs,) = self.world.measure([query])
            if obs.status == OutcomeStatus.PENDING:
                still.append((pid, query, z, action))
                continue
            self.preds.resolve(pid, obs)
            done.append((query, obs, z, action))
        self.pending = still
        return done

    def train(self, triples, steps=300, seed=0):
        g = torch.Generator().manual_seed(seed)
        meta, tgt = build_targets([[(q, o)] for q, o, _, _ in triples])
        z = torch.stack([zz for _, _, zz, _ in triples])[:, None]
        opt = torch.optim.Adam(self.heads.parameters(), lr=3e-3)
        norm = KindNormaliser()
        n = len(triples)
        for _ in range(steps):
            idx = torch.randint(0, n, (min(n, 64),), generator=g)
            out = self.heads(z[idx], self.e_q.expand(len(idx), 1, DM), meta.kind_ids[idx], meta.horizon_s[idx],
                             exposure=meta.exposure[idx])
            loss, _ = world_nll(out, type(tgt)(tgt.y[idx], tgt.observed[idx], tgt.censored[idx]), normaliser=norm)
            opt.zero_grad()
            loss.backward()
            opt.step()


def test_predict_register_act_resolve_learn_and_recalibrate(tmp_path, gen, measure):
    lab = Lab(tmp_path)
    n_per_round = 90

    def round_of_decisions(tag):
        ids = []
        for t in range(n_per_round):
            cands = lab.candidates(f"{tag}{t}")
            assert len(cands) == 18 and all(lab.world.validate_action(c) == [] for c in cands)
            mask = torch.ones(1, len(cands), dtype=torch.bool)
            sel = P.select(P.RULE_POLICY, a_mask=mask, logits=torch.zeros(1, len(cands)), generator=gen)
            assert abs(float(sel.propensity) - 1 / 18) < 1e-12
            rec, _ = lab.predict_register_execute(cands[int(sel.index)])
            ids.append(rec.prediction_id)
        return ids

    first = round_of_decisions("a")
    # nothing has matured: the environment says pending, the registry refuses to resolve, nothing is trainable
    assert len(lab.preds.pending()) == n_per_round and lab.preds.scoreable() == []
    pid, query, _, _ = lab.pending[-1]                                 # the most recent decision: not yet due
    (obs,) = lab.world.measure([query])
    assert obs.status == OutcomeStatus.PENDING
    with pytest.raises(RegistryError):
        lab.preds.resolve(pid, obs)                                    # pending stays pending
    _, tgt = build_targets([[(query, obs)]])
    assert not bool(tgt.mask(torch.tensor([[K_COUNT]])).any())         # and is never a training target
    early = lab.mature()                                               # older decisions may already be due
    assert 0 < len(early) < n_per_round and len(lab.preds.pending()) == n_per_round - len(early)
    # the remaining delayed outcomes arrive
    lab.clock.advance(lab.world.cfg.outcome_horizon_s + 3600.0)
    matured = early + lab.mature()
    assert len(matured) == n_per_round and lab.preds.pending() == []
    assert all(o.status == OutcomeStatus.OBSERVED for _, o, _, _ in matured)
    lab.train(matured)
    # second round: predictions are again registered before acting, now from the updated model
    second = round_of_decisions("b")
    lab.clock.advance(lab.world.cfg.outcome_horizon_s + 3600.0)
    matured2 = lab.mature()
    nll1 = float(np.mean([lab.preds.get(p).nll for p in first]))
    nll2 = float(np.mean([lab.preds.get(p).nll for p in second]))
    measure("labworld_count_nll_before_learning", nll1)
    measure("labworld_count_nll_after_learning", nll2)
    assert nll2 < nll1 - 0.15, (nll1, nll2)
    # the registered predictions track the hidden mechanism: predicted mean vs true expected count within horizon
    truth, predicted = [], []
    for (query, obs, _, action), pid in zip(matured2, second):
        ch, effort = action.arguments["channel"], action.arguments["effort"]
        frac = 1.0 - math.exp(-lab.world.cfg.outcome_horizon_s / lab.world._delay[ch])
        truth.append(lab.world._rate(ch) * effort * frac)
        predicted.append(lab.preds.get(pid).dist.mean)
    from scipy.stats import poisson
    oracle = float(np.mean([-poisson.logpmf(o.value, m) for (_, o, _, _), m in zip(matured2, truth)]))
    measure("labworld_count_nll_oracle", oracle)                       # knowing the hidden rates and delays
    assert nll2 < oracle + 0.25, (nll2, oracle)
    corr = float(np.corrcoef(truth, predicted)[0, 1])
    measure("labworld_predicted_vs_true_mean_corr", corr)
    assert corr > 0.8
    # prospective scoring of everything, grouped; the 90 % interval of the learned predictions covers
    report = lab.preds.score(family_of=lambda r: "lab/base")
    assert report["n_scored"] == 2 * n_per_round and report["refused"] == 0 and report["skipped"] == 0
    later = [lab.preds.get(p) for p in second]
    from cam_live.predict.calibration import score_predictions
    cov = score_predictions(later, family_of=lambda r: "lab/base")["coverage"]["neg_binomial|lab/base|<=1w"]
    measure("labworld_cover90_after_learning", cov["cover90"])
    assert cov["n"] == n_per_round and cov["cover90"] > 0.75
    assert lab.preds.verify_chain() == 4 * n_per_round


def test_registered_randomised_experiment_recovers_the_true_difference(tmp_path, gen, measure):
    lab = Lab(tmp_path)
    rates = [lab.world._rate(c) for c in range(lab.world.cfg.n_channels)]       # ground truth: test-side only
    hi, lo = int(np.argmax(rates)), int(np.argmin(rates))
    arms = [f"channel-{hi}", f"channel-{lo}"]
    probe = lab.candidates("probe")
    arm_action = {arms[0]: next(c for c in probe if c.arguments == {"channel": hi, "effort": 1}),
                  arms[1]: next(c for c in probe if c.arguments == {"channel": lo, "effort": 1})}
    # hypothesis and predictions first
    exp_pred_query = OutcomeQuery(new_id("q"), "experiment.difference", kind=OutcomeKind.REAL, horizon_s=3 * DAY,
                                  issued_at=lab.clock.now())
    from cam_live.predict import dists as D
    rec = lab.exps.register(
        question="Do the two channels differ in adopters per unit effort within the attribution window?",
        hypothesis=f"{arms[0]} yields more adopters than {arms[1]}",
        predictions=[PredictionRecord(new_id("pred"), exp_pred_query, D.Normal(0.0, 3.0).to_record(), lab.clock.now())],
        arms=arms, assignment="randomized", intervention="lab.act effort 1", comparison=arms[1], horizon_s=3 * DAY,
        experiment_id="exp_lab_channels")                             # fixed id: the assignment draw is reproducible
    eid = rec.experiment_id
    units = []
    for t in range(60):
        a = lab.exps.assign(eid, f"trial{t}")
        cands = lab.candidates(f"x{t}")
        chosen = next(c for c in cands if c.arguments == arm_action[a.arm].arguments)
        # the same assignment expressed as a selection over the candidate set: non-arms have probability zero
        arm_mask = torch.tensor([[c.arguments in (arm_action[arms[0]].arguments, arm_action[arms[1]].arguments)
                                  for c in cands]])
        probs = P.selection_probs(P.RULE_RANDOMIZED, a_mask=torch.ones_like(arm_mask), arms=arm_mask, eps_floor=0.0)
        assert float(probs[0, cands.index(chosen)]) == a.probability == 0.5 and float(probs.sum()) == 1.0
        chosen.experiment_id = eid
        prec, _ = lab.predict_register_execute(chosen, experiment_id=eid)
        lab.exps.link_decision(eid, f"dec{t}")
        units.append((a.arm, prec.prediction_id))
    assert lab.exps.mature(eid) is False
    lab.clock.advance(3 * DAY + 1.0)
    arm_of_query = {lab.preds.get(pid).query.query_id: arm for arm, pid in units}
    for query, obs, _, action in lab.mature():
        lab.exps.add_outcome(eid, obs, arm=arm_of_query[query.query_id])
    assert lab.exps.mature(eid) is True
    res = lab.exps.analyse(eid)
    frac = [1.0 - math.exp(-lab.world.cfg.outcome_horizon_s / lab.world._delay[c]) for c in (hi, lo)]
    true_diff = rates[hi] * frac[0] - rates[lo] * frac[1]
    measure("labworld_experiment_true_difference", true_diff)
    measure("labworld_experiment_estimated_difference", res["diff"])
    measure("labworld_experiment_ci", [res["lo"], res["hi"]])
    assert res["valid"] and res["randomized"] and res["lo"] <= true_diff <= res["hi"]
    supported = bool(res["lo"] > 0)
    closed = lab.exps.conclude(eid, supported=supported, test=res,
                               alternative_explanations=["arrival delays differ between channels, so a fixed "
                                                         "window favours the faster one"],
                               measurement_uncertainty=f"Welch 95% CI [{res['lo']:.2f}, {res['hi']:.2f}]")
    assert closed.status == "closed" and closed.supported is (true_diff > 0) and len(closed.actual) == 60
    assert len(closed.decision_ids) == 60 and len(lab.exps.meta(eid)["assignments"]) == 60
    # the experiment-level prediction is resolved by the measured difference, after the fact, exactly once
    lab.clock.advance(1.0)
    from cam_live.contracts import OutcomeObservation
    done = lab.preds.resolve(closed.predicted[0].prediction_id,
                             OutcomeObservation(exp_pred_query.query_id, OutcomeStatus.OBSERVED, res["diff"],
                                                observed_at=lab.clock.now(), source="experiment analysis"))
    assert done.nll is not None and lab.exps.get(eid).predicted[0].resolved_value == res["diff"]
