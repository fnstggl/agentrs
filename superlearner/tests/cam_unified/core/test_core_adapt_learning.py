"""Fast-adaptation learning with the DOM belief composition (port of the audit's exp_adapt.py; plan 2026-11-26 §3 W6
"with the Kalman layer on, Gaussian mechanism NLL(k=2) <= oracle + 0.05 and z-sd in [0.8, 1.2]").

The real BeliefModule, OutcomeHead and BayesLastLayer(fusion=True) are trained with the runtime's support/query
structure of `adapt_loss_all_k`: support steps update the particles (the given action-conditioned features, i.e. the
world-model likelihood path) and the Kalman/NIG layer (residuals against the particle-free head mean); queries are
scored by the gated fusion (`FusedPredictive.joint_log_prob`), with the Kalman layer dropped in half of the updates
and the mixture-only auxiliary term, plus a world-loss-like unconditioned head NLL. The mechanism world: two hidden
mechanisms mu = +-1 (Gaussian outcome, sigma 0.25), K=8 particles, an exact Bayesian oracle over the two mechanisms.

Marked slow (CPU minutes); `CAM_RUN_SLOW=1` runs it. `run_adapt_experiment` is also used to write the W6 evidence.
"""
import math
import os

import pytest
import torch

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.core.bayes import BayesLastLayer
from superlearner.cam_unified.core.belief_runtime import fused_predictive_modules, kalman_base_modules
from superlearner.cam_unified.core.beliefs import BeliefModule
from superlearner.cam_unified.interfaces import Likelihood, OutcomeBatch, TokenKind, TokenSet
from superlearner.cam_unified.world.outcomes import OutcomeHead

KS = (0, 1, 2, 4, 8, 16)
NQ = 4
SIG = 0.25


def _ob(y, B, D, q_tok, observed=True):
    dev = q_tok.device
    q = TokenSet(q_tok.expand(B, 1, D).clone(), torch.ones(B, 1, dtype=torch.bool, device=dev),
                 torch.full((B, 1), int(TokenKind.QUERY), device=dev), torch.zeros(B, 1, device=dev),
                 torch.full((B, 1), -1, device=dev), None)
    return OutcomeBatch(q, torch.full((B, 1), int(Likelihood.GAUSSIAN), device=dev),
                        torch.zeros(B, 1, dtype=torch.long, device=dev), y.view(B, 1).float(),
                        torch.zeros(B, 1, dtype=torch.bool, device=dev),
                        torch.full((B, 1), observed, dtype=torch.bool, device=dev),
                        torch.zeros(B, 1, dtype=torch.long, device=dev), [["y"] for _ in range(B)])


def run_adapt_experiment(seed: int = 0, steps: int = 2500, B: int = 64, d_model: int = 64, K: int = 8, Dp: int = 32,
                         bayes_feat: int = 8, lr: float = 1e-3, kalman_dropout: float = 0.5, mix_aux: float = 1.0,
                         n_eval: int = 512, device=None) -> dict:
    dev = torch.device(device or "cpu")
    torch.manual_seed(seed)
    cfg = make_config("dev", n_layers=2, d_model=d_model, n_heads=4, n_state=8, d_particle=Dp, n_particles=K,
                      bayes_feat=bayes_feat, stoch_groups=8, stoch_classes=8, mem_key_dim=32, reason_layers=1,
                      max_ctx_tokens=96, n_prefix=4, max_entities=32, n_options=0, preset="dom_adapt_test")
    D, S, A = cfg.d_model, cfg.n_state, 4
    bm = BeliefModule(cfg, n_channels=4).to(dev)
    head = OutcomeHead(cfg).to(dev)                         # DOM film init 1/sqrt(Dp)
    bayes = BayesLastLayer(cfg).to(dev)                     # fusion on (dom preset)
    assert bayes.fusion
    params = list(bm.parameters()) + list(head.parameters()) + list(bayes.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    g0 = torch.Generator().manual_seed(1234)
    Z0 = torch.randn(1, S, D, generator=g0).to(dev)
    E_act = torch.randn(A, D, generator=g0).to(dev)
    q_tok = torch.randn(1, 1, D, generator=g0).to(dev)
    MECH = torch.tensor([[-1.0] * A, [1.0] * A], device=dev)

    def Za(x):
        return Z0 + E_act[x].unsqueeze(1)                   # action-conditioned latent (stands in for WM.prior)

    def support(fast, x, y, use_k):
        Bn = x.shape[0]
        ob = _ob(y, Bn, D, q_tok)
        feat = head.features(Za(x), ob)
        fast_n, _ = bm.update(fast, Z0.expand(Bn, S, D), None, ob, head, feat=feat)
        if use_k:
            base0 = kalman_base_modules(head, feat, ob)
            r = OutcomeBatch(ob.q, ob.lik, ob.n_classes, ob.y - base0, ob.censored, ob.y_mask, ob.channel, ob.keys)
            fast_n = bayes.update(fast_n, bayes.features(feat, r), r)
        return fast_n

    def query(fast, x, y, use_k):
        Bn = x.shape[0]
        ob = _ob(y, Bn, D, q_tok)
        feat = head.features(Za(x), ob)
        fp = fused_predictive_modules(head, bayes if use_k else None, fast, feat, ob, kalman=use_k)
        return fp, ob

    def episode(Bn, k, gen, use_k, train=True):
        mid = torch.randint(0, 2, (Bn,), generator=gen).to(dev)
        xs = torch.randint(0, A, (Bn, k + NQ), generator=gen).to(dev)
        ys = MECH[mid].gather(1, xs) + SIG * torch.randn(Bn, k + NQ, generator=gen).to(dev)
        fast = bm.init_fast(Bn, device=dev)
        for t in range(k):
            fast = support(fast, xs[:, t], ys[:, t], use_k)
        res = [query(fast, xs[:, k + j], ys[:, k + j], use_k) for j in range(NQ)]
        world = None
        if train:
            ob = _ob(ys.flatten(), Bn * (k + NQ), D, q_tok)
            world = head.dist(head.features(Za(xs.flatten()), ob), ob).log_prob(ob.y).mean()
        return res, world, xs, ys

    gtr = torch.Generator().manual_seed(seed)
    for step in range(steps):
        k = KS[int(torch.randint(len(KS), (1,), generator=gtr))]
        use_k = float(torch.rand((), generator=gtr)) >= kalman_dropout
        res, world, _, _ = episode(B, k, gtr, use_k)
        lp = torch.stack([fp.joint_log_prob(ob) for fp, ob in res], 1).mean()
        loss = -lp - world
        if mix_aux > 0:
            # mixture-only query term (particles keep their own learning signal on continuous outcomes)
            lpm = torch.stack([torch.logsumexp(torch.log_softmax(fp.logw.float(), -1)
                                               + fp.particle_log_prob(ob).sum(-1), -1) for fp, ob in res], 1).mean()
            loss = loss - mix_aux * lpm
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()

    @torch.no_grad()
    def evaluate():
        ge = torch.Generator().manual_seed(99_000 + seed)
        out = {}
        for k in KS:
            res, _, xs, ys = episode(n_eval, k, ge, True, train=False)
            lp = torch.stack([fp.joint_log_prob(ob) for fp, ob in res], 1)
            lpm = torch.stack([torch.logsumexp(torch.log_softmax(fp.logw.float(), -1)
                                               + fp.particle_log_prob(ob).sum(-1), -1) for fp, ob in res], 1)
            mu = torch.stack([fp.mean()[:, 0] for fp, _ in res], 1)
            var = torch.stack([fp.var()[:, 0] for fp, _ in res], 1).clamp_min(1e-8)
            qy = ys[:, k:]
            z = (qy - mu) / var.sqrt()
            # exact Bayes over the two mechanisms (uniform prior)
            logpost = torch.full((n_eval, 2), -math.log(2), device=dev)
            for j in range(k):
                logpost = logpost - 0.5 * (ys[:, j:j + 1] - MECH[:, 0].view(1, 2)) ** 2 / SIG ** 2
            logpost = logpost - torch.logsumexp(logpost, -1, keepdim=True)
            orc = torch.logsumexp(logpost.unsqueeze(1) - 0.5 * math.log(2 * math.pi * SIG ** 2)
                                  - 0.5 * (qy.unsqueeze(-1) - MECH[:, 0].view(1, 1, 2)) ** 2 / SIG ** 2, -1)
            gate = torch.stack([fp.gate[:, 0] for fp, _ in res if fp.gate is not None]) if res[0][0].gate is not None \
                else None
            out[k] = {"nll_full": float(-lp.mean()), "nll_mix": float(-lpm.mean()), "nll_oracle": float(-orc.mean()),
                      "z_sd": float(z.std()), "cover90": float((z.abs() < 1.645).float().mean()),
                      "gate": None if gate is None else float(gate.mean())}
        return out

    return {"seed": seed, "steps": steps, "familiar": evaluate()}


SLOW = os.environ.get("CAM_RUN_SLOW") == "1"


@pytest.mark.slow
@pytest.mark.skipif(not SLOW, reason="slow learning test: set CAM_RUN_SLOW=1 to run")
def test_gated_fusion_reaches_the_oracle_with_calibrated_predictions():
    for seed in (0, 1, 2):
        # 5000 steps: at 2500 seed 2 stayed at NLL(k=2) 0.129 vs oracle 0.047 (particles not yet separated); at 5000
        # every seed meets the criterion (docs/evidence/dom/W6/adapt_learning_gated_fusion.json "runs_5000")
        r = run_adapt_experiment(seed=seed, steps=5000)["familiar"]
        print(seed, {k: {n: round(v, 3) for n, v in row.items() if v is not None} for k, row in r.items()})
        assert r[2]["nll_full"] <= r[2]["nll_oracle"] + 0.05, (seed, r[2])
        assert 0.8 <= r[2]["z_sd"] <= 1.2, (seed, r[2])
        assert r[16]["nll_full"] <= r[0]["nll_full"]                 # adaptation helps
