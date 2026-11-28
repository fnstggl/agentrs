# CAM Live — mathematical implementation notes (DL-04)

Describes the code at commit `fdab7d3` (read 2026-11-28, 07:15–07:35 UTC). No later commit touches `cam_live/src`,
`tools`, `configs`, `tasks` or `pyproject.toml` (`git log fdab7d3..HEAD -- ...` is empty). The mathematical code was
last changed in these commits: `nn/fast.py` 9680ea7 (G2L-1), `predict/` ef0f3ff (G2D-1, `predict/errors.py`),
`reward/` 4476c27 (KF-E01 / KF-E02 / KF-E05 / LRN-3), `credit/` 6159def (DX round 2), `agent.py` 257267c. So the
verifier snapshots still describe this code: G2-decision round 1 (`f54d8cc`) for `reward/`, G2-learn round 2
(`f54d8cc`) for `nn/fast.py`, G2-learn round 3 (`cf46eb1`) for `credit/`, and G2-decision round 2 (`ef0f3ff`) for
`predict/errors.py`.

Every equation is followed by the code that computes it, as `file:function` or `file:Class.method`. Paths are relative
to `cam_live/src/cam_live/`. Where the design documents (`docs/CAM_LIVE_ARCHITECTURE.md`, `docs/A/DESIGN.md`,
`docs/B|C|D/API.md`, `docs/C/LRN_S5.md`) or a docstring say something else, this note describes the code and lists
the difference in §14. Defaults are the values in force with `configs/cam_live.yaml`. That file has `nn`
(`max_prompt_tokens` only), `storage` and `runtime` sections, but no `predict`, `reward`, `credit` or `agent`
section, so the dataclass defaults apply there.

Five formulas were re-evaluated on the CPU at `fdab7d3` with
`cd cam_live && CUDA_VISIBLE_DEVICES= ../.venv/bin/python -I docs/evidence/DOC/scripts/math_check.py`.
The output is identical, line for line, to the runs at `55a0f4c` and `5ae3b59` (`docs/evidence/DOC/math_check.txt`):
* the NIG predictive variance of §2.2: 0.585551 by `FastBeliefState.predict` and 0.585551 by the closed form;
* the prior of the selection calibration (§6.3): mean of the (beta_q, beta_pi) draws (1.002, 0.0006), mean lambda
  0.357, lambda > 0 in 49.9 % of draws;
* the exploration floor (§6.1): a one-hot vector becomes 0.985 / 0.005 / 0.005 / 0.005;
* the return discount (§8.1): beta = 8.0225e-6 s^-1 = ln 2 / 86400;
* the scale invariance of the batch standardisation (§7.3): 1.2e-12.

One further fact was measured for this note: the overflow of `FastPrior` at large prior variance (§13). Command:
`cd cam_live && ../.venv/bin/python -I -c "from cam_live.nn.fast import FastPrior; FastPrior(4, prior_var=v)"` for
v = 1, 1e5, 5e5 (constructs) and 1e6 (`OverflowError: math range error`). Everything else comes from reading the
code or from the cited verification reports. Nothing here is a new measurement of learning.

## Notation

| symbol | meaning | where it is formed |
|---|---|---|
| `H_t` | stored context items of decision t (experience tokens) | `learning_api.py:encode_decisions` |
| `b` | belief, output of the experiential core on `H_t` | `nn/model.py:CamLiveModel.core_forward` |
| `e_a`, `e_g`, `e_q` | embeddings of a candidate action, the goal text, an outcome query | `nn/model.py:CamLiveModel.encode_actions / CamLiveModel.encode_goals / CamLiveModel.encode_queries` |
| `z` | decision latent `z = trunk(b, e_a, e_g)` | `nn/model.py:CamLiveModel.decision_latent` |
| `phi` | fast-layer features `phi = Phi(z, e_q)`, F = 128 | `nn/decision.py:FeatureMap.forward` |
| `w` | last-layer weights of the fast layer, one vector per outcome kind | `nn/fast.py:FastBeliefState` |
| `theta` | adaptable subset: policy head + decision trunk | `reward/inner.py:AdaptableSubset` |
| `psi` | parameters of the intrinsic reward `R_psi` (group `reward`) | `reward/intrinsic.py:IntrinsicReward.psi` |
| `mu(a)` | logged propensity of the executed candidate | `contracts.py:Decision` (`probabilities`) |
| `U` | verified utility (from signed verifier records only) | `reward/integrity.py:objective_utility / measurement_utility` |

`phi` has unit L2 norm and a constant last coordinate: `raw = RMSnorm(out(...))` over F - 1 entries,
`phi = [raw / sqrt(F), 1 / sqrt(F)]`, so `||phi||^2 = (F - 1)/F + 1/F = 1` (`nn/decision.py:FeatureMap.forward`).

---

## 1. Parameters that set the numbers below

| quantity | default | location |
|---|---|---|
| fast forgetting `gamma`, CUSUM `k`, `h`, clip, change-forget | 0.999, 1.0, 12.0, 9.0, 0.1 | `nn/presets.py:NNConfig`, `nn/fast.py:FastBeliefState.__init__` |
| fast prior variance, noise, `a0`, noise mode | 1.0, 0.5, 2.0, `nig` | `nn/presets.py:NNConfig`, `nn/fast.py:FastPrior` |
| ensemble members / hazard bins (50m core) | 5 / 12 | `predict/heads.py:PredictConfig` |
| Thompson samples `S`, floor `eps` | 32, 0.02 | `predict/heads.py:PredictConfig` |
| VOI samples, candidates, slow path | 8, 16, off | `predict/heads.py:PredictConfig` |
| cost weights (money, seconds) in values | 0.0, 0.0 | `predict/heads.py:PredictConfig.cost_weight / PredictConfig.time_weight` |
| selection rule; goal-free rule | `thompson_policy`; `voi` | `agent.py:AgentSettings.selection_rule / AgentSettings.goal_free_rule` |
| inner step `alpha`, steps `K` | 1e-2, 2 | `configs/cam_live.yaml` `optim`, `reward/metalearner.py:MetaConfig` |
| live trust region: target KL, step range | 0.05, [1e-2, 2.0] | `reward/metalearner.py:MetaConfig` |
| reward lr (Adam), meta-gradient clip | 1e-4, 0.5 | `config.py:OptimConfig` |
| independent groups per meta-update | 8 | `configs/cam_live.yaml` `optim.meta_min_independent_outcomes` |
| ESS floor, weight clip, propensity floor | max(2, 0.05 n), 10, 1e-6 | `reward/metalearner.py:MetaConfig` |
| held-out fraction of groups (mix promotion) | 0.25 | `reward/metalearner.py:MetaConfig.heldout_fraction` |
| bootstrap mix: start, step, min groups, margin | 1.0, 0.25, 4, 0.05 x RMS(U) | `reward/metalearner.py:MetaConfig` |
| return half-life | 86 400 s | `reward/inner.py:InnerConfig.half_life_s` |
| credit discount `beta` | 0 | `agent.py:AgentSettings.credit_beta` |
| live policy update cadence / batch | every 4 rewarded decisions / 16 newest (+ up to 16 replayed) | `agent.py:AgentSettings`, `reward/metalearner.py:MetaConfig.replay_matured` |
| learning rates (experiential, heads, adapters), clips | 3e-4, 3e-4, 5e-5; 1.0, 1.0, 0.5 | `config.py:OptimConfig` |
| frequent trigger, max share, warm-up | 4 independent matured units, 0.9, 8 steps | `credit/learner.py:LearnerSettings` |
| slow step: every, batch, graph items | 8 frequent updates, 4 records, newest 8 context items | `credit/learner.py:LearnerSettings` |
| slow guard: absolute cap, relative cap, z, chunks, floor, drift | 0.01, 0.05, 2, 8, 1e-4 \|L\|, max(0.02, 0.02 \|L\|) | `credit/learner.py:LearnerSettings` |
| slow trust region: back-off, minimum, recovery | x0.5, 1/16, x1.5 | `credit/learner.py:LearnerSettings` |
| held-out split in the agent: kind, fraction | stratified by record, 0.15; experiment lifetimes not held out | `agent.py:CamLiveAgent.__init__`, `credit/learner.py:ContinualLearner._record_held_out` |
| checkpoint gates (validation drop, retention drop) | 0.5, 0.1 | `agent.py:AgentSettings.promotion_max_validation_drop / AgentSettings.promotion_max_retention_drop` |
| promotion cadence | every 24 frequent updates and at each objective end | `agent.py:AgentSettings.promote_every` |

---

## 2. Fast Bayesian beliefs (`m_t`)

There is one posterior `w ~ N(m, Sigma)` per outcome kind (real, count, binary, time-to-event). Categorical outcomes
are not in the fast layer. Everything is float64. The state is created from the learned prior by
`nn/model.py:CamLiveModel.new_fast_state`. It is updated without any gradient step by
`predict/predictor.py:Predictor.observe_outcomes` → `nn/fast.py:FastBeliefState.update`. At a new objective the agent
returns it to the current learned prior (`credit/learner.py:ContinualLearner.reset_fast`, called by
`agent.py:CamLiveAgent.set_objective`).

### 2.1 Real outcome, known noise (Gaussian, exact)

With `s = phi^T Sigma phi`, `S = s + sigma^2`, gain `k = Sigma phi / S`, innovation `e = y - phi^T m`:

    m'     = m + k e
    Sigma' = (I - k phi^T) Sigma (I - k phi^T)^T + sigma^2 k k^T          (Joseph form, symmetrised)
    surprise = 1/2 [ log(2 pi S) + e^2 / S ]
    KL(post || prior) = 1/2 [ -s/S + (e^2/S)(s/S) + log(S / sigma^2) ]

`nn/fast.py:gaussian_update` (with `_joseph`, `_quad`). The batch form uses precision and Cholesky factors,
`Lambda = Lambda0 + Phi^T W Phi / sigma^2`, `m = Lambda^-1 (Lambda0 m0 + Phi^T W y / sigma^2)` (`nn/fast.py:gaussian_fit`,
`_prec_from_cov`). It equals the recursion.

### 2.2 Real outcome, inferred noise (normal-inverse-gamma, the default `noise_mode="nig"`)

`w | s2 ~ N(m, s2 V)`, `s2 ~ InvGamma(a, b)`. With `q = phi^T V phi`:

    k = V phi / (1 + q),  m' = m + k e,  V' = Joseph(V, k) + k k^T,
    a' = a + 1/2,         b' = b + e^2 / (2 (1 + q))
    predictive: Student-t(loc = phi^T m, scale^2 = (b/a)(1 + q), dof = 2a)

`nn/fast.py:nig_update`, `nn/fast.py:predict_nig`; the KL is in closed form in `nig_update`. The reported moments
(`nn/fast.py:FastBeliefState.predict`) are:
* mean `phi^T m`;
* epistemic `E[s2] q`;
* noise `E[s2]`, with `E[s2] = b / (a - 1)` (`nn/fast.py:_nig_noise_mean`, infinite for `a <= 1`).

So epistemic + noise is the Student-t *variance* `(b/(a-1))(1+q)` (re-evaluated above). The linear predictor's
variance is likewise `(b/(a-1)) phi^T V phi` (`nn/fast.py:FastBeliefState.linear`). `FastBeliefState.posterior`
reports the plug-in `b / a` as `noise` (the Student-t scale). Thompson draws sample `s2` jointly:
`w = m + sqrt(s2) L z`, `s2 = b / Gamma(a)` (`nn/fast.py:FastBeliefState.sample_weights`, Cholesky with relative
jitter 1e-12).

### 2.3 Count, binary, time-to-event (assumed-density / Laplace update)

Linear predictor `eta = phi^T w`, prior on it `N(eta0, s)`, `eta0 = phi^T m`, `s = phi^T Sigma phi`.
Likelihoods (`nn/fast.py:_glm_terms`, `_glm_loglik`):

    count          c ~ Poisson(e exp(eta))                    e = exposure
    binary         c ~ Binomial(n, sigmoid(eta))              n = trials (exposure argument)
    time-to-event  delta eta - (t/u) exp(eta)                 Poisson-exposure identity: delta = 1 - censored,
                                                              exposure t/u (u = time unit); right-censoring exact

The posterior mode lies on the line `m + Sigma phi c`. The update therefore solves a scalar, strictly concave
problem:

    eta* = argmax  loglik(eta) / d  -  (eta - eta0)^2 / (2 s)          d = count dispersion (1 for binary / tte)

**Poisson family** (count and time-to-event; `nn/fast.py:_mode_poisson`). With `k = d / s`, find the root of the
concave, decreasing `r(x) = c - e exp(x) - k (x - eta0)`. Its log form `G(x) = log(e exp(x)) - log(c + k (eta0 - x))`
has the same root, is convex and increasing, and is defined where `L = c + k (eta0 - x) > 0`. Each iteration takes
the Newton step on `r` and the Newton step on `G` and keeps the smaller of the two, from either side of the root.
The G-step is used only where `L > 0`, `e exp(x) > 0` and the G-step is finite. The iterates decrease monotonically
to the root (fix of G2L-1; before 9680ea7 the G-step was taken only from the right).

**Binomial** (`nn/fast.py:_mode_binomial`): bracketed Newton with the two tail forms of the score.

Both searches stop when `|step| <= 1e-11 (1 + |eta|)` or after at most 100 iterations (`nn/fast.py:_laplace_1d`;
constants `NEWTON_TOL`, `NEWTON_MAX_ITER`). For the Poisson family, convergence is reported only if, in addition,
the root condition holds at the returned point:

    |r(x)| <= 1e-8 (c + e exp(x) + k |x - eta0|) + 8 eps_64 k (|x| + |eta0| + 1)    or    L > 0 and |G(x)| <= 1e-8

One further Newton step is taken with the graph, so `eta*` carries the implicit-function gradient. With curvature
`h = -loglik''(eta*) / d` and `delta = eta* - eta0`:

    m'     = m + Sigma phi (delta / s)
    Sigma' = Joseph(Sigma, k = Sigma phi h / (1 + h s)) + (h / (1 + h s)^2) (Sigma phi)(Sigma phi)^T
    KL     = 1/2 [ -h s / (1 + h s) + delta^2 / s + log(1 + h s) ]

`nn/fast.py:glm_update`. The surprise is the predictive NLL of the *untempered* likelihood,
`-log ∫ p(y | eta) N(eta; eta0, s) d eta`. It is computed by 24-point Gauss–Hermite quadrature centred on the Laplace
approximation `N(eta*, s / (1 + h s))` of the integrand (`nn/fast.py:_adaptive_gh_nll`, `nll_glm`).

Predictive moments (`nn/fast.py:predict_glm`):
* count: mean `e exp(eta0 + s/2)`, epistemic `mean^2 (exp(s) - 1)`, aleatoric `mean` (times the dispersion,
  `FastBeliefState.predict`);
* time-to-event: mean `u exp(-eta0 + s/2)`;
* binary: mean and variance by Gauss–Hermite quadrature over `eta`.

Impossible observations raise before anything is absorbed (`nn/fast.py:check_observations`): a negative count or
time, successes outside `[0, n]`, a positive count at zero exposure, a censoring flag other than 0/1, a non-finite
value. The change-detection residual uses predictive moments: `z2 = (c - E c)^2 / Var c`, with the probit
approximation `sigmoid(eta0 / sqrt(1 + pi s / 8))` for binary.

How the predictor feeds the fast layer (`predict/predictor.py:Predictor.observe_outcomes`):
* a count is absorbed with exposure = the head's predicted mean count, so the fast layer learns a multiplicative
  correction of the head;
* a real value is absorbed as the residual `(y - loc_metric)/scale_metric - head_loc` (normalised units);
* a time to event is absorbed as `y / head_time` (a proportional correction);
* a binary outcome is absorbed as the raw 0/1 value (no head offset, see §13).

### 2.4 Changing mechanisms

Stabilised forgetting before every update (when `gamma < 1` and the state is not fresh):

    Lambda <- gamma Lambda + (1 - gamma) Lambda0,    Lambda m <- gamma Lambda m + (1 - gamma) Lambda0 m0
    (NIG: a <- gamma a + (1 - gamma) a0,  b likewise)

`nn/fast.py:forget`, `FastBeliefState._flatten`. It is the geometric mixture `p^gamma prior^(1-gamma)`, bounded by the
prior in every direction. A CUSUM runs on the standardised surprise of the pre-update belief:

    C <- max(0, C + min(z2, 9) - 1 - k);   C > h  =>  forget once with gamma = 0.1, reset C, re-apply the datum

`nn/fast.py:FastBeliefState.update`. Defaults `gamma = 0.999`, `k = 1`, `h = 12`.

### 2.5 Learned prior (ALPaCA-style)

`FastPrior` holds per kind (`nn/fast.py:FastPrior.prior`; group `experiential`):
* `m0` and a Cholesky factor `L0` (softplus diagonal floored at 1e-6);
* the log noise variance;
* `a0 = 1 + softplus(raw)`;
* the count dispersion `d = exp(disp_raw)` (exactly 1 at initialisation).

`prior()` returns `cov = L0 L0^T`, `V = cov / sigma^2`, `b0 = a0 sigma^2`. The prior is trained by the prequential
(predict-then-update) NLL of a sequence:

    L_fast = sum_t -log p(y_t | posterior after y_<t)          (masked rows contribute 0; differentiable in phi and prior)

`nn/fast.py:online_nll`. Two providers use it:
* `predict/providers.py:FastPriorLoss` (weight 0.5): one sequence per (objective, kind), rows ordered by observation
  time, at most 48 per sequence, each kind's term divided by a running scale (`_balanced`);
* the prior-predictive term inside `predict/providers.py:WorldModelLoss.fast_term` (weight `fast_prior_weight` 0.25,
  sequences of length 1).

The fast-layer targets are the head-offset inputs of §2.3 (`predict/heads.py:fast_inputs`, `head_offsets`).

---

## 3. Prediction distributions

### 3.1 World-model heads

`M = 5` members. Each is `raw = member_m(x) + prior_m(u)`, with a frozen randomised prior network (Osband et al. 2018;
`prior_scale = 1`), and
`x = SiLU(LN(W_z z + W_q e_q + (W_zg z)(W_qg e_q)/sqrt(h) + W_t tau(horizon) + E_kind + W_phi phi))`
(`predict/world.py:WorldHeads.forward`). Per kind (`predict/world.py:WorldOutput`):

| kind | distribution | parameters |
|---|---|---|
| real | Student-t (or normal) in normalised metric units | `loc`, `scale = softplus(.+0.5413)+1e-3`, `df = 2 + softplus(.+3)` |
| count | negative binomial, mean `exp(log_rate) * exposure`, concentration `r` | `log_rate` clamped ±20, `log r` clamped [-6, 9] |
| binary | Bernoulli | logit |
| time-to-event | discrete hazard on 12 equal bins of `[0, horizon]`, right-censoring | `h_k = sigmoid(logit_k - 2)` |
| categorical | masked softmax over at most 16 classes | logits |

Hazard likelihood (`predict/dists.py:hazard_log_prob`):
* event in bin k: `log h_k + sum_{j<k} log(1 - h_j)`;
* censored at t: the sum over the bins completed by t of `log(1 - h_j)`;
* restricted mean `E[min(T, H)]` with events uniform in their bin (`predict/dists.py:hazard_expected_time`).

The negative binomial is evaluated in log space (`predict/dists.py:neg_binomial_log_prob_from_log`).

Training (`predict/world.py:world_nll`): masked NLL of the *observed or censored* outcomes of the executed candidate
only. Pending or unobservable targets give exactly zero loss and gradient. Member m is weighted by a deterministic
Poisson(1) bootstrap multiplicity of the record key (`predict/world.py:bootstrap_weights`). The loss is averaged per
kind and divided by `max(1, EMA|NLL_kind|)` (`predict/world.py:KindNormaliser`, momentum 0.95). Targets are float64
(`predict/world.py:build_targets`).

Ensemble summary (`predict/world.py:WorldOutput.predictive`): mean = average of the member means; **epistemic** =
variance of the member means (unbiased); **aleatoric** = mean of the member variances (NB: `mu + mu^2 / r`;
Student-t: `scale^2 df/(df-2)`).

### 3.2 Combination with the fast layer (the prediction the agent registers)

`predict/predictor.py:Predictor._means`, per query (head members `m`, fast posterior moments `(f_m, f_v)`):

    real   mean = (mean_m(loc_m) + f_m) * scale + loc,      epi = (var_m(loc_m) + f_v) * scale^2
    count  mean = h_m * f_m,                                epi = h_v f_m^2 + h_m^2 f_v + h_v f_v      (independent factors)
    binary p = mean_m sigmoid(logit_m + shift),  shift = E_fast[eta] - E_ref[eta],
           epi = var_m(sigmoid(.)) + (p(1-p))^2 Var_fast[eta]
    tte    mean = t_m * f_m,                                epi as for counts
    categorical: head only

`ref` is the fast state at the start of the current objective (`predict/predictor.py:Predictor.set_fast_state`, called
by `set_objective`). The registered distribution (`predict/predictor.py:Predictor._distribution`) is moment-matched:
* count → negative binomial with total variance `alea + epi`;
* real → Student-t / normal with `scale^2 = mean_m scale_m^2 + epi / s^2`;
* binary → Bernoulli(p);
* time-to-event → the exact member mixture of the hazard pmf with the proportional-hazards correction
  `h' = 1 - (1 - h)^ratio`, `ratio = exp(E_fast[eta])` (clipped ±10).

Every `PredictionRecord` carries `epistemic_std` separately. It is registered in the hash-chained registry before
`evaluate_candidates` returns (`predict/registry.py:PredictionRegistry.register`).

Joint posterior samples (for Thompson sampling; `predict/predictor.py:Predictor._samples`): sample s draws one ensemble
member index and one fast-weight vector per kind, both shared by all candidates of the decision. The candidate's goal
value per sample is the sum over its goal-relevant queries of the longest horizon
(`predict/predictor.py:Predictor._goal_mask`). It is stored as `extras["ts_<s>"]`.

Goal-value uncertainty after the decision's own outcomes (LRN-3; `predict/predictor.py:Predictor.goal_epistemic_after`):
the decision-time head view of the executed candidate's goal-relevant queries is kept
(`predict/predictor.py:Predictor._keep_goal_view`, at most 512 decisions). Once every one of those queries has
absorbed its own outcome, `sqrt(sum_goal-queries epi)` is recomputed with the current fast state on that view. This is
the same quantity as the decision-time `CandidateEvaluation.uncertainty`. The views are dropped at a new objective
(`predict/predictor.py:Predictor.forget_goal_views`) and checkpointed with the agent.

### 3.3 Goal evaluator, delta V, action quality

`predict/evaluator.py:GoalEvaluator` gives `V(b, g, horizon)` and `Q(z, g, horizon)` read-outs:
* success logit;
* magnitude (negative binomial or Student-t, scaled by the objective scale);
* time-to-achievement (12-bin hazard, censoring-aware);
* resources (normal on `log1p(amount/scale)`).

Loss (`predict/evaluator.py:evaluator_loss`): `1.0 BCE(success) + 0.25 NLL(mag) + 0.25 NLL(time) +
0.25 NLL(resources)`, masked, each prefix weighted `1/(#valid prefixes)`. Targets come only from signature-checked
objective outcomes, applied to every earlier prefix (`predict/evaluator.py:prefix_targets`):
* achieved → success 1 and time as an event;
* failed → success 0 and time right-censored at the remaining horizon;
* pending → all masks false;
* prefixes formed at or after the evaluation are masked.

    Delta V = sigmoid(logit V(b', g)) - sigmoid(logit V(b, g))        (a reward-generator input, detached; not a reward)
    action quality = q_chosen - sum_i pi_i q_i,   regret = max_i q_i - q_chosen

`predict/evaluator.py:delta_v`, `action_quality`. The signals are assembled in `reward/assemble.py:SignalAssembler`.
`success_before` is the state value `V(b)` recorded at decision time, the same function as `success_after`.

### 3.4 Error attribution and its change detector

Each resolved prediction is labelled (`predict/errors.py:classify_error`, `z_noise = 2`):
* `missing_observation` if the outcome is pending or unobservable;
* `within_noise` if `|y - mean| <= 2 aleatoric_std`;
* `changed_conditions` if not, and the metric's change detector is in the detected state;
* `model_uncertain` if `|y - mean| <= 2 sqrt(aleatoric^2 + epistemic^2)`;
* `model_wrong` otherwise.

The detector (`predict/errors.py:ChangeDetector.update`, one per metric in `Predictor.observe_outcomes`) is a
two-sided CUSUM on `r = clip((y - mean) / total_std, ±6)`:

    pos <- max(0, pos + r - 0.5),   neg <- max(0, neg - r - 0.5)
    detected when max(pos, neg) > 8;   re-armed (G2D-1) when max(pos, neg) < 0.25 x 8 = 2

The labels are diagnostics only. They create no training signal and no reward.

---

## 4. Calibration metrics

Only prospective predictions are scored: `made_at < observed_at` strictly (`predict/calibration.py:prospective_guard`,
`score_predictions`).

    Brier = (p - y)^2;   NLL = -[y log p + (1-y) log(1-p)]   (p clipped to [1e-12, 1 - 1e-12])
    recalibration: y ~ sigmoid(alpha + beta logit p)  by Newton (perfect: alpha = 0, beta = 1)
    ECE = sum_b (n_b / n) |mean p_b - freq_b|  (10 equal-width bins)

`predict/calibration.py:brier`, `binary_nll`, `logistic_recalibration`, `reliability_bins`, `ece`. Intervals:
* Brier / NLL: Student-t over cluster means (`predict/calibration.py:studentised_ci`); `score_predictions` clusters by
  experiment, then decision, then prediction;
* slope / intercept: the CR1 cluster-robust sandwich (G - 1 df) when clusters are given, otherwise a Newey–West HAC
  sandwich in row order (Bartlett kernel, Andrews AR(1) bandwidth `1.1447 (alpha n)^(1/3)`;
  `predict/calibration.py:_hac_meat`);
* ECE: a cluster bootstrap (`predict/calibration.py:ece_ci`).

Reports are grouped by objective family and horizon bucket (edges 1 h, 1 d, 7 d, 30 d;
`predict/calibration.py:grouped_binary_report`).

---

## 5. Value of information

Definition used at decision time (`predict/predictor.py:Predictor._voi`). The 16 candidates with the highest total
epistemic variance are taken. Infeasible ones and those without fast-layer queries are then skipped, not replaced.
For each such candidate i, over `S_v = 8` samples:

    for s = 1..S_v:  f_s = clone(fast state)
        for each of i's queries j:  w ~ f_s posterior;  lin = phi_ij^T w;  draw y from the likelihood:
            count Poisson(hc exp(lin)), hc = head mean count;  real lin + sqrt(noise) N(0,1) (NIG: plug-in b/a);
            binary Bernoulli(sigmoid(lin));  tte -log(u) exp(-lin)
            f_s.update(phi_ij, y)                                     (the fast layer's own update, §2)
            KL_s += KL of that update
        Q_s(a') = sum_goal-queries combined mean under f_s (§3.2) - net_cost(a')   for every candidate a'
    VOI_i = max(0,  mean_s max_a' Q_s(a')  -  max_a' mean_s Q_s(a'))          (E[max] - max E)
    info_gain_nats_i = mean_s KL_s

The baseline is the maximum of the sample means, not `max_a' Q(b, a')`. For an exact Bayesian update the two agree
(the posterior mean is a martingale). The assumed-density update is not one, so taking `max Q(b)` would credit the
update's bias as information; that variant is kept as `voi_vs_current`. Also recorded: `voi_stderr`
(`sd_s(max_a' Q_s) / sqrt(S_v)`) and `voi_net = VOI - net_cost`. `info_gain_nats` is 0 for the candidates the loop
skipped. The sampled-outcome loop runs on a copy of the head view on the fast state's device (same numbers on the CPU;
d807a40).

`net_cost = cost_weight * money + time_weight * seconds`. Both weights are 0 by default, so cost, time and
opportunity cost do not enter at default settings. A prediction error alone has no value here: a noisy observation
that cannot change the best candidate gives `E[max] = max E`.

Exact references (tests only, `cam_live/tests/predict/test_voi.py`):
* the linear-Gaussian last layer, `E[max_a (m_a + beta_a Z)]` evaluated exactly on the upper envelope
  (`predict/voi.py:expected_max_affine`, `voi_linear_gaussian`; Frazier et al. 2009);
* the knowledge gradient for independent arms (`predict/voi.py:knowledge_gradient_gaussian`);
* GLM sampling (`predict/voi.py:voi_glm`);
* `VOICosts.net = exp(-r dt) E[max] - max + direct - cost - rate dt` (not used by the agent).

The learned estimator of downstream usefulness, `predict/voi.py:DownstreamValueEstimator` (Student-t output,
`downstream_loss`), is attached as head `downstream` (`predict/heads.py:attach_heads`). At this commit no provider
trains it and no selection rule reads it.

Where VOI is used:
* as the reward signal `voi` (§7.1), and `info_gain_nats` as the reward signal `info_gain` (§7.1);
* by rule `voi` (`softmax(q + ig_weight * VOI)`), which is the goal-free rule (`agent.py:AgentSettings.goal_free_rule`).

The default selection rule does not use it.

---

## 6. Selection rule `thompson_policy`

### 6.1 Probabilities

For one decision with feasible set `A`, the numbers recorded per candidate (`predict/policy.py:DecisionMaker.scores`)
are:
* `q_i = extras["q_goal"] - net_cost_i` (posterior-mean goal value, §3.2);
* samples `q_{s,i} = extras["ts_s"] - net_cost_i` (`S = 32` joint samples, §3.2; when absent, normal marginals
  `q + sd N(0,1)`);
* policy logits `l_i`.

The rule, with the function computing each line:

    z_{s,i} = (q_{s,i} - mean_A q) / sd_A q             (sd over the posterior means; 1 if sd <= 1e-12)    standardised_values
    c^0_i   = log softmax_A(l)_i - mean_A log softmax(l)                                                  centred_log_policy
    c_i     = (c^0_i - kappa zbar_i) / max(1, RMS_A(c^0 - kappa zbar)),
              zbar_i = (q_i - mean q)/sd q,  kappa = <c^0, zbar>_A / <zbar, zbar>_A    (projection removed) policy_preference
    lambda_s = clip( max(beta_pi,s, 0) / max(beta_q,s, 0.05), 0, 10 )                                       policy_weight
    score_{s,i} = z_{s,i} + lambda_s c_i                                                                    thompson_policy_scores
    p_i = (1/S) sum_s 1[i maximises score_s] / #maximisers          (exact ties share the sample)          _tie_share
    p   <- (1 - eps) p + eps / |A|,   eps = 0.02                                                            apply_floor

`predict/policy.py:DecisionMaker.thompson_policy`, `DecisionMaker.probabilities`. The candidate is drawn with
`torch.multinomial(p)` from exactly this vector (`predict/policy.py:DecisionMaker.select`). So the recorded
`Decision.probabilities` are the true propensities of the sampling mechanism, and every feasible candidate has at
least `eps / |A|`. They are a 32-sample Monte-Carlo estimate of the ideal Thompson probability, not that probability.
The diagnostics record the calibration summary, `policy_share_tv = 1/2 ||p(beta_pi) - p(beta_pi = 0)||_1` (before the
floor) and the value-only distribution. G2-decision (`probe_selection.py`, 10/10 at `f54d8cc` and again at `ef0f3ff`)
found the chosen-index frequencies equal to the recorded propensities over 20 000 decisions (max |z| 0.9) and the
vector invariant to `q -> 37 q - 1234.5` and to `logits + 3.7 z`.

With probability `experiment_rate = 0.1`, and an expected order flip >= 0.2 between two candidates
(`predict/policy.py:DecisionMaker.propose_experiment`), the agent instead assigns at random over registered arms
(rule `randomized`: uniform over arms, then the floor over the arms; `predict/policy.py:DecisionMaker.select_experiment`).

### 6.2 What the projection does

`c` keeps only the part of the policy's preference that is orthogonal (over the decision's feasible candidates) to
the world model's standardised ranking. It is bounded like `z` (divided by its RMS once that exceeds 1 nat). A
uniform policy has `c = 0`; a policy that only repeats the world model also has `c = 0`.

### 6.3 Selection calibration (learned weight of the policy)

`predict/policy.py:SelectionCalibration`. It uses each executed decision whose verified progress `y` is final (the
agent's signed annotation, §8.2). The features at decision time are `z_c` (the chosen candidate's `zbar`, winsorised
±4) and `c_c` (its `c`, ±4) (`predict/policy.py:SelectionCalibration.features`). It fits

    y / s = beta_0 + beta_q z_c + beta_pi c_c + noise,        s = sqrt( sum_k w_k y_k^2 / sum_k w_k ),  w_k = 0.99^(n-1-k)

with prior `beta ~ N((0, 1, 0), diag(1, 0.5^2, 0.5^2))`, by weighted least squares with the forgetting weights. The
noise variance is iterated 3 times as `sigma^2 = (nu + sum w r^2) / (nu + sum w)`, `nu = 2`, floor 1e-6
(`predict/policy.py:SelectionCalibration.posterior`). Draws are `(beta_q, beta_pi) ~ N(mean[1:], cov[1:,1:])`
(`SelectionCalibration.sample`). At most 1024 observations are kept, each decision once. It is fed by
`reward/metalearner.py:MetaLearner.refresh_selection` and shared through the policy head
(`predict/policy.py:selection_calibration`).

At the prior, `lambda` is positive in about half of the draws (measured above: mean 0.357; G2-decision G2D-2:
P(lambda > 1) = 0.08). A near-uniform policy has `c ≈ 0`, so the prior has no influence until the policy's logits
differ (§13).

Other rules (`predict/policy.py:selection_probs`): `policy` softmax(l / T); `thompson` the share of samples where `i`
is best; `ucb` argmax(q + sd); `greedy` argmax(q); `voi` (§5). The agent's ablation `policy` + value prior adds
`value_weight * zbar` to the logits (`agent.py:CamLiveAgent._with_value_prior`).

---

## 7. Internal reward `R_psi` and its bootstrap

### 7.1 Signals

A fixed, versioned schema (`reward/signals.py:SCALAR_SPECS`, version 1.0) of 15 scalars plus one-hot result status (6)
and failure class:

| signal | transform | source at fdab7d3 (`reward/assemble.py:record_signals`, `SignalAssembler.assemble`) |
|---|---|---|
| success_before, success_after, delta_v | identity | evaluator head on `b` and `b'` (§3.3) |
| action_quality | symlog | `predict/evaluator.py:action_quality` on `q_goal - net_cost` |
| info_gain | log1p | chosen candidate's `extras["info_gain"]`, else `extras["info_gain_nats"]` (§5): the *expected* fast-posterior KL of its own outcomes, estimated before acting (KF-E02) |
| voi | symlog | chosen candidate's `voi` (§5) |
| cost_money, cost_seconds, cost_compute_s | log1p | measured `ActionResult.cost` |
| elapsed_s | log1p | decision time minus the newest ACTION item in `H_t` |
| prediction_error_std | log1p | `\|outcome - mean\| / std` of the chosen candidate's prospective predictions whose outcomes were observed by reward time |
| epistemic_before | log1p | chosen candidate's `uncertainty` (goal-value epistemic std) |
| epistemic_after | log1p | annotation `reward_signals.epistemic_after`, written once all goal-relevant outcomes of the decision are absorbed (`agent.py:CamLiveAgent._annotate_epistemic_after`, §3.2; LRN-3) |
| verified_progress_delta | symlog | the agent's signed per-decision annotation (§8.2), else the verifier-trust OUTCOME items new in `H_{t+1}` that answer this decision's goal-relevant questions |
| context_novelty | identity | stored value only |

`learning_progress` is an extra input of the bootstrap only (not one of the 15). It is the annotation
`reward_signals.learning_progress` = world-model NLL on the record's matured targets before minus after the next
frequent update, on the same targets (`agent.py:CamLiveAgent._lp_after`, `predict/providers.py:record_world_nll`,
`reward/bootstrap.py:learning_progress`; LRN-3).

Each value is what was known at reward time `max(decision ts + duration, newest post item)`. A signal without a
source is masked (value 0, mask 0). Stored signals are read from `decision.diagnostics["reward_signals"]`, then from
the per-signal annotations `reward_signals.<name>` (`reward/assemble.py:stored_signals`). Annotations are added after
the decision is stored, so they are read by every batch built after they were written. `epistemic_after` stays
masked for a decision whose goal-relevant outcomes arrive after its objective ended (`Predictor.forget_goal_views`).

### 7.2 `R_psi`

    x_t = W_lat [n(b), n(e_a), n(b') - n(b), n(e_g)] + W_sig [n_s(sig) * mask, mask]
    h_{t+1} = GRU(x_t, h_t)                       (lifetime context)
    R_psi  = B tanh( out(SiLU(W SiLU(x_t + W_life h_t))) / B ),   B = 1

`reward/intrinsic.py:IntrinsicReward.step_features`, `IntrinsicReward.head`, `IntrinsicReward.reward_sequence`,
`reward/intrinsic.py:LifetimeContext`; `reward/integrity.py:bound_reward`.

`n(.)` are running mean / variance normalisers (float64, clip ±8, `reward/intrinsic.py:RunningNorm`). They are updated
only by `IntrinsicReward.observe` on real decisions and frozen inside a meta window (`IntrinsicReward.meta_window`).
The statistics in force when an inner batch was built are stored with it and passed to the functional call
(`reward/inner.py:InnerBatch.norm_state`), so the differentiated reward is the one the update used. The goal-free
mode uses a learned null goal. `version` is a hash of the parameters and the normaliser statistics. `psi` is exactly
`IntrinsicReward.parameters()` (`IntrinsicReward.meta_parameters`). Only the MetaLearner's Adam holds them
(`reward/integrity.py:assert_reward_optimizer_exclusive`).

The lifetime context `h` is reset only by `reward/metalearner.py:MetaLearner.new_lifetime`, which the agent never
calls. At a new objective the agent calls `reward/metalearner.py:MetaLearner.new_objective` (KF-E05): that starts a
new segment of the meta-learner's bookkeeping (returns, the return-to-go chain and the discount do not run across
it; §8.1, §9.2) but leaves `h` as it is. So `h` summarises the agent's whole life across its objectives.

### 7.3 Bootstrap reward and mixing

    r_boot = 4 tanh( [ 1.0 tanh(LP) + 1.0 Delta V + 0.5 g_ctx * clip((eps_b - eps_a)^+ / eps_b, 0, 1)
                       + 2.0 symlog(v) * mask_v ] / 4 )

`reward/bootstrap.py:BootstrapReward.__call__` (weights `reward/bootstrap.py:BootstrapWeights`; bound 4 set in
`reward/inner.py:PolicyImprovementLoss.__init__`; inputs mapped by `PolicyImprovementLoss._bootstrap`). The terms:
* `LP` is the learning progress of §7.1 (0 where not yet written);
* `eps_b`, `eps_a` are the epistemic std before / after (§7.1); the exploration term is 0 unless both are present;
* `v` is the verified progress already observed;
* `g_ctx` is the noisy-TV gate: per context key, over the last 6 epistemic values, `g <- g + (1 - g) 0.5` while they
  fall by >= 5 %, else `g <- 0.6 g` (`reward/integrity.py:NoisyTVGate`).

The bootstrap is labelled `bootstrap` and never meta-learned.

Mixing inside the inner loss (`reward/inner.py:PolicyImprovementLoss.intrinsic`), with `n(x) = (x - mean_batch x) /
sqrt(mean_batch (x - mean x)^2 + 1e-12)` (`reward/integrity.py:scale_normalise`):

    r_i = n(r_boot)_i                                   if mix >= 1
        = n(R_psi)_i                                    if mix <= 0
        = mix n(r_boot)_i + (1 - mix) n(R_psi)_i         otherwise
        + g_i                                           goal term (§8.2)

Each component is standardised first, so `R_psi -> alpha R_psi + beta` (alpha > 0) changes nothing at any mix
(re-evaluated above: 1.2e-12). psi acts only through the shape of `R_psi`, with weight `1 - mix`. The value stored on
the record (`ExperienceRecord.intrinsic_reward`) is the unstandardised `mix r_boot + (1 - mix) R_psi`
(`reward/bootstrap.py:RewardMixer.combine`, called once per real decision by `PolicyImprovementLoss.reward_step`). The
record label is `bootstrap` until at least one meta-update was applied **and** a promotion lowered the mix
(`reward/bootstrap.py:RewardMixer.label`).

---

## 8. Policy improvement: the inner loss and the live update

### 8.1 Inner loss

Rows are the support decisions in order of objective segment, then time (`reward/inner.py:PolicyImprovementLoss._order`).
The latents `b, e_a, e_g, b'` are cached constants (`encode_decisions(grad=False)`), so nothing reaches the core or
the backbone. `pi_theta` is evaluated through `torch.func.functional_call` on the trunk and policy head
(`reward/inner.py:AdaptableSubset.forward`).

    G_i  = r_i + gamma_i G_{i+1},   gamma_i = exp(-beta (t_{i+1} - t_i)) within an objective segment, 0 at its end,
           beta = ln 2 / 86400 s                                                    reward/meta.py:discounted_returns
    G_i <- G_i + Gc_i - Gg_i        only if some row carries return-like credit (§8.2)
    Gn   = n(G)                                                                     (standardised over the batch)
    A_i  = Gn_i - V_{theta_sg}(b_i, C_i)                                            (baseline held fixed, see C-9)
    L_inner(theta; psi) = -mean_i log pi_theta(a_i | b_i, C_i) A_i
                          + 0.5 mean_i 1/2 (V_theta - Gn_i)^2
                          - 0.01 mean_i H[pi_theta(. | b_i, C_i)]
                          - 1.0 beta_q^+ mean_{i: ok} sum_j pi_theta(j | i) zbar_ij               (all-candidate term)

`reward/inner.py:PolicyImprovementLoss.functional`, `reward/inner.py:InnerConfig`, `reward/inner.py:InnerBatch.gamma`
(built in `PolicyImprovementLoss.build_batch`). `zbar_ij` is the recorded standardised predicted value of candidate j
(`predict/policy.py:SelectionCalibration.candidate_values`). `beta_q^+` is the posterior mean of the calibration
clipped at 0 (a constant in psi). `theta_sg` are copies of theta that are not variables of the inner derivative but
are still differentiated by the unroll. With plain `.detach()` the path `theta_k(psi) -> V` would be lost for K >= 2
(change request C-9, measured bias 1.0 % at K = 2, 3.3 % at K = 4).

### 8.2 Goal term and credited goal return

Per row, the goal reward `u_i` is the first available of (`reward/inner.py:PolicyImprovementLoss._goal_term`):

1. `verified`: the verified progress attributed to decision i once final (`reward/inner.py:verified_progress`, the
   decision's `diagnostics.reward_signals.verified_progress_delta`);
2. `credit:<est>`: the stored *per-decision* credit of the evidence-selected estimator, when the agent's credit hook
   gives one (KF-E01; below);
3. `predicted`: the calibrated decision-time prediction `s (beta_0 + beta_q^+ zbar_chosen)`
   (`predict/policy.py:SelectionCalibration.predict`), or the raw `extras["q_goal"]` before the calibration has a scale
   (`reward/inner.py:predicted_goal_value`);
4. nothing (0, not counted).

    g_i = c_g u_i / rms(u over rows with a source and no return-like credit),   c_g = 1     PolicyImprovementLoss.goal_term

The verified annotation is written once, from signed verifier readings only, when every goal-relevant answer of the
decision is in (`agent.py:CamLiveAgent._annotate_verified_delta`). It is the sum of the decision's own attributed
outcomes; 0 if the environment attributes outcomes and none are attributed to it; otherwise the signed change of the
objective metric across the decision.

**Credit (KF-E01).** The agent sets `PolicyImprovementLoss.credit_fn` to `agent.py:CamLiveAgent._credit_for` (read by
`reward/inner.py:PolicyImprovementLoss._credit`). For a record it returns the stored credit of
`CreditAssigner.default` (`credit/providers.py:LEARNING_KEY`), or nothing:
* estimator `simple` → a per-decision value (the decision's own linked outcome amounts, §11), used as `u_i` (step 2
  above). It is returned only in objectives whose environment attributes outcomes to decisions
  (`CamLiveAgent.attributing`); elsewhere a linked answer is a reading of the objective level, not a contribution;
* estimator `rudder` / `cocoa` → a *return-like* value `C_i` (RUDDER's redistributed return-to-go, COCOA's credit of
  the action for its own and later outcomes). It replaces the row's discounted goal return:

      Gg   = discounted_returns(g)                                                       PolicyImprovementLoss._goal_return
      Gc_i = c_g C_i / rms(C over credited rows)        at a credited row               PolicyImprovementLoss.credited_goal_return
           = g_i + gamma_i Gc_{i+1}                      elsewhere
      G   <- G + Gc - Gg

  Credited rows are removed from the rms of `g`, and `g_i = 0` there, so `G = discounted(r_intrinsic) + Gc`
  (`reward/inner.py:PolicyImprovementLoss._credit_returns`).

The goal term and the credited returns are constants in psi. They are frozen in the cached batch, so the
differentiated update uses the same values as the live one. G2-decision (§3, `probe_meta_path.py`) recomputed 38 live
updates with this loss, including credited returns, to <= 4.3e-6 relative. In the agent the default estimator stays
`simple` (§11.3), so only step 2 is reached in practice.

### 8.3 Inner step and live trust region

    theta_{k+1} = theta_k - alpha clip_group(grad_theta L_inner(theta_k; psi, D)),   k = 0, 1   (K = 2)

`reward/meta.py:MetaGradient.inner_step`. The per-group norm clip `g <- g min(1, 1/||g||)` is in the graph and applies
separately to the policy head and the trunk (`reward/inner.py:GroupClip`; `MetaConfig.clip_policy / MetaConfig.clip_trunk`).

The live update is `reward/metalearner.py:MetaLearner.policy_update`. The agent calls it every 4 newly rewarded
decisions, on the 16 newest plus up to 16 earlier decisions with final verified progress
(`MetaLearner._replay_matured`). It chooses `alpha` by a trust region on

    KL = mean_rows sum_{a in A} pi_0(a) [log pi_0(a) - log pi_alpha(a)]                         MetaLearner._kl

* target 0.05; a step is accepted when `KL in [0.05/1.5, 1.5 x 0.05]`;
* at most 6 retries;
* the next step is the geometric mean of the bracketing steps once both sides are known, else
  `clamp(alpha sqrt(0.05/KL), 0.01, 2.0)`;
* the search stops when the change is <= 2 %.

The chosen `alpha`, the step search, `theta_0` and the support batch are stored as an `UpdateEvent`, so the
meta-gradient differentiates exactly the SGD step that was applied. The chosen `alpha` is a constant of that function
(G2D-3, §13). The update runs under `PsiGuard` (psi must be bit-identical afterwards).

---

## 9. The meta-gradient (lifetime objective, Zheng et al. 2020)

### 9.1 Objective and estimator

    J(psi) = E[ U_verified(D_query; theta_K(psi)) ]
    surrogate  S(psi) = sum_rows w_i rho_i log pi_{theta_K(psi)}(a_i | x_i) (U_i - b_i)
               [+ (sum_k sum_j log pi_{theta_k}(a_j)) (J_hat - c)   when support step k >= 1 was sampled from pi_{theta_k}]
    dJ/dpsi    = d S / d psi  through the K-step unroll

`reward/meta.py:MetaGradient.inner_update` (create_graph, `mode="unroll"`; `"checkpoint"` gives the same gradient by
an adjoint reverse sweep, `MetaGradient._adjoint`; `"first_order"` drops the inner Hessian chain),
`MetaGradient.outer_surrogate`, `MetaGradient.meta_gradient`. `theta_0` is a constant (truncated window).

In the agent each update event is an episode (`reward/metalearner.py:MetaLearner._episode`, `_steps_of`,
`meta_update`):
* `theta_0` = the event's snapshot;
* support = its batch with `mix = meta_mix = 0` (the pure learned reward: a *shadow* update while the live mix is
  above 0);
* query = the decisions taken after it.

With `unroll_events = 1` (default) no support step is flagged on-policy, so the inner-data score term is not used at
default settings.

### 9.2 Query rows, independence, utility

* Windows (`reward/windows.py:WindowBuilder.cut`): support = eligible decisions before the first query decision,
  query = those after. A query row whose independence group already occurs in the support (drop reason
  `straddles_boundary`), or that lies inside the gap, is dropped. Rows must have propensities and
  the provenance of the query rows.
* Independence (`reward/windows.py:independence_key`, `WindowBuilder._prepare`): experiment id, else parent decision,
  else the decision. Rows that share a verified outcome (`dedup_key`) are merged (union–find). An objective outcome's
  ledger group overrides it for its rows. Groups are counted, never rows.
* Held-out groups: a deterministic hash of the group key with salt `promotion` reserves 25 % of the groups for the
  mix promotion (§10.1) (`reward/windows.py:held_out_groups`).
* Utility (default `utility="return_to_go"`). Objective utilities `U_obj` are the change of
  `objective_utility = 1 if achieved else 0.5 clip(value / max(target, 1), 0, 1)` since the previous admitted
  evaluation. Goal-relevant verifier measurements `m_j` contribute their measured value:

      U_a = U_obj(a) + sum_{j >= a, same objective segment, same held-out side} exp(-beta (t_j - t_a)) m_j

  `reward/metalearner.py:MetaLearner._episode`; `reward/integrity.py:objective_utility`, `measurement_utility`. The
  numbers come from the hash-chained outcome ledger, derived from signed records. A disagreeing caller utility is
  refused (`reward/integrity.py:OutcomeLedger.admit_verified`, `utility_matches`). Only objectives with a recorded
  operator / verifier origin supervise. A row whose goal-relevant answers are not all in stops the return-to-go chain
  there. The outer utility does not read the credit store.
* Baseline (`reward/metalearner.py:MetaLearner.outer_baselines`): the mean utility of the rows outside the row's
  baseline group. With return-to-go that group is the whole chain (objective segment, held-out side), because every
  earlier row's `U` contains this row's outcome. Otherwise it is the running mean of verified utility at the row's
  registration. It does not depend on the row's own action or outcome (G2-decision: max |db| 3.6e-15 over 200
  random chains).

### 9.3 Importance weighting and the ESS rule

The episode is on-policy (`rho = 1`) only if all three hold:
* every event of the episode was applied by `policy_update`;
* replaying it without a graph from `theta_0` with the current psi, statistics and mix gives parameters bit-identical
  to those the live update left (`reward/metalearner.py:MetaLearner.logged_policy_is_current`);
* `max |pi_{theta_K}(a) - mu(a)| <= 1e-3`.

Otherwise (always during the cold start, because the shadow mix 0 differs from the live mix):

    rho_i = pi_{theta_K}(a_i) / mu(a_i)          (from the logged log-propensity; refused if missing, > 1, or mu < 1e-6)
    ESS   = (sum rho)^2 / sum rho^2              per lifetime segment of the query batch (the agent passes one per episode)
    refuse if ESS < max(2, 0.05 n)               -> "effective_sample_size_too_small"
    w_i   = (1/N) min(rho_i, 10)                 (N = all outer rows of the meta-update; not self-normalised)

`reward/meta.py:MetaGradient.outer_surrogate`. The gradients of all supported episodes are summed. Then
(`MetaLearner.meta_update`, `reward/meta.py:MetaGradient.apply`) the step is refused:
* if the norm is not finite;
* if every advantage satisfies `|U - b| <= 1e-9 max(1, max|U|)` (`zero_advantage`; the rows stay available);
* if the norm is <= `min_meta_grad_norm` (0).

Otherwise one Adam ascent step is taken on `-g * min(1, 0.5/||g||)` (lr 1e-4). If psi is bit-unchanged the optimizer
state is restored. Skip reasons recorded in `MetaUpdateRecord.skipped_reason`:
* `no_policy_update_recorded`, `insufficient_independent_outcomes:n<8`;
* `no_supported_episode:<reason>`, with the engine's reasons `empty_or_misaligned_query`, `non_finite_utility`,
  `unverified_outcome_in_query`, `missing_propensities`, `invalid_propensities`, `propensity_below_floor`,
  `effective_sample_size_too_small`, `inner_update_has_no_graph`;
* `non_finite_meta_gradient`, `zero_advantage`, `zero_meta_gradient`, `psi_unchanged_by_the_step`, `not_applied:*`.

### 9.4 Exact reference and independent checks

`reward/exact.py` implements the tabular setting of Zheng et al. 2020 (`r_psi[t, s, a]`, `theta_{k+1} = theta_k +
alpha grad J_int`, `J(psi) = J_ext(theta_K(psi))`) three ways:
* autograd through exact updates;
* central finite differences;
* the sampled REINFORCE estimator with the inner-data score term and leave-one-out baselines (the sampled objective is
  also enumerated exactly).

Tests: `cam_live/tests/reward/test_reward_exact.py` (`test_exact_meta_gradient_equals_finite_differences[K]`,
`test_cutting_the_pathway_gives_exactly_zero[K]`, `test_sampled_estimator_is_unbiased_by_z_score_and_relative_error`),
`cam_live/tests/reward/test_reward_meta.py` (`test_meta_gradient_matches_finite_differences`,
`test_first_order_approximation_is_exact_for_one_step_and_biased_beyond`,
`test_sampled_on_policy_estimator_with_score_term_is_unbiased`), and
`cam_live/tests/reward/test_reward_inner.py` (`test_meta_gradient_matches_finite_differences_on_the_real_adaptable_subset`).

Independent checks (G2-decision §3 at `f54d8cc` and round 2 at `ef0f3ff`; `reward/` unchanged since; not re-run by
me):
* the applied meta-gradient equals an own recomputation (own unroll, ESS rule, leave-group-out baselines, weight 1/N)
  to 2.4e-7 .. 7.9e-7 relative;
* on the real adaptable subset, the engine gradient equals the own one to <= 2.8e-14 and central finite differences to
  about 5e-10, for K = 1, 2, 4, 8 (float64).

---

## 10. Promotion

### 10.1 Bootstrap mix: a non-inferiority test

For each held-out group g (`reward/metalearner.py:MetaLearner._promotion`, `reward/bootstrap.py:paired_policy_value_diffs`):
from the same snapshot and support, `theta_l` = the inner update with the learned reward (mix 0) and `theta_b` = with
the bootstrap (mix 1). On the group's logged query rows:

    d_g = mean_rows [ min(pi_l(a)/mu(a), 10) - min(pi_b(a)/mu(a), 10) ] U

Differences are buffered until there are >= max(2, 4) groups. Then, with mean `dbar`, `se = sd / sqrt(n)`, the
one-sided 95 % Student-t quantile `t_{n-1}` and the margin `delta = 0 + 0.05 RMS(U)`:

    lower  (mix <- max(0, mix - 0.25))   if dbar - t se >= -delta         (learned reward not worse by more than delta)
    raise  (mix <- min(1, mix + 0.25))   if dbar + t se < 0               (shown harm)
    hold                                  otherwise, or fewer than 4 groups, or non-finite

`reward/bootstrap.py:RewardMixer.update`. G2-decision found 300 of 300 decisions equal to its own one-sided t rule.

### 10.2 Checkpoint promotion: gates, not a test

`credit/checkpoint.py:CheckpointManager.promote` with the agent's `PromotionGates(max_validation_drop=0.5,
max_retention_drop=0.1)`, called through `credit/checkpoint.py:promote_learner`. Validation metrics are the negated
held-out provider losses `heldout/neg_<provider> = -L_p` (`ContinualLearner.validate`, each with its provider's
provenance). A checkpoint is promoted iff:
* every validation metric satisfies `new >= old - 0.5` and every retention metric `new >= old - 0.1`, relative to the
  last promoted checkpoint;
* all metrics are finite, and there is at least one of each kind;
* the held-out evidence exists: without any provider held-out loss the refusal says "no held-out evidence" (INT3A-5);
* the checkpoint exists and verifies.

Metrics whose provenance is not in `credit/checkpoint.py:INDEPENDENT_PROVENANCE` (e.g. the agent's own reward) are
ignored. There is no primary metric and no confidence bound. The agent calls it every 24 frequent updates and at the
end of each objective (`agent.py:CamLiveAgent.promote`).

A refusal restores the last promoted learner checkpoint, except two things (LRN-6):
* `R_psi` (whole module, with the meta optimizer) always stays;
* the policy head stays unless `decision_agreement` is among the reasons.

A promotion or rollback re-references the slow-step drift bound (`ContinualLearner.set_slow_reference`, §12). The
agent's checkpoint manager keeps the newest 2 checkpoints and the newest promoted one. Optimizer state is kept only in
the newest kept checkpoint (`credit/checkpoint.py:CheckpointManager.prune`, `keep_optim = 1`; 6159def). A rollback to
a checkpoint without optimizer state starts the learner's optimizers fresh.

---

## 11. Retrospective credit

Credit is recomputed for an objective when its outcomes mature (`credit/providers.py:CreditAssigner.assign`). It is
stored per record with a version history (`credit/replay.py:SqliteExperienceStore.set_credits`).

Simple attribution: `credit = sum_links weight x value x exp(-beta (t_outcome - t_decision))`, declared answers first
(`credit/simple.py:credit_from_links`). The weights come from evidence links. Ambiguous evidence is split with
uncertainty. Negative or malformed amounts are treated as ambiguous, and amounts whose sum overflows become shares
relative to the largest (G2L-3) (`credit/links.py:attribution_amounts`).

### 11.1 RUDDER

`F_xi` is a causal transformer over per-decision latents (+ time features `log1p(tau/60)`, `log1p(dt/60)`, goal),
predicting `G / scale` from every prefix; `pred[0] = F(empty)` (`credit/rudder.py:ReturnDecomposer.forward`).

    loss  = mean over prefixes of matured trajectories (F(H_<t) - G/scale)^2          (linear target; scale = max |G|)
    Gamma_t = exp(-beta tau_t),  tau_t = time since the trajectory's first decision
    c_t = [F(H_<=t) - F(H_<t)] / Gamma_t  (+ F(empty)/Gamma_first on the first, + [G - F(H_<=T)]/Gamma_last on the last)
    sum_t Gamma_t c_t = G   exactly (conservation, for any F)
    learning signal_t = [G - F(H_<t)] / Gamma_t = (sum_{k>=t} Gamma_k c_k) / Gamma_t     (redistributed return-to-go)

`credit/rudder.py:ReturnDecomposer.loss`, `ReturnDecomposer.redistribute`, `Redistribution.conservation_error`.
Unmatured trajectories contribute no loss and get NaN credit. The values are stored as `credit["rudder"]` (the signal,
the return-like value of §8.2) and `credit["rudder_contribution"]`.

### 11.2 COCOA

The hindsight model `h(a | b, u, elapsed)` is defined over the recorded candidates, with logits = `log pi(a|b) +
f(b, u, elapsed, a)`. `f`'s output layer is zero-initialised, so `h = pi` and `w = 0` before training. It is a 4-member
bootstrapped ensemble (`credit/cocoa.py:HindsightModel`), trained by cross-entropy of the executed candidate given a
later outcome. Training calls use Poisson(1) bootstrap weights per member; a held-out evaluation uses none, so it is
deterministic and draws no random number (`credit/providers.py:HindsightProvider.loss`; 5a59e17).

    w(a, u) = clip( mean_members h_m(a | b, u) / pi(a) - 1,  -1, 10 )
    credit(a_t) = own outcomes of t (discounted) + sum_{later outcomes k} w(b_t, a_t, u_k) v_k exp(-beta (tau_k - t))

`credit/cocoa.py:HindsightModel.contribution`; `credit/providers.py:HindsightProvider.credit`, `HindsightProvider._pairs`
(an outcome pairs with t if t did not declare it and it was observed after t). Refusal (`credit/cocoa.py:check_support`,
value NaN, status `unsupported`) happens when:
* propensities are missing;
* they are NaN, > 1, or do not sum to 1 (tolerance 1e-3);
* any candidate is below `min_prob` (1e-3);
* the candidate set is open-ended without tracked proposals (backbone-proposed sets);
* there are no candidates;
* the per-candidate ESS is insufficient, when required.

The association `h - pi`, the counterfactual estimate `w v` and the experimental effect (randomised arms only) are
separate fields. The hindsight model also has `credit/providers.py:HindsightProvider.adapter_sensitivity_loss`: the
same loss with a graph into the adapters, on its own training records, used by the slow step only as a constraint
(§12).

### 11.3 Which credit trains, and estimator selection

`CreditAssigner.default` is the evidence-selected estimator (`credit/select.py:EstimatorSelector.select`). It starts as
`simple`.

Evidence (`credit/providers.py:CreditAssigner.evaluate_experiments`, run by the learner after every learning update
since DX-1 / cf46eb1: `agent.py:CamLiveAgent.__init__` calls `CreditAssigner.attach_experiments`):
* the agent's experiment registry is copied into the experience store before each update
  (`CreditAssigner.record_experiments`);
* for every randomised experiment whose two arms each have >= 2 matured *held-out* units (units in an objective that
  an estimator or the learner trained on are excluded), `CreditAssigner.compare_with_experiment` measures the arm
  difference of the return-to-go and each estimator's estimate of it, with the paired standard error
  `credit/select.py:paired_comparison` (`D = sum_i (w_arm e_i - v_arm y_i)`, `Var(D) = sum_arm n_arm Var_arm(q)`).

An estimator replaces the default only if it passes every gate (`credit/select.py:EstimatorSelector.score`):
* >= 5 scored randomised held-out experiments;
* mean `z^2 <= 3`;
* at least one significant measured effect, with sign agreement >= 0.8;
* its RMSE on the common experiments is at least 10 % below the current default's.

Every pass that cannot compare logs its reason to `credit_selection.jsonl`.

What reaches training:
* the policy update reads the default estimator's stored credit through the goal term (§8.2; KF-E01, 4476c27);
* the continual learner trains `F_xi` and the hindsight model as loss providers (§12);
* the meta-gradient's outer utility does not read credit (§9.2).

In the agent as configured the comparison never runs. Each agent experiment is one randomised decision (one unit), and
the agent sets `holdout_experiments=False`, so every experiment objective is trained on. G2-learn round 3 recorded
11–15 selection-log rows per run, each with "no randomised experiment has 2 or more matured held-out units in each
arm"; G2-decision round 2 recorded 14. So the default stays `simple`, and RUDDER / COCOA credit is computed, versioned
and logged but does not reach the policy update (§13, §14).

---

## 12. Continual learner and retention

**Frequent update** (`credit/learner.py:ContinualLearner._frequent_update`):
* each provider's gradient is taken separately;
* if the largest provider's share of the experiential gradient norm exceeds 0.9, it is scaled by
  `0.9 rest / (0.1 ||g_top||)`;
* per-group norm clip (experiential 1.0, heads 1.0; `config.py:OptimConfig`);
* one AdamW per group (weight decay 0.01 on matrices), linear warm-up over 8 steps.

It is triggered by >= 4 matured independent outcome units (`credit/learner.py:ContinualLearner.poll`,
`LearnerSettings.min_independent_outcomes`).

**Held-out split** (`credit/learner.py:ContinualLearner.is_validation`). The agent uses the stratified record split
(INT3A-5, c9fd652; `ContinualLearner._record_held_out`): the k-th stored record of an objective is held out iff

    floor(u + (k + 1) f) > floor(u + k f),    f = 0.15,   u = seeded hash of the objective

So exactly every 1/f-th record of each objective is held out, and an objective with >= 7 records always has one.

**Slow update** (adapters, after every 8th applied frequent update; `credit/learner.py:ContinualLearner._slow_update`).
The steps, in order:

1. **Held-out cells before.** For each provider p, its fixed validation sample (32 records, seeded) is split into
   `k = min(8, n)` chunks, and the loss `L_{p,j}` of each chunk is measured in eval mode
   (`ContinualLearner._validation_cells`). If no provider has held-out records, the step is **deferred**, not taken
   (`ContinualLearner._defer`; G2L-4).
2. **Gradients.** Model buffers and target scalers are saved (`ContinualLearner._running_state`). The losses are
   computed on 4 records encoded with a graph into the adapters through the newest 8 context items, and the adapter
   gradient `g_p` of each provider is taken. Then the running statistics are restored
   (`ContinualLearner._restore_running_state`; 5a59e17), so the slow step changes the adapters only. A provider without
   held-out records of its own does not move the adapters; if none is left, the step is deferred.
3. **Protected set.** `P` = the providers with `g_p != 0`, plus readers of the representation that train no adapter
   (COCOA's `adapter_sensitivity_loss`, eval mode, its own training records, no random draw;
   `ContinualLearner._reader_sensitivity`; 9a47d7d).
4. **Combination** (`ContinualLearner._balance_adapter_grads`; PCGrad, Yu et al. 2020). For every ordered pair in name
   order with `<g'_i, g_j> < 0`: `g'_i <- g'_i - (<g'_i, g_j> / ||g_j||^2) g_j`. Then the max-share cap of 0.9 is
   applied as for the experiential group, and `g = sum_p g'_p`.
5. **Step.** Norm clip 0.5. `lr = 5e-5 x warm-up factor x step_scale` (`ContinualLearner._set_lr`). One AdamW step.
6. **Projection of the applied update**, when `P` has two or more members (`ContinualLearner._project_update`;
   A-GEM-style, applied to the optimizer's update `Delta`, because AdamW's per-coordinate scaling can turn a
   conflict-free gradient into an update that raises another loss). In up to 4 passes over `P` in name order: while
   `<g_k, Delta> > 1e-12 ||g_k|| ||Delta||`, set `Delta <- Delta - (<g_k, Delta> / ||g_k||^2) g_k`.
7. **Verdict per provider** (`ContinualLearner._provider_verdict`, combined by `ContinualLearner._guard_verdict`;
   G2L-5, G2L-9). For each judged provider, with chunk changes `c_j = L^after_{p,j} - L^before_{p,j}`:

       d_p = mean_j c_j,   se_p = sd_j(c) / sqrt(n),   s_p = |mean_j L^before_{p,j}|
       floor_p = max(1e-4 s_p, 1e-7 max(1, s_p)),   r_p = 0.05 max(s_p, 1e-6)
       tol_p   = max( min(0.01, r_p, max(2 se_p, floor_p)),  min(floor_p, r_p) )
       D_p'    = D_p + d_p,     bound_p = max(0.02, 0.02 s_p)
       accept  iff for every judged p: all chunks computable and finite after the step, d_p <= tol_p, D_p' <= bound_p

   `se_p` counts as 0 when there is one chunk. `D_p` is the provider's summed change over the accepted slow steps since
   the reference (the learner's first slow step, then each promotion or rollback; `ContinualLearner.set_slow_reference`).
8. **Accept or undo.** On acceptance: `D_p <- D_p'` and `step_scale <- min(1, 1.5 step_scale)`. On rejection, weights,
   AdamW moments and step counts, learning rates and the schedule step are restored exactly, and
   `step_scale <- max(1/16, 0.5 step_scale)` (trust region; 2672207). A promotion does not reset `step_scale`.

Measured (G2-learn round 3 at `cf46eb1`; `credit/` unchanged since; not re-run by me):
* slow steps kept on the real configuration 11 of 12; debug agent 58 of 58 (+2 deferred); D's lab 27 of 27;
* kept steps raised no provider's held-out loss by more than 8.2e-5 relative;
* the projection leaves `<g_k, Delta> <= 2e-9 |Delta| |g_k|` on 200 random conflicting sets.

S's evaluation X0 at `257267c` (learner settings of `eval.oe_runner.SUBSTANTIVE_LEARNER`, not the defaults) recorded
34 kept, 2 reverted and 1 deferred of 37 slow steps (`docs/S/RESULTS.md` §3; not re-checked).

**Anchor-set functional regularisation** (`credit/retention.py:RetentionProvider`, `credit/retention.py:AnchorSet.penalty`).
At the end of an objective the last 64 records become anchors, with the model's outputs then (`b`, `z_chosen`). Each
frequent update draws 8 anchors evenly over families and adds `mean_anchors (f(x) - f_ref(x))^2 / mean(f_ref^2)` per
function (KL(ref || current) for probability functions), weight 1.

**Retention sets and `decision_agreement`** (`agent.py:CamLiveAgent.retention_metrics`). The first 24 executed records
of every objective are frozen with the policy's action distributions at that moment. The metrics are
`<family>@<objective>/neg_world_loss` (the negated world-model loss on their realised outcomes) and
`/decision_agreement = (1/n) sum 1[argmax p_ref = argmax p_now]`, with provenance `retention`. They feed §10.2.

---

## 13. Known numerical caveats

State at `fdab7d3`, from the code and the latest verification reports (`docs/verification/G2_learn.md`,
`G2_decision.md`, `G2_integration.md`). "Not re-checked" means I relied on the record and did not verify the claim
at this commit.

| caveat | source | state at `fdab7d3` (evidence) |
|---|---|---|
| GLM updates were not the Laplace update far from the prior predictor (fixed 6–12 Newton steps; rate off ×1.9–×339) | `docs/verification/G_nn.md` D1 | **fixed**. G2-learn §2: time-to-event 700/700 and binary 425/425 grid problems exact to 1e-15 |
| count mode search reported convergence at an absurd mode when `e exp(eta0)` is ~16 orders below the count (4 of 2625 grid problems, e.g. posterior eta 3.33e11 instead of 48.35) | G2-learn G2L-1 | **fixed** in 9680ea7 (G-step from both sides, root condition checked, §2.3). G2-learn round 2: 2625/2625; 60 000 random problems equal the bisection root (worst 3.2e-15), none reported unconverged. `nn/fast.py` unchanged since |
| count dispersion tempered binary / time-to-event | G_nn D2 | **fixed** (`FastBeliefState._apply`, `online_nll`). G2-learn §2: bit-identical posterior at dispersion 3.13 |
| bf16 LM head: generation log-probabilities accurate to 0.1–0.3 nats / token | G_nn D3 | **partly**: float32 head; bf16 trunk still −0.36…+0.12 nats on short completions (`docs/A/DESIGN.md`); not re-measured by G2-learn; not re-checked |
| NIG `predict()` gave `(b/a)(1+q)`, under-stating the variance by `a/(a-1)` | G_nn D4 | **fixed**. Re-evaluated above; G2-learn §2: ratio 1.000000000000 over 10 000 predictions. A Gaussian interval from `predict()` matches the Student-t variance, not its scale |
| impossible observations absorbed | G_nn D5 | **fixed** (`check_observations`; G2-learn §2: 10 cases raise, state bit-identical) |
| ill-conditioned features on realistic inputs (Gram max/min 2.3e4 at init) | G_nn D6 | **open** (initialisation property; the float64 Joseph updates stay exact) |
| text → belief depended on batch composition (1 %) | G_nn D8, G_cl m7 | **fixed**. G2-learn §2: 90 comparisons bit-exact (tiny backbone), max \|diff\| 0 (bf16 GPU) |
| `FastPrior(prior_var > ~5e5)` raises `OverflowError` in `_inv_softplus` (`math.expm1(sqrt(prior_var))`) | G2-learn G2L-1, aside | **open**. Measured at fdab7d3: `prior_var` 5e5 constructs, 1e6 raises. The default is 1.0 |
| return-to-go baseline depended on the row's own outcome (biased outer gradient) | `docs/verification/G_pr.md` D1 | **fixed** (`MetaLearner.outer_baselines`). G2-decision §2: max \|db\| 3.6e-15 |
| caller-chosen utility admitted for a signed outcome | G_pr D2 | **fixed**: derived in the ledger, mismatch refused (G2-decision §2) |
| calibration slope / intercept intervals ignored clustering | G_pr D5 | **fixed**: CR1 / HAC. G2-decision §2: a clustered null excludes 0 in 7 % (was 61 %) |
| NLL intervals not clustered | G_pr D10 | **fixed** (`score_predictions` clusters by experiment / decision / prediction; G2-decision §2) |
| cold-start shadow update switched to on-policy by tolerance | G_pr D7 | **fixed**: structural bit-identity check first (`logged_policy_is_current`; G2-decision §2) |
| world targets stored in float32 | G_pr D11 | **fixed**: float64 (`build_targets`) |
| `v.detach()` baseline dropped a meta path for K >= 2 (1.0 % / 3.3 % bias) | `docs/change_requests/C.md` C-9 | **fixed** (`theta_sg`). G2-decision: engine = own = FD (~5e-10) for K = 1, 2, 4, 8 |
| trust-region step is a constant of the differentiated function: where the step is interior and the live mix < 1, its dependence on psi is dropped | G2-decision G2D-3 | **open** (approximation). Exact at the default live mix 1, where R_psi has weight 0 in the live update |
| trust-region bounds bind often: steps at `max_inner_lr` = 2 with KL far below 0.05, or at 0.01 with KL up to 0.15 | G2-decision §3 | observation. KL per live update 0.035–0.074 (debug), 0.042–0.069 and 0.036–0.063 (real configuration, G2-decision §6 and round 2) |
| negative attribution amounts created credit > outcome | `docs/verification/G_cl.md` M3 | **fixed** (`credit/links.py:attribution_amounts`) |
| attribution amounts that overflow when summed dropped the outcome | G2-learn G2L-3 | **fixed** in 9680ea7 (shares relative to the largest amount; G2-learn round 2) |
| slow-update guard admitted degrading steps; a reverted step kept the Adam state | G_cl M4, m4 | **fixed** (G2-learn §2) |
| slow-step guard failed open without held-out cells | G2-learn G2L-4 | **fixed** in e4186c8 (the step is deferred). G2-learn round 2: no slow step applied without a verdict |
| guard judged the mean over providers, so one provider's degradation was hidden by another's gain | G2-learn G2L-5 | **fixed** in e4186c8 (per-provider verdicts, §12) |
| float-floor tolerance (1e-7 \|L\|) let a provider insensitive to the adapters veto a slow step (7 of 44 rejections) | G2-learn G2L-9 | **fixed** in 0745ae4 (floor 1e-4 \|L\|). G2-learn round 3: real configuration 11/12 slow steps kept (was 1/13 at f54d8cc) |
| the slow step's own loss evaluation moved running statistics (world-model normaliser, target scaler), so held-out changes of 1–8 % were measured that no adapter change caused | DX round 2 (`docs/DX.md`), G2-learn round 3 R3.2 (a) | **fixed** in 5a59e17. G2-learn: cells return bit-exactly after the restore; a harmful perturbation is still refused |
| COCOA's held-out loss used random bootstrap weights (non-deterministic before / after comparison) | DX round 2 | **fixed** in 5a59e17 (no bootstrap weights outside training calls) |
| sub-threshold creep: a consistent rise below the 1e-4 floor passes each step; 0.9e-4 per step was accepted for 224 steps until the 2.04 % drift bound; the bound restarts at each promotion (6.2 % after three) | G2-learn round 3 R3.2 (b) | observation (bounded per promotion cycle; across cycles only the 0.5 / 0.1 promotion gates bound it) |
| projection is first-order on the slow batch's training gradients | `credit/learner.py:ContinualLearner._project_update`; G2-learn round 3 R3.2 (c) | by design (`<g_k, Delta> <= 2e-9 \|Delta\| \|g_k\|` measured); the held-out outcome is judged by the guard |
| first promotion often found no held-out evidence (record-hash split) | G2-learn G2L-8 | **fixed** (914b10a, c9fd652: stratified split, explicit refusal). G2-learn round 2: 5/5 first promotions granted |
| long trajectories: decisions before the RUDDER window got no credit, and the window's discount was referenced at the wrong time | G_cl m3 | **fixed** (`docs/D/FIXES.md`). G2-learn §2: G-cl probe A passes; all 11 decisions credited with `max_len` 3, `sum_t Gamma_t c_t = G` to 1e-9 |
| stored credit was read by no training loss | G2-learn G2L-7 | **fixed** in 4476c27 (KF-E01, §8.2). G2-decision §3: live updates with credited returns equal the documented loss to <= 4.3e-6 |
| estimator selection never ran in the agent | G2-learn G2L-6, G2-decision G2D-4 | **wired** (715417e, cf46eb1). But no comparison can run in the agent as configured (one-unit experiments, `holdout_experiments=False`), so the default stays `simple` by construction (G2-learn round 3 R3.4; G2-decision round 2 RC-08 "partial") |
| verified bootstrap term saturated at bound 1 (1 and 42 adopters gave the same reward) | `docs/change_requests/INT2.md` INT2-6 | **addressed**: bound 4 and the unbounded goal term `u / rms(u)` (§7.3, §8.2); `4 tanh(x/4)` still compresses large values |
| live policy step too small to change behaviour (fixed `alpha = 1e-2`) | INT2-7 | **addressed**: trust region, target KL 0.05 (§8.3); see the trust-region rows above |
| support batches rebuilt from the store after a restore (different meta-step) | INT2-8 | **fixed** (5ad3777: batches of events still to be differentiated are checkpointed) |
| selection probabilities drift ~1e-6 after reload | `docs/change_requests/LRN.md` LRN-5 | withdrawn: 0.0 after the INT2-8 fix (LRN's measurement) |
| rollback after a refused promotion reverted psi and the policy head | LRN-6 | **applied** in `agent.py:CamLiveAgent.promote` |
| bootstrap inputs `epistemic_after`, `learning_progress` never written | LRN-3 | **fixed** in 4476c27 (annotations, §7.1). G2-decision MR-01: "LRN-3 / KF-E02 inputs produced" |
| reward signal `info_gain` had no producer | KF-E02 | **fixed** in 4476c27: the assembler falls back to `info_gain_nats`, an *expected* KL estimated before acting, 0 for candidates outside the 16 VOI candidates (§5, §7.1) |
| binary fast layer has no head offset: the predictor adds the fast posterior's *shift* from the objective-start state to the head logit (approximation) | `docs/change_requests/B.md` B-CR-10 (1) | **open** (`Predictor._means`, `_samples`) |
| fast posteriors are per kind and independent (a real reading cannot inform a count posterior; its fast-layer VOI is 0) | B-CR-10 (2) | **open** (by design) |
| change detector latched: after one detected shift a metric's misses were never `model_wrong` again | G2-decision G2D-1 | **fixed** in ef0f3ff (re-arm below 0.25 x threshold, §3.4). Residual: the statistic is not capped, so re-arming lags with the shift's duration: 223 / 960 / 4918 calibrated residuals after a 3-sigma shift of 40 / 200 / 1000 observations (G2-decision round 2). Labels are diagnostics only |
| prior-only policy weight before any verified outcome: mean lambda 0.36, P(lambda > 1) = 0.08; a peaked preference moves a 6-candidate decision by TV 0.17 | G2-decision G2D-2 | observation (by design; an untrained policy has `c = 0`). Mean lambda re-measured above: 0.357 |
| Thompson probabilities are a 32-sample estimate; before the floor a candidate can have probability exactly 0 | `predict/policy.py:_tie_share` | by design; the floor gives every feasible candidate >= 0.02 / \|A\| |
| meta-update evidence is scarce: one objective segment is one independent group for objective utilities; ESS >= 2 per episode; 8 groups needed | `docs/INTEGRATION.md` §8, §9 e; G2-decision MR-05 | **open** (by design). Real configuration: 0 applied of 98 calls in G2-decision's two runs (`effective_sample_size_too_small` once 4 groups existed). Staging validation: 0 applied, 34 skips for fewer than 3 groups (G2-integration round 2). S X0 at 257267c with a minimum of 4 groups: 4 applied of 123 records (`docs/S/RESULTS.md` §3) |

---

## 14. Not implemented, or different from the design / documentation

* **Credit reaches training only as `simple` credit.** The policy update uses the stored credit of the evidence-selected
  estimator (§8.2; spec 9.2 "use redistributed credit during eligible training updates", KF-E01). The default can move
  to RUDDER or COCOA only through estimator selection on held-out randomised experiments (§11.3). That selection is
  wired (DX-1), but in the agent as configured it can never compare: each experiment is one randomised decision, and
  `holdout_experiments=False` makes every experiment objective trained-on. So in practice only `simple` credit is
  used, only in objectives whose environment attributes outcomes, and only for rows without a final verified
  annotation. RUDDER / COCOA credit is computed, stored and versioned; the policy update would read it only if the
  selection made one of them the default. The meta-gradient's outer utility reads no credit (§9.2).
* **Downstream-usefulness VOI head**: attached and counted, not trained, not used (§5). The architecture document's
  "learned estimator of downstream usefulness that is periodically scored against realised downstream learning" exists
  as code (`DownstreamValueEstimator`, `score_downstream`) but not in the agent's loop.
* **VOI costs**: `cost_weight = time_weight = 0`, so VOI is gross of cost, time and opportunity cost at default
  settings. `VOICosts` is used only by the reference estimators.
* **Selection uses neither VOI nor imagination**: `thompson_policy` explores through the posterior samples. Imagined
  values (planner) are recorded only.
* **Docstring difference**: `predict/policy.py:selection_probs` describes `thompson_policy` as "weighted by
  beta_q[s], plus beta_pi[s] x the centred log-probability". The code (and `DecisionMaker.thompson_policy`) uses
  `z + lambda c` with `lambda = beta_pi^+ / max(beta_q, 0.05)` and the projected, RMS-bounded preference `c` of §6.1.
* **Score term**: implemented in `reward/meta.py`, inactive with `unroll_events = 1` (§9.1).
* **Trust-region step in the meta-gradient**: treated as a constant (G2D-3). That is exact at the default live mix 1;
  otherwise it is an approximation (§8.3, §13).
* **Checkpoint promotion** is a gate with fixed absolute tolerances (§10.2), not a statistical non-inferiority test.
  The non-inferiority test is the bootstrap-mix rule (§10.1).
* **Late bootstrap inputs**: `epistemic_after` and `learning_progress` are written as annotations after the decision
  (§7.1). A batch built before they exist sees them masked, and `epistemic_after` is never written for a decision
  whose goal-relevant outcomes arrive after its objective ended.
* **Objective segments versus lifetime context**: R_psi's lifetime context is not reset between objectives (KF-E05,
  as the architecture's section 4 requires). The meta-learner's returns, discount, return-to-go chains and independence
  bookkeeping run per objective segment (`MetaLearner.new_objective`, §7.2).
* **Slow-step protection is first-order**: PCGrad and the update projection hold on the slow batch's training
  gradients. Held-out safety comes from the per-provider guard (§12).
* `docs/CAM_LIVE_ARCHITECTURE.md` §3 asks for the world-model uncertainty as a "small bootstrapped ensemble or
  last-layer posterior — state which". Both are used, and they are added as variances of independent factors (§3.2).
