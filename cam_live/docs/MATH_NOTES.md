# CAM Live — mathematical implementation notes (DL-04)

Describes the code at commit `55a0f4c` (read 2026-11-27, 23:12–23:55 UTC). The last commit that changed
`src/cam_live/nn/`, `predict/`, `reward/` or `credit/` before that is `5ad3777`; `src/cam_live/agent.py` was last
changed in `1039a6f`. Other agents were editing `agent.py`, `predict/`, `reward/` and the runtime while this was
written: a later commit can differ from what is stated here.

Every equation is followed by the code that computes it, as `path:function` (paths relative to `cam_live/`). Where the
design documents (`docs/CAM_LIVE_ARCHITECTURE.md`, `docs/A/DESIGN.md`, `docs/B|C|D/API.md`, `docs/C/LRN_S5.md`)
or a docstring say something else, the code is described and the difference is listed in §14. Defaults are the
values in force with `configs/cam_live.yaml` and no further configuration section (`predict`, `reward`, `credit`,
`agent` are absent from that file, so the dataclass defaults apply).

Five formulas were re-evaluated on the CPU at `55a0f4c` (script `docs/evidence/DOC/scripts/math_check.py`, run with
`cd cam_live && ../.venv/bin/python -I docs/evidence/DOC/scripts/math_check.py`; re-run at `5ae3b59` with the same
numbers, output `docs/evidence/DOC/math_check.txt`): the NIG predictive variance of §2.2 (0.585551 by
`FastBeliefState.predict` against 0.585551 by the closed form), the prior of the selection calibration (§6.3:
mean of (beta_q, beta_pi) draws (1.002, 0.0006), mean lambda 0.357, lambda > 0 in 49.9 % of draws), the exploration
floor (§6.1: a one-hot vector becomes 0.985 / 0.005 / 0.005 / 0.005), the return discount (§8.1: beta =
8.0225e-6 s^-1 = ln 2 / 86400) and the scale invariance of the batch standardisation (§7.3: 1.2e-12). Everything else
is from reading the code; nothing here is a new measurement of learning.

## Notation

| symbol | meaning | where it is formed |
|---|---|---|
| `H_t` | stored context items of decision t (experience tokens) | `learning_api.encode_decisions` |
| `b` | belief, output of the experiential core on `H_t` | `nn/model.py:CamLiveModel.core_forward` |
| `e_a`, `e_g`, `e_q` | embeddings of a candidate action, the goal text, an outcome query | `nn/model.py:encode_actions / encode_goals / encode_queries` |
| `z` | decision latent `z = trunk(b, e_a, e_g)` | `nn/model.py:decision_latent` |
| `phi` | fast-layer features `phi = Phi(z, e_q)`, F = 128 | `nn/decision.py:FeatureMap.forward` |
| `w` | last-layer weights of the fast layer, one vector per outcome kind | `nn/fast.py:FastBeliefState` |
| `theta` | adaptable subset: policy head + decision trunk | `reward/inner.py:AdaptableSubset` |
| `psi` | parameters of the intrinsic reward `R_psi` (group `reward`) | `reward/intrinsic.py:IntrinsicReward.psi` |
| `mu(a)` | logged propensity of the executed candidate | `contracts.Decision.probabilities` |
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
| cost weights (money, seconds) in values | 0.0, 0.0 | `predict/heads.py:PredictConfig.cost_weight / time_weight` |
| selection rule | `thompson_policy` | `agent.py:AgentSettings.selection_rule` |
| inner step `alpha`, steps `K` | 1e-2, 2 | `configs/cam_live.yaml` `optim`, `reward/metalearner.py:MetaConfig` |
| live trust region: target KL, step range | 0.05, [1e-2, 2.0] | `reward/metalearner.py:MetaConfig` |
| reward lr (Adam), meta-gradient clip | 1e-4, 0.5 | `config.py:OptimConfig` |
| independent groups per meta-update | 8 | `configs/cam_live.yaml` `optim.meta_min_independent_outcomes` |
| ESS floor, weight clip, propensity floor | max(2, 0.05 n), 10, 1e-6 | `reward/metalearner.py:MetaConfig` |
| held-out fraction of groups (promotion) | 0.25 | `reward/metalearner.py:MetaConfig.heldout_fraction` |
| bootstrap mix: start, step, min groups, margin | 1.0, 0.25, 4, 0.05 x RMS(U) | `reward/metalearner.py:MetaConfig` |
| return half-life | 86 400 s | `reward/inner.py:InnerConfig.half_life_s` |
| credit discount `beta` | 0 | `agent.py:AgentSettings.credit_beta` |
| live policy update cadence / batch | every 4 rewarded decisions / 16 newest | `agent.py:AgentSettings` |
| checkpoint gates (validation drop, retention drop) | 0.5, 0.1 | `agent.py:AgentSettings.promotion_*` |

---

## 2. Fast Bayesian beliefs (`m_t`)

One posterior `w ~ N(m, Sigma)` per outcome kind (real, count, binary, time-to-event); categorical outcomes are not
in the fast layer. Everything is float64. The state is created from the learned prior by
`nn/model.py:CamLiveModel.new_fast_state` and updated without any gradient step by
`predict/predictor.py:Predictor.observe_outcomes` → `nn/fast.py:FastBeliefState.update`.

### 2.1 Real outcome, known noise (Gaussian, exact)

With `s = phi^T Sigma phi`, `S = s + sigma^2`, gain `k = Sigma phi / S`, innovation `e = y - phi^T m`:

    m'     = m + k e
    Sigma' = (I - k phi^T) Sigma (I - k phi^T)^T + sigma^2 k k^T          (Joseph form, symmetrised)
    surprise = 1/2 [ log(2 pi S) + e^2 / S ]
    KL(post || prior) = 1/2 [ -s/S + (e^2/S)(s/S) + log(S / sigma^2) ]

`nn/fast.py:gaussian_update` (`_joseph`, `_quad`). The batch form uses precision and Cholesky factors,
`Lambda = Lambda0 + Phi^T W Phi / sigma^2`, `m = Lambda^-1 (Lambda0 m0 + Phi^T W y / sigma^2)` (`nn/fast.py:gaussian_fit`,
`_prec_from_cov`), and equals the recursion.

### 2.2 Real outcome, inferred noise (normal-inverse-gamma, the default `noise_mode="nig"`)

`w | s2 ~ N(m, s2 V)`, `s2 ~ InvGamma(a, b)`. With `q = phi^T V phi`:

    k = V phi / (1 + q),  m' = m + k e,  V' = Joseph(V, k) + k k^T,
    a' = a + 1/2,         b' = b + e^2 / (2 (1 + q))
    predictive: Student-t(loc = phi^T m, scale^2 = (b/a)(1 + q), dof = 2a)

`nn/fast.py:nig_update`, `predict_nig`, closed-form KL in `nig_update`. The reported moments
(`nn/fast.py:FastBeliefState.predict`) are mean `phi^T m`, epistemic `E[s2] q`, noise `E[s2]` with
`E[s2] = b / (a - 1)` (`_nig_noise_mean`, infinite for `a <= 1`), so epistemic + noise is the Student-t
*variance* `(b/(a-1))(1+q)`. The linear predictor's variance is likewise `(b/(a-1)) phi^T V phi`
(`FastBeliefState.linear`). Thompson draws sample `s2` jointly: `w = m + sqrt(s2) L z`, `s2 = b / Gamma(a)`
(`FastBeliefState.sample_weights`, Cholesky with relative jitter 1e-12).

### 2.3 Count, binary, time-to-event (assumed-density / Laplace update)

Linear predictor `eta = phi^T w`, prior on it `N(eta0, s)`, `eta0 = phi^T m`, `s = phi^T Sigma phi`.
Likelihoods (`nn/fast.py:_glm_terms`, `_glm_loglik`):

    count          c ~ Poisson(e exp(eta))                    e = exposure
    binary         c ~ Binomial(n, sigmoid(eta))              n = trials (exposure argument)
    time-to-event  delta eta - (t/u) exp(eta)                 Poisson-exposure identity: delta = 1 - censored,
                                                              exposure t/u (u = time unit); right-censoring exact

The posterior mode lies on the line `m + Sigma phi c`, so the update solves a scalar strictly concave problem

    eta* = argmax  loglik(eta) / d  -  (eta - eta0)^2 / (2 s)          d = count dispersion (1 for binary / tte)

by a monotone safeguarded Newton search (Poisson family: Newton on `r(x) = c - e exp(x) - (d/s)(x - eta0)`, which
lands right of the root from the left, and from the right the smaller of the Newton steps on `r` and on its log
form `G(x) = log(e exp(x)) - log(c + (d/s)(eta0 - x))`, so the iterates decrease monotonically to the root;
binomial: bracketed Newton with the two tail forms of the score) until `|step| <= 1e-11 (1 + |eta|)`,
at most 100 iterations (`nn/fast.py:_mode_poisson`, `_mode_binomial`, `_laplace_1d`; constants `NEWTON_TOL`,
`NEWTON_MAX_ITER`). One further Newton step is taken with the graph, so `eta*` carries the implicit-function
gradient. With curvature `h = -loglik''(eta*) / d`, `delta = eta* - eta0`:

    m'     = m + Sigma phi (delta / s)
    Sigma' = Joseph(Sigma, k = Sigma phi h / (1 + h s)) + (h / (1 + h s)^2) (Sigma phi)(Sigma phi)^T
    KL     = 1/2 [ -h s / (1 + h s) + delta^2 / s + log(1 + h s) ]

`nn/fast.py:glm_update`. The surprise is the predictive NLL of the *untempered* likelihood,
`-log ∫ p(y | eta) N(eta; eta0, s) d eta`, by 24-point Gauss–Hermite quadrature centred on the Laplace
approximation `N(eta*, s / (1 + h s))` of the integrand (`nn/fast.py:_adaptive_gh_nll`, `nll_glm`). Predictive
moments (`nn/fast.py:predict_glm`): count mean `e exp(eta0 + s/2)`, epistemic `mean^2 (exp(s) - 1)`, aleatoric
`mean` (times the dispersion, `FastBeliefState.predict`); time-to-event mean `u exp(-eta0 + s/2)`; binary mean and
variance by Gauss–Hermite over `eta`. Impossible observations (negative count or time, successes outside
`[0, n]`, positive count at zero exposure, censoring flag not 0/1, non-finite) raise before anything is absorbed
(`nn/fast.py:check_observations`). The change-detection residual uses predictive moments
(`z2 = (c - E c)^2 / Var c`, probit approximation `sigmoid(eta0 / sqrt(1 + pi s / 8))` for binary).

How the predictor feeds it (`predict/predictor.py:Predictor.observe_outcomes`): a count is absorbed with
exposure = the head's predicted mean count (the fast layer learns a multiplicative correction of the head); a real
value as the residual `(y - loc_metric)/scale_metric - head_loc` (normalised units); a time to event as `y / head_time`
(proportional correction); a binary outcome as the raw 0/1 value (no head offset, see §14).

### 2.4 Changing mechanisms

Stabilised forgetting before every update (when `gamma < 1` and the state is not fresh):

    Lambda <- gamma Lambda + (1 - gamma) Lambda0,    Lambda m <- gamma Lambda m + (1 - gamma) Lambda0 m0
    (NIG: a <- gamma a + (1 - gamma) a0,  b likewise)

`nn/fast.py:forget`, `FastBeliefState._flatten`. It is the geometric mixture `p^gamma prior^(1-gamma)`, bounded by the
prior in every direction. CUSUM on the standardised surprise of the pre-update belief:

    C <- max(0, C + min(z2, 9) - 1 - k);   C > h  =>  forget once with gamma = 0.1, reset C, re-apply the datum

`nn/fast.py:FastBeliefState.update`. Defaults `gamma = 0.999`, `k = 1`, `h = 12`.

### 2.5 Learned prior (ALPaCA-style)

`FastPrior` holds per kind `m0` and a Cholesky factor `L0` (softplus diagonal floored at 1e-6), the log noise
variance, `a0 = 1 + softplus(raw)` and the count dispersion `d = exp(disp_raw)` (exactly 1 at initialisation);
`prior()` returns `cov = L0 L0^T`, `V = cov / sigma^2`, `b0 = a0 sigma^2` (`nn/fast.py:FastPrior.prior`). Group
`experiential`. It is trained by the prequential (predict-then-update) NLL of a sequence

    L_fast = sum_t -log p(y_t | posterior after y_<t)          (masked rows contribute 0; differentiable in phi and prior)

`nn/fast.py:online_nll`. Two providers use it: `predict/providers.py:FastPriorLoss` (weight 0.5; one sequence per
(objective, kind), rows ordered by observation time, at most 48 per sequence; each kind's term divided by a running
scale, `_balanced`) and the prior-predictive term inside `predict/providers.py:WorldModelLoss.fast_term` (weight
`fast_prior_weight` 0.25, sequences of length 1). The fast-layer targets are the head-offset inputs of §2.3
(`predict/heads.py:fast_inputs`, `head_offsets`).

---

## 3. Prediction distributions

### 3.1 World-model heads

`M = 5` members, each `raw = member_m(x) + prior_m(u)` with a frozen randomised prior network (Osband et al. 2018;
`prior_scale = 1`), `x = SiLU(LN(W_z z + W_q e_q + (W_zg z)(W_qg e_q)/sqrt(h) + W_t tau(horizon) + E_kind + W_phi phi))`
(`predict/world.py:WorldHeads.forward`). Per kind (`predict/world.py:WorldOutput`):

| kind | distribution | parameters |
|---|---|---|
| real | Student-t (or normal) in normalised metric units | `loc`, `scale = softplus(.+0.5413)+1e-3`, `df = 2 + softplus(.+3)` |
| count | negative binomial, mean `exp(log_rate) * exposure`, concentration `r` | `log_rate` clamped ±20, `log r` clamped [-6, 9] |
| binary | Bernoulli | logit |
| time-to-event | discrete hazard on 12 equal bins of `[0, horizon]`, right-censoring | `h_k = sigmoid(logit_k - 2)` |
| categorical | masked softmax over at most 16 classes | logits |

Hazard likelihood (`predict/dists.py:hazard_log_prob`): event in bin k: `log h_k + sum_{j<k} log(1 - h_j)`;
censored at t: `sum over bins completed by t of log(1 - h_j)`; restricted mean `E[min(T, H)]` with events uniform in
their bin (`hazard_expected_time`). Negative binomial in log space (`neg_binomial_log_prob_from_log`).

Training: `predict/world.py:world_nll` — masked NLL of *observed or censored* outcomes of the executed candidate
only (pending / unobservable targets give exactly zero loss and gradient), member m weighted by a deterministic
Poisson(1) bootstrap multiplicity of the record key (`bootstrap_weights`), averaged per kind and divided by
`max(1, EMA|NLL_kind|)` (`KindNormaliser`, momentum 0.95). Targets are float64 (`build_targets`).

Ensemble summary (`WorldOutput.predictive`): mean = average of member means; **epistemic** = variance of member means
(unbiased); **aleatoric** = mean of member variances (NB: `mu + mu^2 / r`; Student-t: `scale^2 df/(df-2)`).

### 3.2 Combination with the fast layer (the prediction the agent registers)

`predict/predictor.py:Predictor._means`, per query (head members `m`, fast posterior moments `(f_m, f_v)`):

    real   mean = (mean_m(loc_m) + f_m) * scale + loc,      epi = (var_m(loc_m) + f_v) * scale^2
    count  mean = h_m * f_m,                                epi = h_v f_m^2 + h_m^2 f_v + h_v f_v      (independent factors)
    binary p = mean_m sigmoid(logit_m + shift),  shift = E_fast[eta] - E_ref[eta],
           epi = var_m(sigmoid(.)) + (p(1-p))^2 Var_fast[eta]
    tte    mean = t_m * f_m,                                epi as for counts
    categorical: head only

`ref` is the fast state at the start of the lifetime (`Predictor.set_fast_state`). The registered distribution
(`Predictor._distribution`) moment-matches: count → NB with total variance `alea + epi`; real → Student-t / normal with
`scale^2 = mean_m scale_m^2 + epi / s^2`; binary → Bernoulli(p); time-to-event → the exact member mixture of the
hazard pmf with proportional-hazards correction `h' = 1 - (1 - h)^ratio`, `ratio = exp(E_fast[eta])` (clipped ±10).
Every `PredictionRecord` carries `epistemic_std` separately; it is registered in the hash-chained registry before
`evaluate_candidates` returns (`predict/registry.py`).

Joint posterior samples (for Thompson): sample s draws one ensemble member index and one fast-weight vector per kind,
both shared by all candidates of the decision (`Predictor._samples`); the candidate's goal value per sample is the
sum over its goal-relevant queries of the longest horizon (`Predictor._goal_mask`), stored as `extras["ts_<s>"]`.

### 3.3 Goal evaluator, delta V, action quality

`predict/evaluator.py:GoalEvaluator` gives `V(b, g, horizon)` and `Q(z, g, horizon)` read-outs: success logit,
magnitude (NB or Student-t, scaled by the objective scale), time-to-achievement (12-bin hazard, censoring-aware),
resources (normal on `log1p(amount/scale)`). Loss (`evaluator_loss`): `1.0 BCE(success) + 0.25 NLL(mag) +
0.25 NLL(time) + 0.25 NLL(resources)`, masked, each prefix weighted `1/(#valid prefixes)`. Targets come only from
signature-checked objective outcomes, applied to every earlier prefix: achieved → success 1 and time as an event;
failed → success 0 and time right-censored at the remaining horizon; pending → all masks false; prefixes formed at or
after the evaluation are masked (`prefix_targets`).

    Delta V = sigmoid(logit V(b', g)) - sigmoid(logit V(b, g))        (a reward-generator input, detached; not a reward)
    action quality = q_chosen - sum_i pi_i q_i,   regret = max_i q_i - q_chosen

`predict/evaluator.py:delta_v`, `action_quality`; signal assembly in `reward/assemble.py` (success_before is the
state value `V(b)` recorded at decision time, the same function as success_after).

---

## 4. Calibration metrics

Only prospective predictions are scored: `made_at < observed_at` strictly (`predict/calibration.py:prospective_guard`,
`score_predictions`).

    Brier = (p - y)^2;   NLL = -[y log p + (1-y) log(1-p)]   (p clipped to [1e-12, 1 - 1e-12])
    recalibration: y ~ sigmoid(alpha + beta logit p)  by Newton (perfect: alpha = 0, beta = 1)
    ECE = sum_b (n_b / n) |mean p_b - freq_b|  (10 equal-width bins)

`predict/calibration.py:brier`, `binary_nll`, `logistic_recalibration`, `reliability_bins`, `ece`. Intervals:
Student-t over cluster means for Brier / NLL (`studentised_ci`); slope / intercept with the CR1 cluster-robust
sandwich (G - 1 df) when clusters are given, otherwise a Newey–West HAC sandwich (Bartlett kernel, Andrews AR(1)
bandwidth `1.1447 (alpha n)^(1/3)`) in row order (`_hac_meat`); ECE by a cluster bootstrap (`ece_ci`). Reports grouped
by objective family and horizon bucket (edges 1 h, 1 d, 7 d, 30 d; `grouped_binary_report`).

---

## 5. Value of information

Definition used at decision time (`predict/predictor.py:Predictor._voi`): the 16 candidates with the highest total
epistemic variance are taken (infeasible ones and those without fast-layer queries are then skipped, not replaced);
for each such candidate i, over `S_v = 8` samples:

    for s = 1..S_v:  f_s = clone(fast state)
        for each of i's queries j:  w ~ f_s posterior;  lin = phi_ij^T w;  draw y from the likelihood:
            count Poisson(hc exp(lin)), hc = head mean count;  real lin + sqrt(noise) N(0,1) (NIG: plug-in b/a);
            binary Bernoulli(sigmoid(lin));  tte -log(u) exp(-lin)
            f_s.update(phi_ij, y)                                     (the fast layer's own update, §2)
        Q_s(a') = sum_goal-queries combined mean under f_s (§3.2) - net_cost(a')   for every candidate a'
    VOI_i = max(0,  mean_s max_a' Q_s(a')  -  max_a' mean_s Q_s(a'))          (E[max] - max E)

The baseline is the maximum of the sample means, not `max_a' Q(b, a')`: for an exact Bayesian update the two agree
(the posterior mean is a martingale), and the assumed-density update is not one, so taking `max Q(b)` would credit
the update's bias as information; that variant is kept as `voi_vs_current`. Also recorded: `voi_stderr`
(`sd_s(max_a' Q_s) / sqrt(S_v)`), `info_gain_nats` (mean belief-change KL), `voi_net = VOI - net_cost`.
`net_cost = cost_weight * money + time_weight * seconds`, both weights 0 by default, so cost, time and opportunity
cost do not enter at default settings. A prediction error alone has no value here: a noisy observation that cannot
change the best candidate gives `E[max] = max E`.

Exact references (tests only, `tests/predict/test_voi.py`): linear-Gaussian last layer, `E[max_a (m_a + beta_a Z)]`
evaluated exactly on the upper envelope (`predict/voi.py:expected_max_affine`, `voi_linear_gaussian`, Frazier et
al. 2009), knowledge gradient for independent arms (`knowledge_gradient_gaussian`), GLM sampling (`voi_glm`), and
`VOICosts.net = exp(-r dt) E[max] - max + direct - cost - rate dt` (not used by the agent). The learned estimator of
downstream usefulness, `predict/voi.py:DownstreamValueEstimator` (Student-t output, `downstream_loss`), is attached
as head `downstream` (`predict/heads.py:attach_heads`) but at this commit no provider trains it and no selection rule
reads it.

Where VOI is used: as the reward signal `voi` (§7.1) and by rule `voi` (`softmax(q + ig_weight * VOI)`), which is the
goal-free rule (`agent.py:AgentSettings.goal_free_rule`). The default selection rule does not use it.

---

## 6. Selection rule `thompson_policy`

### 6.1 Probabilities

For one decision with feasible set `A`, recorded numbers per candidate (`predict/policy.py:DecisionMaker.scores`):
`q_i = extras["q_goal"] - net_cost_i` (posterior-mean goal value, §3.2), samples `q_{s,i} = extras["ts_s"] - net_cost_i`
(`S = 32` joint samples, §3.2; when absent, normal marginals `q + sd N(0,1)`), policy logits `l_i`.

    z_{s,i} = (q_{s,i} - mean_A q) / sd_A q             (sd over the posterior means; 1 if sd <= 1e-12)    standardised_values
    c^0_i   = log softmax_A(l)_i - mean_A log softmax(l)                                                  centred_log_policy
    c_i     = (c^0_i - kappa zbar_i) / max(1, RMS_A(c^0 - kappa zbar)),
              zbar_i = (q_i - mean q)/sd q,  kappa = <c^0, zbar>_A / <zbar, zbar>_A    (projection removed) policy_preference
    lambda_s = clip( max(beta_pi,s, 0) / max(beta_q,s, 0.05), 0, 10 )                                       policy_weight
    score_{s,i} = z_{s,i} + lambda_s c_i                                                                    thompson_policy_scores
    p_i = (1/S) sum_s 1[i maximises score_s] / #maximisers          (exact ties share the sample)          _tie_share
    p   <- (1 - eps) p + eps / |A|,   eps = 0.02                                                            apply_floor

`predict/policy.py:DecisionMaker.thompson_policy`, `DecisionMaker.probabilities`. The candidate is drawn with
`torch.multinomial(p)` from exactly this vector (`DecisionMaker.select`), so the recorded `Decision.probabilities`
are the true propensities of the sampling mechanism; every feasible candidate has at least `eps / |A|`. (They are a
32-sample Monte-Carlo estimate of the ideal Thompson probability, not that probability.) The diagnostics record the
calibration summary, `policy_share_tv = 1/2 ||p(beta_pi) - p(beta_pi = 0)||_1` (before the floor) and the value-only
distribution.

With probability `experiment_rate = 0.1` and an expected order flip >= 0.2 between two candidates
(`DecisionMaker.propose_experiment`) the agent instead assigns at random over registered arms (rule `randomized`,
uniform over arms, then the floor over the arms; `DecisionMaker.select_experiment`, `agent.py`).

### 6.2 What the projection does

`c` keeps only the part of the policy's preference that is orthogonal (over the decision's feasible candidates) to
the world model's standardised ranking, and is bounded like `z` (divided by its RMS once that exceeds 1 nat). A
uniform policy has `c = 0`; a policy that only repeats the world model has `c = 0`.

### 6.3 Selection calibration (learned weight of the policy)

`predict/policy.py:SelectionCalibration`. For each executed decision whose verified progress `y` is final (the
agent's signed annotation, §8.2), the features at decision time are `z_c` (chosen candidate's `zbar`, winsorised
±4) and `c_c` (its `c`, ±4) (`SelectionCalibration.features`). It fits

    y / s = beta_0 + beta_q z_c + beta_pi c_c + noise,        s = sqrt( sum_k w_k y_k^2 / sum_k w_k ),  w_k = 0.99^(n-1-k)

prior `beta ~ N((0, 1, 0), diag(1, 0.5^2, 0.5^2))`, weighted least squares with the forgetting weights, the noise
variance iterated 3 times as `sigma^2 = (nu + sum w r^2) / (nu + sum w)`, `nu = 2`, floor 1e-6
(`SelectionCalibration.posterior`). Draws `(beta_q, beta_pi) ~ N(mean[1:], cov[1:,1:])` (`sample`). At most 1024
observations; each decision once. It is fed by `reward/metalearner.py:MetaLearner.refresh_selection` and shared
through the policy head (`selection_calibration`). At the prior, `lambda` is positive in about half of the draws
(measured above: mean 0.357), but a near-uniform policy has `c ≈ 0`, so it has no influence until its logits differ.

Other rules (`predict/policy.py:selection_probs`): `policy` softmax(l / T); `thompson` share of samples where `i` is
best; `ucb` argmax(q + sd); `greedy` argmax(q); `voi` (§5). The agent's ablation `policy` + value prior adds
`value_weight * zbar` to the logits (`agent.py:CamLiveAgent._with_value_prior`).

---

## 7. Internal reward `R_psi` and its bootstrap

### 7.1 Signals

Fixed, versioned schema (`reward/signals.py`, version 1.0): 15 scalars — success_before, success_after, delta_v,
action_quality (symlog), info_gain (log1p), voi (symlog), cost_money / cost_seconds / cost_compute_s (log1p),
elapsed_s (log1p), prediction_error_std (log1p), epistemic_before / epistemic_after (log1p),
verified_progress_delta (symlog), context_novelty — plus one-hot result status (6) and failure class. Each value is
what was known at reward time `max(decision ts + duration, newest post item)`; a signal without a source is masked
(value 0, mask 0) (`reward/assemble.py:SignalAssembler`). At this commit no runtime code writes `info_gain`,
`epistemic_after` or `learning_progress` (§13), so those are masked in runtime runs.

### 7.2 `R_psi`

    x_t = W_lat [n(b), n(e_a), n(b') - n(b), n(e_g)] + W_sig [n_s(sig) * mask, mask]
    h_{t+1} = GRU(x_t, h_t)                       (lifetime context; reset only at a new lifetime)
    R_psi  = B tanh( out(SiLU(W SiLU(x_t + W_life h_t))) / B ),   B = 1

`reward/intrinsic.py:IntrinsicReward.step_features`, `head`, `reward_sequence`, `LifetimeContext`;
`reward/integrity.py:bound_reward`. `n(.)` are running mean / variance normalisers (float64, clip ±8,
`RunningNorm`) updated only by `IntrinsicReward.observe` on real decisions and frozen inside a meta window
(`meta_window`); the statistics in force when an inner batch was built are stored with it and passed to the
functional call (`reward/inner.py:InnerBatch.norm_state`), so the differentiated reward is the one the update used.
The goal-free mode uses a learned null goal. `version` = hash of parameters and normaliser statistics. `psi` is
exactly `IntrinsicReward.parameters()` (`meta_parameters`); only the MetaLearner's Adam holds them
(`reward/integrity.py:assert_reward_optimizer_exclusive`).

### 7.3 Bootstrap reward and mixing

    r_boot = 4 tanh( [ 1.0 tanh(LP) + 1.0 Delta V + 0.5 g_ctx * clip((eps_b - eps_a)^+ / eps_b, 0, 1)
                       + 2.0 symlog(v) * mask_v ] / 4 )

`reward/bootstrap.py:BootstrapReward.__call__` (weights `BootstrapWeights`; bound 4 set in
`reward/inner.py:PolicyImprovementLoss.__init__`). `LP` learning progress (loss before − after on the same targets),
`eps_b/eps_a` epistemic std before / after, `v` verified progress already observed, `g_ctx` the noisy-TV gate:
per context key, over the last 6 epistemic values, `g <- g + (1 - g) 0.5` while they fall by >= 5 %, else `g <- 0.6 g`
(`reward/integrity.py:NoisyTVGate`). Labelled `bootstrap`, never meta-learned.

Mixing inside the inner loss (`reward/inner.py:PolicyImprovementLoss.intrinsic`), with `n(x) = (x - mean_batch x) /
sqrt(var_batch x + 1e-12)` (`reward/integrity.py:scale_normalise`):

    r_i = n(r_boot)_i                                   if mix >= 1
        = n(R_psi)_i                                    if mix <= 0
        = mix n(r_boot)_i + (1 - mix) n(R_psi)_i         otherwise
        + c_g u_i / rms(u over rows with a value)        goal term, c_g = 1 (§8.2)

Because each component is standardised first, `R_psi -> alpha R_psi + beta` (alpha > 0) changes nothing at any
mix; psi acts only through the shape of `R_psi`, with weight `1 - mix`. The value stored on the record
(`ExperienceRecord.intrinsic_reward`) is the unstandardised `mix r_boot + (1 - mix) R_psi`
(`reward/bootstrap.py:RewardMixer.combine`, called once per real decision by `PolicyImprovementLoss.reward_step`).
The record label is `bootstrap` until at least one meta-update was applied **and** a promotion lowered the mix
(`RewardMixer.label`).

---

## 8. Policy improvement: the inner loss and the live update

### 8.1 Inner loss

Rows are the support decisions in lifetime order; latents `b, e_a, e_g, b'` are cached constants
(`encode_decisions(grad=False)`), so nothing reaches the core or the backbone. With `pi_theta` evaluated through
`torch.func.functional_call` on the trunk and policy head (`reward/inner.py:AdaptableSubset.forward`):

    G_i  = r_i + gamma_i G_{i+1},   gamma_i = exp(-beta (t_{i+1} - t_i)) within a lifetime, 0 at its end,
           beta = ln 2 / 86400 s                                                    meta.discounted_returns
    Gn   = n(G)                                                                     (standardised over the batch)
    A_i  = Gn_i - V_{theta_sg}(b_i, C_i)                                            (baseline held fixed, see C-9)
    L_inner(theta; psi) = -mean_i log pi_theta(a_i | b_i, C_i) A_i
                          + 0.5 mean_i 1/2 (V_theta - Gn_i)^2
                          - 0.01 mean_i H[pi_theta(. | b_i, C_i)]
                          - 1.0 beta_q^+ mean_{i: ok} sum_j pi_theta(j | i) zbar_ij               (all-candidate term)

`reward/inner.py:PolicyImprovementLoss.functional`, `InnerConfig`. `zbar_ij` is the recorded standardised
predicted value of candidate j (`SelectionCalibration.candidate_values`), `beta_q^+` the posterior mean of the
calibration clipped at 0 (a constant in psi). `theta_sg` are copies of theta that are not variables of the inner
derivative but are still differentiated by the unroll; with plain `.detach()` the path `theta_k(psi) -> V` would be
lost for K >= 2 (change request C-9, measured bias 1.0 % at K = 2, 3.3 % at K = 4).

### 8.2 Goal term `u`

`u_i` = the verified progress attributed to decision i once final (`reward/inner.py:verified_progress`: the
decision's `diagnostics.reward_signals.verified_progress_delta`), otherwise (labelled `predicted`) the calibrated
decision-time prediction `s (beta_0 + beta_q^+ zbar_chosen)` (`SelectionCalibration.predict`), or the raw
`extras["q_goal"]` before the calibration has a scale (`PolicyImprovementLoss._goal_term`). The annotation is written
once, from signed verifier readings only, when every goal-relevant answer of the decision is in
(`agent.py:CamLiveAgent._annotate_verified_delta`): the sum of the decision's own attributed outcomes; 0 if the
environment attributes outcomes and none are attributed to it; otherwise the signed change of the objective metric
across the decision. The goal term is constant in psi.

### 8.3 Inner step and live trust region

    theta_{k+1} = theta_k - alpha clip_group(grad_theta L_inner(theta_k; psi, D)),   k = 0, 1   (K = 2)

`reward/meta.py:MetaGradient.inner_step` with the in-graph per-group norm clip `g <- g min(1, 1/||g||)` separately for
the policy head and the trunk (`reward/inner.py:GroupClip`, `MetaConfig.clip_policy / clip_trunk`). The live update
(`reward/metalearner.py:MetaLearner.policy_update`, called every 4 newly rewarded decisions on the 16 newest plus up to
16 earlier decisions with final verified progress) chooses `alpha` by a trust region on

    KL = mean_rows sum_{a in A} pi_0(a) [log pi_0(a) - log pi_alpha(a)]                         MetaLearner._kl

target 0.05, accepted when `KL in [0.05/1.5, 1.5 x 0.05]`; at most 6 retries; the next step is the geometric mean of
the bracketing steps once both sides are known, else `clamp(alpha sqrt(0.05/KL), 0.01, 2.0)`; stop when the change is
<= 2 %. The chosen `alpha`, the step search, `theta_0` and the support batch are stored as an `UpdateEvent`, so the
meta-gradient differentiates exactly the SGD step that was applied. The update runs under `PsiGuard` (psi must be
bit-identical afterwards).

---

## 9. The meta-gradient (lifetime objective, Zheng et al. 2020)

### 9.1 Objective and estimator

    J(psi) = E[ U_verified(D_query; theta_K(psi)) ]
    surrogate  S(psi) = sum_rows w_i rho_i log pi_{theta_K(psi)}(a_i | x_i) (U_i - b_i)
               [+ (sum_k sum_j log pi_{theta_k}(a_j)) (J_hat - c)   when support step k >= 1 was sampled from pi_{theta_k}]
    dJ/dpsi    = d S / d psi  through the K-step unroll

`reward/meta.py:MetaGradient.inner_update` (create_graph, `mode="unroll"`; `"checkpoint"` gives the same gradient by
an adjoint reverse sweep, `_adjoint`; `"first_order"` drops the inner Hessian chain), `outer_surrogate`,
`meta_gradient`. `theta_0` is a constant (truncated window). In the agent each update event is an episode:
`theta_0` = the event's snapshot, support = its batch with `mix = meta_mix = 0` (the pure learned reward: a *shadow*
update while the live mix is above 0), query = the decisions taken after it (`reward/metalearner.py:MetaLearner._episode`,
`_steps_of`, `meta_update`). With `unroll_events = 1` (default) no support step is flagged on-policy, so the
inner-data score term is not used at default settings.

### 9.2 Query rows, independence, utility

* Windows (`reward/windows.py:WindowBuilder.cut`): support = eligible decisions before the first query decision,
  query = those after; a query row whose independence group already occurs in the support (`straddles_boundary`) or
  that lies inside the gap is dropped. Rows must have propensities and the provenance of the query rows.
* Independence (`windows.py:independence_key`, `WindowBuilder._prepare`): experiment id, else parent decision, else
  the decision; rows that share a verified outcome (`dedup_key`) are merged (union–find). An objective outcome's
  ledger group overrides it for its rows. Groups are counted, never rows.
* Held-out groups: a deterministic hash of the group key with salt `promotion` reserves 25 % of the groups for the
  mix promotion (§10.1) (`windows.py:held_out_groups`).
* Utility (default `utility="return_to_go"`): with objective utilities `U_obj` (change of `objective_utility = 1` if
  achieved else `0.5 clip(value / max(target, 1), 0, 1)` since the previous admitted evaluation) and goal-relevant
  verifier measurements `m_j` (their measured value),

      U_a = U_obj(a) + sum_{j >= a, same lifetime, same held-out side} exp(-beta (t_j - t_a)) m_j

  `reward/metalearner.py:_episode`; `reward/integrity.py:objective_utility`, `measurement_utility`. Numbers come from
  the hash-chained outcome ledger, derived from signed records; a disagreeing caller utility is refused
  (`OutcomeLedger.admit_verified`, `utility_matches`). Only objectives with a recorded operator / verifier origin
  supervise. A row whose goal-relevant answers are not all in stops the return-to-go chain there.
* Baseline (`MetaLearner.outer_baselines`): mean utility of the rows outside the row's baseline group — with
  return-to-go the whole chain (lifetime, held-out side), because every earlier row's `U` contains this row's
  outcome — else the running mean of verified utility at the row's registration. It does not depend on the row's
  own action or outcome.

### 9.3 Importance weighting and the ESS rule

On-policy (`rho = 1`) only if every event of the episode was applied by `policy_update`, replaying it without a
graph from `theta_0` with the current psi, statistics and mix gives parameters bit-identical to those the live update
left (`MetaLearner.logged_policy_is_current`), and `max |pi_{theta_K}(a) - mu(a)| <= 1e-3`. Otherwise (always during
the cold start, because the shadow mix 0 differs from the live mix):

    rho_i = pi_{theta_K}(a_i) / mu(a_i)          (from the logged log-propensity; refused if missing, > 1, or mu < 1e-6)
    ESS   = (sum rho)^2 / sum rho^2              per lifetime of the query batch (the agent passes one per episode)
    refuse if ESS < max(2, 0.05 n)               -> "effective_sample_size_too_small"
    w_i   = (1/N) min(rho_i, 10)                 (N = all outer rows of the meta-update; not self-normalised)

`reward/meta.py:MetaGradient.outer_surrogate`. Gradients of all supported episodes are summed; then
(`MetaLearner.meta_update`, `MetaGradient.apply`) the step is refused if the norm is not finite, if every advantage
`|U - b| <= 1e-9 max(1, max|U|)` (`zero_advantage`: rows stay available), or if the norm is <= `min_meta_grad_norm`
(0); otherwise one Adam ascent step on `-g * min(1, 0.5/||g||)` (lr 1e-4). If psi is bit-unchanged the optimizer
state is restored. Skip reasons recorded in `MetaUpdateRecord.skipped_reason`: `no_policy_update_recorded`,
`insufficient_independent_outcomes:n<8`, `no_supported_episode:<reason>` (reasons of the engine:
`empty_or_misaligned_query`, `non_finite_utility`, `unverified_outcome_in_query`, `missing_propensities`,
`invalid_propensities`, `propensity_below_floor`, `effective_sample_size_too_small`, `inner_update_has_no_graph`),
`non_finite_meta_gradient`, `zero_advantage`, `zero_meta_gradient`, `psi_unchanged_by_the_step`, `not_applied:*`.

### 9.4 Exact reference

`reward/exact.py` implements the tabular setting of Zheng et al. 2020 (`r_psi[t, s, a]`, `theta_{k+1} = theta_k +
alpha grad J_int`, `J(psi) = J_ext(theta_K(psi))`) three ways: autograd through exact updates, central finite
differences, and the sampled REINFORCE estimator with the inner-data score term and leave-one-out baselines; the
sampled objective is also enumerated exactly. Tests: `tests/reward/test_reward_exact.py`
(`test_exact_meta_gradient_equals_finite_differences[K]`, `test_cutting_the_pathway_gives_exactly_zero[K]`,
`test_sampled_estimator_is_unbiased_by_z_score_and_relative_error`), `tests/reward/test_reward_meta.py`
(`test_meta_gradient_matches_finite_differences`, `test_first_order_approximation_is_exact_for_one_step_and_biased_beyond`,
`test_sampled_on_policy_estimator_with_score_term_is_unbiased`), `tests/reward/test_reward_inner.py::
test_meta_gradient_matches_finite_differences_on_the_real_adaptable_subset`.

---

## 10. Promotion

### 10.1 Bootstrap mix: a non-inferiority test

For each held-out group g (`MetaLearner._promotion`, `reward/bootstrap.py:paired_policy_value_diffs`): from the same
snapshot and support, `theta_l` = inner update with the learned reward (mix 0), `theta_b` = with the bootstrap
(mix 1); on the group's logged query rows

    d_g = mean_rows [ min(pi_l(a)/mu(a), 10) - min(pi_b(a)/mu(a), 10) ] U

Differences are buffered until >= max(2, 4) groups; then with mean `dbar`, `se = sd / sqrt(n)`, one-sided 95 %
Student-t quantile `t_{n-1}` and margin `delta = 0 + 0.05 RMS(U)`:

    lower  (mix <- max(0, mix - 0.25))   if dbar - t se >= -delta         (learned reward not worse by more than delta)
    raise  (mix <- min(1, mix + 0.25))   if dbar + t se < 0               (shown harm)
    hold                                  otherwise, or fewer than 4 groups, or non-finite

`reward/bootstrap.py:RewardMixer.update`.

### 10.2 Checkpoint promotion: gates, not a test

`credit/checkpoint.py:CheckpointManager.promote` with the agent's `PromotionGates(max_validation_drop=0.5,
max_retention_drop=0.1)`: promoted iff every independent validation metric (held-out provider losses, negated,
`credit/retention.py:validation_metrics`) satisfies `new >= old - 0.5` and every retention metric `new >= old - 0.1`
relative to the last promoted checkpoint, all finite, at least one of each kind; metrics whose provenance is not in
`INDEPENDENT_PROVENANCE` (e.g. the agent's own reward) are ignored. There is no primary metric and no confidence
bound. Called every 24 frequent updates and at the end of each objective (`agent.py:CamLiveAgent.promote`). A refusal
restores the last promoted learner checkpoint except `R_psi` (whole module, with the meta optimizer) and — unless
`decision_agreement` is among the reasons — the policy head (LRN-6).

---

## 11. Retrospective credit

Credit is recomputed for an objective when its outcomes mature (`credit/providers.py:CreditAssigner.assign`) and
stored per record with a version history (`credit/replay.py`). Simple attribution: `credit = sum_links weight x value
x exp(-beta (t_outcome - t_decision))`, declared answers first (`credit/simple.py:credit_from_links`); weights from
evidence links, ambiguous evidence split with uncertainty, negative / malformed amounts treated as ambiguous
(`credit/links.py:attribution_amounts`).

### 11.1 RUDDER

`F_xi` is a causal transformer over per-decision latents (+ time features `log1p(tau/60)`, `log1p(dt/60)`, goal)
predicting `G / scale` from every prefix; `pred[0] = F(empty)` (`credit/rudder.py:ReturnDecomposer.forward`).

    loss  = mean over prefixes of matured trajectories (F(H_<t) - G/scale)^2          (linear target; scale = max |G|)
    Gamma_t = exp(-beta tau_t),  tau_t = time since the trajectory's first decision
    c_t = [F(H_<=t) - F(H_<t)] / Gamma_t  (+ F(empty)/Gamma_first on the first, + [G - F(H_<=T)]/Gamma_last on the last)
    sum_t Gamma_t c_t = G   exactly (conservation, for any F)
    learning signal_t = [G - F(H_<t)] / Gamma_t = (sum_{k>=t} Gamma_k c_k) / Gamma_t     (redistributed return-to-go)

`credit/rudder.py:ReturnDecomposer.loss`, `redistribute`, `Redistribution.conservation_error`. Unmatured trajectories
contribute no loss and get NaN credit. Stored as `credit["rudder"]` (signal) and `credit["rudder_contribution"]`.

### 11.2 COCOA

Hindsight model `h(a | b, u, elapsed)` over the recorded candidates, logits = `log pi(a|b) + f(b, u, elapsed, a)` with
`f`'s output layer zero-initialised (so `h = pi`, `w = 0` before training); 4-member bootstrapped ensemble
(`credit/cocoa.py:HindsightModel`). Trained by cross-entropy of the executed candidate given a later outcome
(Poisson(1) bootstrap weights per member).

    w(a, u) = clip( mean_members h_m(a | b, u) / pi(a) - 1,  -1, 10 )
    credit(a_t) = own outcomes of t (discounted) + sum_{later outcomes k} w(b_t, a_t, u_k) v_k exp(-beta (tau_k - t))

`credit/cocoa.py:HindsightModel.contribution`; `credit/providers.py:HindsightProvider.credit`, `_pairs` (an outcome
pairs with t if t did not declare it and it was observed after t). Refusal (`credit/cocoa.py:check_support`, value
NaN, status `unsupported`): propensities missing; NaN or > 1 or not summing to 1 (tolerance 1e-3); any candidate
below `min_prob` (1e-3); open-ended candidate set without tracked proposals (backbone-proposed sets); no candidates;
insufficient per-candidate ESS when required. Association `h - pi`, counterfactual estimate `w v` and experimental
effect (randomised arms only) are separate fields.

### 11.3 Which credit trains

`CreditAssigner.default` is chosen by `credit/select.py:EstimatorSelector` (held-out randomised experiments, paired
z-scores); it starts as `simple`. At this commit no training loss reads the stored credit: the policy update uses the
intrinsic return plus the goal term (§8), and the meta-gradient uses the return-to-go of §9.2. The credit values are
computed, versioned and logged (see §13).

---

## 12. Continual learner and retention

* Frequent update (`credit/learner.py:ContinualLearner._frequent_update`): each provider's gradient is taken
  separately; if the largest provider's share of the experiential gradient norm exceeds 0.9 it is scaled by
  `0.9 rest / (0.1 ||g_top||)`; per-group norm clip (experiential 1.0, heads 1.0, adapters 0.5; `config.py:OptimConfig`);
  one AdamW per group (weight decay 0.01 on matrices), linear warm-up over 8 steps. Triggered by >= 4 matured
  independent outcome units (`LearnerSettings.min_independent_outcomes`).
* Slow update (adapters, every 8 frequent updates): kept only if the mean paired change `d` of the held-out loss over
  8 chunks satisfies `d <= min(0.01, 0.05 |L_before|, max(2 se(d), 1e-7 max(1, |L_before|)))` and the summed change
  since the last promoted checkpoint stays <= 0.02; otherwise weights, AdamW state, schedule and counters are restored
  (`credit/learner.py:ContinualLearner._guard_verdict`, `_validation_cells`, `set_slow_reference`).
* Anchor-set functional regularisation (`credit/retention.py:RetentionProvider`, `AnchorSet.penalty`): at the end of an
  objective the last 64 records become anchors with the model's outputs then (`b`, `z_chosen`); each frequent update
  draws 8 anchors evenly over families and adds `mean_anchors (f(x) - f_ref(x))^2 / mean(f_ref^2)` per function
  (KL(ref || current) for probability functions), weight 1.
* Retention sets and `decision_agreement` (`agent.py:CamLiveAgent.retention_metrics`): the first 24 executed records
  of every objective, frozen with the policy's action distributions at that moment; metrics
  `<family>@<objective>/neg_world_loss` (negated world-model loss on their realised outcomes) and
  `/decision_agreement = (1/n) sum 1[argmax p_ref = argmax p_now]`, provenance `retention`; they feed §10.2.

---

## 13. Known numerical caveats

Status at `55a0f4c` from reading the code (and, where stated, the fix documents' probe results); "not re-checked"
means I did not verify it at this commit.

| caveat | source | at `55a0f4c` |
|---|---|---|
| GLM updates were not the Laplace update far from the prior predictor (fixed 6–12 Newton steps; rate off ×1.9–×339) | `docs/verification/G_nn.md` D1 | **fixed**: converged monotone search, tolerance 1e-11, cap 100, convergence returned (`nn/fast.py:_laplace_1d`); fix probe in `docs/A/DESIGN.md` |
| count dispersion tempered binary / time-to-event | G_nn D2 | **fixed**: dispersion only for counts (`FastBeliefState._apply`, `online_nll`) |
| bf16 LM head: generation log-probabilities accurate to 0.1–0.3 nats / token | G_nn D3 | **partly**: float32 head; bf16 trunk still −0.36…+0.12 nats on short completions (`docs/A/DESIGN.md`); not re-checked |
| NIG `predict()` gave `(b/a)(1+q)`, under-stating the variance by `a/(a-1)` | G_nn D4 | **fixed** (re-evaluated above). Consequence: a Gaussian interval from `predict()` now matches the Student-t variance, not its scale |
| impossible observations absorbed | G_nn D5 | **fixed** (`check_observations`) |
| ill-conditioned features on realistic inputs (Gram max/min 2.3e4 at init) | G_nn D6 | **open** (initialisation property; the float64 Joseph updates stay exact) |
| text → belief depended on batch composition (1 %) | G_nn D8, G_cl m7 | fixed per `docs/A/DESIGN.md` (fixed-shape forwards); not re-checked |
| return-to-go baseline depended on the row's own outcome (biased outer gradient) | `docs/verification/G_pr.md` D1 | **fixed**: baseline group = whole chain (`MetaLearner.outer_baselines`); `docs/C/FIXES.md` |
| caller-chosen utility admitted for a signed outcome | G_pr D2 | **fixed**: derived in the ledger, mismatch refused |
| calibration slope / intercept intervals ignored clustering | G_pr D5 | **fixed**: CR1 / HAC (`logistic_recalibration`) |
| cold-start shadow update switched to on-policy by tolerance | G_pr D7 | **fixed**: structural bit-identity check first (`logged_policy_is_current`) |
| world targets stored in float32 | G_pr D11 | **fixed**: float64 (`build_targets`) |
| `v.detach()` baseline dropped a meta path for K >= 2 (1.0 % / 3.3 % bias) | `docs/change_requests/C.md` C-9 | **fixed** (`theta_sg`) |
| negative attribution amounts created credit > outcome | `docs/verification/G_cl.md` M3 | **fixed** (`links.attribution_amounts`) |
| slow-update guard admitted degrading steps; reverted step kept Adam state | G_cl M4, m4 | **fixed** (noise-based tolerance + drift bound; full restore) |
| RUDDER window discount referenced at the wrong time | G_cl m3 | fixed per `docs/D/FIXES.md`; not re-checked |
| verified bootstrap term saturated at bound 1 (1 and 42 adopters gave the same reward) | `docs/change_requests/INT2.md` INT2-6 | **addressed**: bound 4 and the unbounded goal term `u / rms(u)` (§7.3, §8.2); `4 tanh(x/4)` still compresses large values |
| live policy step too small to change behaviour (fixed `alpha = 1e-2`) | INT2-7 | **addressed**: trust region, target KL 0.05 (§8.3); not re-measured on the real configuration by INT2 |
| support batches rebuilt from the store after a restore (different meta-step) | INT2-8 | **fixed** (5ad3777: batches of events still to be differentiated are checkpointed) |
| selection probabilities drift ~1e-6 after reload | `docs/change_requests/LRN.md` LRN-5 | withdrawn: 0.0 after the INT2-8 fix (LRN's measurement) |
| rollback after a refused promotion reverted psi and the policy head | LRN-6 | **applied** in `agent.py:CamLiveAgent.promote` |
| bootstrap inputs `epistemic_after`, `learning_progress` never written: exploration and learning-progress terms are 0 in runtime runs | LRN-3 | **open** (no producer in `agent.py` / `predict/`) |
| binary fast layer has no head offset: the predictor adds the fast posterior's *shift* from the lifetime-start state to the head logit (approximation) | `docs/change_requests/B.md` B-CR-10 (1) | **open** (`Predictor._means`, `_samples`) |
| fast posteriors are per kind and independent (a real reading cannot inform a count posterior; its fast-layer VOI is 0) | B-CR-10 (2) | **open** (by design) |
| reward signal `info_gain` has no producer: the assembler reads `extras["info_gain"]`, the predictor writes `info_gain_nats` (a different, prospective quantity) | code reading for this document | **open**: the signal is masked in runtime runs |
| Thompson probabilities are a 32-sample estimate; before the floor a candidate can have probability exactly 0 | `predict/policy.py:_tie_share` | by design; the floor gives every feasible candidate >= 0.02 / |A| |
| meta-update evidence is scarce: one lifetime is one independent group for objective utilities; ESS >= 2 per episode; 8 groups needed | `docs/INTEGRATION.md` §8, §9 e | **open** (by design); real staging run: 0 applied in 4 lifetimes of 7 decisions (INTEGRATION §7.6) |

---

## 14. Not implemented, or different from the design / documentation

* **Credit does not train anything yet** (§11.3): simple / RUDDER / COCOA credit is computed and stored, but no loss
  reads `ExperienceRecord.credit`; spec 9.2 "use redistributed credit during eligible training updates" is not met
  (also noted in `docs/verification/G_cl.md`, Observations).
* **Downstream-usefulness VOI head**: attached and counted, not trained, not used (§5). The architecture document's
  "learned estimator of downstream usefulness that is periodically scored against realised downstream learning" is
  implemented as code (`DownstreamValueEstimator`, `score_downstream`) but not in the agent's loop.
* **VOI costs**: `cost_weight = time_weight = 0`, so VOI is gross of cost, time and opportunity cost at default
  settings; `VOICosts` is used only by the reference estimators.
* **Selection uses neither VOI nor imagination**: `thompson_policy` explores through the posterior samples; imagined
  values (planner) are recorded only.
* **Docstring difference**: `predict/policy.py:selection_probs` describes `thompson_policy` as "weighted by
  beta_q[s], plus beta_pi[s] x the centred log-probability"; the code (and `DecisionMaker.thompson_policy`) uses
  `z + lambda c` with `lambda = beta_pi^+ / max(beta_q, 0.05)` and the projected, RMS-bounded preference `c` of §6.1.
* **Score term**: implemented in `reward/meta.py`, inactive with `unroll_events = 1` (§9.1).
* **Checkpoint promotion** is a gate with fixed absolute tolerances (§10.2), not a statistical non-inferiority test;
  the non-inferiority test is the bootstrap-mix rule (§10.1).
* **Bootstrap inputs** `learning_progress` and `epistemic_after`, and the signal `info_gain`, are not produced in the
  runtime (§13), so the bootstrap reward is in practice `Delta V + 2 symlog(verified progress)` bounded by 4.
* `docs/CAM_LIVE_ARCHITECTURE.md` §3 calls the world-model uncertainty "small bootstrapped ensemble or last-layer
  posterior — state which": both are used and added as variances of independent factors (§3.2).
