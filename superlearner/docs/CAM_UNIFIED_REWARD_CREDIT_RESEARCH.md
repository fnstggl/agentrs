# CAM Unified — research note: learned intrinsic rewards, RUDDER, COCOA, HCA, R-HCA

Workstream H (independent audit, research track), 2026-11-26. Required by `CAM_UNIFIED_SPEC_META_REWARDS.md` Part 1
("Create a technical research note explaining which mathematical components are adopted, which are adapted, and which
assumptions are not valid in CAM's open-ended partially observable environments") and manifest items IR-03..IR-08,
CR-02..CR-09, EX-C, EX-J.

**What was read.** All five primary sources were obtained in full and read from their text (not from summaries):

| # | Source | Obtained | sha256 (first 16) |
|---|---|---|---|
| 1 | Zheng, Oh, Hessel, Xu, Kroiss, van Hasselt, Silver, Singh. *What Can Learned Intrinsic Rewards Capture?* ICML 2020 (PMLR 119) + supplement (derivation, hyperparameters, JAX pseudocode) | proceedings.mlr.press PDF, 11 + 7 pp. | 165761d51a87f837 / 65c685270f0dbdd6 |
| 2 | Arjona-Medina, Gillhofer, Widrich, Unterthiner, Brandstetter, Hochreiter. *RUDDER: Return Decomposition for Delayed Rewards* (arXiv 1806.07857, NeurIPS 2019 version with 93-page appendix) + github ml-jku/rudder (README only links: rudder-demonstration-code, rudder-a-practical-tutorial) | arXiv PDF, 102 pp. | 2b31a83bf2c30954 |
| 3 | Meulemans, Schug, Kobayashi, Daw, Wayne. *Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis* (COCOA), NeurIPS 2023, arXiv 2306.16803v2 incl. appendices A–M + github seijin-kobayashi/cocoa (`ccoa/contribution/causal.py`, `modules/contribution_coefficient.py` read) | arXiv PDF, 51 pp. | 9ea0bb072999d49f |
| 4 | Ramesh, He, Schmidhuber, White. *Improving Reward-Based Hindsight Credit Assignment* (R-HCA), RLJ vol. 7 2026 (RLC 2026 pre-proceedings) incl. supplementary App. A–I + github Aditya-Ramesh-10/Reward-HCA (`agents/hca.py` read) | rlj.cs.umass.edu/2026/papers/Paper133.pdf, 24 pp. | f0eb37ddcecb5c69 |
| 5 | Harutyunyan, Dabney, Mesnard, Azar, Piot, Heess, van Hasselt, Wayne, Singh, Precup, Munos. *Hindsight Credit Assignment* (HCA), NeurIPS 2019, arXiv 1912.02503 incl. appendices | arXiv PDF, 18 pp. | 917be0fd3c22faee |

Nothing about R-HCA had to be guessed: the paper gives the estimator (its Eq. 4), the proofs, the algorithm
(App. G, Algorithm 1), the network (App. F.1), all hyperparameters (App. F) and a public implementation. Where this
note states something the papers do **not** state, it is marked **[derived here]** and was checked numerically.

**What was executed** (scratch, `/tmp/claude-0/audit_research/scripts/`, CPU, float64):

* `credit_exact.py` — exact enumeration (128 trajectories) of a noisy key-to-door MDP; exact expectation and exact
  variance of 11 estimators (REINFORCE, advantage, Q-critic, HCA+, COCOA-reward, R-HCA, HCA-return, 4 RUDDER uses)
  against the exact gradient; closed-form checks of the COCOA App. L and R-HCA App. E counterexamples (§8.1–8.2).
* `metagrad_exact.py` — exact lifetime meta-gradient of a tabular intrinsic reward through stochastic inner
  REINFORCE updates (3^5 × 2 histories enumerated), vs finite differences, Zheng's Eq. 4 Monte-Carlo estimator,
  truncated backpropagation and the episodic ablation (§8.5).
* `audit_meta_sampled.py` — bias of the project's committed `credit/meta_exact.sampled_meta_gradient` (§10).

---------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. All five methods are mathematically well-defined and their central claims reproduce exactly on enumerable problems
   (table §8.1): COCOA, R-HCA, HCA+-state, RUDDER (exact decomposition) are unbiased (relative bias ≤ 6e-31 = float
   round-off); return-conditioned HCA is biased (relative bias 0.72); HCA+-state equals REINFORCE at a decision whose
   action is decodable from later states (t=0 block variance identical, 11.054 vs 11.054); R-HCA's variance at the
   key decision is independent of immediate-reward noise (0.411 at σ0 = 0, 0.6, 1.5) while COCOA's grows
   (0.411 → 2.312 → 12.293); a RUDDER decomposition that "explains away" (puts all prediction on the last step) used as
   an immediate reward gives relative bias 1.00 (the key decision receives no credit). The R-HCA counterexample
   (Var R-HCA 400/9 > Var COCOA 326/9) and the HCA-return bandit bias (A = 1/3, −4/3 instead of 1, −2) reproduce exactly.
2. Zheng et al.'s meta-gradient (Eq. 4) is **exact** only with backpropagation of ∂θ_t/∂η through the whole lifetime;
   truncation to the last N inner updates is biased: relative error 0.60 (N=1) and 0.30 (N=2) on a 5-step lifetime
   (cosine to the exact gradient still 0.9999 — **a cosine test cannot detect this**). Full backprop: relative error
   0.005, max |z| 1.1 (unbiased).
3. **Invalid in CAM as published:** finite enumerable action sets (COCOA/R-HCA/HCA sum over all actions); undiscounted
   absorbing-state episodes (COCOA/HCA/R-HCA/RUDDER all use γ = 1 or a fixed γ); immediate, always-resolved rewards;
   rewards visible to the reward network (Zheng's r_η receives the extrinsic reward r_{t+1} in its input history);
   on-policy hindsight models (CAM's learner is distributed with policy lag); a Markov state s_t (CAM has a learned
   belief b_t that encodes past actions, which makes state-conditioned HCA degenerate — COCOA App. G.5).
4. **Adopted unchanged:** COCOA contribution coefficient definition and estimator (Eq. 4) for a *finite candidate set*;
   R-HCA Eq. 4 (k ≥ 0) for the same set; RUDDER return-equivalence + difference-of-predictions contribution analysis +
   residual correction; Zheng's optimal-reward objective over lifetimes, lifetime value function, truncated
   meta-gradient with value bootstrap.
5. **Adapted (each needs its own exact test, §8):** restricted candidate-policy estimator (§7.2), SMDP discounting
   Γ_{t,t+k} = exp(−κ Σ dt) in contribution coefficients and RUDDER conservation (§4.8, §5.6), delayed/censored verified
   outcomes as rewarding outcomes (§7.4), observation-only reward-network inputs (§7.5), functional differentiable inner
   update on a restricted "fast" parameter subset of the ~1B core (§7.7).
6. **Integration rule (CR-09):** every unit of verified utility must enter exactly one credit path. Recommended
   default, to be confirmed by Exp. C: R-HCA-feature over the recorded candidate set, rewarding outcomes = verified
   outcome events; RUDDER redistribution only for objectives whose utility is defined solely at trajectory end and has
   no attributable outcome events; neither applied to the same utility twice; intrinsic reward R_ψ is a separate,
   inner-loop-only signal (spec 12.4).

---------------------------------------------------------------------------------------------------------------------

## 1. Notation map (papers → CAM)

| Paper symbol | Meaning in paper | CAM symbol / object |
|---|---|---|
| s_t, x_t (HCA), S_t | Markov state | **b_t**: CAM latent belief / state tokens Z_t at decision t (history-dependent, learned, non-stationary across training) plus the recorded candidate set C_t when a candidate estimator is used: s̃_t = (b_t, C_t) |
| a_t, A_t | discrete action | **j_t** ∈ {1..K_t}: index of the executed candidate in C_t = {c_{t,1..K_t}} (concrete executable actions generated by the CAM-conditioned decoder from intent I_t); e_a(c) action embedding |
| π(a\|s) | policy | **σ_θ(j \| b_t, C_t)**: CAM's evaluation/selection policy over recorded candidates (finite); **q_φ(c \| b_t, I_t)**: proposal (decoder) distribution |
| g (UVFA) | goal | **e_g** goal embedding (learned null goal in goal-free mode) |
| u, U_{t+k}, U′ | rewarding outcome encoding f(s,a,r) | **u_{t+k} = U_ω(o^ver_{t+k})**: encoding of an independently verified outcome event (payment, passing test, accepted PR …), or the scalar verified utility increment |
| R_t | reward | Credit target: **ΔU^ver_t** (verified utility increment, training-label channel; never visible to the actor) for the outer/credit estimators; **r^ψ_t** (intrinsic) for the inner loop. Never summed together (spec 12.4) |
| Z_t, G_t | return | verified (discounted) return-to-go over an objective window |
| γ, β | discount / survival | **γ_t = exp(−κ dt_t)** (SMDP clock, `learn/returns.py:27`), Γ_{t,t+k} = Π_{u=t}^{t+k−1} γ_u |
| η (Zheng) | intrinsic reward params | **ψ** (`R_ψ`, `selfeval/intrinsic.py`) |
| φ (Zheng) | lifetime value params | **V_L** (`LifetimeValue`) |
| θ (Zheng) | policy params, reset each lifetime | **θ_fast**: parameters subject to differentiable inner updates (see §7.7); θ_slow: everything else |
| τ_t = (s_0,a_0,r_1,d_1,…,s_t) | lifetime history incl. extrinsic rewards and episode ends | actor-visible lifetime history only (observations, actions, costs, episode/objective boundaries) — no labels, no evaluator state (§7.5) |
| h(a\|s,u), h_β, h_z | hindsight action distribution | **h_κ(j \| b_t, C_t, u, e_g, policy_ctx)** pointer over candidate embeddings (`credit/heads.py: HindsightModel`) |
| g(Δ_{0:t}) (RUDDER) | sequence return predictor | **F_ξ(H_t, e_g)** (`credit/heads.py: ReturnDecomposer`) |

---------------------------------------------------------------------------------------------------------------------

## 2. Zheng et al. 2020 — learned intrinsic rewards across lifetimes

### 2.1 Objective (optimal reward problem, Singh et al. 2009/2010)

Agent: policy π_θ(a_t|s_t). Episode return G^ep = Σ_{t<T_ep} γ^t r_{t+1}. Lifetime = finite sequence of episodes until
the designer ends training; lifetime return G^life = Σ_{t<T} γ^t r_{t+1} (r = extrinsic). Intrinsic reward
r_η(τ_{t+1}) with lifetime history τ_t = (s_0, a_0, r_1, d_1, s_1, …, r_t, d_t, s_t), d_i binary episode terminations.

    J(η) = E_{θ_0 ~ Θ, T ~ p(T)} [ E_{τ ~ p_η(τ|θ_0)} [ G^life ] ]                                   (Zheng Eq. 1)
    p_η(τ|θ_0) = p(s_0) Π_t π_{θ_t}(a_t|s_t) p(d_{t+1}, r_{t+1}, s_{t+1} | s_t, a_t),   θ_t = f(θ_{t−1}, η)

Only η is transferred across lifetimes; θ is re-initialised (θ_0 ~ Θ) and a new task T is sampled per lifetime
(tasks may be non-stationary *within* a lifetime).

### 2.2 Architecture (paper §4.1, supp. Table 1)

* r_η: RNN over the **entire lifetime history** (not reset at episode boundaries): Conv(16,3×3,1)–FC(64)–LSTM(64),
  ReLU, **arctan output**. Inputs include the extrinsic reward and the termination bit.
* Lifetime value V_φ(τ_t): separate RNN, same inputs and architecture, scalar estimate of remaining lifetime return.
* Policy: Conv(16,3×3,1)–FC(64) with policy and (intrinsic) value heads.

### 2.3 Inner update (policy gradient on the intrinsic return)

    J_η(θ) = E_θ[ Σ_{t<T_ep} γ̄^t r_η(τ_{t+1}) ]                                                     (Eq. 2)
    ∇_θ J_η(θ) = E_θ[ G^ep_{η,t} ∇_θ log π_θ(a|s) ],   G^ep_{η,t} = Σ_{k=t}^{T_ep−1} γ̄^{k−t} r_η(τ_{k+1})   (Eq. 3)

As implemented (supp. C.1 pseudocode): an actor-critic on intrinsic rewards — n-step returns of r_in with **episodic**
discount (zero at episode ends) × γ̄ = 0.9, bootstrap from the agent's own value `rollout.v[-1]`, advantage
returns − v; loss = PG + baseline_cost·½(v − returns)² + inner_entropy_cost·entropy; θ_{k} = θ_{k−1} + optimiser
update (SGD lr 0.1; Adam 1e-3 on Key-Box). θ is updated after every rollout of length 4–16.

### 2.4 Outer meta-gradient, truncation, lifetime value

Supp. A derives (γ = 1 for brevity), with π_η(·|τ_t) = π_{θ_t}(·|s_t), θ_t = f_η(τ_t, θ_0):

    ∇_η J(η) = E_{τ_t}[ G_t ∇_η log π_η(a_t|τ_t) ] = E_{τ_t}[ G^life_t ∇_{θ_t} log π_{θ_t}(a_t|s_t) ∇_η θ_t ]   (Eq. 4)

with G^life_t the remaining lifetime return. This is exact (an ordinary likelihood-ratio gradient in which the policy
at time t is a deterministic function of η and the realised history) **iff ∇_η θ_t is propagated through every inner
update since the start of the lifetime**. Practical approximation (Eq. 5–6): truncate the chain rule after N policy
updates and bootstrap the return with the lifetime value

    G^{life,φ}_t = Σ_{k=0}^{n−1} γ^k r_{t+k+1} + γ^n V_φ(τ_{t+n})                                         (Eq. 5)
    φ ← φ + α′ (G^{life,φ}_t − V_φ(τ_t)) ∇_φ V_φ(τ_t)                                                     (Eq. 6)

Facts from the supp. C.2 outer pseudocode that matter for a faithful implementation:

* the window starts from the current learner state θ (treated as a constant: gradients do not flow into earlier
  windows); N + 1 rollouts; for rollout k the logits are computed under θ_k (after k inner updates) **before** the
  inner update with that rollout; the policy-gradient term uses rollouts 1..N ("the updated policy is evaluated by the
  next rollout"), i.e. **every inner rollout after the first is also outer evaluation data**;
* lifetime advantages = lifetime n-step returns (lifetime discount, zero only at lifetime end, × γ = 0.99) − V_φ,
  `backprop=False` (stop-gradient on the advantage);
* outer loss = pg_loss + baseline_cost·½(V_φ − G^life)² (on rollouts 0..N−1 only; the last rollout is used only for
  the meta-gradient) + outer_entropy_cost·entropy of the post-update policies (a meta-level entropy term
  differentiated into η through θ_k);
* η and φ are one parameter set updated by one Adam step (lr 1e-3); 64 lifetimes in parallel, meta-gradients
  averaged; 2·10^5 η updates; outer unroll N = 5.

### 2.5 Algorithm (Algorithm 1, with the pseudocode facts)

    initialise η, φ
    repeat                                           # many lifetimes, 64 in parallel
      sample task T ~ p(T); θ ~ Θ; reset r_η / V_φ RNN state at lifetime start only
      while lifetime not ended:
        θ_0 ← θ  (constant)
        for k = 1..N: roll out with π_{θ_{k−1}}; θ_k ← θ_{k−1} + α ∇_θ J_η(θ_{k−1})   (graph kept, functional)
        one more rollout with π_{θ_N}
        L_outer(η, φ) = −Σ_{rollouts 1..N} sg(A^life) log π_{θ_k}(a) + c_v ½ (V_φ − G^{life,φ})² − c_H H(π_{θ_k})
        (η, φ) ← Adam step on ∇ L_outer;  θ ← θ_N (detached)

### 2.6 Reference environments and hyperparameters (supp. B, Table 1)

| Domain | episode limit | episodes / lifetime | rollout len | entropy | policy opt / lr |
|---|---|---|---|---|---|
| Empty Rooms (4 rooms, invisible goal fixed per lifetime, uniform over cells) | 100 | 200 | 8 | 0.01 | SGD 0.1 |
| Random ABC (5×5; A~U[−1,1], B~U[−0.5,0], C~U[0,0.5] per lifetime; episode ends at object or limit) | 10 | 50 | 4 | 0.01 | SGD 0.1 |
| Key-Box (key needed to open A/B/C, same reward law; >3e9 states) | 100 | 5000 | 16 | 0.01 | Adam 1e-3 |
| Non-stationary ABC (A=±1, B=−0.5, C=−A; A,C swap every 250 episodes) | 10 | 1000 | 4 | 0.05 | SGD 0.1 |

Common: reward/lifetime-VF Adam 1e-3, N = 5, γ̄ = 0.9 (inner), γ = 0.99 (outer). Baselines: Extrinsic-EP,
Extrinsic-LIFE, count-based (Strehl & Littman), ICM; hand-designed near-optimal heuristic for Random ABC.

### 2.7 Findings (including transfer)

* The learned reward encodes cross-episode exploration then exploitation (Empty Rooms: rewards unvisited cells until
  the goal is found, then only the goal; Random ABC: explore A and C, avoid B, then commit), causal invariants
  (Key-Box: reward the key) and anticipation of non-stationarity (negative reward to the exploiting policy a few
  episodes **before** the swap, raising entropy).
* Ablations (Fig. 7): **lifetime-history input is crucial** (FF reward on the current step fails); the lifetime
  objective beats the episodic objective on Random ABC, Non-stationary ABC and Key-Box (episodic value penalises
  exploratory episodes).
* Policy transfer (RL², MAML) is faster on the *same* interface (RL² better than the learned reward; the learned reward
  beats MAML, matches RL² asymptotically). The learned reward **transfers** to permuted and extended action spaces and
  to a different learning algorithm (Q-learning) because it rewards *state changes*, not actions; RL²/MAML fail under
  permuted actions. Knowledge in a reward ("what") acts indirectly and slowly; knowledge in a policy ("how") acts
  immediately but generalises worse.

### 2.8 Assumptions and failure modes

* Assumes the reward network can observe the extrinsic reward (Random ABC is unsolvable otherwise: the value of A is
  only observable through r). **Invalid in observation-only CAM** (§7.5).
* Many lifetimes from a task distribution; the reward only captures regularities shared by that distribution.
* Truncation bias (measured §8.5); meta-gradient variance (64 lifetimes/batch); non-stationarity of θ_0 distribution
  is controlled by re-initialising θ each lifetime — **CAM's continually trained policy is not re-initialised**, so the
  "fresh learner" assumption must be recreated with resets of θ_fast (§7.7).
* No guarantee of policy invariance (the learned reward is not potential-based) — spec 12.4 is right that downstream
  verified evaluation is the only safeguard.

---------------------------------------------------------------------------------------------------------------------

## 3. Hindsight Credit Assignment (Harutyunyan et al. 2019)

### 3.1 State-conditional HCA

h_k(a|x,π,y) = P_{τ~T(x,π)}(A_0 = a | X_k = y); Bayes: h_k(a|x,y)/π(a|x) = P(X_k=y | x, a) / P(X_k=y | x).

    Q^π(x,a) = r(x,a) + E_{τ~T(x,π)}[ Σ_{k≥1} γ^k (h_k(a|x,X_k)/π(a|x)) R_k ]                              (Thm 1)
    A^π(x,a) = r(x,a) − r_π(x) + E_{τ~T(x,π)}[ Σ_{k≥1} (h_β(a|x,X_k)/π(a|x) − 1) γ^k R_k ],  β = γ          (Eq. 3–4)

h_β is the time-independent version (probability of a given that y is reached at some future time with survival β).
Policy gradient, all-actions form (Thm 3, Eq. 7), requires a learned reward model r̂(x,a) and no baseline:
∇V = E[Σ_k γ^k Σ_a ∇π(a|X_k) Q_x(X_k,a)], Q_x(X_s,a) ≈ r̂(X_s,a) + Σ_{t=s+1}^{T−1} γ^{t−s} (h_β(a|X_s,X_t)/π) R_t
+ γ^{T−s} (h_β(a|X_s,X_T)/π) V(X_T) (Algorithm 1; the bootstrap form is Theorem 7, **shown to be incorrect** by COCOA
App. K: a collider path A_0 → S′ ← K → Z invalidates the conditional independence used in its proof).

### 3.2 Return-conditional HCA

h_z(a|x,z) = P(A_0 = a | Z(τ) = z). Theorem 2 (assumes h_z(a|x,z) > 0 for **every** return z reachable from x):

    V^π(x) = E_{τ~T(x,a,π)}[ Z(τ) π(a|x) / h_z(a|x,Z(τ)) ],   A^π(x,a) = E_{τ~T(x,a,π)}[ (1 − π(a|x)/h_z(a|x,Z)) Z ]   (Eq. 5–6)
    ∇V = E[ Σ_k γ^k ∇ log π(A_k|X_k) (1 − π(A_k|X_k)/h_z(A_k|X_k, Z_k)) Z_k ]                                (Eq. 8, Alg. 2)

The baseline b_s = π(A_s|X_s) Z_s / h_z(A_s|X_s,Z_s) is an action-dependent baseline; Prop. 1 claims no bias, but only
under the support assumption, which fails whenever some returns are unreachable after some actions (COCOA App. L:
2-arm bandit, rewards 1/−2, π = 2/3,1/3 gives A = 1/3, −4/3 instead of 1, −2 — reproduced exactly here; in the
key-to-door of §8.1 its relative bias is 0.72–0.75). An **unbiased** return-conditioned variant is COCOA's
"Counterfactual Return" (App. F.3, Eq. 99): Σ_t Σ_a ∇π(a|S_t)(p^π(a|S_t,Z_t)/π(a|S_t) − 1) Z_t; it is high-variance
because the return mixes all subtasks.

### 3.3 Learning, diagnostic tasks, hyperparameters

h_β and h_z are trained by cross-entropy on (X_i, X_j≥i, A_i) resp. (X_i, Z_i, A_i) pairs (returns binned, n_b = 3 for
Delayed Effect, 10 elsewhere; return HCA is sensitive to n_b; baseline lr from {0.1..0.4}; all γ = 1, 100 runs).
Diagnostic tasks (Fig. 2): **Shortcut** (chain of length n=5, at each state one action jumps to the goal, per-step −1,
final +1, 0.1 chance of absorbing), **Delayed Effect** (choice at start, aliased chain, final ±1, white noise σ ∈ {0, 2}
on intermediate rewards; bootstrapping fails, MC is noisy, state-HCA robust), **Ambiguous Bandit** (2 actions, Gaussian
rewards means 1, 2, std 1.5 or 0.5, crossover ε). App. F: with bootstrapping, state HCA helps because h learns faster
than π (not because h is exact).

---------------------------------------------------------------------------------------------------------------------

## 4. RUDDER — return decomposition for delayed rewards

### 4.1 Definitions (main text §2, App. A2)

Finite MDP, γ = 1, finite horizon T or absorbing state, reward at sequence end. Sequence-Markov decision process (SDP):
Markov transitions and policy, reward not necessarily Markov. SDPs P̃, P are **return-equivalent** if they differ only
in reward and ṽ^π_0 = v^π_0 for all π; **strictly** return-equivalent if the expected return is equal for every
episode. Return-equivalent SDPs have the same optimal policies (Thm 1). Expected future rewards:

    κ(m, t−1) = E_π[ Σ_{τ=0}^{m} R_{t+1+τ} | s_{t−1}, a_{t−1} ]                                          (Def. 1)

A redistribution is **optimal** if κ(T−t−1, t) = 0 for all t. Theorem 2: for a delayed-reward MDP and a second-order
Markov redistribution, optimality ⇔

    E[R_{t+1} | s_{t−1}, a_{t−1}, s_t, a_t] = q̃^π(s_t, a_t) − q̃^π(s_{t−1}, a_{t−1})                      (Eq. 1)

Theorem 3: then q^π(s_t,a_t) = r(s_t,a_t) = q̃^π(s_t,a_t) − ψ^π(s_t), ψ^π(s_t) = E[q̃^π(s_{t−1},a_{t−1}) | s_t]; same
advantage function as the original MDP; off-policy: E_{π̆}[R_{t+1}|s_t,a_t] = q̃^π(s_t,a_t) − ψ^{π,π̆}(s_t) (Eq. 2–3).

### 4.2 Return decomposition by contribution analysis (§3, App. A2.8)

A function g predicts the sequence-wide return from the whole state-action sequence (not a value function). To avoid
"explaining away" (later states storing the accumulated contribution), inputs are differences
Δ(s_{t−1},a_{t−1},s_t,a_t) (assumed mutually independent, Eq. A121). With g predicting the return at every step:

    h_0 = g(Δ_{0:0}),   h_t = g(Δ_{0:t}) − g(Δ_{0:t−1})                                                     (A125–A126)
    E[R_1|s_0,a_0] = h_0,  E[R_{t+1}|s_{t−1},a_{t−1},s_t,a_t] = h_t,  R_{T+2} = R̃_{T+1} − Σ_{t=0}^{T} h_t    (A132–A134)

The compensation R_{T+2} makes the new SDP **strictly** return-equivalent for any g; a g with zero prediction error
gives the optimal redistribution. Alternatives (integrated gradients, LRP) introduce non-Markov rewards; RUDDER prefers
differences of predictions. κ-correction variant: R^c_{t+1} = R_{t+1} + κ(m,t) − κ(m,t−1) (A113); κ as an auxiliary
output pushes redistribution further back.

### 4.3 Learning algorithms using a redistribution (App. A2.7)

(A) Q-value estimation as the (exponential moving) mean of the redistributed immediate reward (variant i assumes
optimality; ii corrects with κ; iii eligibility traces); (B) policy gradients with q^π(s,a) replaced by an estimate of
r(s,a) or a sample of the redistributed reward (the offset ψ(s) acts as a baseline); PPO integration (below);
(C) Q-learning on the redistributed reward.

### 4.4 RUDDER as implemented for Atari (App. A4.2) — the losses and the hyperparameters that matter

* Three phases: **safe exploration** (avoid low-Q intervals; start time, length, Q threshold), **lessons replay buffer**
  (≤ 128 episodes; admission/priority = equal-weight combined rank of model error and |return − buffer mean|; sampling
  softmax over losses; samples of 512 transitions), **LSTM return decomposition**.
* LSTM: 64 cells, **no forget gate, no output gate**, sigmoid input gate (bias mean −5, recurrent-only), tanh cell
  input (forward-only), identity output (linear counting dynamics); input: Δ-frame + frame features, PPO vision features
  (no gradient back into PPO), action, approximate in-game time.
* Loss per sample (A268–A275), g = realised return, ĝ_t prediction at t:

        L_m = (g − ĝ_T)²
        L_c = 1/(T+1) Σ_t (g − ĝ_t)²
        L_e = 1/(T−9) Σ_{t≤T−10} (ĝ_{t+10} − (ĝ_{t+10})^_t)²       (predict the prediction 10 steps ahead)
        L_a = (L_a1 + L_a2 + L_a3)/3   (q-value, reward sum of next 10 and next 50 frames)
        L   = L_m + (L_c + L_e + L_a)/10

  Squared losses: g must estimate the **conditional expectation** of the return.
* Redistributed reward r_t = ĝ_t − ĝ_{t−1}; quality = 1 − |g − ĝ_T|/µ · 1/(1−ε) clipped to [0,1] (ε = 0.8, µ = 10);
  advantages a = a_o(1 − quality) + a_r·quality; value loss L_v = L_o + L_r·quality; rewards scaled r·10/g_max;
  train every 10^4 PPO updates or when the buffer changes, 8 samples/epoch, until quality > 0 on the last 5 epochs;
  start only with ≥ 32 buffer samples of different returns. Adam 1e-4, L2 1e-7, grad clip 0.5.

### 4.5 Reference tasks (tabular, ε-greedy 0.2, λ = 0.9, 100 trials, learning time to 90 % of optimum)

(I) Grid World 31×31 "time bomb" (bomb [30,15], start [30−d,15], ⌊1.5d⌋ steps, 1000 w.p. 0.5 if bomb visited,
immediate c·t·h with c = −0.09 toward / +0.1 away); (II) The Choice (tree, first action selects ± subtree;
R_{T+1} = R_C + R_0 + R_b with R_0, R_C cancelling in expectation, E[R] = p(a_0=+)·b); (III) Trace-Back (15×15,
start (7,7), T = 20, optimal up at t=1, right at t=2: −50 immediate, +150 at T; otherwise +50; transitions random for
t>2; optimal return 100); (IV) Charge-Discharge (r(D,d)=1 immediate, r(C,d)=10 delayed to T; LSTM 5 cells, Adam 0.01);
(V) Trace-Back with PPO±GAE. RUDDER exponentially faster than MC/TD(λ)/MCTS/reward shaping in the delay.

### 4.6 Assumptions and failure modes

γ = 1; complete episodes for training g; Markov policy; g deterministic; Δ components independent. Failure modes
stated by the authors: not effective without delay; LSTM problems on very long sequences; **spurious redistributed
rewards** from prediction errors (FNNs much worse than LSTMs because their errors do not cancel); redistribution is
correlational, not causal (spec 6.2 is right). Measured here (§8.1): using the redistributed reward *as the immediate
credit* is unbiased only for an optimal decomposition (exact g: unbiased, variance 2.08 vs REINFORCE 36.4); with an
explaining-away g it has relative bias 1.00; using the redistributed *return-to-go* is unbiased for any g (it differs
from the original return-to-go by g_{t−1}, a function of the past) but gives no variance reduction (35.4 vs 36.4).

### 4.7 [derived here] Discounted / variable-duration conservation

For the SMDP return G = Σ_t Γ_t r_t (Γ_t = Π_{u<t} γ_u = exp(−κ τ_t), τ_t elapsed days before decision t) and a
predictor F(H_{≤t}) of the start-referenced discounted return:

    c_t = [F(H_{≤t}) − F(H_{<t})] / Γ_t,   c_{first} += F(∅)/Γ_{first},   c_{last} += [G − F(H_{≤T})] / Γ_{last}
    ⇒ Σ_t Γ_t c_t = G   (per trajectory, exactly)

The redistributed return-to-go from t, Γ_t^{−1} Σ_{k≥t} Γ_k c_k = [G − F(H_{<t})]/Γ_t, equals the discounted
return-to-go referenced at t minus a function of the past ⇒ unbiased for the discounted objective for any F. Caveat:
1/Γ_t amplifies prediction noise for late decisions of long lifetimes; prefer objective windows with their own
t = 0 (Γ restarts) — `credit/heads.py:158` implements this formula.

---------------------------------------------------------------------------------------------------------------------

## 5. COCOA — counterfactual contribution analysis (Meulemans et al. 2023)

### 5.1 Setting and definitions

Undiscounted MDP with a zero-reward absorbing state reached w.p. 1 (episodic and discounted MDPs are special cases,
App. B: discount = (1−γ) absorption probability per step). Rewarding outcome U′ ~ p(U′|s′,a′,r′) (Def. 1).
Contribution coefficient (Eq. 3; time abstracted via K ~ Geometric(β), β → 1, App. C.1):

    w(s,a,u′) = Σ_{k≥1} p^π(U_{t+k}=u′ | S_t=s, A_t=a) / Σ_{k≥1} p^π(U_{t+k}=u′ | S_t=s) − 1
              = p^π(A_t=a | S_t=s, U′=u′) / π(a|s) − 1

Variants: U = S → state HCA ("HCA+" without reward model); U = R → **COCOA-reward**; U = learned reward features →
**COCOA-feature**. The middle (forward) form stays defined for π(a|s) = 0.

### 5.2 Estimators (Table 1, Eq. 4, App. C.3, G.4)

    ∇̂^U V^π(s_0) = Σ_t [ ∇ log π(A_t|S_t) R_t + Σ_{a∈A} ∇π(a|S_t) Σ_{k≥1} w(S_t, a, U_{t+k}) R_{t+k} ]   (Eq. 4)
    with reward model:        Σ_t Σ_a ∇π(a|S_t) [ r(S_t,a) + Σ_{k≥1} w(S_t,a,U_{t+k}) R_{t+k} ]             (Eq. 26)
    sampled alternatives:     Σ_t ∇ log π(A_t|S_t) R_t + (1/M) Σ_m ∇ log π(a_m|S_t) Σ_{k≥1} w(S_t,a_m,U_{t+k}) R_{t+k},
                              a_m ~ π(·|S_t) drawn **independently of the trajectory's A_t** (A_t may not be reused)   (Eq. 27, 107)
    reordered (credit view):  Σ_t R_t [ ∇ log π(A_t|S_t) + Σ_{k<t} Σ_a w(S_k,a,U_t) ∇π(a|S_k) ]                (Eq. 30)

Advantage identity (Lemma 6): A^π(s,a) = r(s,a) − Σ_{a′} π(a′|s) r(s,a′) + E_{T(s,π)}[Σ_{k≥1} w(s,a,U_k) R_k].

### 5.3 Theory

* **Def. 2 (fully predictive):** p^π(R_k = r | S_0=s, A_0=a, U_k=u) = p^π(R = r | U = u), independent of k.
* **Thm 1:** with ground-truth w and a fully predictive U, Eq. 4 is unbiased.
* **Prop. 2:** if every action sequence leads to a distinct state, HCA+ (U=S) = REINFORCE (measured §8.1: identical t=0
  variance).
* **Thm 3** (single final reward, policy optimised at the single initial step): S ⇒ U′ ⇒ U ⇒ R fully predictive chain
  gives V[∇̂^R] ≼ V[∇̂^U] ≼ V[∇̂^{U′}] ≼ V[∇̂^S] ≼ V[∇̂^{REINFORCE}] — less information in U ⇒ lower variance; R is
  optimal among fully predictive encodings.
* **Thm 4:** each term pushes up the expected number of occurrences O^π(u′,s) = Σ_k p^π(U_k=u′|S_0=s); informative U
  "forgets" unvisited sibling outcomes → variance.
* **Prop. 5:** p^π(a|s,u′,l) = p^π(a|s,u′) for policy logits l(s) → feeding logits to h is free and helps track the
  moving target. **Prop. 8:** p(A_0|S_0,R′) = Σ_{s′} p(S′=s′|S_0,R′) p(A_0|S_0,S′=s′) (reward-aware smoothing of
  state hindsight).

### 5.4 Learning the coefficients (§3.4, App. D, E.8)

* Hindsight classifier h(a | s, u′, l) by cross-entropy on observed (S_t, U_{t+k}, A_t) for all k > t (reference code:
  uniform over pairs, optional `mask_zero_reward_loss` — train only on pairs whose future reward ≠ 0; logits =
  f(s,u,l) + policy logits when `modulate_with_policy`); network: hypernetwork combining policy logits with state and
  hindsight object multiplicatively (⊗) with gated ReLU ReLU-g(x) = ReLU(x_{0:n/2}) − ReLU(x_{n/2:n}); a plain MLP on
  (s, u′) "performs poorly" (App. E.8.1).
* Contrastive alternative (Prop. 7): D(a,s,u′) separating a ~ p^π(a|s,u′) from a ~ π(a|s);
  w = D*/(1−D*) − 1 = exp(logit) − 1; loss Σ_m [−log D(a_m,s_m,u′_m) − Σ_{a′} π(a′|s_m) log(1 − D(a′,s_m,u′_m))] (Eq. 67).
* Successor representations (D.3): Σ_k p(U_k=u′|s,a) = Σ_{s′} p(U′=u′|S′=s′) M(s,a,s′) — the DP ground truth.
* COCOA-feature reward network: input × action-specific squared mask → ReLU (straight-through) → action-independent
  linear readout; MSE to reward, L1 1e-3 (pre-readout) and L2 0.03 (readout) as decoupled weight decay; pre-trained on
  30 batches of a random policy for 20 000 full-batch Adam steps, frozen, features discretised by 1[x > 0.05]
  (task interleaving: L1 0.05, L2 3e-4, 90 batches, 30 000 steps).

### 5.5 Algorithm (App. E.1, Algorithm 1)

    collect M batches with a random policy; pretrain reward features U = f(S,A) (COCOA-feature only)
    for each iteration:
      sample L trajectories from π
      for t < k: train h(A_t | S_t, U_k, π_logits(S_t)) by cross-entropy           # contribution module
      for t: X_a = ∇π(a|S_t) Σ_{k>t} w(S_t,a,U_k) R_k  for all a;  ĝ += ∇ log π(A_t|S_t) R_t + Σ_a X_a
      policy step with ĝ (AdamW, entropy regularisation + ε-greedy)

### 5.6 Discounting and bootstrapping (App. J, K)

Discounted coefficients w_γ(s,a,u′) = Σ_k γ^k p(U_{t+k}=u′|s,a) / Σ_k γ^k p(U_{t+k}=u′|s) − 1 = p^π_γ(a|s,u′)/π(a|s) − 1;
hindsight loss weighted by γ^k: L_γ = E[Σ_t Σ_{k≥1} γ^k CE(h_γ(·|S_t,U_{t+k}), δ_{A_t})] (Eq. 122); estimator
Σ_t γ^t ∇log π(A_t|S_t) R_t + Σ_a ∇π(a|S_t) Σ_{k≥1} γ^k w_γ R_{t+k} (Table 6). **[derived here]** For CAM's SMDP
clock replace γ^k by Γ_{t,t+k} = exp(−κ(τ_{t+k} − τ_t)) in both the coefficient and the CE weights (time-varying survival;
same derivation as App. C.1 with p(K=k) ∝ Γ_{t,t+k}). n-step bootstrapping (Thm 10) needs **state-based** w_n(s,a,S_n)
in front of V(S_n) — spurious contributions return; the authors leave value bootstrapping with outcome coefficients open.

### 5.7 Partial observability and continuous spaces (App. G)

With a belief/internal state that summarises history, past actions are decodable ⇒ HCA-state = REINFORCE; COCOA-reward
averages over belief states that yield the same reward (Eq. 110) and avoids it. Deterministic continuous rewards make
U = R as informative as the state ⇒ degeneration; remedies trade bias for variance: reward binning, stochastic
outcome U ~ N(R, σ), input noise / capacity limits on h. Continuous actions: integral or the independent-sample
estimator (Eq. 107, unbiased, extra variance).

### 5.8 Reference environments, metrics, hyperparameters (App. E)

* **Linear key-to-door** (main testbed): pick key at t=0, distractor apples (left/right, two stochastic values), door;
  9-dim state; 4 actions (pick key, pick left, pick right, open door); lengths L ∈ {20,…,100} (Fig. 3: distance 100);
  distractor reward ~ U({2/L, 18/L}), treasure 4/L. Reward switching: door-less L = 40, treasure sign flipped after
  60k episodes. Reward aliasing: treasure value equal to one distractor value.
* **Tree** (Thm 3 check): depth d, n_a actions, overlap o_s; transition j ← j(n_a − o_s) + a; reward
  ((idx(s) + a·p + seed) mod n_r) − n_r//2, n_r = 5; grouping u = ((idx(s)+a·p+seed) mod (n_r n_g)) − n_r//2; Fig. 2:
  6 actions, depth 4, n_g ∈ {4, 32}.
* **Task interleaving** (Mesnard et al.): C = 5 contexts, O = 2 objects, B = 3 open contexts, γ = 0.95 (death prob.).
* **Ground truth by DP** (E.2): M(s,a,s′,t) = Σ_{s″} p(s″|s,a) Σ_{a″} π(a″|s″)(1[s″=s′] + M(s″,a″,s′,t−1)) (Eq. 76);
  w = M(s,a,u′)/Σ_{a′} π(a′|s) M(s,a′,u′) − 1 (Eq. 77–78); expected advantage of an imperfect ŵ (Eq. 83–85) and the
  expected (biased) gradient (Eq. 88–91). Metrics (Eq. 92–94): SNR = ‖∇V‖² / E‖∇̂ − ∇V‖², Variance and Bias normalised by
  ‖∇V‖², in dB, MC sample size 512; **t = 0 partial gradient** (the key decision); **shadow training** (policy trained by
  ground-truth Q-critic, all estimators evaluated on the same data).
* Hyperparameters (Tables 2–5): batch 8 (512 for metrics), AdamW, lr_agent 3e-4, lr_hindsight 3e-3 (key-to-door) /
  1e-3 (interleaving), lr_features 3e-3, entropy 0.03 (L=20) → 0.01 (L=100) log-log interpolated, gradient-norm
  clipping (1.0 in the long reward-switch runs); 30 seeds for selection + 30 fresh seeds.
* Results: COCOA-reward ≈ ground-truth Q-critic with GT coefficients; SNR robust to key–door distance while HCA/baselines
  decay ∝ 1/distance; COCOA-feature robust to reward aliasing and adapts almost instantly to reward switches; SNR of
  COCOA-reward degrades with multiplicative model noise similarly to Q-critic (App. F.4).

### 5.9 Failure modes

Policy-dependent hindsight targets (on-policy data only); scarce training signal (only rewarding outcomes are useful
for U = R; latent learning from non-rewarding observations is an open problem, App. H); deterministic continuous
rewards; reward aliasing for COCOA-reward; outcome encodings that carry state information; immediate rewards handled by
REINFORCE (the R-HCA failure mode, §6); no validated value bootstrapping; hindsight is not counterfactual in the SCM
sense (App. I.2).

---------------------------------------------------------------------------------------------------------------------

## 6. R-HCA — reward hindsight credit assignment (Ramesh, He, Schmidhuber, White 2026)

### 6.1 What it changes

COCOA-reward (their Eq. 3) handles the immediate reward R_t through the REINFORCE term ∇log π(A_t|S_t) R_t; with noisy
immediate rewards that term dominates the variance even with a perfect hindsight model. R-HCA reweights the immediate
reward through hindsight as well — **the only change is that the sum over k starts at k = 0**:

    ∇̂V^π(S_0) = Σ_t Σ_{a∈A} ∇_θ π(a|S_t) Σ_{k≥0} ( ĥ^π_k(a | S_t, R_{t+k}) / π_θ(a|S_t) ) γ^k R_{t+k}       (R-HCA Eq. 4)

with a time-indexed hindsight model h^π_k(a|s,u) = P(A_t=a | S_t=s, U_{t+k}=u) (k = elapsed steps, k = 0 conditions
on the reward of the same step). No reward model is needed.

* **Prop. 1:** Q^π(s_t,a) = E_{τ~T(s_t,π)}[Σ_{k≥0} γ^k (h^π_k(a|s_t,R_{t+k})/π(a|s_t)) R_{t+k}] (Bayes on
  P(R_{t+k}=r|s_t,a)). Substituting into the score-function PG and converting to all-actions form gives Eq. 4
  (unbiased with the true hindsight; inspired by Alipov et al. 2021, without their assumption that rewards depend only
  on the next state).
* **Prop. 2 (contextual bandit):** Var(ĝ_COCOA) = Var(ĝ_R-HCA) + E[Var(ĝ_COCOA | S, R)] — R-HCA is the
  Rao-Blackwellisation of REINFORCE over A given (S, R): ĝ_R-HCA = R Σ_a ĥ(a|S,R) ∇log π(a|S).
* **Prop. 3 (sparse reward MDP, R_t = 0 for t < H):** Var(R-HCA) ≤ Var(COCOA).
* **Prop. 4 (general MDP):** Var(COCOA_t) − Var(R-HCA_t) = [Var(I_COCOA,t) − Var(I_R-HCA,t)] (≥ 0) + 2 Cov(D_t, G_fut,t),
  I = immediate-reward terms, D_t = I_COCOA,t − I_R-HCA,t, G_fut,t = Σ_a ∇log π(a|S_t) Σ_{k≥1} ĥ_k γ^k R_{t+k};
  R-HCA wins iff Var-difference ≥ −2 Cov(D_t, G_fut,t). **Example 1 (App. E)** gives Var R-HCA = 400/9 > Var COCOA =
  326/9 when a constant step cost acts as a negatively correlated control variate (reproduced exactly, §8.2).
* **Prop. 5 (features):** for U_{t+k} = f(S_{t+k},A_{t+k},R_{t+k}) with R_{t+k} = g(s_t, U_{t+k}) measurable, the same
  identity and the feature-based update Σ_t Σ_a ∇π(a|S_t) Σ_{k≥0} (ĥ_k(a|S_t,U_{t+k})/π) γ^k R_{t+k} hold.
  **[derived here]** Measurability is sufficient, not necessary: COCOA's full-predictiveness (E[R_{t+k} | U_{t+k}, S_t,
  A_t] = r̄(U_{t+k})) also gives unbiasedness; a time-abstracted h (occurrence sums over k ≥ 0, undiscounted) is also
  unbiased — this is the form used in §8.1 (relative bias ≤ 5e-31 for σ0 ∈ {0, 0.6, 1.5}).

### 6.2 Algorithm (App. G, Algorithm 1; reference code `agents/hca.py`)

    for each episode τ = (S_0,A_0,R_0,…,S_H,A_H,R_H) collected with π_θ:
      G(i,a) = Σ_{j≥i} ( ĥ_{j−i}(a|S_i,R_j) / π_θ(a|S_i) ) R_j      (γ^{j−i} in the code; ratio clamped at max 10)
      actor:     L_π = − Σ_i Σ_a π_θ(a|S_i) G(i,a)                  (G detached; plus entropy bonus)
      hindsight: L_h = Σ_i Σ_{j≥i} CE(δ_{A_i}, ĥ_{j−i}(·|S_i,R_j))  (all pairs incl. (i,i); no discount weight)

Reference-code details that matter: the ratios used for the actor step are computed with the hindsight model in eval
mode **before** it is trained on the same batch (avoids the memorisation degeneracy ĥ → δ_{A_i}, which turns the
estimator into REINFORCE); π in the ratio denominator is the behaviour-time softmax; the COCOA variant in the same code
uses ratio (not ratio − 1) because Σ_a ∇π(a) c = 0; future info may be the reward ("categorical" two-hot) or a TD-like
delta ("delta-categorical").

### 6.3 Hindsight model (App. F.1) and hyperparameters (App. F)

Inputs: state, future reward (two-hot over 51 bins in [r_min, r_max]), policy log-probabilities at S_t, elapsed time k
(learned positional embedding added to the reward embedding). State and reward branches: linear to d = 512 →
residual block (3 linear + ReLU + LayerNorm) → projection to d_int × |A| (d_int = 128); non-gated logits =
Σ_{d_int}(state ⊙ future)/√d_int; scalar gate g ∈ [0,1] from the **detached** concatenated features (2-layer MLP +
sigmoid); output softmax(g·f + (1 − g)·centred log π). Init: encoders orthogonal gain 1, others N(0, 0.02). Actor/critic:
separate MLPs 3×512 ReLU, orthogonal init (actor head gain 0.01). Optimisers RMSProp (actor), Adam (hindsight);
γ = 1, episodic rollouts, 4 parallel processes, value-loss coefficient 0 for hindsight methods, ratio clip 10.
Selected (Umbrella): R-HCA lr 5e-5, entropy 1e-4, LRH 5e-5, K = 2 hindsight updates/batch; Planned Catch: R-HCA lr 1e-4,
entropy 0.1, LRH 1e-4, K = 1; two-hot ranges ±0.1 (Umbrella), ±1.5 (Catch), [0,1] (CartPole), ±100 (Lunar Lander).

### 6.4 Reference tasks and findings

* **Umbrella** (H = 50, γ = 1): state (1 − t/H, context bits s², s³, first-action bits s⁴, s⁵); only the first action
  matters; every non-terminal reward is −4/H or 6/H w.p. ½ (pure noise, mean +1/H); final reward +1/H if s²_H = s⁴_H else
  −1/H. Ground-truth hindsight in closed form (App. D). With GT hindsight R-HCA learns much faster than COCOA; with
  learned hindsight R-HCA > COCOA-reward > A2C(λ ∈ {0, 0.8, 1}); State-HCA+ no better than A2C.
* **Planned Catch** (40×5 grid, T = 39, actions L/R/Stay, actions effective only for the first L steps, ±1 at the end,
  intermediate rewards uniform in {−κ, κ}; evaluated with κ = 0): high noise κ = 1.5 — R-HCA strong and flat in L, A2C(0.8)
  fails at L = 4 and catches up at L = 40, COCOA between; low noise κ = 0.5 — A2C(0.8) best for nearly all L.
* **CartPole, Lunar Lander** (fixed 200-step episodes): A2C(0.8) ≥ hindsight methods; on Lunar Lander all hindsight
  methods lag far behind (multi-factor shaped reward is hard to explain by a hindsight model).
* Conclusion: reward-based hindsight pays off when consequential actions are separated from informative feedback by many
  noisy intermediate rewards; otherwise TD(λ)/GAE actor-critic is preferable; hybrid use is suggested but not given.

### 6.5 What the paper does not provide (do not invent)

No discounted/time-abstracted variant beyond γ^k weighting in code; no off-policy / policy-lag correction (hindsight
trained only on the latest data); no estimator for large or continuous action spaces (no analogue of COCOA Eq. 27 is
stated — by analogy it would need independently sampled alternatives, **unverified**); no value bootstrapping (the
critic combination is "future work"); no rewarding-feature experiments (Prop. 5 only). CR-07 can therefore be completed
exactly as published for finite candidate sets; any extension must be labelled as such.

---------------------------------------------------------------------------------------------------------------------

## 7. Adopted, adapted and invalid assumptions in CAM

| # | Assumption in the papers | Status in CAM | Treatment |
|---|---|---|---|
| A1 | Markov state s_t observed (all) | **Invalid** — CAM acts on a learned belief b_t over an open-ended POMDP | Use b_t recorded **at behaviour time** (not recomputed by a later encoder) as S_t; hindsight model input b_t detached (no L_hindsight gradient into the core initially — the core must not be shaped by a model that sees the future); policy-version conditioning |
| A2 | State-conditioned hindsight (HCA, HCA+) is informative | **Invalid** — b_t encodes the agent's own past actions ⇒ HCA-state = REINFORCE (COCOA Prop. 2, App. G.5; measured §8.1) | Use outcome-conditioned coefficients (U = verified outcome event / utility) only; keep HCA+ only as a baseline |
| A3 | Finite enumerable action set, Σ_a ∇π(a\|s) | **Invalid** for executable programs; **adopted** for the recorded candidate set | Restricted candidate-policy estimator, §7.2 |
| A4 | Undiscounted episodes with absorbing state (COCOA/HCA/R-HCA), γ = 1 episodic with reward at end (RUDDER) | **Invalid** for open-ended lifetimes | Objective windows with deadlines (finite horizon, Γ restarts) and SMDP survival weights Γ_{t,t+k} in coefficients, CE weights and RUDDER conservation (§4.7, §5.6) |
| A5 | Rewards observed with the transition and always resolved | **Invalid** — verified outcomes are delayed (days), may stay pending, arrive through the training-label channel | §7.4 |
| A6 | Reward network sees extrinsic reward (Zheng) | **Invalid** under observation-only operation | §7.5 |
| A7 | Fresh learner per lifetime (θ_0 ~ Θ) (Zheng) | **Invalid** for one continually trained CAM | Meta-training lifetimes reset θ_fast (and episodic memory per spec 5.6 transfer tests) while θ_slow is fixed inside a lifetime; §7.7 |
| A8 | Exact gradient through all inner updates (Zheng Eq. 4) | **Infeasible** at 1B scale | Truncated windows K ∈ {1,2,4,8} + V_L bootstrap; truncation bias measured on exact problems (§8.5) |
| A9 | On-policy hindsight targets | **Invalid** under distributed policy lag | Feed behaviour log-probs (Prop. 5) to h, bound lag (`learn/ppo.py` max_policy_lag), recompute coefficients with the *current* σ in the denominator only when lag ≤ bound; unvalidated, must be measured (§8.6) |
| A10 | U fully predictive and minimal | **Adapted** — must be tested, not assumed (spec 7.5) | Outcome encoder trained to predict verified utility; test full-predictiveness: E[(R − r̂(u))²] ≈ E[(R − r̂(u, b_t, j_t, k))²]; bottleneck/discretisation (COCOA-feature recipe) to discard state detail |
| A11 | Rewards defined by the task, independent of the learner | **Adopted** for credit (verified utility); **invalid** for R_ψ (learned) | Credit estimators act on verified utility; R_ψ only drives the inner loop (spec 12.4) |
| A12 | RUDDER g predicts the conditional mean (squared loss) | **Adopted** | Do not regress symlog targets with Huber (§10 item 5) |

### 7.1 Two different "rewards" — accounting

Outer/credit channel: ΔU^ver (verified, delayed, label channel). Inner channel: r^ψ (learned). COCOA/R-HCA/RUDDER are
estimators of ∇ of an expected sum of *one* reward stream. If they are used on the inner stream (credit for intrinsic
rewards), the hindsight model's targets are intrinsic rewards (which change with ψ — version them); if used on the
verified stream, the result is the policy-gradient of verified utility (the "ordinary" CAM actor objective when
intrinsic training is disabled, and the outer query surrogate of §2.4 when meta-training). Never add the redistributed
verified return to r^ψ.

### 7.2 Restricted candidate-policy estimator (adapting A3) — definition

At decision t CAM records C_t = {c_{t,1..K_t}} (independent draws from q_φ(·|b_t, I_t)), the selection distribution
σ_θ(·|b_t, C_t) and the chosen j_t (`semantic/actions.py:787–797` already stores n, probs, chosen, proposal_logp).
Treat s̃_t = (b_t, C_t) as the state of a finite-action decision process whose action is j:

    COCOA:  ∇̂_θ = Σ_t [ ∇_θ log σ_θ(j_t|s̃_t) R_t + Σ_{j=1}^{K_t} ∇_θ σ_θ(j|s̃_t) Σ_{k≥1} w(s̃_t, j, U_{t+k}) R_{t+k} ]
    R-HCA:  ∇̂_θ = Σ_t Σ_{j=1}^{K_t} ∇_θ σ_θ(j|s̃_t) Σ_{k≥0} ( h_κ(j | s̃_t, U_{t+k}, k) / σ_θ(j|s̃_t) ) Γ_{t,t+k} R_{t+k}

Both are exact (Thm 1 / Prop. 1) for the selector parameters **conditional on C_t** (C_t is drawn before σ acts and is
part of the state); they estimate the gradient of V restricted to the proposal's support. They say nothing about
actions never proposed. For the proposal parameters φ, **[derived here]** an unbiased form is the score function of
the whole candidate set, ∇_φ Σ_i log q_φ(c_{t,i}) · (Σ_j σ(j|s̃_t) Â(s̃_t, j) − b(b_t)) with Â(s̃_t, j) the COCOA/R-HCA
per-candidate advantage — unverified at scale; distilling q toward high-Â candidates is a heuristic and must be labelled
so. Report candidate coverage: hindsight mass on the executed candidate, fraction of decisions where a later-verified
better action was in C_t, K_t distribution, proposal entropy.

### 7.3 Policy context for the hindsight model

h_κ(j | b_t, C_t, u, e_g, k, log σ(·|s̃_t)) with the R-HCA gated policy prior: logits = g·f + (1 − g)·log σ (or COCOA's
f + log σ). Without the prior the model must re-learn the policy at every update (COCOA App. E.8.1, R-HCA §2.2).

### 7.4 Delayed, censored, verified outcomes (adapting A5)

* The rewarding outcome is the verified event (e.g. "invoice paid by independent customer X"), placed at the decision
  index t+k at which the outcome *occurred* (event timestamp), not at which verification arrived; Γ uses the elapsed
  simulated time. The hindsight pair (t, t+k) becomes trainable when the label resolves.
* Pending outcomes are never labelled failures (SE-03) and are excluded from hindsight training and from the estimator;
  this conditions on resolution and biases toward fast-resolving outcomes unless (i) credit for a window is computed only
  after the window's resolution deadline, or (ii) a resolution-time model weights resolved samples by 1/P(resolved by
  now) — choose (i) for the default (simple, unbiased given the deadline).
* Retrospective recomputation (spec 11.2): coefficients and redistributions are recomputed when new outcomes resolve;
  the online actor never sees them (future-information firewall: h_κ and F_ξ are only called by the learner).

### 7.5 Observation-only reward network inputs (adapting A6)

Zheng's r_η conditions on extrinsic rewards; CAM's R_ψ may condition only on actor-visible information: observations
(including numbers the world shows the actor, e.g. its own account balance), its own predictions, uncertainty and costs
(`selfeval/intrinsic.py: ExperienceSignals`). Training labels (ΔU^ver) reach ψ **only** through the outer meta-gradient
and V_L targets; the private evaluator state (Crucible private counterfactual forks, sealed objectives) never reaches
any network. Consequence: problems like Random ABC are learnable only to the extent that outcome quality is inferable
from observations; this is the intended difficulty, not a defect. Reward-integrity test (Exp. F): permuting/zeroing
the label channel must leave R_ψ's outputs unchanged at fixed ψ.

### 7.6 Lifetime value (adopting Zheng Eq. 5–6)

V_L(lifetime state) regresses n-step *verified* lifetime utility + bootstrap (never intrinsic rewards). It is (a) the
baseline of the outer surrogate (any action-independent baseline is unbiased) and (b) the bootstrap that replaces the
utility beyond the truncation window — (b) requires V_L in utility units and an estimate of the conditional mean
(squared loss or a distributional head's mean; a symlog regression gives exp-of-mean-log, biased low for heavy tails).

### 7.7 Differentiable inner update at CAM scale (adapting A7/A8)

Second-order differentiation through K updates of the ~1B core is infeasible. Make θ_fast an explicit, small parameter
set (selection/evaluation head, value head for r^ψ, optional per-lifetime low-rank adapters on a few late core layers),
update it functionally (`torch.func.functional_call` with create_graph=True) inside a window; θ_slow is frozen inside the
window and trained first-order outside it. Freeze R_ψ's running normalisation statistics within a window (otherwise the
differentiated reward is not the reward the inner update used). Account compute per window (spec 5.5).

---------------------------------------------------------------------------------------------------------------------

## 8. Exact diagnostic environments (verification targets)

All expectations below are exact (enumeration or dynamic programming), so implementations can be tested to ~1e-9, not
by noisy Monte-Carlo cosine checks.

### 8.1 E1 NoisyKeyDoor-4 (exact enumeration; `credit_exact.py`)

T = 4, 2 actions, state (t, key). t=0: a=1 takes the key; R_0 = ±σ0 (noise). t=1,2: R_t = 0.3·a_t ± 0.8. t=3:
R_3 = 1 if key else 0. Tabular softmax θ[t, key, a] ~ 0.5·N(0,1) (seed 0). 2^4 actions × 2^3 noise = 128 trajectories.
Ground-truth coefficients from occurrence sums (k ≥ 1 for COCOA, k ≥ 0 time-abstracted for R-HCA), exact Q/V,
return-conditioned h_z by enumeration. Metrics normalised by ‖∇V‖² (=0.086751; t=0 block 0.075538).

σ0 = 0.6:

| estimator | rel. bias | rel. var | SNR dB | t0 rel. bias | t0 rel. var | t0 SNR dB |
|---|---|---|---|---|---|---|
| REINFORCE | 1.9e-31 | 36.424 | −15.61 | 4.1e-32 | 11.054 | −10.44 |
| Advantage (exact V) | 1.5e-31 | 19.090 | −12.81 | 5.1e-32 | 9.874 | −9.94 |
| Q-critic (exact Q) | 1.1e-31 | 0.080 | 10.99 | 5.1e-32 | 0.000 | (exact) |
| HCA+ (U = S) | 9.7e-32 | 20.600 | −13.14 | 5.1e-32 | **11.054** (= REINFORCE) | −10.44 |
| COCOA-reward | 9.7e-32 | 12.988 | −11.14 | 5.1e-32 | 2.312 | −3.64 |
| R-HCA (U = R, k ≥ 0) | 1.3e-31 | 5.295 | −7.24 | 9.2e-32 | **0.411** | 3.86 |
| HCA-return | **0.72** | 13.487 | −11.52 | **0.47** | 7.617 | −9.08 |
| RUDDER immediate, exact g | 1.6e-31 | 2.077 | −3.17 | 8.2e-32 | 2.194 | −3.41 |
| RUDDER immediate, explaining-away g | **1.00** | 7.965 | −9.53 | **1.00** | 0.000 | 0.00 |
| RUDDER return-to-go, exact g | 1.7e-31 | 35.382 | −15.49 | 4.1e-32 | 11.054 | −10.44 |
| RUDDER return-to-go, explaining-away g | 1.2e-31 | 56.777 | −17.54 | 4.1e-32 | 11.054 | −10.44 |

σ0 sweep (t0 rel. variance): COCOA 0.411 / 2.312 / 12.293 and R-HCA 0.411 / 0.411 / 0.411 for σ0 = 0 / 0.6 / 1.5;
total-gradient rel. variance COCOA 11.33 / 12.99 / 21.68, R-HCA 5.295 constant. RUDDER conservation Σ_t d_t +
compensation = G asserted on every trajectory (< 1e-12).

**Acceptance tests to port into `tests/cam_unified/credit/`:** exact expectation (not sampled) of COCOA, R-HCA, HCA+,
RUDDER(exact g) equals ∇V to 1e-10 relative; HCA-return bias > 0.5; HCA+ t0 variance equals REINFORCE t0 variance to
1e-10; R-HCA t0 variance invariant to σ0 to 1e-10 while COCOA's is strictly increasing; RUDDER immediate with
explaining-away g has t0 bias 1.0.

### 8.2 E2 Closed-form counterexamples

* HCA-return bandit (COCOA App. L): rewards (1, −2), π = (2/3, 1/3) ⇒ HCA-return advantages (1/3, −4/3) vs true (1, −2).
* R-HCA App. E Example 1: actions a0,a1,a2 with logits (−θ, 0, θ) at θ = 0, R_0 = 1, S1→−5, S3→−15, S2→±(−5,−15) w.p. ½
  ⇒ Var R-HCA = 400/9 ≈ 44.44 > Var COCOA = 326/9 ≈ 36.22, equal means −10/3. (Both reproduced exactly.)
* Gaussian bandit (R-HCA App. C): σ(θ) policy, μ = ±0.5, noise σ = 5, perfect hindsight
  ĥ(a1|r) = σ(θ)N(r;μ1,σ²) / [σ(θ)N(r;μ1,σ²) + (1−σ(θ))N(r;μ0,σ²)]; paper's N = 1000 histograms: COCOA var 5.99 / 2.17,
  R-HCA 0.11 / 0.01 at σ(θ) = 0.5 / 0.9 (compute the exact variances by 1-D quadrature).

### 8.3 E3 COCOA's own DP testbeds (for long delays, Exp. C)

Linear key-to-door with T ∈ {16, 64, 256, 1024} decisions (state (t, key, flag): S ≈ 4T, DP cost O(T²·S) is trivial):
exact SNR/bias/variance of the t = 0 partial gradient for REINFORCE, Advantage, Q-critic, HCA+, COCOA, R-HCA,
RUDDER, learned-coefficient versions, in shadow training. Expected (paper): REINFORCE/HCA SNR ∝ 1/distance; COCOA flat.
Add the Planned-Catch noise structure (intermediate ±κ) to separate R-HCA from COCOA. Tree environment for Thm 3's
variance ordering under its exact conditions (single decision, single final reward).

### 8.4 E4 RUDDER exact tests

κ-test: with exact g, E[Σ_{τ≥0} R̃_{t+2+τ} | s_t, a_t] = 0 for all t (compute by DP); Trace-Back (§4.5) as the canonical
delayed-reward task; Charge-Discharge for "distracting immediate reward". Discounted conservation (§4.7) with random
dt per decision.

### 8.5 E5 Meta-gradient exactness (`metagrad_exact.py`)

3-armed bandit, lifetime L = 5 one-step episodes, tasks r_ext ∈ {(1, −0.5, 0.2), (−1, −0.5, 0.2)} uniform per lifetime,
θ_0 = 0, intrinsic reward table η[arm, sign(r_ext)] (sees the current outcome, as Zheng's r_η does), inner
θ ← θ + 1.0·r_η(a,r)·∇log π(a) after every episode (sampled action), J(η) = E[Σ_t r_ext(a_t)].
Exact J and ∇_ηJ by enumerating 3^5 × 2 histories: J = −0.7306, ‖∇J‖ = 0.7887; finite differences agree to 4.7e-11.
Monte Carlo, 2·10^5 lifetimes:

| estimator | rel. error vs exact | cosine | max \|z\| |
|---|---|---|---|
| Zheng Eq. 4, ∂θ_t/∂η through the whole lifetime, G^life | 0.005 | 1.0000 | 1.1 (unbiased) |
| truncated: ∂θ_t/∂η through the last 1 update | 0.597 | 0.9999 | 208 |
| truncated: last 2 updates | 0.298 | 1.0000 | 65 |
| full backprop, episodic return (myopic ablation) | 0.021 | 0.9999 | 7.0 |
| inner update independent of η | gradient exactly 0 | — | — |

Use (i) exact-vs-truncated **relative error and z-scores**, never cosine alone; (ii) a costly-exploration task
(Exp. J; the committed `test_meta_learned_reward_values_costly_exploration…` on key-to-door with key cost 0.3 is a good
start); (iii) lifetimes with episode boundaries (Random ABC-like 2-object version, 3–5 episodes) to show the lifetime
objective differs from the episodic objective.

### 8.6 E6 Learned-model tests (not exact, but with exact references)

Umbrella (H = 50) and Planned Catch (L ∈ {4, 20, 40}, κ ∈ {0.5, 1.5}) with ground-truth vs learned hindsight; COCOA's
model-perturbation test (multiply GT hindsight by (1 + σε), σ ∈ {0.001 … 0.3}, App. F.4); policy-lag test: train h on
data from π_{v−L}, evaluate the coefficient error against the exact coefficients of π_v for lags L ∈ {0, 1, 4, 16}.

---------------------------------------------------------------------------------------------------------------------

## 9. Integration accounting and gradient routing for these components (UR-03, CR-09)

| Loss | Data | Updates | Gradient stops |
|---|---|---|---|
| L_hindsight = Σ_{t<k} Γ_{t,k} CE(h_κ(·\|b_t^beh, C_t, u_k, e_g, k, log σ^beh), j_t) | resolved verified outcome pairs (label channel) | κ, outcome encoder ω (optional) | b_t detached; σ logits detached |
| L_outcome-encoder = (r̂_ω(u) − ΔU^ver)² + L1 bottleneck | verified outcomes | ω | — |
| L_return (RUDDER) = L_m + (L_c + L_e + L_a)/10 on resolved windows | completed windows only | ξ | inputs detached |
| L_policy (credit) = −Σ_t Σ_j σ_θ(j\|s̃_t) sg(G^cred_t(j)) [R-HCA form] or the COCOA form | on-policy or lag-bounded data | θ (selector + core via σ) | coefficients detached |
| L_inner (meta windows) = actor-critic on r^ψ, functional | real experience in the window | θ_fast (functional copy) | none through r^ψ (meta-gradient wanted); ψ never in the actor optimiser |
| L_meta = −Σ_{query} sg(A^ver) log σ_{θ_K(ψ)} − Σ_{inner rollouts k≥1} sg(A^life) log σ_{θ_k(ψ)} + V_L regression | query + inner rollouts, verified utility | ψ, V_L | advantages stop-grad |

Exactly one of {COCOA, R-HCA, RUDDER-immediate, GAE} provides G^cred for a given utility stream; Exp. C decides the
default. A mixture is allowed only as a declared estimator validated against verified downstream performance.

---------------------------------------------------------------------------------------------------------------------

## 10. Conformance notes on code committed during this audit (6b02749, e7f454b, d975c10)

The master began implementing while this note was written. Checked against the papers (read-only; 13 tests in
`tests/cam_unified/credit` pass in 4.7 s):

1. `credit/estimators.py:86–98` `hca_return_grad` computes Σ_a ∇π(a|S_t)(h(a|S_t,Z_t)/π) Z_t — that is COCOA's
   **unbiased "Counterfactual Return"** estimator (App. F.3, Eq. 99), not Harutyunyan et al.'s HCA-return (Eq. 8 above,
   biased; the docstring says "Return-conditioned HCA (Harutyunyan et al. 2019)"). Rename, and add the true HCA-return
   with the §8.2 bandit test. It is not tested at all at present.
2. **R-HCA is absent** from `credit/estimators.py` and `credit/exact.py`. It is a two-line addition: occurrence counts
   including k = 0 (`now + D[t]` in `future_outcome_counts`) and the all-actions form with coefficient ratio (not ratio−1)
   on every k ≥ 0. `exact.py` adds Gaussian reward noise not encoded in U, so U = R hindsight is not enumerable there:
   add discrete-noise rewards (as E1) or the analytic Gaussian posterior.
3. `credit/meta_exact.py:99–123` `sampled_meta_gradient` differentiates only the query log-probabilities under θ_K;
   it omits the score term of the inner data distribution (b_k ~ π_{θ_k(ψ)}, k ≥ 1), which Zheng's formulation
   contains because rollouts 1..N are evaluated under θ_k(η). Measured on the committed test MDP (key_to_door d=5),
   2000 repetitions, n_inner = 64: K = 4 relative error 0.015, max |z| = 10.2 (biased); with the term
   Σ_{k≥1} Σ_i ∇_ψ log π_{θ_k}(b_k^i)·(Ĵ_query − c) added: 0.007, max |z| = 1.5. Also: self-inclusive batch-mean baselines
   (`:113`, `:120`) scale the inner/outer gradients by (1 − 1/N) — use leave-one-out means. The test
   (`tests/cam_unified/credit/test_credit_meta_exact.py:30`) accepts cosine > 0.95, which cannot detect a 60 % magnitude
   error (§8.5) — test relative error and z-scores.
4. `credit/heads.py:47–66` `HindsightModel` has no policy-logit input / gated policy prior and no elapsed-time input; both
   COCOA (Prop. 5, App. E.8.1) and R-HCA (App. F.1) report that the prior is needed to track the moving policy.
   Add `logp_policy` and `k`/Δτ inputs and the gate g·f + (1−g)·log σ. Coefficient clipping `:75` to [−1, 10] matches
   R-HCA's ratio clip 10.
5. `credit/heads.py:142` `ReturnDecomposer.loss` regresses symlog(G) with smooth-L1 and `:150` symexp's the prediction:
   that is neither E[G|H_t] nor monotone-difference-preserving, so the decomposition cannot be RUDDER-optimal even with
   infinite data (κ ≠ 0); conservation still holds through the residual. Use a squared loss on linearly scaled G
   (RUDDER: r·10/g_max) or the mean of a distributional head; add L_e and the auxiliary L_a; add the lessons buffer and the
   quality gate if used inside PPO. A causal transformer replaces RUDDER's gate-less LSTM — legitimate but unvalidated
   (RUDDER's argument for LSTMs is that predictions change only when information is stored); test with E1/E4.
6. `selfeval/intrinsic.py`: running-statistic normalisation of R_ψ must be frozen inside a meta window (§7.7); the reward
   takes e_a (action embedding) — Zheng's transfer to new action interfaces relied on rewarding state changes, so
   include an ablation without e_a in Exp. G; Zheng's ablation says the reward needs lifetime memory (FF on the current
   step fails) — ensure R_ψ receives a lifetime state not reset at episode boundaries.
7. None of `credit/*` or `selfeval/*` is imported by the runtime (`agent.py`, `model.py`, `learn/*`) — not integrated.

---------------------------------------------------------------------------------------------------------------------

## 11. Reproduction

    /workspace/agentrs/.venv/bin/python -I /tmp/claude-0/audit_research/scripts/credit_exact.py 0.6     # also 0.0, 1.5
    /workspace/agentrs/.venv/bin/python -I /tmp/claude-0/audit_research/scripts/metagrad_exact.py
    /workspace/agentrs/.venv/bin/python -I /tmp/claude-0/audit_research/scripts/audit_meta_sampled.py 2000 64 1,2,4
    # set torch threads to 1 on the shared machine; the scratch directory is not durable — port the tests (§8.1) into
    # tests/cam_unified/credit/ to keep them.
