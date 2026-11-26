# CAM Live 1.0 — architecture and workstream contract

Binding for every workstream. The operator's specification is `CAM_LIVE_SPEC.md` (verbatim); requirement ids are
in `CAM_LIVE_COMPLETION_MANIFEST.json`. Code contracts: `src/cam_live/contracts.py` (records, service
interfaces), `src/cam_live/nn/api.py` (tensors), `src/cam_live/config.py` (configuration). Those three files,
`diagnostics/world.py`, `tests/conftest.py` and this document are master-owned.

## 0. Ground rules

1. **Standalone.** Nothing under `cam_live/` imports `superlearner`, `crucible` or `agentrs`. Tested code from
   those packages may be *copied* into your subpackage with a provenance comment (file + commit `3968583`).
2. **No placeholders.** No `NotImplementedError`, `pass`-bodied stubs, mocked demonstrations, or tests that only
   exercise doubles. If something cannot be finished, it is reported as unfinished in your status file — never
   presented as done. The completion checker scans claimed files and import-reachability from the runtime.
3. **No strategy.** No Discord-growth tactic, niche, posting schedule, marketing framework, community format or
   psychological taxonomy appears anywhere in code, prompts, configs, tool descriptions or tests. Tool
   descriptions say what a tool *does*, never when or why to use it. Nothing in the environment or runtime returns a
   scalar reward or a bonus for "having built a website / posted / called a tool".
4. **Predictions are not outcomes; imagined is not real; bootstrap is not meta-learned.** Each has its own type or
   flag in the contracts. Never synthesise an outcome to complete a training target; mask it.
5. **Environment content is untrusted data.** It is encoded and stored; it is never concatenated into anything
   that is interpreted as an instruction about objectives, permissions or execution, and it never reaches the
   authorization service as anything but the literal action to be judged.
6. **Honest evidence.** Every number in a status file or report comes from a command that was actually run, with
   the command recorded. Failed scientific hypotheses are recorded as failures; they do not block unrelated
   engineering.
7. **Ownership.** Edit only your own paths (table below). For anything else write a change request in
   `docs/change_requests/<workstream>.md` (what, why, exact proposed diff) and work around it locally meanwhile.

### Workstreams and paths (all under `cam_live/`)

| Id | Scope | Owns (src/cam_live/…, tests/…) |
|---|---|---|
| A | Neural architecture | `nn/` (not `api.py`), `tests/nn/` |
| B | Prediction and decision-making | `predict/`, `tests/predict/` |
| C | Intrinsic reward and meta-learning | `reward/`, `tests/reward/` |
| D | Retrospective credit and continual learning | `credit/`, `tests/credit/` |
| EB | Computer environment: adapter, RPC, browser, perception, staging sites | `harness/env.py`, `harness/rpc.py`, `harness/browser.py`, `harness/perception.py`, `harness/staging/`, `tests/harness/env/` |
| EC | Sandbox, background jobs, visual computer use, grounding models | `harness/sandbox.py`, `harness/jobs.py`, `harness/visual.py`, `harness/grounding/`, `tests/harness/compute/` |
| EA | Authorization, credentials, approvals, limits, audit, e-stop, process separation, injection defence | `harness/auth/`, `tests/harness/auth/` |
| ES | Skills, modality selection, failure taxonomy, publishing, Discord, HTTP API | `harness/skills.py`, `harness/modality.py`, `harness/failures.py`, `harness/publishing.py`, `harness/discord.py`, `harness/http_api.py`, `harness/staging_discord/`, `tests/harness/skills/` |
| F | Live runtime: loop, tasks, verifier, logs, journal, dashboard, CLI, doctor | `runtime/`, `tasks/`, `tests/runtime/` |
| S | Scientific evaluation: baselines, experiments, transfer | `eval/`, `tests/eval/` |
| M | Master: contracts, `agent.py`, integration, manifest, checker | everything else |
| G | Independent verification (reads and runs; writes only `docs/verification/`) | `docs/verification/` |

Each workstream writes `docs/status/<Id>.json` (schema in §9) and may add its own `conftest.py` inside its test
directory.

## 1. Processes and trust boundaries

```
 operator ── dashboard/CLI ──► runtime (loop, logs, scheduler)            uid camlearn
                                  │  AgentAPI (in-process)
                                  ▼
                               learner: backbone+LoRA, core, heads, R_psi, replay, checkpoints   (GPU)
                                  │ ExecRPC v1 (unix socket)        │ AuthRPC          │ VerifyRPC
                                  ▼                                 ▼                  ▼
        computer environment (executor)                  authorization service    independent verifier
        ├─ Chromium via Playwright  (uid cambrowser)     policy, grants, limits,  reads permitted evidence
        ├─ sandbox jobs             (uid camsbx)         approvals, audit, e-stop  directly, signs outcomes
        ├─ visual grounder (frozen model)                └─ secret broker (vault)
        └─ api / publishing / discord / skills
```

* The **learner never holds a secret**. Credentials live in a vault directory readable only by the authorization
  service's uid; executors receive a secret at the moment of use through `ExecContext.secrets`, and every
  observation is scrubbed of known secret values before it is returned.
* Every external effect passes `PolicyEngine.authorize` (signed grant bound to the action hash) **and**
  `check_primitive` for each primitive operation inside it, whatever the modality. A skill never has more
  permission than the same operations issued one by one.
* The **verifier** is not a model. It signs `VerifiedOutcome`s with a key the learner's uid cannot read; only signed
  outcomes may supervise the evaluator and the outer (meta) objective. The audit log and verifier log are
  hash-chained and not writable by the learner's uid.
* This instance is an unprivileged container (no user namespaces, no `CAP_NET_ADMIN`, no `CAP_SYS_ADMIN`; we do
  have setuid/setgid, Landlock and seccomp). Isolation therefore uses separate uids + file modes + rlimits +
  Landlock + seccomp. Kernel-level per-destination network filtering is **not** available here; record it as a
  limitation and recommend a VM for live operation. Everything also runs in a single process ("inproc"
  transport) for tests; the separated mode is what `cam-live run` uses and has its own tests (marked
  `privileged`).

## 2. Neural architecture (A)

* **Backbone.** `Qwen/Qwen3-4B` (4.02 B parameters, Apache-2.0, local snapshot pinned in `BackboneConfig`),
  frozen, bf16. `Qwen/Qwen3-8B` is a config change. `name="tiny"` is a small random causal LM used only for CPU
  tests of mathematics and plumbing; it is never reported as the backbone.
* **Semantic adapters.** LoRA on attention projections across all layers, target 10–20 M trainable parameters
  (rank 32 on q/v ≈ 11.8 M on Qwen3-4B; report the measured count), plus the **latent bridge**: a trainable map
  from CAM's belief `b` to `bridge_tokens` virtual prefix embeddings (and optionally gated cross-attention into
  decoder layers) that conditions generation. The bridge must start as an exact no-op (zero-initialised output,
  not a text-initialised prefix: text-initialised prefixes leaked into generated strings before).
* **Semantic embedding.** Mean-pooled hidden states from several depths (`tap_layers`), cached by text hash when
  `with_grad=False`; the cache is invalidated when adapters change.
* **Experience tokens.** One token per `ExperienceItem`: semantic projection + item-kind embedding + continuous-time
  encoding + a **separate value channel** for exact quantities (sign, symlog magnitude at several scales and a
  learned per-metric-name hash embedding), fused after separate normalisation so content cannot swamp values (it did:
  cos(+1,−1)=0.98 before). Raw values stay in `TokenBatch.values`.
* **Experiential core.** Pre-norm transformer: RMSNorm, SwiGLU, causal self-attention with rotary positions over
  event index plus a learned function of elapsed time, cross-attention from the token stream to persistent state
  slots, and a **gated** slot update (gate bias negative at init, small init std) carrying state between calls.
  A direct path from the current event to `b` (attention-mixed history alone could not read the current step
  before). Default `50m` preset ≈ 50 M parameters including tokenizer projections, decision trunk and feature map;
  presets `debug … 1b` in `config.SCALE_PRESETS` (adjust `d_ff`/layers to hit targets; report actual counts).
* **Decision trunk and features.** `decision_latent(b, e_a, e_g) → z` and `features(z, e_q) → phi` are shared by the
  world model, evaluator, policy, VOI and reward: one representation, several light heads.
* **Fast adaptive state** `m_t` = core slots + `FastBeliefState`: Bayesian last-layer posterior(s) `w ~ N(mu, Sigma)`
  over `phi`. Gaussian targets: exact conjugate update in Joseph / Cholesky form, float64. Count, binary and
  time-to-event (with censoring, via the Poisson-exposure identity) targets: assumed-density (Laplace/EKF) GLM
  updates. Forgetting factor / process noise for changing mechanisms, with change detection. The prior mean and
  covariance are learned parameters (`experiential` group) trained ALPaCA-style on held-out later records of the
  same context, using each context's full-data scale (support-estimated spread rewarded variance inflation before).
  It must be compared against exact Bayesian linear regression on raw features, a GRU adapter and running statistics,
  on held-out families; whichever wins is reported.
* **Parameter groups** are exactly `nn.api.GROUPS`; every optimizer, log line, checkpoint and hash uses them.
  **Per-group gradient clipping** always (a global clip scaled core updates by 1e-4 before).

## 3. Prediction and decision-making (B)

* **World model** `p(y | z, q, horizon)`: one head per outcome kind (normal / Student-t; negative binomial with
  exposure; Bernoulli; discrete-time hazard with right-censoring; categorical), queried by `e_q` so targets are not a
  fixed list. Output combines the head and the Bayesian fast layer; epistemic and aleatoric parts are reported
  separately (small bootstrapped ensemble or last-layer posterior — state which). Training: negative
  log-likelihood on observed outcomes with masks for pending/unobservable ones; latent-consistency loss for the
  short-horizon latent transition used in imagination. Classify error sources where the evidence allows
  (noise vs. missing observation vs. changed conditions via change detection).
* **Evaluator** `V(b, g, horizon, policy_ctx)`: success logit (BCE-with-logits on *verified* objective outcomes,
  applied to every earlier prefix), plus magnitude, time-to-achievement (censoring-aware) and resource heads.
  Calibration utilities: Brier, NLL, calibration slope/intercept, reliability bins, by objective family and horizon;
  only prospectively registered predictions are scored. `ΔV` is exposed as an input to the reward generator, not as
  a reward.
* **Policy** `pi(a_i | b, g, C_t)` over a variable candidate set, with recorded probabilities for every candidate.
  Exploration rules: Thompson sampling from the fast posterior, UCB, greedy learned value, decision-relevant
  information gain, and explicit randomised assignment for controlled experiments — all selectable, all logging
  propensities. The policy must depend on CAM's latent: a test permutes `b` across a batch and requires the action
  distribution to change (KL was 4e-9 before, i.e. the LLM decided everything).
* **Candidates.** The backbone proposes typed `ActionSpec`s from the discovered tool descriptors, the current
  observation digest and the latent bridge; a deterministic schema-walker also proposes valid argument settings so
  the agent is never limited to what the LM happens to write; the skill library contributes procedures. Every
  candidate records its provenance. Parsing is strict JSON → `ActionSpec` → `EnvironmentAdapter.validate_action`.
  Novel multi-step programs are `ActionSpec.steps`.
* **Imagination / planning.** Short latent rollouts with uncertainty; never decodes text; horizon grows only while
  measured multi-step error stays under a threshold; outputs carry `imagined=True` and cannot be written through any
  outcome API.
* **VOI.** `VOI(a) = E_y[max_a' Q(b_updated(a,y), a', g)] − max_a' Q(b, a', g)` by sampling `y` from the fast
  posterior and applying its update, minus cost/time/opportunity cost; plus a learned estimator of downstream
  usefulness that is periodically scored against realised downstream learning. Raw prediction error is never used as
  value (noisy-TV test).
* **Experiments.** The agent registers `ExperimentRecord`s (question, hypothesis, arms, predictions) before acting;
  randomised assignment is available on owned resources.

## 4. Intrinsic reward and meta-learning (C)

* `r = R_psi(b_t, e_a, b_next, e_g, signals, life_ctx)`: learned nonlinear function, bounded output, normalisation
  statistics frozen inside a meta window, versioned. `signals` is the named vector in `reward/signals.py`
  (success estimate before/after, ΔV, action-quality estimate, information gained, VOI, cost, elapsed time,
  prediction error, epistemic uncertainty before/after, execution status, …); `life_ctx` is a recurrent summary of
  the lifetime that is **not** reset at objective boundaries.
* **Inner update (real).** `theta' = theta − alpha ∇_theta L_inner(theta; psi, D_support)` with
  `L_inner = −Σ log pi_theta(a|·) · A_psi + value loss on intrinsic returns`, executed functionally
  (`torch.func.functional_call`) over the adaptable subset `theta_adapt` (policy head + decision trunk by default;
  ≤ 50 M parameters, K ≤ 8) on cached latents — never through the backbone. The same update is what the
  continual learner actually applies to the live policy.
* **Outer update.** `J_outer(psi) = E[U_verified(D_query; theta'(psi))]`, estimated with the policy-gradient
  surrogate `Σ log pi_theta'(a_q|·) · (U_verified − baseline)` on query decisions taken **after** the inner update
  (on-policy), or with clipped importance weights and a support check when the acting policy differed. Includes the
  inner-data score term when support actions were sampled from `pi_theta`. `U_verified` comes only from signed
  verifier outcomes/measurements. Reference: Zheng et al. 2020; exact tabular implementation with finite-difference
  agreement, **relative-error and z-score** checks for K ∈ {1,2,4,8} (cosine alone hid a 60 % bias before), and a
  test that detaching the inner reward or removing the unroll zeroes the gradient.
* **Windows from one lifetime.** Support/query windows are separated in time; independence is counted per
  experiment/parent decision, not per row; a meta-update needs `meta_min_independent_outcomes` verified query
  outcomes, otherwise it is skipped and logged with the reason.
* **Cold start.** Bootstrap signal = self-supervised predictive learning progress + grounded ΔV + uncertainty-aware
  exploration + sparse verified outcomes, labelled `bootstrap`. The mix `bootstrap_mix` moves toward the learned
  reward only when held-out query evidence shows the learned reward does not hurt; it can move back.
* **Integrity.** The inner loss never updates `psi`; `psi` has its own optimizer fed only by the outer path; outputs
  bounded and scale-normalised so magnitude cannot be inflated; outcomes de-duplicated by `dedup_key`; verifier
  signature required; repeated trivial self-goals discounted; noisy-TV and fabricated-outcome tests.
* **Goal-free mode.** Budgeted self-chosen learning goals from uncertainty and achievable progress; their usefulness
  is judged on later independently assigned objectives.

## 5. Credit and continual learning (D)

* **Store.** SQLite (WAL) + blob directory implementing `contracts.ExperienceStore`; items, experience records,
  predictions, windows, provenance (`live`/`staging`/`diagnostic` never mixed silently), model versions, eligibility.
* **Linking.** Delayed outcomes attach to earlier decisions through evidence (attribution keys in events, artifact
  ids, experiment arms, time windows); ambiguous evidence yields fractional links with stated uncertainty — never
  "the latest action caused it".
* **RUDDER.** Sequence model `F_xi(H_t, g)` of the eventual verified return, squared loss on the scaled return;
  redistributed credit `c_t = F(H_{t+1}) − F(H_t)` with initial offset and residual correction; conservation checked
  under duration-aware discounting `exp(−beta Δt)`; the *redistributed return-to-go* is used for learning (using
  the difference as an immediate reward biased the key decision to zero before).
* **COCOA.** Exact finite-action reference (validated against enumerated gradients), then a hindsight model
  `h(a | b, u)` with a policy-logit prior and elapsed-time input giving `w = h/pi − 1` over probability-tracked
  candidate sets; refuses (returns "unsupported") where propensities or action support are missing; associations,
  counterfactual estimates and experimentally supported effects are separate fields.
* **Default estimator is chosen by evidence**: simple delayed attribution first; RUDDER/COCOA are exercised and
  compared with randomised-experiment differences, and promoted only when validated.
* **Learner.** Three schedules: fast (belief update, no gradient), frequent (experiential + heads; triggered by
  matured informative outcomes, not by wall-clock or by every correlated row), slow (adapters every
  `adapter_every` frequent updates; `psi` only through the outer path). Loss balancing so the world loss does not
  take >90 % of the core gradient. Replay strata: recent / informative older / successes / failures / delayed /
  objective families. Every update writes a `LearningUpdateRecord` with before/after hashes, per-group deltas and a
  behavioural probe (did predictions or action probabilities actually change).
* **Retention and checkpoints.** Anchor-set functional regularisation + replay; forgetting is measured, not assumed.
  Atomic checkpoints (tmp dir → fsync → rename), architecture signature with tolerant loading of newly added heads,
  promotion gated by independent validation and retention, rollback, weights-only export for transfer.

## 6. Computer environment (EB, EC, EA, ES)

* `harness.env.ComputerEnvironment` implements `EnvironmentAdapter` by composing `ModalityExecutor`s:
  `structured_browser` (EB), `visual` and `code` (EC), `api` and `skill` (ES), `internal` (wait / inspect, EB).
  It owns observation ids, staleness detection, the event stream, pending-outcome bookkeeping, complete execution
  logging and session recovery, and exposes the same interface in-process and over `ExecRPC v1`.
* **Browser (EB).** Playwright Chromium, persistent context per authorized identity, multiple tabs, isolated
  contexts for read-only research, CDP attach to an external Chromium. Observation = URL, title, visible text
  digest, accessibility tree, interactive elements with stable per-observation refs, forms, links, tabs, diff
  against the previous observation; screenshots on demand. Primitives per spec 12A.2. After consequential actions the
  resulting state is inspected and `ActionSpec.verify` evaluated.
* **Visual (EC).** Screenshot + mouse/keyboard primitives; `VisualGrounder` implementations are frozen, pinned by
  revision, licence-checked and code-reviewed before use; a grounder only proposes coordinates, the harness validates
  them against the current observation and policy and then acts. Benchmark on the staging sites.
* **Sandbox (EC).** Python and shell under a dedicated uid with rlimits, Landlock filesystem scope and seccomp
  (no-network by default); background jobs with purpose, limits, deadline, status, cancellation and persisted results.
* **Authorization (EA).** `harness.auth.PolicyService` implements `PolicyEngine`: scoped credentials, allowed
  accounts/destinations, platform-compliance rules, publishing approvals, spending and rate limits, data-use
  permissions, hash-chained audit log, pause, emergency stop, revocation, prohibited classes of spec 12.6.
  Fail-closed. `mode="live"` needs an explicit operator authorization record.
* **Skills (ES).** Generated from successful traces: parameter abstraction, pre/postconditions, sandboxed
  validation, reliability posterior, versions, staleness detection, repair/retire; never more permission than their
  primitives. **Modality selection** starts from the spec's ordering as a prior and is calibrated from experience
  (success probability, latency, compute, primitive count); it is never used to route around a denial.
  **Failures** are classified into `FailureClass`; bounded retries for transient ones; none for `HARD`.
* **Publishing / Discord (ES).** Draft → preview → approval → publish. Discord only through the official bot REST
  API; aggregate metrics; bots and operator-listed test accounts excluded from counts; message content is not
  stored or trained on unless `data_permissions` says so. Without credentials the tools are listed as
  `available=False` with the reason, and contract tests run against the local staging service.
* **Staging (EB).** Local websites exercising: structured HTML, visually rendered controls (canvas), dynamic
  changes, multi-step workflows, unfamiliar variants, interface changes, and delayed consequences.

## 7. Runtime (F) and evaluation (S)

* `runtime.loop.LiveRuntime` drives `AgentAPI` through the 17 steps of spec 15.1 against any
  `EnvironmentAdapter` + `PolicyEngine` + `OutcomeVerifier`; waits on the pending-outcome queue without generating
  activity; enforces `TaskSpec.limits` and stop conditions; checkpoints and pauses on exhaustion; resumes after a
  kill. Logs per `contracts.RunFiles`; journal generated from records (facts vs. inferences marked).
* CLI `cam-live` with every command of spec Part 17; FastAPI dashboard with the controls of 15.3; `doctor`;
  preflight that reports exactly what is missing for live operation.
* `eval/`: baselines A–E, calibration, delayed-credit, meta-gradient and reward-integrity experiments,
  learning-efficiency (`N_q`, `S_q`, AULC with studentised intervals) and fresh-process weights-only transfer.

## 8. Engineering rules

* Python: `/workspace/camlive/.venv/bin/python` (torch 2.15 cu132, transformers 5.24, playwright 1.65 + Chromium).
  Install extra packages only into this venv: `uv pip install --python /workspace/camlive/.venv/bin/python <pkg>`.
* Tests: `cd /workspace/camlive && .venv/bin/python -m pytest cam_live/tests/<yours> -q --rootdir=cam_live`.
  Default tests are CPU, deterministic (seeded) and fast (< 5 min per directory). Mark `gpu`, `slow`, `browser`,
  `privileged`, `network` as appropriate.
* **GPU is shared with another project.** Small models (< 4 GB) may use it freely. Anything loading Qwen3-4B or
  using > 4 GB must take the shared lock (`tests/conftest.py::gpu_lock` / fixture `gpu_heavy`), hold it ≤ 15 min,
  stay under 24 GB, and release memory. Never wrap pytest in `flock`.
* Do not touch `/workspace/agentrs` (another session works there), its venv, other worktrees, or any running
  process. Do not push. Do not contact any external service except: PyPI, Hugging Face downloads, and read-only
  fetches of public documentation needed for your work. No Discord API calls, no posting anywhere.
* Commit your own paths early and often on the current branch:
  `git add <your paths> && git commit -m "cam live [<Id>]: <what>" -- <your paths>` (retry if the index is locked;
  never `git add -A`, never amend or rebase, never commit another workstream's files).
* Record evidence: write JUnit XML of your final test run to `docs/evidence/test_results/<Id>.xml`
  (`--junitxml=`) and any measured numbers to `docs/evidence/<Id>/…`.

## 9. Status file `docs/status/<Id>.json`

```json
{"workstream": "B", "commit": "<sha>", "summary": "...",
 "requirements": {"PI-01": {"state": "implemented|partial|not_started",
                             "files": ["src/cam_live/predict/world.py"],
                             "integration_points": ["cam_live.predict.world.WorldModel.predict"],
                             "tests": ["tests/predict/test_world.py::test_learns_action_conditioned_counts"],
                             "commands": ["..."], "evidence": ["docs/evidence/B/…"],
                             "limitations": "…"}},
 "public_api": ["signature …"], "change_requests": 0,
 "measured": {"…": 0.0}, "not_done": ["…"], "scientific_failures": ["…"]}
```
