# ST — autonomous strategic invention (amendment 2026-11-28, item 5) and agent.py integration

Owner of `src/cam_live/agent.py` (integrator), `src/cam_live/predict/candidates.py`, `src/cam_live/eval/proposal_probe.py`,
`tests/integration/test_st_*.py`. Every number below names the file or command it comes from. "Live run" =
`runs/cam_live/live/discord-100-planning-20261128T083454Z` (read-only for ST; analysed with
`python -m cam_live.eval.proposal_probe analyse-run <run>`), code d1d06dc (agent.py identical to 87eab34).

## 1. Why CAM did not research, hypothesise, plan or sustain strategies (investigation)

Measured on the live run's first 35 decisions (Qwen3-4B + 50m, open-ended generation on), plus the code path:

| cause | evidence | kind |
|---|---|---|
| **Truncated generation.** One 256-token answer was asked for 8 alternatives (`proposer_k` 8, `proposer_max_new_tokens` 256): ~32 tokens per action | 34 of 35 decisions had a cut-off last line (`not_json`, "Unterminated string ..."); 146 items written, 4.1 valid per decision; an action carrying content (a note, a file, a request description) could not be written beside others — the backbone proposed `internal.note` **0** times in 35 decisions | architectural (budget shared by all alternatives) |
| **Action space polluted with actions that can never execute.** The agent filtered tools by the task's interfaces only, not by `resources.authorized_tools`; the schema walker took the authorization's destination PATTERN `https://*` as a URL | 14 of 35 chosen actions were walker `http.request` (http.* not authorized: P-TOOL-NOT-AUTHORIZED), 4 were navigations / downloads to `https://*/` ("bad host '*'"); 33 of 35 chosen actions came from the walker, 2 from the backbone | defect (agent) |
| **No memory of its own work in the generator's context.** The observation digest holds the latest 16 observations / results within 3000 characters; an executed note shows as `internal.note: {"characters": n, "saved": ...}` — its text never re-enters a prompt; earlier actions appear without their arguments | 13 identical repeats of an executed action in 35 decisions; 0 follow-ups that reuse a value first seen in an earlier result | architectural (context management) |
| **Selection cannot value a research step or a note in this phase.** `thompson_policy` scores candidates by Thompson samples of the predicted objective metric (standardised within the decision) plus the policy's preference weighted by a calibration on *verified progress*. In the planning phase the objective metric is 0 at every reading: the policy's weight stays at its prior, and the world model's goal values differ only by fitted noise | first decisions: near-uniform choice over the 16 offered candidates; later: walker `http.request` predicted 0.3-0.8 vs ~0.13 for every other candidate (a metric that was 0 every time) and chosen with probability up to 0.64 — the run concentrated on an action that was refused every time | empirical limitation of goal-value selection without outcome variation (+ noise-fitting); IMP's imagination now adds information value (IMP.md) |
| **The generator never learns as a generator.** LoRA and the latent bridge receive only predictive objectives (`world_model_adapters`, `evaluator_adapters`, `bridge_grounding`); nothing in the learner makes the backbone more likely to propose what CAM's own experience rated well (spec 1.4 "support gradual improvement of the action generator through the pretrained-model adapters") | `credit.learner` slow providers list (MODEL_ARCHITECTURE §7) | architectural gap |

What the backbone did write (live run, 35 decisions): `internal.request_operator` 45, Discord reads (`get_guild_settings`,
`list_channels`, `list_roles`, ...) 80+, `discord.modify_guild` 7, `create_channel` 3; never a browser navigation, never
a sandbox action, never a note.

## 2. Hotfix (commit a0f3411; master's request; live run paused at 35 decisions)

* Tools outside `resources.authorized_tools` (names, globs, modality names, read as `harness.auth.evaluate` reads
  them; a program is judged by its steps) are neither listed in the prompt nor offered (`CamLiveAgent._tool_authorized`,
  `_permitted`, `_feasible`: counted as `tool_not_authorized`).
* Authorization patterns (any value with `*`) are never walker vocabulary (`_resource_vocab`).
* Generation: `proposer_samples` = 4 independent continuations in one batched `generate` call, each asked for
  `proposer_k` = 2 alternatives with `proposer_max_new_tokens` = 384 (8 alternatives per decision as before, 192
  tokens each instead of 32); each continuation parsed on its own and de-duplicated; per sample: lines, items, valid,
  whether its last line was rejected; the raw text of each sample recorded with the decision
  (`decision.diagnostics["generation"]`).
* Regression tests `tests/integration/test_st_hotfix.py` (4 of 5 fail on 87eab34; the fifth is the control "no tool
  list -> interfaces decide").

## 3. Item-5 architecture changes (in the code since a0f3411, inactive by default until the staging evidence)

* **OWN RECORDS** (`agent.proposer_memory_chars`, `CamLiveAgent._memory_digest`, header `predict.candidates.H_MEMORY`):
  a data section of the proposer prompt with every executed action of the objective, oldest first — operation,
  arguments, the expectation it was proposed with, status / refusal detail, the beginning of what it returned — and,
  for what the agent wrote into its own records (tools the environment describes as internal with local side effects:
  the notebook, operator requests), the text it wrote. Within its budget the oldest other actions go first, its own
  written records last; the prompt fitting shrinks it together with the observation digest. Nothing is added by the
  agent: it is what was done, proposed and observed. Persisted with the objective's operations (checkpoint).
* **`expected` in the output format** (`agent.proposer_expected`, `PROPOSAL_INSTRUCTIONS_EXPECTED`): the action
  contract's optional field; a proposal can carry its own prediction, which is kept (`ActionSpec.expected`), rendered
  into the action's encoding (core, world model, policy, R_psi see it) and shown next to the actual result in later
  prompts.
* **The generator learns from CAM's experience** (`agent.proposal_learning`, `predict.candidates.ProposalLearningProvider`,
  slow timescale, LoRA + bridge): advantage-filtered regression on the agent's own successfully executed actions,
  w_i = max(0, A_i), A_i = standardised `intrinsic_reward` (the reward the live policy update uses: bootstrap / R_psi
  mix) within the objective; context = the decision's own experience items + objective under the proposer's headers
  and instruction block, conditioned on the decision's belief through the bridge. Held-out validation, the slow-step
  guard and its trust region apply as to every adapter loss.

* **Per-sample tool-order rotation** (`agent.proposer_rotate_samples`, off by default): sample i reads the same
  complete tool list starting at position i·n/samples (one batched generate call over the rotated prompts). Reason:
  on the real model the 4 identical-prompt samples at temperature 0.7 wrote the same two actions in most decisions.
* **Realised information value** (`agent.information_value`, on; `CamLiveAgent._information_value`): see 3.1.

### 3.1 Can an information-only action be valued at all? (master's question, 12:10 UTC)

Read in the code at HEAD:

* **(a) VOI / information gain** are computed for every candidate, internal and research actions included
  (`Predictor._voi`), but only over the candidate's outcome *queries* — `action.succeeded`, the objective metric after
  the action, the next readings of watched metrics. The content an action returns (a page's text, a note) is no query
  and has no VOI. VOI enters the selection only under rule `voi` (goal-free mode); `thompson_policy` uses Thompson
  samples of the goal value, the policy's preference and (IMP) imagined goal values. R_psi's signal vector contains
  `info_gain` and `voi`, but at bootstrap mix 1.0 the live reward is the bootstrap: learning progress of the
  decision's own outcomes, delta_v, exploration = the drop of the *goal-value* epistemic std after the objective
  reading, verified progress. For a note or a fetched document each of these is what it is for any action.
* **(b) The content is encoded**: the action item (rendered arguments, <= 2,000 characters) and the result's
  observations (<= 2,000 characters) enter the core's context window (<= 256 items) through the frozen backbone's
  embedding (384 tokens, head and tail) -> b -> z -> every prediction. But nothing measured whether it changed later
  predictions, and nothing credited the action that brought it. **An action whose only effect is information had no
  pathway to be valued.**

Fix (generic; no tool is named, nothing tells the generator to write anything): for every executed decision t, once
the next 2 decisions of the objective whose context windows contain the items t added have matured outcomes,

    IV_t = NLL_world(their outcomes | context without t's items) - NLL_world(their outcomes | recorded context)

(same weights, same targets, nats; only the context pathway is removed), written once as
`reward_signals.information_value`. Change request **ST-3** (RW; applied by RW in a055332) makes it a bootstrap input like learning progress
(`w · tanh(IV)`), so the reward the live policy update uses values what an action's result did to CAM's own later
predictions; its path into the selection is then the existing one (bootstrap reward -> `MetaLearner.policy_update` ->
policy preference -> `thompson_policy`). A pre-action estimate (the unused `predict.voi.DownstreamValueEstimator`
head, "realised downstream usefulness") could be trained on IV — designed, not built (IMP's predict/).

None of these names a tool, a niche, a strategy or a step order; the strategy scanner finds nothing in any new
constant text (`test_new_constant_text_carries_no_strategy`).

## 4. agent.py integration (ST is the integrator)

| request | applied | commit | check |
|---|---|---|---|
| IMP (imagination in the decision: `plan(..., predictor=, evaluations=)`, `ImaginedEvaluation.extras()` into every candidate's extras, `planner.state()` / `load_state()`) | yes | dde7d2d | `tests/predict/test_imp_agent.py` 2/2 on a clean snapshot (skipped before) |
| PS-1 (persistent core state chain: `_live()` around decision-time core calls, `commit` per decision, checkpoint / restore report, status, close) | yes; hunks for `evaluate()` (conflict with IMP's planner call) and `status()` merged by hand | dde7d2d | `tests/nn/test_ps_agent_integration.py` 2/2, `test_agent_recovery` 2/2 |
| RW-1 / RW-2 (`meta_restored_fresh` in the restore report; `reward_validity` in status; getattr-guarded) | yes | dde7d2d | tests above; guarded when RW's meta-learner lacks the attributes |
| SG-1..3 (`skill.run` procedural; transfer evidence of assigned objectives; reference prior after restore) | yes | dde7d2d | getattr-guarded; SG's own tests |
| CR-1 / CR-2 / CR-3 (segment split as the agent's default, `segment_decisions` from the configuration, replay of decisions whose credit was revised in the real policy update) | yes, after CR committed credit/ (e8dee3f) | f983cea | `tests/credit/test_cr.py` on the snapshot below |

Clean snapshot of HEAD 00f8913 + the integrated agent.py of dde7d2d: 33 passed, 1 skipped (`-m slow`), 0 failed
(`test_st_hotfix`, `test_st_strategy`, `test_int2_prompts`, `test_int3a_prompt`, `test_imp_agent`,
`test_ps_agent_integration`, `test_agent_recovery`). Clean snapshot of HEAD df1b46e + agent.py of f983cea (JUnit
`docs/evidence/test_results/ST_fix.xml`): 70 tests — 68 passed (the `-m slow` slow-step test run separately and
merged in: 1 passed, 33 s), 2 failed (the two PS-1 regressions below, identical on HEAD) — over `test_st_hotfix`, `test_st_strategy`, `test_st_iv`, `test_int2_prompts`,
`test_int3a_prompt`, `test_int3a_audit`, `test_int3a_bootstrap_inputs`, `test_int3a_lrn6`, `test_agent_recovery`,
`test_agent_runtime`, `test_int2_checkpoints`, `test_imp_agent`, `test_ps_agent_integration`, `tests/credit/test_cr.py`;
1,232 s.

**Regression found by the integrator (reported to the master, not fixed here):** with PS-1 applied (persistent core
state enabled by default) two existing integration tests fail on a clean snapshot of HEAD 1a4adb2 —
`test_agent_runtime.py::test_behaviour_changes_and_lives_in_the_weights` (restoring the initial weights no longer
restores the initial decision distribution: TV 0.0042, bound 1e-9) and
`test_int2_checkpoints.py::test_export_weights_reads_in_a_fresh_process` (fresh-process model from the weights-only
export differs by 6.4e-4, bound 1e-6). The same snapshot with `persistent_state.enabled` false passes both. Cause: the
carried core state is behaviour that is not in the trainable weights nor in the weights-only export — PS's design;
the tests encode the old property. Decision left to PS / the master.

Runtime observations reported by the master (not agent.py, not fixed here): a paused runtime did not apply a queued
stop for > 10 min; commands are applied only at step start, so under GPU contention a pause took ~9 min.

## 5. Evidence

### 5.1 Setup

Staging rehearsal of the planning task (`tasks/staging/discord_100_planning_staging.yaml`: the operator's objective
verbatim; staging Discord + neutral web of `tools/planning_rehearsal.py`), real configuration (Qwen3-4B bf16 + LoRA +
bridge + 50m core, open-ended generation on), `cam-live run --inproc`, one GPU hold per condition, stopped at 30
decisions or 11.5 min after the first decision. All conditions use `nn.max_prompt_tokens` 8192 (overlay; ST-1) and
seed 0. Driver: `<scratch>/ST/st_staging.sh` / `st_resume.sh`; run roots `runs/cam_live/st/<condition>`; measures:
`python -m cam_live.eval.proposal_probe analyse-run <run>` (evaluation-only counts, never shown to CAM) and
`planning_rehearsal.py check`.

| condition | code | agent settings |
|---|---|---|
| live (before) | d1d06dc (the live run itself, 36 decisions, live task) | HEAD defaults |
| before | d1d06dc snapshot | HEAD defaults |
| hotfix | a0f3411 snapshot | defaults of a0f3411 (authorized tools, no pattern vocabulary, 4 x 2 x 384 samples) |
| after | a0f3411 snapshot | + `proposer_memory_chars` 2400, `proposer_expected` true, `proposal_learning` true |
| after_nomem | a0f3411 snapshot | as after, `proposer_memory_chars` 0 (ablation of OWN RECORDS) |

### 5.2 Results

Live run 1 (d1d06dc, live task, 36 decisions; `analyse-run`): cut-off lines in 35 of 36 decisions; 18 of the first 35
chosen actions refused (14 unauthorized `http.request`, 4 `https://*/`); 33 / 35 chosen from the walker; 0 notes.
Live run 2 (a0f3411 = hotfix, the master's resumed live run, first 39 decisions, read-only): 1 cut-off line in 156
samples; 4 of 39 refused; 9 / 39 chosen actions written by the backbone; 6 operator requests; 0 notes; median 24 s per
decision.

Staging (one seed each; `before` got only 2 decisions in its hold because the machine was at load ~95 then):

| measure | before | hotfix | after | after_nomem |
|---|---|---|---|---|
| decisions | 2 | 30 | 30 | 30 |
| chosen backbone / walker | 0 / 2 | 8 / 22 | 7 / 23 | 10 / 20 |
| valid backbone proposals per decision | 5.5 | 4.63 | 3.93 | 5.63 |
| decisions with a cut-off line | 2/2 | 0/30 | 0/30 | 1/30 |
| cut-off lines / samples | 2/2 | 0/120 | 0/120 | 1/120 |
| notes executed / proposed | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| operator requests executed | 0 | 5 | 2 | 2 |
| chosen actions carrying `expected` | 0 | 0 | 6 | 10 |
| follow-ups on earlier results | 0 | 1 | 0 | 1 |
| identical repeats | 0 | 5 | 6 | 0 |
| changed retry after a refusal | 0 | 0 | 2 | 2 |
| experiments registered | 0 | 1 | 3 | 1 |
| results ok / queued / denied | 1 / 0 / 1 | 24 / 5 / 1 | 20 / 5 / 5 | 16 / 11 / 3 |
| slow (adapter) steps | 0 | 0 | 0 | 0 |
| median s per decision | (load ~95) | 19.1 | 18.6 | 23.4 |

Fifth condition **after_rot** (as after + `proposer_rotate_samples`, a0f3411 + the rotation change): 30 decisions,
7 / 23 chosen backbone / walker, 4.87 valid backbone proposals per decision, 2 cut-off lines / 120 samples, 0 notes
executed or proposed, 2 operator requests, 6 chosen with `expected`, 5 identical repeats, 5 experiments registered,
results 24 ok / 6 queued / 0 denied, 18.6 s per decision; 12 distinct operations proposed (8-10 otherwise), dominated
by `discord.create_channel` (85 of 134 proposals).

Notes on the measures: "cut-off lines" are lines the strict parser refused as not JSON (the token budget ended inside
them); "follow-ups" = chosen actions whose arguments reuse a token first seen in an earlier decision's result;
"identical repeats" = re-executions of an identical action; "changed retry" = after a refused / failed action, a later
chosen action of the same operation with other arguments; experiments = randomised comparisons the decider registered.
All are evaluation counts computed from the run records after the run; none is a signal to CAM.

Backbone proposals over each run (offered candidates; `<scratch>/ST/reprop.py`):

| condition | proposals offered | re-proposals of an executed action | distinct operations | most proposed, first half -> second half |
|---|---|---|---|---|
| hotfix | 139 | 12 (9 %) | 10 | request_operator 33, get_guild_settings 13, modify_guild 8 -> request_operator 30, get_guild_settings 14, list_channels 10, create_channel 6 |
| after | 104 | 17 (16 %) | 10 | create_channel 16, request_operator 10, get_guild_settings 9 -> request_operator 19, get_member_events 6, create_channel 6 |
| after_nomem | 169 | 0 (0 %) | 8 | modify_guild 32, request_operator 29, create_channel 9 -> modify_guild 33, create_channel 27, request_operator 15 |
| after_rot | 134 | 5 (4 %) | 12 | create_channel 45, request_operator 10, modify_guild 6 -> create_channel 40, get_guild_settings 5, request_operator 5 |

### 5.3 The four evidence levels per capability

| capability | (a) implementation correctness | (b) integration in the runtime | (c) causal influence on behaviour | (d) measured improvement |
|---|---|---|---|---|
| authorized tools only, no pattern vocabulary (hotfix) | `test_st_hotfix` (4 tests fail on 87eab34, pass on a0f3411) | live run 2 and staging run on the real configuration | refused actions: live 18/35 (d1d06dc) -> 4/39 (a0f3411); staging hotfix 1/30 | decisions no longer spent on actions the authorization service always refuses |
| independent samples with their own budget (hotfix) | `test_st_hotfix` (num_return_sequences, per-sample parse, de-duplication, raw text) | live run 2, staging | cut-off lines: 35/36 decisions -> 1/39 (live) and 0/30 (staging); backbone-written chosen actions 2/35 -> 9/39 (live), 8/30 (staging) | valid proposals per decision not higher (4.1 -> 4.2-4.6: the samples are mostly identical) |
| OWN RECORDS section | `test_st_strategy` (content, order, budget priority, ablation, persistence, compatibility with old entries) | staging 'after' (section 300-2,300 characters, prompts 5.0-6.0 k tokens) | after vs after_nomem (same code, one seed): different proposal mix (create_channel / get_member_events vs modify_guild 65 of 169); no note in either | none established (no reduction of repeats or re-proposals; one seed) |
| `expected` in the output format | `test_st_strategy` (format, ablation) | staging after / after_nomem | 6/30 and 10/30 chosen actions carry the generator's own prediction (0 without the field) | not measured (no learned component scores the text) |
| proposal learning (LoRA + bridge from CAM's own reward) | `test_st_strategy`: positive advantages only, gradient reaches LoRA and bridge, an SGD step raises the rewarded line's log-likelihood more than an unrelated line's; runs in the learner's slow step on a runtime lifetime (`-m slow`) | real model, 'long': in the first slow step its gradient dominated (norm 34.1) and its held-out loss fell 11.320 -> 11.139 nats/word | none on behaviour: the guard reverted the step (bridge_grounding held-out +0.0174 > tol. 0.0021) | not established |
| realised information value (+ ST-3 bootstrap input, RW a055332) | `test_st_iv` (measured once from later predictions; reaches the bootstrap component) | real model, 'long': 23 records measured | IV within ±5e-4 nats for every action type: no differential value reaches the reward | none: the world model's targets do not depend on content (5.5) |
| agent.py integration (IMP, PS-1, RW-1/2, SG-1..3; CR-1..3) | snapshot tests (section 4) | each workstream's own runtime evidence | — | — |

### 5.4 Negative results and their diagnosis

* **No note, plan or written hypothesis emerged in any condition** (0 notes executed, 0 `internal.note` proposed by
  the backbone in 165 staging decisions over six conditions and in 75 live decisions). Diagnosis — an empirical limitation of the
  current generator, not a defect of the plumbing: (1) the frozen Qwen3-4B (thinking off, temperature 0.7) proposes the
  actions its prior associates with the objective text (Discord reads, `modify_guild`, `create_channel`,
  operator requests) whatever the OWN RECORDS section shows; its 4 samples are mostly identical (3.9-4.6 distinct
  valid actions of 8 asked); (2) nothing learned can change what it writes at this experience volume: the only
  learned pathway into the generator (LoRA + bridge: `bridge_grounding` and the new `proposal_learning`) acts in slow
  steps, and **no slow step was even attempted** in the 30-decision 'after' run (`learning_updates.jsonl`: 27 fast,
  14 frequent-timescale records incl. policy updates, 0 slow; the learner triggers one every 8 applied frequent
  updates); (3) the selection cannot value a note: in
  the planning phase the objective metric is 0 at every reading, so `thompson_policy` is near-uniform over C_t (or
  follows fitted noise, ST-2), and a candidate the generator never writes cannot be chosen.
* **No evidence that OWN RECORDS reduces repetition** — in this one-seed comparison the direction is the opposite:
  with the section 17 re-proposals of an already executed action (16 %) and 6 identical repeats, without it 0 and 0
  (after_nomem, whose proposals were dominated by `modify_guild` with varying arguments, which the re-proposal measure
  does not count as identical). The section changes the *kind* of action proposed over time (first-half /
  second-half counts); with 30 decisions per condition and one seed no causal claim is made from either difference.
* **The `expected` field works as a carrier of CAM's own predictions** (every backbone-written chosen action in
  'after' carried one, e.g. "A new text channel named 'Community Announcements' will be created ..."), but these are
  restatements of the action's direct effect, not hypotheses about the objective; they are not scored against the
  outcome by anything learned (only shown next to the result in later prompts).
* Proposal learning: implementation correct and integrated; in the one real-model slow step it improved its own
  held-out objective (-0.181 nats/word) but the step was reverted by the slow-step guard because bridge_grounding's
  held-out loss rose (+0.0174, one validation chunk). Effect on behaviour: none yet. Diagnosis: interference between
  two adapter objectives judged on very little held-out evidence (the guard is doing its job; whether the conflict is
  real or noise cannot be decided from one chunk).
* **Realised information value is ~0 on the real model** (23 measurements, |IV| <= 5e-4 nats) — architectural cause in
  5.5: nothing CAM predicts in this phase depends on content; the master forwarded a request to IMP for a
  content-dependent self-supervised predictive target.

### 5.5 Real-model measurement of the information value (long staging run)

Condition **long** (scratch copy of a0f3411 + rotation + the agent-side information value + the ST-3 bootstrap
patch, i.e. the pathway RW committed in a055332; settings of after_rot + `information_value`; one runtime process
over two lease holds of <= 820 s; the staging task copy's decision limit raised to 90 and external-action limit to
120 in the scratch copy only). First lease hold 13:23-13:37 UTC: 43 decisions (the second hold had not been granted by 14:20: SG, IMP and CR
holds came first). From `analyse-run` / `reprop.py` / `learning_updates.jsonl` / the experience store (copy):

| measure | long (43 decisions) |
|---|---|
| chosen backbone / walker | 14 / 29 |
| valid backbone proposals per decision | 5.49 (192 offered, 14 distinct operations, 1 re-proposal of an executed action) |
| notes executed / proposed | 0 / 0 |
| operator requests / chosen with `expected` | 2 / 14 |
| identical repeats / experiments registered | 10 / 7 |
| results ok / queued / denied | 23 / 17 / 3 |
| learning updates | 26 fast, 18 frequent, **1 slow (adapter) step** |
| realised information value measured | 23 records; every value within ±5e-4 nats (mean 0.0000): get_audit_summary +0.0005, browser.read -0.0000 (max +0.0003), internal.inspect -0.0002, request_operator -0.0001, others 0.0000 |
| median s per decision | 20.0 |

**The first real-model slow step with proposal learning** (`learning_updates.jsonl`, trigger "8 frequent updates"):
the adapter step was computed from bridge_grounding, fast_prior, world_model, world_model_adapters and
proposal_learning (gradient norms 9.1, 2e-4, 2e-3, 9e-4 and **34.1**: the generator's own loss dominated); after the
step **proposal_learning's held-out loss fell from 11.320 to 11.139 nats per word (-0.181)** — the generator became
more likely to write CAM's above-average-reward executed actions on held-out decisions — but the guard **reverted the
step** because bridge_grounding's held-out loss rose by 0.0174 (tolerance 0.0021; one validation chunk, no noise
estimate). So the learned path into the generator works on the real model and improves its own held-out objective,
and the slow-step guard (no provider's held-out loss may worsen) blocked it: an interference between consequence
prediction and proposal learning in the shared LoRA / bridge parameters, judged on very little held-out evidence.

**Diagnosis of the near-zero information value — an architectural limitation of the prediction targets, measured.**
The world model's matured targets in this run are only `action.succeeded`, three executor bookkeeping metrics
(`observed.exec.modality.visual_heavy.{n, expected_cost, recovery_s}`) and the objective metric `membership.confirmed`
(0 at every reading) — 26 / 22 / 22 / 22 / 22 targets in the first 43 records. None of them depends on what a page
says or what a note contains, so removing an action's content from later contexts cannot change their likelihood:
the information value is ~0 for every action type *by construction of what CAM predicts*, not because the pathway
is missing. A generic remedy is a content-dependent self-supervised prediction target whose likelihood the
information can change — e.g. the latent transition model's prediction of later beliefs, or the bridge-grounding
model of observed consequence text once the bridge is trained — so that "this page made my later observations
predictable" becomes measurable. Not built (predict/ is IMP's; the bridge is a no-op until slow steps train it).

## 6. What is not done

* The long run's second lease hold was not granted before the deadline (43 of the planned ~70 decisions); repeated
  seeds of the staging comparisons were not run — every staging difference above is one sample path.
* A content-dependent self-supervised predictive target (so that information value, VOI and imagination can value
  what a document or a note contains) — the architectural cause of 5.5; forwarded to IMP by the master.
* An accepted real-model generator update: the one proposal-learning slow step was reverted by the guard.
* Turning the item-5 mechanisms on in the live run (off by default; the master decides on this evidence).
* ST-1 (window 8192) is open with the configuration owner; without it the fitting shortens OWN RECORDS.
* A learned operation-level steering of generation (the experiential policy choosing which tools the generator
  instantiates, so "which actions are considered" is learned, spec 1.4) — designed, not built.
* The two PS-1 regressions (section 4) are for PS / the master to decide.
