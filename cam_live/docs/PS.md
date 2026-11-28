# PS — persistent internal representations (amendment item 3)

Agent PS of the CAM Live core-intelligence amendment (2026-11-28). Owned: `src/cam_live/nn/**` (not `gpu.py`),
`tests/nn/**`, this file, `docs/status/PS.json`, `docs/change_requests/PS.md`. Code: `e8af02c` (mechanism),
`ff5faca` (batched refresh); the agent-side threading is change request PS-1, **applied by ST in `dde7d2d`** (all 11
hunks, verified by grep). The experiments of §4.2–4.4 ran on a patched copy made before `dde7d2d` (this repository at
`ff5faca`, `git archive`, PS-1 applied to agent.py, nothing else), PYTHONPATH pointing at it; the agent-level tests
also pass in the repository at `dde7d2d`+ (§4.1). Evidence: `docs/evidence/PS/`.

**Summary.** The design before PS had no recurrent / persistent latent state (spec 1.3) — the core's slots were
re-initialised at every call (§1). PS carries the core state across decisions: a chain of committed decisions whose
state is recomputed from a stored anchor K decisions back, so decision-time and replayed beliefs are the same function
and gradients cross K decision boundaries; old checkpoints load with the new retention parameter closed and compute
exactly what they computed before (§2). It is integrated in the runtime and causally changes beliefs and decisions
(§4.2–4.3). Whether it improves learning: on a synthetic memory task it is learned and carries information far beyond the
window (out-of-window accuracy 0.29 → 0.80); on the agent's actual experience — LabWorld, codec and SW / MECH
objectives, and CAM's own live planning run, at debug scale and at full size, paired on identical experience — it
gives **no measurable improvement** in learning to predict (on − off within ±0.02 nats) and no consistent retention
effect; the one detectable effect is tiny — at debug scale the held-out codec domain is predicted better in 21 of 28
pairs (−0.027 nats, p = 0.012). The diagnosis (§5): in this experience what the predictions need is already held by the context window or, in
longer objectives, by the per-objective fast posterior; the mechanism's benefit needs non-stationary or one-off
information needed long after it left the window, which the evaluated experience does not contain — an empirical
limitation of the evidence, not a defect found in the mechanism. Accordingly the chain is **off by default**
(`persistent_state.enabled: false`, §3): implemented, integrated, tested and one configuration line away, but not
switched on without evidence that it helps.

## 1. Does the design before PS satisfy the specification? (audit at 87eab34)

What the code did (KF-E04, MODEL_ARCHITECTURE §3 and §11.1, grep of `core_forward(` at 87eab34): every one of the 13
call sites passed `state=None` and discarded `CoreOutput.state`. Each belief was
`b = F_theta(window, slot_init)`: the learned initial slots, conditioned on "fresh" (no previous call), plus the
re-encoded window — the goal item, the tool-list item and the newest items of the *current* objective, at most
`max_context` (256 at 50m, 48 in the debug configuration), emptied at every new objective
(`agent.py:CamLiveAgent.set_objective`). The fast Bayesian posteriors (`FastBeliefState`) are applied on top of
`phi` downstream of `b`; they do not enter `b`, and they are reset to the learned prior at every objective.

| requirement | text | before PS |
|---|---|---|
| spec 1.3 (experiential backbone) | "Recurrent or persistent latent state." | **not met.** The slots and the gated write exist, but a state that is re-initialised at every call is a learned constant memory, not a persistent state. |
| spec 2.1 | "`b_t = F_θ(H_t, m_t)` … `m_t` is the current fast adaptive state" | **half met.** `b` depended on the window only. The fast posterior is an adaptive state, but not an input of `F_θ`. |
| CAM_LIVE_ARCHITECTURE §2 (binding design) | "a **gated** slot update … carrying state between calls"; "`m_t` = core slots + `FastBeliefState`" | **not met** (MODEL_ARCHITECTURE §11.1 lists it as a difference). |
| spec 1.3 / amendment item 3 | "support long-horizon understanding, prediction and decision-making" | **limited by the window.** Anything that left the window, and everything from earlier objectives, reached a belief only through the weights. |
| spec 10.4 | "Persistent learning across tasks" | met through the weights (unchanged by PS). |

So the re-encoded context plus fast beliefs does **not** satisfy the architectural requirement of a recurrent or
persistent latent state. Why every caller passed `None` (from the code and the history): the core's state API
(`CoreState`, `init_state`) was written for streaming disjoint chunks (G_nn probe "streaming is not equivalent to
one-shot processing"), while the agent re-encodes an overlapping window at every decision, and the learner re-encodes
stored records in random order. Threading a state through those paths needs (i) a definition of *which* state a
replayed record is read with, (ii) a way to train the state's write path from replayed records, and (iii) a way to
keep decision-time and replay-time beliefs the same function. None of these existed, so `None` was the only
consistent choice — it kept training and acting identical at the cost of having no persistent state.

## 2. What PS built

**The carried state** (`nn/persist.py:PersistentStateChain`). Positions `p = 0, 1, …` are the agent's committed
decisions in order over its lifetime (objectives included; the window resets at an objective, the state does not).
Per position the chain stores durably (`<run_dir>/learner/core_state.sqlite`, WAL): the decision's context-window
refs `W_p`, the keys of the context window and of the post-result window as the tokenizer sees them (refs present
in the store, newest `max_context`), and the anchor `S̄_p` — the core slots the decision was made with (float16,
8 × 768 = 12 KB at 50m) and the newest-event time. The state of position `q` is

    S_q = Φ(W_{q-1}, … Φ(W_{q-K}, S̄_{q-K}) …),     Φ(W, S) = ExperientialCore(tok(W), S).state,   K = unroll = 2

recomputed with the current weights (`state_for`). `S̄_0` is the fresh state (the current learned `slot_init`).

**One function at decision time and in replay.** `CamLiveModel.core_forward(tokens, None)` asks the attached chain
for each row's state (`resolve` → `row_positions`): a stored context window → its position `q`; a stored post window
→ `q + 1`; any other window inside a decision-time `live()` block → the next uncommitted position `n`; anything else
→ the fresh state (old records, diagnostics). So the learner's re-encoding of a stored record (`encode_decisions`,
`encode_with_adapter_grad`, `_next_belief`, the meta-learner's cached latents, retention anchors, RUDDER / COCOA
batches) reads the record with the same state the decision was made with — no change in predict/, credit/, reward/
or learning_api.py. Tested to 1e-6 (`tests/nn/test_ps_persist.py::test_decision_time_and_replayed_beliefs_are_the_same_function`).

**How gradients and credit flow.** Every loss on a replayed record back-propagates through the `K` recomputed
transitions into the parameters that write the state (write attention, gate, candidate norm), carry it (retention
`ρ`, elapsed-time conditioning) and read it (slot cross-attention in blocks 0, 2, 4; `state_read`), and into the
tokenizer and core for the `K` earlier windows (finite differences: `test_gradients_flow_through_the_recomputed_transitions`).
The anchor is a constant: the gradient is truncated `K` decisions back (truncated BPTT with stored, burned-in
anchors, R2D2-style). This is not a train/test truncation artefact — the decision used exactly the same truncated
computation — but it is a credit-horizon limit: a write whose value appears more than `K` decisions later is
credited only through the intermediate positions' own losses. The recall diagnostic (§4.1) measures what this costs:
with `K = 1` the state is trained to carry a fact for up to 7 decisions.

**Retention `ρ` and compatibility** (`nn/core.py:ExperientialCore.persist_rho`, `retention()`). A carried-in state is
read as `init + ρ ⊙ (S − init)` per slot (the elapsed-time conditioning mixed the same way), `ρ ∈ [0, 1]` with a
straight-through clamp. A fresh state is unaffected by `ρ`; `ρ = 0` reproduces the fresh computation bit for bit
(tested against the pre-PS core: `check_bitident.py`). New models start at `ρ = 1`. An agent checkpoint without the chain's bookkeeping —
written before PS, or by code that has the PS model but no chain (agent.py without PS-1, as the live run's) — is
restored with `ρ = 0` (`PersistentStateChain.load_state_dict(None)`, reported as `legacy_checkpoint`; a missing
`persist_rho` is additionally reported by the tolerant weight loader), so the restored model computes exactly what
it computed before, and learning opens `ρ` where carrying the state lowers the losses. 8 new
parameters (n_slots) in the `experiential` group: 50m 50,953,773 (was 50,953,765); `arch_signature` unchanged.

**Initial state, reset, export** (`initial()`, `reset()`). The carried state is experience-dependent state like the
fast beliefs: a weights-only export carries no chain, so a model rebuilt from it starts at the initial chain state
(every belief from the fresh state). `initial()` evaluates the live model at that state without changing anything;
`reset()` starts a new segment (the next decision from the fresh state, no recomputation across the boundary, earlier
records keep their states; boundaries are persisted with the chain).

**Other change in the core.** The elapsed time between calls now runs from the previous call's newest event to this
call's newest event (it used this call's first event, which in an overlapping re-encoded window predates the previous
call's end, so it was always clamped to 0). Only carried states see it; fresh states are bit-identical.

## 3. Integration into the runtime (agent side: change request PS-1, applied by ST in `dde7d2d`)

The agent-side threading is the exact diff in `docs/change_requests/PS.md` (PS-1, 11 hunks against agent.py at
87eab34): the chain is attached in `__init__` (`persistent_state.enabled`; file
`<run_dir>/learner/core_state.sqlite`), `_live()` blocks around `_belief`, `update_beliefs`' `belief_value`, and
`evaluate`'s `evaluate_candidates` and `planner.plan`; `commit` in `record_result` (idempotent per decision id: the
re-delivery of an approved result and the re-entry of journalled decisions after a restart (F-6) are handled); the
chain's bookkeeping in the agent checkpoint (`aux_state["agent"]["persistent_state"]`) and in the restore report;
`status()["persistent_state"]`; `close()`. ST applied all hunks in `dde7d2d` (checked by grep at the time of
writing). Nothing in predict/, credit/, reward/, runtime/ or learning_api.py changes: those modules call
`core_forward(tokens, None)` and the model resolves it. Configuration: optional section `persistent_state`
(`enabled`, `unroll` 2, `anchor_cache` 2048, `legacy_rho` 0.0); the `nn` section and `arch_signature` are unchanged.

**Default: off** (`enabled: false`, `551ff3b`). PS-1 first shipped with the chain on by default; ST then found two
integration properties that the carried state changes (restoring the initial weights restores the initial decision
distribution; a weights-only export read in a fresh process gives the agent's distribution). The master's decision:
the chain state is experience-dependent state, like the fast beliefs, which a weights-only export (spec 15.3)
deliberately excludes, so both properties hold *at equal chain state*: `PersistentStateChain.initial()` evaluates at
the initial chain state (what an export computes) and `reset()` starts a new segment; the two tests compare at the
initial chain state (edited with the master's permission), and a PS test checks both properties with the chain on
(`test_weights_carry_the_learned_behaviour_at_equal_chain_state_and_export_starts_initial`). The default stays on
only if the evidence shows the persistent state helps; §4.4 shows it does not measurably help on the evaluated
experience, so the default is off. Turning it on is one configuration line; a run that turns it on later starts from
the initial chain state (a checkpoint without chain bookkeeping takes the legacy path, `ρ = 0`).

## 4. Evidence (amendment item 8: correctness, integration, causal influence, measured improvement)

### 4.1 Implementation correctness (level a)

Tests (`tests/nn/test_ps_persist.py`, 8 tests, debug model on CPU):

* compatibility: no chain, a disabled chain, nothing committed, and `ρ = 0` give bit-identical `b` and `h` to the
  plain core; detaching the chain restores the old path (`test_without_chain_or_commits_or_with_rho_zero_…`). Against
  the pre-PS core (`core.py` of 87eab34 loaded side by side, `docs/evidence/PS/scripts/check_bitident.py`): fresh
  `b`, `h`, slots bit-identical; carried state with `ρ = 0` bit-identical to fresh; with `ρ = 1` the newest-event
  elapsed time changes `b` by at most 6.7e-3 (debug model) relative to the old first-event definition.
* decision time = replay: 8 decisions, each stored window re-encoded alone and in a batch equals the decision-time
  belief to 1e-6; post windows are read with the next position's state (`test_decision_time_and_replayed_…`).
* information beyond the window: two streams that differ only in a reading 6 decisions and one objective boundary
  back (no shared item in the 4-item window) give different beliefs; without the chain identical ones.
* gradients: central finite differences (ε = 1e-2) of a replayed belief projection match autograd for the slot-write
  gate bias, the retention `ρ` (interior value) and a write-attention weight (within 2 %); a window K = 2 back changes
  the replayed belief, a window further back changes it only through its stored anchor (bit-identical when edited).
* durability: the chain reopened from its database restores the live state (1e-6) and every replayed belief;
  commits are idempotent; a checkpoint without the retention parameter is reported missing by the tolerant loader
  and the legacy restore closes `ρ`, after which every decision-time belief equals the old model's bit for bit.

**Can the carried state be learned the way the continual learner trains it?** (`scripts/ps_recall.py`, debug model,
CPU; results `docs/evidence/PS/recall/recall_s{0,1,2}.json`.) A stream of decisions; every 8th observes a fact
("the code is <k>", 4 classes), the others distractors; the window holds the newest 2 items; the target at every
decision is the latest fact. Online and prequential: predict at decision time, commit, then one AdamW step on a
replay batch of 16 stored decisions (the newest + uniform earlier ones) re-encoded through `core_forward(tokens,
None)` — the same resolution and the same K-step recomputation the agent's learner gets. 300 decisions, scored on the
second half; 3 seeds; chance 0.25.

| condition | accuracy, fact in window (lag 0-1) | accuracy, fact out of window (lag 2-7) | NLL out of window | accuracy at lag >= 3 (beyond K) | per seed (out of window) |
|---|---:|---:|---:|---:|---|
| no carried state (`state=None`, before PS) | 0.956 | 0.292 | 1.694 | 0.308 | 0.277 / 0.259 / 0.339 |
| chain, K = 1 | 0.781 | 0.741 | 0.926 | 0.756 | 0.982 / 0.491 / 0.750 |
| chain, K = 2 | 0.860 | 0.795 | 0.589 | 0.806 | 1.000 / 0.446 / 0.938 |

The carried state learns to hold the fact for 7 decisions after it left the window, also at lags beyond the
gradient horizon K (the stored anchors carry what the K-step-trained transitions preserve), in 2 of 3 seeds almost
perfectly and in seed 1 partly (0.45-0.49) within 300 steps. Cost seen here: while it is being learned the carried
state interferes with the in-window readout (seed 1: 0.74 vs 0.95; seed 2 at K = 1: 0.63 vs 1.00) — the state is an
extra input the read-out has to learn to use. The retention stayed at `ρ = 1` (the clamp's upper bound) in every
run. This is an engineering diagnostic (debug model, synthetic stream), not evidence of competence.

### 4.2 Integration into the actual runtime (level b)

* **In the repository** (agent.py at `dde7d2d`, PS-1 applied by ST): `tests/nn/test_ps_agent_integration.py` runs a
  LabWorld objective through `eval.runner.run_lifetime` → `runtime.loop.LiveRuntime`: one chain position per stored
  record, decision-time rows resolved live, the learner's re-encodings resolved to stored positions, a stored
  decision re-encoded = its position's state to 1e-6; checkpoint → rebuilt agent → `load` → consistent restore and
  the same live state; a checkpoint written with the chain off restores with `legacy_checkpoint`, `ρ = 0`, and every
  stored and decision-time belief bit-identical to the old agent's; a third test checks, with the chain on, that the
  learned behaviour lives in the weights at equal (initial) chain state and that a weights-only export rebuilt by
  `load_exported_model` matches the agent at the initial chain state. Final run on a clean snapshot of `551ff3b`
  (default off): `tests/integration` + `tests/nn` 205 passed, 16 skipped (`-m gpu` / `-m slow`), 0 failed, 18 min
  (JUnit `docs/evidence/test_results/PS_fix.xml`); the same snapshot with the default flipped on:
  `test_agent_runtime.py`, `test_int2_checkpoints.py`, `test_agent_recovery.py` 15 passed
  (`docs/evidence/PS/snapshot_551ff3b_chain_on_tests.xml`).
  `tests/integration/test_agent_recovery.py` (kill -9 mid-run and restart without duplicate actions; save → fresh
  process → load gives identical decisions to 1e-6) passes with the chain on: both tests on the patched copy, the
  kill -9 test also in the repository (ST reports the whole file passing on its integration snapshot). (An earlier failure of the kill -9 test on the patched copy was my artefact: that copy had
  `configs/authorization.yaml` of a later commit, whose `standing` key the older policy code rejects, so every action
  was denied; with the matching configuration both tests pass.)
* **Debug ablation runs through the runtime** (§4.4): over the 5 runs with the chain, 189 decisions committed, every
  learner re-encoding resolved to a stored position (`rows_fresh` = 0; 34,770 context and 8,785 post windows); e.g. `on_s0`: 40 commits, 314 decision-time rows,
  7,606 context and 1,879 post windows re-encoded by the learner with their carried state, 17,898 refresh windows.
* **Substantive protocol on CPU (plumbing only, tiny backbone)**: `scripts/ps_oe_paired.py --cpu` (`eval.oe_runner`'s
  `substantive_config` / `run_objective`, generation on, SW then MECH, the agent rebuilt and restored from its
  checkpoint between objectives): restore `consistent`, positions 2 → 4, `ρ` moved by learning (min 0.99996).
* **Real configuration**: §4.4 (full size) and §4.5 (live-run checkpoint).

### 4.3 Causal influence on behaviour (level c)

Within-decision counterfactual (`scripts/ps_eval.py` `CausalProbe`, `scripts/ps_oe.py` `measure`): at every
`select`, the trainable policy's distribution over the offered candidates and the belief `b` are computed twice
with the same weights and context — with the carried state and with the chain switched off. Debug runs with the
chain on (`ρ` from 1, seeds 0–3): mean total variation 0.0094 per decision (max 0.019 per lifetime, max 0.036 per
decision), relative change of `b` 0.265 (20 lifetimes, 149 decisions with ≥ 2 candidates). With `ρ` started at 0 (`on_rho0_s0`, the legacy-restore
setting) the effect is 1e-5–5e-4 TV and 0.1–0.2 % of `b`: the retention opened only to 0.002–0.004 in 40 decisions.
So the carried state changes what the agent believes by a quarter of the belief's norm and changes the policy's
distribution measurably but little at debug scale. Full size: §4.4 (paired substantive objectives).

**Methodological finding.** On-line on / off runs diverge chaotically: `on_rho0_s0`, whose beliefs differ from the
fresh ones by 0.1 %, reached verified progress 0.41 / 0.20 / 0.35 / 0.06 on the four LabWorld objectives where
`off_s0` reached 0.12 / 0.06 / 0.06 / 0.00 — a difference that cannot be caused by the carried state. Outcome
differences between on-line arms at this sample size therefore measure trajectory noise (Thompson sampling,
experiments and learning amplify any numerical difference); the within-decision counterfactual and the fixed-stream
replay below are the valid measurements.

### 4.4 Measured improvement in learning or competence (level d)

**Debug scale, on-line through the runtime** (`scripts/ps_eval.py`, LabWorld F1 ×2 → T1 (interfering mechanism) →
F1 (retention) → codec T2 (held out), 8 decisions each, same seeds; `docs/evidence/PS/ablation_debug/`):

| run | lifetime | family | progress | NLL count (n) | NLL binary (n) | NLL goal-relevant (n) | causal TV | rel. change of b | rho |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| off_s0 | 0 | F1 | 0.119 | 2.317 (15) | 0.559 (16) | 1.710 (15) | - | - | - |
| off_s0 | 1 | F1 | 0.064 | 1.380 (20) | 0.529 (17) | 1.048 (17) | - | - | - |
| off_s0 | 2 | T1 | 0.062 | 1.488 (21) | 0.562 (17) | 1.086 (17) | - | - | - |
| off_s0 | 3 | F1 | 0.000 | 0.653 (19) | 0.519 (17) | 0.803 (18) | - | - | - |
| off_s0 | 4 | T2 | 1.000 | - | 0.594 (8) | 0.693 (4) | - | - | - |
| off_s1 | 0 | F1 | 0.186 | 1.409 (19) | 0.549 (17) | 1.171 (17) | - | - | - |
| off_s1 | 1 | F1 | 0.100 | 2.452 (18) | 0.553 (16) | 1.105 (15) | - | - | - |
| off_s1 | 2 | T1 | 0.292 | 2.876 (18) | 0.528 (17) | 1.470 (16) | - | - | - |
| off_s1 | 3 | F1 | 0.000 | 0.470 (16) | 0.567 (16) | 0.595 (16) | - | - | - |
| off_s1 | 4 | T2 | 1.000 | - | 0.710 (2) | 0.695 (1) | - | - | - |
| off_s2 | 0 | F1 | 0.068 | 1.670 (25) | 0.595 (19) | 1.229 (19) | - | - | - |
| off_s2 | 1 | F1 | 0.236 | 2.674 (25) | 0.542 (18) | 1.717 (19) | - | - | - |
| off_s2 | 2 | T1 | 0.338 | 3.154 (21) | 0.509 (17) | 1.647 (16) | - | - | - |
| off_s2 | 3 | F1 | 0.278 | 1.230 (18) | 0.423 (16) | 0.832 (17) | - | - | - |
| off_s2 | 4 | T2 | 0.958 | - | 0.301 (8) | -2.555 (5) | - | - | - |
| off_s3 | 0 | F1 | 0.034 | 1.115 (21) | 0.497 (18) | 0.964 (18) | - | - | - |
| off_s3 | 1 | F1 | 0.118 | 1.644 (20) | 0.526 (16) | 1.100 (17) | - | - | - |
| off_s3 | 2 | T1 | 0.062 | 4.255 (19) | 0.529 (16) | 3.903 (17) | - | - | - |
| off_s3 | 3 | F1 | 0.000 | 0.457 (17) | 0.525 (16) | 0.594 (17) | - | - | - |
| off_s3 | 4 | T2 | 0.958 | - | 0.359 (9) | -3.859 (4) | - | - | - |
| on_rho0_s0 | 0 | F1 | 0.407 | 1.966 (16) | 0.560 (16) | 2.018 (16) | 0.0000 | 0.001 | 0.0021 |
| on_rho0_s0 | 1 | F1 | 0.200 | 2.842 (18) | 0.565 (15) | 1.398 (15) | 0.0001 | 0.002 | 0.0044 |
| on_rho0_s0 | 2 | T1 | 0.354 | 3.875 (19) | 0.554 (15) | 1.362 (16) | 0.0001 | 0.002 | 0.0044 |
| on_rho0_s0 | 3 | F1 | 0.056 | 0.992 (18) | 0.519 (17) | 0.846 (17) | 0.0005 | 0.002 | 0.0038 |
| on_rho0_s0 | 4 | T2 | 0.958 | - | 0.382 (8) | -4.987 (5) | 0.0000 | 0.001 | 0.0031 |
| on_s0 | 0 | F1 | 0.424 | 3.184 (19) | 0.543 (17) | 1.919 (17) | 0.0071 | 0.200 | 0.9972 |
| on_s0 | 1 | F1 | 0.173 | 2.868 (22) | 0.554 (16) | 1.780 (18) | 0.0135 | 0.272 | 0.9931 |
| on_s0 | 2 | T1 | 0.323 | 3.124 (22) | 0.561 (17) | 1.494 (17) | 0.0077 | 0.264 | 0.9935 |
| on_s0 | 3 | F1 | 0.333 | 1.904 (20) | 0.530 (17) | 1.274 (18) | 0.0074 | 0.280 | 0.9947 |
| on_s0 | 4 | T2 | 0.958 | - | 0.405 (9) | -7.271 (4) | 0.0029 | 0.317 | 0.9956 |
| on_s1 | 0 | F1 | 0.153 | 1.939 (20) | 0.557 (17) | 1.063 (17) | 0.0131 | 0.201 | 0.9963 |
| on_s1 | 1 | F1 | 0.255 | 2.848 (21) | 0.541 (16) | 1.516 (17) | 0.0091 | 0.175 | 0.9913 |
| on_s1 | 2 | T1 | 0.262 | 2.563 (23) | 0.532 (15) | 1.425 (16) | 0.0081 | 0.218 | 0.9924 |
| on_s1 | 3 | F1 | 0.167 | 1.149 (19) | 0.526 (17) | 0.794 (18) | 0.0082 | 0.230 | 0.9926 |
| on_s1 | 4 | T2 | 0.981 | - | 0.397 (8) | -3.814 (4) | 0.0048 | 0.293 | 0.9943 |
| on_s2 | 0 | F1 | 0.136 | 1.487 (24) | 0.599 (19) | 1.153 (19) | 0.0046 | 0.192 | 0.9958 |
| on_s2 | 1 | F1 | 0.245 | 3.035 (28) | 0.511 (18) | 2.066 (19) | 0.0157 | 0.265 | 0.9908 |
| on_s2 | 2 | T1 | 0.246 | 2.258 (23) | 0.546 (17) | 1.347 (17) | 0.0148 | 0.258 | 0.9915 |
| on_s2 | 3 | F1 | 0.111 | 1.016 (19) | 0.542 (17) | 0.868 (17) | 0.0083 | 0.242 | 0.9931 |
| on_s2 | 4 | T2 | 1.000 | - | 0.611 (10) | 1.642 (7) | 0.0097 | 0.451 | 0.9946 |
| on_s3 | 0 | F1 | 0.237 | 2.285 (27) | 0.503 (18) | 1.186 (18) | 0.0054 | 0.194 | 0.9969 |
| on_s3 | 1 | F1 | 0.100 | 2.088 (18) | 0.508 (16) | 1.123 (16) | 0.0175 | 0.270 | 0.9932 |
| on_s3 | 2 | T1 | 0.292 | 3.426 (21) | 0.518 (16) | 2.137 (17) | 0.0037 | 0.180 | 0.9929 |
| on_s3 | 3 | F1 | 0.000 | 0.491 (16) | 0.542 (15) | 0.590 (16) | 0.0069 | 0.218 | 0.9942 |
| on_s3 | 4 | T2 | 1.000 | - | 0.669 (2) | 0.690 (1) | 0.0192 | 0.574 | 0.9969 |

| run | world-model NLL of the first phase's records after phase 0 / 1 / 2 / 3 | change 0 -> 3 |
|---|---|---:|
| off_s0 | 1.4309 / 1.4525 / 1.4714 / 1.5053 | +0.0744 (+5.2 %) |
| off_s1 | 1.7992 / 1.8093 / 1.6932 / 1.7167 | -0.0825 (-4.6 %) |
| off_s2 | 2.4190 / 2.3531 / 2.2495 / 2.1695 | -0.2495 (-10.3 %) |
| off_s3 | 1.3576 / 1.3366 / 1.3502 / 1.4168 | +0.0593 (+4.4 %) |
| on_rho0_s0 | 2.6638 / 2.5229 / 2.5668 / 2.5610 | -0.1028 (-3.9 %) |
| on_s0 | 2.2796 / 2.2533 / 2.2842 / 2.3210 | +0.0414 (+1.8 %) |
| on_s1 | 3.1417 / 3.1505 / 3.1756 / 3.2216 | +0.0799 (+2.5 %) |
| on_s2 | 3.3557 / 3.1643 / 3.2106 / 3.3496 | -0.0061 (-0.2 %) |
| on_s3 | 2.4687 / 2.2903 / 2.2928 / 2.4254 | -0.0433 (-1.8 %) |

seed 0, LabWorld (F1/T1): progress on - off +0.252 (4); goal-relevant NLL on - off +0.454 (4); count NLL on - off +1.311 (4)
seed 0, held-out T2: progress on - off -0.042 (1); goal-relevant NLL on - off -7.964 (1); count NLL on - off -
seed 1, LabWorld (F1/T1): progress on - off +0.064 (4); goal-relevant NLL on - off +0.114 (4); count NLL on - off +0.323 (4)
seed 1, held-out T2: progress on - off -0.019 (1); goal-relevant NLL on - off -4.509 (1); count NLL on - off -
seed 2, LabWorld (F1/T1): progress on - off -0.046 (4); goal-relevant NLL on - off +0.002 (4); count NLL on - off -0.233 (4)
seed 2, held-out T2: progress on - off +0.042 (1); goal-relevant NLL on - off +4.197 (1); count NLL on - off -
seed 3, LabWorld (F1/T1): progress on - off +0.104 (4); goal-relevant NLL on - off -0.381 (4); count NLL on - off +0.205 (4)
seed 3, held-out T2: progress on - off +0.042 (1); goal-relevant NLL on - off +4.549 (1); count NLL on - off -

Over the 16 LabWorld lifetimes of seeds 0–3: verified progress on − off +0.094 (SD 0.144; on higher in 10 of 16),
goal-relevant prospective NLL on − off +0.047 (SD 0.53; on lower in 6 of 16). As explained in §4.3 these on-line
differences are dominated by trajectory divergence (the near-zero-effect run `on_rho0_s0` differs from `off_s0` by
+0.29 progress in its first objective); they are reported, not interpreted.

**Debug scale, paired on a fixed experience stream** (`scripts/ps_replay.py`: prequential world-model NLL of every
stored decision before training on it, then 2 AdamW steps of the agent's own world-model loss provider on a replay
batch of 8; the same decisions, observations and outcomes for both arms; the 8 stores of the on-line runs
`on_s0..3`, `off_s0..3` as streams, model seeds 0–2 (first four) and 3–4 (all eight): 28 pairs;
`docs/evidence/PS/replay_debug/`):

| stream | model seed | scored | NLL on | NLL off | on - off | per block on/off | retention on | retention off | rho (on, end) |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| off_s0 | 0 | 36 | 4.933 | 4.942 | -0.009 | lab/first 5.688/5.693; lab/door 1.349/1.347; lab/first 0.902/0.888; codec 17.144/17.236 | 1.381 -> 1.322 | 1.383 -> 1.325 | rho 0.9829 |
| off_s0 | 1 | 36 | 5.135 | 5.133 | +0.003 | lab/first 5.961/5.959; lab/door 1.369/1.370; lab/first 0.955/0.958; codec 17.729/17.703 | 1.414 -> 1.398 | 1.417 -> 1.401 | rho 0.983 |
| off_s0 | 2 | 36 | 5.024 | 5.031 | -0.008 | lab/first 5.835/5.828; lab/door 1.369/1.371; lab/first 0.942/0.944; codec 17.252/17.341 | 1.432 -> 1.349 | 1.438 -> 1.349 | rho 0.9838 |
| off_s0 | 3 | 36 | 5.235 | 5.257 | -0.023 | lab/first 6.190/6.237; lab/door 1.385/1.371; lab/first 0.989/0.979; codec 17.608/17.667 | 1.465 -> 1.374 | 1.454 -> 1.370 | rho 0.986 |
| off_s0 | 4 | 36 | 5.049 | 5.057 | -0.008 | lab/first 5.879/5.884; lab/door 1.399/1.401; lab/first 0.869/0.874; codec 17.393/17.427 | 1.501 -> 1.435 | 1.495 -> 1.427 | rho 0.9857 |
| off_s1 | 0 | 33 | 4.250 | 4.264 | -0.013 | lab/first 5.751/5.758; lab/door 3.912/3.943; lab/first 1.406/1.412; codec 5.702/5.731 | 1.737 -> 1.562 | 1.741 -> 1.570 | rho 0.9862 |
| off_s1 | 1 | 33 | 4.282 | 4.281 | +0.001 | lab/first 5.848/5.855; lab/door 3.758/3.754; lab/first 1.516/1.500; codec 5.542/5.568 | 1.720 -> 1.566 | 1.725 -> 1.570 | rho 0.9847 |
| off_s1 | 2 | 33 | 4.306 | 4.293 | +0.013 | lab/first 5.867/5.859; lab/door 3.918/3.880; lab/first 1.468/1.465; codec 5.128/5.151 | 1.763 -> 1.539 | 1.758 -> 1.542 | rho 0.9881 |
| off_s1 | 3 | 33 | 4.423 | 4.449 | -0.026 | lab/first 6.061/6.098; lab/door 3.971/3.975; lab/first 1.420/1.432; codec 5.849/6.009 | 1.792 -> 1.573 | 1.797 -> 1.571 | rho 0.9888 |
| off_s1 | 4 | 33 | 4.434 | 4.402 | +0.032 | lab/first 5.964/5.955; lab/door 4.183/4.050; lab/first 1.467/1.492; codec 5.702/5.649 | 1.850 -> 1.637 | 1.830 -> 1.617 | rho 0.9887 |
| off_s2 | 3 | 40 | 5.163 | 5.165 | -0.002 | lab/first 6.532/6.559; lab/door 3.999/3.959; lab/first 1.440/1.444; codec 7.310/7.303 | 2.273 -> 1.870 | 2.263 -> 1.876 | rho 0.9906 |
| off_s2 | 4 | 40 | 5.244 | 5.232 | +0.012 | lab/first 6.447/6.432; lab/door 4.429/4.400; lab/first 1.380/1.392; codec 7.518/7.504 | 2.339 -> 1.960 | 2.337 -> 1.952 | rho 0.986 |
| off_s3 | 3 | 40 | 3.758 | 3.766 | -0.008 | lab/first 5.855/5.872; lab/door 1.697/1.696; lab/first 0.901/0.897; codec 4.482/4.492 | 1.332 -> 1.232 | 1.322 -> 1.232 | rho 0.9908 |
| off_s3 | 4 | 40 | 3.642 | 3.642 | +0.000 | lab/first 5.575/5.574; lab/door 1.641/1.637; lab/first 0.869/0.870; codec 4.548/4.552 | 1.259 -> 1.195 | 1.262 -> 1.196 | rho 0.9849 |
| on_s0 | 0 | 40 | 4.845 | 4.877 | -0.032 | lab/first 6.609/6.616; lab/door 4.683/4.816; lab/first 1.907/1.911; codec 4.417/4.425 | 2.252 -> 1.936 | 2.268 -> 1.949 | rho 0.9822 |
| on_s0 | 1 | 40 | 5.007 | 5.001 | +0.006 | lab/first 6.901/6.900; lab/door 4.672/4.656; lab/first 1.897/1.892; codec 4.664/4.655 | 2.334 -> 1.867 | 2.332 -> 1.864 | rho 0.9806 |
| on_s0 | 2 | 40 | 4.931 | 4.924 | +0.007 | lab/first 6.751/6.734; lab/door 4.687/4.676; lab/first 2.005/2.000; codec 4.461/4.479 | 2.264 -> 1.910 | 2.263 -> 1.919 | rho 0.9827 |
| on_s0 | 3 | 40 | 5.177 | 5.196 | -0.020 | lab/first 7.175/7.225; lab/door 5.071/5.054; lab/first 1.966/1.963; codec 4.495/4.513 | 2.319 -> 1.858 | 2.306 -> 1.850 | rho 0.9829 |
| on_s0 | 4 | 40 | 5.232 | 5.202 | +0.030 | lab/first 6.923/6.910; lab/door 5.881/5.741; lab/first 1.938/1.926; codec 4.497/4.523 | 2.451 -> 1.980 | 2.428 -> 1.981 | rho 0.9825 |
| on_s1 | 0 | 39 | 5.597 | 5.603 | -0.007 | lab/first 7.132/7.127; lab/door 3.669/3.696; lab/first 1.244/1.244; codec 8.566/8.583 | 2.964 -> 2.299 | 2.959 -> 2.307 | rho 0.9843 |
| on_s1 | 1 | 39 | 5.766 | 5.760 | +0.006 | lab/first 7.432/7.443; lab/door 3.710/3.708; lab/first 1.301/1.287; codec 8.698/8.663 | 3.028 -> 2.192 | 3.029 -> 2.187 | rho 0.9848 |
| on_s1 | 2 | 39 | 5.605 | 5.615 | -0.009 | lab/first 7.376/7.383; lab/door 3.604/3.613; lab/first 1.289/1.294; codec 8.132/8.151 | 3.073 -> 2.308 | 3.091 -> 2.311 | rho 0.9853 |
| on_s1 | 3 | 39 | 5.774 | 5.821 | -0.047 | lab/first 7.494/7.565; lab/door 3.755/3.802; lab/first 1.366/1.352; codec 8.511/8.569 | 3.014 -> 2.220 | 3.056 -> 2.232 | rho 0.9891 |
| on_s1 | 4 | 39 | 5.770 | 5.759 | +0.011 | lab/first 7.484/7.457; lab/door 3.812/3.797; lab/first 1.277/1.281; codec 8.551/8.557 | 3.183 -> 2.192 | 3.157 -> 2.188 | rho 0.9844 |
| on_s2 | 3 | 37 | 6.428 | 6.461 | -0.033 | lab/first 7.591/7.633; lab/door 2.488/2.486; lab/first 1.357/1.359; codec 17.125/17.236 | 3.064 -> 2.280 | 3.059 -> 2.270 | rho 0.987 |
| on_s2 | 4 | 37 | 6.341 | 6.353 | -0.012 | lab/first 7.517/7.511; lab/door 2.655/2.640; lab/first 1.365/1.393; codec 16.436/16.526 | 3.312 -> 2.641 | 3.307 -> 2.609 | rho 0.9861 |
| on_s3 | 3 | 33 | 4.600 | 4.625 | -0.025 | lab/first 6.614/6.655; lab/door 3.548/3.565; lab/first 1.441/1.441; codec 6.064/6.108 | 2.354 -> 1.875 | 2.358 -> 1.872 | rho 0.9895 |
| on_s3 | 4 | 33 | 4.591 | 4.580 | +0.011 | lab/first 6.530/6.517; lab/door 3.779/3.769; lab/first 1.385/1.382; codec 5.717/5.667 | 2.465 -> 2.152 | 2.441 -> 2.144 | rho 0.9863 |

prequential NLL on - off: mean -0.005 (SE 0.003) over 28 pairs; on better in 16 of 28
  block 0:lab/first: mean -0.009 over 28 (on better in 15)
  block 1:lab/door: mean +0.008 over 28 (on better in 10)
  block 2:lab/first: mean -0.000 over 28 (on better in 15)
  block 3:codec: mean -0.027 over 28 (on better in 21)
forgetting of the first block (on minus off, NLL change first->last block): mean -0.001 over 28 (on forgets less in 12)

Retention columns: NLL of the first block's records after the first -> last block; rho: mean retention at the end.

| block (family) | pairs | on − off (nats) | SE | on better | sign test p (two-sided) |
|---|---:|---:|---:|---:|---:|
| 0 LabWorld F1 (first exposure) | 28 | −0.009 | 0.004 | 15 | 0.85 |
| 1 LabWorld T1 (interfering mechanism) | 28 | +0.008 | 0.009 | 10 | 0.18 |
| 2 LabWorld F1 again (retention) | 28 | −0.000 | 0.002 | 15 | 0.85 |
| 3 codec T2 (held-out domain) | 28 | −0.027 | 0.009 | 21 | 0.012 |

**No measurable effect on learning to predict overall** (on − off −0.005, SE 0.003, on better in 16 of 28), **none
on retention** of the first block after the interfering one (−0.001; on forgets less in 12 of 28), and one small,
nominally significant effect: on the held-out codec domain the chain predicts better in 21 of 28 pairs (−0.027 nats,
p = 0.012; 0.05 after a Bonferroni correction over the four blocks) — about 0.2–0.5 % of that block's NLL (4–17
nats). The interfering block points the other way (not significant). The learner closed the retention slightly in
every run (`ρ` 1 → 0.98).

**Debug scale, long objectives as fixed streams** (`docs/evidence/PS/replay_long/`): two streams with a 40- and a
16-decision LabWorld objective (F1) followed by a 20- / 11-decision interfering one (T1) — longer than the 48-item
window holds — replayed with and without the chain, 3 model seeds each:

| stream | model seed | scored | NLL on | NLL off | on - off | per block on/off | retention on | retention off | rho (on, end) |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| off_s5 | 0 | 60 | 3.634 | 3.648 | -0.014 | lab/first 3.997/4.002; lab/door 2.910/2.939 | 1.898 -> 1.579 | 1.922 -> 1.601 | rho 0.9771 |
| off_s5 | 1 | 60 | 3.616 | 3.626 | -0.010 | lab/first 4.021/4.034; lab/door 2.807/2.810 | 1.844 -> 1.495 | 1.841 -> 1.515 | rho 0.9785 |
| off_s5 | 2 | 60 | 3.538 | 3.534 | +0.004 | lab/first 4.041/4.054; lab/door 2.530/2.494 | 1.831 -> 1.572 | 1.836 -> 1.577 | rho 0.9852 |
| off_s6 | 0 | 27 | 5.922 | 5.956 | -0.034 | lab/first 7.377/7.398; lab/door 3.805/3.857 | 3.024 -> 2.808 | 3.053 -> 2.810 | rho 0.9868 |
| off_s6 | 1 | 27 | 5.894 | 5.897 | -0.002 | lab/first 7.528/7.527; lab/door 3.518/3.525 | 2.876 -> 2.604 | 2.877 -> 2.586 | rho 0.9852 |
| off_s6 | 2 | 27 | 5.876 | 5.869 | +0.008 | lab/first 7.405/7.411; lab/door 3.653/3.625 | 2.956 -> 2.753 | 2.955 -> 2.759 | rho 0.9868 |

prequential NLL on - off: mean -0.008 (SE 0.006) over 6 pairs; on better in 4 of 6
  block 0:lab/first: mean -0.010 over 6 (on better in 5)
  block 1:lab/door: mean -0.005 over 6 (on better in 4)
forgetting of the first block (on minus off, NLL change first->last block): mean +0.003 over 6 (on forgets less in 3)

Again no significant effect (on − off −0.008, SE 0.006, on better in 4 of 6; first, long block −0.010, 5 of 6).

**Synthetic memory (§4.1)**: where information outside the window matters, the carried state is learned and helps a
lot (out-of-window accuracy 0.29 → 0.80 at K = 2); started closed (`ρ = 0`, `docs/evidence/PS/recall_rho0/`) the
learner opens it (mean `ρ` 0 → 0.33–0.44 in 300 steps) and reaches 0.60 (0.47 / 0.83 / 0.48).

**Full size, paired on fixed real experience** (Qwen3-4B + 50m core on the GPU; holds of 329 s and ~10 min;
`docs/evidence/PS/full_size/replay_real*.json`): the same prequential replay on three stores written by the
full-size agent — two substantive-evaluation runs (`x0_E_s0`: SW:rle → SW:rpn → MECH; `x1_E_s0`: four SW
objectives) and CAM's own live planning run (36 real decisions of the Discord 100-member objective, before it was
stopped) — model seeds 0–2:

| stream | model seed | scored | NLL on | NLL off | on - off | per block on/off | retention on | retention off | rho (on, end) |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| x0_E_s0 | 0 | 41 | 2.365 | 2.384 | -0.019 | openended/sw/rle 3.521/3.559; openended/sw/rpn 1.912/1.964; openended/mech 1.553/1.549 | 1.074 -> 1.209 | 1.090 -> 1.471 | rho 0.9981 |
| x0_E_s0 | 1 | 41 | 2.280 | 2.300 | -0.020 | openended/sw/rle 3.576/3.601; openended/sw/rpn 2.132/2.189; openended/mech 1.281/1.288 | 1.151 -> 1.308 | 1.187 -> 1.372 | rho 0.9885 |
| x0_E_s0 | 2 | 41 | 2.196 | 2.191 | +0.005 | openended/sw/rle 3.517/3.512; openended/sw/rpn 1.846/1.825; openended/mech 1.226/1.226 | 1.115 -> 1.534 | 1.113 -> 1.469 | rho 0.9876 |
| x1_E_s0 | 0 | 43 | 2.662 | 2.653 | +0.009 | openended/sw/rle 3.502/3.517; openended/sw/rpn 2.014/2.048; openended/sw/merge 1.807/1.595; openended/sw/topk 1.379/1.399 | 1.215 -> 1.019 | 1.198 -> 0.949 | rho 1.0 |
| x1_E_s0 | 1 | 43 | 2.696 | 2.674 | +0.022 | openended/sw/rle 3.661/3.645; openended/sw/rpn 2.103/2.055; openended/sw/merge 1.250/1.271; openended/sw/topk 1.539/1.580 | 1.298 -> 1.518 | 1.282 -> 1.557 | rho 1.0 |
| x1_E_s0 | 2 | 43 | 2.702 | 2.733 | -0.031 | openended/sw/rle 3.570/3.581; openended/sw/rpn 2.016/2.037; openended/sw/merge 1.822/1.946; openended/sw/topk 1.577/1.627 | 1.326 -> 1.250 | 1.334 -> 1.124 | rho 1.0 |
| live | 0 | 36 | 0.712 | 0.716 | -0.004 | community/membership 0.712/0.716 | 0.307 -> 0.307 | 0.253 -> 0.253 | rho 0.9821 |
| live | 1 | 36 | 0.710 | 0.693 | +0.017 | community/membership 0.710/0.693 | 0.354 -> 0.354 | 0.277 -> 0.277 | rho 0.9908 |
| live | 2 | 36 | 0.656 | 0.684 | -0.028 | community/membership 0.656/0.684 | 0.339 -> 0.339 | 0.233 -> 0.233 | rho 0.996 |

prequential NLL on - off: mean -0.006 (SE 0.006) over 9 pairs; on better in 5 of 9
  block 0:openended/sw/rle: mean -0.011 over 6 (on better in 4)
  block 1:openended/sw/rpn: mean -0.016 over 6 (on better in 4)
  block 2:openended/mech: mean -0.001 over 3 (on better in 1)
  block 2:openended/sw/merge: mean +0.022 over 3 (on better in 2)
  block 3:openended/sw/topk: mean -0.037 over 3 (on better in 3)
  block 0:community/membership: mean -0.005 over 3 (on better in 2)
forgetting of the first block (on minus off, NLL change first->last block): mean -0.013 over 6 (on forgets less in 3)

**No measurable effect on learning to predict at full size either**: on − off −0.006 (SE 0.006) over 9 pairs, on
better in 5. Retention is mixed (forgetting of the first block −0.013 on − off, 3 of 6); in x0, after the unfamiliar
MECH block, the first block's NLL rose less with the chain in seeds 0 and 1 (+0.135 vs +0.381, +0.157 vs +0.185) and more in seed 2 (+0.419 vs +0.356). On the live run's stream the chain's
model ends with a worse fit of the stream it trained on (0.31–0.35 vs 0.23–0.28 nats). Cost at full size: an on run
took 70–74 s against 23–28 s off (the K-step refresh in every replayed row and the anchor recomputation at every
commit).

**Full size, on-line, both arms in one hold** (`scripts/ps_oe_paired.py`: the substantive protocol of
`eval.oe_runner` — condition E, open-ended generation on, sandboxed SW objective — one shared backbone, each arm's
own checkpoint; `docs/evidence/PS/full_size/paired.json`, `paired_causal_on.jsonl`): SW:rle was achieved by both
arms in 2 decisions (verified value 1.0; on 39 s, off 37 s; prospective NLL 1.891 vs 1.896). The within-decision
counterfactual at the one decision with carried state (position 1): TV 0.0023 between the policy distributions with
and without it, **the policy's most probable candidate changed**, belief changed by 14.9 % relative, the evaluator's
success probability by 7e-5. MECH:7 (the next hold) was still queued for the shared GPU lock at the time of this version.

### 4.5 Compatibility with the live run's checkpoints

On the real configuration (GPU, `docs/evidence/PS/full_size/compat_live.json`): the final checkpoint of CAM's live
planning run `discord-100-planning-20261128T083454Z` (`ckpt_00000039`, 36 decisions; copied with plain reads,
never written) restored by the PS agent: no missing or unexpected weights, no architecture-signature difference;
`restore_report["persistent_state"]` = `legacy_checkpoint: true, rho_initialised: 0.0`. Beliefs `b` and `b_next` of
the run's last 24 stored decisions with the chain attached are bit-identical to those without it. Counterfactual: the
run's 36 decisions replayed into a chain in order and read with the retention open (`ρ = 1`) would have moved those
beliefs by 7.2 % (cosine ≥ 0.997) and the world-model NLL from 0.7585 to 0.7604 — closing `ρ` at the restore is what
keeps an upgrade exact.

Observation: that checkpoint already contains `exp.core.persist_rho` at exactly 1.0 — the live run's learner
process imported the PS model code from the worktree after `e8af02c` (its run directory is in this worktree). Without
a chain (agent.py before PS-1) every core call takes the fresh path, which is bit-identical to the pre-PS core, and
`persist_rho` receives exactly zero gradient, so the run computed what it computed before. The new live run
(`discord-100-planning-20261128T104208Z`, frozen snapshot `a0f3411`: PS model code, no PS-1) is in the same
situation; when it is restarted with PS-1 its checkpoints take the legacy path (`ρ = 0`).

## 5. Diagnosis, negative results and limitations

**Why the carried state does not (yet) improve prediction on the LabWorld / codec objectives — empirical, not a
defect of the mechanism.** (i) The mechanism is correct and learnable: finite differences, decision = replay, and
the synthetic memory task, where the same training path (stored decisions replayed through the chain, one step per
decision) learns to carry a fact 7 decisions past the window and beyond the gradient horizon K. (ii) In the
evaluated objectives the information the predictions need is already carried by something else: an objective of
8–16 decisions fits the window (48 items debug, 256 at 50m); in longer ones (the 40-decision streams) the world's
numeric mechanism is stationary and is accumulated over the whole objective by the fast Bayesian posterior, which is
never truncated within an objective; across objectives LabWorld draws a new hidden mechanism per world, so what the
state carries over is at best family-level knowledge, which the weights also learn. On identical experience the two arms
then learn to predict equally well (debug replay: −0.005 ± 0.003 over 28 pairs; long objectives −0.008 ± 0.006;
full size on three real streams −0.006 ± 0.006 over 9 pairs) and the learner slightly closes the retention. (iii) The
held-out codec block, a different domain after four LabWorld objectives, is the one place with a consistent sign in
favour of the carried state (21 of 28 pairs, p = 0.012 / 0.05 Bonferroni) but the effect is tiny (−0.027 nats,
0.2–0.5 % of the block's NLL); the interfering LabWorld block leans the other way (10 of 28, n.s.). Where the
long-horizon structure that the state exists for occurs — long objectives (the 100-member planning objective will
run for hundreds of decisions, more than the window holds) and information that is observed once and needed much
later — the synthetic result says the mechanism can learn it; it has not been shown on such real experience yet.

**Limitations.**

* **Credit horizon K.** Gradients cross at most `unroll` = 2 decision boundaries; longer dependencies are learned
  through the stored anchors and the intermediate positions' own losses (enough for the 7-decision memory, not shown
  for hundreds of decisions).
* **Anchor staleness.** An anchor is written by the weights of its decision's time and refreshed by K transitions of
  the current weights; records far in the past are read through anchors of much older weights (no re-anchoring pass).
* **Records stored before PS** have no chain position and are read with the fresh state (for the live run: every
  record before the upgrade).
* **Slow opening after a legacy restore.** With `ρ = 0` the restored live run computes exactly what it computed
  before; the learner opens `ρ` at most by the experiential learning rate per step (debug: 0 → 0.004 in 40 decisions;
  synthetic task with lr 3e-3: 0 → 0.4 in 300 steps). The live run will therefore behave as before PS for a long
  time — the safe choice, but it means its persistent state contributes nothing until learning finds it useful.
* **Cost.** K extra core forwards per replayed row and per decision-time block (16 % of a debug decision before the
  refresh tokenisation was batched; the full-size cost is in §4.4).
* **Online on / off outcomes** (progress, prospective NLL) are dominated by trajectory divergence at the sample
  sizes affordable here (§4.3); they are reported, not used as evidence either way.

**When to switch it on.** The switch is `persistent_state.enabled: true`; what would justify it is a paired
fixed-stream replay (`scripts/ps_replay.py`, minutes on the GPU) that shows a gain on the experience it would be used
for. The live planning objective is the natural candidate once it has run longer than the window holds (hundreds of
decisions, with one-off observations such as research results and operator answers needed much later): replaying
its store with and without the chain is the test. A run that switches it on starts from the initial chain state, with
the retention closed if its checkpoint predates the chain, so the switch itself changes nothing until learning opens
`ρ`.

## 6. Reproduce

All CPU commands with `OMP_NUM_THREADS=4 nice -n 10`, `CUDA_VISIBLE_DEVICES=`, one at a time; `S=cam_live/docs/evidence/PS/scripts`.

```
.venv/bin/python -m pytest cam_live/tests/nn -q --rootdir=cam_live                    # 119 passed, 13 skipped
.venv/bin/python $S/check_bitident.py                                                 # vs the pre-PS core
PYTHONPATH=cam_live/src .venv/bin/python $S/ps_recall.py --out OUT --steps 300 --seeds 0 --conds off,K1,K2
PYTHONPATH=cam_live/src .venv/bin/python $S/ps_recall.py --out OUT --steps 300 --seeds 0 --conds K2 --rho-init 0
.venv/bin/python $S/ps_eval.py --cond on --seed 0 --out OUT --phases F1:2,T1:1,F1:1,T2:1 --decisions 8   # and --cond off
.venv/bin/python $S/ps_replay.py --source OUT/off_s0/agent/learner/experience.sqlite --cond on,off --seed 0 --out R.json
.venv/bin/python $S/ps_abl_table.py OUT ; .venv/bin/python $S/ps_replay_table.py R.json
# full size, each takes the shared GPU lock itself (CAM_LIVE_GPU_LOCK when running from a copy):
.venv/bin/python $S/ps_oe_paired.py --plan SW:rle,MECH:7 --objective 0 --decisions 12 --root ROOT --out P.json
.venv/bin/python $S/ps_replay.py --real --source STORE[,STORE] --cond on,off --seed 0 --out RR.json
.venv/bin/python $S/ps_compat_live.py --run-dir RUN --work WORK --out C.json
```
