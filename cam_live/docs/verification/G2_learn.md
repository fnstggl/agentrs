# G2-learn — independent re-verification of the neural model (A) and of credit / continual learning (D)

Verifier: G2, lens `g2learn`. I did not write any of the code under test. Scope: workstream A — NA-01 NA-02 NA-03
NA-04 NA-05 NA-07 NA-09 NA-10 NA-11 NA-12 NA-13; workstream D — NA-08 RC-01 RC-02 RC-03 RC-05 RC-06 RC-07 RC-08 CL-01
CL-02 CL-03 CL-04 CL-05 CL-08 CU-20 (CL-10 is claimed by no status file: not judged). Earlier reports whose defects
are re-checked: `G_nn.md` (D1-D11) and `G_cl.md` (M1-M4, m1-m7).

**Code verified:** commit `5562e65aa3fa50bbd36c930c413148d03f28b48e` (later than `b663d5e`); RC-01 and RC-02 at
`3e29595461c5d5d160e82717af77490b90bbc9a9`, because `agent.py` / `reward/inner.py` changed in between (`nn/` and
`credit/` are identical in both). No file under `nn/`, `credit/` or `agent.py` changed between `b663d5e` and
`5562e65` (`git diff --name-only b663d5e 5562e65 -- cam_live/src` lists only harness/runtime files). The worktree carried *uncommitted* edits of other agents in `agent.py`,
`predict/{candidates,predictor,providers}.py`, `reward/assemble.py` and `eval/proposal_probe.py` while I worked; none
of them is imported by `nn/`, `credit/` or their tests (checked), and every run that imports `agent.py` (integration
tests, `probe_agent_credit.py`) and every real-configuration run was repeated or done on a clean `git archive` export
of `5562e65` (PYTHONPATH pointing there), the agent-dependent ones again on an export of `3e29595`. The real
configuration was checked on the GPU (one probe; the pytest GPU modules could not get the shared lock) and on the CPU. Environment: torch 2.15.0+cu132, Qwen/Qwen3-4B (local, offline), RTX A6000 shared.
I edited no source file and no other agent's test. All commands run from `/workspace/camlive`.

## 1. What was executed

| What | Command | Result |
|---|---|---|
| A's default tests | `.venv/bin/python -m pytest cam_live/tests/nn -q --rootdir=cam_live` | 99 passed, 13 skipped, 56 s |
| D's default tests | `... cam_live/tests/credit -q ...` | 114 passed, 7 skipped, 148 s |
| D's and A's slow tests (clean export) | `... cam_live/tests/credit cam_live/tests/nn -q --rootdir=cam_live -m slow --timeout=7200` | 6 passed (5 credit incl. `test_evidence.py`, 1 nn), 910 s |
| Integration acceptance tests of RC-01 / CL-02 / CL-03 (+ CL-05 integrated) | `... cam_live/tests/integration/test_agent_runtime.py::test_delayed_outcomes_link_to_their_decisions ...::test_learning_changes_experiential_and_adapter_weights cam_live/tests/integration/test_int2_checkpoints.py` | dirty worktree 2 + 2 passed; export of `5562e65` 4 passed (284 s); export of `3e29595` (+ `test_int3a_audit.py`) 7 of 8 (G2L-8) |
| G-nn's CPU probes (unchanged) | `cam_live/docs/verification/repro/nn/probe_{fast,stale_cache,presets,invariants [--scale 50m],saveload_cpu}.py` | fast 43/44, stale cache 3/3, presets 12/12, invariants 23/23 (debug and 50m), save / fresh-process load 2/2 + 8/8 |
| G-cl's CPU probes (unchanged) | `cam_live/docs/verification/repro/cl/probe_{rudder,smdp,store_links --kills 12,providers,learner,cocoa --seeds 200 --n 400}.py` (`CL_THREADS=1`) | 63/63, 8/8, 37/37, 4/4 (+ D prints `unsupported`), 26/26, 47/47 |
| A's fast-adaptation evidence, seed 0 full length | `cd cam_live && PYTHONPATH=src ../.venv/bin/python -m cam_live.nn.fast_compare --steps 2000 --seeds 0 --threads 8 --out X` then `repro/g2learn/compare_evidence.py fast X` | 352 recorded numbers, largest difference 0 (22 min) |
| D's debug-scale experiments on the current tree | `PYTHONHASHSEED=0 .venv/bin/python -m cam_live.credit.diagnose --only learning,credit,retention --threads 1 --out X --work W` then `compare_evidence.py diagnose X` | conclusions reproduce; numbers partly differ (section 6) |
| Own probes (new) | `cam_live/docs/verification/repro/g2learn/probe_*.py` | section 2 / 3; outputs in `repro/g2learn/out/` |

Own probes: `probe_fast_stress.py` (fast layer against float64 references on 3750 extreme scalar problems, implicit
gradients, `online_nll` finite differences, NIG predictive, dispersion, impossible observations; 42/46),
`probe_credit_gaps.py` (append-only history, attribution maps, slow-step guard; 13/18), `probe_long_rudder.py`
(RUDDER credit and conservation on an 11-decision trajectory with `max_len` 3 and a random non-zero decomposer; 5/5),
`probe_agent_credit.py` (D's machinery inside the integrated `CamLiveAgent` + `LiveRuntime` + LabWorld + signing
verifier; 5 clean runs), `probe_real_learn.py [--cpu]` (real configuration: counts, bridge no-op, batch invariance
at every bucket edge (GPU only), cache coherence after `p.data = ...` and `load_trainable_state`, argument
sensitivity of `z`; 13/13 on the GPU, 10/10 on the CPU), `probe_real_cpu_grads.py` (real backbone in float32: finite-difference checks
of the embedding and generation gradient paths into LoRA and the bridge, generation; 6/6), `compare_evidence.py`
(re-runs against the committed evidence files), `probe_real_cl.py [--cpu]` (G-cl's real-configuration probe with the one renamed-attribute line adapted, as D's
copy does), `probe_promotion_evidence.py` (held-out evidence at the agent's first promotion).

## 2. Earlier defects — re-checked

| Defect | Claimed | Probe now | Verdict on the fix |
|---|---|---|---|
| D1 GLM mode not converged far from the prior | fixed | G-nn `probe_fast.py` all "Laplace mode reached" rows PASS; my grid: time-to-event 700/700 and binary 425/425 problems exact to 1e-15, count 2621/2625 (the 4 others: new defect G2L-1) | fixed for every realistic regime |
| D2 dispersion tempers binary / time-to-event | fixed | PASS (G-nn probe; mine: binary / TTE posterior identical bit-for-bit at dispersion 3.13, counts move 0.068) | fixed |
| D3 bf16 head of `generation_logprob` | partly fixed | bf16 head not separately re-measured (A's float32-head tests pass in the default suite; GPU pytest not re-run) | head fixed, trunk error disclosed |
| D4 NIG `predict()` variance | fixed | mine: predict() = Student-t variance (ratio 1.000000000000 over 10 000 predictions), epistemic part = Monte-Carlo variance of `sample_weights` (≤ 0.7 %), Student-t interval on the correct scale covers 0.9000 (n 10 000). G-nn's NIG calibration check now FAILs (0.9238) only because it standardises by the (now correct) variance and applies Student-t quantiles to it — the check itself is invalidated by the fix, as A notes | fixed |
| D5 impossible observations absorbed | fixed | PASS (10 impossible cases raise and leave the state bit-identical; 4 possible boundary cases accepted; masked placeholder rows ignored) | fixed |
| D6 phi ill-conditioned on real inputs | not fixed (reported) | — | disclosed limitation, unchanged |
| D7 one argument barely moves z | fixed | debug: cos(z) 0.639 (click e3 vs e4); real configuration: 0.752 (CPU) / 0.738 (GPU) | fixed |
| D8 embedding depends on batch composition | fixed | tiny backbone: 90 comparisons bit-exact (bucket edges 7/8/9/63/64/65/127/128/129/255/256/257, > max_text_tokens, more rows than one forward); bf16 GPU backbone: the same 90 comparisons max \|diff\| 0, `with_grad` value canonical, belief rel. 2.3e-7 | fixed |
| D9 `arch_signature` misses `max_text_tokens` | fixed | `probe_saveload_cpu.py` 8/8 | fixed |
| D10 caches stale after `.data` writes | fixed | `probe_stale_cache.py` 3/3; mine (tiny): `lora_B.data = new tensor` (new storage) and `load_trainable_state` give bit-exactly the cold-model embedding / logits | fixed |
| D11 tests that cannot fail | fixed for the named tests | — (test code read) | fixed as stated |
| M1 estimator selection in-sample | fixed | G-cl probe C PASS (0.00 of 24 experiment records in the training split); re-run of `diagnose` on the current tree: 0 experiment lifetimes trained on, nothing promoted | fixed in D's lab (but see G2L-6 for the agent) |
| M2 mutable history | fixed | G-cl `--only immut` PASS; mine: protected diagnostics, probabilities, a second result replacement refused; a tampered index column raises `StoreIntegrityError` | fixed for records; outcomes appended through `update()` bypass a guard (G2L-2) |
| M3 negative attribution amounts | fixed | G-cl `--only links` PASS; mine: unknown ids, booleans, strings, zeros: weights ≤ 1, credit ≤ outcome | fixed |
| M4 slow guard admits degrading steps | fixed | debug: a step that raises held-out loss by 0.0012 (tolerance 1e-6) is undone; D's regression tests pass | fixed when held-out cells exist; fail-open and masking remain (G2L-4, G2L-5) |
| m1 open-ended sets supported | fixed | G-cl probe D prints `unsupported`; agent records `proposal_probs` only for sets without backbone proposals | fixed |
| m2 COCOA pairs omit earlier-declared outcomes | fixed | G-cl probe B PASS (0 pairs left out) | fixed |
| m3 long trajectories | fixed | G-cl probe A PASS; mine: every one of 11 decisions credited with `max_len` 3, sum_t Gamma_t c_t = G to 1e-9 with 10 non-zero contributions | fixed |
| m4 revert leaves Adam state | fixed | G-cl `probe_learner.py` PASS (0 adapter tensors keep state) | fixed |
| m5 evidence not regenerable | fixed | D's `diagnostics.json` comes from `e6f51c7`; on the current tree the learning numbers reproduce to 1e-6, credit and retention numbers differ slightly because `nn/backbone.py` changed after `e6f51c7` (section 6) | fixed as a process; numbers are those of `e6f51c7` |
| m6 name-based promotion filter | fixed | G-cl probe PASS; in the agent no registered provider is trained on the internal reward, and `promote_learner` declares provenance per provider | fixed |
| m7 real reload depends on caches | holds after A's e00d28f | CUDA: writer predictions independent of caches (0.0), fresh-process reload bit-exact on 4 records (`probe_real_cl.py`) | fixed |

## 3. New defects

Severity as in G_nn / G_cl: major = a stated property is refuted in normal operation; minor = narrow effect.

### G2L-1 — minor — count mode search reports convergence at an absurd mode for observations far above the prior predictor (NA-07)
`src/cam_live/nn/fast.py:318-354` (`_mode_poisson`). From the left, the first Newton step lands at
x1 ≈ eta0 + c·s/disp. When the prior predictor e·exp(eta0) is below ~1e-16·disp/s, `L = c + k (eta0 - x)` (line 338)
cancels to ≤ 0 in float64 although it is positive in exact arithmetic, so the convex G-step is disabled (`okg` requires
`L > 0`, line 343), the r-step in the `big` branch moves by 1 per iteration, and the *relative* stopping rule
`|step| <= 1e-11 (1 + |x|)` (line 350) is already met when |x1| ≳ 1e11. Result: `converged=True` with the posterior
mean at the overshoot. 4 of 2625 grid problems, e.g. prior eta −30, variance 100, dispersion 0.3, count 1e9 at
exposure 1e-12: posterior eta 3.33e11 instead of 48.35 (1e15 with variance 1e6); the next predicted count is capped
at e·exp(40) = 2.4e5. Reachability in the agent is low: `predict/predictor.py` passes the head's predicted count as
exposure (floor 1e-9), which keeps e·exp(eta0) far above 1e-16 unless the fast posterior's log-multiplier is below
about −16. Repro: `.venv/bin/python cam_live/docs/verification/repro/g2learn/probe_fast_stress.py | grep -i "count"`.
Suggested regression test: the two `FastBeliefState.update` cases printed by the probe must reach the bisection root
or report `converged=False`. (Aside: `FastPrior(prior_var>~5e5)` raises `OverflowError` in `_inv_softplus`.)

### G2L-2 — minor — `SqliteExperienceStore.update()` accepts appended outcomes dated before their query, even before the decision (RC-07, RC-01)
`credit/replay.py:705` (`update` → `_write` → `history_changes`, lines 130-200): appended outcomes are not checked
against the query's `issued_at` / the decision time, while `add_outcome` refuses them (line 727, "observed before the
query was issued"). An outcome observed at t = 900 for a query issued at 1015 (decision at 1010) is stored through
`update()`; so is one dated 500 s before the decision. D's RC-07 limitation "an outcome dated before its query is
refused" holds for `add_outcome` only. The agent uses this path (`agent.py:908-916` at `5562e65`, `:1033-1041` at `3e29595`, `_attach_self_outcome`:
`rec.outcomes.append(obs); self.store.update(rec)`), but checks the timing itself first (`_resolve_observed`:
reading after `made_at` and not before `due`); in seven integrated runs no stored outcome predates its decision.
Repro: `.venv/bin/python cam_live/docs/verification/repro/g2learn/probe_credit_gaps.py --only store`.

### G2L-3 — observation — attribution amounts that overflow when summed drop the outcome (RC-01)
`credit/links.py:258`: `{a: 1e308, b: 1e308}` sums to inf, both shares become 0 and the event gets no link
("no evidence") although it names two decisions with equal amounts. Artificial input; no credit is created. Repro:
`probe_credit_gaps.py --only links`.

### G2L-4 — minor — the slow-step guard fails open when there are no held-out cells (CL-03, CL-08)
`credit/learner.py:741, 778, 798-800`: `val_before = self._validation_cells()` is empty when no provider has a
validation sample; then neither the optimizer state is saved nor a verdict computed, and the step is kept with no
`guard` note. In D's lab with `val_fraction=0` a step that raises the (probe) held-out loss is applied (adapters
changed, verdict `None`); a held-out check that cannot be made is treated as passed. In the integrated agent (debug
scale, 3 lifetimes per run, runs differ because ids carry `os.urandom` bytes) 6 of 7, 1 of 10 (both on the dirty
worktree), 0 of 7, 1 of 9, 1 of 7, 3 of 11, 1 of 5 (export of `5562e65`) and 0 of 7, 2 of 15 (export of `3e29595`)
slow steps ran without a verdict; most guarded steps had one
validation chunk (`paired_se` NaN, tolerance at the float floor). D's claim "a slow step is kept only if the held-out
loss ... rises by at most min(0.01, 5 %, 2 SE)" is therefore not true as written. Repro: `probe_credit_gaps.py --only
guard` (check "without held-out records ...") and `probe_agent_credit.py` ("every applied slow step ... carries a
held-out guard verdict").

### G2L-5 — minor — the guard judges the mean over providers, so one provider's degradation is hidden by another's gain (CL-03)
`credit/learner.py:533-565` (`_guard_verdict`): cells of all providers are averaged per chunk (line 543) before the
paired change is formed, with no per-provider bound and no scale normalisation. Two probe providers in D's lab: one
improves by 0.0012, the other's held-out loss rises by 0.0006 = 10x the tolerance (5.9e-5): the step is accepted
(aggregate −0.0003). In the integrated agent the validation losses at the end of a run were rudder 0.04,
cocoa 0.47, bridge_grounding 19.6 (the world-model / evaluator providers had no validation sample), so the verdict is
essentially that of the grounding loss the slow step itself optimises. Repro: `probe_credit_gaps.py --only guard`.

### G2L-6 — major for RC-08 — evidence-based estimator selection is not connected in the integrated agent
Nothing outside `credit/` calls `CreditAssigner.compare_with_experiment` or `EstimatorSelector.select` (grep over
`src/`); in all seven clean integrated runs (5-9 randomised decisions in each run that counted them)
`<run>/learner/credit_selection.jsonl` stays empty. The agent registers its experiments in
`predict.registry.ExperimentRegistry`, not in the experience store (`store.experiments()` is empty), so
`compare_with_experiment` (`credit/providers.py:888-…`, `st.experiment(...)`) would treat them as not randomised;
each agent experiment is one randomised decision (one run: 5 experiment ids with 1 unit each), while an arm
difference needs at least two units per arm; and the agent sets `holdout_experiments=False` (`agent.py:611-615` at `5562e65`, `:722-726` at `3e29595`), so every
objective with a randomised decision is one the learner trained on (2 of 2 or 3 of 3 per run) and would be excluded from
the comparison. The default therefore stays `simple` by construction, never by evidence, and spec 9.4's "compare
COCOA-inspired estimates with actual observed experimental differences where possible" never happens in the agent.
D's own selection in its lab is correct (M1 fixed). Repro: `probe_agent_credit.py` (INFO lines "estimator selection
log rows", "randomised experiments ...", "objectives with randomised decisions ...").

### G2L-7 — RC-02 gap carried over from G-cl — open at 5562e65, closed at 4476c27 (verified at 3e29595)
At `5562e65`, `CreditAssigner.credited_returns` (`credit/providers.py:871`) had no caller in `src/` and no loss read
`record.credit` (spec 9.2: "Use redistributed credit during eligible training updates"). INT3A's KF-E01 (`4476c27`)
connects it outside `credit/`: `agent._credit_for` hands the evidence-selected estimator's stored credit to
`reward.inner` (`credit_fn`), where it becomes the goal term of the real policy update (return-like for RUDDER /
COCOA, per decision for `simple` in objectives whose environment attributes outcomes). On an export of `3e29595`:
`tests/integration/test_int3a_audit.py` 4/4 (with the default forced to `rudder` the credited return-to-go replaces
the goal return and changes the gradient; without credit the update is bit-for-bit the old one), and in two
integrated runs the hook returns stored `simple` credit for 20 / 20 and 34 / 34 records. RUDDER / COCOA credit can
reach training only once evidence selects them, which the agent never evaluates (G2L-6).

### G2L-8 — minor, regression at 3e29595 — the agent's first promotion often finds no held-out evidence (CL-05 integrated)
`tests/integration/test_int2_checkpoints.py::test_promotion_needs_independent_evidence_and_a_refusal_rolls_back`
passes 3 / 3 runs on an export of `5562e65` and fails 3 / 4 on `3e29595` (first promotion refused: "no independent
validation metric supplied"). `probe_promotion_evidence.py` (the same two lifetimes): 3 / 3 promoted at `5562e65`,
2 / 3 at `3e29595`; in the refused run `learner.validate()` is empty and no record carries a `val:*` flag. The gate
itself behaves correctly (fail-closed); what changed between the commits is the agent (only `agent.py`, `predict/`,
`reward/`, runtime files differ; `credit/` did not), and in both the held-out set of a 14-decision lifetime holds
0-5 records chosen by a record-hash split over ids that carry `os.urandom` bytes, so promotion — like the slow-step
guard (G2L-4) — depends on chance. Not bisected. Repro: run the test or `probe_promotion_evidence.py --reps 3` with
PYTHONPATH on exports of both commits.

### Out of scope, reported for the owner (F, `runtime/logs.py:59`)
The runtime's log redactor treats every run of 17-20 digits as a Discord snowflake and pseudonymises it; floats inside
the JSON string `LearningUpdateRecord.notes` (e.g. `0.00875766109675169`) become `0.member:<hash>`, so in
`RunFiles.LEARNING` 34-42 of 56-65 learning records per run no longer have parseable notes (the learner's own log
`<run>/learner/learner_updates.jsonl` is intact). Repro: last INFO line of `probe_agent_credit.py`.

## 4. Real configuration (GPU) and clean-export integration runs

**Shared GPU.** The shared lock was held by other tenants for most of the session (an S evaluation sequence from
22:41, then INT3B / INT3A runs that wait as *blocking* flock waiters, so my jobs polling with `gpu_lock` never got a
turn). After switching my probes to the project's queued lease (`runtime.gpu_cycle.GpuLease`, a blocking waiter like
the others; `repro/g2learn/_common.queued_gpu_lock`) `probe_real_learn.py` ran on the GPU at 01:13 (waited 852 s,
held the lock 22.6 s, peak 9.3 GB; export of `3e29595`, whose `nn/` equals `5562e65`'s): **13 / 13**:

| Check (real Qwen3-4B bf16 + LoRA + bridge + 50m core, CUDA) | Result |
|---|---|
| counts / groups / frozen | 4,022,468,096 frozen bf16 parameters, none requires grad; LoRA 11,796,480 (72 modules), bridge 4,917,376, experiential 50,953,765; groups disjoint over 704 parameters |
| bridge | logits bit-identical with and without latents of norm 1e-3 / 1 / 1e3 at init; one Adam step on the bridge alone: TV(latent, −latent) 1.0000 |
| D8 batch invariance | 15 texts of 1 / 7 / 8 / 9 / 63 / 64 / 65 / 127 / 128 / 129 / 255 / 256 / 257 / 300 / 434 tokens (the last above `max_text_tokens`), each embedded alone and in 6 shuffled calls with the others and with more texts of one bucket than one forward holds: 90 comparisons, max \|diff\| 0; `embed(with_grad=True)` = canonical value bit-exactly; belief of a context alone vs in a padded batch: rel. 2.3e-7 |
| D10 | after `lora_B.data = new tensor` (72 modules) embedding and no-grad logits equal a cold recomputation exactly; after `load_trainable_state` back to the original bit-exactly |
| D7 | cos(z) click e3 vs e4 0.738, api rows 100 vs 101 0.912 |

A's and D's `-m gpu` pytest modules were not re-run: their fixtures poll the lock (`LOCK_NB` every 5 s) and lost
to the blocking waiters for 30 min twice (pytest's 900 s timeout errored the first attempt). Output:
`repro/g2learn/out/probe_real_learn_gpu.txt`. Then `probe_real_cl.py --part write` / `--part read` (D's learner on the real configuration, CUDA; held the lock
≤ 5.5 min, peak 9.4 GB): `poll()` triggers on 16 independent matured units; one frequent update changed 127
experiential + 56 head tensors and no adapter / reward tensor, predictions moved (max \|db\| 1.11, \|dz\| 1.92,
\|dNLL\| 0.214); one slow step changed 144 / 144 LoRA + 26 / 28 bridge tensors and nothing else (guard: paired
change +0.00025 accepted under tolerance 0.0019, 6 chunks; by-loss adapter gradient bridge grounding 7.0 vs world
0.026); frozen hash unchanged; the writer's predictions do not depend on inference caches; a weights-only export
reloads in a fresh CUDA process with identical group hashes and bit-identical predictions (m7 holds; CL-06 itself is
outside this lens). Outputs `out/probe_real_cl_gpu_{write,read}.txt`.

Before that, the **real configuration was built on the CPU** by the same code (`build_model(CamLiveConfig(device="cpu"))`,
real Qwen3-4B weights; on the CPU the backbone loads in float32, so bf16-specific properties are not judged there):

| Check (real Qwen3-4B + LoRA + bridge + 50m core, CPU) | Result |
|---|---|
| counts from the module tree: frozen / LoRA / bridge / experiential | 4,022,468,096 / 11,796,480 / 4,917,376 / 50,953,765 (= A.json); 72 LoRA modules, rank 32, q/v; groups disjoint over 704 parameters; no frozen parameter requires grad |
| bridge at initialisation | next-token logits bit-identical with and without latents of norm 1e-3, 1, 1e3; greedy text identical ("Red.") |
| one Adam step on the bridge alone | TV(latent, −latent) of the next-token distribution 0.9997 |
| embedding path → LoRA-B | finite differences / analytic 1.0001 (h 1e-2), 0.9997 (h 3e-3); 72 / 72 modules get gradient; `embed(with_grad=True)` value = canonical cached value, gradient rel. diff 7.8e-7 |
| `generation_logprob` → LoRA-B + zero-initialised bridge output | gradient norms 64.5 / 59.4 at init; finite differences / analytic 0.9994 / 0.9999 |
| D10 caches after `lora_B.data = new tensor` (72 modules) and after `load_trainable_state` | embedding and no-grad logits equal a cold recomputation exactly; back to the original bit-exactly |
| D7 | cos(z) click e3 vs e4 0.752 (A: 0.747 on the GPU), api rows 100 vs 101 0.915 |
| D's learner on the real configuration (G-cl's `probe_real` copy, `--cpu`, own parameter snapshots; LabWorld lab, 30 records) | `poll()` triggers on 16 independent matured units; one frequent update changed 127 experiential + 56 head tensors, 0 adapter / reward, predictions moved (max \|db\| 1.10, \|dz\| 1.90, \|dNLL\| 0.214); one slow step changed 144/144 LoRA + 26/28 bridge tensors and nothing else, guard accepted (paired change −0.0063, 6 chunks, tolerance 0.0050), by-loss adapter gradient bridge grounding 6.92 vs world 0.026; frozen hash unchanged; writer predictions independent of inference caches (0.0) |
| weights-only export → fresh CPU process | group hashes identical; predictions differ by ≤ 7e-5 (\|db\|) because the reader ran with another CPU thread count than the writer (16) — G-cl's check is a CUDA bit-exactness claim of CL-06, outside this lens, not judged |

Outputs: `repro/g2learn/out/probe_real_learn_cpu.txt`, `out/probe_real_cpu_grads.txt`, `out/probe_real_cl_cpu_{write,read}.txt`.

**Clean-export integration runs** (`git archive 5562e65`, PYTHONPATH there; repeated on `3e29595` below): the four integration acceptance tests
(RC-01 linking, CL-02 / CL-03 learning, CL-05 promotion and rollback twice) passed in 284 s; `probe_agent_credit.py`
ran 5 times (3 lifetimes each; the agent is not reproducible across processes because ids carry `os.urandom` bytes):
every matured declared outcome linked, per-outcome link weights ≤ 1, no link to a later decision, no outcome before
its decision, empty leakage audit, learned credit written for closed objectives (1-3 of 3 closed per run), no COCOA
value on an open-ended untracked set, every frequent update (16-34 per run) changed experiential weights and
behaviour; slow steps without a guard verdict 0/7, 1/9, 1/7, 3/11, 1/5; estimator-selection log empty in every run;
outputs `repro/g2learn/out/probe_agent_credit_clean_*.txt`.

**Newer commit `3e29595`** (agent.py changed: INT3A's KF-E01 and prompt fitting; `nn/`, `credit/` unchanged): D's
default tests 114 passed / 7 skipped; integration: `test_int3a_audit.py` 4/4, RC-01 linking and learning tests pass,
`test_int2_checkpoints.py` 1 of 2 (G2L-8); `probe_agent_credit.py` 2 runs: 13/13 and 12/13 (the failure: 2 of 15
slow steps without a guard verdict, G2L-4), estimator-selection log empty, stored credit reaches the policy update
(G2L-7). Outputs `repro/g2learn/out/*_3e29595*.txt`.

## 5. Per requirement: basis of the verdict

| Req. | Verdict | Basis (what I ran; what fails) |
|---|---|---|
| NA-01 | verified | Real configuration built by `build_model(CamLiveConfig())`: Qwen/Qwen3-4B, 4,022,468,096 frozen parameters recomputed from the module tree, none requires grad; embeddings, next-token logits, `generation_logprob` and greedy generation ("Red.") run through the real weights (CPU float32 run of the same code path, section 4). On the GPU (bf16, section 4): 4,022,468,096 frozen bf16 parameters without grad, same counts, bridge / cache / batch checks 13 / 13. Qwen3-8B never loaded (disclosed). |
| NA-02 | verified | Experiential group 50,953,765 on the real configuration (= A.json); causality / gradient / recall / preset tests pass; G-nn `probe_invariants.py` 23/23 at debug and 50m scale. |
| NA-03 | verified | LoRA rank 32 on q/v of 36 layers = 11,796,480 (72 modules), bridge 4,917,376, adapters 16,713,856: LoRA and group inside 10-20 M (module tree of the real configuration, CPU and GPU). |
| NA-04 | verified | Groups disjoint and covering all 704 parameters of the real configuration (CPU and GPU); counts = A.json; fresh-process round trip, group-hash and recorded-evidence tests pass; D9 fixed (`probe_saveload_cpu.py` 8/8). |
| NA-05 | verified | Shared z / phi used by every head (tests); D7 fixed (cos(z) click e3 vs e4: 0.639 debug, 0.752 / 0.738 real configuration CPU / GPU, was 0.9997); D8 fixed: bit-exact on the tiny backbone and on the bf16 GPU backbone at every bucket edge (90 comparisons each, section 4), belief rel. 2.3e-7. D6 (phi ill-conditioned at init) and usefulness after training remain disclosed limitations, not refutations of function. |
| NA-07 | verified | All 17 acceptance tests pass. D1 / D2 / D4 / D5 fixed (section 2): on 3750 extreme scalar problems the mode equals a float64 bisection to ≤ 9e-16 (TTE, binary) and the implicit-function gradients equal the closed form to ≤ 4e-14; `online_nll` gradients equal 4th-order finite differences to ≤ 1.5e-7 on a year-in-seconds censored sequence, counts at exposure 1e7 / 1e9, 0 of 1e6 trials, and the stateful path gives the same per-row surprises bit-for-bit. Minor defect G2L-1 (4 of 2625 count problems, prior predictor ≥ 16 orders below the count: wrong mode reported as converged) — outside any regime the predictor produces. |
| NA-09 | verified | Real backbone, float32 (CPU): embedding-path gradient to LoRA-B = central finite differences (ratio 1.0001 / 0.9997), `embed(with_grad=True)` = canonical value + exactly that gradient (rel. 7.8e-7), 72/72 LoRA modules get gradient; generation path to LoRA-B and the zero-initialised bridge output = finite differences (0.9994 / 0.9999); caches follow `p.data = ...` and `load_trainable_state` (D10). Adapters updated by experience in the integrated agent (5-15 slow steps per run) and on the real configuration (CUDA and CPU: 144/144 LoRA + 26/28 bridge tensors per slow step). On the GPU the caches follow `p.data = ...` / `load_trainable_state` bit-exactly; bf16 gradient accuracy (A's fp32-reference GPU test) not re-run. |
| NA-10 | verified | Default + presets tests pass; G-nn's `probe_presets.py` 12/12 on the current code (5m … 1b instantiated with real weights, counts = claims, forward finite, backward ≤ 150m). Only 50m ever ran on the GPU (disclosed). |
| NA-11 | verified | Token tests pass (every record type / item kind, exact float64 values, value channel, time, hashes); argument leaves now reach z (D7). A's GPU token test not re-run (section 4); argument sensitivity of z on the GPU 0.738. |
| NA-12 | verified | Real weights: logits and greedy text bit-identical with / without latents at init (norms 1e-3..1e3), non-zero gradient to the zero-initialised bridge output equal to finite differences, one Adam step on the bridge alone gives TV(latent, -latent) 0.9997. On the GPU (bf16): logits bit-identical at init, TV 1.0000 after one bridge step. Usefulness of a trained bridge not shown (disclosed). |
| NA-13 | verified | Seed 0 of the recorded comparison re-run at full length on the current code: all 352 recorded per-seed numbers (mean and per-support NLL / RMSE, 4 methods x 2 targets x 2 splits) reproduce with difference 0; the stated scientific failures are in the file. Synthetic tasks, seed 1 not re-run. |
| NA-08 | verified | Default tests, G-cl `probe_learner.py` 26/26; integrated agent (5 clean runs): every frequent update changed experiential weights; `diagnose` learning experiment reproduces held-out NLL 2.981 → 2.291 on the current tree. Real configuration (CUDA and CPU, section 4): 127 experiential + 56 head tensors changed, predictions moved, adapters / reward untouched. No real-configuration learning curve (disclosed). |
| RC-01 | verified | Pinned to `3e29595` (`agent.py`, one of RC-01's files, changed after `5562e65`): the acceptance tests pass on exports of both commits, including `tests/integration/test_agent_runtime.py::test_delayed_outcomes_link_to_their_decisions`. M3 fixed (G-cl probe and my attribution-map attacks: weights ≤ 1, credit ≤ outcome). In 7 clean integrated runs (5 at `5562e65`, 2 at `3e29595`) every matured declared outcome is linked, per-outcome weights ≤ 1, none to a later decision. G2L-3 (overflowing amounts drop an outcome) is an artificial-input observation. |
| RC-02 | verified | Pinned to `3e29595`. Decomposer, return-to-go identity, causality, full-trajectory credit (m3) verified (G-cl `probe_rudder.py` 63/63, `probe_providers.py` A, my 11-decision `max_len` 3 run). The gap open at `5562e65` — no training update consumed any credit (G2L-7) — is closed by INT3A's KF-E01: `test_int3a_audit.py` 4/4 and the integrated hook returns stored credit for 20/20 and 34/34 records. RUDDER's credit reaches training only once evidence selects RUDDER, which the agent never evaluates (G2L-6, counted under RC-08). |
| RC-03 | verified | sum_t Gamma_t c_t = G with an own Gamma: G-cl probe (error 0.0, beta 0 … 0.05) and mine over the whole 11-decision trajectory with random non-zero decomposer weights (|err| < 1e-9, 10 non-zero contributions, 36 000 s span). |
| RC-05 | verified | Acceptance tests pass; G-cl `probe_cocoa.py` 47/47; m1 (provenance-based open-endedness, untracked → `unsupported`, G-cl probe D) and m2 (Eq. 4 pairs, probe B) fixed. The agent records inclusion probabilities (`proposal_probs`) only for sets without backbone proposals; no COCOA value on an open-ended untracked set in the integrated runs (on the tiny backbone every set was tracked, so the agent-side refusal was not exercised there). |
| RC-06 | verified | Acceptance tests pass. In the integrated agent (clean export) signed verdicts reach `put_objective_outcome` (1-3 of 3 objectives per run), RUDDER / COCOA credit is written and re-versioned for decisions of closed objectives, the retrain queue is fed; D's "nothing outside credit/ does so yet" is stale. |
| RC-07 | verified | Acceptance tests pass; leakage audit empty, no link to a later decision and no outcome dated before its decision in 5 integrated runs; history immutable (M2 fixed, tampered index detected). Minor defect G2L-2: `update()` accepts appended outcomes dated before their query / decision (the agent's own path checks the timing first). |
| RC-08 | partial | In D's lab the selection is now held out by lifetime with paired standard errors and refuses zero SE (M1 fixed; re-run on the current tree: 0 of 288 experiment lifetimes trained on, RUDDER mean z^2 6.54, COCOA 11.32, nothing promoted, default `simple`). **Fails in the integrated agent** (G2L-6): nothing calls `compare_with_experiment` / `select`; `credit_selection.jsonl` stays empty; the agent's experiments are not in the store and each randomised decision is its own experiment with one unit; `holdout_experiments=False` makes all of them trained-on. The default is `simple` by construction, and estimates are never compared with the agent's experimental differences. |
| CL-01 | verified | 19 acceptance tests pass (kill -9 included); G-cl `probe_store_links.py --kills 12` 37/37 (12 SIGKILLs, no acknowledged record lost; strata quotas exact; provenance never mixed); M2 fixed (immutable history, index checked on every read). G2L-2 concerns `update()`'s outcome timing only. |
| CL-02 | verified | Acceptance tests pass; G-cl `probe_learner.py` 26/26; integrated agent (clean export, 5 runs): every frequent update (16-34 per run) changed experiential weights and the behaviour probe; `diagnose` learning experiment on the current tree reproduces held-out NLL 2.981 → 2.291 (to 1e-6). Real configuration (CUDA and CPU): one frequent update changed 127 experiential + 56 head tensors and the predictions (section 4). |
| CL-03 | partial | Adapter updates occur: debug lab (every `adapter_every`), integrated agent (5-15 slow steps per run changing LoRA and bridge), real configuration on CUDA and on the CPU (144/144 LoRA + 26/28 bridge tensors, nothing else). **Fails:** the claimed guard "a slow step is kept only if the held-out loss rises by at most min(0.01, 5 %, 2 SE)": with no held-out cells the step is kept unguarded (G2L-4; 8 of 61 slow steps in 5 clean integrated runs had no verdict, a degrading step is applied in D's lab), and the bound is on the provider-averaged loss (G2L-5: a provider's held-out loss rose 10x above tolerance and the step was kept). M4's original scenario (a degrading step with a measurable held-out loss) is now refused and undone with its optimizer state (m4 fixed). |
| CL-04 | verified | Acceptance tests pass; `diagnose --only retention` on the current tree: NLL increase on family A 1.768 ± 0.426 naive vs 0.316 ± 0.129 with retention, decision agreement 0.34 vs 0.83 (recorded 1.768 / 0.357 / 0.86, from `e6f51c7`). One family pair, one seed, debug scale, replay and regulariser not ablated (all disclosed). |
| CL-05 | verified | Acceptance tests pass; `tests/integration/test_int2_checkpoints.py` 2/2 on the clean export (integrated promotion needs independent evidence; a refusal rolls back); G-cl promote probe PASS; m6 fixed (provenance). In short integrated runs early promotions are refused for missing validation / retention metrics (fail-closed), later ones succeed. At `3e29595` the INT2 promotion test fails 3 of 4 runs (G2L-8: no held-out evidence in short lifetimes, worse than at `5562e65`, 0 of 6 refused there); CL-05's own files are unchanged, the gate fails closed as it should. |
| CL-08 | verified | Acceptance tests pass; G-cl learner probe (F F S F F F S, trigger counts independent units); integrated agent: every frequent update triggered by matured independent outcomes (`matured_outcomes:k_independent/k_rows`), slow steps every `adapter_every`. G2L-4 affects the guard of the slow timescale, not its scheduling. |
| CU-20 | verified | Acceptance tests pass; G-cl `probe_smdp.py` 8/8 (discount composes, one event credited once over the hierarchy, one learning unit, ledger refuses double counting); per-outcome link weights ≤ 1 in the integrated agent. `CreditLedger` is not wired into `CreditAssigner` (disclosed). |

## 6. Status-file statements that are not true as written

* `D.json` `not_done[0]` "Not integrated: agent.py and the runtime do not construct the learner, the assigner or the
  linker yet"; NA-08 "The learner is not yet called by agent.py"; RC-01 "Not yet called from the agent's reconcile
  step"; RC-06 "nothing outside credit/ does so yet" (`put_objective_outcome`); CL-05 "nothing in the runtime calls
  promote_learner yet" — all stale: `agent.py` constructs and calls all of them (lines 593-700, 873, 931, 1083, 1840,
  2115), and the integration tests exercise them.
* `D.json` RC-07 "an outcome dated before its query is refused" — only through `add_outcome` (G2L-2).
* `D.json` `not_done` "credited returns are exposed but not consumed" — stale since INT3A's `4476c27` (KF-E01, G2L-7).
* `D.json` CL-03 / fix_stage M4 "a slow step is kept only if the held-out loss ... rises by at most ..." — not when
  there is no held-out cell (G2L-4), and only for the provider-averaged loss (G2L-5).
* `D.json` `measured` (credit / retention numbers) come from `e6f51c7`; on the verified commit the same command gives
  COCOA mean z^2 11.32 / RMSE 3.83 (recorded 13.84 / 4.23), RUDDER 6.54 / 3.41 (6.61 / 3.40), retention arm NLL
  increase 0.316 ± 0.129 (0.357 ± 0.135), decision agreement 0.833 (0.856); learning numbers identical to 1e-6.
  Conclusions unchanged (nothing promoted, forgetting reduced ~5x).
* `A.json` NA-07 "the mode is exact ... iterations and convergence are returned" — not in the regime of G2L-1.

## 7. Not checked

Qwen3-8B; presets above 50m on the GPU; that trained representations / a trained bridge / adapter updates improve
anything (no learning curve on the real configuration exists — A and D say so); power-loss durability; the
integrated agent on the real configuration (INT2's evidence, not re-run here); seed 1 of the fast-adaptation run;
CL-10 (unclaimed).
