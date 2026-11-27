# Reward meta-updates with non-zero meta-gradients inside the integrated agent, real configuration (2026-11-27)

`meta_real_20261127T184014Z.json` — `run_meta_real.py` at commit dcb9592 (+ uncommitted logging only), Qwen3-4B
(frozen, bf16) + LoRA + bridge + 50m core on the RTX A6000 under the shared lock (946 s held, peak 10.4 GB). One
`CamLiveAgent` ran 5 LabWorld F1 lifetimes (12 decisions each, k = 16, close-out to the deadline; verified progress
0.69, 0.18, 0.76, 0.48, 0.25). 63 meta-update calls: **4 applied with non-zero meta-gradients**, 56 skipped
`insufficient_independent_outcomes`, 2 `no_policy_update_recorded`, 1 `no_supported_episode`. The reward-parameter
hash changed in lifetimes 1–4.

The pathway of every applied update (`applied_pathway`, recorded by `validation.MetaStepProbe`):

| meta-gradient norm | psi delta (L2) | independent groups / query rows | estimator / ESS | internal reward on the support decisions, mean abs change | inner update theta_0 -> theta_K, mean TV on support | policy change caused by the meta-step (theta_K with psi before vs after), mean TV / max abs prob |
|---|---|---|---|---|---|---|
| 0.050 | 0.096 | 9 / 3 | importance-weighted / 2.92 | 0.067 | 0.015 | 0.027 / 0.0089 |
| 0.831 | 0.073 | 9 / 6 | importance-weighted / 2.59 | 0.017 | 0.010 | 0.006 / 0.0023 |
| 1.837 | 0.061 | 9 / 9 | importance-weighted / 2.40 | 0.052 | 0.015 | 0.004 / 0.0015 |
| 2.886 | 0.064 | 8 / 6 | importance-weighted / 2.11 | 0.061 | 0.014 | 0.008 / 0.0029 |

Read: psi -> R_psi on the support decisions (changes by the step) -> the inner policy update (the policy moves by TV
~0.01–0.015 on its support) -> the query decisions taken afterwards carry the verifier's signed outcomes (utility
mean 1.76 in the first update) -> outer meta-gradient on psi (non-zero) -> psi changes (reward version
`rpsi1-...645872a78a` -> `...fb66d91346`) -> recomputing the same inner update from its snapshot with the new psi gives
a different policy (TV 0.004–0.027): the meta-step changes the next inner update.

Limits: importance weighting (the live policy update uses the bootstrap mix 1, the differentiated one the learned
reward, so the logged policy is not the recomputed one); ESS 2–3; effects are small; LabWorld is a diagnostic world;
whether R_psi's changes make later decisions *better* is not established by this run. The backbone's proposals in
LabWorld were mostly refused by the strict parser for invalid arguments (3 of 4 per decision; see INTEGRATION.md §8).
