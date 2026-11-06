# Experiments

All numbers in this document were produced by code in this repository in the development sandbox
(4 CPU cores, no GPU) and are copied from `results.json` / `metrics.jsonl` files written by the
runs named next to them. Nothing was estimated or extrapolated unless explicitly marked.
Model scale is tiny (Pythia-14M for policy, world model and critic): the results speak to the
platform and to tiny models, **not** to frontier-scale agents.

<!-- RESULTS -->

## Protocol

**Environment.** MiniWeb (simulated web apps with server-side verification); see
`docs/architecture.md`. Reward: 1 if the environment verifier confirms success at termination,
else 0; a −0.1 shaping penalty for invalid actions is applied by the RL objective (not by the
environment). Discount γ = 0.95.

**Splits** (no task or answer from a held-out split can enter training; enforced in code):

| split | families | instances | layouts | goal templates | vocabulary |
|---|---|---|---|---|---|
| train (demos + RL) | 10 train families | seed range 0 | default, reversed | all but the last | train vocabulary |
| val (model selection) | 10 train families | seed range 1M | default, reversed | all but the last | train vocabulary |
| test (held-out instances) | 10 train families | seed range 2M | + `distractors` | all, incl. held-out | disjoint vocabulary |
| test_family (held-out tasks) | 4 unseen families | seed range 3M | all | all | disjoint vocabulary |

The held-out families exercise dynamics seen in training in new combinations
(`dialog_confirm`: open dialog → confirm; `scroll_find`: scroll to reveal; `toggle_settings`:
switches + save; `lookup_then_fill`: read a value on one page, type it on another).

**Configurations** (all share the starting checkpoint `EleutherAI/pythia-14m`, the 400
teacher demonstrations (40 per family, ε = 0.15 noise), the RL task stream, reward, evaluation
sets and hardware class):

| | policy init | demos | RL | world model | planning at decision time | imagined training |
|---|---|---|---|---|---|---|
| A policy-only | pretrained | – | – | – | – | – |
| B imitation | pretrained | BC | – | – | – | – |
| C model-free RL | BC | BC | PPO, 25 × 32 episodes | – | – | – |
| D model-based (planning) | BC | BC + WM | same as C | trained on demos + all real experience | MPC (K=4, H=1) | – |
| D + imagination | BC | BC + WM | same as C | same | MPC | Dyna-style, H=2, uncertainty-filtered |

Because training is deterministic per seed and D without imagination uses exactly C's policy
update stream, **D's policy weights are bit-identical to C's** (verified); D-policy vs D-planner
therefore isolates the effect of world-model planning on fixed weights.

**Model selection.** Hyperparameters were chosen on the `val` split only. `test` and
`test_family` were evaluated once per final checkpoint.

**Uncertainty.** Per-run 95% bootstrap intervals over episodes; across seeds, the mean and a
bootstrap interval over seeds. With 2 seeds, seed-level intervals are very coarse.

**Compute accounting.** FLOPs are counted per backbone forward pass (2·params·tokens) and ×3 for
training passes; reported separately for policy training, world-model training, critic training
and inference (including planning). Real environment interactions are counted in steps and
episodes. Cached shared stages (BC / initial world model / critic) are charged to every run that
uses them.

## Findings during development (measured, including failures)

These were found by tests or by monitoring runs; each changed the code or the protocol.

1. **Hash-seed nondeterminism confounded the first comparison.** `product_names()` iterated a
   Python `set`, so `checkout`/`search_lookup` task instances depended on `PYTHONHASHSEED` and
   differed between experiment processes. The pilot comparison (`results/pilot/`) is therefore
   *not* controlled and is not reported as a result. Fixed; a test now regenerates tasks and
   teacher episodes under three hash seeds and requires identical output. After the fix, an
   RL run is bit-reproducible: the final seed-0 C run reproduced the per-iteration statistics of
   the tuning run exactly.
2. **PPO step size.** With lr 1e-4 (full fine-tuning of Pythia-14M), per-update approximate KL was
   0.1–0.4, the clip fraction 25–43%, KL early stopping fired almost every iteration, entropy rose,
   and validation success fell from 0.30 (BC) to 0.175 after 5 iterations
   (`results/tuning/v2_C_s0_lr1e-4_aborted_metrics.jsonl`). On the val split, lr 1e-5 and 3e-5 gave
   KL ≈ 0.002–0.02 and val success 0.30 / 0.325 after 5 iterations; 3e-5 was selected. All C/D runs
   reported below use the tuned value; the aborted lr 1e-4 runs are not reported as results.
3. **Planner-collected data is strongly off-policy.** Training PPO on episodes collected by the
   softmax planner gave approximate KL ≈ 7.6 against the recorded behaviour probabilities in a
   smoke run (immediate early stop). Main runs therefore collect on-policy and plan at decision
   time; planner data is supported via truncated importance weights or search distillation.
4. **Length bias of sequence-likelihood policies.** An untrained LM policy put ≈99% of its
   probability on the shortest action (`finish()`; entropy 0.014), so on-policy RL from scratch
   never received a reward (all advantages zero, no update). Found by a failing test. BC
   initialisation removes it; `length_norm` is available for RL from scratch.
5. **Small demonstration sets overfit.** With 20 demonstrations per family (pilot), BC raised
   held-out imitation accuracy but *increased* validation NLL (4.94 → 5.06). Doubling to 40 per
   family and halving the learning rate gave NLL 4.83 → 2.11 with accuracy 13.6% → 47.5%.
6. **The critic trained on demonstrations is uninformative.** Teacher demonstrations succeed ≈90%
   of the time, so the initial critic learns "high value everywhere" (V(s₀) AUROC 0.25–0.28 on the
   small validation set, mean V(s₀) 0.76). It keeps training on the agent's own returns during RL.
7. **Memory.** In the RL phase one experiment process grows to ≈2 GB; running 8 in parallel on a
   15 GB machine triggered the OOM killer twice. Runs were restarted from scratch (deterministic);
   `scripts/mem_queue.sh` now admits jobs only when memory is available.
