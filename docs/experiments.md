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
