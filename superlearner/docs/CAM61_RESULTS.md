# CAM 6.1: status and measured results

Branch `feat/cam-6.1`, built from `feat/superlearner-4` (`7909214`) and `feat/crucible-4` (`c453d27`). Code is in
`superlearner/src/superlearner/unified/`. Raw results are in `runs/cam61/` (git-ignored).

## Verdict

**CAM 6.1 has not met its success criterion.** It runs end to end and its weights learn from real Crucible experience,
but it does not beat the strongest simple baseline, and on held-out mechanisms it still scores below leaving the
settings alone. Its recurrent memory failed the mandatory memory test. Later stages depend on memory and adaptation, so
they were not run.

## Audit of earlier versions

No CAM 5.0 or CAM 6.0 implementation or design exists in any commit, ref, reflog or commit message that was searched,
including the tip of `feat/superlearner-4`. The actual lineage is CAM v1, CAM v2 / 2.1 / 2.2, CAM hybrid, pure-CAM
Superlearner, CAM 2.0, Superlearner 4.0. The closest unlabelled work is the continual-learning amendment on
`feat/superlearner-4` (from 2026-11-22). Uncommitted files on the old GPU were checked by file name only, because the
backup transfer did not finish.

Hazards found in earlier code and avoided here:

* The Superlearner hub returns held-out outcomes to the learner (`world/core.py:259-278`).
* The continual reservoir almost always replaces old entries once full (`algo/continual.py:210-216`). CAM 6.1 uses a
  true uniform reservoir, covered by a test.
* `goal_net_cents` encodes the private reference, so CAM 6.1 does not read it.

## What exists

| Part | State |
|---|---|
| Structured observation encoder, query compression | Implemented, trained |
| GRU + 32×32-style categorical RSSM, prior / posterior, KL balancing | Implemented, trained (16×16 at the scale used) |
| Learned slot memory | Implemented; **no benefit shown** |
| Reward, continuation, observation, latent, multi-horizon and action-effect prediction | Implemented, trained |
| Duration-aware λ-returns, V-trace, two-hot symlog critics | Implemented, hand-checked |
| Permutation-equivariant candidate policy | Implemented, trained |
| Executable program language with validation | Implemented. It is a bounded lever language, not arbitrary code |
| Trainable program generator (AWR) | Implemented, passes a toy test |
| One-step imagination with calibration gate | Implemented. Multi-step imagination is not |
| Crucible `Learner` adapter, checkpoints | Implemented |
| Qwen text encoding and proposals | **Not connected** |
| Durable external records | **Not implemented** |
| Explicit meta-RL, plasticity diagnostics, continual protocol | **Not implemented** |

All results use the `small` configuration (about 3M parameters). The `medium` configuration was built but not trained.

Two inputs are hand-designed priors copied from Crucible's baselines, not learned: per-lever intervention slopes, and a
starting preference for small changes. Any result below includes them.

## Decisive experiment

Six training families, 240,000 environment steps, three seeds, frozen evaluation with fresh context, 8 tasks per family.
A score of 0 means leaving the defaults alone and 1 means matching the reference policy. Intervals are 95% bootstrap
over tasks with seeds averaged, so they understate seed-to-seed variation.

| Condition | Familiar, trained | Held-out, untrained | Held-out, trained | Held-out gain |
|---|---:|---:|---:|---:|
| CAM 6.1 full | +0.033 | −0.308 | **−0.066** | +0.242 [+0.166, +0.313] |
| CAM 6.1 without planner | +0.018 | −0.308 | −0.116 | +0.192 [+0.120, +0.262] |
| CAM 6.1 without world-model loss | −0.226 | −0.308 | −0.193 | +0.115 [+0.049, +0.181] |
| PPO-GRU | +0.023 | −0.259 | −0.139 | +0.119 [+0.023, +0.214] |
| Model-based baseline | +0.243 | +0.183 | **+0.170** | −0.013 [−0.051, +0.026] |

Paired on held-out tasks:

* CAM 6.1 full minus model-based: **−0.236 [−0.339, −0.141]**
* CAM 6.1 full minus PPO: +0.073 [−0.037, +0.186], not distinguishable
* CAM 6.1 full minus its no-world-model ablation: +0.127 [+0.036, +0.215]

How to read these:

* Most of CAM's gain is learning to stop making harmful random moves. It starts at −0.31 and ends below 0, so the gain is
  **not** evidence of a newly acquired capability.
* The world-model loss helped relative to CAM's own ablation. That ablation also changes the gradient routing, so the
  effect is not isolated to prediction alone.
* The model-based baseline's score comes from its within-episode regression. Its training added nothing measurable.
* CAM used about 10 times PPO's training compute (2,830 s GPU against 276 s CPU, seed 0).
* The familiar-family seeds were also used while debugging CAM, so that column is optimistic. The held-out families were
  not used until this experiment. Crucible's sealed test families were not touched.

## Mandatory tests

The automated suite has 37 tests: 36 pass and 1 fails.

| # | Test | Result |
|---|---|---|
| 1 | Tensor integrity | Pass |
| 2 | Gradients reach trainable parameters | Pass |
| 3 | Posterior versus prior, no future leakage | Pass |
| 4 | Held-out prediction improves | Pass |
| 5 | Neural memory beats no-memory control | **Fail** |
| 6 | Policy moves toward reward | Pass |
| 7 | Open-ended generation | Pass on a toy task; no frozen-LLM comparison |
| 8 | Duration-aware returns | Pass |
| 9 | Imagination integrity | Pass |
| 10 | Real outcome improvement on held-out tasks | Improved over untrained, **still below holding the defaults** |
| 11 | Parametric internalization | Weights improve from fresh context, same caveat as 10 |
| 12 | Cold mechanism acquisition | **Not established** |
| 13 | Continual learning | Not run |
| 14 | Meta-RL adaptation | Not run |
| 15 | Proposal bottleneck | Not run |
| 16 | Compute-normalized efficiency | Costs recorded; CAM is less efficient than both baselines |

Memory task, three seeds, final score (0 is chance, 0.375 is perfect):

| Configuration | Mean | Seeds |
|---|---:|---|
| Recurrent + slots | −0.037 | −0.030, −0.059, −0.021 |
| Recurrent only | −0.033 | −0.053, −0.035, −0.010 |
| No memory | −0.030 | −0.027, −0.059, −0.003 |

An earlier version of this test was invalid: the visible lever let a memoryless agent store the cue. That was fixed
without changing the pass threshold.

## Experiments

| Experiment | State |
|---|---|
| A. Small unknown mechanisms | Partial, through the decisive run |
| B. Predictive world learning | Partial; no separate multi-step imagination arm |
| C. Memory | Toy task only, negative |
| D. Frozen Qwen versus trainable proposals | Not run |
| E. Meta-RL | Not run |
| F. Continual learning | Not run |
| G. Open-ended Crucible objectives | Not run |
| H. Structural transfer | Partial, four held-out families, negative against model-based |
| I. Scaling | Not run |

## Next step

Fix within-episode adaptation before anything else. The clearest evidence: simple regression inside one episode explains
61–97% of reward variation across the six training families, while CAM's held-out action-effect R² reached at most 0.155
in development. The memory task must pass before meta-RL, continual learning or scaling results would mean anything.

## Reproduce

```bash
cd /workspace/agentrs && source .venv/bin/activate
python -m pytest superlearner/tests/unified --rootdir=superlearner
python -m superlearner.unified.experiments.decisive run
python -m superlearner.unified.experiments.decisive report
```
