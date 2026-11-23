# Continual learning in the weights

This document covers the amendment "continual learning must change the core neural weights": what in CAM now
learns for life, how that is tested, and what the measurements say. Every table is generated from the JSON files
under `docs/experiments/` (`python docs/experiments/render.py --fill docs/CONTINUAL.md`); the lab notes with the
calibration runs, superseded attempts and failures are in `docs/experiments/NOTES.md`.

## 1. Summary

<!-- summary -->
<!-- /summary -->

## 2. What learns, and what does not

CAM's decisions are made by a neural controller that reads the situation as *items* (goal, history, documents,
tools, argument options), each carrying a text embedding, hashed symbolic tokens and numbers. Until now the
controller was trained in one warm-start-plus-RL run, and the text embeddings were frozen vectors from a language
model. Now:

| part | parameters | role | learns for life? |
|---|---|---|---|
| controller: input embeddings | 4,806,784 | turns items into vectors (text projection, token table, numeric and type embeddings) | yes |
| controller: encoder | 10,647,552 | six transformer layers over the items: the representation of the situation | yes |
| controller: recurrent state | 2,217,600 | fast and slow GRU: memory within an episode (reset to zero at every episode start) | yes |
| controller: policy heads | 1,772,162 | which action (strategic choice among every tool, wait, draft and tool-making operation) and which arguments (tactical choice) | yes |
| controller: critic | 223,493 | value ensemble and success predictor | yes |
| world model | 1,057,166 | predicts, for a state and a proposed action: accepted or not, failure class, reward, the next state's encoding, the return | yes, online |
| language backbone (Qwen3-0.6B, the 25 blocks that feed the read-out) | 548,856,064 | embeds every text the controller reads | **Architecture A: frozen. Architecture B: adapters (1.0M), the top four blocks (62.9M) or everything (548.9M)** |
| generator (Qwen3-14B) | 14.8 billion | writes code and free text when an action needs it | **no - frozen** |

*Architecture A* is the first six rows on a frozen backbone: 20.7M parameters that take every consequential
decision and are all updated from experience. *Architecture B* makes the language backbone itself plastic: its
weights receive the gradients of the policy, critic and world-model losses through the embedding table
(`cam/backbone.py`; a gradient cache makes the full 549M-parameter update fit in memory and gives exact gradients).
LoRA adapters are run as the parameter-efficient baseline, not as the answer.

**What stays frozen, and why.** The 14B generator. Its weights take 28 GB at 16-bit precision; a full-parameter
update needs weights, gradients and two optimiser moments at 32-bit precision, about 240 GB, on a 46 GB card.
Updating a few of its top blocks would fit, but those blocks produce the generator's *text*, and the only training
signal this system has for text is a few hundred written values per run, each judged once by the world - far too
little to move 350M parameters per block without damaging what the model already writes. So the backbone that is
trained is the one whose output the controller consumes at every decision (text representations), at a size
(0.6B) where full-parameter training is real rather than nominal. Nothing in the results below is a statement
about the 14B model: **the generator has not learned anything; when this document says the backbone learned, it
means Qwen3-0.6B.** The eight task arms used here need no generated content, so the generator is not even loaded.
The top three blocks of Qwen3-0.6B are also untouched: the embedding is read below them, so no gradient reaches
them.

## 3. The lifelong loop

`superlearner lifetime --config configs/life/<condition>.yaml` (`continual/lifetime.py`) runs a sequence of
*phases*. A phase brings task arms the learner has not met and possibly a kind of world it has not been in:

```
experience  ->  training signal  ->  gradient  ->  parameter update  ->  changed behaviour  ->  more experience
```

1. **Observe.** The learner is shown a few verified demonstrations of the new arms (four per arm) and takes
   supervised optimiser steps on them - with rehearsal, half of every minibatch comes from earlier arms.
2. **Practice.** It acts in fresh worlds of the phase. Every iteration updates the policy and critic (recurrent PPO
   with a KL trust region), replays its own verified successes, and trains the world model on every new
   transition. One episode in eight is exploratory (sampled at a high temperature, or uniformly random); those feed
   the world model and the reservoir of successes but never the policy-gradient term.
3. **Close.** The update rule consolidates; then every phase's held-out task list is evaluated with the current
   weights (one row of the retention matrix), together with the world model's accuracy on held-out transitions of
   every phase so far, plasticity diagnostics, and the size of the weight change per parameter group.

The update rule is `algo/continual.py` (`ContinualPPO`). Its parts:

| mechanism | what it does | setting |
|---|---|---|
| online world model | `joint_loss` on every policy step (a share `wm.repr` of its gradient shapes the controller's state), then `wm_fit`: the controller re-encodes the experience buffer and the model alone takes 300 steps on those transitions | `algo.wm` |
| rehearsal | a bounded reservoir of verified successes **per task arm**, kept across phases; replay draws arms uniformly | `algo.rehearsal` |
| replay consolidation | at the end of a phase, optimiser steps on the reservoir alone, all arms equally | `algo.consolidate_steps` |
| anchors | quadratic pull towards the weights at the end of the last phase: uniform (L2-SP), importance-weighted (EWC, diagonal empirical Fisher accumulated over phases), or only the most important weights | `algo.anchor` |
| plasticity | pull towards the initial weights (L2-init), shrink-and-perturb at phase boundaries, per-group learning-rate multipliers (a group at 0 is frozen) | `algo.l2init`, `algo.shrink_perturb`, `algo.group_lr` |
| backbone | the language backbone's trainable parameters are one more optimiser group | `backbone.mode`, `algo.backbone_lr` |

**Persistence.** One checkpoint file, replaced atomically after every step, holds the controller, the world model,
both optimisers' moments, the reservoirs and the experience buffer, anchors and importance estimates, episodic
memory, every counter and random-number state, and the rows measured so far; a trainable backbone's changed
weights sit next to it. A lifetime that is killed continues at the next step. The test
(`tests/test_continual.py`, `tests/restart_check.py`) rebuilds a learner from the file alone and checks that its
state equals the original's in every part and that one more update on the same episodes gives bit-identical
weights in both. (Two whole lifetimes are *not* compared bit by bit: the simulator puts a wall-clock-dependent value
into some observations, so two identical uninterrupted runs occasionally diverge by themselves; see the notes.)

**Two memories.** Parametric memory is everything in the table of section 2. Non-parametric memory is the agent's
episodic store (what worked for which goal: tools and literal arguments), its tool library and its within-episode
knowledge of entities; it can be switched on (`lifetime.memory`) and is what the "frozen core plus written memory"
condition learns with. Demonstrations and own successes kept for rehearsal are training data, not memory the agent
reads when it decides: at decision time the policy sees only the current episode.

## 4. Parametric Internalization Test

`superlearner pit` (`continual/internalize.py`). One seed, one unfamiliar family, one route of learning:

1. A learner is trained on five familiar families (found a company, fund it, list a deployed service, deploy coded
   software, operate a listed service). Its weights are exported: W_before.
2. Starting from that learner's full state, it meets ONE family it has never been shown or trained on and learns by
   the route's means.
3. The change of every parameter group is recorded.
4. The weights are exported to a file that holds tensors and nothing else: W_after.
5. A **new process** builds a new learner from the configuration, loads W_after, and acts greedily on held-out
   tasks. It starts with no episodic memory, no tools, no replay buffer, no cached embeddings and no optimiser
   (the test asserts all five are empty); the recurrent state is zero at every episode start; worlds, seeds and
   identifiers are from the `select` split, which training never touches.
6. The same tasks under W_before, evaluated the same way, give the paired comparison: same network, same tasks,
   only the weights differ.

Unfamiliar families: **launch** (`service` from a deployed but unlisted service - a composition of known parts:
list it, then operate it until it has served its requests) and **repair** (a listed service that is down or badly
overpriced must be diagnosed and fixed - a mechanism the learner has not met). Held-out lists: the family in the
kinds of world it practised in, in two kinds of world it never saw, related families that were never trained, and
the familiar families (retention).

Routes: *practice* (acting only), *explore_practice* (half of the episodes exploratory, a third of those choosing
actions the online world model is unsure about), *observe* (four demonstrations, no acting), *observe_practice*.
Controls: *frozen_memory* (weights frozen, the same demonstrations and practice written to episodic memory only),
*memory_observe_practice* (weights and memory together, evaluated with the memory and - in the fresh process -
without it), and W_after run under a different actor name.

<!-- table:pit -->
<!-- /table -->

## 5. Lifetimes: six phases, eight conditions

`configs/life/base.yaml`. Phases (four demonstrations per new arm, four practice iterations of 64 episodes):

| phase | arms | kinds of world |
|---|---|---|
| start | found a company; list a deployed service | tiny, bazaar |
| capital | fund the company; deploy coded software | tiny, bazaar |
| operate | operate a listed service; repair a failing one | cheap compute, closed markets |
| launch | take a deployed service to its request target | expensive compute, open distribution |
| build | take coded software to its request target | bazaar; cheap compute, closed markets |
| shift | the four arms of *start* and *capital*, **in the two kinds of world they were not learned in, with no demonstrations** | expensive compute; cheap compute |

After every phase all six held-out lists are evaluated (12 tasks per arm, `select` split), greedy and - for phases
seen so far - sampled; at the end also in two kinds of world no phase trains in.

Conditions (the amendment's five, with the fourth split in two and three amounts of backbone plasticity):

| | condition | weights | episodic memory | protection |
|---|---|---|---|---|
| 1 | frozen core + written memory | trained in the first phase, then frozen | on | - (later phases are written to memory only) |
| 2 | trainable core, no protection | trained in every phase | off | none: each phase uses only its own data |
| 3 | trainable core + external memory | as 2 | on | none |
| 4a | trainable core + rehearsal | as 2 | off | per-arm reservoirs replayed in every update and observation |
| 4b | + consolidation | as 2 | off | 4a and 64 replay-only steps at the end of every phase |
| 5 | broader backbone updates | 4b, and the language backbone: adapters / top four blocks / everything | off | as 4b |

The protection used in 4a/4b was fixed before these runs from a short calibration (notes): rehearsal kept earlier
phases, importance-weighted anchors alone did not at any coefficient from 2e4 to 2e6, and anchors on top of
rehearsal slowed new learning.

<!-- table:life -->
<!-- /table -->

### Independent check on untouched tasks

The protection in conditions 4a/4b was chosen from a short calibration, and the calibration, the lists above and the
meta-search all draw from the same seed range of the `select` split, so the numbers above are not free of selection.
`superlearner continual-verify` therefore evaluates every run's final weights on tasks from a seed range none of them
used, fixed before any outcome was seen, in a new process given the weights file alone.

<!-- table:life_verify -->
<!-- /table -->

### Retention matrices

<!-- table:life_matrix -->
<!-- /table -->

### Backbone plasticity (Architecture A against B)

<!-- table:life_backbone -->
<!-- /table -->

### Plasticity, representation and the world model over a lifetime

<!-- table:life_plasticity -->
<!-- /table -->

## 6. The Meta-Researcher on weight-update rules

`superlearner meta-search --config configs/meta_continual.yaml --full-feedback` (`meta/`, mode `lifetime`).
A candidate is a version of `algo/continual.py`. The search space (`meta/proposals.py:ContinualSpace`) covers which
parameter groups learn and how fast, how many steps an observation gets and at which step size, whether and how
much is rehearsed, which arms replay favours, replay consolidation, the kind, strength and selectivity of anchors,
L2-init and shrink-and-perturb, how strongly the world model shapes the shared representation and how much it is
fitted, and the amount of replay and the RL step size - nineteen genes. The first pool is thirteen single-idea
variants (the classic continual-learning methods, one at a time); later pools mutate the champion or the best rule
screened so far.

Every candidate runs the same short lifetime inside the confinement (uid nobody, Landlock, seccomp, no GPU): three
phases of two new arms each, with the same seeds and the same number of demonstrations and episodes per phase. The
held-out tasks live on a trusted env hub; the learner starts them by index and never sees a seed or a score. The
measures are computed by trusted code from the hub's own records, and a run counts only if it evaluated every task
exactly once after every phase. Primary measure: the mean, over phases, of held-out success on the phases seen so
far - high only if new arms are acquired quickly and earlier ones are kept. Candidates whose update is identical to
the champion's on fixed data (now including two short phases, so that observation and consolidation are exercised)
are rejected before any experience is spent.

<!-- table:meta_continual -->
<!-- /table -->

## 7. What this shows and what it does not

<!-- conclusions -->
<!-- /conclusions -->

## 8. Reproduce

```bash
cd superlearner && export PYTHONPATH=src
python -m superlearner jobs configs/queue_continual.yaml --log-dir runs/queue_continual     # lifetimes + PIT (GPU)
python -m superlearner meta-search --config configs/meta_continual.yaml --full-feedback --run-dir runs/meta/continual1
python -m superlearner pit report --config configs/pit.yaml
python -m superlearner continual-report
python docs/experiments/render.py --fill docs/CONTINUAL.md
python -m pytest -q tests/test_continual.py                                                 # CPU, about two minutes
```

A single lifetime: `python -m superlearner lifetime --config configs/life/consolidate.yaml --set seed=0`; it
continues where it stopped if run again. A CPU-only check of everything: `configs/life_smoke.yaml`,
`configs/pit_smoke.yaml`, `configs/meta_continual_smoke.yaml`.
