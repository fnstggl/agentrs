# Continual learning in the weights

This document covers the amendment "continual learning must change the core neural weights": what in CAM now
learns for life, how that is tested, and what the measurements say. Every table is generated from the JSON files
under `docs/experiments/` (`python docs/experiments/render.py --fill docs/CONTINUAL.md`); the lab notes with the
calibration runs, superseded attempts and failures are in `docs/experiments/NOTES.md`.

## 1. Summary

**CAM now learns for life in its weights, and that learning is what carries its skills.** Across six phases of new
tasks and kinds of world, gradient updates to the 19.7M-parameter controller and an online 1.06M-parameter world model
produce skills that survive a process restart and work on tasks nothing has looked at, from a file of weights alone.
Three parts of the amendment did **not** come out as hoped: a broader trainable backbone is not established as better,
written memory hurt, and the Meta-Researcher did not find a better way to update CAM's own weights.

The numbers below are from the **independent check**: 112 tasks from a seed range no calibration, lifetime list or search
ever used, in a new process given the weights file only (success in kinds of world trained in / never trained in).

| Finding | Evidence |
|---|---|
| **Learning in the weights beats written memory** | trainable core 66% / 65%; frozen core + written memory 25% / 25% (+40.8 points [+33.0, +48.5], three seeds) |
| **Rehearsal prevents forgetting** | with rehearsal 92% / 94%; unprotected 66% / 65% (+28.1 [+16.5, +38.9], two seeds). On the lifetimes' own lists forgetting falls from 31% to 0% |
| **A new skill is internalized in the weights** | `build`, 0% before: **100%** after four demonstrations, on all three seeds, also in worlds never trained in. Frozen weights with the same experience written to memory: 0% |
| **Learning from its own practice is real but unreliable** | `build`, practice only: 0% → 50% (per seed 50, 100, 0); with exploration 75% (100, 94, 31). `repair`, practice only: no gain (19% → 17%). With demonstrations `repair` reaches 77% from 21% |
| **Written memory hurt when trained with** | trainable core + memory 40% / 38%, which is 25.4 points *below* the same core without it [-41.5, -10.3] |
| **A broader backbone is suggestive, not established** | over the same rule with a frozen backbone: full 549M parameters +5.4 points [+0.0, +11.6] (two seeds); LoRA +8.0 and top four blocks +8.9 (one seed each, so those intervals cover tasks, not seeds). Nearly all of it is one arm (`repair`). Updates cost four to six times as long |
| **No loss of plasticity in six phases** | no dormant units, and a stable ability to fit new targets. Too short a lifetime to say more |
| **The Meta-Researcher found no better rule for CAM** | 71 candidate rules, 125 confined lifetimes, two promoted. The first was beaten by a plain control. The second gave +0.56 on the small search learner but on the real learner: no gain at full training steps (-2.7 [-11.6, +5.8]) and **harm at half steps (-21.9 [-32.6, -10.7])** |

**The frozen 14B model has learned nothing.** It is not even loaded here. "Backbone" means Qwen3-0.6B, which embeds the
texts the controller reads.

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

Unfamiliar families: **build** (`service` from coded software: deploy it, list it and operate it until it has served
its requests; 0% before learning) and **repair** (a listed service that is down or badly overpriced must be diagnosed
and fixed, a mechanism the learner has not met). A first choice, `launch` (`service` from a deployed service), turned
out to be solved at 100% before any learning, by composing two familiar skills, so it cannot show acquisition; its one
route is kept as a record. Held-out lists: the family in the kinds of world it practised in, in two kinds of world it
never saw, related families that were never trained, and the familiar families (retention).

Routes: *practice* (acting only), *explore_practice* (half of the episodes exploratory, a third of those choosing
actions the online world model is unsure about), *observe* (four demonstrations, no acting), *observe_practice*.
Controls: *frozen_memory* (weights frozen, the same demonstrations and practice written to episodic memory only),
*memory_observe_practice* (weights and memory together, evaluated with the memory and - in the fresh process -
without it), and W_after run under a different actor name. One more route, *observe_practice_backbone*, repeats observation and practice with
the whole language backbone trainable (Architecture B).

<!-- table:pit -->
### Unfamiliar family `build` (service/0/coded)

Held-out success with the weights before and after, both evaluated in a fresh process from the weights file alone (greedy; percentage points; 95% interval over seeds and tasks).

| route | seeds | weights changed | unfamiliar family: before -> after | difference | tasks gained / lost | in unseen kinds of world | related, never trained | familiar families (retention) | after, sampled | controller weights moved (relative) | decisions used |
|---|---|---|---|---|---|---|---|---|---|---|---|
| practice only | 3 | yes | **0% -> 50%** | +50.0 [+0.0, +100.0] | 24 / 0 | 0% -> 47% | 29% -> 29% | 100% -> 100% | 69% | 0.002 | 5,913 |
| practice, half exploratory | 3 | yes | **0% -> 75%** | +75.0 [+35.4, +100.0] | 36 / 0 | 0% -> 69% | 29% -> 62% | 100% -> 100% | 85% | 0.002 | 5,151 |
| observation only (4 demonstrations) | 3 | yes | **0% -> 100%** | +100.0 [+100.0, +100.0] | 48 / 0 | 0% -> 100% | 29% -> 88% | 100% -> 100% | 100% | 0.045 | 0 |
| observation + practice | 3 | yes | **0% -> 100%** | +100.0 [+100.0, +100.0] | 48 / 0 | 0% -> 100% | 29% -> 75% | 100% -> 100% | 100% | 0.044 | 2,150 |
| observation + practice, backbone fully trainable | 2 | yes | **0% -> 91%** | +90.6 [+71.9, +100.0] | 29 / 0 | 0% -> 100% | 25% -> 50% | 100% -> 100% | 100% | 0.043 | 2,300 |
| control: weights frozen, memory written | 2 | no | **0% -> 0%** | +0.0 [+0.0, +0.0] | 0 / 0 | 0% -> 0% | 0% -> 0% | 95% -> 95% | 3% | 0.000 | 7,409 |
| observation + practice, memory on | 2 | yes | **0% -> 50%** | +50.0 [+0.0, +100.0] | 16 / 0 | 0% -> 50% | 0% -> 38% | 95% -> 85% | 59% | 0.040 | 6,204 |

* practice only, the agent registered under another name: 52% (same weights, same tasks: 50%).
* practice, half exploratory, the agent registered under another name: 67% (same weights, same tasks: 75%).
* observation only (4 demonstrations), the agent registered under another name: 92% (same weights, same tasks: 100%).
* observation + practice, the agent registered under another name: 67% (same weights, same tasks: 100%).
* observation + practice, backbone fully trainable, the agent registered under another name: 69% (same weights, same tasks: 91%).
* control: weights frozen, memory written: with the memory it wrote, 0% -> 0% (+0.0 [+0.0, +0.0]); weights alone, 0% -> 0%.
* observation + practice, memory on: with the memory it wrote, 0% -> 100% (+100.0 [+100.0, +100.0]); weights alone, 0% -> 50%.
* observation + practice, memory on, the agent registered under another name: 50% (same weights, same tasks: 50%).

State of the fresh process before its first episode: {'memory_items': 0, 'tools': 0, 'embedding_rows': 0, 'replay_buffer': 0, 'texts': 0}.

### Unfamiliar family `repair` (repair/0/scratch)

Held-out success with the weights before and after, both evaluated in a fresh process from the weights file alone (greedy; percentage points; 95% interval over seeds and tasks).

| route | seeds | weights changed | unfamiliar family: before -> after | difference | tasks gained / lost | in unseen kinds of world | related, never trained | familiar families (retention) | after, sampled | controller weights moved (relative) | decisions used |
|---|---|---|---|---|---|---|---|---|---|---|---|
| practice only | 3 | yes | **19% -> 17%** | -2.1 [-12.5, +10.4] | 3 / 4 | 0% -> 0% | 46% -> 79% | 100% -> 100% | 29% | 0.002 | 3,687 |
| practice, half exploratory | 3 | yes | **21% -> 19%** | -2.1 [-14.6, +10.4] | 3 / 4 | 0% -> 0% | 54% -> 100% | 100% -> 100% | 25% | 0.003 | 4,022 |
| observation only (4 demonstrations) | 3 | yes | **21% -> 67%** | +45.8 [+27.1, +62.5] | 26 / 4 | 0% -> 53% | 54% -> 75% | 100% -> 97% | 56% | 0.049 | 0 |
| observation + practice | 3 | yes | **21% -> 77%** | +56.2 [+41.7, +70.8] | 27 / 0 | 0% -> 50% | 54% -> 88% | 100% -> 100% | 65% | 0.048 | 2,956 |
| observation + practice, backbone fully trainable | 2 | yes | **16% -> 78%** | +62.5 [+43.8, +81.2] | 20 / 0 | 0% -> 50% | 50% -> 100% | 100% -> 100% | 69% | 0.042 | 3,285 |
| control: weights frozen, memory written | 2 | no | **16% -> 19%** | +3.1 [-9.4, +15.6] | 2 / 1 | 0% -> 0% | 0% -> 0% | 95% -> 95% | 16% | 0.000 | 6,287 |
| observation + practice, memory on | 2 | yes | **12% -> 38%** | +25.0 [+0.0, +56.2] | 8 / 0 | 0% -> 25% | 0% -> 19% | 95% -> 75% | 44% | 0.046 | 4,437 |

* practice only, the agent registered under another name: 27% (same weights, same tasks: 17%).
* practice, half exploratory, the agent registered under another name: 17% (same weights, same tasks: 19%).
* observation only (4 demonstrations), the agent registered under another name: 71% (same weights, same tasks: 67%).
* observation + practice, the agent registered under another name: 69% (same weights, same tasks: 77%).
* observation + practice, backbone fully trainable, the agent registered under another name: 72% (same weights, same tasks: 78%).
* control: weights frozen, memory written: with the memory it wrote, 16% -> 9% (-6.2 [-21.9, +9.4]); weights alone, 16% -> 19%.
* observation + practice, memory on: with the memory it wrote, 12% -> 47% (+34.4 [+3.1, +68.8]); weights alone, 12% -> 38%.
* observation + practice, memory on, the agent registered under another name: 34% (same weights, same tasks: 38%).

State of the fresh process before its first episode: {'memory_items': 0, 'tools': 0, 'embedding_rows': 0, 'replay_buffer': 0, 'texts': 0}.

### Unfamiliar family `launch` (service/0/deployed)

Held-out success with the weights before and after, both evaluated in a fresh process from the weights file alone (greedy; percentage points; 95% interval over seeds and tasks).

| route | seeds | weights changed | unfamiliar family: before -> after | difference | tasks gained / lost | in unseen kinds of world | related, never trained | familiar families (retention) | after, sampled | controller weights moved (relative) | decisions used |
|---|---|---|---|---|---|---|---|---|---|---|---|
| practice only | 1 | yes | **100% -> 96%** | -4.2 [-12.5, +0.0] | 0 / 1 | 100% -> 94% | 50% -> 50% | 100% -> 100% | 88% | 0.003 | 4,519 |

* practice only, the agent registered under another name: 100% (same weights, same tasks: 96%).

State of the fresh process before its first episode: {'memory_items': 0, 'tools': 0, 'embedding_rows': 0, 'replay_buffer': 0, 'texts': 0}.
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

Three further conditions test the rules the Meta-Researcher promoted on this real learner: the first promoted rule (96
consolidation steps), the equal-compute promoted rule at the full supervised steps, and the same rule against
rehearsal at half the supervised steps, where the real learner is not at its ceiling.

The protection used in 4a/4b was fixed before these runs from a short calibration (notes): rehearsal kept earlier
phases, importance-weighted anchors alone did not at any coefficient from 2e4 to 2e6, and anchors on top of
rehearsal slowed new learning.

<!-- table:life -->
Held-out success after the whole lifetime (greedy actions; per-seed values in brackets). *Final, six lists*: mean over the six held-out lists with the final weights. *Taught five*: the five phases that came with demonstrations. *Right after learning*: each phase's own list at the end of that phase. *Forgetting*: for every earlier phase, its best success at any time minus its success at the end. *Unseen worlds*: the same arms in two kinds of world no phase trained in.

| condition | seeds | final, six lists | taught five | sampled actions | right after learning | forgetting | unseen worlds | decisions |
|---|---|---|---|---|---|---|---|---|
| 1 frozen core + written memory | 3 | **21%** (25, 18, 19) | 17% | 25% | 21% | 0% (0, 0, 0) | 20% | 53,521 |
| 2 trainable core, no protection | 3 | **63%** (62, 64, 64) | 66% | 63% | 88% | 31% (30, 33, 30) | 70% | 16,748 |
| 3 trainable core + external memory | 2 | **39%** (28, 50) | 41% | 32% | 50% | 14% (15, 12) | 39% | 26,624 |
| 4a trainable core + rehearsal | 2 | **93%** (97, 89) | 92% | 86% | 93% | 0% (0, 0) | 96% | 17,934 |
| 4b trainable core + rehearsal + consolidation | 3 | **88%** (86, 90, 89) | 86% | 87% | 88% | 2% (3, 0, 2) | 92% | 15,819 |
| first promoted rule (96 consolidation steps) | 2 | **90%** (89, 90) | 88% | 84% | 86% | 2% (4, 0) | 92% | 16,048 |
| equal-compute promoted rule | 2 | **89%** (89, 89) | 87% | 83% | 92% | 5% (10, 0) | 94% | 17,103 |
| half the steps: rehearsal | 2 | **76%** (76, 76) | 77% | 68% | 77% | 1% (0, 2) | 81% | 27,713 |
| half the steps: equal-compute promoted rule | 2 | **54%** (50, 58) | 50% | 50% | 87% | 39% (43, 35) | 58% | 25,981 |
| 5 (LoRA) 4b + backbone adapters | 1 | **94%** (94) | 93% | 95% | 99% | 5% (5) | 100% | 16,507 |
| 5 (top 4 blocks) 4b + backbone top blocks | 1 | **96%** (96) | 95% | 93% | 86% | 3% (3) | 100% | 18,392 |
| 5 (full) 4b + whole backbone | 2 | **91%** (86, 96) | 89% | 83% | 85% | 2% (5, 0) | 96% | 18,832 |

Final success by phase:

| condition | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| 1 frozen core + written memory | 83% | 0% | 0% | 0% | 0% | 42% |
| 2 trainable core, no protection | 50% | 50% | 58% | 72% | 100% | 49% |
| 3 trainable core + external memory | 38% | 25% | 50% | 42% | 50% | 29% |
| 4a trainable core + rehearsal | 100% | 100% | 75% | 83% | 100% | 100% |
| 4b trainable core + rehearsal + consolidation | 100% | 100% | 50% | 81% | 100% | 100% |
| first promoted rule (96 consolidation steps) | 100% | 100% | 50% | 88% | 100% | 100% |
| equal-compute promoted rule | 100% | 100% | 50% | 83% | 100% | 100% |
| half the steps: rehearsal | 100% | 50% | 50% | 83% | 100% | 75% |
| half the steps: equal-compute promoted rule | 100% | 50% | 25% | 75% | 0% | 75% |
| 5 (LoRA) 4b + backbone adapters | 100% | 100% | 100% | 67% | 100% | 100% |
| 5 (top 4 blocks) 4b + backbone top blocks | 100% | 100% | 100% | 75% | 100% | 100% |
| 5 (full) 4b + whole backbone | 100% | 100% | 75% | 71% | 100% | 100% |

The *shift* phase (four earlier arms in two kinds of world they were not learned in, practice only, no demonstrations): success on its held-out list just before and after the phase.

| condition | before | after | per seed (before, after) |
|---|---|---|---|
| 1 frozen core + written memory | 42% | 42% | (50, 50), (33, 33), (42, 42) |
| 2 trainable core, no protection | 35% | 49% | (31, 46), (25, 50), (50, 50) |
| 3 trainable core + external memory | 0% | 29% | (0, 8), (0, 50) |
| 4a trainable core + rehearsal | 100% | 100% | (100, 100), (100, 100) |
| 4b trainable core + rehearsal + consolidation | 100% | 100% | (100, 100), (100, 100), (100, 100) |
| first promoted rule (96 consolidation steps) | 100% | 100% | (100, 100), (100, 100) |
| equal-compute promoted rule | 100% | 100% | (100, 100), (100, 100) |
| half the steps: rehearsal | 62% | 75% | (50, 75), (75, 75) |
| half the steps: equal-compute promoted rule | 100% | 75% | (100, 75), (100, 75) |
| 5 (LoRA) 4b + backbone adapters | 94% | 100% | (94, 100) |
| 5 (top 4 blocks) 4b + backbone top blocks | 100% | 100% | (100, 100) |
| 5 (full) 4b + whole backbone | 100% | 100% | (100, 100), (100, 100) |

Conditions with episodic memory, final weights evaluated with the memory and with it switched off:

| condition | with its memory | memory off | memory records | agent-made tools |
|---|---|---|---|---|
| 1 frozen core + written memory | 21% | 25% | 1747 | 11 |
| 3 trainable core + external memory | 39% | 38% | 1747 | 62 |

Differences between conditions on the same held-out tasks (final weights, all six lists; percentage points; 95% interval from a bootstrap over seeds and tasks):

| comparison | seeds | tasks | difference | tasks only the first solves / only the second | unseen worlds |
|---|---|---|---|---|---|
| 2 trainable core, no protection minus 1 frozen core + written memory | 3 | 432 | +29.2 [+20.6, +37.3] | 200 / 74 | +34.0 [+18.1, +50.0] |
| 3 trainable core + external memory minus 2 trainable core, no protection | 2 | 288 | -20.8 [-38.9, -4.9] | 6 / 66 | -25.0 [-43.8, -8.3] |
| 4a trainable core + rehearsal minus 2 trainable core, no protection | 2 | 288 | +37.5 [+29.9, +45.5] | 110 / 2 | +35.4 [+21.9, +47.9] |
| 4b trainable core + rehearsal + consolidation minus 2 trainable core, no protection | 3 | 432 | +33.1 [+28.0, +38.0] | 152 / 9 | +30.6 [+21.5, +39.6] |
| 4b trainable core + rehearsal + consolidation minus 4a trainable core + rehearsal | 2 | 288 | -4.5 [-12.2, +1.7] | 3 / 16 | -4.2 [-12.5, +0.0] |
| first promoted rule (96 consolidation steps) minus 4a trainable core + rehearsal | 2 | 288 | -3.8 [-10.4, +1.7] | 2 / 13 | -4.2 [-12.5, +0.0] |
| equal-compute promoted rule minus 4a trainable core + rehearsal | 2 | 288 | -4.2 [-10.4, +1.0] | 2 / 14 | -2.1 [-12.5, +7.3] |
| half the steps: equal-compute promoted rule minus half the steps: rehearsal | 2 | 288 | -13.2 [-22.9, -3.8] | 51 / 89 | -14.6 [-28.1, -1.0] |
| 4b trainable core + rehearsal + consolidation minus 1 frozen core + written memory | 3 | 432 | +62.3 [+54.4, +69.0] | 269 / 0 | +64.6 [+53.5, +75.0] |
| 5 (LoRA) 4b + backbone adapters minus 4b trainable core + rehearsal + consolidation | 1 | 144 | +8.3 [+3.5, +13.9] | 14 / 2 | +8.3 [+2.1, +16.7] |
| 5 (top 4 blocks) 4b + backbone top blocks minus 4b trainable core + rehearsal + consolidation | 1 | 144 | +9.0 [+3.5, +14.6] | 15 / 2 | +8.3 [+2.1, +16.7] |
| 5 (full) 4b + whole backbone minus 4b trainable core + rehearsal + consolidation | 2 | 288 | +3.5 [-1.7, +9.4] | 16 / 6 | +4.2 [+0.0, +12.5] |
<!-- /table -->

### Independent check on untouched tasks

The protection in conditions 4a/4b was chosen from a short calibration, and the calibration, the lists above and the
meta-search all draw from the same seed range of the `select` split, so the numbers above are not free of selection.
`superlearner continual-verify` therefore evaluates every run's final weights on tasks from a seed range none of them
used, fixed before any outcome was seen, in a new process given the weights file alone.

<!-- table:life_verify -->
Every lifetime's final weights, exported to a tensors-only file and evaluated in a new process with no memory, tools, replay buffer or cached embedding, on 112 tasks from a seed range that neither the calibration, the lifetimes' own lists nor the meta-search ever used (eight arms). Per-seed values in brackets.

| condition | seeds | in kinds of world it trained in | in kinds of world it never trained in | with its episodic memory |
|---|---|---|---|---|
| 1 frozen core + written memory | 3 | **25%** (25, 25, 25) | **25%** (25, 25, 25) | 21% / 20% |
| 2 trainable core, no protection | 3 | **66%** (64, 67, 68) | **65%** (60, 68, 68) | - |
| 3 trainable core + external memory | 2 | **40%** (25, 56) | **38%** (25, 50) | 47% / 44% |
| 4a trainable core + rehearsal | 2 | **92%** (96, 89) | **94%** (100, 88) | - |
| 4b trainable core + rehearsal + consolidation | 3 | **90%** (90, 90, 90) | **88%** (88, 88, 88) | - |
| first promoted rule (96 consolidation steps) | 2 | **89%** (89, 89) | **85%** (85, 85) | - |
| equal-compute promoted rule | 2 | **91%** (90, 92) | **89%** (85, 92) | - |
| half the steps: rehearsal | 2 | **79%** (78, 81) | **78%** (75, 80) | - |
| half the steps: equal-compute promoted rule | 2 | **58%** (50, 65) | **55%** (50, 60) | - |
| 5 (LoRA) 4b + backbone adapters | 1 | **96%** (96) | **100%** (100) | - |
| 5 (top 4 blocks) 4b + backbone top blocks | 1 | **97%** (97) | **100%** (100) | - |
| 5 (full) 4b + whole backbone | 2 | **95%** (93, 97) | **94%** (88, 100) | - |

By arm (weights only, both lists):

| condition | deploy/0/coded | found/0/scratch | fund/0/scratch | list/0/deployed | repair/0/scratch | service/0/coded | service/0/deployed | service/0/listed |
|---|---|---|---|---|---|---|---|---|
| 1 frozen core + written memory | 0% | 100% | 0% | 100% | 0% | 0% | 0% | 0% |
| 2 trainable core, no protection | 100% | 0% | 0% | 100% | 35% | 100% | 90% | 100% |
| 3 trainable core + external memory | 100% | 0% | 0% | 100% | 6% | 50% | 6% | 50% |
| 4a trainable core + rehearsal | 100% | 100% | 100% | 100% | 47% | 100% | 97% | 100% |
| 4b trainable core + rehearsal + consolidation | 100% | 100% | 100% | 100% | 13% | 100% | 98% | 100% |
| first promoted rule (96 consolidation steps) | 100% | 100% | 100% | 95% | 6% | 100% | 95% | 100% |
| equal-compute promoted rule | 100% | 100% | 100% | 100% | 24% | 100% | 95% | 100% |
| half the steps: rehearsal | 100% | 100% | 0% | 100% | 27% | 100% | 100% | 100% |
| half the steps: equal-compute promoted rule | 0% | 100% | 100% | 100% | 6% | 0% | 95% | 50% |
| 5 (LoRA) 4b + backbone adapters | 100% | 100% | 100% | 100% | 83% | 100% | 100% | 100% |
| 5 (top 4 blocks) 4b + backbone top blocks | 100% | 100% | 100% | 100% | 89% | 100% | 100% | 100% |
| 5 (full) 4b + whole backbone | 100% | 100% | 100% | 100% | 56% | 100% | 100% | 100% |

Differences on these same tasks (percentage points; 95% interval from a bootstrap over seeds and tasks):

| comparison | seeds | tasks | difference | tasks only the first solves / only the second |
|---|---|---|---|---|
| 2 trainable core, no protection minus 1 frozen core + written memory | 3 | 336 | +40.8 [+33.0, +48.5] | 179 / 42 |
| 3 trainable core + external memory minus 2 trainable core, no protection | 2 | 224 | -25.4 [-41.5, -10.3] | 1 / 58 |
| 4a trainable core + rehearsal minus 2 trainable core, no protection | 2 | 224 | +28.1 [+16.5, +38.9] | 69 / 6 |
| 4b trainable core + rehearsal + consolidation minus 2 trainable core, no protection | 3 | 336 | +23.5 [+17.6, +29.5] | 91 / 12 |
| 4b trainable core + rehearsal + consolidation minus 4a trainable core + rehearsal | 2 | 224 | -3.6 [-10.7, +2.2] | 3 / 11 |
| first promoted rule (96 consolidation steps) minus 4a trainable core + rehearsal | 2 | 224 | -5.4 [-12.5, +0.4] | 2 / 14 |
| equal-compute promoted rule minus 4a trainable core + rehearsal | 2 | 224 | -2.7 [-11.6, +5.8] | 6 / 12 |
| half the steps: equal-compute promoted rule minus half the steps: rehearsal | 2 | 224 | -21.9 [-32.6, -10.7] | 29 / 78 |
| 4b trainable core + rehearsal + consolidation minus 1 frozen core + written memory | 3 | 336 | +64.3 [+59.2, +69.3] | 216 / 0 |
| 5 (LoRA) 4b + backbone adapters minus 4b trainable core + rehearsal + consolidation | 1 | 112 | +8.0 [+2.7, +14.3] | 10 / 1 |
| 5 (top 4 blocks) 4b + backbone top blocks minus 4b trainable core + rehearsal + consolidation | 1 | 112 | +8.9 [+3.6, +15.2] | 11 / 1 |
| 5 (full) 4b + whole backbone minus 4b trainable core + rehearsal + consolidation | 2 | 224 | +5.4 [+0.0, +11.6] | 13 / 1 |
<!-- /table -->

### Retention matrices

<!-- table:life_matrix -->
**1 frozen core + written memory** (mean of 3 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **83%** | 0% | 0% | 0% | 0% | 42% |
| capital | 83% | **0%** | 0% | 0% | 0% | 42% |
| operate | 83% | 0% | **0%** | 0% | 0% | 42% |
| launch | 83% | 0% | 0% | **0%** | 0% | 42% |
| build | 83% | 0% | 0% | 0% | **0%** | 42% |
| shift | 83% | 0% | 0% | 0% | 0% | **42%** |

**2 trainable core, no protection** (mean of 3 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **100%** | 0% | 0% | 0% | 0% | 50% |
| capital | 50% | **100%** | 0% | 0% | 0% | 75% |
| operate | 0% | 0% | **100%** | 0% | 0% | 0% |
| launch | 50% | 0% | 50% | **81%** | 0% | 25% |
| build | 25% | 50% | 47% | 28% | **100%** | 35% |
| shift | 50% | 50% | 58% | 72% | 100% | **49%** |

**3 trainable core + external memory** (mean of 2 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **88%** | 0% | 0% | 0% | 0% | 46% |
| capital | 50% | **44%** | 0% | 0% | 0% | 47% |
| operate | 0% | 0% | **50%** | 0% | 0% | 0% |
| launch | 12% | 0% | 25% | **42%** | 0% | 8% |
| build | 0% | 0% | 25% | 42% | **50%** | 0% |
| shift | 38% | 25% | 50% | 42% | 50% | **29%** |

**4a trainable core + rehearsal** (mean of 2 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **100%** | 0% | 0% | 0% | 0% | 50% |
| capital | 100% | **100%** | 0% | 0% | 0% | 100% |
| operate | 75% | 100% | **75%** | 38% | 0% | 88% |
| launch | 100% | 75% | 75% | **83%** | 0% | 88% |
| build | 100% | 100% | 75% | 79% | **100%** | 100% |
| shift | 100% | 100% | 75% | 83% | 100% | **100%** |

**4b trainable core + rehearsal + consolidation** (mean of 3 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **100%** | 0% | 0% | 0% | 0% | 50% |
| capital | 100% | **100%** | 0% | 0% | 0% | 100% |
| operate | 83% | 83% | **50%** | 61% | 33% | 83% |
| launch | 100% | 100% | 50% | **78%** | 0% | 100% |
| build | 100% | 100% | 50% | 89% | **100%** | 100% |
| shift | 100% | 100% | 50% | 81% | 100% | **100%** |

**first promoted rule (96 consolidation steps)** (mean of 2 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **100%** | 0% | 0% | 0% | 0% | 50% |
| capital | 100% | **75%** | 0% | 0% | 0% | 88% |
| operate | 69% | 100% | **50%** | 42% | 8% | 90% |
| launch | 100% | 75% | 50% | **92%** | 0% | 88% |
| build | 100% | 100% | 52% | 83% | **100%** | 100% |
| shift | 100% | 100% | 50% | 88% | 100% | **100%** |

**equal-compute promoted rule** (mean of 2 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **100%** | 0% | 0% | 0% | 0% | 50% |
| capital | 75% | **100%** | 0% | 0% | 0% | 88% |
| operate | 50% | 50% | **75%** | 0% | 0% | 50% |
| launch | 100% | 75% | 50% | **79%** | 0% | 88% |
| build | 100% | 100% | 50% | 75% | **100%** | 100% |
| shift | 100% | 100% | 50% | 83% | 100% | **100%** |

**half the steps: rehearsal** (mean of 2 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **100%** | 0% | 0% | 0% | 0% | 50% |
| capital | 100% | **50%** | 0% | 0% | 0% | 75% |
| operate | 50% | 0% | **50%** | 0% | 0% | 25% |
| launch | 100% | 0% | 50% | **88%** | 0% | 50% |
| build | 75% | 50% | 50% | 38% | **100%** | 62% |
| shift | 100% | 50% | 50% | 83% | 100% | **75%** |

**half the steps: equal-compute promoted rule** (mean of 2 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **100%** | 0% | 0% | 0% | 0% | 50% |
| capital | 50% | **100%** | 0% | 0% | 0% | 75% |
| operate | 50% | 50% | **50%** | 0% | 0% | 50% |
| launch | 100% | 50% | 50% | **96%** | 0% | 75% |
| build | 100% | 100% | 50% | 83% | **100%** | 100% |
| shift | 100% | 50% | 25% | 75% | 0% | **75%** |

**5 (LoRA) 4b + backbone adapters** (mean of 1 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **100%** | 0% | 0% | 0% | 0% | 50% |
| capital | 100% | **100%** | 0% | 0% | 0% | 100% |
| operate | 100% | 100% | **100%** | 83% | 100% | 100% |
| launch | 100% | 100% | 100% | **92%** | 0% | 100% |
| build | 100% | 88% | 100% | 83% | **100%** | 94% |
| shift | 100% | 100% | 100% | 67% | 100% | **100%** |

**5 (top 4 blocks) 4b + backbone top blocks** (mean of 1 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **100%** | 0% | 0% | 0% | 0% | 50% |
| capital | 100% | **100%** | 0% | 0% | 0% | 100% |
| operate | 100% | 100% | **50%** | 42% | 0% | 100% |
| launch | 100% | 100% | 50% | **67%** | 0% | 100% |
| build | 100% | 100% | 100% | 92% | **100%** | 100% |
| shift | 100% | 100% | 100% | 75% | 100% | **100%** |

**5 (full) 4b + whole backbone** (mean of 2 seeds; row: after phase, column: held-out list)

| after | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| start | **94%** | 0% | 0% | 0% | 0% | 50% |
| capital | 90% | **62%** | 0% | 0% | 0% | 78% |
| operate | 75% | 100% | **75%** | 42% | 0% | 89% |
| launch | 100% | 100% | 75% | **79%** | 46% | 99% |
| build | 100% | 100% | 75% | 79% | **100%** | 100% |
| shift | 100% | 100% | 75% | 71% | 100% | **100%** |
<!-- /table -->

### Backbone plasticity (Architecture A against B)

<!-- table:life_backbone -->
The same learner and update rule (condition 4b) with four amounts of backbone plasticity.

| backbone | trainable backbone parameters | moved from pretrained (relative L2) | seeds | final, six lists | right after learning | forgetting | unseen worlds | shift before -> after | update time per lifetime | difference to frozen (paired) |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | - | 3 | **88%** (86, 90, 89) | 88% | 2% | 92% | 100% -> 100% | 14 min | reference |
| lora | 1,024,000 | 6.56e-01 | 1 | **94%** (94) | 99% | 5% | 100% | 94% -> 100% | 90 min | +8.3 [+3.5, +13.9] |
| top:4 | 62,923,776 | 1.87e-02 | 1 | **96%** (96) | 86% | 3% | 100% | 100% -> 100% | 62 min | +9.0 [+3.5, +14.6] |
| full | 548,856,064 | 2.42e-02 | 2 | **91%** (86, 96) | 85% | 2% | 96% | 100% -> 100% | 79 min | +3.5 [-1.7, +9.4] |

Swap test: the final controller evaluated on the six held-out lists with its trained backbone, and again with the backbone put back to its pretrained weights (all text embeddings recomputed). What disappears was carried by the backbone's changed weights.

| backbone | seeds | with its trained backbone | with the pretrained backbone | difference (paired by task) | by phase (trained / pretrained) |
|---|---|---|---|---|---|
| lora | 1 | 94% | 33% | +55.6 [+47.2, +63.2] | start 100/50, capital 100/50, operate 100/50, launch 67/0, build 100/0, shift 100/50 |
| top:4 | 1 | 97% | 8% | +90.3 [+85.4, +94.4] | start 100/0, capital 100/0, operate 100/50, launch 83/0, build 100/0, shift 100/0 |
| full | 2 | 92% | 38% | +56.6 [+44.8, +67.4] | start 100/62, capital 100/0, operate 75/50, launch 79/75, build 100/0, shift 100/38 |
<!-- /table -->

### Plasticity, representation and the world model over a lifetime

<!-- table:life_plasticity -->
Plasticity probe: a copy of the controller is trained for 40 steps to choose arbitrary fixed targets on fixed probe decisions; the loss it reaches (lower = easier to train; choosing at random gives 4.98). Measured after every phase.

| condition | start | capital | operate | launch | build | shift | dormant units (max) | effective rank of the state, first -> last |
|---|---|---|---|---|---|---|---|---|
| 1 frozen core + written memory | 2.63 | 2.63 | 2.63 | 2.61 | 2.64 | 2.64 | 0.0% | 4.27 -> 4.27 |
| 2 trainable core, no protection | 2.45 | 2.78 | 2.94 | 3.27 | 2.90 | 3.09 | 0.0% | 4.14 -> 2.89 |
| 3 trainable core + external memory | 2.21 | 3.72 | 3.00 | 3.23 | 2.90 | 3.07 | 0.0% | 4.09 -> 2.89 |
| 4a trainable core + rehearsal | 2.25 | 2.87 | 2.56 | 2.40 | 2.23 | 2.39 | 0.0% | 3.97 -> 4.56 |
| 4b trainable core + rehearsal + consolidation | 2.69 | 2.43 | 2.62 | 2.74 | 2.45 | 2.40 | 0.0% | 4.39 -> 4.84 |
| first promoted rule (96 consolidation steps) | 2.78 | 3.27 | 2.56 | 2.89 | 2.89 | 2.87 | 0.0% | 4.18 -> 4.66 |
| equal-compute promoted rule | 2.77 | 3.58 | 4.03 | 3.92 | 4.00 | 3.91 | 0.0% | 3.49 -> 4.12 |
| half the steps: rehearsal | 2.52 | 2.74 | 2.81 | 2.67 | 2.65 | 2.68 | 0.0% | 4.16 -> 4.17 |
| half the steps: equal-compute promoted rule | 3.43 | 3.14 | 3.47 | 4.06 | 3.79 | 3.70 | 0.0% | 3.56 -> 4.08 |
| 5 (LoRA) 4b + backbone adapters | 2.46 | 2.34 | 2.36 | 2.31 | 2.73 | 2.80 | 0.0% | 4.03 -> 5.06 |
| 5 (top 4 blocks) 4b + backbone top blocks | 2.87 | 2.90 | 3.08 | 3.72 | 2.98 | 2.84 | 0.0% | 4.24 -> 4.73 |
| 5 (full) 4b + whole backbone | 3.28 | 2.99 | 2.89 | 3.12 | 3.55 | 3.30 | 0.0% | 4.03 -> 4.84 |

World model, held-out transitions (episodes of exploratory behaviour that are never trained on; all phases seen so far): share of actions whose acceptance it predicts correctly, against always predicting the majority outcome; and the same for this iteration's NEW transitions before any update has seen them (predict, then train).

| condition | held-out accuracy at the end | majority | first phase's transitions, after phase 1 -> after phase 6 | next-state cosine (mean predictor) | new transitions, first iteration | new transitions, all later iterations (majority) |
|---|---|---|---|---|---|---|
| 1 frozen core + written memory | 88% | 66% | 93% -> 93% | 0.99 (0.93) | 33% | 76% (69%) |
| 2 trainable core, no protection | 92% | 59% | 93% -> 95% | 0.97 (0.93) | 65% | 92% (73%) |
| 3 trainable core + external memory | 92% | 57% | 92% -> 95% | 0.98 (0.92) | 23% | 85% (71%) |
| 4a trainable core + rehearsal | 91% | 60% | 92% -> 93% | 0.89 (0.69) | 62% | 90% (67%) |
| 4b trainable core + rehearsal + consolidation | 91% | 60% | 85% -> 91% | 0.93 (0.83) | 66% | 90% (70%) |
| first promoted rule (96 consolidation steps) | 90% | 63% | 86% -> 91% | 0.90 (0.71) | 64% | 90% (70%) |
| equal-compute promoted rule | 93% | 59% | 94% -> 94% | 0.96 (0.74) | 64% | 90% (70%) |
| half the steps: rehearsal | 91% | 65% | 93% -> 93% | 0.99 (0.90) | 26% | 89% (65%) |
| half the steps: equal-compute promoted rule | 92% | 62% | 92% -> 93% | 0.99 (0.84) | 43% | 90% (71%) |
| 5 (LoRA) 4b + backbone adapters | 90% | 62% | 93% -> 93% | 0.92 (0.78) | 62% | 90% (73%) |
| 5 (top 4 blocks) 4b + backbone top blocks | 91% | 61% | 93% -> 96% | 0.85 (0.72) | 61% | 90% (70%) |
| 5 (full) 4b + whole backbone | 90% | 61% | 86% -> 86% | 0.92 (0.73) | 56% | 89% (72%) |

How far the controller's weights are from their initial values at the end (relative L2 per parameter group), and whether they changed after the first phase:

| condition | input | encoder | recurrent | policy | critic | all | changed after phase 1 | optimiser steps |
|---|---|---|---|---|---|---|---|---|
| 1 frozen core + written memory | 0.03 | 0.07 | 0.09 | 0.59 | 0.22 | 0.07 | NO | 262 |
| 2 trainable core, no protection | 0.06 | 0.16 | 0.32 | 0.88 | 0.32 | 0.14 | yes | 1,584 |
| 3 trainable core + external memory | 0.06 | 0.14 | 0.33 | 0.86 | 0.31 | 0.13 | yes | 1,594 |
| 4a trainable core + rehearsal | 0.06 | 0.15 | 0.35 | 0.81 | 0.29 | 0.13 | yes | 1,608 |
| 4b trainable core + rehearsal + consolidation | 0.06 | 0.18 | 0.35 | 0.82 | 0.31 | 0.14 | yes | 1,984 |
| first promoted rule (96 consolidation steps) | 0.06 | 0.16 | 0.36 | 0.82 | 0.30 | 0.14 | yes | 2,190 |
| equal-compute promoted rule | 0.11 | 0.30 | 0.77 | 1.38 | 0.56 | 0.26 | yes | 1,626 |
| half the steps: rehearsal | 0.05 | 0.11 | 0.22 | 0.59 | 0.22 | 0.09 | yes | 1,306 |
| half the steps: equal-compute promoted rule | 0.09 | 0.24 | 0.61 | 1.09 | 0.47 | 0.20 | yes | 1,322 |
| 5 (LoRA) 4b + backbone adapters | 0.05 | 0.14 | 0.35 | 0.84 | 0.31 | 0.13 | yes | 2,012 |
| 5 (top 4 blocks) 4b + backbone top blocks | 0.05 | 0.15 | 0.37 | 0.86 | 0.30 | 0.13 | yes | 1,990 |
| 5 (full) 4b + whole backbone | 0.05 | 0.14 | 0.32 | 0.77 | 0.28 | 0.12 | yes | 1,977 |

Learning efficiency over the lifetime: training success in the FIRST practice iteration of each phase (right after observing the demonstrations; the last phase has none):

| condition | start | capital | operate | launch | build | shift |
|---|---|---|---|---|---|---|
| 1 frozen core + written memory | 82% | 0% | 0% | 0% | 0% | 40% |
| 2 trainable core, no protection | 100% | 96% | 74% | 82% | 99% | 44% |
| 3 trainable core + external memory | 75% | 42% | 59% | 27% | 1% | 8% |
| 4a trainable core + rehearsal | 100% | 76% | 53% | 62% | 95% | 94% |
| 4b trainable core + rehearsal + consolidation | 100% | 78% | 61% | 79% | 90% | 90% |
| first promoted rule (96 consolidation steps) | 100% | 74% | 62% | 54% | 88% | 97% |
| equal-compute promoted rule | 100% | 68% | 61% | 63% | 67% | 80% |
| half the steps: rehearsal | 76% | 36% | 42% | 33% | 29% | 67% |
| half the steps: equal-compute promoted rule | 88% | 52% | 62% | 46% | 13% | 65% |
| 5 (LoRA) 4b + backbone adapters | 100% | 74% | 71% | 71% | 93% | 93% |
| 5 (top 4 blocks) 4b + backbone top blocks | 100% | 72% | 57% | 77% | 98% | 94% |
| 5 (full) 4b + whole backbone | 96% | 41% | 65% | 54% | 96% | 93% |
<!-- /table -->

## 6. The Meta-Researcher on weight-update rules

`superlearner meta-search --config configs/meta_continual.yaml --full-feedback` (`meta/`, mode `lifetime`).
A candidate is a version of `algo/continual.py`. The search space (`meta/proposals.py:ContinualSpace`) covers which
parameter groups learn and how fast, how many steps an observation gets and at which step size, whether and how
much is rehearsed, which arms replay favours, replay consolidation, the kind, strength and selectivity of anchors,
L2-init and shrink-and-perturb, how strongly the world model shapes the shared representation and how much it is
fitted, and the amount of replay and the RL step size. The first pool is thirteen single-idea
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

Three searches ran. The first charged candidates for experience but not for gradient steps. The second and third hold
the supervised steps per phase fixed; the second's baseline was already at the ceiling, so it could not show anything.
Rules a search promotes are then tested on seeds and held-out tasks it never used, all rules of a seed running at the
same time, and on the real learner (section 5).

<!-- table:meta_continual -->
#### Search 1: candidates charged for experience only

25 candidate rules, 24 ran a whole confined lifetime, 0 rejected as inert, 4 taken to more seeds. Baseline score per seed: 1: 0.556, 2: 0.472, 3: 0.435, 4: 0.306, 5: 0.435.

Screening (one seed; effect = candidate's score minus the baseline's), best first:

| round | what it changes | effect | final | forgetting | picked by the learned selector |
|---|---|---|---|---|---|
| 2 | observe_steps: 1.0 -> 2.0; rehearsal_keep: True -> False; rehearse_frac: 0.5 -> 0.25 | +0.194 | 0.50 | 0.75 |  |
| 0 | rehearsal plus replay consolidation at the end of every phase | +0.130 | 0.81 | 0.04 | yes |
| 0 | rehearsal, world model trained on a fixed representation | +0.125 | 0.83 | 0.00 | yes |
| 1 | consolidate_steps: 0 -> 96; shrink_perturb: none -> mild | +0.083 | 0.67 | 0.25 |  |
| 1 | anchor: none -> l2sp; anchor_coef: 200000.0 -> 20000.0; consolidate_steps: 0 -> 96; wm_fit: 300 -> 100 | +0.056 | 0.75 | 0.00 | yes |
| 2 | rehearsal_keep: True -> False | +0.028 | 0.50 | 0.50 | yes |
| 1 | consolidate_steps: 0 -> 96; wm_fit: 300 -> 100; wm_replay: 4 -> 0 | +0.019 | 0.72 | 0.00 |  |
| 0 | rehearsal plus shrink-and-perturb at phase boundaries | +0.000 | 0.67 | 0.25 |  |
| 0 | rehearsal with half of all replay reserved for earlier phases | +0.000 | 0.67 | 0.00 |  |
| 2 | consolidate_steps: 0 -> 96; lr: 1.0 -> 0.5; rehearse_frac: 0.5 -> 0.75; replay_steps: 1.0 -> 2.0 | -0.028 | 0.58 | 0.00 |  |
| 0 | no rehearsal (each phase forgets the reservoir of earlier arms) | -0.056 | 0.50 | 0.25 |  |
| 0 | rehearsal plus a pull towards the initial weights (L2-init) | -0.056 | 0.50 | 0.00 |  |
| 0 | rehearsal with the past favoured in replay | -0.056 | 0.50 | 0.25 |  |
| 1 | top_frac: 1.0 -> 0.1; wm_fit: 300 -> 0 | -0.056 | 0.50 | 0.25 |  |
| 1 | anchor: none -> ewc; consolidate_steps: 0 -> 32; lr: 1.0 -> 2.0 | -0.056 | 0.50 | 0.00 |  |
| 2 | anchor: none -> ewc; anchor_coef: 200000.0 -> 20000.0; consolidate_steps: 0 -> 32 | -0.056 | 0.50 | 0.00 |  |
| 2 | replay_steps: 1.0 -> 2.0 | -0.056 | 0.50 | 0.00 | yes |
| 0 | no rehearsal, importance-weighted anchors (EWC) instead | -0.111 | 0.33 | 0.25 |  |
| 0 | rehearsal plus a uniform anchor to the previous phase's weights (L2-SP) | -0.148 | 0.47 | 0.00 |  |
| 1 | rehearse_frac: 0.5 -> 0.75; replay_steps: 1.0 -> 2.0; wm_repr: 0.1 -> 0.5 | -0.148 | 0.72 | 0.00 | yes |
| 0 | rehearsal with a slower trunk (input and encoder at 0.3x) | -0.185 | 0.61 | 0.08 |  |
| 0 | rehearsal plus importance-weighted anchors (EWC) | -0.324 | 0.44 | 0.00 |  |
| 0 | rehearsal, world-model gradient shapes the representation strongly | -0.343 | 0.39 | 0.00 |  |
| 2 | consolidate_steps: 0 -> 32; observe_lr: 0.0003 -> 0.0001; shrink_perturb: none -> mild | -0.361 | 0.33 | 0.00 |  |
| 0 | self-test failed: ['gradients_reach_policy_and_value'] / rehearsal with modular updates: heads only after the first phase's trunk | did not run | - | - |  |

Taken to more seeds (difference to the champion of that moment on the same seeds; bootstrap over seeds):

| what it changes | decision | stage | seeds | score difference | final-success difference |
|---|---|---|---|---|---|
| observe_steps: 1.0 -> 2.0; rehearsal_keep: True -> False; rehearse_frac: 0.5 -> 0.25 | inconclusive | confirm | 5 | +0.248 [+0.193, +0.315] | -0.322 [-0.406, -0.239] |
| rehearsal plus replay consolidation at the end of every phase | promoted | confirm | 5 | +0.289 [+0.226, +0.352] | +0.100 [+0.028, +0.189] |
| rehearsal, world model trained on a fixed representation | harmful | develop | 3 | -0.123 [-0.194, -0.046] | -0.120 [-0.333, +0.111] |
| consolidate_steps: 0 -> 96; shrink_perturb: none -> mild | rejected_screen | screen | 1 | -0.046 [-0.046, -0.046] | -0.139 [-0.139, -0.139] |

**Test after selection**: seeds [6, 7, 8, 9, 10] and held-out tasks the search never used; all rules of a seed ran at the same time. With five seeds the sign-flip p-value cannot go below 0.0625.

| rule | score | final success | score difference to baseline | seeds better / equal / worse | sign-flip p |
|---|---|---|---|---|---|
| baseline (rehearsal) | 0.376 | 0.478 | reference | - | - |
| promoted rule | 0.644 | 0.583 | +0.269 [+0.198, +0.354] | 5 / 0 / 0 | 0.0625 |
| control: same extra steps, no consolidation | 0.965 | 0.944 | +0.589 [+0.522, +0.652] | 5 / 0 / 0 | 0.0625 |

Learned selection of what to test (it chooses before the round's outcomes are known):

| round | pool | selector fitted | its picks | pool mean (random choice) | best in pool |
|---|---|---|---|---|---|
| 0 | 13 | False | +0.127 | -0.081 | +0.130 |
| 1 | 6 | False | -0.046 | -0.017 | +0.083 |
| 2 | 6 | True | -0.014 | -0.046 | +0.194 |

#### Search 2: equal gradient steps, 150 per phase (the baseline is at the ceiling)

23 candidate rules, 22 ran a whole confined lifetime, 0 rejected as inert, 0 taken to more seeds. Baseline score per seed: 1: 1.000.

Screening (one seed; effect = candidate's score minus the baseline's), best first:

| round | what it changes | effect | final | forgetting | picked by the learned selector |
|---|---|---|---|---|---|
| 0 | faster heads (policy and critic at 3x) | +0.000 | 1.00 | 0.00 |  |
| 0 | half of all replay reserved for earlier phases | +0.000 | 1.00 | 0.00 |  |
| 1 | group_lr: uniform -> fast_heads; replay_arms: uniform -> new_heavy | +0.000 | 1.00 | 0.00 | yes |
| 0 | larger observation step size (1e-3) | -0.009 | 0.97 | 0.04 |  |
| 0 | world model trained on a fixed representation | -0.009 | 0.97 | 0.00 |  |
| 1 | group_lr: uniform -> fast_heads; replay_arms: uniform -> old_heavy; wm_repr: 0.1 -> 0.0 | -0.009 | 0.97 | 0.00 |  |
| 0 | a quarter of each observation batch rehearses earlier arms (instead of half) | -0.111 | 0.67 | 0.50 |  |
| 0 | 30 steps moved to consolidation, at the observation step size | -0.139 | 0.83 | 0.00 | yes |
| 0 | uniform anchor to the last phase's weights (L2-SP) | -0.139 | 0.83 | 0.00 |  |
| 0 | shrink-and-perturb at phase boundaries | -0.139 | 0.83 | 0.00 | yes |
| 0 | pull towards the initial weights (L2-init) | -0.139 | 0.83 | 0.00 |  |
| 0 | 30 of the 150 steps moved to end-of-phase consolidation | -0.185 | 0.69 | 0.00 |  |
| 0 | importance-weighted anchors (EWC) on top of rehearsal | -0.194 | 0.67 | 0.00 |  |
| 0 | replay favours earlier arms | -0.194 | 0.67 | 0.00 |  |
| 1 | consolidate_steps: 0 -> 30; group_lr: uniform -> fast_heads; observe_steps: 1.0 -> 0.8; wm_fit: 300 -> 100 | -0.194 | 0.67 | 0.00 |  |
| 0 | 60 steps moved to consolidation, at the observation step size | -0.287 | 0.64 | 0.04 |  |
| 1 | consolidate_steps: 0 -> 60; observe_steps: 1.0 -> 0.6; top_frac: 1.0 -> 0.1; wm_fit: 300 -> 100 | -0.306 | 0.83 | 0.00 | yes |
| 1 | anchor: none -> l2sp; anchor_coef: 200000.0 -> 2000000.0; consolidate_lr: None -> 0.0003; wm_fit: 300 -> 0 | -0.333 | 0.50 | 0.00 |  |
| 0 | slower trunk (input and encoder at 0.3x) | -0.417 | 0.50 | 0.50 |  |
| 1 | consolidate_steps: 0 -> 90; l2init: 0.0 -> 20000.0; observe_steps: 1.0 -> 0.4 | -0.435 | 0.61 | 0.08 |  |
| 0 | 60 of the 150 steps moved to end-of-phase consolidation | -0.463 | 0.61 | 0.00 |  |
| 0 | smaller observation step size (1e-4) | -0.694 | 0.67 | 0.00 |  |
| 0 | FileNotFoundError: [Errno 2] No such file or directory: '/workspace/agentrs/.worktrees/superlearner-4/superlearner/runs/meta/conti | did not run | - | - |  |

Learned selection of what to test (it chooses before the round's outcomes are known):

| round | pool | selector fitted | its picks | pool mean (random choice) | best in pool |
|---|---|---|---|---|---|
| 0 | 17 | False | -0.139 | -0.185 | +0.000 |
| 1 | 6 | False | -0.153 | -0.213 | +0.000 |

#### Search 3: equal gradient steps, 100 per phase

23 candidate rules, 23 ran a whole confined lifetime, 0 rejected as inert, 3 taken to more seeds. Baseline score per seed: 1: 0.500, 2: 0.500, 3: 0.463, 4: 0.481, 5: 0.537.

Screening (one seed; effect = candidate's score minus the baseline's), best first:

| round | what it changes | effect | final | forgetting | picked by the learned selector |
|---|---|---|---|---|---|
| 1 | observe_lr: 0.0003 -> 0.001; replay_arms: uniform -> new_heavy | +0.500 | 1.00 | 0.00 | yes |
| 1 | consolidate_steps: 0 -> 20; observe_lr: 0.0003 -> 0.001; observe_steps: 1.0 -> 0.8; wm_fit: 300 -> 100 | +0.444 | 0.83 | 0.00 |  |
| 0 | larger observation step size (1e-3) | +0.417 | 0.83 | 0.25 |  |
| 1 | observe_lr: 0.0003 -> 0.001; replay_arms: uniform -> old_heavy; wm_repr: 0.1 -> 0.0 | +0.389 | 0.67 | 0.25 |  |
| 0 | faster heads (policy and critic at 3x) | +0.296 | 0.56 | 0.25 |  |
| 0 | world model trained on a fixed representation | +0.111 | 0.83 | 0.00 |  |
| 0 | 40 steps moved to consolidation, at the observation step size | +0.102 | 0.56 | 0.42 |  |
| 0 | replay favours earlier arms | +0.056 | 0.67 | 0.00 |  |
| 0 | half of all replay reserved for earlier phases | +0.056 | 0.67 | 0.00 |  |
| 0 | pull towards the initial weights (L2-init) | +0.037 | 0.61 | 0.00 |  |
| 0 | uniform anchor to the last phase's weights (L2-SP) | +0.009 | 0.53 | 0.00 |  |
| 0 | shrink-and-perturb at phase boundaries | +0.000 | 0.50 | 0.00 | yes |
| 0 | a quarter of each observation batch rehearses earlier arms (instead of half) | -0.014 | 0.33 | 0.38 |  |
| 0 | importance-weighted anchors (EWC) on top of rehearsal | -0.083 | 0.50 | 0.00 |  |
| 1 | anchor: none -> l2sp; anchor_coef: 200000.0 -> 2000000.0; consolidate_lr: None -> 0.0003; wm_fit: 300 -> 0 | -0.139 | 0.33 | 0.00 |  |
| 0 | 20 of the 100 steps moved to end-of-phase consolidation | -0.167 | 0.50 | 0.00 |  |
| 0 | slower trunk (input and encoder at 0.3x) | -0.222 | 0.58 | 0.00 |  |
| 0 | 20 steps moved to consolidation, at the observation step size | -0.250 | 0.50 | 0.00 | yes |
| 0 | three quarters of each observation batch rehearse earlier arms | -0.250 | 0.50 | 0.00 |  |
| 0 | 40 of the 100 steps moved to end-of-phase consolidation | -0.306 | 0.33 | 0.00 |  |
| 0 | smaller observation step size (1e-4) | -0.306 | 0.33 | 0.00 |  |
| 1 | consolidate_steps: 0 -> 40; observe_steps: 1.0 -> 0.6; top_frac: 1.0 -> 0.1; wm_fit: 300 -> 100 | -0.306 | 0.33 | 0.00 | yes |
| 1 | consolidate_steps: 0 -> 60; l2init: 0.0 -> 20000.0; observe_steps: 1.0 -> 0.4 | -0.306 | 0.33 | 0.00 |  |

Taken to more seeds (difference to the champion of that moment on the same seeds; bootstrap over seeds):

| what it changes | decision | stage | seeds | score difference | final-success difference |
|---|---|---|---|---|---|
| observe_lr: 0.0003 -> 0.001; replay_arms: uniform -> new_heavy | promoted | confirm | 5 | +0.409 [+0.369, +0.467] | +0.278 [+0.150, +0.428] |
| consolidate_steps: 0 -> 20; observe_lr: 0.0003 -> 0.001; observe_steps: 1.0 -> 0.8; wm_fit: 300 -> 100 | rejected_screen | screen | 1 | -0.056 [-0.056, -0.056] | -0.167 [-0.167, -0.167] |
| larger observation step size (1e-3) | rejected_screen | screen | 1 | -0.083 [-0.083, -0.083] | -0.167 [-0.167, -0.167] |

**Test after selection**: seeds [6, 7, 8, 9, 10] and held-out tasks the search never used; all rules of a seed ran at the same time. With five seeds the sign-flip p-value cannot go below 0.0625.

| rule | score | final success | score difference to baseline | seeds better / equal / worse | sign-flip p |
|---|---|---|---|---|---|
| baseline (rehearsal) | 0.387 | 0.478 | reference | - | - |
| promoted rule | 0.944 | 0.850 | +0.557 [+0.494, +0.620] | 5 / 0 / 0 | 0.0625 |

Learned selection of what to test (it chooses before the round's outcomes are known):

| round | pool | selector fitted | its picks | pool mean (random choice) | best in pool |
|---|---|---|---|---|---|
| 0 | 17 | False | -0.125 | -0.030 | +0.417 |
| 1 | 6 | False | +0.097 | +0.097 | +0.500 |
<!-- /table -->

## 7. What this shows and what it does not

**What the experiments establish** (on eight task arms in this simulator):

1. **Gradient updates to the controller and world model are the learning mechanism.** Skills acquired this way persist
   through checkpoints and restarts and work on unseen tasks from the weights alone. The controls fail: the same
   experience written to memory with frozen weights leaves `build` at 0%, and the frozen-core lifetime ends at 25%.
2. **Continual learning without protection forgets, and rehearsal fixes it here.** Keeping a per-arm reservoir of
   verified successes and replaying it took forgetting from 31% to 0%.
3. **Demonstrations are by far the strongest route to a new skill.** Four of them gave `build` 100% on every seed.
   Practice alone acquired it on some seeds and not others, and did nothing for `repair`.

**What they suggest but do not establish:**

* **Training the language backbone helps.** Every backbone condition scored above its frozen reference, but with one or
  two seeds, the full-backbone interval touching zero, and the gain concentrated in `repair`. The swap test shows the
  trained backbone matters to the final learner (92% with it, 38% with pretrained weights put back), but the controller
  co-adapted to it, so that is not a measure of how much knowledge the backbone holds.
* **Exploration guided by the world model helps practice.** `build` reached 75% against 50%, on three seeds with wide
  intervals.

**What did not work:**

* **Written memory.** It did nothing for a frozen core and lowered a trainable one by 25 points.
* **Importance-weighted anchors (EWC).** Alone they did not prevent forgetting at any coefficient tried (calibration, one seed), and on top of rehearsal they scored below it in the searches.
* **End-of-phase consolidation** added nothing over rehearsal (-3.6 points [-10.7, +2.2]).
* **The Meta-Researcher's promoted rules, for the real learner.** The searches ran on a small CPU controller. Their
  results replicated on new seeds and tasks for that controller, and one did not survive contact with the real one: a
  larger step size that helped the proxy made the real learner overwrite old skills when steps were scarce (forgetting
  39% against 1%). Its learned selector of what to test was fitted in only one of these seven rounds, so these searches
  say almost nothing about it. **So the requirement that the
  Meta-Researcher be evaluated on discovering better weight updates has been met, and the answer so far is no.** A
  search would need to run on the real learner, which costs about an hour per candidate lifetime.

**What remains unproven:**

* Learning on anything outside this ladder of eight related arms. "Unseen" here means unseen seeds and world families,
  not unseen kinds of problem.
* Any adaptation of the 14B generator, or that adapting a language model improves its *reasoning*. The trained backbone
  only produces embeddings.
* Durability over long lifetimes. Six phases and about 17,000 decisions cannot show loss of plasticity.
* Robustness across seeds: three at most, one for two of the backbone conditions.

## 8. Reproduce

```bash
cd superlearner && export PYTHONPATH=src
python -m superlearner jobs configs/queue_continual.yaml --log-dir runs/queue_continual     # lifetimes + PIT (GPU)
python -m superlearner jobs configs/queue_continual_final.yaml --log-dir runs/queue_continual_final   # untouched tasks, swap test
CUDA_VISIBLE_DEVICES="" python -m superlearner meta-search --config configs/meta_continual.yaml --full-feedback --run-dir runs/meta/continual1
CUDA_VISIBLE_DEVICES="" python -m superlearner meta-search --config configs/meta_continual_equal.yaml --full-feedback --run-dir runs/meta/continual2
CUDA_VISIBLE_DEVICES="" python -m superlearner meta-search --config configs/meta_continual_equal100.yaml --full-feedback --run-dir runs/meta/continual3
CUDA_VISIBLE_DEVICES="" python -m superlearner meta-verify --config configs/meta_continual.yaml --run-dir runs/meta/continual1
CUDA_VISIBLE_DEVICES="" python -m superlearner meta-verify --config configs/meta_continual_equal100.yaml --run-dir runs/meta/continual3
python -m superlearner pit report --config configs/pit.yaml
python -m superlearner continual-report
python docs/experiments/render.py --fill docs/CONTINUAL.md
python -m pytest -q tests/test_continual.py                                                 # CPU, a few minutes
```

A single lifetime: `python -m superlearner lifetime --config configs/life/consolidate.yaml --set seed=0`; it
continues where it stopped if run again. A CPU-only check of everything: `configs/life_smoke.yaml`,
`configs/pit_smoke.yaml`, `configs/meta_continual_smoke.yaml`.
