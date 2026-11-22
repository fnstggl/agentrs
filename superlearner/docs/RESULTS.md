# Results

Everything here was measured on 2026-11-21 on one RTX A6000 (46 GB), 48 logical CPU cores (shared with other
tenants of the host), 188 GB RAM, with Qwen3-14B as the frozen language model. The instance was down from 15:44 to
21:35; what that interrupted and how each run was recovered is in the lab notes. Raw outputs are in `docs/experiments/*.json`; the chronological account,
including every superseded attempt, is `docs/experiments/NOTES.md`. Success always means the task's predicate on
authoritative world state, checked by the evaluator. Intervals are 95% (Wilson for one condition, bootstrap over
seeds for differences).

**Short version.**

1. **The inner learner learns, modestly.** Starting from a controller cloned from two demonstrations per task,
   twelve iterations of recurrent PPO raised held-out success from 65% to 73%, and with replay of its own successes
   to 77% (three seeds, 96 tasks each; 36 tasks gained and 2 lost with replay). On the sealed final list, opened
   once: 63.5% → 76.0%, interval [+0.0, +25.0]. RL repaired skills the warm start half-had; it did not acquire a
   skill the warm start lacked.
2. **It acts for real.** The controller with the frozen LLM built and deployed a working service from nothing in 8
   of 8 new worlds (the LLM wrote the code; a scripted teacher managed 5 of 8; the LLM without the controller 0),
   and scored 70.8% on the whole sealed list. Nobody — controller or script — operated such a service to the
   traffic goal.
3. **The capability is in the trained weights.** Weights kept, memory reset: 77%. Memory kept, weights reset: 0%.
   The frozen LLM alone: 12%; with the grounding layer but no learning: 23%.
4. **Episodic memory hurt this agent** (42% with it, 70% without), and that confound had first looked like RL
   destroying the policy. Four failed versions of the main experiment are documented with their causes.
5. **Transfer is zero-shot only.** Unseen world families: 75% against 78% in familiar ones. No faster adaptation.
6. **Knowledge persists until something new is learned.** Cloning a new skill erased two of three old ones
   (100% → 33%); with rehearsal none (100%).
7. **The Meta-Researcher works as a machine and found nothing.** It generated 24 algorithm changes as code, ran 22
   of them confined against the unchanged baseline under matched budgets, and promoted none: the four best screened
   candidates shrank to +0.003…+0.017 with intervals around zero when given more seeds. Three LLM-written
   "improvements" were found to be no-ops whose scores (up to +0.040) measure the noise of the comparison. Its
   learned selection model did not beat random selection.
8. **Negative:** several learners in one world (no trade, so no competitive learning), RL with LLM-written content
   in the loop (no change in four iterations), planning with the learned world model (+3 points, inside the noise).

## 1. What the frozen language model can and cannot do here

| Probe | Result |
|---|---|
| Qwen3-4B, "write the files of a service for need X" with public documents only, 8 needs | wrote `service.py` and no manifest for all 8 |
| Qwen3-14B, same probe (code executed on 6 hidden inputs) | working code for 5/8: sort.records 1.00, route.shortest 1.00, match.stable 1.00, schedule.jobs 0.95, forecast.series 0.50 (reference 0.68); route.tsp 0, pack.knapsack timed out, text.extract invalid JSON |
| Speed | about 18 tokens/s; 13–37 s per file set |
| One-shot "files" object (JSON inside JSON) | manifests with unbalanced braces in most attempts |
| Entry-by-entry writing, no reasoning | escaping fixed; manifest `params` written as a property map instead of a JSON schema in 2/2 attempts; repair with the error message shown: 1/2 valid |
| Entry-by-entry, reasoning mode for the manifest | 2/2 first attempts valid (43–52 s each) |
| Reasoning mode for a code file | used its 2,700-token budget and returned nothing (153 s); reasoning is therefore limited to structured entries, with a fallback |

Two defects of the interface the earlier Crucible agents used, found while building the grounding layer:

* Crucible's tool renderer shows only the first line of each description. The deploy manifest format is on the
  following lines of `cloud.deploy`. The Qwen agents of Crucible 3.0's final challenge were never shown it.
* Tool retrieval by mean-pooled hidden states is poor. On a probe, the tool that documents the manifest ranked 14th
  for "commit files for a deployable service" and `market.list` ranked 98th for "have an active market listing";
  with prompt-based last-token embeddings from layer −4 they rank 2nd and 1st.

## 2. Demonstrations and the warm start

Scripted teachers act through the controller's own heads and options; only evaluator-verified episodes are kept.

| Measurement | Result |
|---|---|
| Teacher on the eight arms that need no LLM content | 191/191 episodes verified (after fixing a teacher bug: it listed services under a random need, 146/191 before) |
| Teacher on LLM-dependent arms (52 episodes over 13 arms, small n) | apicall 5/6, code 1/2, deploy from scratch 2/7, service from scratch 0/3, revenue 0/2 |
| Cloning, 16 demonstrations per arm, greedy held-out success after 30 / 60 / 100 / 150 epochs | 0.87 / 0.97 / 0.88 / 1.00 (64 tasks) |
| Cloning, 2 demonstrations per arm, critic sharing the controller's state | 0.34 (80 tasks) |
| Same demonstrations and seed, critic detached | 0.75 |
| Same settings again (GPU nondeterminism, 16 demonstrations) | 0.78 and 0.51 |
| One checkpoint, same 80 tasks, five evaluations under different host load | 0.613, 0.650, 0.662, 0.675, 0.688 |

Consequences for method: the critic is detached from the controller's state; compared runs branch from one saved
warm-start checkpoint per seed; differences of a few points between single evaluations are noise.

## 3. A confound that looked like "RL destroys the policy"

Four successive versions of Experiment B showed held-out success falling in the first iterations of RL
(for example 0.775 → 0.500). Two real problems were found and fixed on the way (the first policy update after a
warm start moved every weight by a full learning-rate step: KL 0.13–0.17 after two steps; and prerequisite gating
left warm-started skills unrehearsed). The drop remained. In one run every policy step of iterations 2–4 had been
rolled back, so the weights were the warm-start weights, and it still scored 0.500. The direct test:

| Same checkpoint, same 80 tasks | Greedy success |
|---|---|
| with the episodic memory and tool library accumulated during training | 0.5125 |
| with memory and tools reset | 0.7625 |

The controller had been cloned with an empty memory. As soon as training episodes filled the memory, remembered
episodes and "what worked before" options appeared in its context and shifted its choices. This is a property of
adding non-parametric memory to a policy that was not trained with it, not of the learning rule. Experiment B
below therefore runs with episodic memory and tool-making off, and memory is examined separately (section 5).

## 4. Experiment B — does reinforcement learning improve the controller?

Design. Three seeds. For each seed one warm-start checkpoint (cloned from 2 verified demonstrations per task arm,
8 arms that need no LLM-written content), from which two runs branch: recurrent PPO with GAE, and the same plus
replay of the run's own verified successes. 12 iterations of 96 episodes (10,000–13,000 decisions). Episodic memory
and tool-making off (section 3). Before and after are evaluated on the same 96 held-out tasks (unseen seeds; half of
the worlds from two families never used in training), greedy and with sampled actions.

<!-- table:b -->
### Experiment B (exp_b_multi.json): held-out success before and after RL, 96 tasks, families ['bazaar', 'entrenched_incumbents', 'fragmented_networks', 'tiny']

| seed | warm start (before RL) | PPO alone | PPO + replay of own successes |
|---|---|---|---|
| 0 | 68% [58, 76] | 69% [59, 77] (+1) | 78% [69, 85] (+10) |
| 1 | 76% [67, 83] | 78% [69, 85] (+2) | 77% [68, 84] (+1) |
| 2 | 52% [42, 62] | 73% [63, 81] (+21) | 76% [67, 83] (+24) |

Sampled actions (same tasks):

| seed | warm start (before RL) | PPO alone | PPO + replay of own successes |
|---|---|---|---|
| 0 | 66% [56, 74] | 70% [60, 78] (+4) | 69% [59, 77] (+3) |
| 1 | 61% [51, 71] | 69% [59, 77] (+7) | 73% [63, 81] (+11) |
| 2 | 39% [29, 49] | 48% [38, 58] (+9) | 52% [42, 62] (+14) |

* PPO alone: after - before (held-out success): +8.0 points, 95% CI [+1.0, +20.8], P(>0) 1.00, n=3 seeds, established=True
* PPO alone: after - before (held-out success, sampled actions): +6.9 points, 95% CI [+4.2, +9.4], P(>0) 1.00, n=3 seeds, established=True
* PPO + replay of own successes: after - before (held-out success): +11.8 points, 95% CI [+1.0, +24.0], P(>0) 1.00, n=3 seeds, established=True
* PPO + replay of own successes: after - before (held-out success, sampled actions): +9.4 points, 95% CI [+3.1, +13.5], P(>0) 1.00, n=3 seeds, established=True
* PPO + replay of own successes - PPO alone: held-out success after RL: +3.8 points, 95% CI [-1.0, +9.4], P(>0) 0.96, n=3 seeds, established=False
* PPO + replay of own successes - PPO alone: held-out success after RL (sampled actions): +2.4 points, 95% CI [-1.0, +4.2], P(>0) 0.96, n=3 seeds, established=False
* PPO + replay of own successes - PPO alone: AULC of the in-training curve: +2.6 points, 95% CI [+0.3, +6.3], P(>0) 1.00, n=3 seeds, established=True
* PPO alone: paired over tasks (greedy): +8.0 points, 95% CI [+0.0, +19.5] (bootstrap over seeds and tasks), 288 task pairs: 25 gained, 2 lost, 261 unchanged, established=False
* PPO alone: paired over tasks (sampled): +6.9 points, 95% CI [+1.4, +12.5] (bootstrap over seeds and tasks), 288 task pairs: 35 gained, 15 lost, 238 unchanged, established=True
* PPO + replay of own successes: paired over tasks (greedy): +11.8 points, 95% CI [+1.0, +24.0] (bootstrap over seeds and tasks), 288 task pairs: 36 gained, 2 lost, 250 unchanged, established=True
* PPO + replay of own successes: paired over tasks (sampled): +9.4 points, 95% CI [+2.1, +16.3] (bootstrap over seeds and tasks), 288 task pairs: 44 gained, 17 lost, 227 unchanged, established=True

| task arm | before | PPO alone | PPO + replay of own successes |
|---|---|---|---|
| deploy/0/coded | 81% | 100% | 100% |
| found/0/scratch | 100% | 100% | 100% |
| fund/0/scratch | 0% | 0% | 0% |
| list/0/deployed | 86% | 100% | 94% |
| repair/0/scratch | 17% | 22% | 22% |
| service/0/coded | 78% | 83% | 100% |
| service/0/deployed | 61% | 81% | 100% |
| service/0/listed | 100% | 100% | 100% |

| seed | arm | in-training curve (decisions: success) | AULC | decisions | weights moved (L2, relative) | updates that changed weights | replay steps |
|---|---|---|---|---|---|---|---|
| 0 | PPO alone | 0:66% 4129:64% 8026:64% 12354:67% | 0.649 | 12354 | 0.74, 0.28% | 12/12 | 0 |
| 0 | PPO + replay of own successes | 0:61% 4234:64% 7870:66% 11633:77% | 0.660 | 11633 | 0.84, 0.32% | 12/12 | 176 |
| 1 | PPO alone | 0:78% 4274:77% 8682:77% 13201:75% | 0.765 | 13201 | 0.33, 0.12% | 12/12 | 0 |
| 1 | PPO + replay of own successes | 0:77% 4186:73% 8281:80% 12910:78% | 0.769 | 12910 | 0.56, 0.21% | 12/12 | 176 |
| 2 | PPO alone | 0:52% 3321:62% 6726:66% 10386:70% | 0.632 | 10386 | 0.65, 0.25% | 12/12 | 0 |
| 2 | PPO + replay of own successes | 0:52% 3498:75% 6972:70% 10372:75% | 0.695 | 10372 | 0.77, 0.29% | 12/12 | 176 |
<!-- /table -->

What this shows.

* Held-out success rose in every seed and both arms; nothing collapsed. Pooled over the three seeds: PPO alone
  65.3% → 73.3% greedy and 55.2% → 62.2% sampled; PPO + replay 65.3% → 77.1% greedy and 55.2% → 64.6% sampled.
* With three seeds the interval over seeds is wide. Pairing the same tasks before and after (bootstrap over seeds
  and tasks): PPO + replay is above zero for greedy and sampled actions; PPO alone for sampled actions, and at
  the boundary for greedy ([+0.0, +19.5]). Tasks that flipped: 36 gained and 2 lost (replay, greedy), 25 gained and
  2 lost (PPO, greedy).
* The gain is uneven: the seed with the weakest warm start (52%) gained 21–24 points, the strongest (76%) 1–2.
* It is repair, not discovery. Arms the warm start did partly (deploying a coded service 81%, operating a service
  from a deployed or coded start 61–78%) went to 81–100%. Arms it could not do stayed where they were: `fund` 0%
  greedy before and after (the greedy policy names the same account as source and target; sampled actions get
  past it), `repair` 17% → 22%.
* Replay against PPO alone: +3.8 points after RL (interval includes zero), +2.6 points of area under the
  in-training curve (interval above zero, three seeds).
* The weights moved little: 0.1–0.3% of their norm. The KL check rolled back 4–7 policy steps per run and halved
  the step size each time, so by iteration 12 the policy learning rate was between 1/13 and 1/80 of its start. Later
  iterations therefore contributed little; the rule protects the policy but also throttles it.

One of the six runs (PPO + replay, seed 1) is a second run: the first was interrupted at iteration 9 and could not
be resumed faithfully, because the replay buffer was not in its checkpoint (fixed since; lab notes).

## 5. Part 5 — the seven conditions on the same tasks

Same 64 held-out tasks, same world mechanics, same decision budget; the checkpoint is PPO + replay, seed 0.

<!-- table:baselines -->
### Part 5 comparison (baselines_exp_b_fewdemo_sil_s0.json): 64 held-out tasks, families ['bazaar', 'entrenched_incumbents', 'fragmented_networks', 'tiny']

| condition | success [95% CI] | valid actions | episodes | failures by class |
|---|---|---|---|---|
| 0 uniform random (same action space) | 2% [0, 8] | 29% | 64 | nonexistent_resource 1437, syntax 129, permission 112, unknown_tool 13 |
| 1 frozen LLM alone | 12% [4, 36] | 74% | 16 | invalid_action 86, not_found 2, conflict 2 |
| 2 frozen LLM + grounding, no learning | 23% [14, 34] | 56% | 62 | permission 82, syntax 34, nonexistent_resource 34, prerequisite 31 |
| 3 warm-started CAM before RL | 67% [55, 77] | 37% | 64 | syntax 128, loop_blocked 128, nonexistent_resource 32, permission 4 |
| 4 CAM after RL | 77% [65, 85] | 40% | 64 | loop_blocked 150, syntax 109, nonexistent_resource 32, permission 1 |
| 5 weights restored, memory reset | 77% [65, 85] | 40% | 64 | loop_blocked 150, syntax 110, nonexistent_resource 32 |
| 6 memory preserved, weights reset | 0% [0, 6] | 3% | 64 | syntax 437, nonexistent_resource 358 |
| 7 scripted control | 100% [94, 100] | 100% | 64 |  |

memory items 0, agent tools 0, parameter distance RL vs warm start 0.8384179472923279

### Part 5 comparison (baselines_memory_on_sil_s0.json): 64 held-out tasks, families ['bazaar', 'entrenched_incumbents', 'fragmented_networks', 'tiny']

| condition | success [95% CI] | valid actions | episodes | failures by class |
|---|---|---|---|---|
| 0 uniform random (same action space) | 2% [0, 8] | 31% | 64 | nonexistent_resource 1391, syntax 125, permission 125, unknown_tool 17 |
| 3 warm-started CAM before RL | 66% [53, 76] | 37% | 64 | syntax 129, loop_blocked 128, nonexistent_resource 32, permission 4 |
| 4 CAM after RL | 42% [31, 54] | 31% | 64 | loop_blocked 241, nonexistent_resource 45, syntax 38, permission 23 |
| 5 weights restored, memory reset | 70% [58, 80] | 36% | 64 | syntax 130, loop_blocked 128, nonexistent_resource 32, permission 4 |
| 6 memory preserved, weights reset | 0% [0, 6] | 1% | 64 | syntax 106, nonexistent_resource 55 |
| 7 scripted control | 100% [94, 100] | 100% | 64 |  |

memory items 374, agent tools 0, parameter distance RL vs warm start 0.6898778676986694
<!-- /table -->

The first table is the learner of Experiment B (trained without episodic memory, so conditions 4 and 5 coincide by
construction); the second is a run of the same configuration trained *with* memory and tool-making on (four RL
iterations, 374 remembered episodes), which is where conditions 5 and 6 say something.

* **Frozen LLM alone** (Crucible's original interface: the full tool list in the prompt, one JSON tool call per
  turn, no grounding, no controller; first 16 of the tasks, 24 decisions each): 2 of 16. It founds the company and
  nothing else.
* **Frozen LLM + grounding, no learning** (retrieval of relevant tools, validated calls, classified failures, drafts
  it writes itself): 23% of 62. It founds the company (7 of 8) and passes the one arm where waiting is enough.
  Better interfaces alone took the model from 12% to 23%, not further.
* **Warm-started controller before RL**: 67%. **After RL**: 77%. Ten points on 64 tasks, the same direction and
  size as in Experiment B's three seeds; for this single checkpoint the intervals overlap.
* **Weights restored, memory reset**: 77% (70% for the run trained with memory). **Memory preserved, weights
  reset**: 0% in both. Whatever this agent can do is in its trained weights; memory alone carries nothing.
* **With the memory it accumulated, the memory-trained run scores 42%, with memory reset 70%** ([31, 54] against
  [58, 80]): the confound of section 3, repeated on a different checkpoint and task list.
* **Scripted control**: 100%. The tasks are solvable through the same action interface, and the learned controller
  is still far from a competent script on `fund` (0%) and `repair` (0–25%).
* **Uniform random choices in the same action space**: 2%.

## 6. Experiment C — transfer

<!-- table:c -->
### Experiment C (exp_c_exp_b_fewdemo_sil_s0.json): transfer

| evaluation set | success [95% CI] | valid actions | n |
|---|---|---|---|
| training families, unseen seeds | 78% [67, 86] | 40% | 64 |
| unfamiliar world families (validation) | 75% [63, 84] | 40% | 64 |
| unfamiliar world families (test: fragile infrastructure, novel needs) | 75% [63, 84] | 38% | 64 |
| harder objective (service level 1, never trained) | 94% [80, 98] | 100% | 32 |

adaptation from RL-trained weights: 0 decisions: 73%, 1079 decisions: 75%, 2169 decisions: 73%, 3147 decisions: 73%, 4094 decisions: 73%

adaptation from warm start (before RL): 0 decisions: 66%, 857 decisions: 70%, 2048 decisions: 67%, 3236 decisions: 67%, 4240 decisions: 64%
<!-- /table -->

* Without any new training the controller does as well in world families it never saw (75%, validation and test
  families) as on new seeds of the families it trained in (78%); the intervals overlap almost completely. What
  decides success is the task arm, not the world: six arms are at 100% in all three sets, `fund` at 0%, `repair`
  at 0–25%.
* A harder objective that was never trained (5,000 successfully served requests in a seven-day window instead of
  500, from a deployed or listed start): 94% of 32. From those starts the harder goal mostly needs the same
  behaviour for longer, so this is weak evidence of transfer to a new objective.
* Adaptation by four more RL iterations in the unfamiliar families (about 4,000 decisions) changed nothing
  measurable: 73% → 73% from the RL-trained weights, 66% → 64% from the warm start (one run each; the differences
  along the curve are one to three tasks out of 64). No faster adaptation of the RL-trained controller can be
  claimed.
* The test-family worlds used here have other seeds than the sealed final list (section 12).

## 7. Experiment D — retention

Skill set A: found a company, fund an account, list a deployed service. Unrelated set B: repair a degraded
service, deploy a coded service, operate a listed service.

<!-- table:d -->
### Experiment D (exp_d_retention.json): retention

A = ['found/0/scratch', 'fund/0/scratch', 'list/0/deployed'], B = ['repair/0/scratch', 'deploy/0/coded', 'service/0/listed']; iterations {'A': 4, 'B': 6}

A at warm start 100%; after learning A 100%; B before any B training 0%

| phase-B strategy | A after training on B | forgetting on A | B after training on B | gain on B |
|---|---|---|---|---|
| no protection | 100% [93, 100] | +0 | 0% [0, 7] | +0 |
| replay of own earlier successes | 100% [93, 100] | +0 | 0% [0, 7] | +0 |
| rehearsal (A sampled at a low rate) | 100% [93, 100] | +0 | 0% [0, 7] | +0 |
| demonstration regulariser on A (bc_coef 0.5) | 100% [93, 100] | +0 | 0% [0, 7] | +0 |
<!-- /table -->

Reinforcement learning on B alone, for six iterations, taught the controller nothing about B (no demonstrations
for B; no sampled success on B's tasks during training — the 6–17% seen in the rehearsal run match the share of A
tasks mixed in — and 0% held-out), and A stayed at 100% under all four strategies.
The weights moved by about 0.5% of their norm in that phase (measured for the two strategies run after the outage).
So this version of the experiment says only that A survives a few thousand decisions of unsuccessful RL on other
tasks; it does not test interference from actually learning something new. Phase A and the first two strategies
were run before the outage and are reported from the kept log (success rates only).

The supervised variant does test it: from the same checkpoint, B is learned by cloning 12 verified demonstrations.

<!-- table:d_clone -->
### Experiment D, supervised variant (exp_d_clone.json): A 100%, B 0% before; demonstrations A 12, B 12

| phase-B training | A afterwards | forgetting on A | B afterwards | gain on B | weights moved (L2) |
|---|---|---|---|---|---|
| clone B only | 33% [22, 47] | +67 | 71% [57, 82] | +71 | 12.90 of 262.2 |
| clone B with A's demonstrations mixed in | 100% [93, 100] | +0 | 96% [86, 99] | +96 | 15.14 of 262.2 |
<!-- /table -->

Learning B alone erased most of A (100% → 33%: `fund` and `list` went to 0%, `found` survived) while B reached
71%. With A's twelve demonstrations mixed into the same cloning run, A stayed at 100% and B reached 96%. Knowledge
in the weights persists across checkpoints and across unrelated experience that does not move them much; it does
not survive learning a new skill unless the old one is rehearsed.

## 8. Experiment A — building and operating a service from scratch

The controller here is cloned from verified demonstrations on all thirteen task arms, including the ones where the
frozen LLM has to write the service (teacher success on those arms, which bounds what can be cloned: deploy from
scratch 17/21, code 2/10, API call 14/17, operate from scratch 0/11, revenue 0/12 — so there are no demonstrations
at all for the last two). Test: 16 held-out tasks in new worlds, eight "deploy a working service starting from
nothing" and eight "operate one until it has served 500 requests", every condition on the same tasks.

<!-- table:a -->
### Experiment A (exp_a_real_actions.json): building and operating a service from scratch, 16 tasks

| who acts | success [95% CI] | valid actions | n | LLM-written values | failures |
|---|---|---|---|---|---|
| CAM controller (greedy) | 50% [28, 72] | 76% | 16 | 36 | loop_blocked 55, nonexistent_resource 24, syntax 21 |
| CAM controller (sampled) | 44% [23, 67] | 89% | 16 | 51 | permission 17, syntax 14, nonexistent_resource 9, loop_blocked 9 |
| scripted control through the same interface | 31% [14, 56] | 92% | 16 | 25 | syntax 20 |
| frozen LLM + grounding, no controller | 0% [0, 19] | 14% | 16 | 0 | nonexistent_resource 40, prerequisite 40, unknown_tool 8, loop_blocked 8 |
<!-- /table -->

* **Deploying from nothing works.** The cloned controller succeeded in 8 of 8 deploy tasks (greedy; 7 of 8
  sampled): create an organisation, register a company, open and fund an account, read the market and the format
  documents, create a repository, have the LLM write the service code and its manifest, commit, rent a machine,
  deploy — 13 decisions, each a real recorded command, the result a live service whose code no script supplied.
  The scripted teacher, through the same interface and with the same LLM, did 5 of 8.
* **Operating it to the traffic goal did not work for anyone**: 0 of 8 for the controller, 0 of 8 for the scripted
  teacher (and 0 of 11 when the demonstrations were collected). The controller's failed episodes show `market.list`
  rejected for its arguments several times, then repeated calls to the endpoint of the service it had just
  deployed. Why the teacher's correctly listed LLM-written services do not reach 500 good requests was not
  diagnosed.
* **The frozen LLM with the grounding layer but no controller: 0 of 16**, 14% of its calls accepted; it never got
  past the prerequisites (40 prerequisite and 40 missing-resource failures).
* With 16 tasks the intervals are wide: 50% [28, 72] against 31% [14, 56] is not an established difference between
  controller and teacher; the difference to the LLM without a controller (0% [0, 19]) is.

RL on top of this warm start (PPO + replay, the LLM in the loop for five of the thirteen arms):

| after | decisions | held-out success, greedy (52 tasks, 13 arms) | sampled |
|---|---|---|---|
| warm start | 0 | 61.5% [48, 74] | 63.5% [50, 75] |
| 2 iterations | 3,849 | 67.3% [54, 78] | 63.5% [50, 75] |
| 4 iterations | 8,020 | 65.4% [52, 77] | 63.5% [50, 75] |

Four iterations of 96 episodes took 69 minutes of rollouts, almost all of it waiting for the LLM to write services
(660 values written, 48,000 tokens generated in this run, the rest served from the cache). Two or three tasks out of
52 changed; that is no measurable effect. Sampled success on the training tasks fell from 58% to 43% over the four
iterations while the adaptive curriculum moved towards the arms that fail, so the run does not show learning there
either. The arms that depend on LLM-written content stayed where the warm start left them (deploy from scratch
75–100% of 4, code 0–50% of 4, operate from scratch and revenue 0%). RL with the LLM in the loop was therefore run
at a scale that could not have shown much, and showed nothing.

## 9. Learned world model

A 1.06M-parameter model on the controller's (frozen) state, trained on recorded transitions of one run and tested
on held-out episodes of the same run.

<!-- table:worldmodel -->
### World model (exp_worldmodel_exp_b_fewdemo_sil_s0.json): 3096 training / 667 held-out transitions

| prediction | model (held-out) | trivial predictor |
|---|---|---|
| action accepted (accuracy) | 0.940 | 0.747 (base rate) |
| action accepted (NLL) | 0.176 | 0.592 |
| failure class (accuracy) | 0.829 | 0.747 (majority) |
| reward (MSE) | 0.1651 | 0.2197 (mean) |
| return-to-go (MSE) | 0.6253 | 0.8622 (mean) |
| next context (cosine) | 0.938 | 0.600 (mean vector) |

planning: model-free 77% [65, 85] | 40% | 64; one-step lookahead 80% [68, 88] | 42% | 64; difference +3.1 points
<!-- /table -->

It predicts whether an action will be accepted (94% against a 75% base rate), the failure class, and the next
context far better than the trivial predictors, and reward and return somewhat better. Using it for one-step
lookahead when acting gave 80% against 77% on 64 held-out tasks: three points, inside the noise of one evaluation
(section 2). The model is a working component with measured predictive value; an improvement of behaviour from
planning with it is not established.

## 10. Several learners in one world

Three controllers (copies of the PPO + replay checkpoint, each with its own optimiser, updated online from its own
experience) act in one persistent world for 12 simulated days, each starting with a company and a coded service.
Each run is repeated with one participant idle (the counterfactual), and once with scripted controllers in place of
the learned ones.

<!-- table:selfplay -->
### Self-play (exp_selfplay.json): world {'family': 'bazaar', 'seed': 31}, 12.0 simulated days

| run | participant | live | listed | requests served | independent revenue (cents) |
|---|---|---|---|---|---|
| all three learners | p:ada | 0 | 0 | 0 | 0 |
| all three learners | p:bea | 1 | 0 | 0 | 0 |
| all three learners | p:cy | 0 | 0 | 0 | 0 |
| counterfactual: ada idle | p:ada | 0 | 0 | 0 | 0 |
| counterfactual: ada idle | p:bea | 1 | 1 | 0 | 0 |
| counterfactual: ada idle | p:cy | 0 | 0 | 0 | 0 |
| counterfactual: bea idle | p:ada | 0 | 0 | 0 | 0 |
| counterfactual: bea idle | p:bea | 0 | 0 | 0 | 0 |
| counterfactual: bea idle | p:cy | 0 | 0 | 0 | 0 |
| counterfactual: cy idle | p:ada | 0 | 0 | 0 | 0 |
| counterfactual: cy idle | p:bea | 0 | 0 | 0 | 0 |
| counterfactual: cy idle | p:cy | 0 | 0 | 0 | 0 |
| reference: scripted control for all three (same world, same interface, no learning) | p:ada | 1 | 1 | 0 | 0 |
| reference: scripted control for all three (same world, same interface, no learning) | p:bea | 1 | 1 | 0 | 0 |
| reference: scripted control for all three (same world, same interface, no learning) | p:cy | 1 | 1 | 0 | 0 |

effect of presence (with - without): {"p:ada": {"p:bea": {"requests_served": 0, "independent_revenue": 0.0}, "p:cy": {"requests_served": 0, "independent_revenue": 0.0}}, "p:bea": {"p:ada": {"requests_served": 0, "independent_revenue": 0.0}, "p:cy": {"requests_served": 0, "independent_revenue": 0.0}}, "p:cy": {"p:ada": {"requests_served": 0, "independent_revenue": 0.0}, "p:bea": {"requests_served": 0, "independent_revenue": 0.0}}}

### Self-play (exp_selfplay_tiny7.json): world {'family': 'tiny', 'seed': 7}, 12.0 simulated days

| run | participant | live | listed | requests served | independent revenue (cents) |
|---|---|---|---|---|---|
| all three learners | p:ada | 1 | 0 | 0 | 0 |
| all three learners | p:bea | 0 | 0 | 0 | 0 |
| all three learners | p:cy | 0 | 0 | 0 | 0 |
| counterfactual: ada idle | p:ada | 0 | 0 | 0 | 0 |
| counterfactual: ada idle | p:bea | 0 | 0 | 0 | 0 |
| counterfactual: ada idle | p:cy | 0 | 0 | 0 | 0 |
| counterfactual: bea idle | p:ada | 1 | 0 | 0 | 0 |
| counterfactual: bea idle | p:bea | 0 | 0 | 0 | 0 |
| counterfactual: bea idle | p:cy | 0 | 0 | 0 | 0 |
| counterfactual: cy idle | p:ada | 1 | 0 | 0 | 0 |
| counterfactual: cy idle | p:bea | 0 | 0 | 0 | 0 |
| counterfactual: cy idle | p:cy | 0 | 0 | 0 | 0 |
| reference: scripted control for all three (same world, same interface, no learning) | p:ada | 1 | 1 | 13372 | 40116 |
| reference: scripted control for all three (same world, same interface, no learning) | p:bea | 1 | 1 | 0 | 0 |
| reference: scripted control for all three (same world, same interface, no learning) | p:cy | 1 | 1 | 8590 | 25770 |

effect of presence (with - without): {"p:ada": {"p:bea": {"requests_served": 0, "independent_revenue": 0.0}, "p:cy": {"requests_served": 0, "independent_revenue": 0.0}}, "p:bea": {"p:ada": {"requests_served": 0, "independent_revenue": 0.0}, "p:cy": {"requests_served": 0, "independent_revenue": 0.0}}, "p:cy": {"p:ada": {"requests_served": 0, "independent_revenue": 0.0}, "p:bea": {"requests_served": 0, "independent_revenue": 0.0}}}
<!-- /table -->

* The machinery works: one shared world, interleaved decisions, per-participant outcomes from authoritative state,
  online updates, counterfactual runs, no invariant violations.
* The scripted reference shows what the worlds allow. In the small world all three scripted participants deployed
  and listed a service; two of them served 13,372 and 8,590 requests (revenue $401 and $258) and the third none —
  they compete for the same demand. In the bazaar world all three listed and nobody was sent a request in 12 days.
* The learned controllers did not get that far. In both worlds at most one of the three had a service live, in one
  run of eight one listing appeared, and none served a request. Their calls failed mostly with permission and
  missing-resource errors (4–5 per participant): with other participants' companies, repositories and deployments
  in the same world, the controller's choice of which resource to name goes wrong in a way it did not in the
  single-actor tasks it was trained on (where the same checkpoint lists and operates a service in 100% of held-out
  tasks).
* Consequently there is nothing to say about learning through competition: every presence effect is zero because
  nobody traded. This is a negative result for Part 8; the controllers were never trained with other learners in
  the world, and the arena makes that gap measurable.

## 11. The Meta-Researcher (Experiments E and F)

One search (`runs/meta/search4`): four rounds, each with a pool of six proposals (three composed rule programs,
two functions written by the frozen LLM, one learned update rule), **every** proposal screened by training a fresh
2.5M-parameter controller in confinement under a 4,000-decision budget enforced by the trusted env hub, against the
champion (recurrent PPO with GAE) on the same seed, demonstrations and evaluation tasks. Then the four best screened
candidates were given more seeds. 37 confined training runs, 95,000 training decisions, 7.7 hours of run time.

<!-- table:meta -->
### Meta-Researcher (exp_e_meta.json): 23 candidates screened (22 valid), 4 taken further

champion baseline runs: screen seed 1: AULC 0.598, held-out 0.58 -> 0.58; develop seed 3: AULC 0.719, held-out 0.71 -> 0.73; develop seed 2: AULC 0.731, held-out 0.71 -> 0.75; confirm seed 4: AULC 0.772, held-out 0.83 -> 0.75; confirm seed 5: AULC 0.637, held-out 0.71 -> 0.62

| round | candidate | source | function(s) | changed lines | runs | does anything? | AULC - champion (1 seed) | picked | hypothesis / rejection |
|---|---|---|---|---|---|---|---|---|---|
| 0 | f000-c727ca72a1 | rule_space | total_loss, other | 77 | yes | yes | +0.076 |  | total_loss -> explore_when_failing {'k': 1.083}: exploration is worth most where the agent is failing; scaling the entropy bonus by the batch's failur |
| 0 | f000-ccf8ab943c | rule_space | other | 72 | yes | yes | +0.027 |  | constants {'lam': 0.952, 'vf_coef': 0.368} |
| 0 | f000-0a1e227eca | rule_space | advantages, other | 70 | yes | yes | +0.009 | yes | advantages -> delta_clip {'c': 1.22}: clipping temporal-difference errors before accumulating them bounds the influence of rare large delayed rewards  |
| 0 | f000-55a3b04410 | learned_rule | policy_loss, advantages | 83 | yes | yes | +0.020 | yes | learned update rule, generation 0: per-decision policy-gradient weights from a meta-network, learned lambda 0.960 and clip 0.159 |
| 0 | f000-8c2baf1242 | llm | replay_loss | 7 | yes | no (inert) | -0.006 |  | The current replay loss uses a simple mean-squared error for the value function, which may not account for the noisy advantages. By using a Huber loss |
| 0 | f000-adad7cd94c | llm | total_loss | 2 | yes | no (inert) | +0.038 |  | The policy is learning slowly due to noisy advantages, which are causing unstable credit assignment. By introducing a moving average of advantages ove |
| 1 | f001-34c474b142 | rule_space | advantages, value_loss | 70 | yes | yes | +0.082 |  | advantages -> optimistic {'beta': 0.618}: when an episode did better than the critic expected, the Monte-Carlo advantage is a lower bound worth trusti |
| 1 | f001-959f624e2b | rule_space | total_loss | 73 | yes | yes | +0.000 | yes | total_loss -> head_entropy {'k': 3.933}: which tool to call is the hard exploration problem, arguments mostly follow from it; putting the entropy bonu |
| 1 | f001-4ea6f1e28d | rule_space | value_loss, replay_loss | 82 | yes | yes | -0.001 |  | value_loss -> late_weighted {'k': 1.813}: values near the end of an episode are easier to learn and anchor the earlier ones; weighting them more early |
| 1 | f001-ea86aa3416 | learned_rule | policy_loss, advantages | 83 | yes | yes | -0.014 |  | learned update rule, generation 0: per-decision policy-gradient weights from a meta-network, learned lambda 0.937 and clip 0.241 |
| 1 | f001-6b1160c276 | llm | policy_loss | 2 | yes | yes | +0.049 | yes | The current policy loss function uses a symmetric clipping range around 1.0, which may limit the effective learning signal during large updates. By us |
| 2 | f002-6f3474c3d2 | rule_space | policy_loss | 71 | yes | yes | +0.058 |  | policy_loss -> rollback {'alpha': 0.564}: outside the trust region the clipped objective has zero gradient, so nothing pulls the policy back; a negati |
| 2 | f002-f2837cdb3a | rule_space | value_loss, total_loss | 73 | yes | yes | +0.059 |  | value_loss -> clipped {'c': 0.452}: limiting how far the critic may move from its acting-time estimate in one update keeps the advantages computed fro |
| 2 | f002-c9b9b462f7 | rule_space | replay_loss | 76 | yes | yes | +0.048 |  | replay_loss -> short_first {'steps': 5.68, 'coef': 1.299, 'k': 1.681}: among successful episodes the shorter ones waste fewer decisions; weighting rep |
| 2 | f002-7056aab4ff | learned_rule | policy_loss, advantages | 83 | yes | yes | +0.036 |  | learned update rule, generation 0: per-decision policy-gradient weights from a meta-network, learned lambda 0.969 and clip 0.225 |
| 2 | f002-b02ff7f482 | llm | policy_loss | 5 | yes | no (inert) | +0.040 | yes | The current policy loss function uses a symmetric clipping range around 1.0, which may limit the ability to adapt to noisy advantages. Asymmetrical cl |
| 2 | f002-a25d7ca058 | llm | advantages | 4 | yes | yes | +0.030 | yes | The current advantage estimation may be overly sensitive to noisy value estimates, especially in truncated episodes. Introducing a small amount of dis |
| 3 | f003-a53d08dcb7 | rule_space | value_loss, total_loss | 74 | yes | yes | -0.007 |  | value_loss -> huber {'delta': 0.505}: returns here are mostly zero with occasional jumps of order one; a Huber loss keeps those jumps from dominating  |
| 3 | f003-6f9221862d | rule_space | advantages, total_loss | 75 | yes | yes | -0.005 |  | advantages -> episode_centered {'k': 0.664}: subtracting part of each episode's mean advantage removes the between-episode component (task difficulty) |
| 3 | f003-2ea6bf8cd6 | rule_space | replay_loss | 76 | yes | yes | +0.026 |  | replay_loss -> short_first {'steps': 7.99, 'coef': 1.431, 'k': 1.048}: among successful episodes the shorter ones waste fewer decisions; weighting rep |
| 3 | f003-3fcaaa311a | learned_rule | policy_loss, advantages | 83 | yes | yes | +0.001 |  | learned update rule, generation 0: per-decision policy-gradient weights from a meta-network, learned lambda 0.920 and clip 0.175 |
| 3 | f003-aaee7afebc | llm | advantages | 4 | yes | yes | -0.005 | yes | The current advantage estimation may be overly sensitive to noisy value estimates, especially near the end of episodes. Introducing a small value clip |
| 3 | f003-0442ba98bc | llm | advantages | 9 | no | ? | - | yes | self-test failed: RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been free |

| source | proposed | ran | inert | mean effect | best effect |
|---|---|---|---|---|---|
| rule_space | 12 | 12 | 0 | +0.031 | +0.082 |
| learned_rule | 4 | 4 | 0 | +0.011 | +0.036 |
| llm | 7 | 6 | 3 | +0.025 | +0.049 |

inert candidates (the champion's algorithm run again): 3, effects [-0.0056, 0.0383, 0.0397], mean |effect| 0.028, max 0.040; 6 of 19 active candidates had |effect| above that maximum

| candidate | source | function(s) | decision | stage | AULC difference [95% CI] | seeds | hypothesis |
|---|---|---|---|---|---|---|---|
| f001-34c474b142 | rule_space | algo/rule.py | inconclusive | develop | +0.005 [-0.109, +0.082] | 3 | advantages -> optimistic {'beta': 0.618}: when an episode did better than the critic expec |
| f000-c727ca72a1 | rule_space | algo/rule.py | inconclusive | develop | +0.017 [-0.017, +0.076] | 3 | total_loss -> explore_when_failing {'k': 1.083}: exploration is worth most where the agent |
| f002-f2837cdb3a | rule_space | algo/rule.py | inconclusive | confirm | +0.004 [-0.073, +0.100] | 5 | value_loss -> clipped {'c': 0.452}: limiting how far the critic may move from its acting-t |
| f002-6f3474c3d2 | rule_space | algo/rule.py | inconclusive | develop | +0.003 [-0.064, +0.058] | 3 | policy_loss -> rollback {'alpha': 0.564}: outside the trust region the clipped objective h |

| round | surrogate fitted | value of surrogate picks | pool mean (random) | best of pool | valid fraction |
|---|---|---|---|---|---|
| 0 | False | +0.015 | +0.028 | +0.076 | 1.00 |
| 1 | False | +0.025 | +0.023 | +0.082 | 1.00 |
| 2 | True | +0.035 | +0.045 | +0.059 | 1.00 |
| 3 | True | -0.017 | -0.003 | +0.026 | 0.83 |

online selection (learns from its own picks): advantage over random, all rounds -0.009 [-0.013, -0.002] n=4; once fitted -0.012 [-0.014, -0.01] n=2; established=False; leave-one-out {'n': 13, 'n_valid': 12, 'spearman': -0.182}

offline replay with full feedback: advantage over random -0.009 [-0.024, -0.001] over 3 rounds; mean rank correlation predicted vs measured -0.45; established=False

* round 1: trained on 6, picked -0.000, random +0.023, best +0.082, rank correlation -0.50
* round 2: trained on 11, picked +0.044, random +0.045, best +0.059, rank correlation +0.14
* round 3: trained on 17, picked -0.006, random -0.003, best +0.026, rank correlation -1.00
<!-- /table -->

**Experiment E — can the Meta-Researcher modify an algorithm and test it?** Yes, mechanically: 24 proposals were
generated as code, 23 became separate source trees, 22 passed verification and the confined self-test and were
trained and compared with the unchanged baseline on the trusted measure. One LLM-written function did not parse
and one failed the self-test (it differentiated through a freed graph).

**Did it find a better algorithm?** No. The champion was not replaced.

* The best screened candidate ("optimistic" advantages) was +0.082 AULC at one seed and +0.005 [−0.109, +0.082]
  over three. The next three: +0.017 [−0.017, +0.076] (three seeds), +0.004 [−0.073, +0.100] (five seeds, the only
  one to pass development), +0.003 [−0.064, +0.058] (three seeds). All four verdicts: inconclusive.
* Screening at one seed is mostly noise, and the search measured that itself. Three of the LLM-written candidates
  changed nothing: one added a variable that is never used, one rewrote the replay loss although the champion
  never replays, one replaced the clipping by an "asymmetric" form that is mathematically the same function. They
  passed every check, were trained like the others, and "scored" −0.006, +0.038 and +0.040 against the champion —
  differences between two runs of the *same* algorithm. Only 6 of the 19 candidates that do change something were
  outside that range, and four of those six are the candidates that then failed to hold up over more seeds.
* The champion's own run at the screening seed (AULC 0.598) was low compared with its other seeds (0.64–0.77),
  which is why most candidates, including the inert ones, looked better than it at that seed.
* The three LLM-written candidates that did change something were small edits (a wider upper clip, a discount on
  truncated episodes, clipped temporal-difference errors): +0.049, +0.030, −0.005 at one seed, none followed up.
* The learned update rule completed one evolution-strategy generation (four evaluations, best +0.036, mean +0.011).

**Experiment F — does the learned selection model improve the search's decisions?** Not in this run.

* Online (it learns only from the two candidates it picked per round): the value of its picks was below the pool
  average by 0.009 over four rounds, and by 0.012 in the two rounds after it had been fitted.
* Replayed offline with full feedback (fitted before each round on *every* earlier outcome): −0.009 against random
  over three rounds, mean rank correlation between predicted and measured effects −0.45.
* With targets that are mostly noise there was little to learn, and two to three fitted rounds could not establish
  an advantage even if there were one. The honest reading: the meta-level learner exists and is trained on real
  outcomes; it has not been shown to help, and here it did not.

What the Meta-Researcher did contribute is on the evaluation side. A confined probe that fingerprints what a
candidate's learning code computes on fixed, varied data now identifies candidates that change nothing; it was
added after this search (which is why the inert candidates were run) and is on by default for new searches. Two
earlier versions of the probe were wrong in instructive ways — one never left the trust region, one had no
truncated episodes — and misclassified three real changes as inert; both are kept in the lab notes.

## 12. The sealed final list

`docs/experiments/final_tasks.json` (96 tasks, 12 arms, six world families including two that no training,
curriculum or selection run ever used; SHA-256 seal `8c20b6e9…`, verified on load) was written before any learning
experiment and opened once, after everything above. Which learner to test was decided on the selection split
(section 4): PPO + replay.

<!-- table:final -->
### Sealed final list (exp_final.json, seal 8c20b6e98b41): learner `exp_b_fewdemo_sil` (highest mean held-out success on the selection split: {'PPO alone': 0.7326388888888888, 'PPO + replay of own successes':)

64 of 96 sealed tasks attempted (arms ['deploy/0/coded', 'found/0/scratch', 'fund/0/scratch', 'list/0/deployed', 'repair/0/scratch', 'service/0/coded', 'service/0/deployed', 'service/0/listed']); not attempted: ['apicall/0/scratch', 'code/0/scratch', 'deploy/0/scratch', 'service/0/scratch']

| seed | before RL (greedy) | after RL (greedy) | before RL (sampled) | after RL (sampled) |
|---|---|---|---|---|
| 0 | 64% [52, 75] | 77% [65, 85] | 55% [43, 66] | 67% [55, 77] |
| 1 | 77% [65, 85] | 77% [65, 85] | 64% [52, 75] | 61% [49, 72] |
| 2 | 50% [38, 62] | 75% [63, 84] | 41% [29, 53] | 50% [38, 62] |

* after - before (greedy): +12.5 points, 95% CI [+0.0, +25.0], 192 task pairs: 27 gained, 3 lost; established=False

* after - before (sampled): +6.2 points, 95% CI [-3.6, +15.6], 192 task pairs: 28 gained, 16 lost; established=False

| world family | n | before | after |
|---|---|---|---|
| bazaar | 48 | 58% | 75% |
| cheap_compute_closed_markets | 24 | 67% | 71% |
| expensive_compute_open_distribution | 24 | 54% | 75% |
| labor_rich_fragile_infra | 24 | 71% | 75% |
| novel_needs | 24 | 67% | 88% |
| tiny | 48 | 67% | 75% |
<!-- /table -->

* On the 64 sealed tasks of the eight arms this learner was trained for, held-out success went from 63.5% (warm
  start) to 76.0% (after RL) greedy, and from 53.1% to 59.4% with sampled actions, averaged over three seeds. Tasks
  that flipped, greedy: 27 gained, 3 lost.
* The size and direction repeat the selection-split result (+11.8 there, +12.5 here). The 95% interval over seeds
  and tasks touches zero for greedy ([+0.0, +25.0]) and includes it for sampled actions ([−3.6, +15.6]): two of the
  three seeds improved by 13 and 25 points, the third (the strongest warm start) not at all.
* The gain holds in the two world families that were never seen before the final test (71% → 75% and 67% → 88%).
* The other 32 sealed tasks need LLM-written services and were not attempted by this learner. Counted as failures,
  the whole list is 42.4% → 50.7%.

The learner that includes LLM-written content (thirteen arms, section 8) on the whole sealed list, one run each:

| controller | sealed list, greedy (96 tasks) | accepted calls | by arm (8 tasks each) |
|---|---|---|---|
| cloned warm start | **70.8%** [61, 79] | 77% | 100%: found, fund, list, deploy (coded), deploy (from scratch), service (coded, deployed, listed); 25%: API call, repair; 0%: code, operate from scratch |
| after 4 RL iterations | **70.8%** [61, 79] | 86% | as above except API call 37.5%, repair 37.5%, code 12.5%, deploy from scratch 62.5% |

68 of 96 tasks in both cases. The cloned controller deployed an LLM-written service from nothing in all eight sealed
deploy tasks, in worlds of families it had never been in. Four RL iterations moved single tasks between arms and
changed nothing overall.

## 13. What it cost

| | experience | time | LLM |
|---|---|---|---|
| Experiment B, six runs | 70,856 decisions, 6,760 episodes | 2.5 h of rollouts, 17 min of GPU updates | none (embeddings only) |
| RL with LLM arms, four iterations | 8,020 decisions | 69 min of rollouts | 660 values written |
| Meta-search | 95,048 decisions in 37 confined runs | 7.7 h of run time (CPU, up to six runs at once) | 8 functions written |
| Frozen LLM, everything in this document | — | about 18 tokens per second and stream | 1,400 generations, 291,000 tokens generated, 3.1M prompt tokens; everything else served from cache |

The trainable controller has 19,667,591 parameters (all trained; the 14B language model is frozen and receives no
gradient). Wall-clock numbers are from a host shared with other tenants.
