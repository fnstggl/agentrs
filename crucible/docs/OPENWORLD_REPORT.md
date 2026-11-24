# Crucible 4.0 "OpenWorld": final research report

Crucible 4.0 extends the Crucible 3.0 event-sourced synthetic internet (`11e5613`) into a platform for measuring how fast
learning systems acquire new, transferable capabilities from experience. This report states what was built, what was
measured, and what was not established. Commands for every number are in [REPRODUCE_OPENWORLD.md](REPRODUCE_OPENWORLD.md);
the component contracts are in [OPENWORLD_ARCHITECTURE.md](OPENWORLD_ARCHITECTURE.md).

## 1. Summary

**The environment works as an instrument. None of the learners tested has yet learned to succeed in unfamiliar worlds.**

* All **16 acceptance tests (A-P) pass**, and the **1,057-test regression suite passes**, including everything from 3.0.
* On the **sealed test set** (72 tasks in 9 families no learner trained on, opened once), no trained learner beat a
  simple scripted experimenter. The trained model-based learner was statistically indistinguishable from it; recurrent
  PPO was significantly worse, and worse than doing nothing.
* The learning that was measured is real but small. PPO's weights improved by +0.07 to +0.09 on fresh tasks (significant
  in 3 of 3 seeds), which brought it from below idle to about idle. The model-based learner's score comes from fitting a
  model *within* each episode; training its weights across episodes added nothing detectable.
* A dedicated **positive-control learner failed its criteria**, which were written before any validation score was seen
  (section 3.2). A neural process trained across 1,800 worlds beat the scripted experimenter on new worlds of the training
  families (+0.184 [+0.129, +0.240]), but its gain over its own untrained weights was not distinguishable from zero
  (+0.048 [-0.006, +0.101]), and on held-out families training made it worse (-0.065 [-0.118, -0.014]). No sealed set
  was opened for it.
* A frozen **Qwen3-4B** with a corrected interface scored the same as idle: in all six full-length facility runs it read
  the gauges every turn and **never set a lever**. The existing **CAM 2.0** checkpoint reached 0 of 8 open-economy goals.
* The red team found one critical and 13 high-severity holes in the 3.0 reward system. 23 of 24 findings are closed.
* Worlds stayed live and consistent for up to **365 simulated days** (7 runs, about 5.3 million events): no process
  failures, no invariant violations, exact replay.
* **Nothing in the simulation is calibrated against real-world data.** Of 55 behavioural and economic mechanisms in the
  fidelity register, 37 are plausible hypotheses, 18 are known to be unrealistic and none is calibrated. Results here
  say nothing yet about performance outside Crucible.

### The decisive question

> Can Crucible reliably distinguish an agent that becomes better at learning unfamiliar consequential problems from one
> that merely executes familiar strategies?

**Partly established.** The protocols correctly attribute gains on purpose-built learners that each have exactly one
kind of learning: within-episode only, weights only, a memoriser that replays training trajectories, and none
(`tests/test_research_protocols.py`). On real learners they separate architectures, timescales and transfer categories
with paired intervals. What is still missing is a **positive control on the real task families**: no learner tested so
far actually gets better at learning them, so the instrument has not yet been seen detecting that on hard tasks. One
purpose-built attempt failed (section 3.2). The instrument did refuse to credit it: a large win over the scripted
baseline was not accepted as learning, because the same procedure with untrained weights scored almost as well.

## 2. What was built

| layer | what it adds |
|---|---|
| Hidden mechanisms (`mechanisms/`) | three independently written engines (structural causal models, an agent-based market, executable technical systems), 25 families split into train, validation and test by causal relation, a facility module operated through ordinary tools, a bit-identical simulator, a private evaluator, validity checks |
| Usefulness substrate (`usefulness/`) | 10 procedural families of digital work whose hidden rules differ per world, composed needs, a discovery and usefulness ledger; value is established by executing code on hidden instances, never by an LLM's opinion |
| Population (`economy/behavior.py`) | five structurally different customer models, segments, latent needs and derived demand, pay-on-acceptance |
| Market dynamics (`actors/entry.py`) | founders who decide from public signals, enter, fail and exit; a continuous invariant monitor; long-run liveness measures |
| Invention trials (`evaluation/invention.py`) | matched forks that place consequences by causal order, with removal arms |
| Research platform (`research/`) | an architecture-neutral learner contract, learning curves, three timescales, parametric internalisation, four transfer categories, equal-budget comparison, sealed sets, cost accounting, failure diagnosis |
| Baselines | hold, random, scripted experimenters, recurrent PPO, model-based, evolutionary search, frozen LLM, CAM 2.0 |
| Self-play (`selfplay/`) | six interaction modes, persistent leagues, held-out opponents, cross-play, a collusion audit |
| Assurance | an 81-probe red team, an authorisation-gated real-world bridge (off by default), a fidelity register |
| Interface and scale | searchable tool discovery, dry-run validation, process-parallel worlds, asynchronous rollout workers, benchmarks |

3.0 behaviour is preserved: with the discovery tools switched off and no new configuration, honest worlds evolve
byte-identically to 3.0 under every red-team fix (seven compared scenarios, deterministic cost model). Two bug fixes deliberately change trajectories where they apply: a crash in the worker
process after a developer finishes an issue, and the deterministic cost of a fresh sandbox worker.

## 3. Results

Scores are normalised per task: **0 = leaving every lever alone, 1 = a reference policy found by search on the true
mechanism**. Success means reaching the task's public objective. Intervals are 95%.

### 3.1 Sealed test (one access; 72 tasks, 9 held-out families, 3 learner seeds)

| learner | score | score, compositional | score, structural | score, other engine | objective reached |
|---|---|---|---|---|---|
| scripted experimenter | **+0.101** [+0.022, +0.175] | +0.141 | +0.222 | -0.010 | 15% [11, 21] |
| model-based (trained) | **+0.052** [-0.042, +0.145] | +0.189 | +0.363 | -0.248 | 18% [14, 24] |
| idle | -0.001 [-0.007, +0.006] | -0.003 | -0.007 | +0.006 | 0% |
| recurrent PPO (trained) | **-0.217** [-0.317, -0.123] | -0.041 | +0.076 | -0.524 | 1% [0, 3] |
| random settings | -0.931 [-1.064, -0.796] | -1.100 | -0.580 | -1.111 | 0% |

Against the scripted experimenter (paired, Holm-corrected): model-based -0.048 [-0.115, +0.021], not distinguishable; PPO
-0.317 [-0.407, -0.233], worse. **The technical-systems engine, which no learner trained on, defeated every learner.**
Model-based did best on the structural families and worse than idle on the other engine.

### 3.2 Learning from experience

Equal budget (120,000 environment steps, 3 seeds), validation tasks from the training families:

| learner | final score | area under the curve | objective reached |
|---|---|---|---|
| model-based | +0.240 [+0.160, +0.324] | +0.212 | 19% |
| scripted experimenter (does not learn) | +0.096 [+0.025, +0.176] | +0.096 | 8% |
| recurrent PPO | -0.046 [-0.089, -0.002] | -0.073 | 0% |

| timescale | recurrent PPO | model-based |
|---|---|---|
| **Fast adaptation** (context intact vs cleared) | present, but the policy depends on context rather than improving within the episode | present: +0.22 and +0.25 |
| **Parametric learning** (frozen, fresh context, trained vs untrained) | **present**: +0.078 and +0.088 | **not detected** after correction |
| **Meta-learning** (faster adaptation after training) | **not detected** | **not detected** |

Parametric internalisation (8-step protocol, 60,000 steps, fresh tasks, weights hashed before and after):

| learner | seed 0 | seed 1 | seed 2 | verdict |
|---|---|---|---|---|
| PPO | +0.085 [+0.045, +0.130] | +0.065 [+0.028, +0.111] | +0.069 [+0.034, +0.108] | internalised, 3 of 3 |
| model-based | +0.004 [-0.070, +0.072] | +0.049 [-0.020, +0.121] | -0.017 [-0.083, +0.053] | no gain, 3 of 3 |

Transfer (60,000 source steps; zero-shot gain of the trained checkpoint over the untrained one):

| target | PPO | model-based |
|---|---|---|
| new seeds, same families | +0.071 [+0.047, +0.100] | +0.025 [-0.019, +0.073] |
| compositional | +0.035 [+0.027, +0.044] | -0.021 [-0.173, +0.083] |
| structural | +0.168 [+0.115, +0.224] | **-0.070 [-0.128, -0.019]** (negative) |
| other engine | -0.009 [-0.135, +0.124] | +0.087 [-0.013, +0.183] |

PPO's transfer gains start from an untrained policy that loses money, so they mostly reflect learning to do less harm:
its absolute scores stay near or below idle.

#### Positive-control attempt: `meta_np` (failed)

`meta_np` is a transformer neural process trained across worlds to predict the revenue and cost of untried settings from
the settings tried so far in the episode. Its output is a correction to an exact Bayesian regression, and it starts at
zero, so **an untrained `meta_np` is that regression with the same search procedure**. Each of 3 seeds trained for 300,000
environment steps (about 37,700 updates) on 1,800 worlds of the six training families.

The criteria were recorded before any validation score was seen ([log](experiments/ow_meta_np_log.md)): the trained
learner, averaged over seeds, must beat the scripted experimenter (C1) **and** its own untrained weights (C2), with
paired 95% intervals above zero, before a new sealed set may be opened. Beating `model_based` (C3) was reported either way.

| trained `meta_np` minus | validation, training families (72 tasks) | held-out families (48 tasks) |
|---|---|---|
| *(trained `meta_np` score)* | *+0.292* | *+0.176* |
| scripted experimenter (C1) | **+0.184 [+0.129, +0.240]**, passes | -0.047 [-0.131, +0.037] |
| its own untrained weights (C2) | **+0.048 [-0.006, +0.101]**, fails | **-0.065 [-0.118, -0.014]**, worse |
| `model_based`, untrained (C3) | +0.050 [-0.002, +0.102] | -0.066 [-0.141, +0.009] |
| `model_based`, trained (C3) | +0.052 [+0.003, +0.106] | -0.052 [-0.107, -0.000] |

**Verdict: failed. C2 and C3 do not hold, so no sealed set was opened.** The win over the scripted experimenter is mostly
due to the regression and search procedure, which need no training.

An exploratory breakdown, chosen after seeing these results and not part of the verdict, shows why the average is small.
Trained minus untrained on validation is **+0.174 [+0.113, +0.236]** on the agent-based market families and
**-0.078 [-0.141, -0.020]** on the structural causal families. On held-out families it is +0.012 [+0.008, +0.017] (market),
-0.113 [-0.196, -0.032] (causal) and -0.047 [-0.170, +0.058] (technical systems). The weights learned something useful
about market worlds and lost ground elsewhere. Any claim about that needs a fresh test.

On development tasks, one pilot checkpoint predicted held-out settings better than the regression, the mean of measured
settings and the nearest measured setting: lowest error in 18 of 18 family and context-size cells, best ranking in 15,
and lowest regret when picking a setting in 16 ([data](experiments/ow_meta_np_predcheck.json)). Better prediction did not
reliably become better decisions.

On synthetic worlds, after 8,000 updates, the correction learned a response curve shared across worlds (error 2% of the
regression's) and a pure two-lever interaction (16%), but not a saturating curve whose size differs per lever and per
world (101%) ([data](experiments/ow_meta_np_synthetic.json)).

Disclosures: the design was tuned on development seeds across 8 training configurations and about 35 variants of its
acting settings, all listed in the log. Validation and held-out tasks had already been used to compare earlier learners,
though never to tune this one. Because this learner has now been seen on them, a successor needs fresh validation seeds.

### 3.3 Frozen language models and CAM

| policy | tasks | mean score | objective reached | what it did |
|---|---|---|---|---|
| reference (privileged) | 6 | +0.991 | 6 of 6 | feasibility witness |
| scripted experimenter | 6 | +0.127 | 0 of 6 | experiments, then commits |
| **Qwen3-4B, corrected interface** | 6 | -0.001 | 0 of 6 | read gauges every turn; never set a lever |
| Qwen3-1.7B, corrected interface | 3 | -0.046 | 0 of 3 | mostly filed hypotheses and made invalid calls |
| idle | 6 | -0.001 | 0 of 6 | |
| random settings | 6 | -1.351 | 0 of 6 | |

The interface is no longer the confound: 0 prompts truncated and every tool visible with its full signature in all
Qwen3-4B runs. In the open-economy objective, Qwen3-4B called `search.query` 192 times out of 192 and built nothing; a
scripted control that induces the hidden format from published examples met the goal by day 4.

**CAM 2.0** (MiniWoB-trained checkpoint, frozen Qwen3-1.7B, 65M controller), 8 open-economy goals: **0 of 8**, the same as
its untrained weights and idle; the scripted Builder reached 6 of 8. The trained CAM chose the first tool in the list
(`ads.create_campaign`) on every step with arguments the schema rejected. It was never trained on Crucible, so this
measures how it carries over, not what it could learn here.

### 3.4 Self-play

Units are percent of demand income per tick. Learners are small independent stochastic-approximation and policy-gradient
controllers; 560 matches were checked for conservation with 0 violations, and 12 of 12 trained pairs replayed exactly as
recorded tool calls in a kernel world.

* **Competition (`abm_duopoly`).** What a learner learns depends on its training opponent: own-opponent minus
  other-opponent payoff is +3.41 [+2.12, +5.05]. The effect comes from the passive opponent (+3.13 [+1.42, +5.46]); against
  the aggressive one it is not distinguishable (+0.27 [-0.60, +1.07]).
* **Tacit collusion appeared without being programmed.** The audit classified 8 of 12 trained duopoly pairs as raising
  both profits by extracting more from customers (5% higher price per unit, 13% fewer units), with no communication
  channel. This is recorded, not forbidden.
* **Cooperation.** In `abm_complement`, co-trained pairs beat pairs of separately trained learners: +1.61 [+1.20, +2.11].
  In `tech_handoff`, a trained pair far outperforms a learner with an untrained partner (+5.22 [+3.80, +6.84]), **but
  separately trained learners do just as well** (-0.08 [-1.20, +0.94]), so that task needs both halves done, not
  co-adaptation.
* **Mixed motive.** Learners settled close to joint restraint in `abm_commons` (0.87 of the way from mutual over-use to
  restraint) and above scripted restraint in `tech_shared_pool` (1.35).
* **Cross-play.** Gaps were small except in `abm_complement` (+0.71 [+0.35, +1.07]), the one clear case of fitting a partner.
* **Held-out family.** On `abm_duopoly_fx`, the policy-gradient learners reached +6.87 and the SPSA learners learned
  nothing (+0.01, the same as defaults).

### 3.5 Long-horizon worlds

| family | days | seeds | events (each) | process failures | invariant violations | replay | sector collapses (unrecovered) | firms entered / exited | final week vs median |
|---|---|---|---|---|---|---|---|---|---|
| `ow_market` | 365 | 3 | 1.14-1.30M | 0 | 0 | exact | 5 (0) | 33 / 11 | 1.01-1.05 |
| `ow_frontier` | 180 | 2 | 0.58-0.62M | 0 | 0 | exact | 5 (0) | 19 / 3 | 1.00-1.01 |
| `ow_heldout` | 90 | 2 | 0.24-0.27M | 0 | 0 | exact | 1 (0) | 16 / 2 | 1.05 |

Each run includes a forced bankruptcy of the largest firm. Output did not decay in these families; the original 3.0
120-day configuration was not re-run. These runs have no focal agent.
Every collapse recovered, so these runs did not exercise a market that stays dead.

### 3.6 Invention and second-order effects (matched forks, 3 seeds, 30 days)

* **Autoscaling (`open_tight_compute`): established under the simulator's assumptions, in 3 of 3 seeds.** The artifact
  was adopted and imitated, created a capacity-exhaustion loss absent from the control, and moved provider prices and
  investment. Another participant published a response in 3 of 3 seeds; firms adopted it in 1 of 3. Withdrawing only the original did **not** remove the new loss, because imitators kept
  it alive. Withdrawing the whole lineage did (0.9-1.3 events a day against 15-21).
* **Peak pricing:** adopted in 3 of 3 seeds; a new loss mechanism in 1 of 3.
* **Better software waking latent demand (`ow_frontier`):** a demand change in 1 of 3 seeds. Weak.

All inventors are scripted controls, so this shows what the world's mechanics support, not that a learner finds it.

### 3.7 Red team, validity, performance

* **Red team:** 81 probes, each a real attack using only in-world powers. Before: 51 held, 16 false successes, 9 integrity
  breaks, 3 leaks, 1 false failure, 1 not applicable. After: **79 held, 1 open, 1 not applicable**.
* **Validity:** 288 of 300 first draws of generated mechanisms pass (96%); rejected draws are redrawn and recorded.
* **Performance** (`--quick` only, on a shared machine with an 11.5-core quota): 4,400 ledger transactions/s, 26,000
  replayed events/s, 10 simulated days/s across 8 parallel small worlds (1.2 for one), 7.5 real-browser actions/s, 40 MB
  for a world with 100,000 consumers.

## 4. Acceptance tests

Each result is what pytest reported, with measured numbers in [`experiments/acceptance/`](experiments/acceptance) and the
caveats in [ACCEPTANCE.md](ACCEPTANCE.md).

| test | result | what it establishes, and its main limit |
|---|---|---|
| A Real software creation | pass | a deployed app with persistent state, found and used by another agent; the builder is scripted |
| B Multiple viable strategies | pass | selling software and operating a facility both meet one objective in 3 of 4 worlds; **horizon raised from 10 to 20 days after the first attempt failed**; software still loses money in world 3 |
| C Unknown mechanism discovery | pass | informed beats deprived by +1.57 on causal-model families; **does not hold on technical systems** |
| D Structural transfer | pass | the protocol measures it; the measured effect was not distinguishable from zero |
| E Autonomous competition | pass | policies depend on the opponent; small controllers |
| F Cooperation | pass | removing a partner changes the outcome in 3 worlds; n=3 |
| G Invention, second-order effects | pass | matched forks, one seed in the test (three in 3.6); scripted inventor |
| H Long-term sustainability | pass | 28 days in the test; recorded runs up to 365 days also checked |
| I Genuine customer value | pass | garbage service earned nothing despite a better listing and price; customers are simulated |
| J Reward-hacking resistance | pass | known attacks fail; this cannot show that no attack exists |
| K Architecture-neutral evaluation | pass | same interface and budget across architectures |
| L Parametric learning | pass | +0.36 on **easy hand-made** tasks; far smaller on the benchmark families |
| M Reproducibility | pass | replay, regeneration, forks, anchors; fork equality needs the deterministic cost model |
| N Clean installation | pass | core and dev extras in an empty environment |
| O Independent generalization | pass | establishes the procedure; final numbers are in 3.1 |
| P Security | pass | process sandbox holds against the probes; **not VM-grade**, which is refused |

## 5. Status

| status | items |
|---|---|
| **Fully implemented and end-to-end tested** | kernel extensions, mechanism engines and facility, usefulness substrate, behaviour models, market entry, invariant monitor, long-run harness, discovery tools, diagnostics, protocols, sealed sets, self-play arena and league, red-team framework, parallel worlds and rollouts |
| **Experimentally validated (under the simulator's assumptions)** | long-horizon liveness and accounting; capacity second-order chain; informed-vs-deprived discovery on causal-model families; reward-hack resistance against 81 probes; opponent-dependent learning; collusion detection |
| **Partially validated** | transfer measurement (no learner shows strong transfer); invention classes other than capacity; cooperation (depends on family); real-world bridge (loopback only, HTTPS path untested); performance (quick suite only) |
| **Inconclusive** | whether the instrument detects meta-learning on real families (no learner exhibits it); which architecture is better on the sealed set between model-based and scripted |
| **Failed / negative results** | no trained learner beat the scripted experimenter on the sealed set; the `meta_np` positive control failed its pre-declared criteria, and training made it worse on held-out families; meta-learning not detected; model-based gained nothing from weight training; frozen Qwen3-4B never acted on the facility; CAM 2.0 reached 0 of 8; discovery procedure fails on technical systems |
| **Not attempted** | calibration against real data; real-world pilots; VM-grade isolation (unavailable on this host); frontier-scale models; hundreds of concurrent LLM agents; self-play inside the full open economy (self-play runs on hidden-mechanism facilities); the full 30-day benchmark table |

## 6. Limitations that matter most

1. **The learners are modest.** Budgets of 60,000-300,000 steps with small networks on shared hardware cannot show what a
   strong architecture would do. The results describe these baselines, not the difficulty ceiling of the benchmark.
2. **No external validity.** Customers, firms and mechanisms are models. Success here does not predict success in reality.
3. **Most competence in the world is scripted.** Every successful inventor, builder and experimenter is a scripted control.
4. **Remaining exploits.** Two formally unrelated colluders can still trade across the netting window; a paid one-cent order
   buys a rating; a padded doorway page can outrank thin content; a store-level log rewrite needs an external anchor
   to detect. All are listed in [LIMITATIONS.md](LIMITATIONS.md).
5. **The sandbox is process-level.** Landlock, seccomp and an unprivileged uid share the host kernel, and workers share one uid.
6. **Sample sizes are small** for several claims (3 seeds for invention classes, n=2-3 worlds in cross-play).
7. **The sealed set is spent.** Any new comparison needs a new set.

## 7. Recommended next steps

1. **Establish a positive control.** The first attempt (`meta_np`, section 3.2) failed. A successor must beat its own
   untrained weights as well as the scripted experimenter, on fresh validation seeds and then a new sealed set. The
   obvious lead is to keep the gain on market worlds without the loss on causal ones, for example by letting each
   episode decide how far to trust the learned correction from its own prediction errors. Without a positive control,
   the instrument's sensitivity to real meta-learning stays unproven.
2. **Train CAM on Crucible** through the existing adapter; its tool-selection failure is the first thing to fix.
3. **Give language models a scaffold or a stronger model** to separate "cannot act" from "cannot discover".
4. **Calibrate one mechanism family** against public data, starting with adoption and retention.
5. **Run self-play inside the open economy**, where competition can involve real software.
