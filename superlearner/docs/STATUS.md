# Status: implemented versus demonstrated

Branch `feat/superlearner-4` (cut from `feat/crucible-internet` 11e5613). Numbers and their caveats are in
[RESULTS.md](RESULTS.md); this page says, per requirement, what exists and what an experiment actually showed.

* **implemented** — the code exists, runs, and is covered by a test or a recorded run;
* **demonstrated** — an experiment with a control measured the claimed effect;
* **negative / inconclusive** — the experiment ran and did not show the effect.

## Acceptance criteria (Part 13)

| # | Criterion | Implemented | Experimentally demonstrated |
|---|---|---|---|
| 1 | Open-ended behaviour: plans and programs that were not enumerated as strategies | yes — actions are programs over the world's current primitives (`actions/program.py`); the controller composes any available tool with arguments, LLM-written content, drafts and its own tools (`cam/agent.py`); 13 interpreter/grounding tests | partly — recorded episodes use 19–27 distinct tools, service code and manifests written by the LLM at the controller's request, and calls to endpoints of services the agent itself had just deployed (tools that did not exist when the episode began). The heads for proposing plans, running drafts and making tools exist and execute but the trained controller did not choose them; agent-made tools were never verified in a later episode (0 in the tool library) |
| 2 | Tool reliability: discovers current capabilities, meaningful feedback on failure | yes — `CapabilityIndex`, `GroundedExecutor` (ten failure classes, schema hints, retry guard), tested | partly — every rejected call is classified (tables in RESULTS list them); retrieval puts the right tool first or second where mean-pooled embeddings put it 14th and 98th. Share of calls accepted by the world: scripted control 92–100%, cloned controller with LLM 76–89%, RL-trained controller on the eight no-LLM arms 40% (it repeats failing calls), random 29–31% |
| 3 | Genuine neural updates: trainable weights receive gradients and change | yes — 19,667,591 trained parameters; tests check gradients reach every component and that a PPO update changes the weights | yes — in all six Experiment B runs every one of the 12 updates changed the weights (L2 distance from the warm start 0.33–0.84, 0.1–0.3% of the norm); weights reset to initialisation give 0% where the trained weights give 77% |
| 4 | Real interaction: acts through Crucible's real mechanics | yes — every action is a recorded Crucible command; success is the evaluator's predicate on authoritative state; idle and random agents earn exactly 0 (tested) | yes — e.g. a live service built from nothing in 13 recorded commands, 8 of 8 held-out worlds (Experiment A) |
| 5 | Long-horizon learning: many decisions, delayed outcomes | architecture: recurrent two-timescale state, waits up to 3 days, whole-episode credit assignment, drafts and agent tools as temporal abstraction | partly — trained episodes are 3–40 decisions with outcomes up to 10 simulated days later; longer goals were not trained; a 5,000-request goal was met zero-shot from advanced starts (94% of 32) |
| 6 | Persistent learning: checkpoints preserve capability without conversation context | yes — checkpoints (weights, optimiser, memory, tools, curriculum, replay buffer), `resume`, separate restore of weights and memory (tested) | yes for weights — every evaluation in RESULTS loads a checkpoint in a new process; with the weights restored and memory reset the agent keeps its capability (77%; 70% for the run trained with memory), with memory kept and weights reset it has none (0%). Retention after learning something new: lost without rehearsal (100% → 33%), kept with it (100%) |
| 7 | Training effectiveness: at least one pre/post RL comparison, positive or negative | yes | yes, positive with qualifications — held-out success after RL, three seeds: 65% → 77% (PPO + replay, 36 tasks gained, 2 lost; interval above zero), 65% → 73% (PPO alone; interval touches zero for greedy actions). Four earlier versions of the experiment that showed a drop are documented with their causes |
| 8 | Independent evaluation: training and locked evaluation tasks separated | yes — four disjoint splits enforced by `check_split`; sealed final list (`docs/experiments/final_tasks.json`, SHA-256 `8c20b6e9…`); held-out tasks served by index (tested) | yes — the sealed list was opened once, after selection: 63.5% → 76.0% on its 64 applicable tasks (interval [+0.0, +25.0]) |
| 9 | Algorithm modification: the Meta-Researcher generates and executes a nontrivial learner-code change | yes — LLM-written functions, composed rule programs, learned-rule parameters; every candidate is a separate source tree run under confinement | yes — 24 changes generated as code (12 composed rule programs of 70–80 changed lines, 4 learned-rule programs, 8 LLM-written functions); 22 passed verification and the confined self-test and were trained. Caveat: 3 of the 6 LLM-written candidates that ran were functionally inert |
| 10 | Algorithm evaluation: a modified learner trained and compared with the unchanged baseline | yes — matched seeds, demonstrations, budget (enforced by the env hub), tasks; scores from the hub's own results | yes — 22 candidates trained and compared with the unchanged champion on the trusted measure at one seed; the four best at three to five seeds |
| 11 | Self-improvement evidence: claims backed by matched experiments and uncertainty | the comparison machinery reports bootstrap intervals and a predeclared verdict; a confined probe identifies candidates that change nothing | **no improvement established** — no candidate was promoted; the four followed-up candidates ended at +0.003 to +0.017 AULC with intervals around zero. Inert candidates measured the comparison's noise (up to +0.040 at one seed) |
| 12 | Reproducibility: configurations, seeds, versions, checkpoints, learning curves | yes — run directories and candidate directories hold all of these; `docs/REPRODUCE.md`; restartable queue | the commands in REPRODUCE are the ones used; an unplanned six-hour outage was recovered from files alone (lab notes) |
| 13 | Security: generated code cannot silently escape to unrestricted host execution | yes — world code only in Crucible's sandbox; research code under uid + rlimits + Landlock + seccomp, probed from inside; fail-closed | the probe passes on this host (`superlearner doctor`, `tests/test_meta.py`); VM-grade isolation is not available here |
| 14 | Compatibility: existing Crucible tests still pass | no Crucible file was modified | yes — 305 passed (0:12:24) on the final tree, after all experiments |
| 15 | Working software: training, evaluation and meta-search from documented commands | yes — `doctor`, `llm-serve`, `train`, `resume`, `evaluate`, `report`, `inspect-trajectory`, `lock-final`, `meta-search`, `experiment`, `jobs`; CLI test | all reported results were produced with these commands |

## Experiments (Part 12)

| Experiment | Ran | Outcome |
|---|---|---|
| A — can CAM perform real actions (build and operate a service)? | yes, 16 held-out tasks, four conditions | **building: yes** (8/8 deployed from nothing with LLM-written code; scripted control 5/8; LLM without controller 0/8). **operating to the traffic goal: no** (0/8 for controller and for scripted control) |
| B — does neural learning improve behaviour? | yes, 3 seeds × 2 algorithms × 96 held-out tasks | **yes, modestly**: +8 points (PPO), +12 (PPO + replay); repairs partly learned skills, does not acquire missing ones; strongest for the weakest warm start |
| C — does learning transfer? | yes | **zero-shot: yes** to unseen world families (75% vs 78% in familiar ones). **faster adaptation: not shown** (flat over 4,000 decisions) |
| D — does knowledge persist? | yes, RL variant and supervised variant | across checkpoints and unrelated unsuccessful RL: **yes**. After really learning a new skill: **forgotten without rehearsal (100% → 33%), kept with rehearsal** |
| E — can the Meta-Researcher modify an algorithm? | yes — 4 rounds, 23 candidates screened, 37 confined training runs | **mechanically yes, no better algorithm found**: nontrivial code changes generated, verified, executed confined, compared under matched budgets; none promoted |
| F — does meta-learning improve algorithm-search decisions? | yes — every pool screened in full so that selection can be scored | **no**: the learned selection model's picks were below the pool average (−0.009 over four rounds online, −0.009 over three rounds replayed with full feedback); too few rounds and too noisy targets to learn from |
| Part 5 — seven-way comparison | yes, 64 held-out tasks (frozen LLM alone on the first 16); memory conditions on a second run trained with memory | random 2% · frozen LLM alone 12% · LLM + grounding 23% · warm start 67% · after RL 77% · weights only 77% (70%) · memory only 0% · scripted control 100% |
| Part 2.5 — learned world model | yes | predicts acceptance (94% vs 75% base rate), failure class, next context; one-step lookahead +3 points, **not established** |
| Part 8 — several learners in one world | yes, two worlds, counterfactuals, scripted reference | **negative**: scripted participants trade and compete in the small world; the learned controllers never got a listed, used service, so no effect of competition on learning could be measured |

## What remains unproven

* **That reinforcement learning can teach this agent a skill it was not shown.** Every gain measured is on skills
  the demonstrations had partly installed. Arms at 0% stayed at 0%.
* **That the gain from RL is robust.** Three seeds; the sealed-list interval is [+0.0, +25.0]; one of three seeds
  did not improve at all. More seeds are the obvious next measurement.
* **That episodic memory or agent-made tools help.** Memory hurt the one agent tested with it; that agent had been
  cloned without memory. No tool the agent made was ever verified and reused.
* **That learning transfers as faster adaptation.** Zero-shot success carried over to unseen world families; four
  further RL iterations there changed nothing.
* **That the agent can operate what it builds.** It deploys LLM-written services; none reached the traffic goal,
  under the controller or under scripted control.
* **That RL works with LLM-written content in the loop.** Four iterations, no change.
* **That planning with the learned world model improves behaviour.** +3 points on 64 tasks is inside the noise.
* **That several learners in one world learn from each other.** They did not trade.
* **That the Meta-Researcher can find a better learning algorithm, or that its learned selection helps.** In one
  search of 23 candidates it found none and its selection was no better than random. The search's inner task is
  small (a 2.5M-parameter controller, 4,000 decisions per run) and screening at one seed is dominated by noise;
  whether a longer search with several seeds per candidate would find something is unknown.
* **Recursive self-improvement.** Nothing was promoted, so nothing compounded.
* **Long horizons.** Episodes of 3–40 decisions; no task of hundreds of decisions was trained.
* **Isolation stronger than uid + rlimits + Landlock + seccomp** is not available on this host.
