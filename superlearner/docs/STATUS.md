# Status: implemented versus demonstrated

Branch `feat/superlearner-4` (cut from `feat/crucible-internet` 11e5613). Numbers and their caveats are in
[RESULTS.md](RESULTS.md); this page says, per requirement, what exists and what an experiment actually showed.

* **implemented** — the code exists, runs, and is covered by a test or a recorded run;
* **demonstrated** — an experiment with a control measured the claimed effect;
* **negative / inconclusive** — the experiment ran and did not show the effect.

## Acceptance criteria (Part 13)

| # | Criterion | Implemented | Experimentally demonstrated |
|---|---|---|---|
| 1 | Open-ended behaviour: plans and programs that were not enumerated as strategies | yes — actions are programs over the world's current primitives (`actions/program.py`); the controller composes any available tool with arguments, LLM-written content, drafts and its own tools (`cam/agent.py`); 13 interpreter/grounding tests | RESULT_1 |
| 2 | Tool reliability: discovers current capabilities, meaningful feedback on failure | yes — `CapabilityIndex`, `GroundedExecutor` (ten failure classes, schema hints, retry guard), tested | RESULT_2 |
| 3 | Genuine neural updates: trainable weights receive gradients and change | yes — 19,667,591 trained parameters; tests check gradients reach every component and that a PPO update changes the weights | RESULT_3 |
| 4 | Real interaction: acts through Crucible's real mechanics | yes — every action is a recorded Crucible command; success is the evaluator's predicate on authoritative state; idle and random agents earn exactly 0 (tested) | RESULT_4 |
| 5 | Long-horizon learning: many decisions, delayed outcomes | architecture: recurrent two-timescale state, waits up to 3 days, whole-episode credit assignment, drafts and agent tools as temporal abstraction | partly — trained episodes are 3–40 decisions with outcomes up to 10 simulated days later; longer goals were not trained |
| 6 | Persistent learning: checkpoints preserve capability without conversation context | yes — checkpoints (weights, optimiser, memory, tools, curriculum), `resume`, separate restore of weights and memory (tested) | RESULT_6 |
| 7 | Training effectiveness: at least one pre/post RL comparison, positive or negative | yes | RESULT_7 |
| 8 | Independent evaluation: training and locked evaluation tasks separated | yes — four disjoint splits enforced by `check_split`; sealed final list (`docs/experiments/final_tasks.json`, SHA-256 `8c20b6e9…`); held-out tasks served by index (tested) | RESULT_8 |
| 9 | Algorithm modification: the Meta-Researcher generates and executes a nontrivial learner-code change | yes — LLM-written functions, composed rule programs, learned-rule parameters; every candidate is a separate source tree run under confinement | RESULT_9 |
| 10 | Algorithm evaluation: a modified learner trained and compared with the unchanged baseline | yes — matched seeds, demonstrations, budget (enforced by the env hub), tasks; scores from the hub's own results | RESULT_10 |
| 11 | Self-improvement evidence: claims backed by matched experiments and uncertainty | the comparison machinery reports bootstrap intervals and a predeclared verdict | RESULT_11 |
| 12 | Reproducibility: configurations, seeds, versions, checkpoints, learning curves | yes — run directories and candidate directories hold all of these; `docs/REPRODUCE.md` | commands in REPRODUCE were the ones used for the reported runs |
| 13 | Security: generated code cannot silently escape to unrestricted host execution | yes — world code only in Crucible's sandbox; research code under uid + rlimits + Landlock + seccomp, probed from inside; fail-closed | the probe passes on this host (`superlearner doctor`, `tests/test_meta.py`); VM-grade isolation is not available here |
| 14 | Compatibility: existing Crucible tests still pass | no Crucible file was modified | RESULT_14 |
| 15 | Working software: training, evaluation and meta-search from documented commands | yes — `doctor`, `llm-serve`, `train`, `resume`, `evaluate`, `report`, `inspect-trajectory`, `lock-final`, `meta-search`, `experiment`; CLI test | all reported results were produced with these commands |

## Experiments (Part 12)

| Experiment | Ran | Outcome |
|---|---|---|
| A — can CAM perform real actions (build and operate a service)? | RESULT_A_RAN | RESULT_A |
| B — does neural learning improve behaviour? | RESULT_B_RAN | RESULT_B |
| C — does learning transfer? | RESULT_C_RAN | RESULT_C |
| D — does knowledge persist? | RESULT_D_RAN | RESULT_D |
| E — can the Meta-Researcher modify an algorithm? | RESULT_E_RAN | RESULT_E |
| F — does meta-learning improve algorithm-search decisions? | RESULT_F_RAN | RESULT_F |
| Part 5 — seven-way comparison | RESULT_P5_RAN | RESULT_P5 |
| Part 2.5 — learned world model | RESULT_WM_RAN | RESULT_WM |
| Part 8 — several learners in one world | RESULT_SP_RAN | RESULT_SP |

## What remains unproven

UNPROVEN_PLACEHOLDER
