# Architecture

Superlearner Lab is a separate package (`superlearner/`) that uses Crucible 3.0 (`crucible/`) as a library. No
Crucible file was changed. Measured results are in [RESULTS.md](RESULTS.md).

```
                         TRUSTED                                   CANDIDATE-EDITABLE LEARNER
 Crucible world  <--  world/core.py  SessionCore  <---- protocol ---->  cam/agent.py  CAMAgent
 (kernel, services)   (clock, budget, Evaluator)   observe/tools/        |  grounding/ (knowledge, executor, tools)
                      world/hub.py   env hub        execute/wait/        |  actions/program.py (interpreter)
                      world/tasks.py splits         end_decision/summary |  llm/ (frozen model client)
                      evalkit/metrics.py                                 v
                                                              cam/model.py  CAMController (trained)
                                                              cam/memory.py working + episodic memory
                                                              cam/worldmodel.py (optional)
                                                                         |
                               train/rollout.py  workers --> episodes --> algo/ (rule.py, ppo.py, sil.py)
                               train/trainer.py  curriculum, checkpoints, evaluation, reports

 meta/researcher.py:  diagnose -> propose (meta/proposals.py) -> select (meta/surrogate.py) -> candidate tree
                      (meta/workspace.py) -> confined run (meta/isolate.py + meta/candidate_run.py) against the
                      env hub -> trusted comparison (meta/compare.py) -> ledger (meta/ledger.py) -> learn -> repeat
```

## 1. The trusted side

`world/core.py SessionCore` owns one Crucible world, the actor's identity, the clock and the task's `Evaluator`.
A learner can only do what an in-world actor can do, through six calls: `info`, `observe`, `tools`,
`execute(tool, args)`, `wait(seconds)`, `end_decision`, plus `summary` at the end. It never receives the world
object, an evaluation task's seed, or the evaluator.

* **Time.** Every executed command costs its tool's duration (at least 1 s); every decision costs 60 s of think
  time; `wait` lets time pass. An episode terminates on verified success or when the task horizon has elapsed, and
  is truncated when the decision budget is used up.
* **Reward.** `Evaluator`: r = (Φ(sₜ) − Φ(sₜ₋₁)) + [first verified success]. Φ is the task contract's progress
  metric on authoritative state, in [0, 1]. Potential differences cannot be farmed by cycles. There is no reward
  for valid calls, activity, code volume or claims; an idle agent and a uniform-random agent earn exactly zero
  (tested).
* **Remote form.** `world/hub.py` serves the same protocol over a unix socket, one forked server per connection.
  It writes every episode's evaluator summary to a results file that the learner cannot write, stamps it with the
  number of training decisions served so far, refuses new training tasks when the experience budget is spent, and
  starts held-out tasks only by index.

`world/tasks.py` defines the task families (outcomes on authoritative state: `found`, `fund`, `code`, `apicall`,
`deploy`, `list`, `service`, `revenue`, `repair`), scripted start states, and four disjoint splits
(`train`, `curriculum`, `select`, `final`) by seed range and world family; `check_split` enforces them and the
final list is sealed with SHA-256. `evalkit/metrics.py` holds the evaluator-only metrics (for example
`sl_solver_quality` runs the agent's repository in the sandbox on hidden inputs).

## 2. Grounding (`grounding/`)

| Piece | What it does |
|---|---|
| `CapabilityIndex` | the tools the actor can call *now*, discovered from the world; semantic retrieval over whole tool descriptions; `consumers(tool)` = tools that operate on the same kind of thing |
| `Knowledge` | identifiers and numbers the actor has seen (by JSON key), documents it has read, stable aliases (`ACCT1`) for prompts |
| `GroundedExecutor` | alias resolution → tool lookup → schema validation → retry guard → the world; classifies every failure |
| `ToolLibrary` | tools the agent made: macros generalised from its own successful calls, or Python run in the world's sandbox; provenance, version, dependencies, permission footprint, outcome statistics; `verified` only after a successful run in a later episode |

Failure classes: `syntax`, `unknown_tool`, `unresolved_reference`, `nonexistent_resource`, `permission`,
`prerequisite`, `rate_limited`, `runtime`, `loop_blocked`, `budget`. The executor never repairs a call; it says what
was wrong (the argument, the schema fragment, identifiers of that kind already seen). An identical call that has
failed twice is not sent a third time until the actor has changed something. Strategic failure (valid actions, no
outcome) is decided at episode level by the evaluator.

Two findings from building this layer explain part of the earlier agents' failures:

* Crucible's prompt renderer keeps only the **first line** of each tool description. `cloud.deploy` documents the
  required manifest on its following lines, so the earlier LLM agents were never shown it. `tool_doc` here renders
  whole descriptions.
* Mean-pooled hidden states of the frozen model retrieve tools poorly. Prompt-based last-token embeddings from a
  late layer ("… means in one word:", layer −4, centred) are much better: on a probe, `cloud.deploy` for "commit
  files for a deployable service" moved from rank 14 to 2, `market.list` for "have an active market listing" from
  98 to 1.

## 3. The frozen language model (`llm/`)

One server process owns Qwen3-14B (`llm/server.py`: batched generation in one thread, embeddings in another,
persistent sqlite cache). Its weights are loaded with `requires_grad=False`; `fingerprint()` records them.
It supplies:

1. **text embeddings** (512-d) of tool descriptions, goals, documents and normalised failure messages — the
   controller's semantic input;
2. **argument values** the controller asks it to write (`gen` options): code, text, structured objects, from a
   document the controller chose. Objects of named texts (files) are written entry by entry; structured entries
   may use the model's reasoning mode; after a failure the next attempt is shown the previous value and what went
   wrong (error-specific repair);
3. **draft programs** (`propose`): a free-form plan and a multi-step program in the action language.

Greedy decoding per (prompt, variant) with a cache, so repeated situations cost nothing. Usage is counted
separately as tokens generated by the model and tokens served from cache.

Serving. Waiting greedy requests are answered in one forward pass even when their length limits and reasoning
flags differ (every row stops at its own limit), grouped by the size of the limit so that a short tool-call
proposal does not wait behind a reasoned manifest; a batch is also limited by estimated tokens, and cached GPU
memory is released after each batch, because the same card runs the training updates. Clients wait and reconnect
when the server is restarted (requests are idempotent: answers are cached by content).

## 4. The trainable controller (`cam/model.py`)

19,667,591 parameters, all trained (`superlearner doctor` prints the count).

```
context items (goal, history, last failure, documents, draft steps, plan, memories) ─┐
observation features (512 hashed numeric features) + 16 auxiliary numbers ───────────┴─> 6-layer transformer encoder
     c_t ──> fast GRU (every decision) ──┬─> z_t
             slow GRU (ticks on time jumps) ┘
z_t × head items   ──> head logits      (every available tool, wait, ask the LLM, run/skip draft, define/use own tool)
z_t, head, argument, earlier arguments × option items ──> option logits (one choice per argument)
z_t ──> value ensemble (4 members; mean = critic, spread = uncertainty), success probability
```

Everything scored is an *item*: a frozen text embedding (through a trained projection), hashed symbolic tokens and
numeric features. There are no per-tool or per-strategy output units, so tools the controller has never seen
(services it discovers, tools it makes) are scored like any other, and probabilities are always over what can
actually be chosen (dynamic candidate sets, masked softmax).

**What the frozen model and the controller exchange.** Frozen → trained: embeddings of every text item, proposed
steps and suggested argument values (as items marked with their source). Trained → frozen: *when* to ask for a
draft, *which* tool call needs LLM-written content, *which document* that content should be based on, which
sampled variant to use, and whether a draft step is followed, edited (argument by argument), skipped or ignored.

## 5. Memory

| Kind | Where | Lifetime | Reset in experiments by |
|---|---|---|---|
| working | `WorkingMemory`: steps, successful calls, plan, draft, subgoals | one episode | every episode start |
| episodic | `EpisodicMemory` (goal embedding → outcome, tools that worked, literal arguments) and the `ToolLibrary` | across episodes and worlds; saved in checkpoints | `evaluate --reset-memory` |
| neural | controller weights; recurrent state (within an episode) | checkpoints | `evaluate --reset-weights` |

## 6. Learning (`algo/`, `train/`)

`algo/rule.py` is the learning rule (advantages, policy / value / entropy / auxiliary losses). `algo/ppo.py` is
recurrent PPO over whole episodes: likelihoods are recomputed by unrolling the controller from each episode's
start (no stale hidden states), so `evaluate(...)["logp"]` equals the acting-time log-probability (tested to
2e-4). `algo/sil.py` adds a replay buffer of the agent's own verified-successful episodes. `bc_update` is
behaviour cloning for the warm start. `algo/base.py` is the interface for further algorithms.

`train/rollout.py` runs CPU worker processes; `train/trainer.py` runs the loop
experience → episodes → update → changed weights → checkpoint, with periodic greedy evaluation on tasks that are
never trained on; `train/curriculum.py` chooses task arms from competence, uncertainty, prerequisites and failure
patterns. Details: [TRAINING.md](TRAINING.md).

## 7. The Meta-Researcher (`meta/`)

See [META_RESEARCHER.md](META_RESEARCHER.md). In short: candidates are complete source trees in which only the
learner files (`cam/`, `algo/`, `train/curriculum.py`) may differ; they run confined (`meta/isolate.py`) and get
experience and scores only from the env hub; comparisons use the hub's own results.

## Departures from the suggested architecture

* **One controller, pointer outputs, instead of a separate action generator and selector.** A single network
  generates actions choice by choice over dynamic items; LLM proposals enter as items rather than as a fixed list
  to rank.
* **The trainable pathway into LLM-written content is choice, not gradient.** The controller decides what the
  frozen model is asked and what it sees; no gradient flows into the frozen model and no adapter is trained on it.
  A learned bias on the frozen model's token logits was considered and not built (see LIMITATIONS).
* **Whole-episode backpropagation** instead of truncated segments, because episodes here are at most a few dozen
  decisions thanks to temporal abstraction (waits, drafts, agent tools).
* **Scores for algorithm comparison are computed by the environment side**, not by the learner, so that candidate
  code cannot influence them.
