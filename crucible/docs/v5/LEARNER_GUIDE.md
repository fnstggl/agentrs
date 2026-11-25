# Connecting a learner to Crucible

Crucible is an environment. It does not contain, assume or wait for any particular learning system (Clarification 1).
Any learner - a scripted agent, a statistical learner, a small neural network, a language-model agent, a future CAM
version - connects through one public interface, `crucible.experience`, without any change to the environment. This
guide is the complete contract. It is proven by `examples/experience_external_adapter.py`, an adapter written only
against this interface, and by the gate test `tests/v5/test_gate_SP_11_6.py`, which checks that running it imports
nothing else from Crucible and changes nothing in it.

Run everything from `crucible/` with `PYTHONPATH=src`.

## 1. Ten lines

```python
from crucible.experience import make

env = make(1, "mech:scm_additive", seed=0, reward_mode="observation_only")   # any tier, same interface
agent = env.agents()[0]
for step in range(100):
    obs = env.observe(agent)                       # Observation (section 2)
    action = my_policy(obs, obs["tools"])          # {"tool": "<host>.<endpoint>", "args": {...}} or None
    if action is not None:
        result = env.execute(agent, action)        # {ok, t, tool, cost_s, result | error, action_seq}
    env.advance(3600)                              # time passes only here
    r = env.reward(agent)                          # 0.0 in observation_only, always
outcome = env.verify()                             # evaluator side: never give this to the learner
```

`crucible.experience.run_learner(env, learner, steps, store=...)` is this loop with schema checks, experience
storage and update scheduling, for any object with `act(obs, tools)` and optionally `observe(reward, result,
next_obs, done)` and `update()`.

## 2. What the learner observes

Every tier returns the same `Observation` (a dict of JSON-native values, `crucible.experience.observation`):

| field | contents |
|---|---|
| `t` | world time, integer seconds |
| `agent` | the observing principal (`p:<handle>`) |
| `numbers` | exact scalars by dotted path: money in integer cents (`bank.balances.acct_000017`), readings as full floats (`rig.seats.r1.readings.le76`), counts |
| `arrays` | numeric vectors (`rig.r1.readings`, `rig.r1.settings`, lever bounds); `semantic.<sha256 ref>` embeddings of readable v5 artifacts when workstream D's encoders are present |
| `text` | natural-language items as `"<source path>: <text>"` (briefs, messages, posts) |
| `artifacts` | references to readable content: `{ref, kind, owner?, title?, t, event}`; v5 artifacts are `sha256:<hex>`, 4.0 content objects `repo:..`, `commit:..`, `post:..`, `mail:..`, `url:..`, `deployment:..`, `listing:..` |
| `tools` | MCP-shaped tool specs: `name`, `description`, `inputSchema` (JSON schema), `annotations` (`readOnlyHint`, `costSeconds`, `host`, `builtin`, `version`, `category`) |
| `events` | every event visible to the agent since its previous observation: `{seq, t, type, actor, payload, cause}` |
| `entities` | ids the agent's view mentions, their kind and relation to the agent (`{id, kind, relations: [{rel, to}], attrs}`) |
| `resources` | the agent's own usage: actions, time used, actions this hour and the hourly limit, balance, accounts, rejected actions |
| `analytics` | public aggregates anyone could compute (counts of public events, facility facts) |
| `last_result` | the result of the agent's previous action |
| `missing` | what the agent does not know (yet): pending outcomes of its experiments and self-goals with their due times, hidden mechanisms present in this world, truncation notices |
| `cursor` | the event seq up to which `events` were delivered |
| `objective` | the agent's public objective text, if it has one |

Rules a learner can rely on:

* **Exact numbers.** Quantities are never rendered into text to be parsed back. Money is integer cents.
* **No silent truncation.** `observe` delivers at most `max_events` events; if more are waiting it says so in
  `missing` and sets `cursor` to the last delivered seq. `observe_since(agent, cursor)` returns every visible event
  after any cursor (pages of 100,000; the returned cursor is where the page ended).
* **Provenance without privileged answers.** An event caused by one of the agent's *own* actions carries
  `cause = {action_seq, tool, t, delay_s, via}`: `via` is `direct` (same transaction), `scheduled` (a process the
  action scheduled, transitively) or, in Tier 1, `settings_in_force` (a facility report produced under settings the
  agent chose). `execute` returns the same `action_seq`, so a learner can join consequences to its actions across
  weeks. Consequences of other participants' actions never carry a cause.
* **No evaluator data.** No reward, goal progress, objective status, hidden mechanism parameter or other
  participant's private state appears in any observation, result, tool list or event (tested by
  `crucible.experience.leakage`, Gate R).
* `validate_observation(obs)` checks the schema; `kinds_present(obs)` says which of the 13 kinds of MASTER_SPEC 11.1
  an observation actually carries.

## 3. Acting

One action shape in every tier: `{"tool": "<host>.<endpoint>", "args": {...}}` (optionally `idempotency_key`).
Discover tools with `env.tools(agent)` (or `obs["tools"]`); `inputSchema` is the exact JSON schema the world
validates against. Tools fall in 13 categories (`annotations.category`, `crucible.experience.TOOL_CATEGORIES`):
browser, search, code_execution, file_creation, program_deployment, communication, financial, research, experiments,
organization, content_creation, tool_composition, agent_services. Services other participants deploy appear as new
tools once the agent learns their host (`world.describe`), with `category = agent_services`.

`execute` never moves the clock. Its result reports the action's duration `cost_s`; the driver decides how much
time passes with `advance(seconds)`. This is a semi-Markov decision process in every tier. Failures inside the world
are results (`ok: false`, `error: {code, message}`), never exceptions. Permissions, prices, quotas, rate limits and
visibility are the world's own.

## 4. Reward modes

`reward_mode` is chosen when the environment is made and applies to every adapter:

| mode | `reward(agent)` | what else the learner gets |
|---|---|---|
| `legacy_shaped_reward` | the 4.0 contracts byte for byte (net income per tick in Tier 1; potential-shaped verified progress plus success bonus with an objective; change of own balance without one) | everything 4.0 gave, including Gym `info["objective"]` |
| `verified_terminal_outcome` | 1.0 once when the independent verifier confirms the objective, else 0.0 | only `success`/`failed` labels |
| `observation_only` | 0.0 always (Gymnasium needs a scalar: it is **not a learning signal**) | world consequences only |

In observation-only mode a learner evaluates itself: what it values is its own business (MASTER_SPEC 11.3). The
reference learner, for instance, trains on the change of the balance it observes plus a self-supervised prediction
loss. Independent verification keeps running on the evaluator side (`env.verify()`); authorised outcome labels can be
supplied separately for calibration by the operator, never through the observation.

The 4.0 environments take the same switch: `MechVecEnv`, `KernelMechEnv`, `SelfPlayEnv`, `KernelSelfPlayEnv`,
`CrucibleEnv` (Gymnasium), `CrucibleParallelEnv` (PettingZoo) and `scale.rollout.WorldEnv` accept
`reward_mode=...`; the default is legacy and is byte-identical to Crucible 4.0 (`leakage.legacy_check`). In
observation-only mode the numeric observation's reward slot (scalars[0]) is 0 while revenue, cost, funds and time
(scalars[1:5]) are unchanged; Gym/PettingZoo infos have no `objective` and episodes end only by truncation.

## 5. Worlds

```python
make(1, "mech:<family>", seed)        # a 4.0 hidden-mechanism task (validate=False: unvalidated training task)
make(1, "rapid:<family>", seed)       # a v5 Tier 1 family (crucible.rapid, workstream H)
make(2, "<domain>[:<config>]", seed)  # a v5 Tier 2 domain (crucible.domains, workstream I); also "d5_<domain>_<config>"
make(3, "<profile>", seed)            # a v5 civilisation (crucible.civ.generate, workstream J)
make(0, "<4.0 world family>", seed, agents=[...])   # any 4.0 family or WorldSpec (KernelExperienceEnv)
```

A tier whose package has not landed raises `TierUnavailable`; nothing falls back to another world type.
`python -m crucible.experience tiers` lists what this checkout has. Kernel worlds take `agents=[handles]` (several
learners in one world), `endowment`, `objectives` (a `GoalContract` or any `begin/step_reward/status` object per
agent), `extra_modules`. The Tier 1 facility uses exactly the tools and principals of its kernel twin (`rig.*`,
`bank.accounts`, `p:agent`), so a learner moves between the batched simulator and the kernel-hosted facility
without change (tested tick for tick in `tests/test_v5_experience_core.py`).

## 6. Experience, checkpoints, parallel worlds

* `ExperienceStore(dir)` - append-only shards with a content hash per record, a hash-chained manifest and
  content-addressed blobs for repeated tool lists; `iterate(agent=..., env={...}, reward_mode=...)` verifies every
  hash. `make_record(...)` is the standard record (`crucible-exp/1`).
* `crucible.experience.reference.ReferenceLearner` - the reference neural learner: `act`, `observe`, `update`,
  `learn_from_records(store.iterate())`, `save`/`load` (exact: weights, optimiser, RNG states), `params_blob` /
  `load_params` for workers.
* `VecExperience(envs)` steps several environments in one process; `FarmRun(learner, worlds, n_workers=...)` runs
  them in worker processes through the unchanged `crucible.scale.RolloutFarm`, stores the experience, updates the
  learner and publishes new parameters (`make_world`, `make_policy` are the farm hooks).

## 7. Self-generated experiments

Kernel worlds made through `crucible.experience` contain the `exp` module: `exp.register {hypothesis, measure,
prediction, deadline_in_s}` where `measure` names any read-only tool the agent may call and a selection of its
result; at the deadline the world measures again, as the agent, and records `exp.resolved`. `exp.result` returns the
evidence (baseline, samples, measured value, whether the prediction held, actions taken meanwhile). Nothing is ever
paid for an experiment, easy or hard.

## 8. Existing learner contracts (unchanged)

* **Numeric learners** (`crucible.research.learner.Learner`, `EnvSpec` with 61 floats in, 12 out) keep running on
  `MechVecEnv`, `KernelMechEnv`, `research.protocols` and `learners.training` exactly as in 4.0. The Tier 1 experience
  environment exposes its numeric view as `env.numeric()` (same reward mode). CAM 7.0's `Cam7Learner` uses this
  contract and reads scalars[1..4] (revenue, cost, funds, time) and ignores the reward argument, so it runs unchanged
  in every mode; learners that read scalars[0] (`probe_exploit`, `hill_climb`) see 0 there in observation-only mode by
  design. Families registered externally with `mechanisms.base.mech_family` (as CAM 7's `worlds.py` does) are
  available through `make(1, "mech:<family>", ...)` without any change.
* **Tool-world policies** (`crucible.agents.base.Policy`: `reset`, `act(obs, tools)`, `observe_result`) run under
  `Runner`/`AsyncRunner` as before; `ReferenceLearner` implements the same three methods.
* **Gymnasium / PettingZoo** wrappers keep their observation and action spaces; `reward_mode` is a new keyword.

## 9. How a future CAM version connects

A CAM version (for example the CAM Unified contract v2 `EnvAdapter`) maps one to one onto this interface:

| CAM Unified EnvAdapter | crucible.experience |
|---|---|
| `reset(seeds)` | `make(tier, name, seed, reward_mode)` per environment |
| `observe()` | `env.observe(agent)` |
| `available_tools()` -> `ToolSchema(name, description, parameters, cost_hint, duration_hint)` | `env.tools(agent)`: `inputSchema` -> parameters, `annotations.costSeconds / 86400` -> duration_hint |
| `execute(actions)` -> `ExecutionResult(ok, observations, error, duration, started_t, finished_t)` | `env.execute(agent, {"tool", "args"})`: `cost_s / 86400` -> duration; the clock does not move |
| `collect_events(since)` (lossless) | `env.observe_since(agent, cursor)` |
| `advance_time(days)` | `env.advance(days * 86400)` |
| `goals()` | `obs["objective"]` |
| `GoalEvaluator.evaluate_goal` (never fed to the learner) | `env.verify()` |
| experience / replay | `ExperienceStore` |

`examples/experience_external_adapter.py` is this mapping as working code (batched over several worlds, with a small
external learner and experience storage). To connect a new learner:

1. write an adapter like that file in the learner's own repository, importing only `crucible.experience`;
2. choose `reward_mode` (observation-only if the learner evaluates itself);
3. drive it with your own loop or `run_learner`; store experience with `ExperienceStore` if you want Crucible's
   format; checkpoint your own learner.

No Crucible file changes. `tests/v5/test_gate_SP_11_6.py` re-checks this on every run: the adapter's imports, a hash
of every source file of the environment before and after, and the identity of every public attribute.

## 10. Commands

```bash
PYTHONPATH=src python -m crucible.experience tiers
PYTHONPATH=src python -m crucible.experience run --tier 1 --name mech:scm_additive --seed 0 --steps 300 \
    --reward-mode observation_only --store /tmp/exp --checkpoint /tmp/ref.pt --out /tmp/run.json
PYTHONPATH=src python -m pytest -q tests/v5/test_gate_R.py tests/v5/test_gate_SP_11_6.py
```

The `run` command is reproducible: the same command gives the same experience-store digest, learner parameter hash
and world digest (Gate S.8).
