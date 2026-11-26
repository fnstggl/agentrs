# Connecting a learner to Crucible

Crucible is an environment. It does not contain, assume or wait for any particular learning system (Clarification 1).
Any learner - a scripted agent, a statistical learner, a small neural network, a language-model agent, a future CAM
version - connects through one public interface, `crucible.experience`, without any change to the environment.
Crucible provides experience; the learner decides what deserves reinforcement; independent evidence (the evaluator
side) establishes whether learning and objective achievement occurred (Clarification 2).

This guide is the complete contract. It is proven by two programs written only against it:
`examples/experience_external_adapter.py` (the CAM Unified `EnvAdapter` shape, Gate SP-11.6) and
`examples/external_learner/curiosity_learner.py` (an architecture-neutral learner that computes its own reward,
Gate OO.6). Run everything from `crucible/` with `PYTHONPATH=src`.

## 1. Ten lines

```python
from crucible.experience import make

env = make(1, "rapid:net_routing", seed=0)         # any tier, same interface; reward_mode defaults to observation_only
agent = env.agents()[0]
for step in range(100):
    obs = env.observe(agent)                       # Observation (section 2)
    action = my_policy(obs, obs["tools"])          # {"tool": "<host>.<endpoint>", "args": {...}} or None
    if action is not None:
        result = env.execute(agent, action)        # {ok, t, tool, cost_s, result | error, action_seq}
    env.advance(3600)                              # time passes only here
    r = env.reward(agent)                          # 0.0 in observation_only, always: not a learning signal
outcome = env.verify()                             # evaluator side: never give this to the learner
```

`run_learner(env, learner, steps, store=...)` is this loop with schema checks, experience storage and update
scheduling, for any object with `act(obs, tools)` and optionally `observe(reward, result, next_obs, done)` and
`update()`. `run_learners(env, {agent: learner}, steps)` drives several learners in one world (each acts on its own
observation; Tier 1 rapid team worlds run their actions as one joint tick).

## 2. What the learner observes

Every tier returns the same `Observation` (a dict of JSON-native values, `crucible.experience.observation`):

| field | contents |
|---|---|
| `t` | world time, integer **simulated seconds** in every tier and adapter (`TIME_UNIT = "s"`, also `resources.time_unit` and `describe()["time_unit"]`) |
| `agent` | the observing principal (`p:<handle>`; `a0`, `a1`, ... in Tier 1 rapid worlds) |
| `numbers` | exact scalars by dotted path: money in integer cents (`bank.balances.acct_000017`), readings as full floats (`rig.seats.r1.readings.le76`), counts, `world.time_s`, and `world.tick_s` where the world ticks |
| `arrays` | numeric vectors (gauge vectors, lever bounds); `semantic.<sha256 ref>` embeddings of readable v5 artifacts when the encoders are present |
| `text` | natural-language items as `"<source path>: <text>"` (briefs, messages, posts) |
| `artifacts` | references to readable content: `{ref, kind, owner?, title?, t, event}` |
| `tools` | MCP-shaped tool specs: `name`, `description`, `inputSchema` (JSON schema), `annotations` (`readOnlyHint`, `costSeconds`, `host`, `builtin`, `version`, `category`) |
| `events` | every event visible to the agent since its previous observation: `{seq, t, type, actor, payload, cause}` |
| `entities` | `{id, kind, relations: [{rel, to}], attrs}` (a rapid message becomes `{id: "message:<n>", kind: "message", relations: [{rel: "from", to}], attrs: {content}}`) |
| `resources` | the agent's own usage: actions, time used, rate limits, money (`balance_cents`, or `cash_cents` in rapid worlds), rejected actions, `cost_s_total`, `cost_cents_total` (rapid) |
| `analytics` | public aggregates anyone could compute |
| `last_result` | the result of the agent's previous action (in a rapid team world: of its last joint tick) |
| `missing` | what the agent does not know (yet): pending outcomes with due times, hidden mechanisms, truncation notices, and - in observation-only worlds - that evaluator data are not observable |
| `cursor` | the event seq up to which `events` were delivered |
| `objective` | the agent's public objective text, if it has one (none at all is a normal state: section 7) |

Rules a learner can rely on:

* **Exact numbers.** Quantities are never rendered into text to be parsed back. Money is integer cents.
* **No silent truncation.** `observe` delivers at most `max_events` events and says in `missing` how many wait;
  `observe_since(agent, cursor)` returns every visible event after any cursor.
* **Each event once, only if visible.** Kernel worlds use the world's own visibility (`World.visible`); Tier 1 rapid
  worlds use the rapid world's rule (a message is seen by its sender and its recipient only), with one global `seq`.
* **Provenance without privileged answers.** An event caused by one of the agent's *own* actions carries
  `cause = {action_seq, tool, t, delay_s, via}`: `direct` (same transaction), `scheduled` (a process the action
  scheduled, transitively), in Tier 1 `settings_in_force` (a mechanism report produced under settings the agent
  chose) or `in_force` (a rapid tick observation after the agent's latest effectful action). `execute` returns the
  same `action_seq`, so consequences can be joined to actions across weeks (Gate OO.7 retrieves 31- and 40-day
  delayed outcomes after 45 unobserved days). Consequences of other participants' actions never carry a cause.
* **No evaluator data** - section 4 says exactly what that means and how it is tested.
* `validate_observation(obs)` checks the schema; `kinds_present(obs)` says which of the 13 kinds of MASTER_SPEC 11.1
  an observation actually carries.

## 3. Acting

One action shape in every tier: `{"tool": "<host>.<endpoint>", "args": {...}}` (optionally `idempotency_key`).
Discover tools with `env.tools(agent)` (or `obs["tools"]`); `inputSchema` is the exact JSON schema the world
validates against. Tools fall in 13 categories (`annotations.category`): browser, search, code_execution,
file_creation, program_deployment, communication, financial, research, experiments, organization, content_creation,
tool_composition, agent_services. Services other participants deploy appear as new tools once the agent learns their
host (`world.describe`). Tier 1 rapid worlds expose their action kinds as `rapid.<kind>` tools.

`execute` never moves the clock in kernel and mechanism worlds: it reports the action's duration `cost_s` and the
driver advances time. A Tier 1 rapid world advances in ticks (`tick_s` seconds, default 3600): with one participant
an action runs its tick at once and that tick counts against the next `advance`; in team worlds actions are queued
(`result.queued`) and run together when every participant has acted or time advances - the real result then arrives
as `last_result` and as an `action.result` event. Failures inside the world are results (`ok: false, error: {code,
message}`), never exceptions. Permissions, prices, quotas, rate limits and visibility are the world's own.

## 4. Reward modes and evaluator data

`reward_mode` is chosen when the environment is made. **The default everywhere is `observation_only`**
(`check_mode(None)`): `make`, every adapter, `MechVecEnv`, `KernelMechEnv`, `SelfPlayEnv`, `KernelSelfPlayEnv`,
Gymnasium `CrucibleEnv` and `ExperienceGymEnv`, PettingZoo `CrucibleParallelEnv` and `ExperienceParallelEnv`,
`scale.rollout.WorldEnv`, the HTTP and MCP servers (`--reward-mode`), the AGENTRS CAM 2.0 and superlearner adapters,
`ReferenceLearner`, the research protocols, the training loop and the self-play harness (`reward_mode=` on every entry
point). Gate OO.1 checks every environment family of every tier (Tier 1 rapid and mechanism families, every Tier 2 domain configuration and Tier 3 profile present, the 4.0 world families) and the constructors and entry points, default-constructed; Gate OO.9 checks each public entry point and that an explicit selection is recorded.

| mode | `reward(agent)` | what else the learner gets |
|---|---|---|
| `observation_only` (default) | 0.0 always (Gymnasium needs a scalar: it is **not a learning signal**) | world consequences only; evaluator tools and fields hidden |
| `verified_terminal_outcome` (explicit) | 1.0 once when the independent verifier confirms the objective at the terminal outcome, else 0.0 | only `success`/`failed` labels; evaluator data hidden as above |
| `legacy_shaped_reward` (explicit) | the 4.0 contracts byte for byte | everything 4.0 gave (the historical baseline, OO.8) |

**What "no evaluator data" means.** Some 4.0/5.0 modules computed evaluator quantities inside participant tools.
`crucible.core.evaluator_only` is the one registry of them, and the kernel enforces it for every participant in a
world whose spec says `evaluator_access: "evaluator"` (every learner-facing constructor builds its world that way
unless legacy is named; a raw `World.create` keeps the 4.0 contract):

* evaluator tools are not listed, not described and refused (`forbidden`): `goals.check_now` (an on-demand
  goal-progress oracle), `use.report` and `use.probe` (the hidden usefulness grader); a module marks its own with
  `endpoint(..., evaluator_only=True)`;
* evaluator fields are removed from tool results, events and module summaries: the relatedness verdicts
  (`independent*`) of `caps.*` and `pkg.*`, the mechanism's creator-defined objective quantity `score` in `rig.*`
  (and in the Tier 1 mechanism adapter's observations and `SelfPlayEnv` infos);
* evaluator metrics (integrity-netted revenue, MRR, independent customers, hidden-grader quality, the mechanism's
  score, evaluator baselines: 25 in all) cannot appear in a participant-authored predicate - self-goals, charter
  conditions, commitment milestones and contract deliverables refuse them, so no institution or agreement can be
  turned into a probe of the evaluator; `exp` refuses evaluator tools as measures and measures what the agent sees;
* a Tier 1 rapid `query spec` answer does not carry the generated world's identity (family, seed, category,
  evaluation split);
* the science world's replication reports keep their statistical verdict (`consistent`/`refuted`: a test on the
  reporter's own readings) but lose `related_to_author`, `independent` and `independent_consistent` (the evaluator's
  relatedness analysis) in `sci.*` results and `sci.replication` events; evaluator-only metrics are the union of this
  registry and the lists other modules keep (`goals.metrics.EVALUATOR_ONLY_METRICS`, `commerce.observables`);
* the filters apply to every call made **as** a participant, not only to calls a participant makes itself: a
  deferred workflow job (`wf.run_later`), a trigger, a flow, and a simulated participant acting through tools from a
  module process receive participant views (before wave 3 a deferred job returned the `score` a direct `rig.read`
  strips);
* the verifier lives on the evaluator side. In one process Python cannot stop a learner from reading the object it
  drives, so evaluated runs that need isolation host the world in its own process:
  `part, ev = crucible.experience.host({"tier": 1, "name": ..., "seed": ...})` gives the learner `part` (the
  participant protocol only; anything else is refused by the world's process) and the evaluator `ev.verify()`.

Legitimate world observations stay: revenue, costs, balances, readings, requests, purchases, analytics a participant
could compute. Leakage is tested by information-flow probes (`crucible.experience.probes` and
`crucible.experience.probes_tiers`, Gates R, OO.2, OO.14, OO.16): twelve probe kinds (reward scalar, key name,
canary, exact value, free text, regression across perturbed worlds, counterfactual worlds, predicate oracle,
termination, tool sweep, nested call, identity) try to recover verifier values from everything a participant reaches
on ten surfaces (including a Tier 3 civilisation), and per tier on five channels: observation, auxiliary/info fields
(the Gymnasium wrapper's `info`), tool responses (every read-only tool, called with identifiers harvested from what
the participant has seen), event metadata (`observe_since`) and free text. Before this design they recovered `independent_revenue` exactly through
`goals.check_now` (21 queries) and the mechanism's objective quantity through `info["score"]`; a positive control
re-runs them on the 4.0 contract and must find leaks of at least three kinds.

In observation-only mode a learner evaluates itself; what it values is its own business (MASTER_SPEC 11.3). The
reference learner trains on the change of the money it observes (`bank.balances.*`, else `resources.cash_cents`)
plus a self-supervised prediction loss; the external example uses prediction-error curiosity. Verified outcome labels
may supervise evaluation and meta-learning on the operator's side, never as handcrafted intermediate rewards. The
self-play harness computes an individual's own signal from its own observation (revenue minus cost) outside legacy
mode and refuses objectives that need privileged data (another seat's private net, the mechanism's score).

**Verified outcome labels** reach a learner only through `crucible.experience.LabelChannel(evaluator,
authorised_by=..., purpose="evaluation" | "meta_learning" | "calibration")`, opened on the evaluator side (an
environment or a `RemoteEvaluator`): `issue()` asks the independent verifier and returns hash-chained
`crucible-label/1` records (optionally appended to a JSONL file). Nothing in Crucible opens a channel by default,
labels are never part of an observation or experience record, and issuing them changes nothing the learner observes
or receives (Gate OO.12 compares twin runs).

## 5. Worlds

```python
make(1, "mech:<family>", seed)        # a 4.0 hidden-mechanism task (validate=False: unvalidated training task)
make(1, "rapid:<family>", seed)       # a v5 Tier 1 family (crucible.rapid); one or several participants
make(2, "<domain>[:<config>]", seed)  # a v5 Tier 2 domain (crucible.domains.base.Domain); also "d5_<domain>_<config>"
make(3, "<profile>", seed)            # a v5 civilisation (crucible.civ.generate)
make(0, "<4.0 world family>", seed, agents=[...])   # any 4.0 family or WorldSpec (KernelExperienceEnv)
```

A tier whose package has not landed raises `TierUnavailable`; nothing falls back to another world type.

* **Tier 2** consumes the canonical domain contract `crucible.domains.base.Domain` (decision D-17): the world is
  `domains.base.spec(domain, config, seed, overrides)`, the learner occupies `Domain.participant_principal(world)`
  (else a new participant), its objective is `Domain.objective(world, principal, difficulty)` (public text in
  `obs["objective"]`), and `Domain.evaluate(world, principal, objective)` is the evaluator in `verify()`. The domain's
  scripted `reference` is never reachable. The path is tested with a minimal conforming domain in
  `tests/test_v5_experience_oo.py`; no v5 domain is registered in this checkout yet (`crucible.domains.base.REGISTRY`
  is empty: the domains become configurations of the civilisation generator, D-17).
* **Tier 3** expects of the civilisation generator (`crucible.experience.registry.CIV_ENTRY_POINT`):
  `crucible.civ.generate(profile: str, seed: int, *, population: str = "development", overrides: dict | None = None)
  -> World` and `crucible.civ.PROFILES`. The registry passes `overrides = {"params": {"evaluator_access": ...}}` and
  expects them merged into the spec before `World.create` (a generator that ignores them is adopted with a runtime
  override); optionally `crucible.civ.participant_seats(world)` and `crucible.civ.evaluate(world, principal)`. The
  learner-side modules `goals` (self-set goals) and `exp` (self-generated experiments) are added to the
  civilisation's spec (`registry.civ_world`: `crucible.civ.spec(...)` plus those modules, with `generate`'s world-id
  rule); without them the world is `crucible.civ.generate(...)` itself. A civilisation has no objective of its own:
  an operator assigns one with `env.set_objective(agent, GoalContract(...))`. Tier 2 domains become configurations of
  the same generator (D-17).
* Kernel worlds take `agents=[handles]`, `endowment`, `objectives` (a `GoalContract` or any
  `begin/step_reward/status` object per agent), `extra_modules`.

## 6. Experience, checkpoints, parallel worlds, external surfaces

* `ExperienceStore(dir)` - append-only shards with a content hash per record, a hash-chained manifest, a HEAD anchor
  (a truncated tail is detected), content-addressed blobs for repeated tool lists (only at the dedupe paths each line
  lists: content a participant wrote is stored verbatim), one writer at a time (`writer.lock`; read with
  `readonly=True`) and crash-safe sealing. Records (`crucible-exp/1`, `make_record`) carry `done` and `episode`, so
  offline learning never chains transitions across an episode reset.
* `ReferenceLearner` - `act`, `observe`, `update` (GAE bootstraps V(s_{t+1}) from the next observation),
  `learn_from_records` (recurrent state reset whenever the agent, world or episode changes), `save`/`load` (weights,
  optimiser, RNG, and the recurrent state with the buffered and in-flight transitions: a mid-episode resume continues
  exactly; `weights_only` loading), `params_blob`/`load_params`.
* `VecExperience(envs)` steps environments in one process; `FarmRun(learner, worlds, n_workers=...)` runs them in
  worker processes through `crucible.scale.RolloutFarm`; its concurrency is measured from worker-side timestamps and
  bounded by the number of distinct worker processes.
* `ExperienceGymEnv(tier, name, seed)` / `ExperienceParallelEnv(...)` - Gymnasium and PettingZoo over any tier:
  observation `Dict(t, numbers (float64, slot table in info["numbers_keys"]), numbers_mask, json)`, where `json` is
  the whole observation, lossless; actions are `{"tool", "args"}` dicts or their JSON; reward 0.0 in the default
  mode; `terminated` is never outcome-based. The 4.0 `api.gym_env` / `api.pettingzoo_env` (text observations over the
  WorldServer) remain for kernel worlds.

## 7. Operation without an objective, self-generated experiments

No objective is a normal state: `obs["objective"]` is None, the reward is 0.0, and `verify()` still reports
evaluator-side measures (invariants, digest, the agent's cash, its organisations' independent revenue). Gate OO.5 runs
Tier 1 worlds for hundreds of ticks across episodes and a 4.0 kernel world for 90 simulated days within configured
limits (memory growth, wall time, events held in memory).

Kernel worlds made through `crucible.experience` (Tiers 0, 2 and 3) contain the `exp` module: `exp.register
{hypothesis, measure, prediction, deadline_in_s}` where `measure` names a read-only tool the agent may call (evaluator
tools are refused) and a selection of its result as the agent receives it; at the deadline the world measures again
and records `exp.resolved`. The Tier 1 adapters offer the same three tools (`crucible.experience.selfexp`) over the
numbers of the agent's own observation (`measure = {"number": "<key of obs['numbers']>"}`), resolved privately at the
deadline and linked to the registering action. Nothing is ever paid for an experiment, easy or hard. `goals.propose` self-goals accept only metrics
a participant can observe and report met/missed at their due time (`goals.mine`).

## 8. Existing learner contracts and migration

* **Numeric learners** (`research.learner.Learner`, `EnvSpec`: 61 floats in, 12 out) run on `MechVecEnv`,
  `KernelMechEnv`, `research.protocols` and `learners.training`. In the default mode the reward and the reward slot
  (scalars[0]) are 0.0; revenue, cost, funds and time (scalars[1:5]) are unchanged. Learners that read the reward or
  scalars[0] (`probe_exploit`, `hill_climb`, `ppo_gru`, the self-play `pg`/`spsa`/`climber`) learn nothing there by
  design. **To reproduce a 4.0 result name `reward_mode="legacy_shaped_reward"`** on the environment or protocol call
  (byte-identical to 4.0: `leakage.legacy_check`, Gates R.6 and OO.8); the 4.0 test suite does exactly that.
  Self-play leagues saved before reward modes existed reload in the mode they ran in (legacy).
* **Tool-world policies** (`agents.base.Policy`) run under `Runner`/`AsyncRunner` as before; 4.0 world families,
  scripted agents and recorded logs are unchanged (a raw world keeps the 4.0 evaluator-access contract).
* **AGENTRS superlearner**: families `crucible/<goal>` are observation-only (no native invalid-call penalty, no
  termination on the verifier's outcome, `goal()` without the evaluator's metric name and target, `evaluate_goal`
  reports the verified label only); `crucible-legacy/<goal>` are the 4.0 adapter.

## 9. How a future CAM version connects

| CAM Unified EnvAdapter | crucible.experience |
|---|---|
| `reset(seeds)` | `make(tier, name, seed)` per environment (observation_only unless named) |
| `observe()` | `env.observe(agent)` |
| `available_tools()` -> `ToolSchema(name, description, parameters, cost_hint, duration_hint)` | `env.tools(agent)`: `inputSchema` -> parameters, `annotations.costSeconds / 86400` -> duration_hint |
| `execute(actions)` -> `ExecutionResult(ok, observations, error, duration, cost, ...)` | `env.execute(agent, {"tool", "args"})`: `cost_s / 86400` -> duration |
| `collect_events(since)` (lossless) | `env.observe_since(agent, cursor)` |
| `Observation.caused_by` | `event["cause"]["action_seq"]` (own actions only) |
| `advance_time(days)` | `env.advance(round(days * 86400))` |
| `goals()` -> `GoalSpec` | `obs["objective"]` (public text) |
| `GoalEvaluator.evaluate_goal` (never fed to the learner) | `env.verify()`, or `ev.verify()` of a hosted world |
| experience / replay | `ExperienceStore` |

Compatibility gaps found by reading the CAM 7 checkout (read only, never imported):

* `superlearner/unified/runtime/rollout.py` and `cam7/experiments/exp1b.py`, `exp2.py` build `MechVecEnv(tasks,
  spec)` with the default: they now run observation-only. `Cam7Learner` reads scalars[1..4] and is unaffected; a CAM
  experiment that reads the reward must pass `reward_mode="legacy_shaped_reward"` to reproduce its historical runs.
* `ExecutionResult.cost` (money): kernel and mechanism `execute` results carry no monetary cost field - money shows
  in the observation (`bank.balances.*`, `resources.balance_cents`) and in bank events; rapid results carry
  `cost_cents` (the cash the agent's account lost in that tick).
* `GoalSpec.metrics` (numeric targets): Crucible gives the public objective as text only; the evaluator's metric
  definition and target stay evaluator-side in observation-only mode. A CAM adapter parses the public text or takes
  public targets from the operator.
* `StepRecord.reward` ("verified return increment from the harness"): in an observation-only run a CAM harness must not
  turn `verify()` output into per-step rewards (that would re-create shaping); verified labels are for evaluation and
  meta-learning.
* Durations: CAM uses float days, Crucible integer seconds (`TIME_UNIT`).

To connect a new learner: write an adapter like `examples/experience_external_adapter.py` in the learner's own
repository, importing only `crucible.experience`; decide what the learner values (it computes its own reward, as
`examples/external_learner/curiosity_learner.py` does); drive it with your own loop or `run_learner`; host the world
with `crucible.experience.host` when the evaluation must be isolated. No Crucible file changes
(`tests/v5/test_gate_SP_11_6.py`, `tests/v5/test_gate_OO.py::test_OO6_an_external_learner_supplies_its_own_rewards`).

## 10. Commands

```bash
PYTHONPATH=src python -m crucible.experience tiers
PYTHONPATH=src python -m crucible.experience run --tier 1 --name rapid:net_routing --seed 0 --steps 300 \
    --store /tmp/exp --checkpoint /tmp/ref.pt --out /tmp/run.json           # observation_only by default
PYTHONPATH=src python -m crucible.experience.probes                         # the information-flow probes
PYTHONPATH=src python examples/external_learner/curiosity_learner.py 1 rapid:net_routing 200
PYTHONPATH=src python -m pytest -q tests/v5/test_gate_R.py tests/v5/test_gate_OO.py tests/v5/test_gate_SP_11_6.py
```

The `run` command is reproducible: the same command gives the same experience-store digest, learner parameter hash
and world digest (Gate S.8, for a mechanism and a rapid world).

## 11. What is not there yet

* No v5 Tier 2 domain is registered (`crucible.domains.base.REGISTRY` is empty; D-17 makes the domains
  configurations of `crucible.civ`). Every gate that needs Tier 2 fails and says so.
* Tier 3 is reached through `crucible.civ`; on the wave-2a integration commit its generator raised (a duplicate
  `space` module, fixed by the concurrent integration work). The Tier 3 gates fail until that fix is merged and pass
  through the same code once it is (verified locally against the integration branch's tree).
* In-world simulated participants that act through tools now receive participant views (the nested-call fix); what a
  module reads directly from world state for its own physics is reviewed per module, not filtered (population and
  goods choices read the artifact- and physics-derived attributes, not creator labels; D-15/D-18). The 4.0
  innovation adopters (`innovation/adopt.py`, not part of v5 worlds) write their own test scores into forum posts:
  scores of tests the adopter ran on inputs it generated itself, i.e. its own experience, not an evaluator value.
* Operator-only experiment controls of the population (`pop_adaptive.set_learning`, `freeze_copy`) are listed to
  participants but refused (`forbidden`).
* The 4.0 training export (`training.trajectories.RecordingRunner`, unchanged 4.0 code) records an objective's
  shaped step reward when given one; it is not a learner interface, and the v5 experience records carry the
  environment's reward (0.0 by default).
