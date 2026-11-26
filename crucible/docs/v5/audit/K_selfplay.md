# Audit K: civilization-wide self-play (Part 12, Gate Q, SP-12.1, SP-12.3)

Auditor: independent audit (workstream K area). Worktree `feat/crucible-5-complete` at `a881d18`. I changed no existing
file. Probe scripts and logs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/K_selfplay/`.

## 1. Verdict in one paragraph

Civilization-wide self-play does not exist. `src/crucible/multiagent/__init__.py` is a 5-line docstring stub. No
`multiagent/modules.py`, no `tests/v5/test_gate_Q.py`, no `tests/v5/test_gate_SP_12_1.py` and no
`tests/v5/test_gate_SP_12_3.py` exist. `python -m crucible.acceptance.v5 run --requirement Q.1|SP-12.1|SP-12.3`
prints "no test files exist yet for this selection" and exits 2. The Tier 3 world that Gate Q needs is missing too:
`crucible.civ` is a stub, `experience.registry.civ_available()` returns `False`, and `domains()` returns `{}`. Workstream K was
scheduled for wave 3 and never started (docs/v5/agent_reports/INDEX.json has no K agent).

Three pieces can be reused:

1. **The 4.0 numeric self-play stack (`crucible.selfplay`).** Its seat isolation, kernel twin, league, archive,
   held-out pool, PFSP and reproducibility are sound and reproduced. It is numeric-facility only, which Gate Q
   explicitly rejects. It defaults to the shaped legacy reward, and its training loop has two reward leaks in
   observation-only mode.
2. **The class-3 strategic groups (`population/strategic.py`, workstream B).** They are the only independently
   updated neural policy groups that act through ordinary tools in a shared world. They have hash-chained updates and
   frozen historical copies, but their action space is a templated 4-head choice. They have no cooperative actions,
   no measured competition and no held-out or cross-play evaluation.
3. **The evaluator-side integrity machinery** (`commerce/integrity.py`, `goals/metrics.related_parties`,
   `training.league.measure_agent`). It can exclude controlled-identity revenue, but only when the harness registers
   a shared `controller`, and no multi-agent harness does.

## 2. What I ran

| # | Command / probe | Result |
|---|---|---|
| T1 | `pytest tests/test_selfplay.py` | 28 passed (44 s) |
| T2 | `pytest tests/test_tooling.py -k league` (3.0 `training.League`) | 1 passed |
| T3 | `pytest tests/test_v5_population_core.py` (pop_strategic plumbing) | 11 passed (66 s) |
| T4 | `python -m crucible.acceptance.v5 run --requirement {Q.1,SP-12.1,SP-12.3}` | exit 2, "no test files exist yet" |
| P1 | `probe_arena_modes.py`: reward-mode defaults, leakage, learning with/without reward | see 3.1 |
| P2 | `probe_pop_strategic.py`: class-3 groups, hashes, freeze_copy, interactions | see 3.3 |
| P3 | `probe_controlled_identity.py`: same learner, two identities, subscription | see 3.4 |
| P4 | `probe_league.py`: SelfPlayLeague versions, held-out kinds, provenance, reproducibility | see 3.2 |
| P5 | `probe_pop_strategic_learning.py 10` (slot scheduler, 1 CPU): behaviour/outcome change of class-3 learning vs frozen copy | see 3.3 |
| P6 | `probe_joint_objective_leak.py`: what the "joint" objective gives a learner | see 3.1 |

All runs were single-threaded (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`). P5 exceeded 5 minutes and went
through `crucible.scale.slots --cpu 1 --gpu-mb 0`.

## 3. Findings with evidence

### 3.1 `selfplay.arena`: sound isolation, but legacy reward by default and two leaks

**Isolation is real.** Each seat's observation is built from its own record only (`arena.py:229-252`). The kernel
twin `KernelSelfPlayEnv` matches tick for tick. Individuals share no tensors, optimisers, generators or buffers
(`tests/test_selfplay.py:237-275`, re-run in T1). The tests check storage pointers (`data_ptr`) and memory sharing
(`np.shares_memory`), not only object identity.

**Defaults violate the observation-only amendment.** Probe P1 showed:

- `check_mode(None)` returns `legacy_shaped_reward` (`experience/modes.py:33-34`). The module docstring says "default
  everywhere".
- These all default to `legacy_shaped_reward`:
  - `SelfPlayEnv.__init__` (`arena.py:153`)
  - `KernelSelfPlayEnv.__init__` (`arena.py:425`)
  - `KernelExperienceEnv.__init__` (`experience/kernel.py:59`)
  - `experience.registry.make` (`registry.py:99`)
  - `MechExperienceEnv` and `RapidExperienceEnv` (`tier1.py:61,472`)
  - `CrucibleParallelEnv` (PettingZoo, `api/pettingzoo_env.py:42`) and the Gym env (`api/gym_env.py:71`)
- A default `SelfPlayEnv` step returns a nonzero reward and writes it into the observation's reward slot
  (`obs_reward_slot = -0.062`).

**The selfplay learners cannot learn without the env reward.** Probe P1 trained against a `hold` opponent on
`abm_duopoly` seed 3, 4 slots:

| learner, mode | updates counted | policy weights changed | net/tick, first 3 iterations → last 3 | frozen eval |
|---|---|---|---|---|
| pg, legacy | 38 | yes | 508 → 4075 | 7579 |
| pg, observation_only | **72** | **no** (hash `326e6037…` unchanged) | 301 → 31 | 117 (= defaults) |
| climber, legacy | – | – | 7048 → 8369 | 9011 |
| climber, observation_only | – | – | -75 → -97 | -56 |
| spsa, legacy | 12 | – | 478 → 9525 | 11924 |
| spsa, observation_only | 12 | no (plays defaults) | -47 → -185 | 117 (= defaults) |

The PG learner reports 72 "updates" in observation-only mode while its policy weights do not change: the advantage
is zero, but optimizer steps are still counted (`learners.py:440`). So "number of updates" is not evidence of learning.

The climber reads its own net from the observation's reward slot (`learners.py:527`), which is zero in
observation-only mode. None of the learners derives a signal from the legitimate observed revenue and cost scalars
(`obs[scalars][1:3]`), which stay present in that mode.

**Leak 1: `info["score"]` in observation-only mode.** `SelfPlayEnv.step` returns `info["score"]`, the mechanism's own
objective quantity, which `experience/modes.py:30-31` classifies as an operator record key. It is returned in every
mode (`arena.py:368`): probe P1 found it nonzero on 10/10 ticks in observation-only mode.

`play()` feeds it to the learner through `Objective("score")` (`arena.py:747-751`). In probe P1, a PG learner
"trained in observation_only" with objective `score` changed its weights and reached 489 → 4122 net/tick, frozen eval
7629. That matches legacy performance: the observation-only switch is bypassed by the harness.

The leakage battery `experience/leakage.py:132-166` (`_numeric_slot`) inspects only `out[0]` and `out[1]` (line 147).
It never scans the info dict, so it cannot detect this.

**Leak 2: the "joint" objective hands a learner its partner's private income.** `Objective.reward` with
`kind="joint"` returns `reward.mean(axis=0)`, the mean of all seats' private nets (`arena.py:570-575`). Probe P6
recovered the partner's per-tick private net from the spy's training signal, as `2*signal - own`, with maximum error
**6.8e-09** over 120 ticks in the default mode.

This is a private-information channel between participants (MASTER_SPEC 12.3, "unauthorized access to private
observations"). The 4.0 "cooperative" mode therefore manufactures cooperation by evaluator-side reward sharing.
Part 12.3 requires that "cooperation should emerge through permitted world interactions". In observation-only mode the
same signal is identically zero (0/120 nonzero ticks), so cooperative training does not work at all.

**Scope.** Every self-play family is a two-seat hidden-mechanism facility:

- `abm_duopoly`
- `abm_duopoly_fx`
- `abm_commons`
- `abm_complement`
- `tech_handoff`
- `tech_shared_pool`

See `modes.py:52-66`. Gate Q says "numeric facility-only self-play does not pass".

### 3.2 `selfplay.league` and `training.league`: working league machinery, missing provenance and learned held-out families

Probe P4: two independently trained PG individuals on `abm_duopoly`, 8 training matches, snapshots at stages 2 and 5,
then the standard held-out population and `evaluate_heldout`. Results:

- **Independent updates.** 8 distinct weight hashes per individual, and the two individuals' final hashes differ.
- **Archive.** The snapshots `A@2, B@2, A@5, B@5` are frozen and reloaded from disk (re-verified in T1).
- **Reproducibility.** The whole run, repeated in a fresh directory, reproduces the hash sequence and the held-out
  evaluation exactly (`reproducible_hash_sequence: true`, `reproducible_heldout_eval: true`).
- **Held-out population.** It is `ho:hold`, `ho:random`, `ho:climber` and two climber funding variants
  (`league.py:139-164`): scripted operators only. There is **no held-out learned policy family**. (The probe's
  `heldout_contains_learned_policy: true` comes from `Individual.learns` (`arena.py:644-645`), which wrongly counts
  the scripted `climber` as a learner.)
- **Provenance.** No weights or model hash appears in `league.json` entries, `info`, match records or individual
  `meta.json` (`any_weights_hash_in_league_json_or_meta: false`). Match records carry only the task `spec_hash`.
  There is no training-data provenance, no per-update record and no population-composition snapshot. Part 12.2 lists
  all of these.
- **3.0 `training.League`** (`training/league.py:362-403`) is the right shape for a civilization league:
  - every policy is its own actor in a freshly generated kernel world;
  - seat handles are neutral and policy ids are not written into the world;
  - outcomes come from `measure_agent` (independent revenue with related-party exclusion);
  - it has a held-out pool and PFSP.

  But its policies are `agents.scripted` factories, and it plays in 4.0 economy families. T2 passes.

### 3.3 Class-3 strategic groups (`population/strategic.py`, owned by workstream B): the only learned groups in a shared world

Probe P2 used a population world of 200 individuals and 6 strategic members in groups A (MLP, REINFORCE) and B (GRU,
A2C) with small hidden sizes. After 2 days I ran `pop_strategic.freeze_copy` (A → A@d2, share 0.5), then ran 3 more
days. Results:

- **Default sizes meet the class-3 range:** `StrategicMLP` h1024 has 2,162,715 parameters and `StrategicGRU` h576 has
  2,362,203 (spec: 2–20M).
- **Real, hash-chained updates:**
  - A: 14 updates, 112 optimizer steps, 15 distinct hashes, 8 checkpoints.
  - B: 19 updates, 152 optimizer steps, 20 distinct hashes, 10 checkpoints.
  - The `prev_hash` chain is intact for both groups.
  - Replay rebuilds and checks every hash (`strategic.py:233-257`).
- **Frozen historical copy works:** A@d2 kept its hash unchanged while A kept learning.
- **Participants act through ordinary tools as themselves:** `code.create_repo`, `code.commit`, `cloud.rent`,
  `cloud.deploy`, `id.grant` and `forum.post` (`strategic.py:395-435`). The log shows 119 deployments, 78 forum
  posts and 699 compute charges. The learning signal is the change in the company's own balance
  (`strategic.py:357`). That is an observable consequence, not a creator-assigned score.

Limits relevant to Gate Q:

- **No strategy space.** The policy picks 4 categorical heads (topic of 20, price of 5, content of 3, promote of 2)
  over one site template (`sites.make_spec`, `strategic.py:153-156`). It cannot trade, negotiate, contract, hire, join
  organisations, collaborate on software or create services that other agents call.
- **No cooperation.** There are 0 transfers between strategic members' accounts, and the ledger kinds touching their
  accounts are only endowment, compute and invoice. No cooperative action exists in the action space.
- **Competition is incidental and unmeasured.** No topic was shared across groups in P2
  (`topics_shared_by_groups: {}`). Revenue is tiny: 0–2 purchases per site, balances 499,770–501,677 against 500,000
  at the start. Mean reward per decision was 0.017 (in units of 1000 cents).
- **No held-out, cross-play or league evaluation** of these groups exists anywhere.
- **The identities are not linked to their controller.** Members are created by `ids.ensure_principal` without a
  `controller`, so `commerce.integrity.related_set` cannot tell that the members of one group are one controller.
- **Shared randomness.** Both groups sample their actions from one random stream per decision process
  (`strategic.py:335`). This is weak hidden coupling between competitors: it carries no content, but group B's draws
  depend on how many group A members are due.
- **ARCHITECTURE.md overstates the integration.** Line 83 says class 3 acts "via `crucible.experience`". It actually
  uses in-module `ctx.call`, and no external learner shares these worlds with them in any test.
- **Probe P5 (behaviour/outcome change): no evidence that learning improves anything.** The run had 12 strategic
  members over 10 simulated days and took 326 s under the slot scheduler. `freeze_copy` at day 2 moved 4 of A's 6
  members into the frozen A@d2, which starts from identical weights and serves as a natural control.

  The networks did update: A had 48 updates and B had 79. Mean realised reward per decision, early half vs. late half
  of days 2-10:

  | group | early | late |
  |---|---|---|
  | A (learning) | 0.022 | **-0.012** |
  | A@d2 (frozen) | 0.042 | 0.011 |
  | B (learning) | 0.032 | 0.052 |

  Final balance change per member, mean: A +92c, frozen A@d2 +1,528c, B +1,344c. The frozen copy did better than
  its learning twin. Topic and price entropy stayed near their maximum, so there was no policy sharpening: A's topic
  entropy went 2.51 → 2.61 and B's 2.63 → 2.59, against a maximum of ln 20 = 3.0.

  The sample is small and the signal tiny, so this proves no harm either. But Gate Q's "evidence of neural learning"
  cannot rest on these groups' weight-hash changes.

### 3.4 SP-12.3: controlled-identity transfers are counted as independent success today

Probe P3 used the commerce gate kit: founder `acme-f` forms a media business with a monthly plan of 2000 cents, and a
second identity of the same learner (`seat-b`) subscribes with its own endowment.

| admission | MRR | active subscriptions | independent customers | independent revenue |
|---|---|---|---|---|
| as the multi-agent harnesses admit (no `controller`) | **2000** | 1 | **1** | **2000** |
| with a shared `controller` | 0 | 0 | 0 | 0 |

None of the multi-agent harnesses passes a controller to `admin.join`:

- `KernelExperienceEnv.add_agent` (`experience/kernel.py:132-138`)
- `KernelSelfPlayEnv` (`arena.py:442-443`)
- `training.League.play` (`training/league.py:380`)

So with today's harnesses, a learner that controls several identities earns "independent" recurring revenue from
itself. The detection mechanism exists (`commerce/integrity.py:63-90`, `plant_same_controller` at line 418); it only
has to be wired in. `selfplay.audit.transfer_invariance` and `bilateral_transfers` cover only the rig metric
`rig_net`.

### 3.5 Collusion checks

`selfplay.audit.classify_pair` (`audit.py:272-356`) tells value-creating cooperation from tacit collusion and from
mutual cost restraint. It does this from engine-private diagnostics: units served, price per unit and work
completed. It passes its test (T1, `test_audit_tells_value_creating_cooperation_from_tacit_collusion`), but it is
specific to the `abm` and `techsys` engines.

Nothing detects collusion in kernel economies:

- price coordination among learner-run businesses;
- bid rigging in `market`, `contracts` or `labor`;
- wash trading among a learner's identities (beyond the related-party netting in `commerce.integrity`);
- coordination through hidden channels;
- collusive benchmark exploitation (two participants inflating each other's evaluator metrics).

### 3.6 Population diversity (SP-12.1)

No population object or world mixes the seven member kinds of 12.1 (current, historical, independent, statistical,
compact neural, semantic and held-out policies):

- `SelfPlayLeague` holds live learners, snapshots and scripted held-out fixtures.
- The population world holds class 1 (statistical), class 2 (adaptive neural) and class 3 (strategic) participants.
- Class 4 semantic participants (`semantic/participants.py`, `SemanticParticipant(Policy)`) are `agents.base.Policy`
  objects. They can be hosted by a `Runner` or by `training.League`, but no world combines them with the other
  kinds.
- No held-out learned policy family exists anywhere.

## 4. Transfer to the real internet: shortcuts and creator-assigned signals

- **`Objective` kinds are evaluator-side reward engineering.** `joint`, `mixed` and `score` decide what a learner
  values and pass it privileged quantities: the partner's private income, and the mechanism's objective quantity.
  A learner trained this way learns to exploit a team-reward channel that the open internet does not provide.
- **Strategic groups learn a lookup over a template site.** They choose a topic/price/content/promote cell, and the
  site itself is generated by `sites.make_spec`. That is closer to tuning a bandit over a simulator template than to
  learning business strategy.
- **"Updates" counters overstate learning.** PG counts optimizer steps with zero advantage. The `ReferenceLearner`'s
  world-model loss changes weights with no reward at all (`experience/reference.py` docstring, lines 19-21). A
  `weight_hash_changed` criterion is met without any policy improvement.
- **Held-out evaluation uses scripted opponents only.** Scripted operators (hold, random, climber) in numeric
  facilities say little about robustness to unfamiliar *learned* strategies.
- **Gate S skips silently.** `tests/v5/test_gate_S.py:135` drops the Tier 3 world when `civ_available()` is false
  (`if civ_available():`). Any K gate test must fail, not skip, when Tier 3 is absent.

## 5. Subsystem classification

| subsystem | verdict | effort | key reason |
|---|---|---|---|
| `multiagent/` civilization self-play harness | IMPLEMENT | XL | 5-line stub. No harness, registry, league, tests or evidence. |
| `selfplay.arena` envs (`SelfPlayEnv`, `KernelSelfPlayEnv`) as Tier 1 multi-seat envs | REPAIR | S | Isolation sound. Defaults are legacy, and `info["score"]` leaks in observation-only mode. |
| `selfplay.arena` `Objective` and `play` training loop | REDESIGN | M | Creator-assigned team/score signals. "joint" leaks the partner's private net. With observation-only nothing learns. |
| `selfplay.learners` (spsa, pg, climber) | REPAIR | S | They learn only from the env reward. They need an option to derive their own signal from observations. The update counter counts no-op steps. PG has about 10k parameters (Tier 1 only). |
| `selfplay.league` / `training.league` | EXTEND | M | Archive, held-out pool, PFSP, persistence and reproducibility verified. No model hashes, training-data provenance, learned held-out families or civilization match type. |
| `selfplay.modes` / `experiments` / `report` (6 modes) | EXTEND | M | Cross-play, best-response, history and cycle analyses are reusable. They run only on numeric facilities with the legacy reward. |
| `selfplay.audit` (transfer invariance, conservation, `classify_pair`) | EXTEND | L | Correct for rig facilities. No kernel-economy collusion battery. |
| Class-3 `pop_strategic` groups (as K's learned groups) | EXTEND | L | Real hash-chained neural updates, frozen copies, tool-based acts. Template action space, no cooperation, no measured competition, no evaluation against held-out opponents. |
| Controlled-identity accounting in multi-agent harnesses | REPAIR | S | `commerce.integrity` works with a `controller`. The harnesses never pass one, so 2000c MRR is counted from self. |
| Multi-agent experience adapters (`KernelExperienceEnv` with N agents, PettingZoo, `ReferenceLearner`) | EXTEND | M | Multi-agent and observed-signal learning are supported. Never run with ≥2 independent learners in one world. Legacy default. |
| Gate Q / SP-12.x tests and evidence | IMPLEMENT | M | No test files exist, and the checker exits "no test files exist yet". |

## 6. Requirements

| id | verdict | reason | work needed | criteria weak |
|---|---|---|---|---|
| Q.1 | IMPLEMENT | No Tier 3 world (`civ` stub, `civ_available()` = False), and no harness runs ≥2 learned policies in one shared world. | After J delivers a civilization: a `multiagent` harness that runs ≥2 independent learner instances in one civilization through `KernelExperienceEnv` in observation-only mode, plus the class-2/3 groups. Each instance must take world-changing actions. Evidence: per-policy action counts and hashes. | yes: a count of 2, with no check that the world is Tier 3 or that the policies acted |
| Q.2 | IMPLEMENT | The only independent groups are pop_strategic A/B in a population world. Competition between them is unmeasured (no shared topic in P2). | Two or more groups with disjoint parameters, optimisers, buffers and RNG streams (storage checks plus distinct hash chains) that contend for a resource, with a counterfactual effect of one group's policy on the other's outcome. | yes: a count only |
| Q.3 | IMPLEMENT | No cooperative world actions exist for learned agents. The 4.0 cooperation is team reward sharing. | Cooperative interactions through tools (contracts, joint ventures, co-developed repositories, trade) with no shared reward. Report counterfactual value created vs. no-cooperation, and competition measured from world events. | yes: ≥1 interaction of each kind |
| Q.4 | IMPLEMENT | No civilization self-play run exists. | Consequence chains in the shared world caused by learner A and observed by learner B (`cause` provenance), across domains. | yes: ≥1 event |
| Q.5 | EXTEND | pop_strategic records a version/hash chain. `SelfPlayLeague` records no hashes or provenance. | One policy registry: version id, weights hash, parent, algorithm, training-data provenance (experience ids, world ids, seeds), the update log, population composition per match, and reproducible cross-play between versions. | yes: 4 versions, no hash, provenance or reproducibility |
| Q.6 | EXTEND | Held-out pool exists but is scripted only and numeric-facility only. | Held-out learned policy families (architectures or algorithms never met in training), evaluated frozen in the civilization on sealed seeds with statistics. | yes: one evaluation against anything |
| Q.7 | EXTEND | Neural updates are real (PG in legacy mode, pop_strategic, `ReferenceLearner`), but selfplay learners do not learn in observation-only mode. In P5 the class-3 groups updated (A 48×, B 79×) but showed no outcome or behaviour gain over a frozen copy: A's late reward was -0.012 vs. 0.011 frozen, and entropy did not fall. | Show behavioural/outcome improvement under observation-only against a frozen-copy control and held-out opponents, not just hash changes. | yes: P1 showed 72 "updates" with an unchanged hash. The world-model loss alone changes hashes. |
| Q.8 | EXTEND | `classify_pair` works only on numeric engines. `commerce.integrity` plants revenue schemes. There is no civilization collusion battery. | A planted-scheme battery in the civilization: price fixing, bid rigging, wash trading among controlled identities, hidden-channel coordination, mutual metric inflation. Report detection rate and false positive rate. | partly: needs a minimum number of scheme types and a false-positive bound |
| Q.n1 | IMPLEMENT | Only numeric-facility self-play exists. | A gate test that refuses rig-only runs and requires a Tier 3 world with tool-based learners. | yes: `civilization_selfplay_runs >= 1` does not define what a civilization run is |
| SP-12.1 | IMPLEMENT | No population mixes the 7 kinds. Held-out learned families are missing. | A population manifest of current, historical, independent, statistical, compact neural, semantic and held-out policies, where every kind acts in one world, with a homogeneity/diversity measure. | yes: counts kinds, not participation or diversity |
| SP-12.3 | REPAIR | Controlled-identity subscription counted (2000c MRR, P3). The "joint" objective leaks the partner's private net (P6). `info["score"]` leaks (P1). | Harnesses register a `controller` per learner/group. Evaluator uses `commerce.integrity` for all self-play outcomes. Remove privileged training signals. Add tests for hidden shared state, private-observation access and shared evaluator information. | yes: covers only 1 of the 4 prevented items in 12.3 |

## 7. Spec requirements with no manifest entry

1. The Part 12 capability list: found competing businesses, collaborate on software, trade, negotiate, join
   organisations, develop competing technologies, create services used by other agents, form cooperative ventures,
   compete for scarce resources, respond to changing strategies. Only Q.3 touches it.
2. 12.2: model hashes, participant identity, training-data provenance, experience/action/outcome history, policy
   updates and population composition. Q.5 counts versions only.
3. 12.2: reproducible cross-play between policy versions, and league or population-based evaluation.
4. 12.1: "some learner instances may contribute to a shared training system; others must maintain independently
   updated policies", and avoiding a homogeneous population.
5. 12.3: no unauthorised access to private observations, no shared evaluator information, no collusive benchmark
   exploitation, and cooperation that emerges through permitted world interactions rather than shared reward.
6. Amendment (a) for self-play: observation-only as the default of every multi-agent env and harness. Self-play
   learners must learn from observed consequences only. Self-play results must be verified on the evaluator side
   (`measure_agent` / `commerce.integrity`), never from learner claims or reward streams.
7. 12.1 / 3.2 class 3: diverse learning algorithms, some participants with unrelated objectives, and some adapting
   throughout a civilization's lifetime. Partly covered by G.4 (B), but not inside self-play.

## 8. What can be reused for K

- `selfplay.arena.Individual`: identity, lineage, budget, persistence. Remove `Objective`'s privileged kinds, keep
  "own".
- `SelfPlayLeague` / `training.League`: archive, held-out pool and log, PFSP, Elo/TrueSkill, neutral seat handles,
  `measure_agent`. Generalise the match into a civilization episode hosting `ExperienceEnv` agents.
- `modes.cross_play`, `head_to_head` (cycle detection), `best_response`, `history_ratings`: as analyses over
  civilization match records.
- `population.strategic` groups and `population.adaptive` (class 2): in-world learned participants with
  hash-chained updates and `freeze_copy`. Extend the action space to the commerce/contract/labour tools and register
  a controller per group.
- `experience.reference.ReferenceLearner` (signal "observed"): an architecture-independent learner that already
  trains in observation-only mode. Run several instances (different seeds/configurations) as independent groups.
- `commerce.integrity.related_set` / `related_flows` / the planted `SCHEMES`: SP-12.3 and part of Q.8.
- `selfplay.audit.classify_pair`: its logic (joint gain vs. value delivered vs. price per unit) carries over to
  kernel markets if value delivered and prices are measured from the ledger and the population's purchases.

## 9. Work plan (minimum to make Gate Q meaningful)

1. **Prerequisites (other workstreams):**
   - a Tier 3 civilization (J);
   - observation-only defaults everywhere (L: `check_mode(None)` and every adapter default);
   - the leakage battery scanning `info` dicts.
2. **`multiagent/registry.py`:** a policy and version registry (hash, parent, algorithm, data provenance, update log)
   and a population composition manifest.
3. **`multiagent/harness.py`:** run N independent learners plus population groups in one civilization through
   `KernelExperienceEnv`. Each learner or group gets its own `controller`. Each runs in its own process or has its
   own RNG, so no hidden shared state is possible.
4. **`multiagent/league.py`:** civilization match = episode. Adds held-out learned families, sealed seeds,
   reproducible cross-play and PFSP.
5. **`multiagent/integrity.py`:** controlled-identity exclusion through `commerce.integrity`, plus a collusion battery
   with planted schemes and false-positive controls.
6. **Gate tests** `tests/v5/test_gate_Q.py`, `test_gate_SP_12_1.py`, `test_gate_SP_12_3.py`. They must fail (not skip)
   without Tier 3, and must require behavioural improvement against frozen controls, not counters.


## Skeptic review

Reviewer: skeptic pass on area K. Worktree `feat/crucible-5-complete`, now at `dd745b9`. The only commit after the audited
`a881d18` adds Clarifications 2-3 and manifest entries OO.1-OO.8 and RP.1-RP.4. It changes no source file under
`src/crucible/{selfplay,population,experience,multiagent}`, so the audit's code references still hold.

I changed no existing file except this one and `K_selfplay.json`, where I added a `skeptic` key. Probe scripts and logs are in
`/tmp/claude-0/-root/8e151463-101a-4f91-9360-1d4fa593f479/scratchpad/audit/K_selfplay_skeptic/`. Everything ran
single-threaded. S2 and S5 ran through `crucible.scale.slots --cpu 1 --gpu-mb 0`.

### S.0 What I ran

| # | Probe / command | What it tests | Result (short) |
|---|---|---|---|
| T1 | `pytest tests/test_selfplay.py -k "each_seat_sees_only or audit_tells or individuals_share_no_weights or heldout_population_never"` | The tests behind the auditor's "isolation is real" and collusion claims | 4 passed (10.5 s). They pass despite S1 and S4. |
| S1 | `probe_score_cross_seat_leak.py` | Does the learner-facing multi-seat interface leak a participant's private data in `observation_only`? | **Yes.** See S.1. |
| S2 | `probe_strategic_rule.py` (slots) | Can the class-3 learning rule learn at the world's reward scale? | It learns strong signals but is unstable: 4 runs collapse after reaching P(best) ≥ 0.9, and a run with no signal sharpens anyway. See S.3. |
| S3 | `probe_two_learners_kernel.py` | Can ≥2 independent neural learners share one kernel world through `ExperienceEnv`? | Parameters are isolated, but there is no driver, no interaction and no signal, and hashes change anyway. See S.4. |
| S4 | `probe_classify_reference.py` | Is `classify_pair` a collusion detector? | **No.** Its verdict follows the analyst's reference. See S.2. |
| S5 | `probe_strategic_competition.py 4` (slots) | Checks a harsh verdict (Q.2): does group B's presence change group A's outcomes? | See S.5. |
| S6 | `probe_obs_derived_net.py` | Is the arena "legacy shaped reward" more than the observable own net? | **No.** It equals observed revenue minus cost to within 3e-8. |
| S7 | `probe_strategic_step_scale.py` | Does the size of a class-3 update depend on the evidence? | **No.** The step is set by lr and the optimizer reset. |

### S.1 New leak: `score` carries the partner's private net into observations in `observation_only` (S1)

The auditor's Leak 1 concerned `info["score"]`, which the harness reads (`arena.py:368`). The same quantity is also in
**every learner-facing channel** of the Tier 1 multi-seat adapter and of the kernel facility, in `observation_only`:

- `MechExperienceEnv` (the `ExperienceEnv` wrapper around `SelfPlayEnv` for multi-seat tasks):
  - `_report_view` (`experience/tier1.py:222-224`)
  - the `rig.report` events (`tier1.py:314`)
  - the observation's `rig.seats.r1.score` (`tier1.py:342`)
- The kernel `rig.read` response (`mechanisms/module.py:398-410`; the docstring calls it "the quantity you are paid for").

What `score` contains depends on the engine:

- **abm families:** the seat's own net. S1 measured `score == net_cents` on 40/40 ticks.
- **`tech_handoff`:** the facility objective is `team_net`, and every seat's score is `(sum(revenue) - sum(cost)) / n_seats`
  (`mechanisms/techsys.py:868-869`). That family is designed with `private_gauges`.

S1 drove `tech_handoff` seed 3 in `observation_only` for 40 ticks. Results:

- `score` was present on 40/40 ticks in the observation numbers, in the `rig.read` result and in the `rig.report` events.
- Native reward was zero on all ticks.
- The partner's private net, recovered as `2*score - own_net`, had a maximum absolute error of **1.60 cents** (integer
  rounding). The partner's mean |net| was 349 cents.
- In the kernel twin (`KernelSelfPlayEnv`, through `World.execute` `rig.read`) the maximum error was **1.61 cents**.

The leakage battery cannot see this, for two reasons:

- `experience/leakage.py:37-39` `PRIVILEGED_KEYS` does not contain `score`, although `experience/modes.py:29` lists it
  among `OPERATOR_RECORD_KEYS`.
- The Tier 1 invariance test uses only the single-seat `scm_additive` task (`leakage.py:340`), and it plants no canary
  in the other seat.

`tests/test_selfplay.py::test_each_seat_sees_only_its_own_view...` (T1) checks only the numeric `SelfPlayEnv` view,
on a synthetic engine whose score is per-seat. It passes on an interface that leaks.

Consequences for the audit:

- (a) The auditor's "**isolation is real**" holds for memory (tensors, optimizers, buffers). It does **not** hold for
  information. The `selfplay.arena` REPAIR stays, but its scope and effort grow (S → M). The work is:
  - make `score` mode-aware, or a declared public rule of the world;
  - add a cross-participant derivability check to the battery, with a canary in the other seat's private record.
- (b) SP-12.3 gains a second concrete channel besides the `joint` objective, one that **survives
  `observation_only`**. This also bears on OO.2 (workstream L): "no native ... reward ... through observations,
  auxiliary fields, tool responses and event metadata".

**Correcting the auditor's Leak 1.** For `abm_duopoly`, the family the auditor's PG run used, `score` is exactly the
seat's own net. S6 shows that the arena's "legacy" reward also equals the observation's `(revenue - cost)/scale` to
within 3e-8, on `abm_duopoly`, `tech_handoff` and `abm_commons`. So the auditor's result (PG with `Objective("score")`
reaching legacy level in `observation_only`) shows the harness bypassing the mode with a reward channel. It does not
show privileged information. Privileged leakage happens through `score` only in `team_net` families (above), and in
legacy mode through `Objective("joint")` (the auditor's Leak 2, which stands).

The same fact **confirms the learners' REPAIR (S)**: a learner that computes its own signal from the observed own
revenue minus cost reproduces the legacy signal exactly, which OO.6 permits.

### S.2 `classify_pair` is not a collusion detector (S4) → `selfplay.audit` and Q.8 become REDESIGN

`classify_pair(pair, reference)` (`selfplay/audit.py:272-356`) labels a pair only relative to a reference that the
caller chooses. S4 used the test's own fixtures (`abm_duopoly` seed 0, scripted profiles, the same 4 noise seeds):

| pair | reference | class |
|---|---|---|
| collude (price 1.15·p0) | compete (0.90·p0): the unit test's case | `tacit_collusion` |
| **compete (0.90·p0)**: the test's *competitive* reference | price war (0.70·p0) | **`tacit_collusion`** |
| **collude (1.15·p0)**: the test's planted cartel | harsher cartel (1.30·p0) | **`value_creating_cooperation`** |
| mild coordinated rise (0.97·p0) | compete | `tacit_collusion` |
| collude, with **1e6 cents** reported paid between the pair | compete | `tacit_collusion`, `circular_manipulation.possible = False` |

The same competitive behaviour is "collusion" against one reference, and a planted cartel is "value-creating
cooperation" against another. Other problems:

- The function has no detected/flagged output.
- `circular_manipulation.possible` is the constant `False` (`audit.py:350`), whatever payments are reported. That is
  true for `rig_net` only, and false in any world where outcomes are bank-ledger based.
- It needs engine-private diagnostics (served units, price per unit, work completed) that kernel economies do not
  emit per seat.

Q.8's criterion `planted_collusion_undetected == 0` would be met by planting exactly the profile the reference is
chosen to expose. That makes the criterion circular.

Verdicts:

- `classify_pair`: **REDESIGN**. A detector needs:
  - pre-registered counterfactuals (for example the same learners with communication or contract channels disabled),
    or reference-free statistics (price-cost margins against a competitive benchmark computed from the ledger,
    synchronised price moves, bid-rotation patterns, related-party flows);
  - a false-positive rate measured on independent competitive learners.
- `transfer_invariance` and `conservation` stay **EXTEND**. They are correct, but for rig facilities only.

### S.3 Class-3 strategic groups: the learning rule is evidence-insensitive and unstable (S7, S2) → REDESIGN

`StrategicGroup.learn` builds a **new optimizer on every call** (`population/strategic.py:121`, Adam for A and
RMSprop for B). Each update's 8 steps are therefore first steps of a freshly initialised optimizer: about
`lr·sign(g)` for Adam, and about `10·lr·sign(g)`, decaying, for RMSprop. The step size does not depend on the size of
the gradient.

S7 ran the same 6-decision batch with rewards scaled by 1e-3, 1 and 1e3:

| configuration | per-update weight change (rewards ×1e-3 / ×1 / ×1e3) |
|---|---|
| A default (MLP h1024, REINFORCE) | max \|Δw\| 2.39e-3 / 2.42e-3 / 2.43e-3. That is 8·lr, the same for evidence 10⁶× apart. |
| B default (GRU h576, A2C) | mean \|Δw\| 5.4e-3 / 6.2e-3 / 5.7e-3, about **20 % of the mean \|w\| per update**, whatever the evidence |

S2 used a synthetic stationary bandit. The reward distribution matches the world's (auditor P2: rewards are mostly
-0.012, and about 2.6 % carry a ~1.5-unit purchase), and only the price head matters. The calls are exactly as
`_learn` makes them: 6 members per round, each example used once, 8 steps. Condition **signal** gives the best price a
purchase rate of 0.10 against 0.026; **strong** gives it 0.6; **null** gives every price the same rate.

- **The mechanism is not dead.** The small P5 configuration (MLP h64) learns *strong* on 8/8 seeds (mean P(best)
  0.998 at 67 updates).
- **It is unstable at realistic signal.**
  - h64 *signal*: mean P(best) 0.544 at 67 updates and 0.70 at 100, then **0.454 at 200**. Two seeds fell from ≥ 0.92
    to ≤ 0.17.
  - Default A *signal*, seed 1: P(best) 0.991 at update 10, then 0.004 at update 66 with price entropy 0.024. It
    **locked onto a worse price** after having found the best one.
  - Default B *strong*, seed 2: 0.999 at update 45, then 0.051 at update 66.
- **Sharpening is no evidence of learning.** Default A *null*, seed 1, reached price entropy 0.41 (maximum 1.61) on a
  non-best arm with no signal at all.

A 90-day default-config civilization gives about 45-67 updates per group (decisions every 2 days, 6 members). So
whether a group ends near the best arm or locked on a wrong one is close to a coin flip, even in a world far simpler
than the real one. This explains the auditor's P5 null result (the frozen copy did better than its learning twin)
better than "small sample" alone.

The defects combine:

- the learning rule's step size is independent of the evidence, and the rule is unstable;
- the action space is a template: 4 categorical heads executed by a scripted tool sequence (`_carry_out`,
  `strategic.py:389-423`), so the learned part never chooses tools;
- there is no cooperative action.

So the groups cannot serve as K's learned policy groups by extension: **EXTEND → REDESIGN** (effort L). The work
needed:

- persist the optimizer state, and log it so replay still reproduces;
- scale and normalise the advantages, and check steps against evidence (a no-signal control must not sharpen);
- replace template cells with a tool-level policy (or let these groups drive an `ExperienceEnv` learner) that covers
  the Part 12 actions;
- register a controller per group.

### S.4 Two neural learners in one kernel world (S3): the plumbing works, nothing else does

S3 put two `ReferenceLearner`s (seeds 1/2, hidden 96/64) into one `tiny` world through
`KernelExperienceEnv(agents=[alpha, beta], reward_mode="observation_only")`, and interleaved them for 160 steps.

What works:

- No parameter storage, optimizer or generator is shared.
- The hash chains are disjoint.
- Native reward was zero.

What does not:

- **There is no multi-agent driver.** `run_learner` (`experience/loop.py:21`) drives one agent and advances the clock
  on every step, so a Gate Q harness cannot use it as is.
- About 75 % of actions were rejected (alpha 124/159, beta 117/159). Each learner made only 3 world-changing actions.
- **Zero events** caused by one learner were seen by the other. There was no shared-world consequence at all.
- The observed signal (own balance change) was **0 cents** for both learners over 160 steps. Even so, 5/5 update
  calls changed the parameter hash (world-model and value losses).
- `add_agent` has no `controller` parameter (`experience/kernel.py:132`), although `admin.join` accepts one
  (`services/admin.py:17-23`).

The adapters stay **EXTEND**. But `ReferenceLearner` must not be counted as a learned policy group for Q.1, Q.2 or
Q.7: it meets "weight_hash_changed" with no learning signal.

### S.5 Do the class-3 groups compete? (S5, checking the harsh Q.2 IMPLEMENT)

S5 ran one population world twice: seed 11, 300 individuals, 6 sites, 6 A members (MLP) and 6 B members (GRU),
decisions every 6 h, 4 simulated days, 206 s under slots. Both groups had `lr = 0`, so the weights were identical in
both runs: the hashes A `2991dcbe57b4` and B `a9be492cb3b2` matched. In the counterfactual run, B's members decided
but their `_carry_out` was skipped, so they had no site and posted nothing.

| A's totals (6 members) | B present (6 B sites, 210 B visits) | B absent |
|---|---|---|
| visits | **252** | **336** (+33 %) |
| first visits | 205 | 275 |
| purchases | 5 | 2 |
| revenue | 2,855 c | 1,966 c |

Per member, 5 of A's 6 members had more visits without B: 31→46, 45→63, 35→50, 32→76, 51→52, and 58→49. The topics
the two groups chose overlap on {3, 10}. A's later actions differed between the runs although its weights did not,
because its observations (visits) differed. That is itself a shared-world consequence of B's behaviour on A.

So the two independently updated groups **do contend for the same population's attention, measurably**. This is a
single paired run: the visit effect is consistent (5/6) but not statistically established, and purchases (7 in
total) are noise.

The auditor's "competition incidental and unmeasured" therefore holds only in the sense that nothing measures it
today. Competing, independently updated groups exist. What is missing is:

- a Tier 3 host;
- a sound update rule (S.3);
- a controller per group;
- a counterfactual competition metric in the gate test.

**Q.2: IMPLEMENT → EXTEND.** The harsh verdict is partly unjustified. It is consistent with the auditor's own Q.5
EXTEND, which rests on the same groups.

### S.6 Other checks

- **`selfplay.modes` / `experiments` / `report`: EXTEND → REDESIGN.**
  - The auditor calls these "reusable as analyses over civilization match records", but they are not record
    analyses. `modes.evaluate` (`modes.py:83-96`) builds a `SelfPlayEnv` from a `Task`.
  - `cross_play` (`modes.py:212-240`) is a 2-seat matrix of `evaluate(task, [a, b])`. `best_response`, `run_pair`,
    `run_population`, `run_historical` and `run_cross_play` all take a `Task` and arena `Individual`s.
    `SelfPlayLeague.head_to_head` (`league.py:284-310`) plays arena matches.
  - `report.report_A/report_B` are write-ups of the 4.0 experiments.
  - Only the formulas carry over (the cross-play gap, cycle detection over a win matrix, `lagged_response`,
    `commons_position`). A civilization with N participants per world, ledger outcomes and `ExperienceEnv` learners
    needs a new match engine underneath, so this is not an extension.
- **`selfplay.league` / `training.league`: EXTEND confirmed.**
  - The `training.League` base is generic: entries, ratings, PFSP, held-out flags, neutral handles and
    `measure_agent`.
  - `SelfPlayLeague` hard-wires `SelfPlayEnv([task]*n, spec)` with the legacy default (`league.py:60`), and offers no
    choice of mode.
  - The held-out `climber` reads the observation's reward slot (`learners.py:527`), so it degenerates to its defaults
    in `observation_only`.
- **Population class 4 is a label.** `population/core.py:323` assigns class-4 ("semantic") rows `policy_group = 200`,
  but no module operates group 200 (grep over `src/crucible`). The population world's semantic class is therefore not
  a semantic participant. This supports the auditor's harsh SP-12.1 IMPLEMENT.
- **Controlled identities (REPAIR S) confirmed.**
  - Participants cannot create persons in-world: `admin.join` is system-only, and `id.create_org` produces orgs that
    `related_parties` already covers.
  - So the controller has to be registered by the harness, as the auditor says.
  - SP-12.3 must also cover S1's channel and the harness `info` dict (`arena.py:368` returns every seat's
    `net_cents`, `revenue_cents` and `cost_cents`). Neither currently reaches a learner through `MechExperienceEnv`
    (`tier1.py` drops `info`), but a K harness must not forward it.
- **The auditor's harsh verdicts Q.1, Q.3, Q.4, Q.n1, the multiagent harness and the gate tests are confirmed.**
  `multiagent/__init__.py` is 5 lines, `civ_available()` is False, `domains()` is `[]`, and S3 saw zero cross-learner
  events.

### S.7 Verdict changes

| target | auditor | skeptic | why (probe) |
|---|---|---|---|
| selfplay.arena envs | REPAIR (S) | REPAIR (**M**) | Information isolation is broken: `score` / `team_net` reveals the partner's private net to ±1.6 c in `observation_only`, through observations, `rig.read` and events (S1). |
| selfplay.arena Objective + play | REDESIGN | REDESIGN (Objective) / **REPAIR (play)** | Leak 1 is own net for abm families (S1: 40/40), and the legacy reward equals observable revenue minus cost (S6). The loop only needs a learner-side signal. |
| selfplay.audit | EXTEND (L) | **REDESIGN** (L) | The verdict follows the reference: compete → tacit_collusion, cartel → value_creating_cooperation. `circular_possible` is hard-coded False with 1e6 c paid (S4). |
| selfplay.modes/experiments/report | EXTEND (M) | **REDESIGN** (M) | Task- and SelfPlayEnv-bound 2-seat code, not record analyses (code: `modes.py:83-240`). |
| population.strategic class-3 groups | EXTEND (L) | **REDESIGN** (L) | Fresh optimizer per update, so step size is independent of the evidence (S7). Lock-in onto a wrong arm, collapse, and sharpening without signal on a synthetic bandit (S2). Template action space. |
| Q.7 | EXTEND | **REDESIGN** | Every existing learner meets "≥100 updates + hash changed" without learning: PG 72 updates with the hash unchanged (auditor), ReferenceLearner 5/5 hash changes with signal 0 (S3), class-3 evidence-independent steps (S7). |
| Q.8 | EXTEND | **REDESIGN** | S4. The criterion is circular. |
| SP-12.3 | REPAIR | REPAIR (larger) | Adds the `team_net` score channel, which survives `observation_only` (S1). |
| Q.2 | IMPLEMENT | **EXTEND** | Two independently updated groups measurably contend for attention in one shared world: A's visits are 252 with B present against 336 without, 5/6 members (S5). Still needed: Tier 3 host, sound update rule, controller per group, counterfactual metric. |
| everything else | — | confirmed | see S.6 |

**Not overturned (with evidence):**

- The learners' REPAIR (S6).
- The controlled-identity REPAIR.
- The league EXTEND.
- The multi-agent adapters' EXTEND (S3).
- The IMPLEMENT verdicts.

### S.8 Requirements with no manifest entry (added)

1. **Cross-participant information isolation at the learner interface.** Nothing in observations, tool responses or
   events may let one participant reconstruct another's private quantities. This is a derivability test with a canary
   in the *other seat's* private record, not a key-name scan (S1).
2. **Learning-evidence validity.** Update size must scale with evidence. A no-signal control must not sharpen. Claims
   need a behavioural gain over a frozen copy, with confidence intervals (S2, S7). Q.7's counters cannot establish
   this.
3. **Collusion detection without an analyst-chosen reference**, with a measured false-positive rate on independent
   competitive learners (S4).
4. **A multi-agent `ExperienceEnv` driver** that interleaves ≥2 external learners in one world, gives each its own
   clock budget, and records `cause` provenance across learners. This is a prerequisite of any K harness (S3).
