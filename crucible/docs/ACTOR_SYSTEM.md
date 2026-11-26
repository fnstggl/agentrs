# Actors and populations

Policies are separate from world mechanics. Every actor — learning agent, NPC, human — acts only through the
public tools, with the same permissions, prices and limits. Three execution tiers keep large populations cheap:

| tier | representation | decisions | cost |
|---|---|---|---|
| background consumers | numpy arrays in the state (`economy/consumers.py`) | vectorised, hourly; from own experience and observable signals | ~1 s per simulated day for 1k–25k consumers, 2.2 s for 100k (measured, SCALING.md) |
| simulated persons and firms (NPCs) | records in `npc_*` tables + ordinary principals (`actors/npcs.py`) | event-driven / periodic rule-based and stochastic policies, executed as tool calls | 2,000 NPC workers add no measurable time per simulated day when idle |
| agents | external policies (`agents/`) | one tool call per turn, any architecture | the policy's own cost + ~19 ms of world time per agent step (3 agents, measured) |

## Consumers

See ECONOMY.md §3. Heterogeneous by need, rate, willingness to pay, preference weights, loyalty, awareness and
social position; they shop among offerings they know, using verified ratings, prices, search results, ads and
neighbours. They never observe hidden quality directly — only through their own experience and others' reports.

## NPC society (`actors/npcs.py`)

* **Firms** (incumbents, entrants): daily policy with bounded rationality — reprice toward observed competition
  and capacity, scale instances when overloaded, start R&D when out-rated (employees implement better tiers;
  otherwise post a bounty), hire support staff when the ticket queue grows. Inputs: their own `cloud.status`,
  their tickets, public `market.browse` data.
* **Workers**: reservation wages, skills by role; search postings, apply, decide on offers with risk adjustments
  from public signals (company age, public payment breaches, reputation); quit or blacklist employers that do
  not pay and post a public warning; developers implement assigned issues as real code commits (quality by
  skill), support staff resolve tickets, sales staff post about their employer.
* **Lenders**: evaluate loan contracts against verified independent revenue (only with a `bank.read` grant),
  enforce rate/term/principal limits; defaults become claims and can trigger bankruptcy.
* **Persona accounts** carry consumer reviews and ratings so that word of mouth is visible in the world.

All NPC behaviour is in policy code; changing it never changes mechanics. NPC decisions are deterministic given
the world seed (they draw from `ctx.rng`), so worlds regenerate exactly.

## Policy classes available to researchers

| class | where | notes |
|---|---|---|
| rule-based / stochastic NPCs | `actors/npcs.py` | inside the world, scheduled |
| scripted controls | `agents/scripted.py` | Builder, Employer, Composer, BountyHunter |
| random / idle probes | `agents/scripted.py` | exploit and inactivity probes |
| frozen LLM | `agents/llm.py` | local Qwen3 (see RL_INTEGRATION.md) |
| CAM 2.0 (+ frozen LLM) | AGENTRS `src/agentrs/cam2/envs/crucible_adapter.py` (moved under v5 D-81) | AGENTRS learner through its EnvironmentAdapter |
| AGENTRS superlearner | AGENTRS `src/agentrs/sl/crucible_world.py` (moved under v5 D-81) | sl `World` adapter |
| small neural RL learner | `agents/rl.py` | online, from verified consequences |
| human | web frontend (`web/`) or `crucible actor` CLI | same tools |
| league / self-play populations | `training/league.py` | opponent pools, held-out populations |

## Time and scheduling

The kernel is a discrete-event simulator: background processes (demand, billing, NPC decisions, deliveries,
contract payments, shocks) are scheduled jobs; nothing runs per tick unless scheduled. `agents.base.Runner`
gives each agent one action per round in a seeded order and then advances the clock by `dt`. Agents can be
sleeping (not in the runner) while the world keeps living.
