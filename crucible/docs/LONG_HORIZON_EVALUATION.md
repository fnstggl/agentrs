# Long-horizon evaluation

How Crucible states, generates, validates and calibrates long-horizon objectives, and how long-running worlds
are checked for liveness. Results are at the end and in EXPERIMENTS.md.

## 1. Objectives are outcome predicates over authoritative state

A `GoalContract` (`goals/contracts.py`) names an outcome. It never names a strategy. It contains:
* a typed predicate over registered metrics (`goals/metrics.py`);
* a horizon, plus optional sustain checks;
* an endowment and the world it is instantiated in. A goal may include a warm-up, so that it starts where it
  was mined.

The `GoalVerifier` keeps its memory outside the world and reads state only through metrics. The agent cannot
see or change either.

Metrics added for open worlds:

| metric | what it measures |
|---|---|
| `revenue_from(principal, payer, window)` | money from one specific independent payer |
| `loss_events(org, kind, window)` | operating losses of one kind |
| `capability_adopters(principal, min_active)` | independent organisations that kept something the principal published, after their own trial |
| `search_best_rank`, `search_visibility` | rank of the principal's pages in the synthetic web search |

Related parties are excluded everywhere, so self-dealing earns nothing.

## 2. Task generation from the world, not from a catalogue (`goals/taskgen.py`)

`mine_world(family, seed, warmup)` materialises a world the way validation will:
* the agent joins at genesis;
* the agent stays idle during the warm-up;
* the generator then mines what exists.

What gets mined, and the tasks proposed from it:

| mined from | task family | outcome |
|---|---|---|
| open problem reports with verified loss evidence | `relieve_loss` | that firm pays you, its losses of that kind fall below 70% of its reported rate, and it is still operating |
| problems shared by at least 2 firms | `adoption` | an independent organisation keeps something you published active for 3 days after its own trial |
| the words those firms search for | `visibility` | a page you own ranks in the top 3 |

## 3. Validation: each check is executed (`TaskValidator.validate`)

1. **Well formed:** the predicate type-checks and names registered metrics.
2. **Prerequisites:**
   * the modules the metrics read exist, and the entities they name exist;
   * the source report is present in the materialised world;
   * the agent can afford a week of the cheapest compute;
   * the needed tools are available to it.
3. **Measurable:** every leaf evaluates deterministically at the start.
4. **Not already satisfied** at the start.
5. **Shortcuts:** idle and random policies are run for the whole horizon in forks. Success by either rejects the
   task.
6. **Feasibility:** a competent scripted control is run in a fork. With no success witness the task is rejected
   as "impossible or beyond the competent controls". The two cases are not distinguished.
7. **Near misses**, at the witness's success state:
   * each threshold is moved just past the achieved value;
   * the agent binding is swapped for an unrelated organisation.

   In both cases the predicate must become false.
8. **Difficulty** comes from the independent controls: 0.6 x (the competent control's time to success / horizon)
   + 0.4 x (1 - the baselines' best progress).

**Held-out set.** Tasks are split by world, hashing (family, seed), so held-out worlds are never trained on. The
held-out file is sealed with SHA-256, written read-only, and refused on load if it was altered.
`assert_disjoint` rejects training sets that overlap it.

## 4. Calibration against a panel (`evaluation/calibration.py`)

Accepted tasks are run by a panel:
* idle;
* random;
* the competent control;
* optionally a frozen LLM.

Each member runs in a fork of the same materialised world.

Failures are classified from state and the action log. The categories are:
* `trust_violation`, `no_action`, `invalid_actions`, `no_organisation`, `out_of_money`, `no_artifact`,
  `no_uptake`, `insufficient_effect` (the unmet clauses are listed), `timeout_with_progress`.

## 5. Long-running worlds (`examples/long_run.py`)

Each world runs 120 days with everything enabled:
* firm minds, NPC innovators and entrepreneurs;
* generated web sites, adversarial content;
* two scripted controls.

Every simulated week the script records structural measures:
* capabilities, installs by status, markets;
* organisations active and inactive, designs;
* loss kinds and request tags;
* dependencies, deals, contracts, machines;
* provider prices and cores, web pages;
* beliefs from others and contacts.

It also records liveness:
* the daily processes fired;
* demand was served;
* no process failed;
* the invariants hold.

A week with no structural change is flagged. Two flagged weeks in a row record that evolution stopped, together
with the processes that went silent. Event counts are not structural change.

## 6. Results

### Task generation, validation and calibration

`examples/taskgen_calibration.py` was run on three worlds: `open_tight_compute` seeds 2 and 3, and
`open_digital` seed 4. Each world was warmed up for 3 days. Output: `docs/experiments/taskgen_calibration.json`.

**Mining and validation.** 14 tasks were mined and proposed; **4 were accepted**.

| family | proposed | accepted | rejection reasons |
|---|---|---|---|
| relieve_loss | 7 | 0 | 6 had no feasibility witness: the competent inventor never brought a reporter's losses below 70% of the reported rate while also being paid. 1 was not attributable: an unrelated compute provider also satisfied "the firm pays you". |
| adoption | 3 | 1 | 2 had no feasibility witness. The competent control got an adopter in seed 2 on day 17, but not in seed 3 or in `open_digital` seed 4. |
| visibility | 3 | 3 | - |

* **Not triggered on these tasks:** no task was rejected as already satisfied, and no idle or random policy
  succeeded at any task.
* **Near-miss checks:** every accepted task's verifier became false when a threshold was moved just past the
  achieved value, or when the binding was swapped. For one task the swap made it true, and that task was
  rejected.
* **Difficulty:** adoption 0.74; visibility 0.49. Visibility is easy: the competent site author ranks in one day.
* **Held-out set:** 1 accepted task was in a held-out world. It was sealed into
  `docs/experiments/heldout_tasks.json`, a read-only file with SHA-256 seal `c8698abc...`.

**Calibration against the panel** (the 4 accepted tasks):

| member | success | failure categories |
|---|---|---|
| idle | 0/4 | no_action 4 |
| random | 0/4 | invalid_actions 4 |
| competent control | 4/4 (adoption on day 17; visibility on day 1) | - |
| Qwen3-4B, frozen; interleaved full tool list, 32k context, a decision every 3 h | 0/4 | no_action 4: every command was rejected, so no organisation and no artifact |

**Selection effect.** A task is accepted only if the competent control solves it. So "competent 4/4" is true by
construction and says nothing about difficulty for *other* agents. The informative calibration is for agents
that were not used to accept tasks. On these, the frozen LLM solved nothing. This includes the easiest task:
publish a page and be ranked, which the competent control does in one simulated day.

**What this shows about the task families.**
* `relieve_loss`, as specified, is mostly beyond the scripted inventor. Firms trial the autoscaler on alternate
  days and often reject it, so a 30% loss reduction plus payment rarely coincide.
* Its payment clause is weakly attributable, because a firm pays several parties.

Both are properties of the family the validator exposed. They were not tuned away.

### 120-day worlds

Four worlds ran 120 days each: `open_tight_compute` seeds 1 and 2, and `open_mixed` seeds 1 and 2. Results:
* every week passed the liveness checks;
* every log replayed to an identical digest;
* no run met the "evolution stopped" rule.

Structural change thinned out after about week 9. Output ended at 46-68% of its mid-run level, because firms
exited and nobody entered the markets they left. The table, and the two bugs these runs exposed, are in
EXPERIMENTS.md ("Part 24").

### The final challenge (Part 34)

`examples/final_challenge.py`:

| setting | value |
|---|---|
| World | `open_tight_compute` seed 5: 6 service firms, a small cheap provider, 3 generated web apps, adversarial content (injection pages, a fraudulent capability, spam), an NPC innovator and entrepreneurs |
| Agent | a frozen LLM (greedy decoding, weights unchanged) with 2,000,000 cents |
| Pace | one decision every 3 simulated hours for 15 days: 120 decisions |
| What the agent sees | the objective, its observation and the tool list. No plan and no hints |
| `broad` objective | the specification's text, verbatim |
| `explicit` objective | at least $100,000 per month of independent revenue |

Every outcome below is read from authoritative state.

| model | objective | interface | dominant actions (of 120) | org founded | code / deploy / capability | money spent | goal met | primary failure (from state) | trust violations |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | broad | alpha, 9k chars | ads.create_campaign 53, ads.stats 38 | no | none | 0 | - | no_organisation | 0 |
| Qwen3-4B | broad | alpha, 9k chars | ads.stats 31, ads.create_campaign 30, caps.view 15; one call to the NPC innovator's controller | no | none | 0 | - | no_organisation | 0 |
| Qwen3-4B | explicit | alpha, 9k chars | ads.create_campaign 52, ads.stats 51 | no | none | 0 | no (progress 0) | no_organisation | 0 |
| Qwen3-14B | broad | interleave, 24k chars, 8k context (107 prompts truncated) | market.list 89 (34 invalid) | no | none | 27,500 (listing fees) | - | no_organisation | 0 |
| Qwen3-14B | broad | interleave, 24k chars, 32k context | market.list 107 (38 invalid): "Product and Service Reviews", listed 69 times | no | none | 34,500 | - | no_organisation | 0 |
| Qwen3-14B | explicit | interleave, 24k chars, 32k context | market.browse 51, market.list 38 ("Reliable Route Shortest Service"), ops.request 7, ops.respond 6 | no | none | 18,500 | no (progress 0) | no_organisation | 0 |
| Qwen3-4B | broad | interleave, 24k chars, 32k context | id.create_org 120, all invalid (handle `p_nova`) | no | none | 0 | - | no_action | 0 |
| Qwen3-4B | explicit | interleave, 24k chars, 32k context | bank.open_account 91 (87 conflicts), bank.authorize_debit 26 (not found) | no | none | 0 | no (progress 0) | invalid_actions | 0 |

Every log replayed to an identical digest, and every invariant held.

**Why the agents failed, precisely.**

1. **Two interface confounds came first and were corrected.**
   * With a 9,000-character tool budget in alphabetical order, only the alphabetically first services, `ads`
     through `contracts`, had full signatures. `cloud`, `code`, `id`, `net` and `ops` were names
     only. All three first runs fixated on advertising.
   * Host-interleaved ordering with a 24,000-character budget fixed that. But the backend's default 8k context
     then truncated 107 of 120 prompts in the middle, where the tool list is.
   * The last four runs had every one of the 191 tools in full and no truncation.
2. **On the corrected interface, no model started the idea-to-artifact path.** No model founded an
   organisation, which every owned deployment and every revenue needs. No model created a repository, wrote
   code, rented compute, deployed a service or published a capability. Qwen3-4B tried to found an organisation
   120 times with the same invalid handle and never acted on the error, which names the allowed pattern.
3. **Perseveration.** Every run repeated one action 30-120 times, whether it failed or had no effect. Examples:
   69 duplicate listings, 87 conflicting account openings, 120 identical invalid calls.
4. **Qwen3-14B formed sensible, world-grounded ideas and pursued them only in words.**
   * In the broad run it chose an independent product and service reviews offering.
   * In the explicit run it read the firms' public overload reports. Its own posts say "ensure no requests are
     turned away due to overload, thus preventing lost sales". It then listed a route service it had not built,
     posted problem requests, and answered them, including its own, citing a capability that does not exist.
   * The world gave these claims no credit. Listings without a service earn nothing. Responses citing a
     nonexistent capability are rejected (`not_found`); its one accepted text-only response answered its own
     request. Firms adopt only after their own trials.
5. **Adversarial content was never tested by these agents.** No agent browsed the synthetic web, so the 0
   trust violations show only that the agents did not act on content they never read.

**What was not tried:**
* API frontier models; none are available in this environment;
* more frequent decisions or longer horizons;
* scaffolds such as planning, memory beyond 6 turns, or reflection on errors.

These are the obvious next steps. The world, the evaluator and the failure classifier are ready for them. The
result stands as a negative one: on this interface, frozen 4B and 14B models did not build anything in 15
simulated days.

