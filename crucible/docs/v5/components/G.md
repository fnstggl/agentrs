# Workstream G: innovation and software ecosystem

Package `crucible/src/crucible/innovation/`, registered by `crucible.load_all_modules()` through
`innovation/modules.py`. It implements INTERFACE_CONTRACTS section 7 (capability registry `caps`, needs graph `needs`,
software ecosystem `pkg`) and the measurements for Gate J, Gate L, SP-6.3, SP-6.4 and SP-6.5.

## What exists

| file | what it is |
|---|---|
| `caps.py` | Kernel module `caps` (host `caps`), the v5 capability registry. It subclasses the 4.0 registry (`services/capabilities.py`) and replaces it under the same module name. Every 4.0 endpoint, applier and process is inherited unchanged. The v5 additions are `caps.declare`, `caps.search {query}`, `caps.try`, `caps.verify`, `caps.adopt_service`, `caps.drop_service`, `caps.declaration`, `caps.declarations` and `caps.adoptions`. |
| `needs.py` | Kernel module `needs` (host `needs`): the needs graph with six relation types and the `needs.update` process. Endpoints: `needs.mine`, `needs.publish`, `needs.requests`, `needs.graph`. Evaluator check: `special_case_triggers(world)`. |
| `pkg.py` | Kernel module `pkg` (host `pkg`), built on 4.0 `code` and `issues`. Endpoints: `register`, `release`, `yank`, `search`, `info`, `star`, `depend`, `upgrade`, `exec`, `ci`, `deploy`, `pr_open`, `pr_diff`, `pr_review`, `pr_merge`, `fork`, `report`, `advisory`. |
| `adopt.py` | Kernel module `adopt`: need-driven participants. They discover services, test them on their own inputs, adopt and use them, recommend them and publish the needs they hold. |
| `formats.py` | Problem formats beyond the process-global 4.0 `NEEDS`, for example `route.batch`, a composite technology. `lookup(name)` searches both. |
| `worlds.py` | `innovation_spec(seed, ...)` (a lean world containing all of the above), the `infra` module (compute providers at genesis), `FORMAT_SCHEMAS` and `STANDARD_INTERFACES`. |
| `reference.py` | Scripted reference agents, labelled `kind = "scripted"`: `ServiceCreator`, `BatchCreator`, `drive()`, and `software_history()`, a full package history acted through tools. |
| `scenarios.py` | The scenarios the gate tests run: `market`, `capacity`, `batch`, `social` and `software`, plus `arms()` (forks of one snapshot). |
| `measure.py` | Evaluator-only helpers for Gate J and L. See the measurement section below. |

### Capability registry (`caps`)

`caps.declare {service_host, capability: {name, inputs, outputs, doc, endpoint?, tags?}, tests_ref?}` binds one
endpoint of a live service to a named capability. `inputs` and `outputs` are JSON schemas, checked with
`schema.check_schema`.

- Only the operator can declare (`cloud.deploy` on the owner).
- Re-declaring the same name supersedes the previous declaration and removes it from the search index.
- Declarations are public and indexed for search under kind `capability`.

`caps.search {query}` matches exact names and hosts first, then ranks declarations by BM25.

- Every result carries public statistics: independent trials and their OK rate, independent adopters, public
  verifications.
- A search records the searcher's discovery (`caps.seen`, visible to the searcher only).
- The 4.0 form `caps.search {q}` keeps its exact 4.0 behaviour.

`caps.try {service_host, capability, input}` validates the input against the declared schema. It then executes the
real endpoint through `cloud.call_endpoint`, so the caller pays the price, the request runs in the sandbox and is
metered against the instance's capacity. Finally it checks the response against the declared outputs. The caller
gets the output back and judges it with its own tests; the registry never scores usefulness.

`caps.verify` runs a test-suite repository against the live endpoint in the sandbox, on seeds drawn fresh for each
verification. The suite has `suite.py` with `generate` and `check`. A suite written by the provider (`tests_ref`) is
labelled `provider_supplied` and is never counted as independent.

`caps.adopt_service` requires a successful `caps.try` by the adopter itself. After adoption, every real call the
adopter makes is counted (`caps.use`). Independence is always decided by `goals.metrics.is_related`.

The v5 tables and endpoints exist only with module config `{"registry": true}`. Worlds without it, including every
4.0 family, keep their exact tool catalogue, state digests and event logs: the v5 endpoints are withdrawn at install
and `caps.search` keeps its 4.0 schema there. This is checked by `tests/test_diagnostics.py`, which reproduces the 3.0
catalogue of 191 tools, and by `test_registry_is_opt_in_so_4_0_worlds_keep_their_tools_and_state`.

### Needs graph (`needs`)

The graph has node kinds activity, resource, capability, constraint and need, and the relation types requires,
consumes, produces, constrains, substitutes and complements. The structure is read off world state at every update;
it is never written per case:

| relation | source in world state |
|---|---|
| `svc:H consumes compute:I`, `capacity:I constrains compute:I` | a live deployment of H on instance I |
| `svc:H produces cap:K` | an active declaration of K by H |
| `svc:A substitutes svc:B` | two live producers of the same capability |
| `cap:K1 complements cap:K2` | K1's output schema satisfies K2's required inputs |
| `svc:A requires cap:K` | service A observed calling a producer of K (`cap_deps`) |
| `user:P requires cap:K` | P adopted K, or a participant's own task (`needs_demand` hook) |
| `repo:D requires pkg:N` | `pkg.depend` |
| `pkg:N@v produces pkg:N` | a release |
| `adv:X constrains pkg:N` | a verified advisory |

World generators may add standard interfaces and relations as data (module config `interfaces` and `edges`).

The rules are keyed only by relation type:

- **consumes + constrains (+ produces)**: a binding resource constraint yields a `supply` need for what its consumers
  produce. A constraint binds when peak hourly utilisation reaches at least 0.97, or when requests are refused for
  capacity.
- **requires**: something required that has no live producer gives a `provide` need. When every producer the holder
  uses is failing, or an active advisory affects the holder's locked version, the rule gives an `alternative` need.
- **complements**: when K1 has adopters and volume and a complementary K2 has no producer, the rule gives a `provide`
  need for K2.
- **substitutes and produces**: these decide when a need is addressed. A need is addressed when a producer created
  after the need arose offers its target. It is resolved when the rule stops deriving it for two updates.

Holders are the principals whose own activity is affected: operators, users who adopted the strained producer, and
dependents. Need events are visible to holders only. Users never see operator metrics; `needs.mine` shows evidence to
the operator role only. A holder can publish a need as a public request; entrepreneurs read those with
`needs.requests`.

The update runs every 6 simulated hours by default. Its parameters are module config: `period_s`, `util`,
`unmet_share`, `min_unmet`, `min_volume`, `min_adopters`, `error_share` and `resolve_after`.

### Software ecosystem (`pkg`)

Packages are public, open-licensed `code` repositories. Each package gets a public issue project in the 4.0 `issues`
module.

- **Releases.** A release pins a semver version to a commit. It runs the repository's tests in the sandbox and
  extracts the public API (function arities, by importing the module in the sandbox). It compares that API with the
  previous release (removed or changed functions, semver conformance) and publishes `README.md` as documentation.
- **Dependencies.** `pkg.depend` writes a flat lock, including transitive locks, into the dependent's
  `crucible.json`. 4.0 hosting and cloud deployments therefore mount every dependency at `deps/<repo>/`.
  `pkg.upgrade` re-resolves the version only if the dependent's own tests pass with the new release; otherwise the
  incompatibility is recorded.
- **Adoption.** A package counts as adopted only by execution. `pkg.exec` and `pkg.ci` run the dependent's code in
  the sandbox under a `sys.setprofile` tracer and record how many calls entered each dependency's directory.
  Adoption is a (package, dependent) pair where the package's code ran inside the dependent's own successful run.
  Stars and declared dependencies never count.
- **Pull requests.** A pull request is the head of another repository, tested when it is opened. A review counts as
  independent only when the reviewer is unrelated to the author and has read the diff (`pkg.pr_diff` is recorded).
  A merge needs an independent approval of that exact commit, passing tests and an unchanged target.
- **Advisories.** An advisory is verified only when the reporter's reproduction tests fail against the affected
  release in the sandbox. Every later release is re-checked against the reproduction, so `fixed_in` is measured.

### Participants (`adopt`)

Config: `n`, `needs` (format name to weight), `channels` (search and/or forum), `per_day`, `wtp_cents`,
`threshold`, `tests`, `candidates`, `start_hours`, `retry_hours`, `recommend_p`, `publish_p`, `endowment`,
`tolerance`, `window` and `switch_hours`.

Each participant is a principal with its own bank account. It acts only through tools, as itself. Every decision
comes from its own experience: its own generated test inputs, scored by its own problem's verifier, its failures and
the prices it pays.

## How to use it

```python
import crucible; crucible.load_all_modules()
from crucible.core.kernel import World
from crucible.innovation.worlds import innovation_spec
from crucible.innovation.reference import ServiceCreator, drive
from crucible.innovation import measure as M

w = World.create(innovation_spec(1, participants=20))          # caps(registry) + needs + pkg + adopt + infra
drive(w, [ServiceCreator("ana", "route.shortest", "anaroute", at=6 * 3600)], 3 * 86400)
prod = M.agent_products(w)[0]
M.discoverers(w, prod), M.users(w, prod), M.independent_transactions(w, prod)
```

To compose the modules into another world, add `{"name": "caps", "config": {"registry": True}}`, `needs` and `pkg`.
`pkg` depends on identity, bank, code and issues. Add `adopt` and `infra` when participants or providers are wanted.

Run from `crucible/` with `PYTHONPATH=src`:

```bash
python -m crucible.scale.slots --cpu 2 --gpu-mb 0 --label "ws-G: unit" -- python -m pytest -q tests/test_v5_innovation_caps.py tests/test_v5_innovation_needs.py tests/test_v5_innovation_pkg.py
CRUCIBLE_V5_AUTHOR=ws-G python -m crucible.scale.slots --cpu 2 --gpu-mb 0 --label "ws-G: gates" -- python -m pytest -q tests/v5/test_gate_J.py tests/v5/test_gate_L.py tests/v5/test_gate_SP_6_3.py tests/v5/test_gate_SP_6_4.py tests/v5/test_gate_SP_6_5.py
```

On the integration branch, evidence is produced with `python -m crucible.acceptance.v5 run --gate J`, `--gate L` and
`--requirement SP-6.3` / `SP-6.4` / `SP-6.5`.

## Measurement helpers (`measure.py`)

**Gate J:**
- `agent_products`, `discoverers`, `users`, `adopters`
- `independent_transactions`
- `compute_use`
- `fisher_exact`

**Gate L:**
- `emerged_needs`
- `needs_addressed_by_new_services`
- `Stage`, `chain_record`
- `counterfactual`; `scenarios.arms` handles several arms

**SP-6.5:** `software_features`, `adoptions_by_execution`

**L.n1:** `special_case_bonuses`

These helpers read only authoritative state and the log, and they work for any participants. When the population
(workstream B) exists, its principals are counted the same way. A participant is independent when it is not related
to the creator.

## Measured (this branch, gate tests run 2026-11-25, 91 s wall for all 17 gate tests)

| requirement | metric | value |
|---|---|---|
| J.1 | agent_created_executable_products | 4 (1 service, 3 packages) |
| J.2 | published_products | 4 |
| J.3 | independent_discoverers | 24 (service), 1 (geokit) |
| J.4 | independent_users | 24 (service), 1 (geokit, executed by a dependent) |
| J.5 | measured_behavior_adoption_pvalue | 2.0e-11 (24/24 adopters as built vs 2/24 with the weaker implementation) |
| J.6 | independent_transactions | 1,092 sales from 24 independent payers |
| J.7 | resource_consumption_delta | +58,364 compute units vs the fork without the product; +9,100 cents compute rent |
| J.n1 | product_categories | 2 (api_service, software_package) |
| L.1 | capability_effects_measured | 5 |
| L.2 | emerged_needs | 7 (supply, alternative, provide/complement, package alternative) |
| L.3 | needs_addressed_by_new_services | 2 (34 holders switched to the new service; 1 dependent upgraded to the fixed release) |
| L.4 | independent_chains / min_chain_stages | 4 / 4 |
| L.5 | counterfactual_removals_confirming | 3 of 3 |
| L.n1 | special_case_bonuses | 0 |
| SP-6.3 | novel_services_discovered_tested_adopted | 1 |
| SP-6.4 | reusable_need_relations / special_case_need_triggers | 6 / 0 |
| SP-6.5 | software_features / adoptions_by_execution | 15 / 2 |

Notes on these numbers:

- **SP-6.3.** All 24 participants found, tested, adopted and used the correct service. The invalid-route service was
  tested by all 24 and adopted by none. The independent suite scored them 1.00 and 0.38.
- **L.5.** The first stage of each chain was removed in a fork of one snapshot:
  - Withdrawing the service right after creation removed the supply need, the new service and the 62 later
    adoptions.
  - Never building the composite technology: base-service compute 5.69M vs 3.10M compute units, overloads 133 vs 0,
    supply need 1 vs 0.
  - Deleting the forum posts: forum-reader adopters 12 vs 1, search-channel adopters 12 vs 12.
- **SP-6.4.** The three counted checks are all zero: needs without a valid relation path, literals in the rule code
  naming entities, and differences after renaming every agent and host.

Cost: 20 participants and one service run 3 simulated days in about 2 s. The market scenario takes about 8 s for
24 participants, 4 days and three counterfactual arms. A 36-participant capacity world runs 7 days in about 9 to
11 s per arm. The scripted package
history takes 2.3 s. Sandbox calls run on persistent workers: deployment about 90 ms, call about 6 ms including the
world.

## Limitations

- **Who acts.** Creators and developers are scripted reference agents, which establish feasibility and are not
  learners. Participants are workstream G's rule-based need-driven adopters, not the workstream B population; the
  helpers will count B's principals unchanged. The chains are independently generated in the sense that each comes
  from its own world, seed and structure, and that every stage after the first is produced by generic mechanisms.
  The first act of each chain is scripted.
- **Social channel.** It is the 4.0 forum. Workstream C's platforms are not used yet.
- **Needs observables.** They cover services, compute instances, declared capabilities and packages only. Content,
  communities, education, organisations and supply chains are not observed yet. `complements` comes only from JSON
  schema structure or from relations configured as world data. Only `supply` needs and package `alternative` needs
  have been addressed by new things in the scenarios; the `route.render` complement need is never served.
- **Objective usefulness.** It is measured only for problem formats that have a verifier (4.0 `NEEDS` and
  `formats.FORMATS`). Subjective usefulness from diverse participant responses is not implemented here.
- **Adoption tracing.** It counts Python-level calls into `deps/<repo>/` under `sys.setprofile`. Code that only
  imports constants, without calling anything, is not counted as adoption.
- **Determinism.** Exact forks need `cost_model: deterministic`. `innovation_spec` sets it.
- **Registry is opt-in.** The v5 registry must be enabled per world, with `caps` config `registry: true`.
- **Substrate (workstream A) not used.** Nothing from `crucible.substrate` is used because it was not merged when
  this was built. Packages are `code` repositories rather than content-addressed artifacts, and cross-module effects
  use the 4.0 hooks (`on_service_call`, `needs_demand`), not `bus.subscribe`.
