# One-world contracts (binding for wave 2; decisions D-14 to D-21)

These contracts let the wave-2 workstreams build in parallel and still produce one world. They extend
INTERFACE_CONTRACTS.md; where they conflict, this file governs (D-17 supersedes its section 9 registry contract).

## 1. One population (D-15) - owner: CORE-POP (workstreams B + C)

`crucible.population` is the only population of every v5 Tier 2 and Tier 3 world. Its package `__init__` exports a
stable API (other workstreams import only these names):

```python
register_activity(world, name, handler, *, owner, prior=0.0, goals=())   # existing; handler(ctx, idx, now) -> utility array
n_individuals(world) -> int
features(world, idx) -> dict[str, np.ndarray]        # public decision features of individuals (no evaluator data)
interests(world, idx) -> np.ndarray                  # (n, semantic.space.DIM) float32, unit norm (D-16)
spendable(world, idx) -> np.ndarray                  # cents an individual can spend now
pay(ctx, i, to_account, cents, *, memo) -> str       # materialise principal + bank.move from the individual's own account
wake(ctx, idx, at, activity)                         # existing
class_members(world, cls) -> np.ndarray              # existing
record_decisions(ctx, kind, idx, values: dict, *, model_id, model_version, fidelity_id, mode="set", table=None) -> str
                                                     # columns.batch carrying decision provenance (who decided, with which
                                                     # model); the ONLY way a subsystem records individual decisions
experience(ctx, idx, activity, utility, *, item_vecs=None, satisfaction=None)
                                                     # updates satisfaction, habits and interests from realised outcomes
```

Rules: a subsystem takes part in the population's life only by registering activities. A handler receives the due
cohort, generates candidates through its own mechanisms (platform recommenders, search, markets, job boards, course
catalogues, service registries), lets individuals choose by their own state (semantic affinity in the shared space,
artifact-intrinsic features, price, social signals, history, fitted decision models from `crucible.calibration.runtime`
where available), applies consequences through its own appliers and the bank, records decisions through
`record_decisions`, and returns realised utility. No subsystem keeps its own population or a second copy of a per-person
fact (follows, shares, memberships, subscriptions, employment live in exactly one owner table each; D-21 check).
`crowd` (C2), `patrons` (F1), innovation adopters and 4.0 consumers/shoppers are not used by v5 worlds; they stay only
for their historical tests and 4.0 families.

Population profiles (Part 3.3): `development`, `standard`, `large`, `internet` with the class counts of the spec; class 4
rows are real semantic participants (bound to the semantic participant runtime), not inert rows.

## 2. Shared semantic space (D-16) - owner: CORE-SEM (workstream D)

```python
crucible.semantic.space.DIM: int; SPACE_ID: str; SPACE_VERSION: str
item_vector(world, ref) -> np.ndarray          # (DIM,) float32 unit norm for an artifact of any modality; cached by
item_vectors(world, refs) -> np.ndarray        # (artifact hash, encoder ids+versions, DERIVED preprocessing, space version)
text_vector(world, text) -> np.ndarray         # queries, needs, interests, descriptions
item_features(world, ref) -> dict[str, float]  # artifact-intrinsic features computed from the bytes (e.g. coherence /
                                               # LM log-likelihood per token, information density, language validity,
                                               # duration_s, modality, production features); never creator-declared
interest_prior(world, rng, n) -> np.ndarray    # (n, DIM) initial interests from embeddings of a real topic corpus
```

All computation goes through recorded inference (replay never reruns encoders). The same space serves text, image,
audio, video and code (alignment maps fitted and documented; their error measured). Gibberish, keyword stuffing and
token-optimised text must not beat informative content on equal topics (tested).

## 3. Activities, needs and couplings (D-18) - owners: ECON (F + G) for needs/capabilities, CIV (J) for couplings

* `crucible.innovation` provides capabilities: typed interfaces, discovery, trials, adoption. Participants try a
  service or package on their OWN activity inputs; adoption follows their own measured outcomes (latency, cost,
  failures, downstream acceptance of their outputs). World physics may use ground truth to produce consequences
  (e.g. a wrong invoice causes a dispute event), but no participant reads hidden truth or evaluator verdicts.
* Needs arise from activity bottlenecks through ONE reusable relation graph (activity requires capability/resource;
  capability lowers cost or raises outcome; usage loads capacity; scarcity moves price; skills come from education),
  not from special-case triggers. Novel capabilities find demand only when they relieve a bottleneck of some
  activity (matched by declared interface + semantic description + trial), and adopting them may change activities
  (new needs).
* `crucible.civ.couplings` registers the cross-domain relations used by civilizations (Part 10.2) as reusable relation
  records over shared state; each coupling is removable for counterfactual experiments.

## 4. World generation (D-17) - owner: CIV (workstream J)

```python
crucible.civ.PROFILES: dict[str, Profile]          # >= 10 structurally different civilization profiles (Part 10.1)
crucible.civ.generate(profile, seed, *, population="development", overrides=None, store=None) -> World
crucible.civ.recipe(profile, seed, *, population, overrides) -> dict   # the 4.0 world-family form, inspectable
```

A Tier 2 domain (`crucible.domains.base.Domain`) builds a civ recipe restricted to the subsystems, mechanism variants,
couplings, topology and population its configuration needs, plus its objective, evaluator and hidden reference.
`crucible.experience` consumes both (`make("tier2:<domain>/<config>")`, `make("tier3:<profile>")`).

## 5. Reward modes (D-19)

Everything defaults to `observation_only`. Evaluator-only tools and fields are marked in the kernel tool registry and
filtered in that mode (workstream L provides the mechanism in wave 1).

## 6. Ownership in wave 2

| workstream | owns | must not change |
|---|---|---|
| CORE-POP (B+C) | `population/`, `social/` | `semantic/` internals (use section 2), `commerce/`, `innovation/` |
| CORE-SEM (D) | `semantic/` | `population/`, `social/` (except the narrow bridge noted in integration_notes) |
| ECON (F+G) | `commerce/`, v5 parts of `economy/`, `innovation/` | `population/` (use section 1 only) |
| CIV (J) | `civ/`, new `worlds/` files | subsystem internals (compose them; request changes via integration_notes) |
