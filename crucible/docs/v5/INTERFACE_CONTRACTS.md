# Crucible 5.0 interface contracts

These contracts are binding. A workstream implements the interfaces it owns exactly as written and uses other
workstreams' interfaces only as written. A change goes through the orchestrator and is recorded in DECISION_LOG.md.
Paths are relative to `crucible/`. Everything runs with `PYTHONPATH=src` from `crucible/` (the package is not installed
in the shared environment).

## 0. Rules that apply to every workstream

1. **Kernel rules (4.0, unchanged).** Handlers read state and emit events; only appliers write state. No global
   randomness: use `ctx.rng(*keys)` or `world.rng_for(...)`. Money moves only through `bank` (`ctx.call("bank.move", ...)`
   or the bank module's helpers). Event payloads are JSON-native and are never mutated after `emit`. Nothing an
   in-world actor can call returns hidden parameters, evaluator internals or other participants' private state.
   New modules pass `crucible.worldgen.admission.admit`.
2. **No fabricated outcomes.** No counter may be written except by an applier of a recorded decision event. No
   creator-assigned quality, relevance or virality score may influence any participant. No special-case success
   bonuses. No mock, stub vector, placeholder or "TODO" in shipped code or acceptance tests.
3. **Determinism.** Same spec and seed give the same event hashes. Nondeterministic computation (neural inference,
   media rendering, external tools) goes through `crucible.substrate.inference.recorded(...)` or `ctx.external`, so
   replay and regeneration use the recorded outcome.
4. **Compute.** Any command that uses more than one CPU thread, any GPU memory, or more than about 60 seconds runs
   through the slot scheduler:
   `PYTHONPATH=src python -m crucible.scale.slots --cpu N --gpu-mb M --label "<ws>: <what>" -- <command>`.
   Never start a GPU job outside it. Never kill, signal or `nice` processes you did not start.
5. **Packages.** Never upgrade or downgrade an installed package in `/venv/main` (other tenants use it). To add a
   package: `flock /tmp/crucible-pip.lock /venv/main/bin/python -m pip install --dry-run <pkg>` must show no change to
   an installed package; then install it the same way without `--dry-run`, and add it to an optional extra in
   `pyproject.toml` (`v5-<area>`).
6. **Disk.** About 26 GB are free for everything. Datasets go under `$CRUCIBLE_DATA` (default
   `/workspace/data/crucible`), model weights in the existing HF cache. Never commit a file larger than 5 MB. Budgets:
   calibration datasets 8 GB in total, encoder weights 3 GB.
7. **Tests.** Unit and integration tests go in `tests/test_v5_<area>_*.py`. Each gate owner writes
   `tests/v5/test_gate_<G>.py`: one test per requirement it owns, which measures the requirement on the real system
   and calls `crucible.acceptance.v5.evidence.record(...)`. Tests must not skip silently; a test that cannot run fails.
   Long tests carry `@pytest.mark.slow` and a `@pytest.mark.timeout(...)`.
8. **Manifest.** Do not edit `docs/v5/ACCEPTANCE_MANIFEST.json`. Report, per requirement, the implementation files and
   test files in `docs/v5/workstreams/<WS>.json` (format in section 13); the orchestrator merges them.
9. **Git.** Commit on your own branch (your worktree). Commit often; a WIP commit is better than lost work. Never
   force-push, never push, never commit secrets, never commit evidence files under `docs/v5/evidence/` (those are
   produced on the integration branch).

## 1. Substrate (`crucible.substrate`, workstream A)

### 1.1 Artifacts: `crucible.substrate.artifacts`

Kernel module `artifacts`, host `artifact`. Content-addressed and immutable.

```python
REF = "sha256:<64 lowercase hex>"
def put(ctx, data: bytes, mime: str, *, owner: str, parents: Sequence[str] = (), meta: dict | None = None,
        kind: str = "file") -> str            # stores bytes, emits artifact.stored, returns REF (idempotent)
def get(world, ref: str) -> bytes             # verifies SHA-256; raises CrucibleError(not_found) if absent
def info(world, ref: str) -> dict             # {ref, mime, size, owner, parents, meta, kind, published, t}
def publish(ctx, ref: str, *, audience: str = "public") -> None   # emits artifact.published
def readable(world, principal: str, ref: str) -> bool             # owner, granted, or published
def put_array_blob(ctx, arrays: dict[str, np.ndarray]) -> str     # npz (no pickle) owned by 'sys', kind 'batch'
def get_array_blob(world, ref: str) -> dict[str, np.ndarray]
```

Events: `artifact.stored {ref, mime, size, owner, parents, meta, kind}` (vis: owner, internal for `sys`),
`artifact.published {ref, audience}` (public). Bytes are kept in the world's store under blob key `art/<hex>` (memory
when the world has no store) and in `world.runtime["artifacts"]`; state keeps the metadata table `artifacts`. Limits:
32 MiB per artifact, `quota_bytes` per owner (default 512 MiB). Tools: `artifact.put {data_b64, mime, kind?, parents?,
meta?}`, `artifact.get {ref}` (returns base64, readable only), `artifact.info {ref}`, `artifact.list {owner?}`.
`artifact.export(world, dest_store)` copies referenced bytes when a world moves to another store.

### 1.2 Columns and batch events: `crucible.substrate.columns`

```python
def declare(world, table: str, n: int, columns: dict[str, tuple[str, tuple[int, ...], float | int]]) -> None
    # in Module.install: arrays named f"{table}.{col}" with dtype, trailing shape and fill value
def col(world, table: str, name: str) -> np.ndarray              # read-only view
def batch(ctx, table: str, kind: str, idx: np.ndarray, values: dict[str, np.ndarray], *, mode: str = "set",
          vis="internal", extra: dict | None = None) -> str
    # mode "set" | "add" | "max" | "min"; writes the arrays to an array blob and emits
    # cols.batch {table, kind, ref, n, cols, mode, extra}; the applier verifies the blob and applies it.
def grow(ctx, table: str, n_new: int) -> None                     # emits cols.grow; appends rows with fill values
```

Batch payload sizes are bounded by the blob, not the event. All population-scale state changes go through `batch`.

### 1.3 Identity indices: `crucible.substrate.ids`

`pid(i: int) -> "p<i>"`, `index(principal: str) -> int | None`, `ensure_principal(ctx, i, kind="person") -> str`
(creates the kernel principal and person record on first use; idempotent). A principal created this way shares the
individual's column state; it never duplicates it.

### 1.4 Typed cross-module events: `crucible.substrate.bus`

```python
def event_type(name: str, schema: dict, owner: str, doc: str = "") -> None   # registers "<host>.<what>"
def emit(ctx, name: str, payload: dict, vis=None) -> Event                     # validates in strict worlds
def subscribe(world, name: str, fn: Callable[[Ctx, Event], None], owner: str) -> None
    # called in the emitting transaction right after the applier; a failure rolls the whole command back
def types() -> dict[str, dict]                                                 # name -> {schema, owner, doc}
```

Subscribers are registered in `Module.install` and must be deterministic. A subscriber may emit further events and
call tools through `ctx.call`; recursion depth is limited by the kernel.

### 1.5 Shared primitives: `crucible.substrate.primitives`

One owner per shared fact. `owner(kind) -> {"module": str, "table": str}` for the kinds: `identity`, `organization`,
`account`, `money`, `asset`, `resource_inventory`, `content_artifact`, `product`, `service`, `software_repository`,
`relationship`, `community`, `institution`, `contract`, `knowledge_artifact`, `tool_definition`, `time`,
`provenance`, `permission`, `visibility`. `check_no_duplicates(world)` fails if two modules declare a table for the
same kind. New kinds are added only by workstream A.

### 1.6 Attention and time: `crucible.substrate.attention`

Columns `pop.attention_budget_min` and `pop.attention_left_min` (float32), reset daily by the population.

```python
def spend(ctx, idx: np.ndarray, minutes: np.ndarray, *, activity: str) -> np.ndarray   # granted minutes (<= left)
```
emits one `cols.batch` (`kind="attention.<activity>"`); invariant: left >= 0 and spent per day <= budget.

### 1.7 Recorded inference: `crucible.substrate.inference`

```python
def recorded(ctx, kind: str, model_id: str, model_hash: str, inputs: Any, fn: Callable[[], Any]) -> Any
    # wraps ctx.external; the record holds {model_id, model_hash, inputs_digest, output}; replay and regeneration
    # return the recorded output without running fn
def weights_hash(module_or_state_dict) -> str     # SHA-256 over sorted parameter tensors (dtype, shape, bytes)
```

## 2. Population (`crucible.population`, workstream B)

Kernel module `pop` (host `pop`). Config:
`{profile: "development"|"standard"|"large"|"internet", n?: int, interest_dim: 32, classes?: {adaptive, strategic,
semantic}, income_per_day_cents?: [lo, hi], seed_mix?: {...}}`. `crucible.population.PROFILES` holds the four
profiles of Part 3.3.

Columns of table `pop` (at least): `interests[d] f16`, `knowledge[d] f16`, `skills[k] f16`, `goals[g] u8`,
`attention_budget_min`, `attention_left_min`, `wallet_cents i64` (spendable money; individuals that transact get bank
accounts through `ids.ensure_principal` and their payments are real `bank` moves), `satisfaction f32`,
`trust[...]` (sparse, owned by social), `habit[platform] f32` (visit propensity), `employer i32`, `beliefs[b] f16`,
`next_active_at i64`, `recent[k] i32` (recent items), `class u8`, `policy_group i16`, `memory[m] f16` (class 2).

```python
def register_activity(world, name: str, handler, *, owner: str) -> None
    # handler(ctx, idx: np.ndarray, now: int) -> None is called with the cohort that activates for `name`
def wake(ctx, idx: np.ndarray, at: int, activity: str) -> None     # schedule activations (messages, notifications)
def features(world, idx: np.ndarray) -> dict[str, np.ndarray]     # documented individual features for models
def class_members(world, cls: int) -> np.ndarray
```

Process `pop.tick` (period configurable, default 15 simulated minutes) activates individuals whose
`next_active_at <= now` and dispatches them to activities by learned habit. Daily process resets attention.
Class-2 policies: `crucible.population.adaptive.AdaptivePolicy` (shared torch weights, per-individual `memory`
column, `act_batch(features, memory) -> (actions, new_memory)`, `learn(batch) -> {loss, weights_hash, n}`); every
update emits `pop.policy_update {group, step, loss, weights_hash, n_examples}`.

## 3. Social internet (`crucible.social`, workstream C)

Content catalogue table `content` (module `content`, host `content`):
`{id, ref (artifact REF), kind: video_short|video_long|post|image|article|newsletter|audio|thread|app|package,
platform, creator (principal), published_at, duration_s, title, tags, removed}`.

```python
def publish(ctx, creator: str, ref: str, kind: str, platform: str, title: str, *, tags=(), duration_s=None) -> str
def consumption(ctx, platform: str, viewers: np.ndarray, items: np.ndarray, watch_s: np.ndarray,
                completed: np.ndarray, shared: np.ndarray, followed: np.ndarray, liked: np.ndarray) -> None
    # the ONLY way audience state changes: one cols.batch of kind "consume.<platform>"; appliers update per-item
    # viewer bitsets and counters; a view requires watch_s >= min(30, 0.5 * duration_s)
def impressions(ctx, platform: str, viewers: np.ndarray, items: np.ndarray, slot: np.ndarray) -> None
def audience(world, item: int, *, principal: str | None = None) -> dict
    # exact for the evaluator; creator view (principal = creator) is delayed and noisy as configured
```

Audience quantities: `impressions, unique_exposed, views, unique_viewers, watch_s, completions, repeat_views, shares,
subscribers, returning_viewers`. Recommenders:

```python
class Recommender(Protocol):
    name: str; version: str
    def rank(self, world, platform: str, viewers: np.ndarray, candidates: np.ndarray, k: int, rng) -> np.ndarray
        # [len(viewers), k] item ids
crucible.social.recsys.REGISTRY: dict[str, type[Recommender]]
```
Platform config: `{"recommenders": [{"name": ..., "weight": ..., "params": {...}}], "schedule": [{"day": d,
"recommenders": [...]}]}`; parameters live in evaluator-only state (`world.runtime["recsys_private"]`), never in
events visible to actors. Communities, follows, subscriptions, messages and shares are module state with endpoints
for agents and batch decisions for the population.

## 4. Semantics (`crucible.semantic`, workstream D)

```python
def encode(world, ref: str, modality: str) -> np.ndarray      # float32 [dim]; modality text|image|audio|video|code
def encoder_info(modality: str) -> dict                       # {id, version, dim, params, preprocessing_hash}
def cache_key(ref, modality) -> str                           # f"{ref}|{encoder_id}@{version}|{preprocessing_hash}"
class SemanticService:                                        # one shared model per (model_id, revision)
    def __init__(self, model_id="Qwen/Qwen3-0.6B", device="auto", max_batch=16, max_new_tokens=128): ...
    def generate(self, ctx_or_none, requests: list[dict]) -> list[dict]   # {messages, max_new_tokens, temperature, seed}
    def info(self) -> dict                                    # {model_id, revision, params, weights_hash, device}
```
Encoders run real pretrained models (no stubs); embeddings are recorded through `substrate.inference.recorded` and
cached on disk under `$CRUCIBLE_CACHE/embeddings`. Semantic participants are `crucible.agents.base.Policy`
implementations that act only through tools, sharing one `SemanticService`.

## 5. Calibration (`crucible.calibration`, workstream E)

```python
def datasets() -> dict[str, dict]       # name -> {url, license, sha256 per file, size, schema, provenance}
def acquire(name, dest=None) -> str     # downloads, verifies checksums, returns the directory
def load(model_name: str) -> FittedModel  # e.g. "content_consumption@1"
class FittedModel:
    name; version; features: list[str]; target; metrics: dict; baseline_metrics: dict; provenance: dict
    limits: str                         # extrapolation limits, in words
    def predict(self, X: np.ndarray) -> np.ndarray
```
Checkpoints (< 5 MB each) live in `docs/v5/calibration/models/<name>/` with `model_card.json`; evaluation reports in
`docs/v5/calibration/reports/`. World code depends only on `load`; it never reads datasets.

## 6. Commerce (`crucible.commerce`, workstream F)

Module `subs` (host `subs`): `subs.create_plan {product, price_cents, interval: month|year, trial_days?}`,
`subs.subscribe {plan}` (the caller's own account pays), `subs.cancel {subscription, at_period_end}`,
`subs.refund {invoice, amount_cents?, reason}`, `subs.plans`, `subs.mine`; process `subs.bill` charges due
subscriptions through `bank`. Evaluator metrics (goals/metrics.py `@metric`): `mrr_cents`, `arr_cents` (= 12 x MRR),
`active_subscriptions`, `churn_rate`, `independent_customers`, `profit_cents`; all exclude related parties and
circular flows (`crucible.commerce.integrity.related_flows(world, owner)`). Products, licensing, one-time sales,
bankruptcy and institutions are further modules in this package.

## 7. Innovation (`crucible.innovation`, workstream G)

Capability registry module `caps` (host `caps`): `caps.declare {service_host, capability: {name, inputs schema,
outputs schema, doc}, tests_ref?}`, `caps.search {query}`, `caps.try {service_host, capability, input}`. Needs graph:
`crucible.innovation.needs` with relation types `requires`, `consumes`, `produces`, `constrains`, `substitutes`,
`complements`; process `needs.update` derives new needs from world state through these relations only. Software:
module `pkg` (packages, versions, dependency edges, releases, adoption only when a dependent deploys or executes it).

## 8. Tier 1 (`crucible.rapid`, workstream H)

```python
@dataclass
class ActionSpace:  kinds: dict[str, dict]        # name -> JSON schema of one action of that kind
    # kinds used: levers, discrete, graph_edit, sequence, query, experiment, message, program, allocate, coordinate
class RapidEnv(Protocol):
    family: str; seed: int; spec: dict
    def reset(self) -> dict                         # observation (crucible.experience Observation dict)
    def step(self, actions: dict[str, Any]) -> dict # {observation, result: {ok, error?}, events}
    def outcome(self) -> dict                       # evaluator side only
def make(family: str, seed: int, **difficulty) -> RapidEnv
crucible.rapid.primitives.REGISTRY  # 28 primitives, versioned: name -> {version, doc, params schema, implementation}
crucible.rapid.families.REGISTRY    # name -> {engine, category (8.1-8.10), split: train|val|test, fingerprint, doc}
def fingerprint(family) -> dict     # motifs, topology, temporal, observation, action interface, objective, engine
def admit(family, seeds) -> dict    # the Part 7.4 checks, oracle evaluator-side only
```
Numeric-lever families also expose the 4.0 `EnvSpec`/`MechVecEnv` view so existing learners keep working. 4.0
validation and sealed families are never moved into training.

## 9. Tier 2 domains (`crucible.domains`, workstream I)

Each domain is `crucible/domains/<name>.py` with `DOMAIN = Domain(name, configs: {train: [>=3], eval_compositional:
[...], eval_structural: [...]}, build(config, seed) -> WorldSpec, objective(world, difficulty) -> dict, evaluate(world,
principal, objective) -> {success, score, details}, reference(world, principal) -> scripted policy (hidden from
agents))` and registers world families `d5_<name>_<config>`.

## 10. Tier 3 civilizations (`crucible.civ`, workstream J)

`generate(profile: str, seed: int, population_profile: str = "standard") -> WorldSpec`; `PROFILES` (10, structurally
different); couplings are module-level subscriptions and needs relations, never scripted narrative events;
`chains.trace(world, from_event) -> list[stage]`, `chains.counterfactual(world, at_seq, removal) -> effect`.

## 11. Learner interface (`crucible.experience`, workstream L)

```python
class Observation(TypedDict):          # every tier
    t: int; agent: str; numbers: dict[str, float]; arrays: dict[str, list]; text: list[str]
    artifacts: list[dict]; tools: list[dict]; events: list[dict]; entities: list[dict]; resources: dict
    analytics: dict; last_result: dict | None; missing: list[str]; cursor: int; objective: str | None
class ExperienceEnv(Protocol):
    reward_mode: Literal["legacy_shaped_reward", "verified_terminal_outcome", "observation_only"]
    def agents(self) -> list[str]
    def observe(self, agent: str) -> Observation
    def execute(self, agent: str, action: dict) -> dict          # {ok, result | error, cost_s, t}
    def observe_since(self, agent: str, cursor: int) -> tuple[list[dict], int]
    def tools(self, agent: str) -> list[dict]
    def advance(self, seconds: int) -> None                       # worlds with time
    def reward(self, agent: str) -> float                         # 0.0 in observation_only, always
    def verify(self) -> dict                                      # evaluator side; never part of an Observation
def make(tier: int, name: str, seed: int, reward_mode=..., **kw) -> ExperienceEnv
class ExperienceStore:  append(record), iterate(...), shards on disk, content hashes
crucible.experience.reference.ReferenceLearner   # small neural learner, real updates, save/load
```

In `observation_only` no observation field may carry reward, shaped progress, objective satisfaction or evaluator
data; legitimate world consequences (balances, revenue, costs, analytics a participant could see) remain.

## 12. Acceptance evidence (`crucible.acceptance.v5`, orchestrator)

`evidence.record(requirement_id, metrics: dict, *, command=None, artifacts=(), notes="", datasets=None, labels=None)` -
metrics must include every metric named in the requirement's manifest criteria, and must be measured: the producing
test asserts on the measured values (`assert not checker.unmet(rid, metrics)` is the idiom), records non-constant
metrics, imports at least one of the requirement's implementation modules, and never skips, returns early, swallows
exceptions, mocks or leaves patches in place. `datasets={name: path}` records checksums; `labels` mark how a result was
obtained (`{"scripted": True}` for scripted reference policies, `{"learned": True}`, `{"calibrated": True}`,
`{"approximation": True}`); the final report separates results by them. Evidence is produced on the integration
branch by `python -m crucible.acceptance.v5 run --gate <G> --agent <name>` in a fresh clone of the committed tree (a
record written anywhere else is never evidence); an independent reviewer re-executes and reproduces it with
`... review --requirement <id> --agent <other>`; both are committed. The verdict comes only from
`python -m crucible.acceptance.v5 --verify-all` (decision D-24, `docs/v5/components/P.md`). Workstreams never edit
the manifest or the ledger; their reports (section 13) are merged by `merge-workstreams`.

## 13. Workstream report file (`docs/v5/workstreams/<WS>.json`)

```json
{"workstream": "C", "branch": "...", "commit": "...",
 "requirements": [{"id": "C.1", "implementation_files": ["src/crucible/social/content.py"],
                   "tests": ["tests/v5/test_gate_C.py", "tests/test_v5_social_content.py"],
                   "state": "implemented|partial|not_started", "notes": "..."}],
 "dependencies_used": ["A"], "limitations": ["..."], "integration_notes": ["..."]}
```
