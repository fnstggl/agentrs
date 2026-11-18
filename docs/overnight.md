# Overnight multi-domain economic continual RL (single RTX A6000)

`python -m agentrs.econ.overnight --config configs/overnight/a6000.yaml` — one shared Qwen3-4B + LoRA
policy learns from verified outcomes across six locally executable domains. Supervisor programs in
`deploy/overnight/` run the trainer, the HN/Bluesky live observer, the GitHub archive observer and a
watchdog/dashboard independently of any interactive session.

## Domains and task families (32 families, all with hidden verifiers)

| domain | families (held-out = never trained; transfer test) | state / verification |
|---|---|---|
| software | bugfix, implement_function, api_repair, config_fix, dependency_fix | generated Python repos; hidden unit tests in a sandbox (nobody user, rlimits, timeout, `python -I`) |
| support (τ²-style **custom**, not τ²-Bench) | return_items, exchange_item, address_update, modify_reservation, cancel_reservation | SQLite store + airline DB, scripted customer; DB state + policy compliance (confirmation before writes, declines) |
| browser | login, checkout, form_validation, search_lookup, fill_field, dropdown; **lookup_then_fill** (held-out) | existing MiniWeb apps, server-side verifiers; actions generated as text |
| finance | payment_matching, expense_categorization, discrepancy_detection, balance_check; **budget_variance** (held-out) | SQLite ledger; exact state / numeric checks |
| crm | contact_matching, lead_qualification, pipeline_update, proposal_prep; **meeting_scheduling** (held-out) | SQLite CRM, scripted prospect; exact state checks |
| research | fact_extraction, comparison, growth_calc, ranking; **fact_verification** (held-out) | document collections with conflicting sources; numeric/exact answer + required evidence |

Splits: `train` (fresh task indices every cycle), `val` (promotion / best-checkpoint selection) and a
sealed `test` split that uses disjoint name/company/template vocabularies and is evaluated only at
stages A (pretrained), B (after BC) and C (final / best). Every family's scripted expert solves
100% of sampled tasks and an immediate-finish policy solves 0% (checked before launch).

## Learning

* **Policy**: Observation + goal + previous actions → one complete JSON tool call (or a MiniWeb
  action string), sampled token by token at T=1. Token ids and behaviour log-probs are stored;
  re-scoring them under the same weights matches to ~2e-4 nats/token (measured).
* **BC** (stage B) on verified expert demonstrations, only for training families whose untrained
  validation success is below 0.5.
* **RL** (stage C): per cycle 16 domain-balanced tasks × 4 rollouts (GRPO groups); reward = hidden
  verifier success − 0.05 × invalid actions; token-level clipped PPO, one epoch, plus a 0.05-weighted
  BC replay loss against forgetting. Samples from any other policy version are rejected.
* **Curriculum**: per-family EMA of training success; weight ∝ 4p(1−p) (learning progress), floor
  0.08, bonus for unsolved families; domain sampled uniformly first.
* **Promotion**: every 2 cycles the candidate is evaluated on validation (training families only);
  promote if ≥ promoted − 0.04, otherwise roll back to the promoted weights and halve the LR.
* **World model** (Qwen3-0.6B + LoRA): (state, action, elapsed time) → validity, outcome
  (continue/success/failure/truncated) and result text; evaluated on validation episodes against
  majority baselines. **Critic** (Qwen3-0.6B + LoRA): discounted verified return from a state.

## Checkpoints, recovery, stopping

`runs/overnight/ckpt/{initial,after_bc,latest,promoted,best,final,snap_HHMM}` — each atomic
(written to `.tmp`, renamed), with policy/critic/world-model LoRA weights, optimizer + scheduler
state, RNG state, run state (curriculum, counters, task indices) and model/code versions. The trainer
resumes from `latest` after a crash; repeated crashes (>6) jump to the final phase. The monitor
restarts a stalled trainer (≤3×), creates `STOP` on low disk or spending cap, and stops everything
after the deadline. Stop manually with `touch runs/overnight/STOP` (the trainer finishes the cycle,
evaluates on the sealed test set, writes the report and exits).

### GitHub checkpoint backups (mandatory destination)

`agentrs_backup` runs `python -m agentrs.gh_backup watch`: an initial checkpoint, then at most every
3 h (`AGENTRS_GH_BACKUP_EVERY_H`; requirement ≤ 6 h), on every new best validated policy (≥ 45 min
apart) and the final checkpoint at the end. Each upload is one GitHub Release with a split tar
(parts ≤ 1.9 GiB, under GitHub's 2 GiB asset limit), `MANIFEST.json` (parts, sizes, sha256, tar
sha256, model ids + revisions, policy version, training step, timestamp, release URL, restore
instructions) and `RESTORE.md`. The release is a draft until every asset is verified (GitHub's
server-side sha256 digest, else re-download), then published. Status: `runs/overnight/gh_backup_status.json`.

The bundle holds the policy LoRA (Qwen3-4B), world-model and critic LoRA weights, AdamW + LR
scheduler states, RNG states, the run state (curriculum, task indices, counters, evals, promotions,
task-factory state), all stored trajectories (replay), the HN/Bluesky delayed-outcome ledger,
forecasts, cursors, prequential scores and observer model, the GitHub observer model, configs and the
git commit. **The repository is public**, so third-party observer content (post text, author
pseudonyms, item references, the pseudonymisation key) is redacted from the uploaded copy; everything
learned (weights/optimizer) and all numeric outcomes are kept. Point `AGENTRS_GH_CKPT_REPO` at a
private repository the token can access for unredacted backups.

```bash
python -m agentrs.gh_backup list                                     # all checkpoint releases
python -m agentrs.gh_backup restore --dest /tmp/r --install-run runs/overnight   # latest complete one
python scripts/verify_restore.py --run runs/overnight --dest /tmp/rt # download+verify+reload+compare predictions
```

Legacy targets: `python -m agentrs.backup` (`hf://<repo>`, `rclone:<remote>:<path>`, `AGENTRS_BACKUP_TARGET`).

### Sessions

A run can be continued in a new session (same weights, optimizer, replay, curriculum; new deadline):
`echo 8 > runs/overnight/NEW_SESSION` before starting the trainer. A run stopped after BC resumes from
`ckpt/after_bc` without repeating BC. The spending cap applies per session.

## Task factory (`agentrs.task_factory`)

Every training task is admitted by the factory: environment builds, the initial state does **not**
already satisfy the hidden verifier, the scripted expert solves it in the real environment, and it is
not a duplicate instance (generator crashes are rejected, not fatal). Each task has a `TaskSpec`
(domain/family, environment + version hash, generator version, goal, deliverable, initial observation,
tools with write/terminal flags, rules, observable schema, names of the hidden private keys,
counterparties, step budget, expert steps, split, seed, provenance/permissions); admitted specs are
logged to `runs/overnight/factory/admitted_specs.jsonl`. Sampling is domain-balanced, then
learning-progress weighted per family (mastered families drop to a 3 % floor, unsolved ones get a bonus).
`python -m agentrs.task_factory.acceptance` runs the acceptance test (real expert actions, verdict flip,
persisted trajectories, memorised-solution replay across instances).

## Delayed consequences in the economic world model

Besides next-step validity/outcome/result, the world model learns (state, action, horizon, real elapsed
seconds) → task status at +1/+2/+4 actions and at the end of the task (running/success/failure/
truncated), only from finished episodes (all labels observed). Validation reports per-horizon accuracy
vs majority and the Brier score of P(success at end) vs the base rate.

**World-model-assisted action selection** (MuZero-inspired, optional): every `wm_assist_every` cycles
the same validation tasks are also run with the policy proposing greedy + `wm_assist_k` sampled actions
re-ranked by log P(valid) + log P(success at end) from the world model; one pre-registered run on the
sealed test set at the end. Direct generation remains the default; the comparison is reported as is.

## Inspecting a run

```bash
cat runs/overnight/latest_status.md            # human-readable dashboard (updated every minute)
jq . runs/overnight/latest_status.json          # machine-readable
tail -f runs/overnight/trainer.log
supervisorctl status | grep agentrs
cat runs/overnight/report.md                    # written at the end
```

## Observation pipelines (separate CPU processes)

* HN + Bluesky: the existing live learner (`agentrs.live`), workdir `runs/overnight/live`.
* GitHub: `agentrs.econ.github_observer` replays GH Archive hours in order, forecasts
  closed/merged/commented within 10m/1h/6h before reading later events, trains an online model on
  matured outcomes and scores it against a running base rate. The 2026 archive is a reduced sample
  with trimmed PR payloads, so outcomes mean "recorded in the archive", and issue→fix links are rarely
  recoverable. None of this data is reward-eligible for the economic policy.

## Asynchronous self-play runtime (`agentrs.selfplay`)

The policy acts by **direct generation**: observation + goal + history + tool schema → Qwen3-4B →
one complete JSON tool call (tool name, arguments, identifiers, generated text/code), sampled token by
token at T = 1 with no top-k/top-p and no logit masks (the only constraint is the assistant prefill
`{"tool": "` and stopping at a newline, both identical at collection and training time), so the
stored behaviour log-probs are exactly those of the sampling distribution. The candidate-scoring
policy (`agentrs.models.policy`, up to 40 candidates) is not used by this run; it remains the legacy
baseline / compatibility path.

`collector: async` (or `<run>/runtime.json`, switchable at a cycle boundary) replaces lock-step
collection with N persistent logical workers sharing **one** engine over the trainer's own policy
object (no second copy of Qwen): each worker owns its task, isolated environment instance (on a
dedicated thread — SQLite objects are thread-bound), history, budgets and timeouts; the engine
batches whatever requests are waiting (bounded queue = backpressure), records exact sampled ids,
behaviour log-probs, sampling config and the served policy version; finished episodes are appended
durably (fsync) to `experience/stream_cycle_NNNN.partial.jsonl`. Schedule per cycle: collect (all
workers on one fixed policy version) → PPO update → world model + critic → checkpoint → every 2 cycles
validation + promote/rollback. Samples whose version differs from the learner's are rejected.

`touch runs/overnight/BENCH` runs the end-to-end benchmark (lock-step 64 vs async 1/4/8/16/32 workers,
180 s each, real environments and weights) at the next cycle boundary, writes `bench.json`, and
switches to the best async configuration if it beats lock-step by > 10 % actions/h.
Tests: `tests/selfplay/test_async_runtime.py` (batching, isolation, slow env not blocking others,
failure containment, version pinning, durable sink, real Qwen3 exact log-probs + non-zero RL gradient).

## Adaptive capacity (`agentrs.capacity`, research pathway, default `capacity_mode: fixed`)

Rank expansion (function-preserving, AdamW moments transferred, checkpoints restore grown shapes
automatically), bounded expert LoRA adapters with a learned per-domain router (shared core always on;
router + experts are part of the trainable state and therefore of every GitHub checkpoint),
consolidation by distilling the routed model into the shared adapter and retiring experts, and an
`ExpansionAdvisor` that only *proposes* growth from evidence (validation plateau while the policy keeps
changing; per-family drops = interference). `scripts/capacity_benchmark.py` runs fixed vs expanded-rank
vs experts from the same checkpoint on identical tasks and update budgets. Tests:
`tests/selfplay/test_capacity.py`.
