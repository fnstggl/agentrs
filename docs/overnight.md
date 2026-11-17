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

**External backups are only as good as the configured target**: `python -m agentrs.backup` supports
`hf://<repo>` and `rclone:<remote>:<path>` with verified uploads and tested restores, but nothing is
uploaded unless a target and credentials are configured (`AGENTRS_BACKUP_TARGET`).

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
