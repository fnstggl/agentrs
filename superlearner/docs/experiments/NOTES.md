# Lab notes (chronological, 2026-11-21)

Raw findings as they happened, including the runs that were stopped or superseded. Run directories named here are
under `runs/` (not committed); archived ones under `runs/archive/`. The curated account is `docs/RESULTS.md`.

## Setup facts

* Host: 1× RTX A6000 (46 GB), 48 logical cores (2× Xeon Gold 6226), 188 GB RAM, unprivileged Docker container
  (pids.max 2816), `/workspace` is not a volume. Qwen3-0.6B/1.7B/4B/14B cached locally. No API model.
* Git: `feat/crucible-internet` 11e5613 and `feat/crucible-open-world` 58c9b8b were already on `origin` (the brief
  said "reportedly local-only"); `feat/cam-2` c25e6fd was one commit ahead of its upstream and unpushed. A bundle of
  all local branches was written to `/workspace/backups/agentrs-all-branches-2026-11-21.bundle` (verified).
* Baseline: 305 Crucible tests pass on the new branch before any change (305 dots, 0 F/E).

## Frozen LLM (before any controller existed)

* Probe "write the files of a service for need X" with only public documents in the prompt (6 hidden inputs each):
  Qwen3-4B wrote `service.py` and omitted the manifest for all 8 needs. Qwen3-14B produced running code for 5/8
  (sort.records 1.0, route.shortest 1.0, schedule.jobs 0.95, match.stable 1.0, forecast.series 0.5 vs reference
  0.68; route.tsp 0, pack.knapsack timeout, text.extract JSON error). ~18 tokens/s; 13–37 s per answer.
* Crucible's `render_tool` shows only the first line of tool descriptions; the deploy manifest format is on later
  lines of `cloud.deploy`. Earlier LLM agents never saw it.
* Tool retrieval with mean-pooled hidden states: rank of `cloud.deploy` for "commit files for a deployable service"
  14; `market.list` for "have an active market listing" 98. Prompt-based last-token embedding, layer −4, centred:
  2 and 1. (Layer −1: 7 and 9; layer −12: 3 and 14.)
* One-shot JSON-in-JSON files: manifests with unbalanced braces. Entry-by-entry writing fixed the escaping; the
  manifest's `params` was then written as a property map instead of a JSON schema in every non-reasoning attempt
  (2/2) and correctly in reasoning mode (2/2 first attempts valid). Reasoning mode for *code* entries exhausted its
  2,700-token budget and returned nothing (153 s wasted); it is now used for structured entries only, with a
  fallback.
* LLM reasoning mode for algorithm proposals: 2,400 tokens, 578 s, empty answer. Turned off for the proposer.
  Without it: `advantages` proposal in 222 s (valid code: "moving average of value estimates"), `policy_loss`
  proposal in 70 s returned the unchanged function with a hypothesis claiming a change (now rejected as a no-op).

## Demonstrations and cloning

* Teacher scripts through the controller's interface, all 13 arms, 52 episodes: apicall 0.83, code 0.5,
  deploy/scratch 0.29 (2/7: LLM-written service deployed live), service/scratch 0/3, revenue 0/2 (small n).
* The teacher listed services under a randomly chosen need, not the one its code implements (found in the arena:
  330–400 requests/day, all errors). After fixing it to read its own repository name: 191/191 demonstration
  episodes verified on the eight LLM-free arms (before: 146/191; earlier 55/94).
* Cloning, 16 demonstrations per arm, full controller, shared critic: greedy held-out success 0.87 / 0.97 / 0.88 /
  1.00 after 30 / 60 / 100 / 150 epochs (64 tasks).
* Cloning, 2 demonstrations per arm, 60 epochs, same seed and demonstrations: 0.338 with the critic sharing the
  controller's state, 0.750 with the critic detached. Two detached-critic runs with identical settings: 0.775 and
  0.512 (GPU nondeterminism + 16 demonstrations) — hence shared warm-start checkpoints for comparisons.
  Three shared warm starts (seeds 0/1/2): 0.613, 0.762, 0.500 on the 80-task curriculum-split list.
* The same checkpoint on the same 80 tasks evaluated in different processes under different host load: 0.613,
  0.650, 0.662, 0.675, 0.688 (measured-CPU cost model outside `tiny`).

## Reinforcement learning from the warm start (superseded attempts)

1. `archive/exp_b_fewdemo_ppo_shared_critic_v1`: PPO, shared critic, gated curriculum. Held-out success
   0.338 → 0.263 → 0.250 at iterations 0 / 5 / 10. Stopped.
2. `archive/exp_b_fewdemo_sil_v1_gated`: PPO + replay, detached critic, gated curriculum. 0.750 → 0.425 at
   iteration 5. Sampled training success on `fund` rose 0.09 → 0.82 in 7 iterations (the replay learns it), but arms
   the warm start had solved and the curriculum had not yet unlocked (service from deployed / coded) went from 1.0
   to 0.0: forgetting by interference. Gating switched off for warm-started runs.
3. `archive/expb_v3_unstable_first_update` (three arms, shared warm starts, ungated, affinity feature): all dropped
   at iteration 4 (0.675 → 0.475, 0.762 → 0.562, 0.650 → 0.475). Diagnosis from metrics: the first policy update
   took 2 gradient steps and measured KL 0.13–0.17; sampled success on deploy-from-coded went 0.42 → 0.0 in that one
   update. Cause: a fresh Adam state moves every weight by the full learning rate on its first steps. Fix: linear
   learning-rate warm-up (40 steps) and a KL check after every step with rollback and learning-rate halving.

4. `archive/expb_v4_memory_confound` (warm-up and rollback added): still 0.650 → 0.388, 0.775 → 0.500 (PPO) and
   0.600 → 0.512 (PPO + replay) at iteration 4. In the PPO seed-1 run every policy step of iterations 2–4 had been
   rolled back, so the weights that scored 0.500 were the warm-start weights that had scored 0.775. Direct test,
   same checkpoint, same 80 tasks: **0.5125 with the episodic memory and tool library accumulated during training,
   0.7625 with them reset.** The drop in all four attempts was (at least largely) memory and agent-made tools
   entering the context of a policy that had been cloned without them, not RL damaging the weights. Experiment B
   now runs with memory and tool-making off; memory is studied separately.
   Also found here: the warm-up counted the 36 critic-only steps, so the first policy step was at 93% of the
   learning rate and was rolled back in every update until the KL-adapted scale reached 1/8–1/16. Fixed (warm-up
   counts policy steps).
   Also: greedy evaluation of `fund` is 0.0 while the same policy sampled succeeds 0.50–0.75 of the time (the
   argmax repeats one failing call; samples get past it). Evaluation now reports greedy and sampled success.

## Infrastructure incidents

* Rollouts first ran at 0.55 s per decision instead of ~0.03 s: every failure message and every listing was a new
  text sent to the 14B model for embedding. Fixed by embedding normalised failure messages, only the four most
  recent documents per decision, and at most eight documents per result (200 demonstration episodes: 65 s).
* Embeddings queued behind long generations: the server now serves them from a second thread.
* Starting four runs at once exhausted the container's thread limit (each spawned interpreter starts 48 BLAS
  threads while importing numpy); workers now start with single-thread environment variables. Two arms were lost
  to GPU out-of-memory with five trainers next to the 31 GB LLM server; updates now restore and retry.
* Stopped runs left worker processes running episodes; workers now ask the kernel to kill them when the parent dies.
* A few world seeds fail in Crucible's world generation (`InsufficientFunds` at a genesis bounty); fixed task lists
  skip them.
* Inside the confinement: no CUDA; `/dev/shm` not available (weights are sent as numpy); `ctypes.util.find_library`
  tries to run programs (libseccomp is loaded by soname).

## Meta-Researcher

* Smoke run (`configs/meta_smoke.yaml`): a learned-rule candidate passed the confined self-test (likelihood error
  9.5e-7, all nine checks), was screened and developed against the champion on the same seeds; with a 400-decision
  budget both were identical (difference 0.0000) — the pipeline ran, the budget was too small to learn anything.
* Confined champion baseline, first inner configuration: cloning had not converged (loss 9.4 → 7.0 in 40
  one-step epochs), held-out success 0.17 flat. With 240 cloning steps: warm start 0.58, PPO flat at 0.58
  (AULC 0.538), 6 of 10 updates stopped early on KL, explained variance of the critic ≤ 0.
* `runs/meta/search1` (kept as a record, not used for results): the champion's screening run finished (AULC 0.696,
  held-out success 0.67 → 0.71), then all six candidates of the first pool were rejected by the confined self-test
  with the same failing check, "gradients reach policy and value". The check was wrong, not the candidates: it
  measured the loss of the first update, which under the configured critic-only warm-up deliberately sends no
  gradient to the policy. The self-test now tests the loss used after the warm-up; the search was restarted from
  scratch as `runs/meta/search2` (new frozen source snapshot, new champion baseline). The smoke configuration now
  uses the same warm-up settings so the test suite covers this path.
* GPU contention again (15:00): with the 32 GB language-model server, three Experiment B trainers, the Experiment A
  cloning run and three experiment processes on one 46 GB card, the transfer and retention experiments died of
  out-of-memory — the retention run inside the code that restores the weights after a failed update (a 2 MB
  allocation). Restores and checkpoint loads now wait and retry; experiments that only evaluate keep their
  controller on the CPU. Both experiments were rerun from the start (logs of the failed attempts:
  `runs/x_c_oom.log`, `runs/x_d_oom.log`).
* `runs/meta/search2` (kept, not used): the champion's screening run needed more than its 2,400 s wall-clock limit
  because the host was running eight other jobs (load average 42; 313 of 400 training episodes after 1,974 s; the
  same run had taken 1,376 s an hour earlier). A wall-clock limit that depends on host load would have invalidated
  the baseline, so the search was stopped before the limit and restarted as `runs/meta/search3` with a smaller
  per-run budget (8 iterations of 32 episodes, 4,000 decisions) and a 5,400 s limit that only catches runaway
  candidates. The experience budget, enforced by the env hub, is what makes runs comparable; wall time is recorded.
* `exp_b_fewdemo_sil_s1` (PPO + replay, seed 1) died of the same GPU out-of-memory during its tenth update
  (15:27). It was resumed from its iteration-8 checkpoint with the retrying restore; iteration 9 was therefore run
  twice (the first attempt's record is kept in `metrics_before_oom_resume.jsonl`, the curve uses the second).

## Outage and recovery (2026-11-21, 15:44 → 21:35)

* The GPU instance ran out of credit at 15:44; every process stopped. On return (21:35): the filesystem was intact,
  no process had survived, the GPU was empty. State found on disk: five of the six Experiment B runs complete
  (12 iterations, checkpoints); `exp_b_fewdemo_sil_s1` at iteration 9 of 12; world-model and memory-ablation results
  saved; the transfer, retention, Part 5 and Experiment A runs interrupted before writing their results; the
  third meta-search attempt interrupted during its champion baseline.
* Two faults of the code that the interruption exposed:
  1. The replay buffer of a run's own successes was not part of its checkpoints, so a resumed "PPO + replay" run
     would have continued with an empty buffer. Checkpoints now store it (`latest.pt.replay.pkl.gz`), and a resumed
     run moves the log records of iterations after its checkpoint to `*.interrupted`. Because the interrupted
     seed-1 replay run could not be resumed faithfully, it was archived
     (`runs/archive/exp_b_fewdemo_sil_s1_interrupted_at_iter9`) and run again from its warm start.
  2. Experiments wrote their results only at the very end. They now write every finished part to
     `runs/partial/` and continue from there; the meta-search stores the trusted measures of every finished
     confined run and reuses them when it is started again on the same directory.
* Scheduling. Before the outage nine jobs shared one GPU with the language-model server, whose memory had grown from
  31 to 35 GB (attention caches of batched 7,000-token prompts). Now: the server limits a batch by estimated tokens
  and releases cached GPU memory after every batch (28.5 GB after loading); clients reconnect if the server is
  restarted; evaluation-only experiments stay on the CPU; and all remaining work is one queue
  (`configs/queue_experiments.yaml`, `superlearner jobs`) that admits at most two GPU-training jobs and 46 rollout
  workers at a time and skips jobs whose results exist.
* Retention experiment: phase A and two of the four phase-B strategies had finished and printed their success rates
  (all exactly 48/48 or 0/48) before the outage. Those parts were entered into the partial-results file from the
  kept log (`runs/x_d.log`), marked as such, instead of being run again; the other two strategies were run after
  the outage from the saved phase-A checkpoint.

## After the recovery (21:45 → 00:20)

* All remaining work ran as two queues (`configs/queue_experiments.yaml`, `configs/queue_llm_arms.yaml`); nothing
  died of out-of-memory. Iterations took about a third of the time they had taken before the outage (58 s of
  rollouts and 2–15 s of updates per Experiment B iteration instead of 160 s and 10–25 s).
* LLM throughput was the next bottleneck: about 3 generations per minute, because only requests with identical
  length limit and reasoning flag were batched. The server now batches greedy requests with different limits (each
  row stops at its own limit; checked against one-by-one generation on Qwen3-0.6B: same text for five mixed
  requests), grouped by the size of the limit so that short requests do not wait for long ones. After two restarts
  of the server under running jobs (103 and 72 client reconnects, no failure): 6–8 generations per minute. A
  batched reasoning request produced the same answer through a different reasoning text than unbatched — batching
  is not bit-identical; the cache fixes whichever was produced first.
* Meta-search, fourth attempt (`runs/meta/search4`), ran to the end: champion baseline 714 s, four rounds of 12–30
  minutes, follow-ups. The first two candidates the frozen LLM wrote passed every check and trained normally; read
  as code, one added a variable that is never used and the other edited the replay loss, which the champion never
  calls. Their "effects" (+0.038, −0.006) are noise.
* Inert-change probe, three versions. (1) One whole update on the demonstration batch, compared by fingerprint
  with the champion: called six candidates inert, among them a rule-space "rollback" policy loss and an
  LLM-written wider clip — false positives, because that single update never left the trust region (the KL stop
  ended it). (2) Rule loss and gradient with perturbed old log-probabilities added: four inert, among them an
  LLM-written discount on truncated episodes — a false positive, because the demonstration batch has no truncated
  episode. (3) Every second episode of the batch marked truncated and failed: three inert, each confirmed by
  reading the diff (unused variable; replay loss never reached; "asymmetric" clipping that is the same function
  under the `min`). The first two versions' outputs are kept in `runs/meta/search4/probe_v1_single_update` and
  `probe_v2_no_truncation`.
* The search's own follow-up took only the best screened candidate (+0.082 at seed 1 → +0.005 over three seeds).
  Because of the noise finding, the search was resumed with `develop_top: 4`; the next three candidates ended at
  +0.017, +0.004 (five seeds) and +0.003. The first candidate's record appears twice in the ledger (the resumed run
  re-read its stored runs and wrote it again); summaries use the last record per candidate.
* Self-play in the bazaar world: no participant, scripted or learned, was sent a request in 12 days, so that world
  says nothing. Added a scripted-control reference and a second world (`tiny:7`) where scripted participants do
  trade (13,372 and 8,590 requests). The learned controllers traded in neither.
* Retention: RL on the new task set learned nothing (0% → 0%), so nothing could be forgotten. Added the supervised
  variant (cloning the new set from the same checkpoint), which shows forgetting (100% → 33%) and its prevention
  by rehearsal (100%).
* Crucible's 305 tests pass on the final tree (12 min 24 s); 41 superlearner tests pass.
