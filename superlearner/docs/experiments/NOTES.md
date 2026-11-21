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
