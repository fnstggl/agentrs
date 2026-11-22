# Reproduction commands

All commands run from `superlearner/` with `PYTHONPATH=src` (or after `pip install -e .`, with `superlearner`
instead of `python -m superlearner`). Python is `/venv/main/bin/python` on the original host. Run directories go
to `runs/` (override with `SUPERLEARNER_RUNS`). GPU: one 46 GB card was used; the frozen-LLM server takes 28–34 GB,
a training process 1–5 GB during its updates, so at most two or three training runs fit next to the server
(section 2 has the queue that enforces this).

## 0. Checks

```bash
python -m superlearner doctor                         # hardware, local models, confinement probe, CPU end-to-end check
python -m pytest -q tests                             # the test suite (needs root for the confinement test)
(cd ../crucible && python -m pytest -q)               # Crucible's own 305 tests (unchanged by this work)
python -m superlearner train --config configs/smoke_cpu.yaml      # one-minute CPU smoke run without a language model
python -m superlearner meta-search --config configs/meta_smoke.yaml   # smallest end-to-end Meta-Researcher round
```

## 1. Frozen-LLM server

```bash
python -m superlearner llm-serve --model Qwen/Qwen3-14B &         # socket runs/llm.sock, cache runs/llm_cache.sqlite
```

Weights come from the local Hugging Face cache (`local_files_only=True`). Generations are cached by prompt, so a
second run of anything is much cheaper in LLM time than the first. The server limits a batch by size and by
estimated tokens (`SUPERLEARNER_LLM_MAX_BATCH`, default 8; `SUPERLEARNER_LLM_MAX_BATCH_TOKENS`, default 14000) and
releases cached GPU memory after each batch; clients wait and reconnect if it is restarted.

## 2. Everything after the warm starts, as two restartable queues

```bash
# shared warm starts: 2 verified demonstrations per arm, 60 cloning epochs (one per seed)
for s in 0 1 2; do python -m superlearner train --config configs/warmstart_few.yaml --name warm_s$s --set seed=$s; done
# Experiment A warm start: thirteen arms including LLM-written services, cloning only
python -m superlearner train --config configs/exp_a_warm.yaml

python -m superlearner jobs configs/queue_experiments.yaml                     # meta-search, B, C, D, Part 5, A, final list, self-play
python -m superlearner jobs configs/queue_llm_arms.yaml --log-dir runs/queue_llm   # RL with LLM arms + sealed list for that learner
```

`superlearner jobs` starts a job only when a GPU-training slot (`max_gpu`, default 2 next to the LLM server) and
enough CPU workers (`max_cpu`) are free, writes `runs/queue/status.json` and one log per job, and skips jobs whose
result file exists. Running the same command again after an interruption continues: training runs resume from
their last checkpoint (`train --resume`), experiments from `runs/partial/<name>.json`, the meta-search from its
ledger and the stored measures of every finished confined run.

The jobs of the queue, for running them one at a time:

```bash
# Experiment B: PPO alone and PPO + replay of own successes, branching from the same warm start
for s in 0 1 2; do for arm in exp_b_fewdemo exp_b_fewdemo_sil; do
  python -m superlearner train --config configs/$arm.yaml --name ${arm}_s$s --resume --set seed=$s --set workers=14 \
      --set iters=12 --set episodes_per_iter=96 --set eval_every=4 --set ckpt_every=4 \
      --set "init_from=runs/warm_s$s/ckpt/bc.pt"
done; done
python -m superlearner experiment b_multi --seeds 0,1,2            # -> docs/experiments/exp_b_multi.json
python -m superlearner experiment final --seeds 0,1,2              # sealed final list, once -> exp_final.json

python -m superlearner report --experiment exp_b_fewdemo_sil_s0
python -m superlearner inspect-trajectory --trajectory runs/exp_b_fewdemo_sil_s0/traj/eval_iter0012.json.gz
python -m superlearner resume --experiment exp_b_fewdemo_sil_s0 --iters 4

# Part 5 comparison, transfer, retention, world model, self-play
python -m superlearner experiment baselines --checkpoint runs/exp_b_fewdemo_sil_s0/ckpt/latest.pt
python -m superlearner experiment baselines --checkpoint runs/memory_on_sil_s0/ckpt/latest.pt --no-llm   # memory ablation
python -m superlearner experiment c --checkpoint runs/exp_b_fewdemo_sil_s0/ckpt/latest.pt
python -m superlearner experiment d --config configs/exp_d.yaml
python -m superlearner experiment d_clone                           # supervised phase B from D's checkpoint
python -m superlearner experiment worldmodel --run-dir runs/exp_b_fewdemo_sil_s0
python -m superlearner experiment selfplay --checkpoint runs/exp_b_fewdemo_sil_s0/ckpt/latest.pt --days 12

# Experiment A and the learner with LLM arms
python -m superlearner experiment a --checkpoint runs/exp_a_warm/ckpt/bc.pt --episodes 16
python -m superlearner train --config configs/exp_a_warm.yaml --name exp_a_rl --resume --set iters=4 --set eval_every=2 \
    --set ckpt_every=1 --set workers=12 --set "init_from=runs/exp_a_warm/ckpt/bc.pt"
python -m superlearner evaluate --checkpoint runs/exp_a_rl/ckpt/latest.pt --final docs/experiments/final_tasks.json

# the sealed final list is committed as docs/experiments/final_tasks.json; re-creating it gives the same seal
python -m superlearner lock-final --out /tmp/final_tasks.json --n 96
```

`runs/memory_on_sil_s0` is a link to the archived run `runs/archive/expb_v4_memory_confound/exp_b_fewdemo_sil_s0`
(the configuration of `exp_b_fewdemo_sil` with `use_memory: true, use_tools: true`, four RL iterations).

## 3. Meta-Researcher

```bash
# Experiments E and F in one run: four rounds in which every proposal of a pool of six is screened (confined, CPU,
# one seed), then the best screened candidate goes through development (3 seeds) and confirmation (5 seeds).
# --resume continues an interrupted search in the same directory (and starts a new one if there is nothing to resume).
python -m superlearner meta-search --config configs/meta_search.yaml --full-feedback --run-dir runs/meta/search4 --resume
python -m superlearner meta-search --config configs/meta_search.yaml --full-feedback --run-dir runs/meta/search4 --resume --rounds 2   # two more
python -m superlearner report --experiment runs/meta/search4
python -m superlearner experiment e --run-dir runs/meta/search4     # -> docs/experiments/exp_e_meta.json
```

Each candidate's directory (`runs/meta/search4/candidates/<id>/`) holds its full source tree, `proposal.diff`,
`proposal_full.json`, per-run job files, logs and checkpoints; the trusted results and measures are in
`runs/meta/search4/trusted/<id>/`; the ledger is `runs/meta/search4/ledger.jsonl`.

## 4. Continual learning in the weights

```bash
# calibration of the protection (short, single seed; chose conditions 4a/4b before the main runs)
for c in 0 20000 200000 2000000; do python -m superlearner lifetime --config configs/life/calib.yaml --run-dir runs/calib/ewc_$c --set algo.anchor.coef=$c; done
python -m superlearner lifetime --config configs/life/calib.yaml --run-dir runs/calib/reh --set algo.rehearsal.keep=true

# lifetimes (eight conditions, 22 runs) and the Parametric Internalization Test (2 families x 6 routes x 3 seeds),
# as one restartable queue; each run hosts its own Qwen3-0.6B backbone, no language-model server is needed
python -m superlearner jobs configs/queue_continual.yaml --log-dir runs/queue_continual
python -m superlearner jobs configs/queue_continual_b.yaml --log-dir runs/queue_continual_b   # PIT with a trainable backbone

# the Meta-Researcher's search over continual update rules (CPU, confined; hours)
python -m superlearner meta-search --config configs/meta_continual.yaml --full-feedback --run-dir runs/meta/continual1 --resume

# aggregate and regenerate the tables
python -m superlearner pit report --config configs/pit.yaml
python -m superlearner continual-report
python docs/experiments/render.py --fill docs/CONTINUAL.md
```

A lifetime that is interrupted continues from its last completed step when the same command is run again; the
queue restarts failed jobs (twice) and skips finished ones. `gpu:` in the queue file counts units of about 5 GB.

## What is and is not deterministic

* Worlds are generated from (family, seed) and Crucible replays them exactly; task lists are fixed.
* `tiny` worlds use Crucible's deterministic cost model; other families use measured CPU time, so outcomes there
  depend slightly on host load.
* GPU cloning and updates are not bit-reproducible; compared runs branch from a saved warm-start checkpoint.
* Some observations contain a value that depends on wall-clock timing (it enters `obs_features`), so two identical
  runs can differ in the sixth digit of a log-probability and, about one time in five in the smoke lifetime, in a
  sampled action. A learner rebuilt from its checkpoint is nevertheless the same learner: on the same episodes its
  next update gives bit-identical weights (`tests/restart_check.py`).
* The LLM is greedy per (prompt, variant) and cached, so identical prompts give identical text.
