# Reproduction commands

All commands run from `superlearner/` with `PYTHONPATH=src` (or after `pip install -e .`, with `superlearner`
instead of `python -m superlearner`). Python is `/venv/main/bin/python` on the original host. Run directories go
to `runs/` (override with `SUPERLEARNER_RUNS`). GPU: one 46 GB card was used; the frozen-LLM server takes about
31 GB, each training process 4–7 GB.

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
second run of anything is much cheaper in LLM time than the first.

## 2. Inner learner

```bash
# shared warm starts: 2 verified demonstrations per arm, 60 cloning epochs (one per seed)
for s in 0 1 2; do python -m superlearner train --config configs/warmstart_few.yaml --name warm_s$s --set seed=$s; done

# Experiment B: PPO alone and PPO + replay of own successes, branching from the same warm start
for s in 0 1 2; do for arm in exp_b_fewdemo exp_b_fewdemo_sil; do
  python -m superlearner train --config configs/$arm.yaml --name ${arm}_s$s --set seed=$s --set workers=14 \
      --set iters=12 --set episodes_per_iter=96 --set eval_every=4 --set ckpt_every=4 \
      --set "init_from=\"runs/warm_s$s/ckpt/bc.pt\""
done; done
python -m superlearner experiment b_multi --seeds 0,1,2            # -> docs/experiments/exp_b_multi.json
python -m superlearner experiment final --seeds 0,1,2              # sealed final list, once -> exp_final.json

python -m superlearner report --experiment exp_b_fewdemo_sil_s0
python -m superlearner inspect-trajectory --trajectory runs/exp_b_fewdemo_sil_s0/traj/eval_iter0012.json.gz
python -m superlearner resume --experiment exp_b_fewdemo_sil_s0 --iters 4

# Part 5 comparison, transfer, retention, world model, self-play
python -m superlearner experiment baselines --checkpoint runs/exp_b_fewdemo_sil_s0/ckpt/latest.pt
python -m superlearner experiment baselines --checkpoint <a run trained with memory on>/ckpt/latest.pt --no-llm
python -m superlearner experiment c --checkpoint runs/exp_b_fewdemo_sil_s0/ckpt/latest.pt
python -m superlearner experiment d --config configs/exp_d.yaml
python -m superlearner experiment worldmodel --run-dir runs/exp_b_fewdemo_sil_s0
python -m superlearner experiment selfplay --checkpoint runs/exp_b_fewdemo_sil_s0/ckpt/latest.pt --days 12

# Experiment A: all thirteen arms including LLM-written services (cloning only; `resume` adds RL iterations)
python -m superlearner train --config configs/exp_a_warm.yaml
python -m superlearner experiment a --checkpoint runs/exp_a_warm/ckpt/bc.pt --episodes 16

# the sealed final list is committed as docs/experiments/final_tasks.json; re-creating it gives the same seal
python -m superlearner lock-final --out /tmp/final_tasks.json --n 96
```

The memory ablation of the Part 5 comparison used the archived run `runs/archive/expb_v4_memory_confound/
exp_b_fewdemo_sil_s0` (same configuration as `exp_b_fewdemo_sil` with `use_memory: true, use_tools: true`).

## 3. Meta-Researcher

```bash
# Experiments E and F in one run: four rounds in which every proposal of a pool of six is screened (confined, CPU,
# one seed), then the best screened candidate goes through development (3 seeds) and confirmation (5 seeds)
python -m superlearner meta-search --config configs/meta_search.yaml --full-feedback --run-dir runs/meta/search2
python -m superlearner meta-search --config configs/meta_search.yaml --full-feedback --run-dir runs/meta/search2 --resume --rounds 2
python -m superlearner report --experiment runs/meta/search2
python -m superlearner experiment e --run-dir runs/meta/search2     # -> docs/experiments/exp_e_meta.json
```

Each candidate's directory (`runs/meta/search2/candidates/<id>/`) holds its full source tree, `proposal.json`,
`proposal.diff`, per-run job files, logs and checkpoints; the trusted results are in
`runs/meta/search2/trusted/<id>/`; the ledger is `runs/meta/search2/ledger.jsonl`.

## What is and is not deterministic

* Worlds are generated from (family, seed) and Crucible replays them exactly; task lists are fixed.
* `tiny` worlds use Crucible's deterministic cost model; other families use measured CPU time, so outcomes there
  depend slightly on host load.
* GPU cloning and updates are not bit-reproducible; compared runs branch from a saved warm-start checkpoint.
* The LLM is greedy per (prompt, variant) and cached, so identical prompts give identical text.
