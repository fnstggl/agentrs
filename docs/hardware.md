# Hardware profiles and resource requirements

Numbers marked **measured** were measured in the development sandbox (4 CPU cores, 15 GB RAM, no
GPU, Python 3.13, torch 2.15 CPU). Numbers marked **estimate** are calculations, not runs.

## Profiles

| profile | config | policy / world model / critic | where it runs |
|---|---|---|---|
| smoke | `configs/models/smoke_scratch.yaml` | tiny random GPT-NeoX (48–64 hidden, 2 layers) + local BPE tokenizer | any CPU, offline; tests |
| development | `configs/models/dev_pythia14m.yaml` | Pythia-14M (pretrained, full fine-tuning) ×3 | CPU; all executed experiments |
| development (larger) | `configs/models/dev_pythia31m.yaml` | Pythia-31M ×3 | CPU (≈3× slower) |
| research | `configs/models/research_qwen3.yaml` | Qwen3-4B + LoRA / Qwen3-0.6B + LoRA ×2, bf16, full-vocabulary policy | GPU — **not executed** |
| scaled | (not implemented) | distributed actors + learner | see below |

## Measured CPU throughput (4 threads)

| model | params | inference tokens/s | forward+backward, batch 8×256 |
|---|---|---|---|
| Pythia-14M | 14M | ≈49,000 (logits kept for 16 positions) | 0.17 s |
| Pythia-31M | 31M | ≈16,400 | 0.45 s |
| Pythia-70M | 70M | ≈7,500 | 1.8 s |
| SmolLM2-135M-Instruct | 135M | ≈920 (full logits) | 13.3 s (full logits) |
| Qwen3-0.6B | 596M | ≈260 | full fp32 fine-tuning ran out of memory (15 GB) |

Training-step costs (Pythia-14M, batch 16, measured): BC ≈0.5 s/step with 4 threads
(≈1.1–1.7 s/step with 1 thread while 4–9 experiment processes share the CPU); world-model step
≈0.6 s/step on 4 threads (≈3 s with 6 concurrent processes). Candidate scoring with a shared
prefix cache is ≈4.5× faster than scoring full sequences; restricting token normalisation to the
candidate vocabulary removed the 50k-way softmax that otherwise dominated (≈2.5× end-to-end BC
speedup).

Memory: one experiment process with three Pythia-14M models uses ≈0.6–1.5 GB RAM (measured).

## Storage (measured)

* Experience: ≈405 bytes per transition in compressed Parquet (observations, candidates,
  prompts and diffs stored as JSON records).
* Checkpoints: Pythia-14M full fine-tune ≈54 MB weights; ≈162 MB with AdamW state.
  A v2 experiment run writes 3–5 checkpoints (≈0.3 GB); the per-seed stage cache ≈0.4 GB.
* Screenshots (if recorded) go to the content-addressed blob store, not into Parquet.

## Research profile (estimate, not executed)

Qwen3-4B policy, bf16 weights ≈8 GB. LoRA r=16 on all linear layers ≈33M trainable parameters
(36 layers × 7 projections); fp32 AdamW state for adapters ≈0.5 GB. With gradient checkpointing,
a policy batch of 4 prompts × 1,536 tokens fits comfortably; the prefix-cache replication used for
candidate scoring costs ≈0.23 GB of KV cache per prompt per candidate copy for this model
(36 layers × 2 × 8 KV heads × 128 × 1,536 tokens × 2 bytes), so 4 prompts × 20 candidates ≈18 GB.
Budget: one 80 GB GPU for the policy learner, plus ≈4–8 GB each for the Qwen3-0.6B world model
and critic (same or second GPU). For larger candidate sets, score candidates in chunks or switch
`prefix_cache: false` (full sequences, more compute, less memory).

## Scaled profile (design, not implemented)

The actor/learner/evaluator separation, atomic Parquet shards, SQLite registry and fixed
checkpoint versions per actor are the hooks for scaling: many actor processes/machines writing
shards to shared storage, one learner (or data-parallel learners) consuming them, evaluators
gating promotion. A distributed implementation was deliberately not built before the
single-machine loop was validated.
