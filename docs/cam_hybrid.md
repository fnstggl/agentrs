# CAM + frozen pretrained semantic foundation

Branch `feat/cam-hybrid`, built on `feat/cam-v2`. Experiments run from the frozen worktree
`/workspace/agentrs-cam-hybrid-exp` (tag `cam-hybrid-exp-v1`) through the supervisor program
`agentrs_cam_hybrid_queue`; results land in `runs/cam_hybrid/<job>`. These are additional arms: the original
CAM, CAM-v2/v2.1 and every earlier queue are unchanged.

## Architecture

```
natural-language goal + observation (rendered from the agent's own tokens only)
  → frozen pretrained LM (Qwen3-0.6B or Qwen3-4B; no LoRA, no gradients, weights fingerprinted)
      mask-mean pooled hidden states of the middle and last layers (forward hooks; cached by exact text)
  → trainable semantic bridge (LayerNorm → MLP), part of CAM's world model, trained by CAM's own objectives
  → CAM-v2 RSSM belief and world model → CAM strategy (options), value, grammar-masked action decoder
  → executed action → verified consequences → CAM updates (PPO actor and critic; world model from replay)
```

CAM chooses every action with its own decoder (mechanism 1). The LLM never selects actions in the hybrid.

## Natural-language layer (`src/agentrs/cam/text/render.py`)

- **Information-equivalent:** texts are rendered from the agent's observation and goal tokens, so they
  contain no hidden state. Tests show the rule-based parser recovers the exact tokens from training wording
  over thousands of real states.
- **Training wording:** several phrasings per fact, shuffled sentence order.
- **Held-out wording:** a different vocabulary ("vendor", "order volume", "cash on hand", "retained
  customers"), different number formats and different sentence structures. It is injective (no information
  lost) and unreadable by the parser. It is used only to test language shift.
- **Structured control:** the original tokens.

## Arms (`configs/cam_hybrid/`)

| Arm | Run | What it is |
|---|---|---|
| A | `hy_A_struct_s{1,2}` | Structured CAM learned from scratch (tokens). Same learner, rewards, budget and evaluation as B. |
| B | `hy_B_q06_s{1,2}`, `hy_B_q4b_s1` | CAM plus frozen Qwen3-0.6B (or Qwen3-4B) embeddings of natural-language observations and goals. |
| C | final phase of `hy_A_*` | The structured CAM deployed on natural language through the rule-based parser, or through frozen Qwen3-4B extraction into CAM's tokens (few-shot, training-wording examples only). |
| D | `hy_D_llm_*` | The frozen LLM acting directly. Same information, tasks, seeds and step limits; grammar-constrained likelihood choice of every action slot. |
| E | final phase of every run | The trained CAM is reset with the frozen LM kept. Three variants: everything, only the semantic bridge, or only the actor. Evaluated on validation and test in both wordings. |

Transfer runs: bizdev learned from scratch, and bizdev initialised from the trained A or B checkpoint, with
negotiation and product monitored for forgetting.

**Learner.** All arms use PPO inside CAM-v2: an RSSM world model trained from replay, the contract reward
with shaping, a training-only product curriculum, a risk-budget Lagrangian and light Go-Explore starts.
CAM-v2's off-policy learners had not learned negotiation at matched budgets when these runs were frozen.
PPO's ratios are exact because the actor's stop-gradient inputs are stored at collection time.

**Measurements**
- Milestones at 10k, 50k, 100k, 200k, 300k and 400k real interactions: validation in both wordings, plus a
  kept checkpoint.
- Sealed test in both wordings.
- World-model open-loop error.
- Frozen-LM fingerprint at start and end.
- Relative change of each CAM parameter group, and change of decision log-probabilities between the initial
  and trained CAM.
- Compute broken down into collection, learning, evaluation and semantic encoding.

## Not done (yet)

- **Mechanism 2** (CAM controlling an LLM language generator) is not built; the primary comparison isolates
  semantic understanding first.
- **TERMS-Bench and other independent negotiation benchmarks** are not integrated.
- **An unseen task family (Test F)** is not built.
