# Parameterized semantic action learning

The agent learns **what to do and what to write**: generated content is part of the action, the
world model reads it, the planner compares content variants, and the content-generating weights
are updated by gradient steps from verified outcomes. It is built into the existing action
representation, policy, world model, planner, experience store and training loop — not a separate
text generator.

## 1. Content is part of the action

`Action = (operation, generated_content, parameters)` is implemented as a typed `SUBMIT` action
with a strict grammar, e.g. `submit(send_email, "Hi Dana, can we meet Mon 9am?")`, and recorded as a
`SemanticAction` (`src/agentrs/content/semantic_action.py`):

| field | purpose |
|---|---|
| `operation`, `content`, `content_ref` | the operation and the COMPLETE content (SHA-256 reference for out-of-line storage) |
| `modality`, `format`, `parameters`, `target_env` | what kind of content, its format, structured parameters, where it is executed |
| `goal`, `context`, `generation_prompt` | what it was generated for and the exact conditioning text |
| `policy_checkpoint` | which policy version generated it |
| `op_candidates`, `op_index`, `op_logprob` | the operation choice and its log-probability |
| `token_ids`, `token_logprobs`, `sampling_temperature` | the exact sampled tokens and their behaviour log-probabilities |
| `source`, `execution_status`, `permission` | policy / planner / teacher / human; executed / denied / declined |

Content is never replaced by a summary. Arbitrary length: the action keeps the full text (up to
64k characters inline, more by reference); `ContentActionEncoder` splits long content into token
chunks and the world model pools over all chunks. Every executed content action from any source is
stored in the experience Parquet (schema v3) together with the environment's content outcome.

## 2. Content-generating policy (`content/policy.py`)

One network — the existing `LMPolicy`, shared with high-level actions — produces both factors:

    pi(a | s) = pi(op | s) * prod_i pi(c_i | s, op, c_<i)

The operation is chosen from the environment's operation set exactly like a primitive action; the
content is sampled token by token after `submit(<op>, "` until the closing `")` (the terminator is
part of the content distribution). The policy supports multiple drafts
(`CandidateContentGenerator`), greedy or sampled generation, and revision rounds through the
planner. Exact token ids are stored because re-tokenising decoded text can produce different ids
(observed in this codebase on the first try).

## 3. Content-conditioned world model (`content/world_model.py`)

`ContentConditionedWorldModel` extends the existing world model (same trunk):

* input = goal, recent history, observation and the complete action including content;
* the generative diff head predicts the environment's textual response (a reply, test results,
  a moderator verdict) — sampling gives several possible responses;
* event / reward / termination ensemble heads as before;
* new ensemble heads for content outcomes: P(format valid), P(objective met), P(partial ≥ 0.5),
  P(positive response), with ensemble spread as epistemic uncertainty.

It trains on real (content, context, outcome) transitions from the agent and from authorised
third parties, with its own optimiser; transitions without a content outcome contribute no
content-head loss, so the same model keeps learning MiniWeb dynamics.

## 4. Consequence-based planning (`content/planner.py`)

Goal + belief → operation → K content drafts → each complete action through the world model →
score = P(objective) − λ·uncertainty → optional revision round → permission check → execute the best
authorised draft, or **decline** (`finish()`) when the best predicted score is below a threshold,
uncertainty is too high, or the operation is not authorised. Predictions are stored next to the real
outcome and never count as success. Adjustable: candidates, temperature, revision rounds, target
score, uncertainty penalty, decline thresholds.

## 5. Updating content-generation weights (`content/training.py`)

`ContentPolicyRLTrainer` optimises a token-level clipped surrogate with group-relative advantages
from verified rewards:

    L = - sum_{u in {op, c_1..c_N}} min(rho_u A, clip(rho_u, 1±eps) A) + beta * sum_i KL_i(pi || pi_ref)

* `rho_u = exp(log pi_theta(u) - log mu(u))` from the stored ids and stored behaviour log-probs:
  equals the plain `-A Σ log P(c_i)` gradient on fresh on-policy data, corrected when the data
  came from an older checkpoint;
* masks restrict the loss to content tokens; variable lengths are handled per sequence;
* episode returns come only from the environment; delayed rewards (a second attempt after
  feedback) are credited to every content action of the episode;
* exact per-position KL over the full vocabulary against a frozen reference (the imitation
  policy); gradient clipping, NaN guards, optimiser and checkpoint state as everywhere else;
* **eligibility:** only content sampled by the policy at temperature 1 with stored behaviour
  log-probabilities enters the policy gradient. Teacher / human / dataset content and
  planner-selected drafts are excluded (they train imitation and the world model);
* one outcome per document is treated as a noisy sequence-level signal; no token is assumed to be
  "the cause".

Imitation (`train_content_bc`) teacher-forces demonstrated content and operations, weights
demonstrations by verified outcome, and can rehearse high-level MiniWeb demonstrations so content
learning does not erase old skills. Imagined training (`imagined_content_steps`) scores on-policy
drafts with the world model, drops uncertain ones, and down-weights the rest.

## 6. Environment and verification (`environments/content/env.py`)

| domain | operation | content | verified outcome | held out |
|---|---|---|---|---|
| structured config | `apply_config` | `key=value` pairs | each required key set correctly (partial credit) | new keys, larger numbers |
| email scheduling | `send_email` | message proposing a time | a simulated recipient accepts morning slots with p = 0.9, afternoons with p = 0.1 | new recipients |
| forum answer | `post_reply` | answer text | correct fact and no wrong fact; **engagement** (views) reported separately, inflated by "!!!", shouting and a posting-hour confounder | new places and cities |
| code | `submit_code` | expression for f(x) | safe evaluator (no `eval`) against visible + hidden tests | larger coefficients |
| document | `write_document` | short note | mentions required items, avoids a forbidden one, word limit | **entire domain** (never trained) |

Outcomes depend on the content itself: identical operations and identical lengths get different
outcomes (tested). The last goal template of each domain is reserved for test splits.

## 7. Safeguards

* Content operations must be on the permission allowlist; operations marked as publishing outside
  the sandbox require human approval (default: deny); per-episode submission rate limit.
* Reward = verified objective; engagement is logged but never optimised; `RewardHackingMonitor`
  flags engagement or proxy features rising without verified success (tested with spam content).
* Predicted rewards are stored separately from verified rewards; imagined drafts are labelled.
* Observational associations are labelled non-causal; `randomized_variant_test` requires explicit
  authorisation and is meant for sandbox environments only.
* Regressions of the shared network (high-level MiniWeb skill, general-text perplexity) are
  measured and reported by `regression_check` instead of passing silently.

## 8. Tests (`tests/content/test_content_learning.py`, 15 tests)

1. different content → different world-model predictions (trained model separates a morning from
   an afternoon proposal by > 0.3 in P(objective)); 2. the world model trains on exact
   action-content-outcome pairs and is evaluated on held-out content (AUROC > 0.7);
3./4./5. nonzero content-RL gradients, a real optimiser step changes parameters, checkpoint reload
preserves the updated policy (and stored behaviour log-probs match recomputation to 1e-4);
6. the planner scores several distinct drafts, executes the best, revises, and declines when
unauthorised or too uncertain; 7. a world model predicting certain success does not make episodes
succeed, imagined drafts are labelled, and the model learns from the real failure; 8. imitation
of a third party's content raises its probability on held-out tasks; 9. regressions of the shared
network are detected; 10. content RL from verified outcomes raises the log-odds of the accepted
proposal on held-out tasks (> 1 nat); plus the end-to-end loop goal → generate → predict →
execute → outcome → reward → update → save → reload → evaluate, permissions/rate limits,
engagement ≠ reward with hacking flagged, and causal-label tests.

## 9. Experiments

Configurations A–E (mixed domains, 2 seeds) and B/C/D on the email domain (2 seeds) were run with
Pythia-14M on CPU. In short: imitation gives the measurable gain (0% → 8–33% on held-out instances);
verified-reward RL gave a small consistent gain on email (+4 to +12 points) and none on the mixed
suite; planning over content drafts and imagined training gave none; nothing transferred to the
unseen document domain. Full tables and caveats: `docs/experiments.md`, section "Content learning".
