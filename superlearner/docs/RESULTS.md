# Results

Everything here was measured on 2026-11-21 on one RTX A6000 (46 GB), 48 logical CPU cores, 188 GB RAM, with
Qwen3-14B as the frozen language model. Raw outputs are in `docs/experiments/*.json`; the chronological account,
including every superseded attempt, is `docs/experiments/NOTES.md`. Success always means the task's predicate on
authoritative world state, checked by the evaluator. Intervals are 95% (Wilson for one condition, bootstrap over
seeds for differences).

**Short version.**

SUMMARY_PLACEHOLDER

## 1. What the frozen language model can and cannot do here

| Probe | Result |
|---|---|
| Qwen3-4B, "write the files of a service for need X" with public documents only, 8 needs | wrote `service.py` and no manifest for all 8 |
| Qwen3-14B, same probe (code executed on 6 hidden inputs) | working code for 5/8: sort.records 1.00, route.shortest 1.00, match.stable 1.00, schedule.jobs 0.95, forecast.series 0.50 (reference 0.68); route.tsp 0, pack.knapsack timed out, text.extract invalid JSON |
| Speed | about 18 tokens/s; 13–37 s per file set |
| One-shot "files" object (JSON inside JSON) | manifests with unbalanced braces in most attempts |
| Entry-by-entry writing, no reasoning | escaping fixed; manifest `params` written as a property map instead of a JSON schema in 2/2 attempts; repair with the error message shown: 1/2 valid |
| Entry-by-entry, reasoning mode for the manifest | 2/2 first attempts valid (43–52 s each) |
| Reasoning mode for a code file | used its 2,700-token budget and returned nothing (153 s); reasoning is therefore limited to structured entries, with a fallback |

Two defects of the interface the earlier Crucible agents used, found while building the grounding layer:

* Crucible's tool renderer shows only the first line of each description. The deploy manifest format is on the
  following lines of `cloud.deploy`. The Qwen agents of Crucible 3.0's final challenge were never shown it.
* Tool retrieval by mean-pooled hidden states is poor. On a probe, the tool that documents the manifest ranked 14th
  for "commit files for a deployable service" and `market.list` ranked 98th for "have an active market listing";
  with prompt-based last-token embeddings from layer −4 they rank 2nd and 1st.

## 2. Demonstrations and the warm start

Scripted teachers act through the controller's own heads and options; only evaluator-verified episodes are kept.

| Measurement | Result |
|---|---|
| Teacher on the eight arms that need no LLM content | 191/191 episodes verified (after fixing a teacher bug: it listed services under a random need, 146/191 before) |
| Teacher on LLM-dependent arms (52 episodes over 13 arms, small n) | apicall 5/6, code 1/2, deploy from scratch 2/7, service from scratch 0/3, revenue 0/2 |
| Cloning, 16 demonstrations per arm, greedy held-out success after 30 / 60 / 100 / 150 epochs | 0.87 / 0.97 / 0.88 / 1.00 (64 tasks) |
| Cloning, 2 demonstrations per arm, critic sharing the controller's state | 0.34 (80 tasks) |
| Same demonstrations and seed, critic detached | 0.75 |
| Same settings again (GPU nondeterminism, 16 demonstrations) | 0.78 and 0.51 |
| One checkpoint, same 80 tasks, five evaluations under different host load | 0.613, 0.650, 0.662, 0.675, 0.688 |

Consequences for method: the critic is detached from the controller's state; compared runs branch from one saved
warm-start checkpoint per seed; differences of a few points between single evaluations are noise.

## 3. A confound that looked like "RL destroys the policy"

Four successive versions of Experiment B showed held-out success falling in the first iterations of RL
(for example 0.775 → 0.500). Two real problems were found and fixed on the way (the first policy update after a
warm start moved every weight by a full learning-rate step: KL 0.13–0.17 after two steps; and prerequisite gating
left warm-started skills unrehearsed). The drop remained. In one run every policy step of iterations 2–4 had been
rolled back, so the weights were the warm-start weights, and it still scored 0.500. The direct test:

| Same checkpoint, same 80 tasks | Greedy success |
|---|---|
| with the episodic memory and tool library accumulated during training | 0.5125 |
| with memory and tools reset | 0.7625 |

The controller had been cloned with an empty memory. As soon as training episodes filled the memory, remembered
episodes and "what worked before" options appeared in its context and shifted its choices. This is a property of
adding non-parametric memory to a policy that was not trained with it, not of the learning rule. Experiment B
below therefore runs with episodic memory and tool-making off, and memory is examined separately (section 6).

EXPERIMENT_SECTIONS_PLACEHOLDER
