# Research questions Crucible is built to study

None of these is answered by the existence of the platform. For each: the experiment the infrastructure supports,
and what result would count against the hypothesis.

| # | question | experiment supported | falsifying evidence |
|---|---|---|---|
| 1 | Does environmental complexity produce more general learning? | train the same learner on families with mechanisms dropped (`drop_modules`) vs full; evaluate on test families | no transfer gain from richer training worlds at matched experience |
| 2 | Can agents discover strategies not demonstrated by humans? | imitation from scripted/LLM trajectories vs RL; cluster strategy fingerprints (emergence.py) | learned agents only reproduce demonstrated tool-usage clusters |
| 3 | Can agents invent tools that improve their future learning? | measure learning speed with vs without self-built services available (counterfactual forks) | removing self-built tools does not slow later learning |
| 4 | Does experience in one world accelerate learning in another? | sequential training train-family → test-family, learning-curve AUC vs scratch | AUC on the target equals scratch |
| 5 | Continual acquisition without catastrophic forgetting? | goal suites revisited after training on other families | success on earlier families drops to baseline |
| 6 | Does cooperation become more sophisticated under simple incentives? | league/self-play in shared worlds; contracts formed, joint ventures, their counterfactual value | no contracts or only trivial ones; removing them changes nothing |
| 7 | Do institutions emerge from repeated interactions? | long canonical worlds; contract/organisation statistics over time | no persistent organisations beyond the scripted ones |
| 8 | Do larger, more diverse populations create new learning opportunities? | vary NPC/agent population size and heterogeneity; measure strategy diversity and learner gains | no change with population |
| 9 | Does self-play beat fixed populations? | league vs fixed NPC opponents; evaluate against a held-out population | no advantage on held-out opponents |
| 10 | Does architecture discovery find faster learners in new environments? | outer learner submits candidates; paired evaluation on val families, final on locked test | no candidate beats the baseline with independent seeds |
| 11 | Do discovered architectures generalise across generator mechanisms? | evaluate on `engineering_lab_heldout`, `novel_needs` | gains vanish on held-out generators |
| 12 | Can technical methods transfer between domains? | lab-learned routing improvements vs economy route needs | no transfer |
| 13 | Does a pretrained LLM accelerate or constrain discovery? | frozen LLM vs CAM+LLM vs pure CAM at matched budgets | LLM-free learners match or beat LLM-based ones on held-out goals |
| 14 | Can an agent surpass its frozen backbone through experience? | CAM+LLM vs the same frozen LLM alone, same tools/tokens | CAM never exceeds the LLM-only baseline (the CAM 2.0 result on MiniWoB) |
| 15 | How do richness, model size, compute and experience interact? | factorial runs over families × models × budgets | — (descriptive) |
| 16 | Does persistence beat independent episodes? | `reset_mode="continue"` vs `"fork"` in the Gym adapter | no difference |
| 17 | Durable knowledge across changed tools/institutions/markets? | shocks (new needs, compute shortage, entrants) | performance collapses after every shock and recovers no faster than scratch |
| 18 | What reward hacking appears in rich worlds? | integrity audit over long training runs | — (descriptive; each new exploit becomes a test) |
| 19 | Which mechanisms create genuine causal reasoning requirements? | ablate hidden information / delays / noise; measure the gap between heuristics and learners | heuristics match learners everywhere |
| 20 | What would falsify "Crucible enables increasingly general intelligence"? | all of the above with pre-registered metrics, budgets and seeds | learners plateau at scripted-heuristic level on held-out families despite more experience and capacity |

The first measurements are in EXPERIMENTS.md; they establish solvability, multiple strategies, exploit
resistance, reproducibility and throughput — not answers to the questions above.
