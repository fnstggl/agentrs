"""SP-3.2b: benchmark of semantic-model sizes on the eight capabilities of MASTER_SPEC Part 3.2 class 4.

* `tasks` builds the evaluation sets from world situations (kernel worlds of the 4.0 open economy: real documents,
  mail, invoices, contracts, forum posts, tool catalogues, failures) with train / dev / test splits; test items come
  from templates never used for training or prompt selection.
* `models` evaluates a candidate: a language model behind `crucible.semantic.service.SemanticService` (option
  scoring and bounded generation), or a task-specific encoder (frozen pretrained MiniLM sentence encoder with a head
  trained on the train split).
* `run` evaluates every model on every capability and writes a report with accuracies, Wilson intervals, p-values
  against chance, memory and latency, and the smallest effective model per capability.

    PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --gpu-mb 9000 --label "ws-D2: SP-3.2b" -- \
        python -m crucible.semantic.capbench --out $CRUCIBLE_DATA/semantic/sp32b.json
"""
CAPABILITIES = ("content_understanding", "situation_interpretation", "tool_choice", "communication",
                "incentive_adaptation", "negotiation", "programming", "strategic_decisions")
