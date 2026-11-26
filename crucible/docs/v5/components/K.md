# Component K: civilisation-wide self-play (`crucible.multiagent`)

Owner: workstream K. Spec: MASTER_SPEC Part 12, Gate Q (Q.1-Q.8, Q.n1), SP-12.0, SP-12.1, SP-12.1b, SP-12.2, SP-12.3,
SP-12.3b. Decisions: D-60, D-61, D-62. Audit: docs/v5/audit/K_selfplay.md.

## What it is

Several independently updated neural policy groups, semantic generalists and a frozen historical checkpoint act in
one Tier 3 civilisation (`crucible.civ`, observation-only) next to the civilisation's own population, all through
the universal interaction contract and multi-agent driver of `crucible.experience` (`run_learners`). Nothing is
shared between participants but the world; cooperation and competition exist only as world actions; every outcome
is the evaluator's.

```
knowledge.py     a participant's private memory (ids by field name, own things, words, its own tool record)
grammar.py       argument candidates from tool schemas + own knowledge; scripted syntax for files/terms/input
policy.py        PolicyGroup: hierarchical host -> tool -> slot policy (MLP or GRU), REINFORCE / A2C / PPO,
                 evidence-scaled updates, no-signal controls, behaviour cloning, checkpoints
objectives.py    learner-side drives (profit, reach, adoption) + novelty / rejection, from own observations only
agents.py        LearnerAgent (one identity of a group), SemanticAgent (class 4)
registry.py      versions, hashes, identities, provenance, experience, actions, outcomes, updates, composition
harness.py       Match: admission with controllers, run_learners in chunks, composition, finish
evaluate.py      evaluator outcomes (cash, independent revenue via commerce.integrity, same-controller counted)
analysis.py      parties; cooperative / competitive interactions; consequences; Part 12 capabilities; TV / JS
integrity.py     independence audit, canary taps, shared-reward audit, controlled identities, collusion battery
league.py        MatchSpec, play / replay (reproducible cross-play), held-out evaluation on sealed seeds, Elo, PFSP
demonstrator.py  SCRIPTED workflow demonstrator (behaviour-cloning data and feasibility only)
campaign.py      demonstrations -> pre-training -> held-out family -> civ run -> learning evidence -> strategy switch
```

## How a learned participant acts

Each decision: `Knowledge` ingests the observation; the group scores every tool whose required slots it can fill
(plus `wait`): first a host (module), then a tool of that host, then one candidate per argument slot. Candidates
are the slot's enum values, numeric bins within bounds (money bins capped by what it holds), fresh names that fit
the schema's pattern (for creating tools), its own accounts and organisations for acting-for slots, ids seen under
the same field name, composed texts, or a syntax grammar (code commit files, contract clause lists, service
inputs). It cannot name what it has never seen. The environment's reward is 0.0 and is never read; the training
signal is the identity's own `Objective` over its own previous and current observation and action result.

Learning: each group's own buffer, optimiser (persisted), generators; an update every 4 decisions per identity;
advantages divided by a running RMS of the signal (no per-batch renormalisation); a batch without signal is not
trained on; an update counts only when the weights hash changes. Every learning group feeds a no-signal twin the
same batches with signal 0 (it must not change). Pre-training: behaviour cloning on decisions recorded from the
scripted demonstrator in CPU training worlds (open_digital + v5 commerce/software modules), never the evaluated
civilisation; groups share the demonstration corpus but have their own seeds and batch orders.

## Parties and integrity

Identities are admitted with `admin.join(controller=...)`: a group's identities (and its historical copies) are one
party. All interactions are between parties. `commerce.integrity` excludes payments between identities of one
controller from independent revenue; `controlled_identity_accounting` reports what is still counted (must be 0) next
to a naive count. The collusion battery plants six schemes with clean controls in the civilisation after the run.
The canary audit writes private canaries into each identity's records and scans everything every participant
receives; an owner's own later disclosure is reported separately. The shared-reward audit recomputes every learner's
signal across a private change to another party (positive control: a team objective).

## 4.0 numeric self-play (`crucible.selfplay`)

Kept (28 4.0 tests pass) and repaired for Clarification 2 (D-62): no other seat's money or status in the step info
outside legacy mode (`last_info` for the operator); the climber reads its observed revenue minus cost; PG counts an
update only when its policy changed; scripted operators are not learners. Numeric facility self-play does not pass
Gate Q (`harness.is_civilisation`).

## Running

Unit tests (CPU, ~1 min): `tests/test_v5_selfplay_obsonly.py`, `tests/test_v5_selfplay_multiagent.py`.
Gate tests (civilisation, GPU text path): `tests/v5/test_gate_Q.py` and `test_gate_SP_12_0.py` (full campaign: three
demonstration worlds of 48 h, 12 epochs of cloning, a held-out GRU/PPO family, 96 simulated hours with 9 admitted
identities, held-out evaluation on two sealed seeds, strategy switch) and `test_gate_SP_12_1/1b/2/3/3b.py` (24 h):

    PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 4000 --ram-mb 6000 --timeout 7200 \
        --label "K: gate Q" -- python -m pytest -q tests/v5/test_gate_Q.py

## Measured on the Tier 3 civilisation (unofficial runs, wave 3b)

| gate file | commit | result |
|---|---|---|
| test_gate_Q.py (full, 72 h) | 9ea4901 | 9/9 passed (1009 s): 4 learned groups acting, 4 independent competing groups, 4 cooperative + 6 competitive interactions between learning parties, 125 cross-party consequence events, 172 update versions, 4 held-out evaluations, 144 RL updates (no-signal twins unchanged), 6/6 planted schemes detected with 0 clean controls flagged |
| test_gate_SP_12_1.py | 2981d88 | passed: 7 member kinds acting in one civilisation |
| test_gate_SP_12_1b.py | 9ea4901 | passed: 3 algorithms, 2 unrelated drives, 4 independently updated groups |
| test_gate_SP_12_2.py | 2981d88 | passed: 9 record kinds, cross-play replay 0 mismatches, Elo league |
| test_gate_SP_12_3.py | 9ea4901 | passed: 0 controlled-identity cents counted (10,200 moved) |
| test_gate_SP_12_3b.py | 9ea4901 | crashed (tap called `update` of a semantic participant) - fixed in 2981d88; at small size on the civilisation every assertion holds; full size not yet re-run |
| test_gate_SP_12_0.py (full) | 2981d88 / 5dfa700 | failed: 7, then 8 of the required 10 capability kinds (DEF-K-7) |

What the runs show about learning (reported, not required to be large): behaviour on recorded situations changes
(TV 0.41 vs 0 for the no-signal twins) but outcomes barely improve; the trained alpha scored worse than its initial
copy against the held-out family (-50,000 cents: it spent more); the external drives were nonzero on 17 of 1008
learner decisions, so reinforcement came mostly from the intrinsic novelty/rejection terms; most rejected actions
were attempts on other parties' repositories, which the own-things memory (5dfa700) addresses. Behaviour cloning is
the bottleneck for capability coverage: held-out agreement with the demonstrator's tool choice rose from 0.03 to
0.22 (argmax) with the BC learning rate raised to 1e-2 (e179c64, not yet re-measured on the civilisation).

## Known limits

See docs/v5/workstreams/K.json `limitations`: small reference learners whose reinforcement learning within a
gate run is small (measured, not assumed); scripted syntax for code, clauses and inputs; semantic participants use
one scoring pass per tool choice; tacit collusion without any world channel is not flagged as collusion.
