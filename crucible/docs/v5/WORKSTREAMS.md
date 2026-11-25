# Crucible 5.0 workstreams

The orchestrator owns architecture, contracts, the manifest, integration and the release decision. Each workstream owns
the paths listed and may touch other code only at the extension points named in INTERFACE_CONTRACTS.md. Work runs in
isolated git worktrees; the orchestrator merges verified work into `feat/crucible-5-openworld`. Compute is scheduled
through `crucible.scale.slots`; at most one GPU-heavy job per workstream at a time.

Waves: 1 = foundations that need nothing new (now); 2 = systems that need the substrate; 3 = worlds that need the
systems; the repair loop and the final QA audit follow. A wave starts as soon as its inputs are merged.

| WS | name | owned paths | wave | depends on | requirements owned |
|---|---|---|---|---|---|
| A | Core world architecture | `src/crucible/substrate/` | 1 | - | 2 |
| B | Autonomous population | `src/crucible/population/` | 2 | A, E (fitted models when available) | 19 |
| C | Social internet | `src/crucible/social/` | 2 | A, B, D | 26 |
| D | Semantic and multimodal intelligence | `src/crucible/semantic/` | 1 | A (recorded inference) | 16 |
| E | Empirical calibration | `src/crucible/calibration/, docs/v5/calibration/` | 1 | - | 11 |
| F | Economic and institutional expansion | `src/crucible/commerce/` | 1 | A | 10 |
| G | Innovation and software ecosystem | `src/crucible/innovation/` | 1 | A, F | 17 |
| H | Rapid scientific environments | `src/crucible/rapid/` | 1 (foundation), 2 (families) | L (observation shape) | 14 |
| I | Semantic digital environments (15 domains) | `src/crucible/domains/` | 3 | B, C, D, F, G, H | 8 |
| J | Integrated civilizations | `src/crucible/civ/` | 3 | all world systems | 17 |
| K | Multi-agent learning and self-play | `src/crucible/multiagent/` | 3 | B, J, L | 11 |
| L | Architecture-independent learner interface (replaces CAM integration, Clarification 1) | `src/crucible/experience/` | 1 | A | 22 |
| M | Performance engineering | `src/crucible/scale/, benchmarks` | 2-4 | B, C, D | 12 |
| N | Research evaluation | `src/crucible/research/ (v5 additions), src/crucible/rapid/curriculum` | 3 | H, L | 25 |
| O | Security and adversarial testing | `src/crucible/redteam/ (v5 probes)` | 2-4 | every world module | 11 |
| P | Independent quality assurance | `docs/v5/reviews/, tests/v5/test_gate_X.py` | continuous, final | everything | 20 |

## Requirements by workstream

* **A** (2): SP-2.1, SP-2.3
* **B** (19): B.1, B.2, B.3, B.4, B.5, B.6, G.1, G.2, G.3, G.4, G.5, G.6, G.7, G.n1, SP-3.1, SP-3.2, SP-3.3, SP-3.4, SP-4.1
* **C** (26): C.1, C.2, C.3, C.4, C.5, C.6, D.1, D.2, D.3, D.4, D.5, I.1, I.2, I.3, I.4, I.5, I.6, I.7, C.n1, I.n1, SP-4, SP-4.2, SP-4.5, SP-4.6, SP-4.7, SP-4.8
* **D** (16): E.1, E.2, E.3, E.4, E.5, H.1, H.2, H.3, H.4, H.5, H.6, H.7, H.8, E.n1, H.n1, SP-3.2b
* **E** (11): F.1, F.2, F.3, F.4, F.5, F.6, F.7, F.8, F.n1, SP-5.4, SP-5.5
* **F** (10): K.1, K.2, K.3, K.4, K.5, K.6, K.7, K.n1, SP-6.1, SP-6.6
* **G** (17): J.1, J.2, J.3, J.4, J.5, J.6, J.7, L.1, L.2, L.3, L.4, L.5, J.n1, L.n1, SP-6.3, SP-6.4, SP-6.5
* **H** (14): M.1, M.2, M.3, M.4, M.5, M.6, M.7, M.8, M.n1, SP-7.1, SP-7.3, SP-7.4, SP-8.11, SP-8.12
* **I** (8): N.1, N.2, N.3, N.4, N.5, N.6, N.7, N.n1
* **J** (17): O.1, O.2, O.3, O.4, O.5, O.6, O.7, P.1, P.2, P.3, P.4, P.5, P.6, O.n1, P.n1, SP-10.1, SP-10.3
* **K** (11): Q.1, Q.2, Q.3, Q.4, Q.5, Q.6, Q.7, Q.8, Q.n1, SP-12.1, SP-12.3
* **L** (22): R.1, R.2, R.3, R.4, R.5, R.6, R.7, S.1, S.2, S.3, S.4, S.5, S.6, S.7, S.8, R.n1, SP-11.1, SP-11.2, SP-11.3, SP-11.4, SP-11.5, SP-11.6
* **M** (12): V.1, V.2, V.3, V.4, V.5, V.6, V.7, V.8, V.9, V.n1, SP-13.7, SP-13.8
* **N** (25): T.1, T.2, T.3, T.4, T.5, T.6, T.7, T.8, U.1, U.2, U.3, U.4, U.5, U.6, U.7, U.8, I.n2, K.n2, S.n1, SP-14.1, SP-14.2, SP-14.4, SP-15.3, SP-15.5, SP-15.6
* **O** (11): W.1, W.2, W.3, W.4, W.5, W.6, W.7, W.8, W.9, W.n1, SP-16
* **P** (20): A.1, A.2, A.3, A.4, A.5, X.1, X.2, X.3, X.4, X.5, X.6, X.7, X.8, X.9, X.n1, SP-1.3, SP-1.5, SP-18, SP-20, SP-21

Gate tests: the owner of a gate writes `tests/v5/test_gate_<G>.py`; a requirement owned by another workstream
inside that gate is tested in the same file by that workstream (coordinate through the orchestrator).
