# Related systems: what was checked, what was reused, what was rejected

Checked 2026-11-20 against the projects' repositories, PyPI and docs (not published claims). "Usable here" means
inside this unprivileged Docker container (no Docker-in-Docker, no KVM, no user namespaces).

| Project | License | Last activity | Needs Docker/KVM | Usable here | Decision |
|---|---|---|---|---|---|
| OpenEnv (openenv/OpenEnv, PyPI `openenv` 0.14.0) | BSD-3 | 2026-11-20 | no (uv provider) | yes | **Protocol reused**: Crucible's HTTP server speaks OpenEnv's `/reset`, `/step`, `/state`, `/health`, `/metadata`, `/schema`, `/ws` and `/mcp` shapes, implemented without depending on the package. OpenEnv does not provide a shared living world. |
| MCP spec (2026-07-28) | — | — | — | yes | `tools/list`, `tools/call`, `notifications/tools/list_changed` used for dynamic tool discovery. |
| Snowflake Agent World Model | no license file detected | 2026-05 | no | technically | Not imported (license unclear). Pattern (SQLite-backed tool envs with SQL verifiers) informed verifier design. |
| RUC Agent-World | MIT | 2026-10 | no | yes | Not imported: its tools are `exec`-ed code over per-env JSON/DB fixtures — disconnected environments, the opposite of a shared causal world. |
| VeriEnv | none detected | 2026-04 | optional | partly | Pattern only (generated site + SDK-based judge) informed the module admission checks. |
| TheAgentCompany | MIT | 2025-11 | **yes** | no | Blocked (each task is a Docker image). Checkpoint grading idea reused in goal contracts. |
| AppWorld | Apache-2.0 (encrypted parts) | 2026-11 | optional | yes (used by CAM 2.0) | Not used as Crucible substrate (fixed apps, per-task DB resets). |
| BrowserGym | Apache-2.0 | 2026-11 | no | yes | Observation conventions (AXTree/DOM) followed for browser mode; Crucible serves its own sites. |
| WebArena | Apache-2.0 | 2025-11 | **yes** | no | Blocked. |
| OSWorld (desktop-env) | Apache-2.0 | 2026-11 | **yes** (VM) | no | Blocked. |
| WebWorld (Qwen) | code: none detected | 2026-02 | no (GPU) | partly | Rejected as an authority: an LLM that predicts next pages cannot enforce conservation or ownership. |
| DeepMind Concordia | Apache-2.0 | 2026-11 | no | yes | Design reference (game master resolves actions). Crucible's "game master" is deterministic code, not an LLM. |
| OASIS (camel-ai) | Apache-2.0 | 2026-11 | no | yes | Design reference for large social populations; Crucible's social layer is mechanistic and shares state with the economy. |
| Emergence World | research-only / non-commercial | 2026-10 | — | no engine source | Docs only; nothing reused. |
| Melting Pot | Apache-2.0 | 2026-10 | no | yes (already used by the AGENTRS superlearner) | Substrate/scenario split with held-out background populations reused as an evaluation principle. |
| Neural MMO (NeuralMMO/environment) | MIT | 2024-08 | no | yes | Reference for large-population PettingZoo envs; not imported. |
| AgentSociety | Apache-2.0 (except `commercial/`) | 2026-11 | unverified | partly | Reference for pluggable environment modules. |
| Google Chrome "Mirror Dimension" | — | announced 2026-11-19 | — | **no public code** | Only press descriptions are available. Synthetic `*.crucible` hostnames are used instead of impersonating real domains. |

## Architectural decisions and rejected alternatives

1. **Event-sourced in-memory state machine + durable event store** (chosen) vs. state living in SQL tables with one
   SQL transaction per action. SQL-resident state gives database constraints for free but costs a round trip per
   transition, makes forks expensive (database clones) and makes exact replay hard. The kernel keeps the
   authoritative state in memory, persists the hash-chained event log and snapshots (SQLite or PostgreSQL), and
   enforces invariants in code with dedicated checks. Replay and regeneration are exact and tested.
2. **Deterministic mechanisms decide outcomes; language models only produce text arguments.** No LLM judge
   decides success, quality or transactions anywhere in the kernel.
3. **One shared state for all services** vs. per-app fixtures (AppWorld, Agent-World, VeriEnv). Shared identity,
   ledger, permissions and event log are what make the services causally connected.
4. **Synthetic hostnames (`<name>.crucible`)** vs. mirroring real domains. No TLS impersonation; the browser runs
   with host-resolver rules that send only `*.crucible` to the local server and block everything else.
5. **Sandbox = unprivileged uid + Landlock (filesystem, TCP) + seccomp (no inet sockets) + rlimits**, because the
   container has no user namespaces (bubblewrap/nsjail unavailable) and no nested containers. Agent programs talk
   to the synthetic internet only through a brokered RPC channel to the kernel, which authorises and meters calls.
6. **Vectorised background populations** (numpy arrays inside the state) vs. one object or LLM per simulated
   person. Consumers are cheap, persist across snapshots and still decide from their own observations.
