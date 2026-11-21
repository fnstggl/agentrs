# Synthetic open internet: what existing projects implement, and what Crucible should take from them

Checked 2026-11-21. Sources are the projects' own READMEs, selected source files, arXiv abstracts or HTML, Hugging
Face cards and the GitHub REST API (license field, `pushed_at`). The source is linked inline for every claim.
"Unknown" means the primary sources do not say. No figures were estimated. This document extends the shorter
decision table in [RELATED_WORK.md](RELATED_WORK.md) and does not replace it.

"Usable here" means usable in Crucible's environment: Python, a single machine, an unprivileged Docker container,
no Docker-in-Docker, no KVM, no user namespaces (see [SECURITY.md](SECURITY.md) and the sandbox backend layer).

---

## 1. Summary table

| Project | Implemented (vs. proposed) | License | Interfaces | State: executable or model-generated | One causal world shared by services | Persistent multi-agent | Generates envs/tasks | Validation of generated content |
|---|---|---|---|---|---|---|---|---|
| **Google Chrome "Mirror Dimension"** | The post describes it as built. No code, data or paper is published. | none published | headless Chrome in gVisor; DOM, ARIA, images, WebMCP tool responses | Agents write the sites (HTML/CSS/JS). The serving backend is not described. | unknown | no ("clean, identical baseline" each run) | yes: research, builder and critic agents work from human-defined user journeys | critic agents drive the rendered site through Chrome DevTools MCP and check ARIA semantics, form-validation edge cases, console errors and responsive layout |
| **Microsoft WildAWM** | Code, 341 env-task pairs and two checkpoints released 2026-10-19 | **no license file** (GitHub: none) | GUI (WebHarbor sites through Playwright MCP), MCP services, CLI and files | executable: files, MCP app databases and Flask-mirror state, all authored by LLMs | one persona's computer W(c,d) combines files, MCP DBs and web state. Runtime coupling between apps is not described. | no (one solver per task, Docker per instance) | yes: day-by-day worlds plus cross-interface tasks | clean-fails/golden-passes matrix; heterogeneous model×harness rollouts; difficulty bands; step gate; required-interface gate; three-level judge agreement; counterfactual "pressure probe"; diversity gate |
| **Snowflake Agent World Model (AWM)** | Pipeline, 1,000 environments and RL models released | code: **no license file**; dataset AgentWorldModel-1K: CC-BY-4.0 | MCP (`list_tools`/`call_tool`), CLI, OpenEnv wrapper | executable: generated FastAPI + SQLAlchemy over SQLite | no (one scenario = one server + one DB) | no | yes: scenario → tasks → schema → sample data → API spec → code → verifier | server must start and list tools (`check_all`); regeneration with LLM-summarised errors; generated verifier code smoke-executed; embedding dedup of scenarios |
| **RUC Agent-World** | 563 of 1,978 environments, 1,432 Q/A/rubrics and the task-synthesis script. No RL trainer, no weights. | MIT | Python tool functions (`exec`), CLI, OpenAI-compatible model server. No MCP server. | executable tool code over JSON "database" files | no (one directory per env) | no | yes: tasks from random walks over a tool-dependency graph | answers come from real execution; the chain is replayed on a clean copy and only stable observations are kept; unanswerable chains are dropped; structural `--check` |
| **TheAgentCompany** | 175 tasks, self-hosted servers, LLM NPCs | MIT (bundled GitLab/Plane/ownCloud/RocketChat keep their own licenses) | browser, terminal/code, chat | executable real services; NPCs are LLMs (Sotopia) | partly: one fictional company spread over 4 services with separate DBs | one agent plus LLM NPCs; per-task reset | no (hand-written tasks) | checkpoint graders (deterministic and LLM); evaluators encrypted |
| **BrowserGym** (ServiceNow) | Framework plus benchmark adapters | Apache-2.0 | Playwright Chromium: AXTree, DOM snapshot, screenshot, set-of-marks, `bid`-addressed actions, chat | whatever the task serves | n/a | one agent (with user chat) | no (framework) | per-task `validate()`; `cheat()` oracle |
| **WebArena** | 812 tasks, self-hosted sites | Apache-2.0 | browser (AXTree/HTML/screenshot) | executable real apps (shop, CMS, forum, GitLab, map, wiki) | no (separate sites) | no | no | functional checks: string match, URL match, `program_html` locators, LLM fuzzy match |
| **OSWorld** | 369 tasks (+ OSWorld-Verified) | Apache-2.0 | screenshot / a11y tree / set-of-marks; `pyautogui` actions | executable real OS in a VM | yes, inside one VM (a real OS) | no | no | per-task execution-based evaluator functions with result/expected getters |
| **Qwen WebWorld** | 8B/14B/32B world models, data and demo | README: models and data Apache-2.0; repo has **no LICENSE file** | text page states (A11y tree, HTML, XML, Markdown, natural language) | **model-generated** (an LLM predicts the next page) | no | no | generates simulated trajectories | intrinsic WebWorld-Bench factuality and "Web Turing" scores |
| **DeepMind Concordia** | Library (`gdm-concordia`) | Apache-2.0 | Python API, GUI/CLI editor | an LLM game master narrates outcomes; optional grounded Python components (inventory, payoff matrix) | the GM holds the world | yes (sequential, simultaneous and asynchronous engines) | scenario prefabs, not task generation | none for tasks (questionnaires and metrics) |
| **CAMEL OASIS** | Library (`camel-oasis`), "up to one million agents" | Apache-2.0 | Python API, 23 actions | executable SQLite platform; LLM agents choose actions | one platform (X-like or Reddit-like) | yes | agent profiles | n/a (replicates social phenomena) |
| AppWorld (extra) | 9 apps, 457 APIs, 750 tasks | Apache-2.0, plus an encrypted protected portion under extra terms | Python/HTTP APIs, MCP server | executable (SQLite base DBs) | about 100 simulated people shared across the 9 apps | no | no | state-based unit tests, including collateral damage |
| τ²-bench (extra) | Domains, LLM user simulator, text and voice | MIT | tool calls; text and full-duplex voice | executable domain DBs plus an LLM user | per domain | agent plus LLM user | no | final DB state vs. goal state; pass^k |
| WebHarbor (extra; WildAWM's GUI layer) | 153 Flask mirrors of real sites, 3,518 tasks | **no license file** | browser | executable Flask apps with seeded DBs | no | no | coding agents with human audit extend the sites | per-task deterministic verifiers (navigation, answer facts, read-only DB) |
| OpenApps (extra, Meta) | Python apps with configurable variants | **CC-BY-NC-4.0** | browser GUI | executable Python | several apps | no | appearance/content variants via Hydra config | state-based rewards |

**Key finding.** None of these projects has the combination Crucible already has: one persistent, event-sourced
world that services share causally, is inhabited by many agents and can be forked or replayed. The open-internet
expansion can still borrow five things:

1. Validation gates for generated tasks (WildAWM, Agent-World, AWM).
2. Critics for generated sites (Mirror Dimension, AWM).
3. Hermetic networking tricks (Mirror Dimension, TheAgentCompany).
4. Observation and action conventions for browser agents (BrowserGym).
5. Game-master grounding for LLM actors (Concordia, OASIS, τ²-bench, TheAgentCompany).

---

## 2. Projects

### 2.1 Google Chrome "Mirror Dimension" for agents

Source: [blog.google/security/mirror-dimension-for-agents](https://blog.google/security/mirror-dimension-for-agents/).
The authors are Aaron Selya and Charles Qian (Chrome AI Security). The page shows no publication date.

* **Implemented vs. proposed.** The post describes these as built:
  * the gVisor sandbox;
  * the interception proxy;
  * on-the-fly TLS leaf certificates;
  * the multi-agent site-generation harness;
  * the red-teaming pipeline.

  It reports "more than 1,000 tests candidates every day per security engineer", against "a few dozen realistic
  attacks a day" by hand. Site counts are inconsistent within the post: "thousands of simulated websites" in the
  summary, "hundreds of interactive, stateful web pages" in the body. Two things are stated as aspirations: that
  hermetic sandboxes "will be the baseline", and that the approach could be extended to coding agents.
* **Open source / license.** No repository, data or paper is linked. No license is stated.
* **Interfaces.** Agents use a headless Chrome. The injection surfaces listed are visible text, images, DOM, ARIA
  labels and WebMCP tool responses ([WebMCP](https://developer.chrome.com/docs/ai/webmcp)).
* **State.** The sites are generated: a research agent writes the specification, critical user journeys and UI
  mock-ups (with Nano Banana), and a builder agent writes HTML, CSS and JavaScript. They are described as
  "interactive, stateful web pages". Whether there is a server-side backend is not stated. The post requires
  "side effects from one trajectory must never persist into the next". The reset mechanism is not described.
* **Shared world / multi-agent.** Unknown / no. Multi-agent applies only to the generation and red-teaming
  harnesses.
* **Generation and validation.** Humans define user journeys. Critic agents check:
  * "accessibility tree semantics (ARIA)";
  * form-validation edge cases;
  * console JavaScript errors;
  * element layout across screen sizes.

  They do this through Chrome DevTools MCP ([repo](https://github.com/ChromeDevTools/chrome-devtools-mcp),
  Apache-2.0), interacting with the rendered page rather than reading source. The post does not say how agent task
  success is scored.
* **Isolation / networking.**
  * Chrome runs inside gVisor. "The sandbox captures the network connection and routes it to a local proxy that
    serves our generated site." The post does not describe the DNS set-up in detail.
  * Agents navigate to **real URLs**. The team rejected sandbox domains because they cause eval awareness.
  * Plain HTTP was rejected because of "Not Secure" warnings. Instead, "the proxy mints TLS leaf certificates on
    the fly that chain back to a custom Root Certificate Authority (CA)", which is pre-installed in the sandboxed
    Chrome.
  * Injection payloads are selected by **request headers** such as `X-Sandbox-Injection-Key`, which the sandbox
    server strips. This leaves the visible URL "completely pristine". The team had first used URL parameters, but
    agents noticed them.
* **Reusable here.**
  * The ideas, not code (there is none).
  * Header-selected variants work with Playwright request headers.
  * Per-host certificates chained to a world-local root CA can be minted with Python `cryptography` (installed in
    `/venv/main`).
  * The critic checklist can be implemented deterministically with Playwright and
    [axe-core](https://github.com/dequelabs/axe-core) (MPL-2.0).
  * gVisor is not available here.
* **Beyond Crucible / lacks.** It goes beyond Crucible on visual realism, TLS realism, large-scale agent-built
  sites and systematic prompt-injection red-teaming. It lacks persistence, shared state, multiple agents and an
  economy. Its realism comes from **impersonating real domains**, which Crucible does not do (see §4, "Do not
  import").

### 2.2 Microsoft WildAWM ("Agent World Model in the Wild")

Sources:
* [README](https://github.com/MSR-Orchard/WildAWM)
* [arXiv 2610.26828](https://arxiv.org/abs/2610.26828)
* source files `synthesis/wildawm/synthesis/tasks/{orchestrator,probe,verifier,_selfcheck,capability_contract}.py`
  and `synthesis/configs/synthesis.example.jsonc`

Last push 2026-10-21.

* **Implemented.** The pipeline, a dashboard, evaluation harnesses for 9 benchmarks, 341 environment-task pairs
  ([HF dataset](https://huggingface.co/datasets/ChilleD/WildAWM), validation split 33) and the WildAWM-35B-A3B and
  WildAWM-4B checkpoints. The abstract says that tasks "can require over 100 agent steps" and that the strongest
  model it reports solves "only 60%". The README says "the entire pipeline is powered by open-source LLMs".
* **License.** The repository has no LICENSE file (GitHub reports none) and the HF dataset declares none. Treat it as
  all rights reserved.
* **Interfaces.** It is explicitly cross-interface ("XAI"):
  * local files and command-line workflows;
  * MCP services backed by application databases;
  * GUI websites from [WebHarbor](https://github.com/aiming-lab/WebHarbor), used through Playwright MCP.

  Its evaluation harness, OpenCode, is "equipped with GUI, MCP, and shell tools". The task generator rejects a
  task if a completed trajectory did not use **all three** of `cli`, `mcp` and `web` (`orchestrator.py`,
  rule `all-three-interfaces`, verdict `reject_interface_bypass`).
* **State.** Executable. A world is "the cumulative state of one persona's computer: local files, MCP application
  databases, WebHarbor state, and optional per-world code overlays". Each day, an LLM plans the day and then writes
  "a deterministic, offline replay program". Each day is then pre-baked into a snapshot.
  `--verify_rebake yes` rebuilds the chain from `delta.sh` and compares it with the snapshots.
* **Shared world.** The world is one snapshot spanning files, MCP DBs and web state. Cross-application history is
  authored by the day programs. Whether the apps interact causally at runtime is not described.
* **Multi-agent.** No. Multiple models and harnesses are used only as a validation ensemble.
* **Generation and validation.** This is the strongest pipeline found (`orchestrator.py` docstring: "sketch,
  author, validate, ensemble-solve, gate, probe, accept"):
  1. *Sketch and diversity gate.* Task sketches are drafted from real files, MCP records and WebHarbor state. A
     diversity gate steers new sketches away from crowded areas.
  2. *Validate matrix.* On scratch copies, the verifier scripts must crash on neither world, the **clean world must
     fail** and the world after `golden.py` **must pass**. A lightweight LLM judge must agree on both
     (`validate_matrix`).
  3. *Heterogeneous ensemble.* Rollouts are drawn from a model × harness pool. The harnesses are `opencode`,
     `codex`, `claude-code`, `nanobot`, `openclaw` and `hermes`. There are hinted tiers and "bare" (task.md-only)
     rollouts.
     * A task is `reject_unsolvable` if it has zero completions across both hint tiers.
     * Difficulty comes from bare completions out of `live_test_n = 3`: hard means at most 1, medium means exactly 2.
     * `step_gate = 30` rejects a task when every hinted rollout completes in fewer LLM calls.
     * Re-authoring rounds (`max_deepen`, `max_ease`, ...) rewrite the public task when it is too easy or too hard.
       Changes made only to the hidden verifier are explicitly rejected.
  4. *Three-level agreement.* "V = V_LLM = V_agent". The programmatic verifier, a lightweight LLM judge and a heavy
     agent-as-judge must give the same 3-class verdict (completed / failed / world_imperfect).
  5. *Counterfactual "pressure probe"* (`probe.py`, `counterfactual_probe_rate = 1.0`). It has two stages:
     * a blind judge forms its own verdict from the before/after snapshots;
     * the verifier author is then told that an independent audit found the **opposite** verdict, and is allowed to
       edit the verifier.

     Any edit to the verifier files counts as a "cave" and the task is rejected as `reject_verifier_fragile`.
  6. *Capability contract.* Fields stamped by tools (which an agent cannot set) are discovered from live artifacts,
     so verifiers do not gate on them.
* **Isolation.** It needs Docker: there are creator, agent, MCP, WebHarbor and worldctl images, and the default
  evaluation sandbox is one container per task. Orchard (Kubernetes) and Modal backends are optional.
* **Reusable here.** All of the gates, as **patterns**, re-implemented over Crucible forks (§4 D1). Nothing can be
  copied because there is no license, and the runtime is Docker-bound.
* **Beyond Crucible / lacks.** It goes beyond Crucible on:
  * rigorous acceptance of generated tasks;
  * mandatory cross-interface tasks;
  * persona-driven long histories (files and documents);
  * export to the [Harbor](https://github.com/harbor-framework/harbor) format (Apache-2.0).

  It lacks other agents, an economy, persistent consequences across tasks and a shared causal world.

### 2.3 Snowflake Agent World Model (AWM)

Sources:
* [README](https://github.com/Snowflake-Labs/agent-world-model)
* [arXiv 2602.10090](https://arxiv.org/abs/2602.10090) (ICML 2026 per README)
* source files `awm/core/{env,db,scenario,verifier}.py` and `awm/prompts.py`
* [OpenEnv env README](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/agent_world_model_env)
* [HF dataset](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K)

Last push 2026-05-28.

* **Implemented.** The synthesis CLI (`awm gen scenario|task|db|sample|spec|env|verifier|all`), environment
  management (`awm env start|check|check_all|reset_db`), an agent demo, verification, and adapted benchmarks
  (BFCLv3, τ²-bench-verified, MCP-Universe). Also 1,000 environments with 10 tasks per scenario (the OpenEnv README
  says "10,000 tasks"), and the Arctic-AWM 4B, 8B and 14B models.
* **License.** The code repository has no LICENSE file. The AgentWorldModel-1K dataset is **CC-BY-4.0** (HF card).
* **Interfaces.** A unified MCP interface at `http://localhost:8001/mcp`, built with FastAPI and `fastapi-mcp`
  (`pyproject.toml`). OpenEnv wraps it as `reset/list_tools/call_tool`, plus special `verify` and `done` tools.
* **State.** Executable and SQL-backed. The abstract says environments "are code-driven and backed by databases,
  providing more reliable and consistent state transitions than environments simulated by LLMs". Each run gets an
  isolated folder with `initial.db` and `final.db`.
* **Shared world / multi-agent.** No. Each scenario is its own server process and SQLite file.
* **Generation pipeline** (schema-first): seed names → 1,000 scenarios → 10 tasks per scenario (the tasks "are also
  serving as the requirements for building the environment") → DB schema → sample data for the initial state →
  API spec → MCP environment code → verifier.
* **Validation.**
  * Scenario dedup by embedding similarity (`embedding_similarity_threshold = 0.85`, `scenario.py`).
  * DB generation is retried when more than 10% of tables fail (`error_threshold = 0.1`).
  * Environment code is regenerated up to `max_retry = 4` times, with an LLM summary of the previous error messages
    fed back (`env.py`).
  * The environment test starts the server on a copy of the DB and checks that it comes up and lists tools.
  * Generated verifier code is executed against a read-only DB and checked only for running and returning the
    right format (`verifier.py`). This is a smoke test. It is **not** a check that the verifier fails on the
    initial state.
  * Two verifier modes: `sql` is "code-augmented LLM-as-a-Judge". The generated function queries
    `initial_db`/`final_db` and an LLM judges the results. `code` is purely code-based.
* **Isolation.** Plain local processes. The README mentions no Docker. The HF Space uses the Docker SDK.
* **Reusable here.** The pipeline shape and its retry/dedup mechanics. The CC-BY-4.0 dataset could, with
  attribution, seed **themes and schemas** for Crucible's generated sites. The code cannot be copied (no license).
  The per-scenario servers are disconnected, so they should not be imported as environments.
* **Beyond Crucible / lacks.** It goes beyond Crucible on breadth (1,000 domains) and cheap automated generation.
  It lacks cross-service causality, identity and money, other actors, persistence and strong verifier validation.

### 2.4 RUC Agent-World

Sources:
* [README](https://github.com/RUC-NLPIR/Agent-World)
* [GRAPH_SYNTHESIS.md](https://github.com/RUC-NLPIR/Agent-World/blob/main/GRAPH_SYNTHESIS.md)
* [arXiv 2604.18292](https://arxiv.org/abs/2604.18292)

MIT. Last push 2026-10-20.

* **Implemented vs. proposed.** Released:
  * 563 environments, selected from the paper's 1,978: 500 from high-usage Smithery MCP server themes and 63 from
    industrial PRDs;
  * 8,927 executable tools (paper: 19,822) and 67,096 records;
  * 1,432 Q/A/rubric examples;
  * the graph synthesis pipeline;
  * an SFT set of about 44K examples on HF.

  The paper's "continuous self-evolving agent training" and arena are **not** in the repository. The README says the
  environments are "stateful runtimes, not a drop-in RL trainer". No model weights are included.
* **Interfaces.** Each tool is a Python source string. It is `exec`'d and bound to a DB directory through the
  `MCP_DB_DIR` environment variable. The README documents no MCP server. Synthesis talks to an OpenAI-compatible
  vLLM server.
* **State.** Executable tool code over per-environment JSON "database" files. An LLM writes queries and rubrics; it
  does not simulate tool responses.
* **Shared world / multi-agent.** No. Because tools find their DB through a process-wide variable, environments
  must run in separate **processes** (GRAPH_SYNTHESIS.md).
* **Generation and validation** (seven stages, GRAPH_SYNTHESIS.md):
  1. Analyse each tool's reads, writes and dependencies. A strong dependency means a parameter can only come from
     another tool's output.
  2. Build a dependency graph.
  3. Take random walks that prefer edges whose parameters actually match.
  4. **Execute** the walk on a private DB copy. Parameters come from upstream outputs or from the same record;
     nothing is ever made up. Errors, empty results and duplicate responses are discarded. "Whatever the tool
     returns is the reference answer."
  5. An LLM writes the user question without leaking the concrete values.
  6. An LLM derives the answer and rubric from the observations. If the observations cannot determine an answer it
     returns `unanswerable` and the item is dropped.
  7. **Replay** the same chain on a clean copy and keep only what is the same in both runs, which removes clock
     stamps and fresh ids.

  Reported run: 602 questions over 274 of the 563 environments, an average of 3.9 real tool calls per question,
  about 75 minutes on two 8-GPU L20 machines. "About half of the environments" cannot produce a reproducible chain.
  `run.sh --check` is structural only (syntax, checkpoints, DB pairing).
* **Isolation.** The README advises an isolated process or container for running tool code, and never to use
  production credentials.
* **Reusable here.** Graph-walk task synthesis with execution-derived answers and a replay-stability filter. This
  maps directly onto Crucible's tool registry and `World.fork` (§4 D2). The environments themselves are
  disconnected fixtures, already rejected in RELATED_WORK.md.
* **Beyond Crucible / lacks.** It goes beyond Crucible on tool breadth taken from the real MCP ecosystem and on
  graph-based task synthesis. It lacks cross-environment state, agents, time and persistence.

### 2.5 TheAgentCompany

Sources:
* [README](https://github.com/TheAgentCompany/TheAgentCompany)
* [docs/EVALUATION.md](https://github.com/TheAgentCompany/TheAgentCompany/blob/main/docs/EVALUATION.md)
* [servers/rocketchat/npc/NPC.md](https://github.com/TheAgentCompany/TheAgentCompany/blob/main/servers/rocketchat/npc/NPC.md)
* [arXiv 2412.14161](https://arxiv.org/abs/2412.14161)

MIT. Last push 2025-11-17.

* **Implemented.** 175 tasks across SDE, PM, data science, HR, finance and admin roles. Each task is a Docker image
  with `/utils/init.sh`, an encrypted `evaluator.py.enc`, `/instruction/task.md` and `/workspace`. Self-hosted
  GitLab, Plane, ownCloud and RocketChat run "with pre-baked data". Setup needs "docker and docker compose
  installed, and 30+ GB of free disk space".
* **Interfaces.** Web browsing, terminal and code, and chat with simulated colleagues. Any agent platform can be
  used.
* **State.** Real services. The NPC colleagues are LLM characters built on
  [Sotopia](https://arxiv.org/abs/2310.11667) (MIT). Each has a profile with name, role, responsibilities, project
  and channels. The baselines used `claude-3-5-sonnet-20241022` both for NPCs and for the LLM graders (paper).
* **Shared world.** One fictional company spread over four services, each with its own DB. There is no shared
  ledger or identity beyond accounts.
* **Multi-agent.** One agent plus NPCs. Every task resets the services ("might take up to 10 minutes").
* **Validation.** Checkpoints with deterministic or LLM graders. The paper defines partial credit as
  `S_partial = 0.5 · Result/Total + 0.5 · S_full`. Evaluators are encrypted "to prevent the agent from peeking"; the
  key is published in the docs.
* **Isolation / networking.** `init.sh` "would add the server's IP to the `/etc/hosts` file, so that the agent can
  visit the services using the synthetic `the-agent-company.com` hostname". It uses `docker run --network host`.
* **Reusable here.** These patterns:
  * the checkpoint partial-credit formula;
  * NPC persona tables;
  * the synthetic company hostname;
  * hiding evaluators from agents.

  The stack itself is blocked (Docker per task, 30+ GB).
* **Beyond Crucible / lacks.** It goes beyond Crucible on real third-party software (GitLab and so on) and on LLM
  colleagues that agents must ask for information. It lacks persistence across tasks, an economy and other learning
  agents.

### 2.6 BrowserGym and WebArena

**BrowserGym**

Sources:
* [README](https://github.com/ServiceNow/BrowserGym)
* source files `browsergym/core/src/browsergym/core/{env,observation,task,constants}.py` and `action/highlevel.py`
* [arXiv 2412.05467](https://arxiv.org/abs/2412.05467)

* **License.** The LICENSE file is Apache-2.0. The GitHub API reports "NOASSERTION" because of the file's format.
  Last push 2026-11-18. `browsergym-core` 0.14.3 is **already installed** in `/venv/main`.
* **Implemented.** A gymnasium environment over Playwright Chromium, with adapters for MiniWoB, WebArena,
  WebArena-Verified, VisualWebArena, WorkArena, AssistantBench, WebLINX, OpenApps, TimeWarp and KNOWS. It is the
  base for AgentLab.
* **Interfaces.**
  * The observation dict holds `chat_messages`, `goal_object`, `open_pages_urls`, `screenshot`, `dom_object`,
    `axtree_object`, `extra_element_properties` (visibility, bounding box, set-of-marks), `focused_element_bid` and
    `last_action_error`.
  * Every element is tagged with a `bid` attribute (`BROWSERGYM_ID_ATTRIBUTE = "bid"`).
  * Action subsets are `chat`, `infeas`, `bid`, `coord`, `nav` and `tab`, plus benchmark-specific sets.
* **Task API.** `AbstractBrowserTask` has `setup(page) -> (goal, info)`, `validate(page, chat_messages) -> (reward,
  done, msg, info)`, `cheat(page, chat_messages)` (an oracle solution) and `teardown()`. A new benchmark "simply
  requires to inherit" this class.
* **State, shared world, multi-agent.** Not applicable. BrowserGym is a harness and the state lives in whatever site
  the task serves.
* **Reusable here.** Directly. It is pip-installable, Playwright-based and needs no Docker for core or MiniWoB.
  Crucible can expose browser-mode goals as BrowserGym tasks (§4 D5).

**WebArena**

Sources:
* [README](https://github.com/web-arena-x/webarena)
* source file `evaluation_harness/evaluators.py`
* [arXiv 2307.13854](https://arxiv.org/abs/2307.13854)

Apache-2.0. Last push 2025-11-26.

* **Implemented.** 812 tasks over self-hosted sites: shopping (7770), shopping admin (7780), a Reddit-like forum
  (9999), GitLab (8023), map (3000) and a Wikipedia mirror (8888). They are configured through environment variables
  and deployed with Docker or a prebuilt AMI. Auto-login cookies are prepared beforehand.
* **Observation.** `accessibility_tree`, HTML or screenshot. Actions address elements by id
  (`create_id_based_action("click [id]")`).
* **Validation.** Hand-annotated functional evaluators:
  * `StringEvaluator`: `exact_match`, `must_include`, LLM `fuzzy_match`, LLM `ua_match` for unachievable tasks;
  * `URLEvaluator`;
  * `HTMLContentEvaluator` (`program_html`: locate elements on the final page and check required contents).

  The abstract reports 14.41% success for the best GPT-4 agent against 78.24% for humans.
* **Shared world / multi-agent / generation.** No / no / no. Sites are reset after an evaluation run.
* **Reusable here.** The evaluator taxonomy, in particular checking the final page state with locators and checking
  unachievable tasks. The sites need Docker (blocked; see RELATED_WORK.md).

### 2.7 OSWorld

Sources:
* [README](https://github.com/xlang-ai/OSWorld)
* `run.py`
* `evaluation_examples/`
* [arXiv 2404.07972](https://arxiv.org/abs/2404.07972)

Apache-2.0. Last push 2026-11-14.

* **Implemented.** 369 tasks on real Ubuntu and Windows apps (the abstract reports 72.36% success for humans and
  12.24% for the best model at the time), and OSWorld-Verified (2025-07). Providers are VMware, VirtualBox, Docker
  with KVM, AWS, Modal and Daytona.
* **Interfaces.** `--observation_type` is one of `screenshot`, `a11y_tree`, `screenshot_a11y_tree` or `som`. The
  default action space is `pyautogui`.
* **State.** A real OS in a VM. Within the VM, all apps share real state: files, browser profile and settings.
* **Task format** (worth copying). A task is a JSON file with:
  * `instruction`;
  * `config`, a list of setup steps such as `launch`, `download`, `open`, `execute` and `sleep`;
  * `related_apps`;
  * `evaluator`, made of `postconfig` steps, a `func` such as `match_in_list`, and typed `result` and `expected`
    getters.

  There is a separate `test_infeasible.json` split.
* **Shared world / multi-agent / generation.** Inside one VM / no / no.
* **Reusable here.** The declarative task-and-evaluator schema and the infeasible-task split. The runtime is
  blocked here (VM or KVM, or a cloud provider).
* **Beyond Crucible / lacks.** It goes beyond Crucible on desktop GUIs and real application software. It lacks
  synthetic services, other agents and persistence.

### 2.8 Qwen WebWorld

Sources:
* [README](https://github.com/QwenLM/WebWorld)
* [arXiv 2602.14721](https://arxiv.org/abs/2602.14721)
* [HF model card](https://huggingface.co/Qwen/WebWorld-8B) (apache-2.0)

Last push 2026-02-25.

* **Implemented.** WebWorld-8B, 14B and 32B (on Qwen3), the training trajectories
  ([Qwen/WebWorldData](https://huggingface.co/datasets/Qwen/WebWorldData)), WebWorld-Bench and a demo. The README
  claims "1M+ real-world web interaction trajectories" and "multi-turn simulation up to 30+ steps". The abstract
  reports +9.2% on WebArena for Qwen3-14B trained on WebWorld trajectories.
* **License.** The README says "All models and data are licensed under Apache 2.0". The repository has no LICENSE
  file, so the license of the code itself is not stated.
* **Interfaces.** Text page states in A11y tree, HTML, XML, Markdown or natural-language form. The model is
  prompted with a state and an action and predicts the next state.
* **State.** **Model-generated.** There is no backend. "Consistent state tracking" is a learned property, measured by
  a "Factuality Score" and a "Web Turing Score".
* **Shared world / multi-agent / validation of generated tasks.** No / no / none beyond the intrinsic benchmark.
* **Reusable here.** As an *agent-side* imagination model, for lookahead search or cheap pretraining data. Never as
  an environment authority, because it cannot enforce conservation, ownership or permissions. This was already
  decided in RELATED_WORK.md.

### 2.9 DeepMind Concordia

Sources:
* [README](https://github.com/google-deepmind/concordia)
* source files `concordia/components/game_master/*`, `concordia/prefabs/game_master/*` and
  `concordia/environment/engines/*`
* [arXiv 2312.03664](https://arxiv.org/abs/2312.03664)

Apache-2.0. PyPI `gdm-concordia`, Python ≥ 3.12. Last push 2026-11-19.

* **Implemented.** Entities are built from components (memory, instructions, action selection) and instantiated
  from prefabs. The engines are `sequential`, `simultaneous` and `asynchronous`. Game-master components include:
  * `event_resolution`, `make_observation`, `next_acting`, `switch_act`, `world_state`, `scene_tracker` and
    `terminate`;
  * `inventory`: "a grounded inventory tracking amounts of items in python";
  * `payoff_matrix`;
  * interrupt-driven variants.

  GM prefabs include `marketplace`, `async_social_media`, `game_theoretic_and_dramaturgic` and `scripted`.
* **Interfaces.** Python API, GUI/CLI editor. "Agents take actions by describing what they want to do in natural
  language. The GM then translates their actions into appropriate implementations". In digital settings this can
  include API calls to apps.
* **State.** Mostly **LLM-generated**. The GM resolves events with an LLM, and the `world_state` component holds
  string-to-string variables maintained by the LLM. Grounded Python components (inventory, payoff matrix) are
  optional.
* **Shared world / multi-agent.** Yes. The GM holds the shared world, and several GMs can hand over
  (`next_game_master`).
* **Generation / validation.** Scenarios are configured from prefabs. There are no task generation or verifiers.
* **Reusable here.** The **pattern**: actors express intent, and an authoritative component decides the outcome. In
  Crucible the authority is the deterministic kernel, not an LLM. The engine taxonomy and persona and memory
  components are also useful. The library could be used *inside* LLM actors (Apache-2.0, pure Python).
* **Beyond Crucible / lacks.** It goes beyond Crucible on rich language-mediated social behaviour, memory and
  persona tooling. It lacks hard invariants and exact replay. Outcomes are only as consistent as the GM's LLM.

### 2.10 CAMEL OASIS

Sources:
* [README](https://github.com/camel-ai/oasis)
* source files `oasis/social_platform/{platform,recsys}.py` and `schema/*.sql`
* [arXiv 2411.11581](https://arxiv.org/abs/2411.11581)

Apache-2.0. PyPI `camel-oasis`. Last push 2026-11-16.

* **Implemented.** Simulations of X-like and Reddit-like platforms, "up to one million agents", with "23 actions"
  (follow, comment, repost, search, trend, mute and others). Recommender systems are random, Reddit hot-score and
  personalised (the personalised one uses TwHIN-BERT embeddings in `recsys.py`). There is a clock module.
* **Interfaces.** `env = oasis.make(agent_graph, platform, database_path)` followed by async `env.step({agent:
  action})`. `ManualAction` (scripted) and `LLMAction` (the LLM chooses) can be mixed in the same step.
* **State.** **Executable**. The platform is code over SQLite, with tables for users, posts, comments, likes,
  follows, mutes, groups, products, recommendations and a `trace` log of every action. The LLM chooses actions; the
  platform applies them.
* **Shared world / multi-agent.** One platform DB shared by all agents, which persist for a run.
* **Cost.** The README gives a measured reference: 100 agents for one step with activation probability 1 used
  335,600 input and 16,750 output tokens. An activation probability per step controls cost.
* **Validation.** Not applicable. The paper replicates phenomena: information spread, polarisation, herd effects.
* **Reusable here.** These patterns:
  * mixing scripted and LLM actions in one step;
  * activation probability as the cost knob for LLM actors;
  * the hot-score and interest recommender as forum-feed mechanisms;
  * the token-budget table as a planning reference.

  The package pulls in CAMEL and a transformer recommender. Do not import it into the kernel.
* **Beyond Crucible / lacks.** It goes beyond Crucible on LLM social populations at scale and on recommender
  dynamics. It lacks other services, money and agents that are learners.

### 2.11 Briefly: AppWorld, τ²-bench, WebHarbor, OpenApps

* **AppWorld** ([README](https://github.com/StonyBrookNLP/appworld), [arXiv 2407.18901](https://arxiv.org/abs/2407.18901)).
  9 apps with 457 APIs over about 100 fictitious users, and 750 tasks. It offers `appworld serve` with or without
  Docker (Podman is suggested without root), and an MCP server. Evaluation is "database-state-based unit tests ...
  without causing any collateral damage". **License:** the public part is Apache-2.0. The protected portion (app
  code, tasks, tests) is in encrypted `.bundle` files under Apache-2.0 "with an additional requirement that any of
  its public redistribution ... should be done in an encrypted format". It is already used by CAM 2.0. For Crucible,
  reuse the *collateral-damage* check as a pattern.
* **τ²-bench** ([README](https://github.com/sierra-research/tau2-bench), [τ-bench arXiv 2406.12045](https://arxiv.org/abs/2406.12045)).
  MIT. Domains are airline, retail, telecom, banking_knowledge, intake and mock. An LLM user simulator follows a
  persona; the agent follows a domain policy. Grading compares "the database state at the end of a conversation
  with the annotated goal state". The reliability metric is **pass^k**. There is an optional gymnasium interface
  (`--extra gym`). For Crucible, reuse pass^k and the LLM-user-against-a-policy pattern.
* **WebHarbor** ([README](https://github.com/aiming-lab/WebHarbor), [arXiv 2611.03634](https://arxiv.org/abs/2611.03634)).
  153 Flask mirrors of **real brands**. One example is `sites/amazon/`, with `app.py`, `seed_data.py`, Jinja
  templates and per-task `verify/verify_N.py`. There are 3,518 tasks. A control plane resets a site by killing it,
  copying `instance_seed/` and respawning it ("sub-second database resets"). Distribution is a Docker image of about
  20 GB. The verifier checks:
  * navigation, so that the agent cannot shortcut;
  * answer facts;
  * that the instance DB is unchanged on read-only tasks.

  **No license file.** For Crucible, reuse the verifier style. Do not take the brand mirrors.
* **OpenApps** ([README](https://github.com/facebookresearch/OpenApps)). Python apps that need "no Docker or OS
  emulators". Hydra config generates "thousands of versions of each app" by changing appearance, layout and content,
  and rewards are read from app state. **License: CC-BY-NC-4.0**, so it is a pattern only.

---

## 3. Cross-cutting observations

1. **Executable state is the norm. LLM-simulated state is the exception.** AWM, Agent-World, WildAWM, WebHarbor,
   OASIS, AppWorld and τ²-bench all run code over a database. WebWorld, and Concordia's LLM GM, are the
   model-generated exceptions. AWM states the reason directly: code-driven environments give "more reliable and
   consistent state transitions than environments simulated by LLMs". This matches Crucible's rule that
   deterministic mechanisms decide outcomes.
2. **Generated worlds are disconnected.** Every generator found (AWM, Agent-World, WildAWM, Mirror Dimension,
   WebHarbor) produces per-scenario or per-site state, which is reset after every trajectory. WildAWM comes closest
   to a coherent world (one persona's computer across interfaces), but its cross-app history is authored offline.
   Crucible's distinctive contribution is that generated sites live in **one causal world**: shared identity, bank,
   search, mail and other actors.
3. **Validation maturity varies widely.** In increasing order:
   * structural checks (Agent-World `--check`);
   * server starts and verifier runs (AWM);
   * answers derived from execution plus a replay-stability filter (Agent-World graph synthesis);
   * clean-fails/golden-passes matrix, heterogeneous solver ensemble, difficulty bands, interface gate, three-judge
     agreement and a counterfactual probe (WildAWM).

   Only WildAWM tests whether the **verifier** is robust. It does so with LLM pressure, because its verifiers are
   LLM-written scripts.
4. **Isolation.** Almost everything realistic depends on Docker or a VM: TheAgentCompany, WebArena, OSWorld,
   WebHarbor, WildAWM, and Mirror Dimension (gVisor). The Python-only exceptions are AWM, Agent-World, OASIS,
   Concordia, BrowserGym core and OpenApps. Hermetic networking is done in three ways: an interception proxy plus a
   minted CA (Mirror Dimension), `/etc/hosts` entries (TheAgentCompany), or ports on localhost (WebHarbor, WebArena).
   Crucible's approach is Chromium `--host-resolver-rules`, a deny-all route handler and a brokered RPC channel for
   sandboxed code. It already matches the first approach's hermeticity without needing privileges.
5. **Interfaces.** BrowserGym and OSWorld have settled on observations of AXTree, DOM and screenshot plus
   set-of-marks, with element-id actions. MCP has become the common tool interface (AWM, AppWorld, WildAWM).
   WildAWM is the only project that *requires* tasks to span GUI, CLI and MCP.

---

## 4. Design decisions for Crucible

Context: this branch already has, uncommitted at the time of reading, `services/net.py` and
`worldgen/sitegen.py`.

* `services/net.py` provides recorded `net.fetch` for API agents and browsers alike, `net.register`/`net.point`
  names, a crawler, and an `appdb` document store that "is world state: it is snapshotted, forked and replayed".
* `worldgen/sitegen.py` generates schema-first site specs and refers to a future `crucible.web.critic` that
  generated code must pass.

The decisions below are written to plug into those. Each is a pattern to implement in Crucible's own code. None
requires importing another project's codebase.

### D1. Admission gates for generated tasks (WildAWM-style, re-implemented on forks)

Apply these to every generated `GoalContract` and every generated site task before it enters a suite.

1. **Clean-fail.** The goal predicate is evaluated on the start snapshot and **must be false**. This is WildAWM's
   "verifier fails on the clean state".
2. **Golden-pass.** A scripted golden program runs in `World.fork` and **must make the predicate true**. It plays the
   role of BrowserGym's `cheat()` and WildAWM's `golden.py`. The program must act through ordinary recorded
   commands as an ordinary principal.
3. **Counterfactual verifier checks, made deterministic.** In forks, apply *near-miss* variants of the golden
   program and require the verifier to reject every one:
   * the wrong recipient or amount;
   * a partial completion;
   * the same effect achieved by another principal;
   * the effect achieved and then reverted;
   * the effect achieved through a forbidden interface or a sock-puppet organisation.

   This is mutation testing of the verifier. It replaces WildAWM's LLM "pressure probe", which is only needed
   because WildAWM's verifiers are LLM-written. Keep the LLM probe only for goals that are judged by an LLM.
4. **Heterogeneous solver attempts.** Run an ensemble of cells in forks: {scripted policy, frozen-LLM tool agent,
   LLM browser agent (BrowserGym obs), CAM 2.0} × seeds.
   * Accept a task only if at least one cell completes it (solvability).
   * Label difficulty from bare attempts, using WildAWM's bands over 3 attempts: hard ≤ 1, medium = 2.
   * Reject a task when every completion takes fewer than K recorded actions (WildAWM `step_gate`).
   * Chromium concurrency is the binding constraint (SYNTHETIC_INTERNET.md §7), so run browser cells serially.
5. **Interface gate.** Record the interface of every command in its event metadata: tool, HTTP, MCP, browser or
   sandboxed code. A task labelled cross-interface is accepted only if completions actually used the required
   interfaces. This follows WildAWM's `all-three-interfaces` rule. The event log makes the check exact, where
   WildAWM has to infer it from trajectories.
6. **Agreement when judges exist.** If any part of a goal is LLM-judged (for example free text), require predicate
   = lightweight judge = agent judge, as in WildAWM's "V = V_LLM = V_agent". Otherwise reject the goal.
7. **Diversity gate.** Deduplicate tasks by embedding or by tool-chain signature before acceptance. AWM uses cosine
   similarity 0.85 for scenarios.
8. **Collateral-damage check** (AppWorld, WebHarbor). Diff events and state between the before and after forks.
   Changes outside the declared footprint void success. Crucible's event log makes this a query rather than a
   heuristic.

### D2. Long-horizon tasks synthesised from execution (Agent-World graph walks over Crucible's tool registry)

1. Build a dependency graph of tools across hosts from the endpoint schemas. Strong edges are ids that only appear
   in outputs; examples are `id.create_org` → `biz.register` → `bank.open_account` → `cloud.deploy`.
2. Random-walk 3 to N steps across **several services**.
3. Execute the walk in a fork, taking arguments only from prior outputs or from the same record. Never invent
   arguments.
4. Replay the walk in a **second fork with a different stochastic seed**. Keep only the facts that agree, so the
   goal cannot depend on clock values or fresh ids. Drop walks with too little stable content, or that cannot be
   answered.
5. Turn the walk's final state delta into a `GoalContract` predicate over authoritative state, not a reference
   answer string. Then send it through D1.

Because Crucible worlds persist, a walk can also span simulated days. Examples are "a deployment that has stayed
solvent for 7 days" or "a contract whose clause fired". Disconnected fixtures cannot express such tasks.

### D3. Schema-first generated sites (AWM pipeline, with Crucible keeping the authority)

* **Pipeline.** Adopt AWM's stage order for `sitegen`/`codegen`: theme → tasks-as-requirements → entity schema →
  seed data → route/API spec → code → verifiers. Add two pieces of AWM mechanics:
  * bounded regeneration, with an LLM-summarised error fed back (AWM uses up to 4 retries);
  * dedup of themes by embedding.

  The CC-BY-4.0 AgentWorldModel-1K schemas could seed theme diversity, with attribution.
* **Authority.** Keep authority where `net.py` puts it. Generated code runs as an agent-deployed service in the
  sandbox, stores data only in `appdb` (world state), moves money only through bank invoices and identifies users
  only through `net.login`. Never admit generated code as a kernel module. This is consistent with the module
  admission rules in SYNTHETIC_INTERNET.md, and it is what keeps replay and forks exact. WebHarbor and AWM use
  standalone SQLite files, which would break this.
* **Critic (`crucible.web.critic`) = Mirror Dimension's checklist, made deterministic.** Use Playwright with
  axe-core (MPL-2.0). axe-core can be loaded into the page as an unmodified script, so the MPL obligations attach
  only to that file. The critic must check:
  * ARIA and AXTree semantics, including that every form control has an accessible name;
  * that there are zero console errors;
  * form-validation edge cases: empty fields, oversize input, wrong types and HTML or script payloads;
  * viewport sweeps for layout overlap;
  * that every route returns non-5xx for the owner and for an anonymous visitor;
  * that private pages carry `noindex`;
  * that the CSRF and authorisation rules in SYNTHETIC_INTERNET.md §6 are followed.

  Then run AWM-style smoke tests (the site installs and its seeded routes work) and at least one D1-gated task per
  site.
* **Variants** (OpenApps pattern, implemented independently because of CC-BY-NC). Seed-driven theme, layout and
  copy variations over the same spec. These test robustness to appearance without changing semantics.

### D4. Hermetic synthetic DNS and TLS for real browsers (Mirror Dimension, without impersonation)

1. **Names.**
   * Keep synthetic names and resolve them from world state: `net.register`/`net.point` plus `<host>.crucible`.
   * Generate the browser's `--host-resolver-rules` from the registry, keeping `MAP * ~NOTFOUND` and the deny-all
     route handler that `BrowserSession` already has.
   * If plausible variety is wanted (several TLDs), draw suffixes only from namespaces guaranteed never to resolve
     publicly: RFC 2606 `.test`, `.example`, `.invalid`, `.localhost`, and `.internal`, which ICANN permanently
     reserved for private use on 2024-07-29 ([CircleID](https://circleid.com/posts/20240810-icann-reserves-.internal-domain-for-private-use),
     [The Register](https://www.theregister.com/2024/08/08/dot_internal_ratified/)). `.crucible` is not a reserved
     name, so a future gTLD round could delegate it.
2. **Optional HTTPS.** This is a proposal, not yet verified in this container.
   * Generate a world-local root CA at server start and mint per-host leaf certificates on demand with
     `cryptography`, served through `ssl.SSLContext.sni_callback`. This is Mirror Dimension's minting proxy, without
     interception.
   * Trust it in Chromium without root, using one of:
     * an NSS DB in the browser's `$HOME/.pki/nssdb` (needs `certutil`, which is **not** installed here);
     * the `--ignore-certificate-errors-spki-list` flag;
     * at minimum, Playwright's `ignore_https_errors`.

   The value of HTTPS is secure contexts (`window.isSecureContext`, `Secure` cookies, WebCrypto) and `https://` URLs.
   It is not about warnings, which headless screenshots do not show. A cheaper alternative to evaluate is
   `--unsafely-treat-insecure-origin-as-secure` for the registered origins.
3. **Header-selected variants** (the `X-Sandbox-Injection-Key` analogue). The harness sets an out-of-band header
   through Playwright request headers or a route handler, and the web front end strips it. The header selects a
   **presentation** variant for that session only, such as an injected prompt-injection payload in a page, ad or
   forum post. It must never change world state. Record the variant id in the session's event metadata so that
   replay is exact. This gives prompt-injection red-teaming and A/B counterfactuals without visible URL changes.
4. **Sandboxed agent code.** Keep the brokered RPC with no inet sockets. Mirror Dimension relies on gVisor, which is
   unavailable here and is already refused by the sandbox backend layer.

### D5. BrowserGym compatibility for browser mode

Wrap browser-mode goals as `AbstractBrowserTask` subclasses:

* `setup` creates or forks the world, logs in and returns the goal;
* `validate` runs `GoalVerifier` on authoritative state, **not** on the page;
* `cheat` runs the D1 golden program;
* `teardown` discards the fork.

Use BrowserGym's observation (AXTree with `bid`, DOM snapshot, screenshot, set-of-marks) and action subsets
(`bid`, `nav`, `tab`, `chat`, `infeas`). Existing BrowserGym and AgentLab agents can then be evaluated on Crucible
unchanged. `browsergym-core` 0.14.3 is already in `/venv/main` (Apache-2.0). Keep Crucible's own `BrowserSession`
as the hermetic launcher, or pass the same Chromium flags to BrowserGym's Playwright context, because BrowserGym
does not provide network confinement.

### D6. GUI + CLI + MCP in one world

* **What exists.** Crucible has tools and MCP, HTTP and browser interfaces, plus sandboxed code. It does not have the
  WildAWM-style *personal workstation*: files, a shell and long personal histories.
* **Proposal: an event-sourced per-principal filesystem.** Content-addressed blobs as `fs.*` endpoints, with quotas.
* **Proposal: a restricted CLI tool.** `sh.run` executes a small command set, plus agent programs in the existing
  sandbox, against that filesystem and the agent's grants.
* **Interface tagging.** Interface tags (D1.5) then make GUI + CLI + MCP tasks checkable.
* **Persona histories.** WildAWM-style persona histories correspond in Crucible to simulated days run by Tier-2
  actors before the agent arrives. They are generated by the world's own mechanisms rather than by offline replay
  scripts, so they are causally consistent by construction.

### D7. LLM-driven social actors grounded by an authoritative game master

These patterns come from Concordia, OASIS, τ²-bench and TheAgentCompany.

* **Grounding.** LLM actors are ordinary principals. They propose actions as tool calls or as natural language
  that a parser turns into tool calls, and the **kernel resolves them**. This is Concordia's GM pattern with the GM
  replaced by deterministic code. Never let an LLM decide transactions, ownership or success. Concordia's `inventory`
  component (Python enforces bounds) is the right precedent; its LLM `event_resolution` is not.
* **Cost control** (OASIS). Wake each actor per tick with an activation probability. Mix scripted Tier-2 actions
  and LLM actions in the same tick, as OASIS does with `ManualAction` and `LLMAction`. Budget tokens from measured
  references such as OASIS's 100-agent figure. Fall back to the Tier-2 rule policy when the budget is exhausted.
* **Personas.** Give each actor a persona as data: role, responsibilities, projects, channels and private
  reservations, as in TheAgentCompany's NPC table and Crucible's existing minds. Memory is an actor-side component
  and can use `gdm-concordia` there (Apache-2.0). Engine semantics (sequential, simultaneous, asynchronous) map
  onto Crucible's scheduler priorities.
* **Uses.** τ²-style customers in `support` tickets who must be satisfied under a policy, and colleagues who hold
  information an agent must ask for (TheAgentCompany). Score with authoritative state and report **pass^k**
  (τ-bench).
* **Replay.** LLM outputs are recorded as command arguments in the event log. Replay is therefore exact without
  re-querying the model; only regeneration (re-running policies) is non-deterministic. Document this distinction.

### D8. Evaluation conventions to adopt

* TheAgentCompany's partial credit, `S_partial = 0.5 · checkpoints_passed/total + 0.5 · S_full`, for multi-stage
  goals. Checkpoints must be verified metrics, not agent claims.
* OSWorld-style declarative task files (setup steps, evaluator function, result and expected getters) as the
  serialisation format for generated goals, with an explicit **infeasible** split. These match WebArena's
  `ua_match` and BrowserGym's `report_infeasible`.
* WebHarbor-style anti-shortcut checks: required navigation or interface evidence. Crucible derives these from
  events rather than URLs.
* Evaluators stay out of the agent's reach, as now. TheAgentCompany's encryption with a published key is weaker than
  Crucible's process separation and is not needed.
* Optional: export accepted tasks in the Harbor package format (Apache-2.0) for external consumers. Harbor tasks
  assume containers, so this is for export only.

### D9. WebWorld only as an agent-side model

If used at all, WebWorld serves as a learned lookahead model *inside* agents, for planning or for cheap
pretraining trajectories. Its predictions are never used as environment state (§2.8).

### What NOT to import, and why

| Do not import | Reason |
|---|---|
| WebArena, TheAgentCompany, OSWorld, WebHarbor and WildAWM runtime stacks | They need Docker, docker compose or a VM with KVM: TheAgentCompany "30+ GB" with `--network host`; a WebHarbor image of about 20 GB; WildAWM creator/agent/MCP/worldctl images and one container per task; OSWorld VMware/VirtualBox/KVM. None of these run in an unprivileged container without Docker-in-Docker or KVM. |
| WildAWM, AWM and WebHarbor **code** | There is no license file, so the default is all rights reserved. Re-implement the patterns from the public descriptions. AWM's **dataset** is CC-BY-4.0 and may be used with attribution. |
| WebHarbor brand mirrors; Mirror Dimension's real-URL approach | Impersonation and trademark risk, and against Crucible's policy of synthetic hostnames (RELATED_WORK.md decision 4). The eval-awareness cost is accepted and should be measured, not avoided by mimicking real sites. |
| Agent-World environments | Each tool is `exec`'d from a JSON string over a per-env JSON DB, and tools locate their DB through a process-global variable. These are disconnected fixtures with no causal coupling. Their provenance (Smithery themes, third-party notices) would need its own review. |
| AWM per-scenario servers as environments | One SQLite file per scenario, with no identity, money or other actors. They would be islands outside the shared world. |
| AppWorld protected portion | Any public redistribution of the protected code or its derivatives must stay encrypted. Its per-task DB resets are also the opposite of persistence. |
| OpenApps code | CC-BY-NC-4.0 (non-commercial). |
| WebWorld as an environment | An LLM-predicted state cannot enforce conservation, ownership or permissions. The repository also has no LICENSE file, although the models and data are Apache-2.0. |
| Concordia's LLM game master as outcome authority | LLM-resolved events would break exact invariants and determinism. Only actor-side components are acceptable. |
| OASIS package inside the kernel | It pulls in the CAMEL framework and a transformer recommender (TwHIN-BERT). Re-implement the hot-score and interest feed mechanically, and use OASIS only as a design reference or an external comparison. |
| gVisor, Firecracker, bubblewrap isolation | Unavailable here (no KVM, no user namespaces). The sandbox backend layer already refuses them. |

### Open items to verify before relying on them

1. In this container, whether Playwright's Chromium 1.44 trusts a world-local CA through `--ignore-certificate-errors-spki-list`, or what installing `certutil` would take (D4.2).
2. axe-core injection cost per page, and its runtime under Chromium concurrency limits (D3).
3. The token and latency budget for D1.4 LLM solver cells and D7 LLM actors, measured with the local Qwen3 set-up
   in RL_INTEGRATION.md.
