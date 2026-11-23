# Crucible

A persistent, event-sourced synthetic digital civilization for training and evaluating continually learning
agents. One authoritative world state is shared by 17 interconnected services (identity, bank, mail, forum,
search, docs, code hosting, issues, calendar, support, ads, market, cloud hosting, company registry, contracts,
jobs, market research), a vectorised consumer population that buys verified work from deployed software, an NPC
society of firms, workers and lenders, and any number of agents. Agents can write and run real code in a sandbox,
deploy it as services other agents and consumers use, form companies, hire, contract, borrow and compete. A
second world (the engineering laboratory) shows the kernel is not specific to economics.

* Architecture: [ARCHITECTURE.md](ARCHITECTURE.md) · status ledger: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
* Docs: [kernel](docs/WORLD_KERNEL.md) · [synthetic internet](docs/SYNTHETIC_INTERNET.md) · [economy](docs/ECONOMY.md) ·
  [actors](docs/ACTOR_SYSTEM.md) · [agent API](docs/AGENT_API.md) · [world generation](docs/WORLD_GENERATION.md) ·
  [self-play](docs/SELF_PLAY.md) · [RL integration](docs/RL_INTEGRATION.md) · [evaluation & goals](docs/EVALUATION.md) ·
  [scaling](docs/SCALING.md) · [security](docs/SECURITY.md) · [research questions](docs/RESEARCH_QUESTIONS.md) ·
  [experiments](docs/EXPERIMENTS.md) · [limitations](docs/LIMITATIONS.md) · [related work](docs/RELATED_WORK.md)


## Crucible 4.0 "OpenWorld"

A platform for measuring how fast learning systems acquire new, transferable capabilities: hidden causal worlds from three
independent engines, open economies whose value is established by executing code, autonomous firms and founders, self-play,
learning-curve and transfer protocols, a sealed test set and an adversarial red team.

**Current finding: all 16 acceptance tests pass, and no learner tested so far beats a scripted experimenter on unseen worlds.**
Read [docs/OPENWORLD_REPORT.md](docs/OPENWORLD_REPORT.md) first, then [docs/REPRODUCE_OPENWORLD.md](docs/REPRODUCE_OPENWORLD.md).

```bash
cd crucible && python -m pytest -q --ignore=tests/test_acceptance.py   # regression suite
python examples/run_acceptance.py                                      # acceptance tests A-P
python examples/ow_benchmark.py list                                   # learners, families and sealed sets
```

## Install

```bash
cd crucible
pip install -e ".[web,browser,postgres,rl,dev]"     # or: uv pip install -e ...
playwright install chromium                          # only for browser mode
```

Requirements: Python ≥ 3.11, Linux with Landlock + seccomp for running agent code (the sandbox refuses to run
untrusted code otherwise), root to drop privileges to `nobody`. PostgreSQL binaries only for the Postgres tests.

## Quickstart (Python)

```python
import crucible; crucible.load_all_modules()
from crucible.core.kernel import World
from crucible.worldgen import generate

w = World.create(generate("bazaar", seed=1))                   # a generated civilization
me = w.system_call("admin.join", {"handle": "alice", "endowment": 1_000_000})["principal"]
print([t["name"] for t in w.available_tools(me)][:10])          # discoverable tools
print(w.execute(me, {"tool": "market.browse", "args": {"kind": "api", "sort": "price"}}).to_dict())
w.advance_time(w.now + 86400)                                   # a simulated day: demand, billing, NPCs, shocks
snap = w.snapshot_blob(); fork = w.fork("experiment-a")         # isolated counterfactual copy
```

Agents: `crucible.agents.base.Runner` drives any `Policy` (scripted controls in `agents/scripted.py`, LLM in
`agents/llm.py`, RL in `agents/rl.py`). Goals: `crucible.goals.contracts.generate_goal(...)`,
`crucible.evaluation.harness.run_goal(goal, policy)`.

## Command line

`crucible --help` lists: `world create|list|inspect|run|snapshot|fork|replay|verify`, `actor join|run`,
`experiment run`, `selfplay run`, `evaluate`, `trace`, `report`, `dashboard`, `serve`. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the exact reproduction commands of every reported result.

## Tests

```bash
cd crucible && python -m pytest -q             # unit, property, integration, sandbox, Postgres, browser tests
```

## Status

This is a research platform. It is not evidence that a superlearner exists: the first experiments establish that
the world is consistent, solvable in several ways, resistant to the obvious reward exploits, reproducible and fast
enough for experimentation — and they report learner failures honestly (docs/EXPERIMENTS.md).


## Open world (branch `feat/crucible-open-world`)

Open families (`open_digital`, `open_industry`, `open_mixed`, `open_scarce_energy`, `open_fragile_suppliers`,
held-out `open_weak_institutions`, `open_perishable`) replace "eight problem formats" as the boundary of value with
outcome-based demand: firms run by Tier-2 minds measure their own books, publish problems, and adopt software or
process knowledge invented by agents only when randomised trials on their own operations show it pays. Agents can
also negotiate binding agreements with simulated people, constitute organisations with enforced governance, and
build in a production economy with hidden physics, energy, freight and household demand over item attributes.

```bash
cd crucible && python -m pytest -q tests/test_open_*.py
PYTHONPATH=crucible/src python crucible/examples/demo_open_invention.py 1 40      # demonstration 1
PYTHONPATH=crucible/src python crucible/examples/demo_open_techchange.py 1 45     # demonstration 2
PYTHONPATH=crucible/src python crucible/examples/demo_open_negotiation.py 1       # demonstration 3
PYTHONPATH=crucible/src python crucible/examples/demo_open_supplychain.py 1 40    # demonstration 4
PYTHONPATH=crucible/src python crucible/examples/demo_open_civilization.py 1 90   # demonstration 5
PYTHONPATH=crucible/src python crucible/examples/open_ablations.py 30 1 2
```

Read `ARCHITECTURE.md` (open-world layer), `docs/OPEN_ENDEDNESS_AUDIT.md`, `docs/EXPERIMENTS.md` and
`IMPLEMENTATION_STATUS.md` (open-world ledger and acceptance criteria).
