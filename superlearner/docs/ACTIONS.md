# Open-ended action generation

The learner is not a controller that picks one of "build website / start business / buy compute". What it can do
is defined by the world's primitive operations as they are *at that moment*, and it composes them itself.

## What an action is

An action is a **program** over primitive operations (`actions/program.py`), small JSON so that the frozen LLM can
write it, the controller can build and edit it, and the agent can store it as a tool:

```
{"plan": "free text", "steps": [
  {"call": "tool.name", "args": {...}, "as": "x"},               tool call, typed arguments, result bound to x
  {"wait": 3600},
  {"if": COND, "then": [...], "else": [...]},                    conditional branch
  {"repeat": 3, "do": [...]}, {"while": COND, "do": [...]}, {"foreach": "$x.items", "as": "it", "do": [...]}
  {"code": "def main(input, caller): ...", "input": ..., "as": "y"}   the agent's own code, run in the world's sandbox
  {"use": "tool_the_agent_made", "args": {...}},
  {"define_tool": {"name", "description", "params", "steps" | "code"}},
  {"subgoal": "free text"}, {"note": "..."}, {"stop": "reason"} ]}
```

Values may refer to earlier results (`"$x.account"`), compute (`{"expr": ["mul", "$x.balance", 0.5]}`) or ask the
language model (`{"gen": "what to write"}`). The interpreter enforces an operation budget and bounded loops; a
failed call stops the program and the trace says where and why. Every primitive operation goes through the grounded
executor: validated, recorded, timed.

## The two generators

**A. The frozen LLM** writes free-form plans and programs from the goal, the situation, the retrieved tool
documentation and recent outcomes (`propose`), and writes argument content the controller asks for (`gen`).

**B. The trainable controller** generates actions itself, one *decision* at a time. A decision is a head and its
arguments (`cam/agent.py`):

| Head | Meaning |
|---|---|
| any tool the actor can call now | a primitive world operation: 136–191 built-in tools, plus services the actor discovered, plus API endpoints of other organisations |
| `wait d` | let simulated time pass (10 min … 3 days) |
| `propose v` | ask the frozen LLM for a draft program (variant v); costs LLM compute; budgeted per episode |
| `draft_run` / `draft_skip` | run the whole remaining draft as one decision, or delete its next step |
| `define k` | turn the last k successful calls into a reusable, parameterised tool |
| `use t` | call a tool the agent made |

For a tool head the controller then chooses one **option per argument** (`cam/features.py build_slots`):

| Option source | Example |
|---|---|
| identifiers the actor has seen under that key, or recently | `account = acct_000018` |
| identifiers the actor created or controls (`mine`) | `owner = o:agent-co` |
| schema values | enum values, booleans, a log-spaced number grid |
| numbers the actor has seen, and simple multiples of them | `amount = 0.7 × balance`, `price = 0.9 × rival's price` |
| a fresh handle | new organisation / repository / host names |
| what the current draft suggests | follow the LLM's step, or override one argument of it |
| what worked before (this episode or a remembered one) | literal values only |
| **let the frozen LLM write this value from this document** | `files = gen(the format document)`, `* = gen(the API description)` |

So the controller can: build a sequence of calls from nothing; follow a draft step exactly; follow it with some
arguments replaced (edit); insert its own steps between draft steps; skip draft steps (delete); run the rest of a
draft in one go; decide what the LLM writes code *about* by pointing at a document; and package what it did into
a tool. The distribution over all of this is the trained network's, and reinforcement learning changes it.

Nothing in the head or option sets names a strategy. Scripted teachers (`train/demos.py`) and the uniform-random
control act through the same interface.

## Why a learned policy over this space is possible

* **Items, not units.** Heads and options are embedded from text embeddings, hashed tokens and numeric features;
  the output layer does not grow with the world.
* **Factored choices with exact likelihoods.** log π(action) = log π(head) + Σ log π(optionⱼ | head, earlier
  options); candidate sets are masked before the softmax; the decision record stores the items and choices, and the
  likelihood is recomputed exactly during training.
* **Temporal abstraction.** Waits, drafts and agent tools make one decision cover many primitive operations, so
  episodes are tens of decisions, not thousands.
* **Credit for LLM-translated decisions.** The controller's decision to ask for LLM-written content, and what to
  base it on, is an ordinary choice with a log-probability; its consequence (the code works or does not, the
  deployment is accepted or rejected) arrives as reward and as the next observation. The frozen model's own
  sampling is not credited or trained.

## Tools made by the agent

`define k` generalises the agent's last k successful calls (`macro_from_trace`): a value produced by an earlier
step becomes a reference to that step, an identifier that existed before becomes a parameter, the rest stays
literal. Programs can also define tools explicitly, including code tools. A tool made in one world runs in another
(tested: a "found a company" macro made in a tiny world creates a registered company in a bazaar world).
The library records source, parameters, provenance, dependencies, hosts touched, version and real outcome counts.
Defining a tool earns nothing; a tool counts as verified only after it has succeeded in a later episode.

## Measured limits

See [RESULTS.md](RESULTS.md) for how often each head was actually used after training, and
[LIMITATIONS.md](LIMITATIONS.md): the controller's use of `propose`, `define` and `use` after training is the
honest test of whether these pathways matter, and it is reported there.
