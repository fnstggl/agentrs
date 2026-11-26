"""S04 / spec Part 2.4: the seven semantic acceptance tests, through the integrated CamUnifiedModel (real tokenizer,
entity encoder, backbone, reasoner, beliefs, options, ActionGenerator with Qwen3-0.6B + LoRA + bridge) - slow; run with

    python -m pytest -q -s -m slow superlearner/tests/cam_unified/semantic/test_semantic_acceptance.py

Protocol (identical for all seven): a family of single-decision tasks with **randomised entity identifiers, tool names
(fresh random names in every episode, so a name never carries information) and wording**; the model is trained by
supervised imitation of the correct action (-log pi(a* | context), `demo_init.demo_log_prob`: gradients reach the
core and the decoder path) on a training split and evaluated by greedy decoding on a held-out split whose semantic
content is new (unseen tool functions, entity types, documents' wording, website format, error types or report
wording). Every test has a control in which the semantically meaningful content is removed or scrambled while names
and structure stay: understanding (not memorised names or positions) is shown by held-out accuracy well above chance
*and* above the control. Each test records its numbers in `benchmarks/s04_semantic_acceptance.json`."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, replace
from typing import Optional

import pytest
import torch

from superlearner.cam_unified.interfaces import Action, GoalSpec, Observation, ObsKind, ToolSchema
from superlearner.cam_unified.semantic import demo_init as DI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "benchmarks", "s04_semantic_acceptance.json")
_SYL_C, _SYL_V = "bdfgklmnprstvz", "aeiou"


def rname(rng: random.Random, prefix: str = "") -> str:
    return prefix + "".join(rng.choice(_SYL_C) + rng.choice(_SYL_V) for _ in range(3)) + f"_{rng.randrange(10, 99)}"


def rid(rng: random.Random, prefix: str = "e") -> str:
    return f"{prefix}{rng.randrange(16 ** 5):05x}"


@dataclass
class Ep:
    obs: list
    tools: list
    goal: GoalSpec
    action: Action
    evidence: Optional[list] = None       # explicit evidence labels (else derived from the argument values)
    meta: Optional[dict] = None


def _record(name: str, rec: dict) -> None:
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d[name] = rec
    with open(OUT, "w") as fh:
        json.dump(d, fh, indent=1)


@pytest.fixture(scope="module")
def system(enc, cfg):
    """A factory of fresh integrated models (shared frozen Qwen; every CAM part newly initialised, the shared LoRA
    reset to its initial values for every model and restored afterwards)."""
    from superlearner.cam_unified.model import build_model
    lora0 = enc.trainable_state_dict()

    def make(seed: int = 0):
        enc.load_trainable_state_dict(lora0)
        torch.manual_seed(seed)
        c = replace(cfg, use_imagination=False, max_ctx_tokens=128)
        return build_model(c, components={"semantic": enc}).to(enc.device)
    yield make
    enc.load_trainable_state_dict(lora0)


def prepare(model, eps: list, seed: int = 0, batch: int = 16) -> list:
    from superlearner.cam_unified.model import BudgetState
    out = []
    for c0 in range(0, len(eps), batch):
        ch = eps[c0: c0 + batch]
        n = len(ch)
        state = model.init_state(n, seed=seed + c0)
        dec = DI.teacher_forced_decide(model, state, [list(e.obs) for e in ch], [1.0] * n, [list(e.tools) for e in ch],
                                       [e.goal for e in ch], [e.action for e in ch],
                                       BudgetState.make(n, device=model.device))
        for j, e in enumerate(ch):
            out.append(DI.ForcedStep(dec.inputs.select([j]), e.action, e.evidence, "s04", c0 + j, False))
    need = [s for s in out if s.evidence is None]
    DI._attach_evidence(model, need, model.cfg.max_evidence)
    for s in out:
        assert s.unrepresentable is None
    return out


def train(model, eps: list, steps: int = 120, batch: int = 16, lr: float = 1e-3, lr_sem: float = 3e-4,
          seed: int = 0) -> list:
    """Supervised imitation of the correct actions (the inputs are re-encoded every 40 updates so the stored initial
    latent state follows the trained weights)."""
    opt = DI.make_optimizer(model, DI.BCConfig(lr=lr, lr_sem=lr_sem))
    rng = random.Random(seed)
    data = prepare(model, eps, seed)
    hist = []
    for k in range(steps):
        if k and k % 40 == 0:
            data = prepare(model, eps, seed)
        b = rng.sample(data, min(batch, len(data)))
        model.train()
        parts, _ = DI.demo_log_prob(model, DI._stack_inputs(b), [s.action for s in b], DI._evidence_arg(b))
        loss = -sum(parts.values()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        model.action_generator.enc.mark_updated()
        model.bump_weights_epoch()
        hist.append(float(loss.detach()))
    return hist


def greedy(model, eps: list, seed: int = 1000) -> list:
    return DI.greedy_actions(model, prepare(model, eps, seed), batch=16)


def run_task(system, make_ep, n_train: int, n_test: int, score, control=None, steps: int = 120, seed: int = 0,
             name: str = "", before: bool = True) -> dict:
    rng_tr, rng_te = random.Random(seed), random.Random(seed + 10_000)
    tr = [make_ep(rng_tr, "train") for _ in range(n_train)]
    te = [make_ep(rng_te, "test") for _ in range(n_test)]
    model = system(seed)
    res = {}
    if before:
        res["before"] = sum(score(e, a) for e, a in zip(te, greedy(model, te))) / len(te)
    hist = train(model, tr, steps, seed=seed)
    res["loss_first_last"] = [hist[0], sum(hist[-10:]) / 10]
    acts = greedy(model, te)
    res["held_out"] = sum(score(e, a) for e, a in zip(te, acts)) / len(te)
    res["train_fit"] = sum(score(e, a) for e, a in zip(tr[:32], greedy(model, tr[:32]))) / min(32, len(tr))
    if control is not None:
        ce = [control(e, random.Random(i)) for i, e in enumerate(te)]
        res["control"] = sum(score(e, a) for e, a in zip(ce, greedy(model, ce))) / len(ce)
    res["examples"] = [{"goal": e.goal.text[:120], "want": e.action.to_dict()["args"] | {"tool": e.action.tool},
                        "got": a.args | {"tool": a.tool}} for e, a in list(zip(te, acts))[:4]]
    res["n_train"], res["n_test"], res["steps"] = n_train, n_test, steps
    print(name, json.dumps({k: v for k, v in res.items() if k != "examples"}))
    _record(name, res)
    del model
    torch.cuda.empty_cache()
    return res


def same_tool(e: Ep, a: Action) -> float:
    return float(a.tool == e.action.tool)


def same_action(e: Ep, a: Action) -> float:
    return float(a.tool == e.action.tool and json.dumps(a.args, sort_keys=True) ==
                 json.dumps(e.action.args, sort_keys=True))


# ---------------------------------------------------------------------------- 1. unseen tool schemas
FUNCS = [  # (function descriptions (2 paraphrases), requests (2 paraphrases), argument schema properties)
    (["Returns the weather forecast for a city.", "Gives tomorrow's temperature and rain for a town."],
     ["Will it rain in {x} tomorrow?", "What is the weather going to be in {x}?"], {"city": "string"}),
    (["Converts an amount of money between two currencies.", "Exchanges a sum from one currency into another."],
     ["How many euros do I get for {n} dollars?", "Convert {n} pounds to yen."], {"amount": "number"}),
    (["Translates a text into another language.", "Renders a sentence in a different language."],
     ["Say '{x}' in Spanish.", "How do you write '{x}' in German?"], {"text": "string"}),
    (["Sends an email message to a recipient.", "Delivers a written message to someone's inbox."],
     ["Email {x} that the meeting moved.", "Write to {x} about the delayed order."], {"to": "string"}),
    (["Looks up the current share price of a listed company.", "Quotes the stock market value of a firm."],
     ["What is {x} trading at right now?", "Give me the stock price of {x}."], {"ticker": "string"}),
    (["Schedules a meeting in the calendar.", "Books an appointment slot in the agenda."],
     ["Set up a call with {x} next Tuesday.", "Book a meeting with {x} for Friday."], {"with": "string"}),
    (["Computes driving directions between two places.", "Plans a road route from one location to another."],
     ["How do I drive from {x} to the airport?", "Route me by car to {x}."], {"destination": "string"}),
    (["Gives the definition of an English word.", "Explains what a word means."],
     ["What does '{x}' mean?", "Define the word '{x}'."], {"word": "string"}),
    (["Pays an outstanding invoice from a bank account.", "Settles a bill by transferring the money owed."],
     ["Settle the invoice from {x}.", "Pay what we owe to {x}."], {"invoice": "string"}),
    (["Searches the web for documents on a topic.", "Finds pages on the internet about a subject."],
     ["Find articles about {x}.", "Look up information on {x} online."], {"query": "string"}),
    (["Sets an alarm or countdown timer.", "Starts a clock that rings after a delay."],
     ["Wake me up in {n} minutes.", "Remind me with a beep in {n} minutes."], {"minutes": "integer"}),
    (["Plays a song by an artist.", "Starts music playback of a track."],
     ["Put on something by {x}.", "I want to hear a song from {x}."], {"artist": "string"}),
]
WORDS = ["Lyon", "Oslo", "Kenji", "Amara", "Acme", "sunrise", "harbour", "Delta Corp", "Lima", "violin", "Quito",
         "Nadia", "Tromso", "Basel", "Ruth", "glacier"]


def _tool(name: str, desc: str, props: dict) -> ToolSchema:
    return ToolSchema(name, desc, {"type": "object", "properties": {k: {"type": v, **({"maxLength": 40} if v == "string"
                                                                                       else {"minimum": 0,
                                                                                             "maximum": 1000})}
                                                                    for k, v in props.items()},
                                   "required": [], "additionalProperties": False})


def ep_unseen_tool(rng: random.Random, split: str, k: int = 5) -> Ep:
    pool = list(range(8)) if split == "train" else list(range(8, 12))
    tgt = rng.choice(pool)
    others = rng.sample([i for i in range(12) if i != tgt], k - 1)
    p = 0 if split == "train" else 1                       # held-out: unseen tools, described in other words
    tools = []
    for i in [tgt] + others:
        tools.append(_tool(rname(rng), FUNCS[i][0][p if i == tgt else rng.randrange(2)], FUNCS[i][2]))
    rng.shuffle(tools)
    req = FUNCS[tgt][1][rng.randrange(2)].format(x=rng.choice(WORDS), n=rng.randrange(2, 99))
    name = next(t.name for t, i in zip(tools, range(99)) if t.description in FUNCS[tgt][0])
    return Ep([Observation(ObsKind.MESSAGE, req, t=1.0, source=rid(rng, "p:"))], tools,
              GoalSpec(rid(rng, "g"), "Handle the user's latest request with the right tool."), Action(name, {}))


def ctl_unseen_tool(e: Ep, rng: random.Random) -> Ep:
    """Control: the same episode with every tool's description and argument schema blanked (names only)."""
    tools = [ToolSchema(t.name, "A tool.", {"type": "object", "properties": {}, "additionalProperties": False})
             for t in e.tools]
    return Ep(e.obs, tools, e.goal, e.action)


@pytest.mark.slow
def test_s04_1_interprets_an_unseen_tool_schema(system):
    r = run_task(system, ep_unseen_tool, 256, 64, same_tool, ctl_unseen_tool, steps=150, name="1_unseen_tool_schema")
    # 5 tools per decision (chance 0.2); held-out tools are of 4 functions never seen in training
    assert r["held_out"] >= 0.45 and r["held_out"] >= r["control"] + 0.2 and r["held_out"] > r["before"] + 0.2


# ---------------------------------------------------------------------------- 2. objective -> entities and actions
KINDS = [("bakery", "bakes bread and croissants every morning"), ("law firm", "gives legal advice on contracts"),
         ("garage", "repairs cars and changes tyres"), ("florist", "sells roses and tulips"),
         ("clinic", "treats patients and runs blood tests"), ("print shop", "prints posters and flyers"),
         ("school", "teaches children to read and write"), ("hotel", "rents rooms to travellers"),
         ("pharmacy", "sells medicines on prescription"), ("brewery", "brews beer and ales"),
         ("tailor", "alters suits and dresses"), ("veterinary practice", "treats sick dogs and cats")]


def ep_entities(rng: random.Random, split: str, k: int = 4) -> Ep:
    pool = list(range(8)) if split == "train" else list(range(8, 12))
    tgt = rng.choice(pool)
    others = rng.sample([i for i in range(12) if i != tgt], k - 1)
    ids = [rid(rng, "o:") for _ in range(k)]
    order = [tgt] + others
    obs = [Observation(ObsKind.TEXT, f"Directory entry {ids[j]}: this business {KINDS[i][1]}.", entity=ids[j],
                       entity_type="organisation", t=1.0, source="directory") for j, i in enumerate(order)]
    perm = list(range(k))
    rng.shuffle(perm)
    obs = [obs[j] for j in perm]
    pay = ToolSchema(rname(rng), "Pay an organisation the amount it invoiced.",
                     {"type": "object", "properties": {"to": {"type": "string", "maxLength": 40}}, "required": ["to"],
                      "additionalProperties": False})
    look = ToolSchema(rname(rng), "Show the opening hours of the town hall.",
                      {"type": "object", "properties": {}, "additionalProperties": False})
    goal = GoalSpec(rid(rng, "g"), f"Pay the invoice of the {KINDS[tgt][0]}.")
    return Ep(obs, [pay, look], goal, Action(pay.name, {"to": ids[0]}))


def ctl_entities(e: Ep, rng: random.Random) -> Ep:
    """Control: the descriptions are shuffled across the entities (ids, structure and goal unchanged)."""
    texts = [o.content.split(": ", 1)[1] for o in e.obs]
    rng.shuffle(texts)
    obs = [replace_content(o, o.content.split(": ", 1)[0] + ": " + t) for o, t in zip(e.obs, texts)]
    return Ep(obs, e.tools, e.goal, e.action)


def replace_content(o: Observation, content: str) -> Observation:
    d = o.to_dict()
    d["content"] = content
    return Observation.from_dict(d)


@pytest.mark.slow
def test_s04_2_associates_an_objective_with_entities_and_actions(system):
    r = run_task(system, ep_entities, 256, 64, same_action, ctl_entities, steps=150, name="2_objective_entities")
    # 4 candidate entities (chance 0.25 for the right id); held-out business kinds never seen in training
    assert r["held_out"] >= 0.5 and r["held_out"] >= r["control"] + 0.2


# ---------------------------------------------------------------------------- 3. unseen mechanism document
DOCS_TRAIN = ["Operating notes for the {p}: opening the {a} valve raises the {q}; opening the {b} valve lowers it.",
              "{p} manual. The {a} valve increases the {q} when opened, the {b} valve decreases it.",
              "Engineer's memo ({p}): to push the {q} up use the {a} valve, to bring it down use the {b} valve."]
DOCS_TEST = ["Handbook of the {p}. If you open the {b} valve the {q} falls, whereas the {a} valve makes it climb.",
             "Shift log, {p}: {q} goes down after the {b} valve is opened and goes up after the {a} valve is opened."]
PLANTS = ["boiler room", "brewery", "paper mill", "greenhouse", "laundry", "bakery oven", "ice rink", "fish farm"]
QUANT = ["pressure", "temperature", "flow rate", "humidity", "water level", "steam output"]
COLORS = ["red", "blue", "green", "amber", "violet", "black", "white", "orange", "silver", "copper", "teal", "pink"]


def ep_document(rng: random.Random, split: str) -> Ep:
    a, b = rng.sample(COLORS[:8] if split == "train" else COLORS[8:], 2)
    doc = rng.choice(DOCS_TRAIN if split == "train" else DOCS_TEST)
    q = rng.choice(QUANT[:4] if split == "train" else QUANT[4:])
    p = rng.choice(PLANTS)
    up = rng.random() < 0.5
    ta = ToolSchema(rname(rng), f"Open the {a} valve.", {"type": "object", "properties": {}})
    tb = ToolSchema(rname(rng), f"Open the {b} valve.", {"type": "object", "properties": {}})
    tools = [ta, tb] if rng.random() < 0.5 else [tb, ta]
    obs = [Observation(ObsKind.FILE, doc.format(p=p, a=a, b=b, q=q), t=1.0, source="docs/" + rname(rng) + ".txt")]
    goal = GoalSpec(rid(rng, "g"), f"{'Raise' if up else 'Lower'} the {q} of the {p}.")
    return Ep(obs, tools, goal, Action(ta.name if up else tb.name, {}))


def ctl_document(e: Ep, rng: random.Random) -> Ep:
    """Control: the document is replaced by an unrelated one (same kind, same valve names mentioned)."""
    words = [w for t in e.tools for w in t.description.split() if w not in ("Open", "the", "valve.")]
    d = replace_content(e.obs[0], f"Visitor notes: the {words[0]} and {words[1]} valves were painted last spring.")
    return Ep([d], e.tools, e.goal, e.action)


@pytest.mark.slow
def test_s04_3_understands_an_unseen_document_describing_a_mechanism(system):
    r = run_task(system, ep_document, 256, 64, same_tool, ctl_document, steps=150, name="3_unseen_document")
    # 2 valves (chance 0.5); held-out documents use unseen wording, valve names and quantities
    assert r["held_out"] >= 0.75 and r["held_out"] >= r["control"] + 0.15


# ---------------------------------------------------------------------------- 4. executable code for a novel task
CODE_TASKS = [
    (["Write a Python program that prints the sum of {a}, {b} and {c}.", "Print the total of {a} plus {b} plus {c}."],
     lambda a, b, c, w: f"print({a} + {b} + {c})", lambda a, b, c, w: str(a + b + c)),
    (["Write a Python program that prints the word {w} in capital letters.", "Output {w} in upper case."],
     lambda a, b, c, w: f"print('{w}'.upper())", lambda a, b, c, w: w.upper()),
    (["Write a Python program that prints how many letters the word {w} has.", "Print the length of {w}."],
     lambda a, b, c, w: f"print(len('{w}'))", lambda a, b, c, w: str(len(w))),
    (["Write a Python program that prints the word {w} backwards.", "Print {w} reversed."],
     lambda a, b, c, w: f"print('{w}'[::-1])", lambda a, b, c, w: w[::-1]),
    (["Write a Python program that prints the product of {a} and {b}.", "Print {a} multiplied by {b}."],
     lambda a, b, c, w: f"print({a} * {b})", lambda a, b, c, w: str(a * b)),
]
CODE_WORDS_TR = ["apple", "river", "candle", "mountain", "pencil", "garden", "window", "silver", "rocket", "violin"]
CODE_WORDS_TE = ["harbor", "thunder", "lantern", "meadow", "compass", "falcon"]


def ep_code(rng: random.Random, split: str) -> Ep:
    k = rng.randrange(len(CODE_TASKS))
    texts, sol, out = CODE_TASKS[k]
    a, b, c = rng.randrange(10, 99), rng.randrange(10, 99), rng.randrange(10, 99)
    w = rng.choice(CODE_WORDS_TR if split == "train" else CODE_WORDS_TE)
    text = texts[0 if split == "train" else 1].format(a=a, b=b, c=c, w=w)
    tool = ToolSchema(rname(rng), "Run a Python program and return what it prints.",
                      {"type": "object", "properties": {"code": {"type": "string", "maxLength": 2000}},
                       "required": ["code"], "additionalProperties": False})
    obs = [Observation(ObsKind.TEXT, "Reminder: the office closes at six today.", t=0.5, source="board"),
           Observation(ObsKind.MESSAGE, f"Task: {text}", t=1.0, source=rid(rng, "p:"))]
    return Ep(obs, [tool], GoalSpec(rid(rng, "g"), "Solve the programming task you were sent."),
              Action(tool.name, {"code": sol(a, b, c, w)}), evidence=[1], meta={"expected": out(a, b, c, w)})


def ctl_code(e: Ep, rng: random.Random) -> Ep:
    obs = [e.obs[0], replace_content(e.obs[1], "Task: (the attachment could not be opened)")]
    return Ep(obs, e.tools, e.goal, e.action, e.evidence, e.meta)


_SBX = None


def runs_correctly(e: Ep, a: Action) -> float:
    global _SBX
    from superlearner.cam_unified.semantic.sandbox import CodeSandbox
    if a.tool != e.action.tool or not isinstance(a.args.get("code"), str):
        return 0.0
    _SBX = _SBX or CodeSandbox()
    res = _SBX.run(code=a.args["code"])
    out = " ".join(o.content for o in res.observations if o.kind is ObsKind.EXEC_RESULT)
    return float(res.ok and e.meta["expected"] in out.split())


@pytest.mark.slow
def test_s04_4_generates_executable_code_for_a_novel_task(system):
    from superlearner.cam_unified.semantic.sandbox import crucible_sandbox_capabilities
    if not crucible_sandbox_capabilities().get("available"):
        pytest.skip("sandbox unavailable")
    r = run_task(system, ep_code, 192, 40, runs_correctly, ctl_code, steps=120, name="4_novel_code")
    # held-out: unseen words and numbers, every task phrased differently from training; executed in the sandbox
    assert r["held_out"] >= 0.5 and r["held_out"] >= r["control"] + 0.3


# ---------------------------------------------------------------------------- 5. unfamiliar website
PRODUCTS_TR = ["red kettle", "blue umbrella", "garden hose", "desk lamp", "wool scarf", "coffee grinder", "yoga mat",
               "bike lock"]
PRODUCTS_TE = ["paper shredder", "rain boots", "travel pillow", "chess set"]


def ep_website(rng: random.Random, split: str, k: int = 4) -> Ep:
    prods = rng.sample(PRODUCTS_TR if split == "train" else PRODUCTS_TE + PRODUCTS_TR, k)
    if split == "test" and not any(p in PRODUCTS_TE for p in prods):
        prods[0] = rng.choice(PRODUCTS_TE)
    ids = [rid(rng, "el-") for _ in prods]
    tgt = rng.randrange(k) if split == "train" else next(i for i, p in enumerate(prods) if p in PRODUCTS_TE)
    if split == "train":                       # an accessibility tree
        page = "\n".join([f"document 'Shop {rname(rng)}'"] + [f"  button id={i} name='Add {p} to basket'"
                                                              for i, p in zip(ids, prods)])
        obs = Observation(ObsKind.A11Y, page, t=1.0, source="browser")
    else:                                      # an unfamiliar site: raw HTML, other wording
        page = "<html><body><ul>" + "".join(f'<li><a href="#" id="{i}">Put the {p} in my cart</a></li>'
                                            for i, p in zip(ids, prods)) + "</ul></body></html>"
        obs = Observation(ObsKind.HTML, page, t=1.0, source="browser")
    click = ToolSchema(rname(rng), "Click a page element given its id.",
                       {"type": "object", "properties": {"element": {"type": "string", "maxLength": 40}},
                        "required": ["element"], "additionalProperties": False})
    back = ToolSchema(rname(rng), "Go back to the previous page.", {"type": "object", "properties": {}})
    goal = GoalSpec(rid(rng, "g"), f"Add the {prods[tgt]} to the shopping basket.")
    return Ep([obs], [click, back], goal, Action(click.name, {"element": ids[tgt]}))


def ctl_website(e: Ep, rng: random.Random) -> Ep:
    """Control: the product names on the page are replaced by 'item' (ids and structure unchanged)."""
    c = e.obs[0].content
    for p in PRODUCTS_TR + PRODUCTS_TE:
        c = c.replace(p, "item")
    return Ep([replace_content(e.obs[0], c)], e.tools, e.goal, e.action)


@pytest.mark.slow
def test_s04_5_uses_an_unfamiliar_website(system):
    r = run_task(system, ep_website, 256, 64, same_action, ctl_website, steps=150, name="5_unfamiliar_website")
    # trained on accessibility trees; tested on HTML pages with other wording and unseen products (chance 0.25)
    assert r["held_out"] >= 0.5 and r["held_out"] >= r["control"] + 0.2


# ---------------------------------------------------------------------------- 6. failures <-> execution feedback
ERRORS_TR = [("ZeroDivisionError", "division by zero"), ("KeyError", "'total'"),
             ("IndexError", "list index out of range")]
ERRORS_TE = [("TypeError", "unsupported operand type(s) for +: 'int' and 'str'"),
             ("AttributeError", "'NoneType' object has no attribute 'items'")]


def ep_failure(rng: random.Random, split: str, k: int = 3) -> Ep:
    fns = [rname(rng).split("_")[0] for _ in range(k)]
    paths = [f"src/{rname(rng).split('_')[0]}.py" for _ in range(k)]
    bad = rng.randrange(k)
    obs = [Observation(ObsKind.CODE, f"def {f}(x):\n    return compute(x)\n", t=0.5, source=p)
           for f, p in zip(fns, paths)]
    et, msg = rng.choice(ERRORS_TR if split == "train" else ERRORS_TE)
    tb = (f"Traceback (most recent call last):\n  File \"tests/test_main.py\", line {rng.randrange(5, 60)}, in "
          f"test_run\n  in {fns[bad]}\n{et}: {msg}")
    obs.append(Observation(ObsKind.ERROR, tb, t=1.0, source="pytest"))
    open_f = ToolSchema(rname(rng), "Open a source file in the editor.",
                        {"type": "object", "properties": {"path": {"type": "string", "maxLength": 60}},
                         "required": ["path"], "additionalProperties": False})
    rerun = ToolSchema(rname(rng), "Run the test suite again.", {"type": "object", "properties": {}})
    goal = GoalSpec(rid(rng, "g"), "A test failed. Open the source file that contains the failing function." if
                    split == "train" else "The build is red: inspect the module whose code raised the error.")
    return Ep(obs, [open_f, rerun], goal, Action(open_f.name, {"path": paths[bad]}), evidence=[bad, k])


def ctl_failure(e: Ep, rng: random.Random) -> Ep:
    """Control: the traceback no longer names the failing function."""
    err = e.obs[-1]
    lines = [ln for ln in err.content.split("\n") if not ln.startswith("  in ")]
    return Ep(e.obs[:-1] + [replace_content(err, "\n".join(lines))], e.tools, e.goal, e.action, None)


@pytest.mark.slow
def test_s04_6_associates_software_failures_with_execution_feedback(system):
    r = run_task(system, ep_failure, 256, 64, same_action, ctl_failure, steps=150, name="6_failure_feedback")
    # 3 files (chance 1/3); held-out: unseen error types and goal wording, random function and file names
    assert r["held_out"] >= 0.6 and r["held_out"] >= r["control"] + 0.2


# ---------------------------------------------------------------------------- 7. decisions follow meaning
REPORTS_TR = ["Road report: the bridge to {i} is {s} today.", "Notice: {i} bridge {s} until tonight."]
REPORTS_TE = ["Traffic update - crossing the strait towards {i} by car is {s2} this morning.",
              "Radio: drivers heading for {i} find the bridge {s} at the moment."]
ISLANDS = ["Arran", "Mull", "Skye", "Islay", "Jura", "Bute", "Coll", "Tiree"]


def ep_report(rng: random.Random, split: str, closed: Optional[bool] = None) -> Ep:
    closed = rng.random() < 0.5 if closed is None else closed
    isl = rng.choice(ISLANDS)
    tpl = rng.choice(REPORTS_TR if split == "train" else REPORTS_TE)
    txt = tpl.format(i=isl, s="closed" if closed else "open", s2="impossible" if closed else "possible")
    drive = ToolSchema(rname(rng), "Drive the van across the bridge.", {"type": "object", "properties": {}})
    ferry = ToolSchema(rname(rng), "Take the van on the ferry.", {"type": "object", "properties": {}})
    tools = [drive, ferry] if rng.random() < 0.5 else [ferry, drive]
    obs = [Observation(ObsKind.TEXT, f"Parcel {rid(rng, 'pk')} is loaded in the van.", t=0.5, source="depot"),
           Observation(ObsKind.MESSAGE, txt, t=1.0, source="radio")]
    return Ep(obs, tools, GoalSpec(rid(rng, "g"), f"Deliver the parcel to {isl}."),
              Action(ferry.name if closed else drive.name, {}), meta={"closed": closed})


@pytest.mark.slow
def test_s04_7_decisions_change_when_a_meaningful_observation_changes(system):
    """Paired held-out episodes differ only in the meaning of one report (closed / open, unseen wording): the decision
    must follow it; a surface-only change (core.invariance SurfaceRandomizer: renamed tools and entities, reordered
    observations; no change of meaning) must not change it."""
    from superlearner.cam_unified.core.invariance import SurfaceRandomizer
    rng_tr, rng_te = random.Random(7), random.Random(70_007)
    tr = [ep_report(rng_tr, "train") for _ in range(256)]
    pairs = []
    for i in range(32):
        s = rng_te.randrange(2 ** 30)
        pairs.append((ep_report(random.Random(s), "test", True), ep_report(random.Random(s), "test", False)))
    model = system(7)
    flat = [e for p in pairs for e in p]
    before = greedy(model, flat)
    train(model, tr, 150, seed=7)
    acts = greedy(model, flat)
    right = [same_tool(e, a) for e, a in zip(flat, acts)]
    flips = sum(right[2 * i] and right[2 * i + 1] for i in range(len(pairs))) / len(pairs)
    flips_before = sum(same_tool(flat[2 * i], before[2 * i]) and same_tool(flat[2 * i + 1], before[2 * i + 1])
                       for i in range(len(pairs))) / len(pairs)
    # surface-only views of the same held-out episodes: renamed tools / entity ids, permuted observations
    sr = SurfaceRandomizer(seed=3, rename_entities=True, rename_tools=True, rescale_units=False, permute=True,
                           reword=False, presentation=False)
    views, maps = [], []
    for e in flat:
        sm = sr.make_map(e.obs, e.tools, [e.action])
        views.append(Ep(sm.apply_observations(e.obs), sm.apply_tools(e.tools), sm.apply_goal(e.goal),
                        sm.apply_action(e.action)))
        maps.append(sm)
    vacts = greedy(model, views)
    same = sum(sm.invert_action(va).tool == a.tool for sm, va, a in zip(maps, vacts, acts)) / len(flat)
    res = {"paired_flip_rate": flips, "paired_flip_rate_before": flips_before,
           "held_out_accuracy": sum(right) / len(right), "surface_view_same_decision": same,
           "n_pairs": len(pairs), "n_train": len(tr), "chance_flip_rate": 0.25}
    print("7_meaningful_change", json.dumps(res))
    _record("7_meaningful_change", res)
    assert flips >= 0.6 and same >= 0.8 and flips > flips_before + 0.25
