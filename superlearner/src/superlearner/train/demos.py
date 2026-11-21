"""Demonstrations for warm-starting: scripted controls that act through the controller's own action interface.

A teacher is a script that says, decision by decision, which head it wants and which option it wants for each
argument. `TeacherChooser` finds those among the heads and options the agent actually has at that moment, so a
demonstration is a sequence of choices the controller itself could have made, executed for real in the world.
Where the script asks for content only a language model can write (service code, an API payload), the
demonstration chooses the "let the frozen LLM write this from that document" option, and the real LLM output is
what gets committed and judged.

Provenance. Every demonstration episode records the script name, how many of its wishes had no exact option
(``misses``), the task and the seed; only episodes the evaluator verified as successful are used for cloning
(`train/trainer.py`). Demonstrations are generated on training-split tasks only.

The scripts encode ordinary developer knowledge (found a company before trading, read the format before writing
code, rent a machine before deploying). They are controls and a warm start; they are not given to the learner as
instructions, and nothing prevents the learner from departing from them.
"""
from __future__ import annotations

import re

import numpy as np

from ..cam.agent import Chooser

DAY = 86400


class Want:
    def __init__(self, fn, desc: str = ""):
        self.fn, self.desc = fn, desc

    def pick(self, options: list) -> int | None:
        return self.fn(options)


def V(x) -> Want:
    def f(opts):
        for i, o in enumerate(opts):
            if o.source not in ("gen", "omit") and o.value == x:
                return i
        return None
    return Want(f, f"value {x!r}")


def SRC(source: str, rank: int = 0) -> Want:
    def f(opts):
        hits = [i for i, o in enumerate(opts) if o.source == source]
        return hits[min(rank, len(hits) - 1)] if hits else None
    return Want(f, f"source {source}")


def TEXT() -> Want:
    """A plain short text (the neutral token), or the text that worked last time."""
    def f(opts):
        for src in ("text", "prev"):
            for i, o in enumerate(opts):
                if o.source == src and isinstance(o.value, str):
                    return i
        return None
    return Want(f, "plain text")


def NEAR(x: float) -> Want:
    def f(opts):
        best, bi = None, None
        for i, o in enumerate(opts):
            if isinstance(o.value, (int, float)) and not isinstance(o.value, bool):
                d = abs(o.value - x)
                if best is None or d < best:
                    best, bi = d, i
        return bi
    return Want(f, f"number near {x}")


def GEN(pred=None, variant: int = 0) -> Want:
    def f(opts):
        gens = [i for i, o in enumerate(opts) if o.source == "gen" and o.variant == variant]
        if pred is not None:
            for i in gens:
                if opts[i].doc is not None and pred(opts[i].doc):
                    return i
            return None
        return gens[0] if gens else None
    return Want(f, "language-model-written value")


def tool(tool_name: str, /, **wants):
    return ("tool", tool_name, {k: (w if isinstance(w, Want) else V(w)) for k, w in wants.items()})


def wait(seconds: int):
    return ("wait", int(seconds), {})


# ================================================================================================ scripts
def _res(step, default=None):
    return step.result if (step is not None and step.ok and step.result is not None) else default


def s_company(ag, st: dict, fund: bool = True):
    r = _res((yield tool("bank.accounts")), [])
    mine = [a for a in r if a.get("owner") == ag.actor]
    st["personal"], st["balance"] = (mine[0]["account"], mine[0]["balance"]) if mine else (None, 0)
    r = _res((yield tool("id.create_org", handle=SRC("fresh"), name=TEXT(), kind="company")), {})
    st["org"] = r.get("org")
    yield tool("biz.register", org=st["org"], account=st["personal"])
    if not fund:
        return
    r = _res((yield tool("bank.open_account", owner=st["org"])), {})
    st["acct"] = r.get("account")
    yield tool("bank.transfer", from_account=st["personal"], to=st["acct"], amount=NEAR(0.7 * st["balance"]))


def s_code(ag, st: dict, need: str):
    r = _res((yield tool("docs.list", scope="public")), [])
    doc = next((d["doc"] for d in r if d.get("title", "").endswith(need)), None)
    if doc is not None:
        yield tool("docs.get", doc=doc)
    kw = {"owner": st["org"]} if st.get("org") else {}
    r = _res((yield tool("code.create_repo", name=SRC("fresh"), **kw)), {})
    st["repo"] = r.get("repo")
    st["need_pred"] = lambda d: need in d.text and d.source == "docs.get"
    yield tool("code.commit", repo=st["repo"], files=GEN(st["need_pred"]))


def s_deploy(ag, st: dict):
    if not st.get("repo") or not st.get("acct"):
        return
    r = _res((yield tool("cloud.providers")), [])
    provs = sorted((p for p in r if p["cores_free"] >= 1 and not p["in_outage"]),
                   key=lambda p: (p["price_core_hour_cents"], p["provider"]))
    if not provs:
        return
    r = _res((yield tool("cloud.rent", provider=provs[0]["provider"], cores=1, account=st["acct"], owner=st["org"])), {})
    st["inst"] = r.get("instance")
    for attempt in range(3):
        kw = {"host": st["host"]} if st.get("host") else {"host": SRC("fresh")}
        out = yield tool("cloud.deploy", repo=st["repo"], instance=st["inst"], **kw)
        if out is not None and out.ok:
            st["host"] = out.args.get("host", st.get("host"))
            eps = (out.result or {}).get("endpoints") if isinstance(out.result, dict) else None
            st["endpoint"] = (eps[0] if isinstance(eps, list) and eps else None)
            return
        if out is None or out.fclass != "syntax" or not st.get("need_pred") or attempt == 2:
            return
        # the deployment was rejected because of the repository's content: write the files again (the language
        # model is shown its previous files and the rejection) and try once more
        yield tool("code.commit", repo=st["repo"], files=GEN(st["need_pred"]))


def s_list(ag, st: dict, need: str):
    if not st.get("host"):
        return
    r = _res((yield tool("market.browse", kind="api", need=need, sort="price")), [])
    prices = [x["price_cents"] for x in r if x.get("need") == need and x.get("seller") != st.get("org")]
    st["rival"] = min(prices) if prices else 5
    if not st.get("endpoint"):          # which endpoints does my deployment actually expose?
        r = _res((yield tool("cloud.deployments")), [])
        mine = [d for d in r if d.get("host") == st["host"]]
        st["endpoint"] = (mine[0].get("endpoints") or ["solve"])[0] if mine else "solve"
    yield tool("market.list", kind="api", price_cents=NEAR(st["rival"]), title=TEXT(), seller=st["org"],
               host=st["host"], endpoint=st["endpoint"], need=need)


def _pick_need(ag, rng, r) -> str:
    needs = sorted({x["need"] for x in r if x.get("need")})
    return str(needs[int(rng.integers(0, len(needs)))]) if needs else "sort.records"


def _prime(ag, st: dict) -> None:
    """Identifiers created by a scripted start state (they are in the actor's own history)."""
    for h in ag.info.get("start_history", []):
        res = (h.get("result") or {})
        r = res.get("result") if res.get("ok") else None
        t = h["tool"]
        if not isinstance(r, dict):
            if t == "bank.accounts" and isinstance(r, list):
                mine = [a for a in r if a.get("owner") == ag.actor]
                if mine:
                    st["personal"], st["balance"] = mine[0]["account"], mine[0]["balance"]
            continue
        if t == "id.create_org":
            st["org"] = r.get("org")
        elif t == "bank.open_account":
            st["acct"] = r.get("account")
        elif t == "code.create_repo":
            st["repo"] = r.get("repo")
        elif t == "cloud.rent":
            st["inst"] = r.get("instance")
        elif t == "cloud.deploy":
            st["host"] = h["args"].get("host")
        elif t == "market.list":
            st["listing"] = r.get("listing")
            st["need"] = h["args"].get("need")
        if t == "code.commit" and "need" not in st:
            st["need"] = None


def script_build(ag, rng, upto: str):
    """found < fund < code < deploy < list < service/revenue, continuing from whatever the start state already did."""
    st: dict = {}
    _prime(ag, st)
    fam = ag.info["task_family"]
    need = st.get("need")
    m = re.search(r"`([a-z]+\.[a-z]+)`", ag.goal)
    if m:
        need = m.group(1)
    if "org" not in st:
        yield from s_company(ag, st, fund=upto != "found")
    if upto in ("found", "fund"):
        return
    if need is None:
        r = _res((yield tool("market.browse", kind="api")), [])
        need = _pick_need(ag, rng, r)
    if "repo" not in st:
        yield from s_code(ag, st, need)
    if upto == "code":
        return
    if "host" not in st or fam == "repair":
        if "host" not in st:
            yield from s_deploy(ag, st)
    if upto == "deploy":
        return
    if "listing" not in st:
        yield from s_list(ag, st, need)


def script_repair(ag, rng):
    st: dict = {}
    _prime(ag, st)
    r = _res((yield tool("cloud.status", host=st.get("host"))), {})
    if r.get("status") != "live" or r.get("availability") not in ("ok", None):
        yield from s_deploy(ag, st)
    r = _res((yield tool("market.browse", kind="api", need=st.get("need"), sort="price")), [])
    rivals = [x["price_cents"] for x in r if x.get("seller") != st.get("org")]
    own = [x for x in r if x.get("seller") == st.get("org")]
    if rivals and own and own[0]["price_cents"] > 1.5 * min(rivals):
        yield tool("market.update", listing=own[0]["listing"], price_cents=NEAR(min(rivals)))


def script_apicall(ag, rng):
    r = _res((yield tool("market.browse", kind="api", sort="price")), [])
    if not r:
        return
    host = r[int(rng.integers(0, len(r)))]["host"]
    yield tool("world.describe", host=host)
    n = int(re.search(r"at least (\d+)", ag.goal).group(1)) if re.search(r"at least (\d+)", ag.goal) else 3
    for _ in range(n + 2):
        yield tool(f"{host}.solve", **{"*": GEN(lambda d: d.source == "world.describe" and host in str(d.raw))})


def script_auto(ag, rng):
    fam = ag.info["task_family"]
    if fam == "apicall":
        yield from script_apicall(ag, rng)
    elif fam == "repair":
        yield from script_repair(ag, rng)
    else:
        yield from script_build(ag, rng, fam if fam in ("found", "fund", "code", "deploy", "list") else "list")
    while True:
        yield wait(DAY if fam in ("service", "revenue", "repair") else 3600)


SCRIPTS = {"auto": script_auto}


class TeacherChooser(Chooser):
    name = "teacher"

    def __init__(self, script: str = "auto", seed: int = 0):
        self.script, self.rng = script, np.random.default_rng(seed)
        self.gen = None
        self.intent = None
        self.misses: list[str] = []
        self.n = 0

    def head(self, ag, items, logits):
        try:
            if self.gen is None:
                self.gen = SCRIPTS[self.script](ag, self.rng)
                self.intent = next(self.gen)
            else:
                self.intent = self.gen.send(ag.last_out)       # the full outcome of the previous decision
        except StopIteration:
            self.intent = wait(DAY)
        self.n += 1
        kind, name, _ = self.intent
        for i, m in enumerate(items.meta):
            if m[0] == kind and m[1] == name:
                return i
        self.misses.append(f"head {kind}:{name}")
        self.intent = wait(3600)
        return next(i for i, m in enumerate(items.meta) if m[0] == "wait" and m[1] == 3600)

    def option(self, ag, tool_name, slot, items, logits):
        want = self.intent[2].get(slot.param)
        opts = slot.options
        if want is None:
            if not slot.required:
                return 0                       # optional arguments the script did not mention are omitted
            self.misses.append(f"{tool_name}.{slot.param}: unspecified required argument")
            return 0
        i = want.pick(opts)
        if i is None:
            self.misses.append(f"{tool_name}.{slot.param}: no option for {want.desc}")
            return 1 if (not slot.required and len(opts) > 1) else 0
        return i

    def report(self) -> dict:
        return {"script": self.script, "intents": self.n, "misses": self.misses[:20]}
