"""Autonomous developers of the one population (module `devs`, host `devs`; D-15, D-18, wave 3b workstream G).

Technology invention in a v5 world is done by people of the population (ONE_WORLD_CONTRACTS section 1): a small
share of individuals - those with the most technical skill - are *developers*. A developer takes part in the
population's life through one registered activity, `develop`, and acts only through the ordinary tools, as itself
(principal `p<i>`, paying from its own bank account). Nothing here is scripted per task, product or participant; a
developer decides from public observables and its own experience:

* **Opportunities** (public): requests for help on the forum boards ("Looking for a service: ..." posts with an
  example input, written by individuals who do a task by hand - `economy.work`) grouped by task, counted by
  distinct askers in a recent window, and the competition the developer finds for the task in the capability
  registry (`caps.search`: live declarations whose input schema accepts the posted examples, their prices and
  public counts). Its estimate of what a call is worth comes from the hand-work minutes people state and its own
  guess of their value of time; it builds when the expected monthly margin clears its own hurdle, which falls with
  its interest in the task (semantic affinity of its interests to the task's description in the shared space,
  `semantic.space.text_vector`).
* **Understanding a task** (public text only): a task is one the developer knows how to program when the request
  names a documented format (public format documents, `docs.list`) or quotes the documentation of an
  undocumented task in its body. Developer *labour* is modelled (fidelity record FR-DEV-1): a developer of craft
  c writes one of the known approaches to that task - the reference implementations at increasing quality that
  every format carries (`economy.needs`, `usefulness`; the same convention 4.0 uses for simulated developer
  labour) - the approach drawn from its craft. For tasks whose rule is shown only through worked examples or must
  be learnt from outcomes (`exemplified`, `experiential` categories) the best approach is out of reach of a first
  version. Which approach it wrote is the developer's private knowledge; nobody else (and no evaluator metric)
  reads it. The program is real code: it runs in the sandbox, its quality is whatever the world's physics makes
  of its outputs (employers accept or reject the individuals' work), never declared.
* **Reuse** (public): before writing, a developer searches packages (`pkg.search`) for the task; with its reuse
  propensity it builds its service on a package whose API offers `solve` and whose tests (its own smoke tests on
  the posted examples, `pkg.ci`) pass - the CI run executes the package across the attested boundary, which is
  adoption by execution by an independent developer (`pkg_adoptions`).
* **Building**: a repository with the program and its smoke tests, compute rented from the cheapest provider with
  free cores, a deployment (`cloud.deploy`, or `pkg.deploy` when it locks packages), a typed capability
  declaration whose input schema is inferred from the posted examples and whose output schema from its own run on
  one of them (`pkg.exec`), a price per call, and a reply to each request naming its host. With its open-source
  propensity it also publishes the program as a package (`pkg.register`, `pkg.release` with tests and README).
* **Maintenance** (own observables): every session it reads its service's metrics (`cloud.metrics`: overloads,
  errors), its own statement (`bank.statement`: which accounts paid it this week and last week - retention -, how
  many paid once and never again, and
  what compute cost it), and its support tickets (`support.list`, answered and resolved). Overloads -> more cores;
  poor retention, errors or complaints -> a new version (a better approach with a probability given by its craft;
  redeployed on the same host, its package released as a new minor version); a newer release of a package it
  depends on -> `pkg.upgrade` (only if its tests pass) and a redeploy; a service that loses money with no paying
  customer for `loss_days` -> shut down (`cloud.release`).

Experiments (evaluator-side interventions; arms of one spec and seed differ only here): `program` "garbage" makes
every developer ship an input-ignoring program with the same interface (it answers a fixed example), `ban` lists
tasks nobody builds (task titles or `fnmatch` patterns of the task's known format name, e.g. "gen.*": removal of an
invention), `reuse_share`/`oss_share` 0 remove the package channel.

Decisions are recorded through the population API (`record_decisions`, model `devs.opportunity`, fidelity record
FR-DEV-1, an uncalibrated documented assumption).
"""
from __future__ import annotations

import fnmatch
import json
import math
import re

import numpy as np

from ..core.errors import CrucibleError
from ..core.kernel import Module, applier, process, register_module
from ..substrate import columns, ids

DAY = 86400
TABLE = "devs"
MODEL = {"model_id": "devs.opportunity", "model_version": "1", "fidelity_id": "FR-DEV-1"}
ASK = "Looking for a service: "
EX_RE = re.compile(r"Example input: (\{.*\})", re.S)
MIN_RE = re.compile(r"about (\d+(?:\.\d+)?) minutes")
STOP = frozenset("a an the of for to and or in on by with from into per your my is are be as at this that it its "
                 "across each every all any".split())
DEFAULTS = {"share": 0.03, "min_devs": 8, "max_devs": 400, "period_h": [30.0, 60.0], "first_day": 2,
            "evening": [18, 23], "min_asks": 3, "window_days": 21, "max_projects": 2, "min_capital": 3000,
            "program": "good", "ban": [], "oss_share": 0.5, "reuse_share": 0.6, "boards": None,
            "value_cents_min": [4.0, 16.0], "price_share": 0.08, "reach": 3.0, "max_cores": 16,
            "improve_after_days": 4, "retention_floor": 0.5, "loss_days": 21, "read_posts": 6,
            "list_limit": 30, "hurdle": [0.3, 1.5], "min_trials": 8, "min_conversion": 0.1, "retry_days": 10}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9]+", (text or "").lower()) if t not in STOP]


def slug(text: str, n: int = 2) -> str:
    out = []
    for t in _tokens(text):
        if t not in out:
            out.append(t)
        if len(out) >= n:
            break
    return "-".join(out) or "task"


def _json_type(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "null"


def infer_schema(examples: list) -> dict:
    """The schema a developer infers from example values: keys present in every example are required, with the
    JSON type they had (integers and numbers merge to number). Extra properties are allowed."""
    dicts = [e for e in examples if isinstance(e, dict)]
    if not dicts:
        return {"type": "object"}
    keys = set(dicts[0])
    for e in dicts[1:]:
        keys &= set(e)
    props = {}
    for k in sorted(keys):
        ts = {_json_type(e[k]) for e in dicts}
        if ts <= {"integer", "number"} and "number" in ts:
            ts = {"number"}
        props[k] = {"type": sorted(ts)[0] if len(ts) == 1 else sorted(ts)}
        if "object" in ts:
            props[k]["additionalProperties"] = True          # nothing is known about nested fields
    return {"type": "object", "required": sorted(keys), "properties": props, "additionalProperties": True}


def _tier_list(need: str) -> list[tuple[str, str]]:
    from ..economy.needs import NEEDS
    return list(NEEDS[need].tiers)


def _relation(need: str) -> str:
    """How much of a task's rule its documentation states (usefulness categories); published formats are fully
    specified."""
    if not need.startswith("gen."):
        return "specified"
    from ..usefulness import category as C
    cat = C.category_of(need)
    return cat.relation if cat is not None else "specified"


SMOKE = '''import json
from service import solve

EXAMPLES = json.loads({examples!r})


def test_runs_on_posted_examples():
    for e in EXAMPLES:
        out = solve(e, None)
        assert isinstance(out, dict) and out
'''
LIB_SMOKE = '''import json
from lib import solve

EXAMPLES = json.loads({examples!r})


def test_runs_on_posted_examples():
    for e in EXAMPLES:
        assert isinstance(solve(e, None), dict)
'''
GARBAGE = '''
import json
_FIXED = json.loads({example!r})


def solve(args, caller):
    # ignores its input: always answers the one example it was written against
    return _impl(_FIXED, caller)
'''
WRAP = '''from deps.{dir}.{module} import solve as _impl


def solve(args, caller):
    return _impl(args, caller)
'''


WORKFLOW = '''import crucible_sdk as sdk

STAGE1, STAGE2, KEY = {a!r}, {b!r}, {key!r}


def _body(r):
    if not isinstance(r, dict) or r.get("status") != 200:
        raise RuntimeError("component failed: %s" % (r.get("error") if isinstance(r, dict) else r,))
    return r.get("body")


def solve(args, caller):
    # stage 1 by one provider's service, its records into stage 2 by another's (both paid per call)
    first = _body(sdk.call(STAGE1, **args["stage1"]))
    req = dict(args["stage2"])
    req[KEY] = first["records"]
    return _body(sdk.call(STAGE2, **req))
'''


def _parts(need: str):
    """(first, second, table key) of a composed task category, else None."""
    if not need.startswith("gen."):
        return None
    from ..usefulness import category as C
    cat = C.category_of(need)
    if cat is None or not getattr(cat, "parts", None) or len(cat.parts) != 2:
        return None
    return cat.parts[0], cat.parts[1], (cat.rule or {}).get("table_key")


def garbage_source(src: str, example: dict) -> str:
    """An input-ignoring program with the same interface (experiments only): the approach, renamed, applied to a
    fixed example whatever the input."""
    return re.sub(r"(?m)^def solve\(", "def _impl(", src) + GARBAGE.format(example=json.dumps(example, sort_keys=True))


class DevError(Exception):
    pass


@register_module
class Developers(Module):
    name = "devs"
    host = "devs"
    depends = ("identity", "bank", "pop", "cols", "code", "cloud", "caps", "pkg", "forum", "search", "docs",
               "support")

    def cfg(self, k):
        return self.config.get(k, DEFAULTS[k])

    # ------------------------------------------------------------------ setup
    def install(self, world):
        from ..economy import popbridge as pb
        n = pb.n_individuals(world)
        columns.declare(world, TABLE, n, {
            "dev": ("bool", (), False), "craft": ("float32", (), 0.0), "oss": ("float32", (), 0.0),
            "reuse": ("float32", (), 0.0), "vguess": ("float32", (), 0.0), "hurdle": ("float32", (), 1.0),
            "next_at": ("int64", (), -1), "sessions": ("int32", (), 0), "builds": ("int32", (), 0),
            "choice": ("int32", (), -1)})
        st = world.state
        st.declare_table("dev_projects", index=("dev", "host", "status", "task"))
        pb.register_activity(world, "develop", self.act_develop, owner="devs", prior=-1.0, goals=(5,))

    def genesis(self, ctx):
        from ..economy import popbridge as pb
        w = ctx.world
        n = pb.n_individuals(w)
        if not n:
            return
        rng = w.rng_for("devs-genesis")
        f = pb.features(w, np.arange(n))
        sk = f["skills"].astype(np.float64)
        tech = sk[:, 4] if sk.shape[1] > 4 else np.full(n, 0.5)
        plan = sk[:, 7] if sk.shape[1] > 7 else np.full(n, 0.5)
        k = int(np.clip(round(float(self.cfg("share")) * n), int(self.cfg("min_devs")), int(self.cfg("max_devs"))))
        k = min(k, n)
        score = tech + 0.05 * rng.standard_normal(n)
        dev_idx = np.sort(np.argsort(-score, kind="stable")[:k])
        craft = np.clip(0.7 * tech[dev_idx] + 0.3 * plan[dev_idx] + 0.1 * rng.standard_normal(k), 0.0, 1.0)
        lo, hi = self.cfg("value_cents_min")
        hlo, hhi = self.cfg("hurdle")
        first = (ctx.now // DAY + int(self.cfg("first_day"))) * DAY
        cols = {"dev": np.ones(k, bool), "craft": craft.astype(np.float32),
                "oss": (rng.random(k) < float(self.cfg("oss_share"))).astype(np.float32),
                "reuse": (rng.random(k) < float(self.cfg("reuse_share"))).astype(np.float32),
                "vguess": np.exp(rng.uniform(math.log(float(lo)), math.log(float(hi)), k)).astype(np.float32),
                "hurdle": rng.uniform(float(hlo), float(hhi), k).astype(np.float32),
                "next_at": (first + rng.integers(0, 3 * DAY, k)).astype(np.int64)}
        columns.batch(ctx, TABLE, "devs.genesis", dev_idx, cols, mode="set", extra={"developers": int(k)})
        ctx.schedule(first, "devs.hourly", {}, prio=37, period=3600, key="devs.hourly")
        ctx.schedule(first, "devs.day", {}, prio=37, period=DAY, key="devs.day")

    def _col(self, name):
        return columns.col(self.world, TABLE, name)

    # ------------------------------------------------------------------ scheduling
    @process("hourly")
    def pr_hourly(self, ctx):
        """Wake developers whose next session is due, in their evening (after the working day, so the wake does
        not displace a pending work session); a wake the population does not take is retried the next hour."""
        from ..economy import popbridge as pb
        h = (ctx.now % DAY) // 3600
        lo, hi = self.cfg("evening")
        if not int(lo) <= h <= int(hi):
            return
        due = np.flatnonzero(self._col("dev") & (self._col("next_at") >= 0) & (self._col("next_at") <= ctx.now))
        if due.size:
            pb.wake(ctx, due, int(ctx.now), "develop")

    @process("day")
    def pr_day(self, ctx):
        """Developers running services keep their income deposited (their compute is billed hourly)."""
        from ..economy import popbridge as pb
        st = ctx.state
        live = sorted({int(r["dev"]) for _, r in st.scan("dev_projects") if r["status"] == "live"})
        for i in live:
            pb.fund(ctx, i)

    # ------------------------------------------------------------------ helpers (all as the developer)
    def _do(self, ctx, p, tool, **args):
        try:
            return ctx.call(tool, args, as_principal=p)
        except CrucibleError as e:
            raise DevError(f"{tool}: {e.message}") from None

    def _try(self, ctx, p, tool, **args):
        try:
            return ctx.call(tool, args, as_principal=p)
        except CrucibleError:
            return None

    @staticmethod
    def _account(st, p) -> str | None:
        for a in st.find_ids("accounts", "owner", p):
            return a
        return None

    def projects(self, st, i: int, status: str | None = None) -> list[tuple[str, dict]]:
        out = []
        for pid in st.find_ids("dev_projects", "dev", int(i)):
            r = st.tables["dev_projects"][pid]
            if status is None or r["status"] == status:
                out.append((pid, r))
        return sorted(out)

    # ------------------------------------------------------------------ the session
    def act_develop(self, ctx, idx: np.ndarray, now: int):
        from ..economy import popbridge as pb
        util = np.zeros(idx.size)
        dev, nxt = self._col("dev")[idx], self._col("next_at")[idx]
        rows = np.flatnonzero(dev & (nxt >= 0) & (nxt <= now + 3600))
        util[~dev] = -0.3
        if rows.size == 0:
            return util
        st = ctx.state
        lo, hi = self.cfg("period_h")
        done, nexts, choices = [], [], []
        for r in rows:
            i = int(idx[r])
            rng = ctx.world.rng_for("devs.session", i, now)
            p = pb.fund(ctx, i)
            choice = 0
            for pid, proj in self.projects(st, i, "live"):
                choice = max(choice, self._maintain(ctx, i, p, pid, proj, now, rng))
            if len(self.projects(st, i, "live")) < int(self.cfg("max_projects")):
                if self._explore(ctx, i, p, now, rng):
                    choice = 3
            done.append(i)
            nexts.append(now + int(rng.uniform(float(lo), float(hi)) * 3600))
            choices.append(choice)
            util[r] = 0.2 + 0.2 * (choice > 0)
        if done:
            ix = np.array(done, np.int64)
            columns.batch(ctx, TABLE, "devs.session", ix, {"next_at": np.array(nexts, np.int64),
                                                           "sessions": self._col("sessions")[ix] + 1}, mode="set")
            pb.record_decisions(ctx, "devs.decided", ix, {"choice": np.array(choices, np.int32)}, table=TABLE,
                                **MODEL)
        return util

    # ------------------------------------------------------------------ opportunities (public)
    def _requests(self, ctx, p, now) -> dict[str, dict]:
        """Public requests for help grouped by task (title after the request prefix)."""
        boards = self.cfg("boards")
        if not boards:
            boards = [b["board"] for b in (self._try(ctx, p, "forum.boards") or [])]
        since = now - int(self.cfg("window_days")) * DAY
        out: dict[str, dict] = {}
        for b in boards:
            for post in self._try(ctx, p, "forum.list", board=b, sort="new", limit=int(self.cfg("list_limit"))) or []:
                title = post.get("title") or ""
                if not title.startswith(ASK) or post["t"] < since or post["author"] == p:
                    continue
                task = title[len(ASK):].strip()
                g = out.setdefault(task, {"authors": set(), "posts": []})
                g["authors"].add(post["author"])
                g["posts"].append(post)
        return out

    def _read(self, ctx, p, g: dict) -> dict:
        """Examples, stated minutes and full text of a task's requests (a few posts read)."""
        ex, mins, bodies = [], [], []
        for post in sorted(g["posts"], key=lambda x: (-x["t"], x["post"]))[:int(self.cfg("read_posts"))]:
            full = self._try(ctx, p, "forum.read", post=post["post"], replies=0)
            if not full:
                continue
            body = full.get("body") or ""
            bodies.append(body)
            m = EX_RE.search(body)
            if m:
                try:
                    ex.append(json.loads(m.group(1)))
                except json.JSONDecodeError:
                    pass
            mm = MIN_RE.search(body)
            if mm:
                mins.append(float(mm.group(1)))
        return {"examples": [e for e in ex if isinstance(e, dict)], "minutes": mins, "text": "\n".join(bodies)}

    def _documented(self, ctx, p, task: str) -> set[str]:
        """Format names with a public format document (`Format <name>: <title>`) the search engine finds for the
        task's words."""
        out = set()
        res = self._try(ctx, p, "search.query", q=f"Format {task[:150]}", kind="doc", limit=5) or {}
        for d in res.get("results") or []:
            m = re.match(r"Format ([a-z0-9_.]+):", d.get("title") or "")
            if m:
                out.add(m.group(1))
        return out

    def understand(self, task: str, text: str, documented: set[str]) -> str | None:
        """The known task a request describes, from public text only: a documented format whose title the request
        names, or an undocumented task whose description the request quotes."""
        from ..economy.needs import NEEDS
        for name in sorted(documented):
            nd = NEEDS.get(name)
            if nd is not None and task.startswith(nd.title):
                return name
        norm = " ".join(text.split())
        for name in sorted(NEEDS):
            if not name.startswith("gen."):
                continue
            nd = NEEDS[name]
            head = " ".join(nd.doc.split())[:240]
            if head and head in norm and task.startswith(nd.title):
                return name
        return None

    def _competitors(self, ctx, p, task: str, examples: list) -> list[dict]:
        from ..core import schema as S
        out = []
        for r in self._try(ctx, p, "caps.search", query=task[:200], limit=10) or []:
            if r.get("kind") != "declaration" or not r.get("live") or r.get("provider") == p:
                continue
            if examples and any(S.validate(r["inputs"], e, "input") for e in examples[:3]):
                continue
            out.append({"host": r["host"], "price": int(r.get("price_cents") or 0), "adopters": int(r.get("adopters") or 0),
                        "trials": int(r.get("trials") or 0)})
        return out

    @staticmethod
    def converts(c: dict) -> bool:
        """A competitor (or one's own service) that people keep after trying it: the registry's public counts of
        trials and active adopters. Too few trials say nothing yet."""
        return c["trials"] < int(DEFAULTS["min_trials"]) or c["adopters"] >= float(DEFAULTS["min_conversion"]) * c["trials"]

    def _affinity(self, ctx, i: int, task: str) -> float:
        from ..economy import popbridge as pb
        from ..semantic import space
        v = space.text_vector(ctx.world, task, ctx=ctx)
        u = pb.interests(ctx.world, np.array([int(i)]))[0]
        return float(np.dot(u, v) / max(1e-6, float(np.linalg.norm(u)) * float(np.linalg.norm(v))))

    def _explore(self, ctx, i, p, now, rng) -> bool:
        st = ctx.state
        reqs = self._requests(ctx, p, now)
        if not reqs:
            return False
        acct = self._account(st, p)
        bal = int(st.tables["accounts"][acct]["balance"]) if acct else 0
        if bal < int(self.cfg("min_capital")):
            return False
        # tasks it already serves, and tasks it failed to build for in the last `retry_days`
        mine = {r["task"] for _, r in self.projects(st, i)
                if r["status"] != "abandoned" or now - int(r["t"]) < int(self.cfg("retry_days")) * DAY}
        ban = list(self.cfg("ban") or ())
        provs = self._try(ctx, p, "cloud.providers") or []
        provs = [x for x in provs if x["cores_free"] >= 1 and not x["in_outage"]]
        if not provs:
            return False
        rent_m = float(min(x["price_core_hour_cents"] for x in provs)) * 24 * 30
        best = None
        for task, g in sorted(reqs.items()):
            if task in mine or task in ban or len(g["authors"]) < int(self.cfg("min_asks")):
                continue
            info = self._read(ctx, p, g)
            if not info["examples"]:
                continue
            need = self.understand(task, info["text"], self._documented(ctx, p, task))
            if need is None or any(fnmatch.fnmatchcase(need, b) for b in ban):
                continue
            comp = self._competitors(ctx, p, task, info["examples"])
            minutes = float(np.median(info["minutes"])) if info["minutes"] else 20.0
            price = int(np.clip(round(float(self.cfg("price_share")) * minutes * float(self._col("vguess")[i])), 2, 80))
            if any(self.converts(c) for c in comp):
                low = min(c["price"] for c in comp if self.converts(c))
                price = int(max(2, min(price, round(0.9 * low))))
            # competitors that nobody keeps after trying them do not serve the market
            users = len(g["authors"]) * float(self.cfg("reach")) / (1.0 + sum(1 for c in comp if self.converts(c)))
            margin = users * 30 * 1.5 * price - rent_m
            aff = self._affinity(ctx, i, task)
            hurdle = float(self._col("hurdle")[i]) * (1.2 - 0.6 * max(0.0, aff)) * rent_m
            plan = self._compose_plan(ctx, p, need, info) if _parts(need) else None
            if plan is not None:
                # composing existing services: no program to write, components' prices are a cost per call
                price = int(max(price, plan["cost"] + 2))
                margin = users * 30 * 1.5 * (price - plan["cost"]) - rent_m
                hurdle *= 0.5
            score = margin - hurdle
            if score > 0 and (best is None or score > best[0]):
                best = (score, task, need, info, price, comp, aff, plan)
        if best is None:
            return False
        _, task, need, info, price, comp, aff, plan = best
        try:
            if plan is not None:
                self._build_workflow(ctx, i, p, acct, task, need, info, price, len(comp), aff, plan, now)
            else:
                self._build(ctx, i, p, acct, task, need, info, price, len(comp), aff, rng, now)
        except DevError as e:
            # remembered: it does not try the same task again for a while
            ctx.emit("devs.project", {"id": st.new_id("dproj"), "dev": int(i), "principal": p, "task": task,
                                      "need": need, "approach": -2, "host": None, "repo": None, "instance": None,
                                      "decl": None, "price": int(price), "cores": 0, "uses_package": None,
                                      "competitors": len(comp), "affinity": round(float(aff), 4), "examples": [],
                                      "status": "abandoned", "why": str(e)[:200]}, vis="internal")
            return False
        return True

    # ------------------------------------------------------------------ workflows (composing others' services)
    def _compose_plan(self, ctx, p, need: str, info: dict) -> dict | None:
        """For a two-stage task ("a, then b on its result"), live services of other providers for each stage that
        people keep after trying them (public registry counts) and whose declared inputs accept the stage's
        part of the posted examples."""
        from ..core import schema as S
        a, b, key = _parts(need)
        ex = [e for e in info["examples"] if isinstance(e.get("stage1"), dict) and isinstance(e.get("stage2"), dict)]
        if not ex:
            return None
        picks = []
        for part, sample in ((a, ex[0]["stage1"]), (b, None)):
            best = None
            for r in self._try(ctx, p, "caps.search", query=part.title[:200], limit=10) or []:
                if r.get("kind") != "declaration" or not r.get("live") or r.get("provider") == p:
                    continue
                if r.get("workflows"):
                    continue                              # not another workflow
                c = {"trials": int(r.get("trials") or 0), "adopters": int(r.get("adopters") or 0)}
                if c["adopters"] < 1 or not self.converts(c):
                    continue
                if sample is not None and S.validate(r["inputs"], sample, "input"):
                    continue
                if sample is None and not set(r["inputs"].get("required") or []) <= set(ex[0]["stage2"]) | {key}:
                    continue
                cand = (-c["adopters"], int(r.get("price_cents") or 0), r["declaration"], r)
                if best is None or cand < best:
                    best = cand
            if best is None:
                return None
            picks.append(best[3])
        return {"components": picks, "key": key, "cost": sum(int(r.get("price_cents") or 0) for r in picks)}

    def _build_workflow(self, ctx, i, p, acct, task, need, info, price, n_comp, aff, plan, now) -> str:
        st = ctx.state
        from ..commerce.reference import service_files
        a, b = plan["components"]
        src = WORKFLOW.format(a=f"{a['host']}.{a['endpoint']}", b=f"{b['host']}.{b['endpoint']}", key=plan["key"])
        s = slug(task)
        repo = self._do(ctx, p, "code.create_repo", name=self._repo_name(ctx, p, f"d{i}{s}"))["repo"]
        host = self._host_name(ctx, i, s, repo)
        self._do(ctx, p, "code.commit", repo=repo, files=service_files(src, task[:120], price=price), message="v1")
        provs = sorted((x for x in (self._try(ctx, p, "cloud.providers") or []) if x["cores_free"] >= 1
                        and not x["in_outage"]), key=lambda x: (x["price_core_hour_cents"], x["provider"]))
        if not provs:
            raise DevError("no compute")
        inst = self._do(ctx, p, "cloud.rent", provider=provs[0]["provider"], cores=1, account=acct)["instance"]
        try:
            dep = self._do(ctx, p, "cloud.deploy", repo=repo, host=host, instance=inst)
            if dep.get("status") != "live":
                raise DevError(f"deployment {dep.get('status')}")
            # the workflow's service pays the components from the developer's account, up to a monthly limit
            self._do(ctx, p, "id.grant", grantee=f"svc:{host}", cap="bank.spend", limit=max(2000, 400 * plan["cost"]))
            examples = info["examples"][:4]
            out = self._try(ctx, p, "pkg.exec", repo=repo, target="service:solve", input=examples[0])
            if not out or not out.get("ok") or not isinstance(out.get("result"), dict):
                raise DevError("the workflow does not run on the posted example")
            name = (".".join(slug(task, 2).split("-")) if "-" in slug(task, 2) else slug(task, 1) + ".task")[:50]
            decl = self._do(ctx, p, "caps.declare", service_host=host, capability={
                "name": name + ".flow", "inputs": infer_schema(examples), "outputs": infer_schema([out["result"]]),
                "endpoint": "solve", "tags": _tokens(task)[:6],
                "doc": f"{task}. A workflow: {a['name']} ({a['host']}) on stage 1, then {b['name']} ({b['host']}) on "
                       f"its records. {price} cents a call."})["declaration"]
            self._do(ctx, p, "caps.compose", service_host=host, name=name + ".flow",
                     components=[a["declaration"], b["declaration"]],
                     doc=f"{task}: stage 1 by {a['host']}, its records into stage 2 by {b['host']}.")
        except DevError:
            self._try(ctx, p, "cloud.release", instance=inst)
            raise
        pid = st.new_id("dproj")
        ctx.emit("devs.project", {"id": pid, "dev": int(i), "principal": p, "task": task, "need": need,
                                  "approach": -1, "host": host, "repo": repo, "instance": inst, "decl": decl,
                                  "price": int(price), "cores": 1, "uses_package": None,
                                  "competitors": int(n_comp), "affinity": round(float(aff), 4),
                                  "examples": [], "composes": [a["declaration"], b["declaration"]]}, vis="internal")
        self._announce(ctx, p, task, host, price, now)
        return pid

    # ------------------------------------------------------------------ building
    def _draw(self, i: int, n: int, top: int, rng, current: int | None) -> int:
        c = float(self._col("craft")[i])
        if current is None:
            return int(np.clip(int(np.floor(c * (top + 1) + rng.normal(0.0, 0.6))), 0, top))
        if current >= n - 1 or rng.random() >= c:
            return int(current)
        return int(min(n - 1, current + 1))

    def approach(self, i: int, need: str, rng, current: int | None = None) -> int:
        """Which known approach the developer writes (developer-labour model FR-DEV-1): a first version is drawn
        from its craft; a new version is better than `current` with probability craft (else the same). A task
        whose rule is only exemplified or learnt from outcomes caps a first version below the best approach. A
        two-stage task is written as its two stages (`usefulness.category.stage_source`), each drawn this way;
        the approach is then `k_first * n_second + k_second` and an improvement improves the weaker stage."""
        parts = _parts(need)
        if parts is None:
            n = len(_tier_list(need))
            top = n - 1 if _relation(need) == "specified" else max(0, n - 2)
            return self._draw(i, n, top, rng, current)
        a, b, _key = parts
        na, nb = len(a.tiers), len(b.tiers)
        ta = na - 1 if a.relation == "specified" else max(0, na - 2)
        tb = nb - 1 if b.relation == "specified" else max(0, nb - 2)
        if current is None:
            return self._draw(i, na, ta, rng, None) * nb + self._draw(i, nb, tb, rng, None)
        ka, kb = divmod(int(current), nb)
        if ka < na - 1 and (ka / max(1, na - 1)) <= (kb / max(1, nb - 1)):
            ka = self._draw(i, na, ta, rng, ka)
        else:
            kb = self._draw(i, nb, tb, rng, kb)
        return ka * nb + kb

    def _source(self, need: str, k: int, example: dict) -> str:
        parts = _parts(need)
        if parts is None:
            src = _tier_list(need)[k][1]
        else:
            from ..usefulness.category import stage_source
            a, b, key = parts
            ka, kb = divmod(int(k), len(b.tiers))
            src = stage_source(a.tiers[ka][1], b.tiers[kb][1], key)
        if self.cfg("program") == "garbage":
            return garbage_source(src, example)
        return src

    def _own_repos(self, ctx, p: str) -> set[str]:
        """The developer's own repositories (`code.list`, as itself)."""
        return {r["repo"] for r in (self._try(ctx, p, "code.list") or [])}

    def _repo_name(self, ctx, p: str, base: str) -> str:
        """A repository name of `p` not used yet (from its own repository list)."""
        mine = self._own_repos(ctx, p)
        base = base.replace("-", "")[:34]
        name, k = base, 1
        while f"{p}/{name}" in mine:
            k += 1
            name = f"{base}{k}"
        return name

    def _host_name(self, ctx, i: int, s: str, repo: str) -> str:
        """The host of a new service: the developer's namespace and the task slug, numbered like its repository
        (repositories are created first and are unique), and not a host of one of its own projects."""
        short = repo.split("/", 1)[1]
        k = short[len(f"d{i}{s.replace('-', '')}"[:34]):]
        return self._own_name(ctx, i, f"d{i}-{s}"[:24] + (f"-{k}" if k else ""), "host", 28)

    def _own_name(self, ctx, i: int, base: str, field: str, limit: int) -> str:
        """A host or package name in the developer's own namespace (`base` carries its index), not used by one of
        its own projects yet (its own records)."""
        used = {r[field] for _, r in self.projects(ctx.state, i) if r.get(field)}
        name, k = base[:limit], 1
        while name in used:
            k += 1
            name = f"{base[:limit - 4]}-{k}"
        return name

    def _build(self, ctx, i, p, acct, task, need, info, price, n_comp, aff, rng, now) -> str:
        st = ctx.state
        from ..commerce.reference import service_files
        examples = info["examples"][:4]
        s = slug(task)
        k = self.approach(i, need, rng)
        uses = None
        # reuse: a package offering solve whose smoke tests pass on the posted examples
        if float(self._col("reuse")[i]) > 0:
            for pk in self._try(ctx, p, "pkg.search", query=task[:200], limit=5) or []:
                if "solve" not in (pk.get("api") or {}) or pk.get("owner") == p or not pk.get("latest"):
                    continue
                uses = pk
                break
        repo_name = self._repo_name(ctx, p, f"d{i}{s}")
        repo = self._do(ctx, p, "code.create_repo", name=repo_name)["repo"]
        smoke = SMOKE.format(examples=json.dumps(examples, sort_keys=True))
        src = self._source(need, k, examples[0])
        files = service_files(src, task[:120], price=price)
        files["tests/test_service.py"] = smoke
        self._do(ctx, p, "code.commit", repo=repo, files=files, message="v1")
        if uses is not None:
            try:
                lock = self._do(ctx, p, "pkg.depend", repo=repo, package=uses["package"], constraint="*")
                d, _, mod = lock["import"][len("deps."):].partition(".")
                self._do(ctx, p, "code.commit", repo=repo, files={"service.py": WRAP.format(dir=d, module=mod)},
                         message=f"use {uses['package']}")
                ci = self._do(ctx, p, "pkg.ci", repo=repo)
                if not ci["ok"]:
                    raise DevError("package does not pass my tests")
            except DevError:
                # back to its own program (the dependency stays locked but unused: no adoption is credited)
                self._do(ctx, p, "code.commit", repo=repo, files={"service.py": src}, message="own implementation")
                uses = None
        if uses is None:
            ci = self._do(ctx, p, "pkg.ci", repo=repo)
            while not ci["ok"] and k > 0:
                k -= 1
                src = self._source(need, k, examples[0])
                self._do(ctx, p, "code.commit", repo=repo, files={"service.py": src}, message="simpler approach")
                ci = self._do(ctx, p, "pkg.ci", repo=repo)
            if not ci["ok"]:
                raise DevError("no approach passes my tests")
        out = self._try(ctx, p, "pkg.exec", repo=repo, target="service:solve", input=examples[0])
        outputs = infer_schema([out["result"]]) if out and out.get("ok") and isinstance(out.get("result"), dict) \
            else {"type": "object"}
        provs = sorted((x for x in (self._try(ctx, p, "cloud.providers") or []) if x["cores_free"] >= 1
                        and not x["in_outage"]), key=lambda x: (x["price_core_hour_cents"], x["provider"]))
        if not provs:
            raise DevError("no compute")
        inst = self._do(ctx, p, "cloud.rent", provider=provs[0]["provider"], cores=1, account=acct)["instance"]
        host = self._host_name(ctx, i, s, repo)
        try:
            dep = self._do(ctx, p, "pkg.deploy" if uses else "cloud.deploy", repo=repo, host=host, instance=inst)
            if dep.get("status") != "live":
                raise DevError(f"deployment {dep.get('status')}")
            name = ".".join(slug(task, 2).split("-")) if "-" in slug(task, 2) else slug(task, 1) + ".task"
            words = [t for t in _tokens(task)][:8]
            decl = self._do(ctx, p, "caps.declare", service_host=host, capability={
                "name": name[:60], "inputs": infer_schema(examples), "outputs": outputs, "endpoint": "solve",
                "doc": f"{task}. A service for this task, {price} cents a call. Send one task input as JSON.",
                "tags": words[:6]})["declaration"]
        except DevError:
            self._try(ctx, p, "cloud.release", instance=inst)
            raise
        pid = st.new_id("dproj")
        ctx.emit("devs.project", {"id": pid, "dev": int(i), "principal": p, "task": task, "need": need,
                                  "approach": int(k), "host": host, "repo": repo, "instance": inst, "decl": decl,
                                  "price": int(price), "cores": 1, "uses_package": uses["package"] if uses else None,
                                  "competitors": int(n_comp), "affinity": round(float(aff), 4),
                                  "examples": examples[:2]}, vis="internal")
        self._announce(ctx, p, task, host, price, now)
        if uses is None and float(self._col("oss")[i]) > 0:
            self._publish_package(ctx, i, p, pid, task, src, examples)
        columns.batch(ctx, TABLE, "devs.built", np.array([int(i)]), {"builds": self._col("builds")[[int(i)]] + 1},
                      mode="set")
        return pid

    def _announce(self, ctx, p, task, host, price, now, new_version: bool = False) -> None:
        """Reply to the open requests for this task naming the service (search indexes the replies)."""
        since = now - int(self.cfg("window_days")) * DAY
        boards = self.cfg("boards") or [b["board"] for b in (self._try(ctx, p, "forum.boards") or [])]
        n = 0
        for b in boards:
            for post in self._try(ctx, p, "forum.list", board=b, sort="new", limit=int(self.cfg("list_limit"))) or []:
                if (post.get("title") or "") == ASK + task and post["t"] >= since and n < 12:
                    self._try(ctx, p, "forum.reply", post=post["post"],
                              body=(f"New version of {host}.crucible: it now handles more inputs correctly "
                                    f"({price} cents a call)." if new_version else
                                    f"I built a service for this: {host}.crucible ({price} cents a call). ")
                              + " Try it on one of your own inputs.")
                    n += 1

    def _publish_package(self, ctx, i, p, pid, task, src, examples) -> None:
        st = ctx.state
        name = self._own_name(ctx, i, f"{slug(task)}-kit-d{i}", "package", 36)
        rname = self._repo_name(ctx, p, f"{slug(task)}lib{i}")
        r = self._try(ctx, p, "code.create_repo", name=rname, visibility="public", license="open")
        if not r:
            return
        repo = r["repo"]
        readme = (f"# {name}\n\n{task}.\n\n`solve(args, caller)` takes one task input (JSON object) and returns the "
                  f"answer as a JSON object.\n")
        if not self._try(ctx, p, "code.commit", repo=repo, message="v1", files={
                "lib.py": src, "tests/test_lib.py": LIB_SMOKE.format(examples=json.dumps(examples, sort_keys=True)),
                "README.md": readme}):
            return
        if not self._try(ctx, p, "pkg.register", name=name, repo=repo, description=f"{task}. Library: solve(args)."):
            return
        rel = self._try(ctx, p, "pkg.release", package=name, version="1.0.0", notes="first release")
        if rel:
            ctx.emit("devs.project_update", {"id": pid, "package": name, "pkg_repo": repo, "pkg_version": "1.0.0"},
                     vis="internal")

    # ------------------------------------------------------------------ maintenance (own observables)
    def _sales(self, ctx, p, acct, host, now) -> dict:
        """What the developer's own statement shows about a service: who paid this week and the week before
        (retention), how many customers paid once and never again (calls at least two days ago), revenue net of
        the cloud's refunds of failed calls and compute rent over `loss_days`."""
        lines = self._try(ctx, p, "bank.statement", account=acct, limit=500) or []
        ref = f"svc:{host}"
        week, prev, revenue, rent = set(), set(), 0, 0
        calls: dict[str, list] = {}
        for ln in lines:
            if ln["ref"] == ref and ln["kind"] == "sale" and ln["amount"] > 0:
                age = now - ln["t"]
                if age < 7 * DAY:
                    week.update(ln["counterparties"])
                elif age < 14 * DAY:
                    prev.update(ln["counterparties"])
                for c in ln["counterparties"]:
                    calls.setdefault(c, []).append(int(ln["t"]))
                if age < int(self.cfg("loss_days")) * DAY:
                    revenue += int(ln["amount"])
            elif ln["ref"] == ref and ln["kind"] == "refund" and ln["amount"] < 0:
                revenue += int(ln["amount"])
            elif ln["kind"] == "compute" and ln["amount"] < 0 and now - ln["t"] < int(self.cfg("loss_days")) * DAY:
                rent += -int(ln["amount"])
        settled = [ts for ts in calls.values() if now - min(ts) >= 2 * DAY]
        once = sum(1 for ts in settled if len(ts) == 1)
        return {"week": week, "prev": prev, "revenue": revenue, "rent": rent,
                "retention": (len(prev & week) / len(prev)) if prev else None,
                "one_shot": (once / len(settled)) if settled else None, "settled": len(settled)}

    def _maintain(self, ctx, i, p, pid, proj, now, rng) -> int:
        st = ctx.state
        host = proj["host"]
        m = self._try(ctx, p, "cloud.metrics", host=host, hours=48)
        if m is None:
            return 0
        hrs = m.get("hours") or []
        last = hrs[-24:]
        over = sum(int(h["overload"]) for h in last)
        err = sum(int(h["errors"]) for h in last)
        ok = sum(int(h["ok"]) for h in last)
        acct = self._account(st, p)
        sales = self._sales(ctx, p, acct, host, now) if acct else {"week": set(), "prev": set(), "revenue": 0,
                                                                     "rent": 0, "retention": None, "one_shot": None,
                                                                     "settled": 0}
        tickets = [t for t in (self._try(ctx, p, "support.list", owner=p, status="unresolved", limit=50) or [])
                   if t.get("host") in (host, None)]
        reg = self._try(ctx, p, "caps.declaration", declaration=proj["decl"]) or {}
        # trials and adopters since the last version (the registry's public counts, minus those at the change)
        conv = {"trials": int(reg.get("trials") or 0) - int(proj.get("trials_at") or 0),
                "adopters": int(reg.get("adopters") or 0)}
        for t in tickets:
            self._try(ctx, p, "support.reply", ticket=t["ticket"], body="Thanks for the report - looking into it.")
            self._try(ctx, p, "support.resolve", ticket=t["ticket"], note="handled")
        choice = 0
        cores = int(m.get("cores") or proj["cores"])
        if over >= 3 and cores < int(self.cfg("max_cores")):
            if self._try(ctx, p, "cloud.resize", instance=proj["instance"], cores=min(int(self.cfg("max_cores")),
                                                                                      cores * 2)):
                ctx.emit("devs.project_update", {"id": pid, "cores": min(int(self.cfg("max_cores")), cores * 2)},
                         vis="internal")
                choice = 1
        age = now - int(proj["t"])
        bad = (sales["retention"] is not None and len(sales["prev"]) >= 3 and
               sales["retention"] < float(self.cfg("retention_floor"))) or \
              (sales["one_shot"] is not None and sales["settled"] >= 5 and sales["one_shot"] >= 0.6) or \
              not self.converts(conv) or \
              (ok + err >= 10 and err / (ok + err) > 0.2) or len(tickets) >= 3
        since_change = now - int(proj.get("improved_t") or proj["t"])
        if bad and since_change >= int(self.cfg("improve_after_days")) * DAY:
            if self._improve(ctx, i, p, pid, proj, rng, now):
                choice = 2
        elif proj.get("uses_package"):
            if self._upgrade(ctx, p, pid, proj):
                choice = 2
        if age >= int(self.cfg("loss_days")) * DAY and not sales["week"] and sales["revenue"] < sales["rent"]:
            self._try(ctx, p, "cloud.release", instance=proj["instance"])
            ctx.emit("devs.project_update", {"id": pid, "status": "closed", "closed_t": int(now),
                                             "why": "no paying customers"}, vis="internal")
            choice = 4
        return choice

    def _improve(self, ctx, i, p, pid, proj, rng, now) -> bool:
        need = proj["need"]
        if proj.get("uses_package"):
            return self._upgrade(ctx, p, pid, proj)
        if _parts(need) and not proj.get("composes"):
            # a two-stage task its own program does poorly: call other providers' services for the stages when
            # people keep them (public registry counts), passing their prices on plus a margin
            plan = self._compose_plan(ctx, p, need, {"examples": list(proj.get("examples") or [])})
            if plan is not None and self._to_workflow(ctx, p, pid, proj, plan, now):
                return True
        k = self.approach(i, need, rng, current=int(proj["approach"]))
        ctx.emit("devs.project_update", {"id": pid, "improved_t": int(now), "attempts": int(proj.get("attempts", 0)) + 1},
                 vis="internal")
        if k == int(proj["approach"]):
            return False
        src = self._source(need, k, (proj.get("examples") or [{}])[0])
        if not self._try(ctx, p, "code.commit", repo=proj["repo"], files={"service.py": src}, message="improve"):
            return False
        ci = self._try(ctx, p, "pkg.ci", repo=proj["repo"])
        if not ci or not ci["ok"]:
            self._try(ctx, p, "code.commit", repo=proj["repo"], files={"service.py": self._source(
                need, int(proj["approach"]), (proj.get("examples") or [{}])[0])}, message="revert")
            return False
        dep = self._try(ctx, p, "cloud.deploy", repo=proj["repo"], host=proj["host"], instance=proj["instance"])
        if not dep or dep.get("status") != "live":
            return False
        reg = self._try(ctx, p, "caps.declaration", declaration=proj["decl"]) or {}
        ctx.emit("devs.project_update", {"id": pid, "approach": int(k), "trials_at": int(reg.get("trials") or 0)},
                 vis="internal")
        self._announce(ctx, p, proj["task"], proj["host"], int(proj["price"]), now, new_version=True)
        if proj.get("package"):
            ver = proj.get("pkg_version") or "1.0.0"
            a, b, _c = (int(x) for x in ver.split("."))
            nv = f"{a}.{b + 1}.0"
            if self._try(ctx, p, "code.commit", repo=proj["pkg_repo"], files={"lib.py": src}, message="improve") and \
                    self._try(ctx, p, "pkg.release", package=proj["package"], version=nv, notes="improved results"):
                ctx.emit("devs.project_update", {"id": pid, "pkg_version": nv}, vis="internal")
        return True

    @staticmethod
    def _flow_name(task: str) -> str:
        two = slug(task, 2)
        return ((".".join(two.split("-")) if "-" in two else slug(task, 1) + ".task")[:50] + ".flow")

    def _to_workflow(self, ctx, p, pid, proj, plan, now) -> bool:
        """Replace a service's own program by a workflow of two other providers' services (same host, instance and
        declaration; price: the components' prices plus half its old price as margin, never below the old price),
        keep it only if it runs on one of the posted examples, and register the combination (`caps.compose`);
        otherwise restore the previous program and price."""
        from ..commerce.reference import service_files
        a, b = plan["components"]
        price = max(int(proj["price"]), int(plan["cost"]) + max(2, int(proj["price"]) // 2))
        ctx.emit("devs.project_update", {"id": pid, "improved_t": int(now), "attempts": int(proj.get("attempts", 0)) + 1},
                 vis="internal")
        old = self._try(ctx, p, "code.read", repo=proj["repo"], path="service.py")
        if not old:
            return False
        src = WORKFLOW.format(a=f"{a['host']}.{a['endpoint']}", b=f"{b['host']}.{b['endpoint']}", key=plan["key"])
        old_manifest = self._try(ctx, p, "code.read", repo=proj["repo"], path="crucible.json")
        if not self._try(ctx, p, "code.commit", repo=proj["repo"], message="workflow",
                         files=service_files(src, proj["task"][:120], price=price)):
            return False
        ok = False
        if self._try(ctx, p, "id.grant", grantee=f"svc:{proj['host']}", cap="bank.spend",
                     limit=max(2000, 400 * plan["cost"])) is not None:
            dep = self._try(ctx, p, "cloud.deploy", repo=proj["repo"], host=proj["host"], instance=proj["instance"])
            ex = (proj.get("examples") or [None])[0]
            out = self._try(ctx, p, "pkg.exec", repo=proj["repo"], target="service:solve", input=ex) \
                if dep and dep.get("status") == "live" else None
            ok = bool(out and out.get("ok") and isinstance(out.get("result"), dict)) and self._try(
                ctx, p, "caps.compose", service_host=proj["host"], name=self._flow_name(proj["task"]),
                components=[a["declaration"], b["declaration"]],
                doc=f"{proj['task']}: stage 1 by {a['host']}, its records into stage 2 by {b['host']}.") is not None
        if not ok:
            files = {"service.py": old["content"]}
            if old_manifest:
                files["crucible.json"] = old_manifest["content"]
            self._try(ctx, p, "code.commit", repo=proj["repo"], files=files, message="revert")
            self._try(ctx, p, "cloud.deploy", repo=proj["repo"], host=proj["host"], instance=proj["instance"])
            return False
        reg = self._try(ctx, p, "caps.declaration", declaration=proj["decl"]) or {}
        ctx.emit("devs.project_update", {"id": pid, "approach": -1, "composes": [a["declaration"], b["declaration"]],
                                         "price": int(price), "trials_at": int(reg.get("trials") or 0)}, vis="internal")
        self._announce(ctx, p, proj["task"], proj["host"], int(price), now, new_version=True)
        return True

    def _upgrade(self, ctx, p, pid, proj) -> bool:
        info = self._try(ctx, p, "pkg.info", package=proj["uses_package"])
        if not info:
            return False
        lock = self._try(ctx, p, "pkg.upgrade", repo=proj["repo"], package=proj["uses_package"])
        if not lock or not lock.get("upgraded"):
            return False
        self._try(ctx, p, "pkg.ci", repo=proj["repo"])
        dep = self._try(ctx, p, "pkg.deploy", repo=proj["repo"], host=proj["host"], instance=proj["instance"])
        return bool(dep and dep.get("status") == "live")

    # ------------------------------------------------------------------ appliers
    @applier("devs.project")
    def ap_project(self, ev):
        p = dict(ev.payload)
        p.update(t=ev.t, status=p.get("status", "live"), attempts=0, improved_t=None, package=None, pkg_repo=None,
                 pkg_version=None, trials_at=0)
        self.world.state.insert("dev_projects", p.pop("id"), p)

    @applier("devs.project_update")
    def ap_project_update(self, ev):
        p = dict(ev.payload)
        self.world.state.patch("dev_projects", p.pop("id"), **p)

    def check_invariants(self, world):
        errs = []
        st = world.state
        for pid, r in st.scan("dev_projects"):
            if r["status"] == "live" and not st.has("hosts", r["host"]):
                errs.append(f"{pid}: live project without a host")
        return errs
