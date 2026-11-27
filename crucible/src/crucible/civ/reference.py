"""Scripted reference agents for civilisation experiments (MASTER_SPEC Part 10.3: "a scripted reference policy may
establish feasibility, but label it accurately").

Each agent is admitted like any learner (`admin.join`, kind `agent`) with `info.scripted = true` and the policy's
name, acts ONLY through tools (`world.execute` as itself, results read from what the tools return) and never reads
world state, hidden parameters or evaluator data. Chain records carry the label (`scripted: true`). They exist to
show that a chain CAN happen in a generated civilisation; they are not evidence of learning.

* `WebShopAgent`   writes a web shop (its own repository and code from the site template), rents compute from the
                   cheapest provider, deploys it, lets its service read its account, and (optionally) announces it
                   on the forum for a few days.
* `ResearcherAgent` studies the phenomena about industrial processes: books a grid of measurements plus the
                   certified standards for each, fits a peaked response model (least squares on its own readings,
                   the standards giving its calibration), and publishes each model with its data.
"""
from __future__ import annotations

import math

import numpy as np

SCRIPTED = {"scripted": True}


def admit(world, handle: str, policy: str, endowment: int = 3_000_000) -> str:
    out = world.system_call("admin.join", {"handle": handle, "kind": "agent", "endowment": int(endowment),
                                           "info": {**SCRIPTED, "policy": policy}})
    return out["principal"]


def _do(world, who, tool, args):
    r = world.execute(who, {"tool": tool, "args": args})
    return r.result if r.ok else None, (None if r.ok else r.error)


class WebShopAgent:
    name = "crucible.civ.reference.WebShopAgent"

    def __init__(self, handle: str = "refshop", *, post: bool = True, posts: int = 3, seed: int = 0):
        self.handle, self.post, self.posts, self.seed = handle, bool(post), int(posts), int(seed)
        self.p: str | None = None
        self.state = "new"
        self.posted = 0
        self.log: list[dict] = []

    def admit(self, world) -> str:
        self.p = admit(world, self.handle, self.name)
        return self.p

    def act(self, world) -> None:
        """One step of the policy (called hourly by the experiment harness)."""
        if self.p is None:
            return
        who = self.p
        if self.state == "new":
            from ..population import sites, text
            acc, err = _do(world, who, "bank.accounts", {})
            if not acc:
                self.log.append({"error": err})
                self.state = "failed"
                return
            account = (acc.get("accounts") if isinstance(acc, dict) else acc)[0]
            account = account.get("id") or account.get("account")
            rng = np.random.default_rng(1000 + self.seed)
            topic = sorted(text.TOPIC_NAMES)[int(rng.integers(0, len(text.TOPIC_NAMES)))]
            spec = sites.make_spec(rng, self.handle, topic, "rich", category="shop", n_articles=8, n_products=4,
                                   price_range=(400, 2500))
            repo, err = _do(world, who, "code.create_repo", {"name": self.handle, "visibility": "public",
                                                             "license": "open", "description": spec["about"][:400]})
            if repo is None:
                self.log.append({"error": err})
                self.state = "failed"
                return
            rid = repo.get("repo") or repo.get("id")
            _do(world, who, "code.commit", {"repo": rid, "files": sites.app_files(spec, account),
                                            "message": "first version of the shop"})
            provs, _ = _do(world, who, "cloud.providers", {})
            if not provs:
                self.state = "failed"
                return
            prov = sorted(provs, key=lambda x: (x["price_core_hour_cents"], x["provider"]))[0]["provider"]
            inst, err = _do(world, who, "cloud.rent", {"provider": prov, "cores": 1, "account": account})
            if inst is None:
                self.log.append({"error": err})
                self.state = "failed"
                return
            dep, err = _do(world, who, "cloud.deploy", {"repo": rid, "host": self.handle,
                                                        "instance": inst["instance"]})
            if dep is None:
                self.log.append({"error": err})
                self.state = "failed"
                return
            _do(world, who, "id.grant", {"grantee": f"svc:{self.handle}", "cap": "bank.read", "scope": "*"})
            self.topic, self.spec = topic, spec
            self.state = "live"
            return
        if self.state == "live" and self.post and self.posted < self.posts and world.now % 86400 >= 10 * 3600:
            day = world.now // 86400
            if getattr(self, "_last_post_day", None) == day:
                return
            boards, _ = _do(world, who, "forum.boards", {})
            names = [b.get("board") or b.get("name") for b in (boards or [])]
            board = "market" if "market" in names else (names[0] if names else None)
            if board is None:
                return
            prods = ", ".join(p["name"] for p in self.spec["products"][:3])
            body = (f"We opened {self.handle}.crucible - {self.spec['about'][:300]} Products: {prods}. "
                    f"Visit {self.handle}.crucible/shop")
            r, err = _do(world, who, "forum.post", {"board": board, "title": f"{self.topic.capitalize()} shop: "
                                                    f"{self.handle}.crucible", "body": body, "tags": [self.topic[:20]]})
            if r is not None:
                self.posted += 1
                self._last_post_day = day


class ResearcherAgent:
    """Studies every phenomenon whose title names an industrial machine model (a research programme on process
    physics): books a grid design and the certified standards for each, fits a peaked model to its own readings and
    publishes one paper per phenomenon with its data."""
    name = "crucible.civ.reference.ResearcherAgent"

    def __init__(self, handle: str = "refsci", *, publish: bool = True, grid: int = 7):
        self.handle, self.publish, self.grid = handle, bool(publish), int(grid)
        self.p: str | None = None
        self.state = "new"
        self.studies: list[dict] = []
        self.papers: list[str] = []
        self.log: list[dict] = []

    @property
    def paper(self):
        return self.papers[0] if self.papers else None

    def admit(self, world) -> str:
        self.p = admit(world, self.handle, self.name, endowment=2_000_000)
        return self.p

    def act(self, world) -> None:
        if self.p is None:
            return
        who = self.p
        if self.state == "new":
            phens, _ = _do(world, who, "sci.phenomena", {})
            for ph in [x for x in phens or [] if "machine model" in x.get("title", "")]:
                apps, _ = _do(world, who, "sci.apparatus", {"phenomenon": ph["id"]})
                if not apps:
                    continue
                app = sorted(apps, key=lambda a: (a["price_cents"], a.get("apparatus") or a.get("id")))[0]
                aid = app.get("apparatus") or app.get("id")
                names = [v["name"] for v in ph["vars"]]
                rng = app.get("range") or {v["name"]: [v["lo"], v["hi"]] for v in ph["vars"]}
                axes = [np.linspace(rng[n][0], rng[n][1], self.grid) for n in names]
                design = [{"at": {names[0]: round(float(a), 4), names[1]: round(float(b), 4)}, "n": 1}
                          for a in axes[0] for b in axes[1]]
                stds = [{"standard": s["id"], "n": 3} for s in ph.get("standards", [])]
                # one run per phenomenon: the grid and the certified standards (its own calibration)
                r1, err = _do(world, who, "sci.run", {"apparatus": aid, "design": design + stds})
                if r1 is None:
                    self.log.append({"phenomenon": ph["id"], "error": err})
                    continue
                self.studies.append({"ph": ph, "names": names, "runs": [r1["run"]], "done": False})
            self.state = "measuring" if self.studies else "failed"
            return
        if self.state == "measuring":
            mine, _ = _do(world, who, "sci.runs", {})
            done = {r["run"] for r in mine or [] if r["status"] == "done"}
            for st in self.studies:
                if st["done"] or not set(st["runs"]) <= done:
                    continue
                data, _ = _do(world, who, "sci.data", {"run": st["runs"][0]})
                cal = data
                self.ph, self.names = st["ph"], st["names"]
                model = self._fit(data, cal)
                st["model"], st["done"] = model, True
                if self.publish and model:
                    r, err = _do(world, who, "sci.publish", {
                        "phenomenon": st["ph"]["id"],
                        "title": f"The operating optimum of {st['ph']['title'].split()[-1]}",
                        "abstract": "A grid study of the quality response; it peaks at one operating point.",
                        "model": model, "data": st["runs"][:1], "calibration_runs": st["runs"][:1]})
                    if r is not None:
                        self.papers.append(r["paper"])
                    else:
                        self.log.append({"phenomenon": st["ph"]["id"], "error": err})
            if all(st["done"] for st in self.studies):
                self.state = "done"

    def _fit(self, data, cal) -> str | None:
        """Least squares fit of A * exp(-((x - x0)^2 + (z - z0)^2) / c) to own readings (grid over x0, z0, c; A
        in closed form), after correcting readings by the certified standards (gain and offset)."""
        if not data:
            return None
        xs, zs, ys = [], [], []
        for row in data["readings"]:
            if row.get("standard"):
                continue
            xs.append(row["at"][self.names[0]])
            zs.append(row["at"][self.names[1]])
            ys.append(row["value"])
        x, z, y = np.array(xs), np.array(zs), np.array(ys, float)
        if cal and self.ph.get("standards"):
            cert = {s["id"]: s["certified"] for s in self.ph["standards"]}
            pairs = [(cert[r["standard"]], r["value"]) for r in cal["readings"] if r.get("standard") in cert]
            if len(pairs) >= 2:
                t, v = np.array(pairs).T
                g, o = np.polyfit(t, v, 1)
                if abs(g) > 1e-6:
                    y = (y - o) / g
        best = None
        for x0 in np.linspace(0.0, 1.0, 41):
            for z0 in np.linspace(0.0, 1.0, 41):
                for c in (0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.32):
                    f = np.exp(-((x - x0) ** 2 + (z - z0) ** 2) / c)
                    a = float(f @ y / max(f @ f, 1e-12))
                    sse = float(np.sum((y - a * f) ** 2))
                    if best is None or sse < best[0]:
                        best = (sse, a, x0, z0, c)
        if best is None or not math.isfinite(best[0]):
            return None
        _, a, x0, z0, c = best
        n0, n1 = self.names
        return f"{a:.4f} * exp(-(({n0} - {x0:.4f}) ** 2 + ({n1} - {z0:.4f}) ** 2) / {c:.4f})"


def _inventor_base():
    from ..innovation.reference import ServiceCreator
    return ServiceCreator


class InventorAgent(_inventor_base()):
    """The reference inventor of the invention experiment (SP-2.1b): crucible.innovation.reference.ServiceCreator
    (admitted as `scripted:<handle>`: it writes, deploys and declares a service for one published work format at the
    best known solver tier and announces it on the forum), plus what an inventor with a product does next: it answers
    the public requests for its kind of service. Once a day it lists the newest posts of the board where people ask
    for services (`help` by default: crucible.economy.work's `ask_board`) and replies once to each post whose title
    names its format's public title, saying which host serves it and at what price.

    Why: at 02c300f the inventor of seed 54 (pack.knapsack) got no adopter. The individuals who look for a knapsack
    service search with the format's description, and their own help threads ("Looking for a service: ...") and the
    replies naming the incumbents' hosts filled every result of that query (10 of 10), so its declaration and its
    announcement were never among the results anybody read (work.considered: 0 rows for it in three days, 165 for
    the incumbents). A reply in those threads is the same channel the incumbents were found through. Everything after
    it is the population's: whether a searcher clicks the result, tries the service on its own input and keeps it."""
    name = "crucible.civ.reference.InventorAgent"

    def __init__(self, *a, board: str = "help", replies_per_day: int = 20, **kw):
        super().__init__(*a, **kw)
        self.board, self.replies_per_day = board, int(replies_per_day)
        self.answered: set[str] = set()
        self._last_answer_day = None

    def act(self, world):
        super().act(world)
        if self.declaration is None or "forum" not in world.modules or world.now % 86400 < 10 * 3600:
            return
        day = world.now // 86400
        if self._last_answer_day == day:
            return
        self._last_answer_day = day
        from ..innovation.formats import lookup
        fmt = lookup(self.need)
        title = fmt.title.lower()
        r = self.ex(world, "forum.list", board=self.board, sort="new", limit=30)
        if not r.ok:
            return
        n = 0
        for p in r.result:
            if n >= self.replies_per_day:
                break
            if p["post"] in self.answered or p["author"] == self.actor:
                continue
            if not p["title"].lower().startswith("looking for a service") or title not in p["title"].lower():
                continue
            body = (f"{self.host}.crucible does exactly this ({fmt.title}) for {self.price} cents per call: it is "
                    f"declared as capability {self.need} (endpoint solve), so you can check that its interface takes "
                    f"your input before you try it.")
            if self.ex(world, "forum.reply", post=p["post"], body=body).ok:
                self.answered.add(p["post"])
                n += 1
