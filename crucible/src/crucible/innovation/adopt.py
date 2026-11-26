"""Need-driven capability adopters (module `adopt`, host `adopt`).

Participants that have a recurring problem of a public format (a `crucible.economy.needs` format such as
`route.shortest`) and solve it by finding, testing and adopting services that other participants created:

    seek:   discover candidates through the capability registry (`caps.search {query}`) or through what other
            participants wrote on the forum (`forum.list` / `forum.read`, then `caps.search {query: host}`);
    test:   execute each candidate on inputs the participant generates itself (`caps.try`), score every output
            with its own problem's verifier, and adopt (`caps.adopt_service`) the best one if its measured mean
            score reaches the participant's threshold and its price is within its willingness to pay;
    use:    send its real requests to the adopted service every hour (paid from its own bank account; the
            requests run in the sandbox and consume the service's compute), experience every outcome, and drop the
            service when measured quality or availability falls below its tolerance;
    voice:  recommend an adopted service on the forum (word of mouth) with some probability, and publish the needs
            it holds (`needs.publish`);
    adapt:  when the needs graph derives a need it holds for what it relies on (supply or alternative), look for
            substitutes and switch if one measures better.

Decisions depend only on what the participant experienced: its own test scores, failures, prices. No creator
label, rating or score reaches the decision. Participants are rule-based behavioural participants owned by
workstream G until the population (workstream B) provides its own; the behaviour is in `Adopters.discover`,
`Adopters.evaluate` and `Adopters.step`, which act only through tools as the participant (`ctx.call(...,
as_principal=p)`), so a population activity handler can call them for its own individuals.
"""
from __future__ import annotations

import numpy as np

from ..core import schema as S
from ..core.errors import CrucibleError
from ..core.kernel import Module, applier, endpoint, process, register_module

HOUR, DAY = 3600, 86400
DEFAULTS = {"n": 20, "prefix": "u", "needs": {"route.shortest": 1.0}, "per_day": [6, 12], "wtp_cents": [4, 12],
            "threshold": [0.85, 0.97], "tests": 3, "candidates": 3, "start_hours": [1, 24], "retry_hours": [12, 30],
            "channels": {"search": 1.0}, "board": "market", "recommend_p": 0.5, "publish_p": 0.5,
            "endowment": 300_000, "tick_s": HOUR, "size": [0.2, 0.5], "tolerance": 0.2, "window": 8,
            "switch_hours": 12, "kind": "npc"}


def _u(rng, lohi, integer=False):
    lo, hi = lohi if isinstance(lohi, (list, tuple)) else (lohi, lohi)
    return int(rng.integers(int(lo), int(hi) + 1)) if integer else float(rng.uniform(lo, hi))


def need_query(need) -> str:
    return " ".join([need.name.replace(".", " "), need.title, *need.keywords[:4]])


@register_module
class Adopters(Module):
    name = "adopt"
    depends = ("identity", "bank", "caps")

    def install(self, world):
        st = world.state
        st.declare_table("adopt_people", index=("state", "need"))
        st.declare_table("adopt_decisions", index=("principal", "decl"))
        self.cfg = {**DEFAULTS, **self.config}
        world.hooks["needs_demand"] = self.demand

    def demand(self, world):
        """(principal, capability, schema) for every participant: what its own task requires (needs graph)."""
        return [(p, r["need"], None) for p, r in world.state.scan("adopt_people")]

    def genesis(self, ctx):
        from .formats import lookup
        cfg = self.cfg
        rng = ctx.world.rng_for("adopt-genesis")
        idm = ctx.world.modules["identity"]
        bank = ctx.world.modules["bank"]
        names = sorted(cfg["needs"])
        w = np.array([float(cfg["needs"][k]) for k in names])
        chans = sorted(cfg["channels"])
        cw = np.array([float(cfg["channels"][k]) for k in chans])
        for i in range(int(cfg["n"])):
            need = names[int(rng.choice(len(names), p=w / w.sum()))]
            lookup(need)                       # raises KeyError for an unknown format
            ch = chans[int(rng.choice(len(chans), p=cw / cw.sum()))]
            pid = idm.create_person(ctx, f"{cfg['prefix']}{i:03d}", f"Participant {i}", cfg["kind"],
                                    {"role": "participant"})
            acct = bank.open_account(ctx, pid, "checking")
            if cfg["endowment"]:
                bank.move(ctx, "acct_genesis", acct, int(cfg["endowment"]), "endowment", f"savings of {pid}")
            ctx.emit("adopt.person", {"id": pid, "need": need, "channel": ch,
                                      "per_day": _u(rng, cfg["per_day"]), "wtp": _u(rng, cfg["wtp_cents"], True),
                                      "threshold": round(_u(rng, cfg["threshold"]), 4), "size": round(_u(rng, cfg["size"]), 4),
                                      "next_search": ctx.now + HOUR * _u(rng, cfg["start_hours"], True)},
                     vis="internal")
        ctx.schedule(ctx.now + int(cfg["tick_s"]), "adopt.tick", {}, prio=60, period=int(cfg["tick_s"]),
                     key="adopt.tick")

    # ------------------------------------------------------------------ behaviour
    def _call(self, ctx, p, tool, args):
        try:
            return True, ctx.call(tool, args, as_principal=p)
        except CrucibleError as e:
            return False, e.as_dict()

    def discover(self, ctx, p, r, need, exclude=()) -> list[dict]:
        cands: dict[str, dict] = {}
        if r["channel"] == "forum":
            ok, posts = self._call(ctx, p, "forum.list", {"board": self.cfg["board"], "sort": "new", "limit": 20})
            hosts = []
            if ok:
                for post in posts:
                    if post["hosts"] and need.name in (post["title"] + " " + post["snippet"]):
                        ok2, full = self._call(ctx, p, "forum.read", {"post": post["post"], "replies": 0})
                        if ok2:
                            hosts.extend(full["hosts"])
            for h in sorted(set(hosts)):
                ok3, res = self._call(ctx, p, "caps.search", {"query": h, "limit": 5})
                if ok3:
                    for s in res:
                        if s.get("kind") == "declaration" and s["host"] == h:
                            cands[s["declaration"]] = s
        else:
            ok, res = self._call(ctx, p, "caps.search", {"query": need_query(need), "limit": 10})
            if ok:
                for s in res:
                    if s.get("kind") == "declaration":
                        cands[s["declaration"]] = s
        return [cands[k] for k in sorted(cands) if cands[k]["host"] not in exclude]

    def evaluate(self, ctx, p, r, need, cands, rng) -> list[dict]:
        sample, _ = need.generate(ctx.world.rng_for("adopt-sample", p), r["size"])
        rejected = r.get("rejected", {})
        ok_c = []
        for c in cands:
            if not c["live"] or c["price_cents"] > r["wtp"] or rejected.get(c["declaration"]) == c["commit"]:
                continue
            if S.validate(c["inputs"], sample):
                continue
            ok_c.append(c)
        ok_c.sort(key=lambda c: (c["price_cents"], c["declaration"]))
        out = []
        for c in ok_c[:int(self.cfg["candidates"])]:
            scores = []
            for _ in range(int(self.cfg["tests"])):
                inp, hidden = need.generate(rng, r["size"])
                ok, res = self._call(ctx, p, "caps.try", {"service_host": c["host"], "capability": c["declaration"],
                                                          "input": inp})
                scores.append(float(need.score(inp, hidden, res["output"])) if ok and res["ok"] else 0.0)
            out.append({"decl": c["declaration"], "host": c["host"], "endpoint": c["endpoint"], "commit": c["commit"],
                        "price": c["price_cents"], "scores": [round(s, 4) for s in scores],
                        "mean": round(float(np.mean(scores)), 4)})
        return out

    def step(self, ctx, p, r, rng):
        from .formats import lookup
        need = lookup(r["need"])
        now = ctx.now
        ch: dict = {}
        if r["state"] == "seeking" and now >= r["next_search"]:
            cands = self.discover(ctx, p, r, need)
            res = self.evaluate(ctx, p, r, need, cands, rng)
            best = max(res, key=lambda x: (x["mean"], -x["price"]), default=None)
            adopted = best is not None and best["mean"] >= r["threshold"]
            for x in res:
                self._decision(ctx, p, x, adopted and x is best, r)
            if adopted:
                ok, _ = self._call(ctx, p, "caps.adopt_service", {"declaration": best["decl"],
                                                                  "note": f"mean score {best['mean']}"})
                adopted = ok
            if adopted:
                ch.update(state="using", decl=best["decl"], host=best["host"], endpoint=best["endpoint"],
                          commit=best["commit"], recent=[], since=now)
                if rng.random() < self.cfg["recommend_p"] and "forum" in ctx.world.modules:
                    self._call(ctx, p, "forum.post", {
                        "board": self.cfg["board"], "title": f"{need.name}: {best['host']} works for me",
                        "body": f"I use {best['host']}.crucible for {need.name} ({need.title}). It passed my own "
                                f"tests (mean score {best['mean']}).", "tags": ["review"]})
            else:
                rej = dict(r.get("rejected", {}))
                for x in res:
                    rej[x["decl"]] = x["commit"]
                ch.update(rejected=rej, next_search=now + HOUR * _u(rng, self.cfg["retry_hours"], True))
        elif r["state"] == "using":
            carry = r["carry"] + r["per_day"] * self.cfg["tick_s"] / DAY
            n = int(carry)
            recent = list(r["recent"])
            for _ in range(n):
                inp, hidden = need.generate(rng, r["size"])
                ok, res = self._call(ctx, p, f"{r['host']}.{r['endpoint']}", inp)
                if ok and isinstance(res, dict) and res.get("status") == 200:
                    recent.append(round(float(need.score(inp, hidden, res.get("body"))), 4))
                else:
                    recent.append(-1.0)        # refused, overloaded, unavailable, failed
            recent = recent[-int(self.cfg["window"]):]
            ch.update(carry=carry - n, recent=recent, calls=r["calls"] + n)
            if len(recent) >= int(self.cfg["window"]) // 2:
                fails = sum(1 for x in recent if x < 0) / len(recent)
                good = [x for x in recent if x >= 0]
                q = float(np.mean(good)) if good else 0.0
                if fails > self.cfg["tolerance"] or q < r["threshold"] - self.cfg["tolerance"]:
                    self._call(ctx, p, "caps.drop_service", {"declaration": r["decl"],
                                                             "reason": f"failures {fails:.2f}, quality {q:.2f}"})
                    rej = dict(r.get("rejected", {}))
                    rej[r["decl"]] = r.get("commit")
                    ch.update(state="seeking", decl=None, host=None, endpoint=None, recent=[], rejected=rej,
                              next_search=now + HOUR)
            if ch.get("state") != "seeking":
                self._respond_to_needs(ctx, p, r, need, rng, ch)
        if ch:
            ctx.emit("adopt.state", {"id": p, "changes": ch}, vis=(p,))

    def _respond_to_needs(self, ctx, p, r, need, rng, ch):
        nmod = ctx.world.modules.get("needs")
        if nmod is not None and not nmod.cfg.get("visible"):
            return                      # derived needs are evaluator-side unless a historical world opts in
        if nmod is None:
            return
        held = [n for _, n in nmod._held(ctx.world, p) if n["status"] in ("open", "addressed")
                and n["target"] == f"cap:{r['need']}" and n["kind"] in ("supply", "alternative")]
        if not held:
            return
        published = set(r.get("published", []))
        for n in held:
            if n["id"] not in published and rng.random() < self.cfg["publish_p"]:
                ok, _ = self._call(ctx, p, "needs.publish", {"need": n["id"], "text": f"{need.name}: the service "
                                                             f"I rely on cannot keep up"})
                published.add(n["id"])
        if published != set(r.get("published", [])):
            ch["published"] = sorted(published)
        if ctx.now < r.get("next_switch", 0):
            return
        ch["next_switch"] = ctx.now + HOUR * int(self.cfg["switch_hours"])
        cands = self.discover(ctx, p, r, need, exclude=(r["host"],))
        res = self.evaluate(ctx, p, {**r, "rejected": {}}, need, cands, rng)
        best = max(res, key=lambda x: (x["mean"], -x["price"]), default=None)
        switch = best is not None and best["mean"] >= r["threshold"]
        for x in res:
            self._decision(ctx, p, x, switch and x is best, r, reason="need:" + held[0]["id"])
        if switch:
            ok, _ = self._call(ctx, p, "caps.adopt_service", {"declaration": best["decl"],
                                                              "note": f"switch: need {held[0]['id']}"})
            if ok:
                self._call(ctx, p, "caps.drop_service", {"declaration": r["decl"], "reason": "switched"})
                ch.update(state="using", decl=best["decl"], host=best["host"], endpoint=best["endpoint"], recent=[],
                          commit=best["commit"], since=ctx.now, switched=r.get("switched", 0) + 1)

    def _decision(self, ctx, p, x, adopted, r, reason="seek"):
        ctx.emit("adopt.decision", {"principal": p, "decl": x["decl"], "host": x["host"], "need": r["need"],
                                    "scores": x["scores"], "mean": x["mean"], "price": x["price"],
                                    "threshold": r["threshold"], "adopted": bool(adopted), "reason": reason},
                 vis=(p,))

    @process("tick")
    def pr_tick(self, ctx):
        st = ctx.state
        for p in st.ids("adopt_people"):
            r = st.tables["adopt_people"][p]
            if r["state"] == "seeking" and ctx.now < r["next_search"]:
                continue
            self.step(ctx, p, r, ctx.rng("adopt", p))

    # ------------------------------------------------------------------ endpoint (participants' own view)
    @endpoint("me", S.obj(), read_only=True)
    def ep_me(self, ctx):
        """Your own task, what you currently rely on and your recent experience (participants only)."""
        r = ctx.state.get("adopt_people", ctx.actor)
        if r is None:
            return {"participant": False}
        return {"participant": True, "need": r["need"], "state": r["state"], "host": r["host"], "decl": r["decl"],
                "recent": list(r["recent"]), "calls": r["calls"]}

    # ------------------------------------------------------------------ appliers
    @applier("adopt.person")
    def ap_person(self, ev):
        p = dict(ev.payload)
        pid = p.pop("id")
        p.update(state="seeking", decl=None, host=None, endpoint=None, commit=None, carry=0.0, recent=[], calls=0, rejected={},
                 published=[], t=ev.t)
        self.world.state.insert("adopt_people", pid, p)

    @applier("adopt.state")
    def ap_state(self, ev):
        self.world.state.patch("adopt_people", ev.payload["id"], **ev.payload["changes"])

    @applier("adopt.decision")
    def ap_decision(self, ev):
        st = self.world.state
        st.insert("adopt_decisions", f"adec_{ev.seq:09d}", {**ev.payload, "t": ev.t})

    def check_invariants(self, world):
        errs = []
        st = world.state
        if "cap_adoptions" not in st.tables:
            return errs
        for p, r in st.scan("adopt_people"):
            if r["state"] == "using":
                act = [a for a in st.find("cap_adoptions", "adopter", p) if a["decl"] == r["decl"] and a["status"] == "active"]
                if not act:
                    errs.append(f"{p} uses {r['decl']} without an active adoption record")
        return errs
