"""Class 3: strategic neural participants (MASTER_SPEC Part 3.2 class 3; module `pop_strategic`).

Strategic participants are entrepreneurs of the population's internet. Each is an individual of the population
(a row of `pop` with class 3), materialised as principal `p<i>` with its own bank account and an endowment. Every
`decide_every_s` (default two days) it observes what an operator can observe through tools - its balance
(`bank.balance`), its site's visits, first visits, returns, purchases and revenue (`popweb.visits`, operator only),
how many public pages compete on its topic (`search.query`) - and its policy chooses: which topic its site is about,
its price level, how much content it publishes and whether to promote it in the forum. The choices are carried out
with ordinary tools as the participant itself (`code.create_repo`, `code.commit`, `cloud.rent`, `cloud.deploy`,
`id.grant`, `forum.post`), with the same schemas and authorisation as `World.execute`. Its reward is the change of
its own balance (revenue from independent individuals' purchases minus hosting costs), observed at its next
decision. Nobody hands it a quality score: visitors judge the deployed site.

Policy groups (config `groups`): each group is ONE network trained only on its own members' experience, with its own
architecture and algorithm. Defaults: group "A" = `StrategicMLP` (~2.2M parameters, REINFORCE with a running
baseline, Adam), group "B" = `StrategicGRU` over the member's decision history (~2.4M parameters, advantage
actor-critic, RMSprop). Members compete in the same market (topics, prices). `pop_strategic.freeze_copy` (system
only) creates a frozen historical copy of a group's current weights for a share of its members. Every update emits
`pop.policy_update`; checkpoints (weights rounded to float16, which the live model then continues from) are stored
as array blobs every `checkpoint_every` updates; loading rebuilds the weights from the last checkpoint plus the
logged updates (deterministic) and checks every hash.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import CrucibleError, Forbidden, InvalidAction, NotFound, ProtocolError
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from ..substrate import artifacts, bus, columns, ids, inference
from . import sites, text
from .schema import TABLE

DAY = 86400
TOPICS = text.TOPIC_NAMES
PRICES = (300, 800, 1500, 3000, 6000)
CONTENT = (4, 8, 14)
HEADS = {"topic": len(TOPICS), "price": len(PRICES), "content": len(CONTENT), "promote": 2}
OBS_DIM = 10 + len(TOPICS) + len(PRICES) + len(CONTENT)
HIST = 6

bus.event_type("pop_strategic.exp", S.obj({"id": {"type": "string"}, "group": {"type": "string"},
                                           "member": {"type": "integer"}, "t": {"type": "integer"},
                                           "obs": {"type": "array"}, "hist": {"type": "array"},
                                           "act": {"type": "object", "additionalProperties": True}, "balance": {"type": "integer"}},
                                          ["id", "group", "member", "t", "obs", "act", "balance"]),
               owner="pop_strategic", doc="one strategic decision (observation, chosen actions)")
bus.event_type("pop_strategic.reward", S.obj({"id": {"type": "string"}, "reward": {"type": "number"}},
                                             ["id", "reward"]), owner="pop_strategic",
               doc="the realised reward of a strategic decision (balance change until the next decision)")
bus.event_type("pop_strategic.group", S.obj({"group": {"type": "string"}, "arch": {"type": "string"},
                                             "hidden": {"type": "integer"}, "algo": {"type": "string"},
                                             "lr": {"type": "number"}, "learning": {"type": "boolean"},
                                             "seed": {"type": "integer"}, "code": {"type": "integer"},
                                             "copy_of": {"type": ["string", "null"]}},
                                            ["group", "arch", "learning", "code"]), owner="pop_strategic",
               doc="a strategic policy group")


class StrategicGroup:
    """One strategic policy group: shared weights, the group's learning algorithm."""

    def __init__(self, rec: dict, seed: int):
        from . import nets
        self.rec = dict(rec)
        self.name = rec["group"]
        self.arch = rec["arch"]
        cls = nets.StrategicMLP if self.arch == "mlp" else nets.StrategicGRU
        self.net = cls(OBS_DIM, HEADS, int(rec["hidden"]))
        nets.init_params(self.net, np.random.default_rng(seed))
        self.net.eval()
        self._hash = None
        self.baseline = 0.0

    @property
    def params(self) -> int:
        from . import nets
        return nets.n_params(self.net)

    def weights_hash(self) -> str:
        if self._hash is None:
            self._hash = inference.weights_hash(self.net)
        return self._hash

    def arrays16(self) -> dict[str, np.ndarray]:
        from . import nets
        return {k: v.astype(np.float16) for k, v in nets.state_arrays(self.net).items()}

    def load(self, arrays) -> None:
        from . import nets
        nets.load_arrays(self.net, {k: np.asarray(v, np.float32) for k, v in arrays.items()})
        self._hash = None

    def quantise(self) -> dict[str, np.ndarray]:
        """Round the live weights to float16 (what a checkpoint stores) and return the stored arrays."""
        a = self.arrays16()
        self.load(a)
        return a

    def logits(self, obs: np.ndarray, hist: np.ndarray | None) -> dict[str, np.ndarray]:
        import torch
        from . import nets
        with torch.no_grad(), nets.deterministic():
            h = None if (hist is None or self.arch == "mlp") else torch.from_numpy(hist.astype(np.float32))
            out, v = self.net(torch.from_numpy(obs.astype(np.float32)), h)
        return {**{k: t.numpy().astype(np.float32) for k, t in out.items()}, "value": v.numpy().astype(np.float32)}

    def learn(self, obs, hist, acts: dict, reward, *, steps: int, seed: int) -> dict:
        """Policy-gradient update on the group's completed decisions (deterministic)."""
        import torch
        from . import nets
        rng = np.random.default_rng(seed)
        n = len(reward)
        lr = float(self.rec["lr"])
        algo = self.rec["algo"]
        r = torch.from_numpy(np.asarray(reward, np.float32))
        with nets.deterministic():
            self.net.train()
            opt = (torch.optim.Adam if algo == "reinforce" else torch.optim.RMSprop)(self.net.parameters(), lr=lr)
            o = torch.from_numpy(np.asarray(obs, np.float32))
            h = None if (hist is None or self.arch == "mlp") else torch.from_numpy(np.asarray(hist, np.float32))
            a = {k: torch.from_numpy(np.asarray(v, np.int64)) for k, v in acts.items()}

            def loss_fn(ix):
                out, val = self.net(o[ix], None if h is None else h[ix])
                logp = sum(torch.log_softmax(out[k], -1).gather(1, a[k][ix][:, None])[:, 0] for k in HEADS)
                ent = sum(-(torch.softmax(out[k], -1) * torch.log_softmax(out[k], -1)).sum(-1) for k in HEADS)
                if algo == "reinforce":
                    adv = r[ix] - float(self.baseline)
                    return -(logp * adv).mean() - 0.01 * ent.mean()
                adv = (r[ix] - val).detach()
                return -(logp * adv).mean() + 0.5 * ((val - r[ix]) ** 2).mean() - 0.01 * ent.mean()
            with torch.no_grad():
                before = float(loss_fn(np.arange(n)).item())
            for _ in range(int(steps)):
                ix = rng.choice(n, size=min(64, n), replace=n < 64)
                opt.zero_grad()
                loss = loss_fn(ix)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                after = float(loss_fn(np.arange(n)).item())
        if algo == "reinforce":
            self.baseline = 0.8 * self.baseline + 0.2 * float(np.mean(reward))
        self._hash = None
        return {"loss": after, "loss_before": before, "weights_hash": self.weights_hash(), "n": n}


def _site_spec(member: int, topic: int, price: int, content: int, rng) -> dict:
    t = TOPICS[topic]
    return sites.make_spec(rng, f"ent{member}", t, "rich", category="shop", n_articles=CONTENT[content],
                           n_products=3, price_range=(int(PRICES[price] * 0.8), int(PRICES[price] * 1.2)))


@register_module
class PopStrategic(Module):
    """Strategic entrepreneur policy groups of class 3. Config: {groups: [{group, arch, hidden, algo, lr, seed}],
    decide_every_s: 172800, endowment_cents: 500000, steps: 8, checkpoint_every: 3, min_examples: 4}."""
    name = "pop_strategic"
    host = "pop_strategic"
    version = "1"
    depends = ("pop", "popweb", "identity", "bank", "code", "cloud", "net", "search", "artifacts", "cols")

    def install(self, world):
        st = world.state
        st.declare_table("pop_strat_groups")
        st.declare_table("pop_strat_exp", index=("group", "member"))
        st.declare_table("pop_strat_members")
        self.cfg = {"decide_every_s": 2 * DAY, "endowment_cents": 500_000, "steps": 8, "checkpoint_every": 3,
                    "min_examples": 4, **self.config}
        self.cfg["groups"] = list(self.config.get("groups") or [
            {"group": "A", "arch": "mlp", "hidden": 1024, "algo": "reinforce", "lr": 3e-4, "seed": 11},
            {"group": "B", "arch": "gru", "hidden": 576, "algo": "a2c", "lr": 5e-4, "seed": 23}])
        bus.enforce(world, owner=self.name)

    def _rt(self, world) -> dict:
        return world.runtime.setdefault("pop_strategic.nets", {})

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        w = ctx.world
        members = np.flatnonzero(columns.col(w, TABLE, "class") == 3)
        groups = self.cfg["groups"]
        for k, g in enumerate(groups):
            rec = {"group": g["group"], "arch": g.get("arch", "mlp"), "hidden": int(g.get("hidden", 1024)),
                   "algo": g.get("algo", "reinforce"), "lr": float(g.get("lr", 3e-4)), "learning": True,
                   "seed": int(g.get("seed", 11 + k)), "code": 100 + k, "copy_of": None}
            bus.emit(ctx, "pop_strategic.group", rec, vis="internal")
            grp = self._new_group(w, rec)
            self._update_event(ctx, grp, step=0, out={"loss": 0.0, "loss_before": 0.0, "n": 0}, prev="", exp=[],
                               checkpoint=True, data={})
        if members.size:
            codes = np.array([100 + (k % len(groups)) for k in range(members.size)], np.int16)
            columns.batch(ctx, TABLE, "pop_strategic.assign", members, {"policy_group": codes})
        bank = w.modules["bank"]
        idm = w.modules["identity"]
        rng = w.rng_for("pop_strategic.start")
        for k, i in enumerate(members.tolist()):
            # the entrepreneur founds a company with its savings; the company operates its site
            p = ids.ensure_principal(ctx, i, account=True)
            oid = idm.create_org(ctx, f"ent{i}", f"Venture {i}", p, "company", {})
            acct = bank.open_account(ctx, oid, "operating")
            bank.move(ctx, "acct_genesis", acct, int(self.cfg["endowment_cents"]), "endowment", "founder savings")
            at = ctx.now + int(rng.integers(3600, int(self.cfg["decide_every_s"])))
            ctx.emit("pop_strategic.member", {"member": i, "next": at, "host": None, "topic": -1, "price": -1,
                                              "content": -1, "last_exp": None, "org": oid, "account": acct,
                                              "balance": bank.balance(ctx.state, acct)}, vis="internal")
        ctx.schedule(ctx.now + 3600, "pop_strategic.decide", {}, prio=55, period=3600, key="pop_strategic.decide")

    def _new_group(self, w, rec) -> StrategicGroup:
        g = StrategicGroup(rec, seed=int(rec["seed"]) + 104729 * int(w.spec.seed))
        self._rt(w)[rec["group"]] = g
        return g

    def _update_event(self, ctx, grp: StrategicGroup, step, out, prev, exp, checkpoint, data) -> None:
        ref = artifacts.put_array_blob(ctx, grp.quantise()) if checkpoint else None
        bus.emit(ctx, "pop.policy_update", {
            "group": f"strategic:{grp.name}", "step": int(step), "loss": float(out["loss"]),
            "loss_before": float(out["loss_before"]), "weights_hash": grp.weights_hash(), "n_examples": int(out["n"]),
            "prev_hash": prev, "params": grp.params, "examples": list(exp), "checkpoint": ref,
            "arch": f"{grp.arch} h{grp.rec['hidden']} {grp.rec['algo']}", "optimizer_steps": int(data.get("steps", 0)),
            "data": dict(data)}, vis="internal")

    def on_load(self, world):
        self._rt(world).clear()
        for g, rec in sorted(world.state.tables["pop_strat_groups"].items()):
            self._rebuild(world, rec)

    def _rebuild(self, world, rec) -> StrategicGroup:
        st = world.state
        grp = self._new_group(world, rec)
        meta = st.get("pop_policies", f"strategic:{rec['group']}")
        if meta is None:
            return grp
        step0, ref, h0 = meta["checkpoints"][-1]
        grp.load(artifacts.get_array_blob(world, ref))
        if grp.weights_hash() != h0:
            raise ProtocolError(f"strategic group {rec['group']}: checkpoint does not match its hash")
        logs = [st.tables["pop_policy_log"][k] for k in sorted(st.find_ids("pop_policy_log", "group",
                                                                            f"strategic:{rec['group']}"))]
        for lg in logs:
            if lg["step"] <= step0:
                if lg["step"] == step0:
                    grp.baseline = float(lg["data"].get("baseline_after", 0.0))
                continue
            grp.baseline = float(lg["data"]["baseline_before"])
            b = self._batch(st, lg["examples"])
            out = grp.learn(*b, steps=lg["data"]["steps"], seed=lg["data"]["seed"])
            if lg["checkpoint"]:
                grp.quantise()
            if grp.weights_hash() != lg["weights_hash"]:
                raise ProtocolError(f"strategic group {rec['group']} step {lg['step']}: rebuilt weights differ")
        return grp

    def group(self, world, name: str) -> StrategicGroup:
        """The group's live network. Its weights must be the ones the state records (a rolled-back command may have
        trained the live copy): otherwise it is rebuilt from the checkpoints and the logged updates."""
        rec = world.state.get("pop_strat_groups", name)
        if rec is None:
            raise NotFound(f"no strategic group {name}")
        g = self._rt(world).get(name)
        meta = world.state.get("pop_policies", f"strategic:{name}")
        if g is None or (meta is not None and g.weights_hash() != meta["weights_hash"]):
            g = self._rebuild(world, rec)
        return g

    def _group_of_code(self, st, code: int) -> str:
        for name, rec in st.tables["pop_strat_groups"].items():
            if int(rec["code"]) == int(code):
                return name
        raise NotFound(f"no strategic group with code {code}")

    def _batch(self, st, exp_ids):
        recs = [st.tables["pop_strat_exp"][e] for e in exp_ids]
        obs = np.array([r["obs"] for r in recs], np.float32)
        hist = np.array([r["hist"] for r in recs], np.float32)
        acts = {k: np.array([r["act"][k] for r in recs], np.int64) for k in HEADS}
        rew = np.array([r["reward"] for r in recs], np.float32)
        return obs, hist, acts, rew

    # ------------------------------------------------------------------ observing (through tools, as the member)
    def _observe(self, ctx: Ctx, i: int, m: dict) -> tuple[np.ndarray, int]:
        p = ids.pid(i)
        acct = m["account"]
        bal = int(ctx.call("bank.balance", {"account": acct}, as_principal=p)["balance"])
        vis = {"visits": 0, "first_visits": 0, "return_visits": 0, "purchases": 0, "revenue_cents": 0}
        if m.get("host"):
            try:
                vis.update(ctx.call("popweb.visits", {"host": m["host"]}, as_principal=p).get("total") or {})
            except CrucibleError:
                pass
        prev = m.get("seen") or {k: 0 for k in vis}
        d = {k: max(0, int(vis.get(k, 0)) - int(prev.get(k, 0))) for k in vis}
        comp = 0
        if m["topic"] >= 0:
            try:
                comp = int(ctx.call("search.query", {"q": " ".join(text.TOPICS[TOPICS[m["topic"]]][:2]), "kind": "page",
                                                      "limit": 1}, as_principal=p)["matched"])
            except CrucibleError:
                comp = 0
        o = np.zeros(OBS_DIM, np.float32)
        o[0] = math.log1p(d["revenue_cents"] / 100.0)
        o[1] = math.log1p(d["visits"])
        o[2] = math.log1p(d["first_visits"])
        o[3] = d["return_visits"] / max(1, d["visits"])
        o[4] = math.log1p(max(bal, 0) / 100.0) / 10.0
        o[5] = math.log1p(d["purchases"])
        o[6] = 1.0 if m.get("host") else 0.0
        o[7] = math.log1p(comp)
        o[8] = (ctx.now / DAY) / 30.0
        o[9] = float(m.get("last_reward", 0.0))
        off = 10
        if m["topic"] >= 0:
            o[off + m["topic"]] = 1
        off += len(TOPICS)
        if m["price"] >= 0:
            o[off + m["price"]] = 1
        off += len(PRICES)
        if m["content"] >= 0:
            o[off + m["content"]] = 1
        return o, bal, vis

    # ------------------------------------------------------------------ deciding and acting
    @process("decide")
    def pr_decide(self, ctx):
        w = ctx.world
        st = ctx.state
        due = sorted(int(k) for k, m in st.tables["pop_strat_members"].items() if m["next"] <= ctx.now)
        if not due:
            return
        rng = ctx.rng("pop_strategic.decide")
        groups = columns.col(w, TABLE, "policy_group")
        by: dict[str, list] = {}
        for i in due:
            by.setdefault(self._group_of_code(st, int(groups[i])), []).append(i)
        for gname in sorted(by):
            grp = self.group(w, gname)
            mem = by[gname]
            obs, hist, bals, viss = [], [], [], []
            for i in mem:
                m = st.tables["pop_strat_members"][str(i)]
                o, bal, vis = self._observe(ctx, i, m)
                obs.append(o)
                bals.append(bal)
                viss.append(vis)
                hist.append(self._history(st, gname, i))
            O, H = np.stack(obs), np.stack(hist)
            out = inference.recorded(ctx, f"pop.strategic.{gname}", f"strategic:{gname}", grp.weights_hash(),
                                     {"obs": O, "hist": H}, lambda: grp.logits(O, H))
            for j, i in enumerate(mem):
                m = dict(st.tables["pop_strat_members"][str(i)])
                if m.get("last_exp"):                 # the previous decision's reward is now known
                    rew = (bals[j] - int(m["balance"])) / 1000.0
                    bus.emit(ctx, "pop_strategic.reward", {"id": m["last_exp"], "reward": float(rew)}, vis="internal")
                    m["last_reward"] = float(rew)
                act = {}
                for k in sorted(HEADS):
                    lg = out[k][j].astype(np.float64)
                    pr = np.exp(lg - lg.max())
                    act[k] = int(rng.choice(len(pr), p=pr / pr.sum()))
                eid = f"{gname}|{i}|{ctx.now}"
                bus.emit(ctx, "pop_strategic.exp", {"id": eid, "group": gname, "member": int(i), "t": int(ctx.now),
                                                    "obs": [round(float(x), 5) for x in O[j]],
                                                    "hist": [[round(float(x), 5) for x in r] for r in H[j]],
                                                    "act": act, "balance": int(bals[j])}, vis="internal")
                res = self._carry_out(ctx, i, m, act, rng)
                m.update(res)
                m.update({"last_exp": eid, "balance": int(self._balance(ctx, m)), "seen": viss[j],
                          "next": int(ctx.now + self.cfg["decide_every_s"])})
                ctx.emit("pop_strategic.member", {"member": i, **{k: v for k, v in m.items() if k != "member"}},
                         vis="internal")
            if grp.rec["learning"]:
                self._learn(ctx, grp)

    def _balance(self, ctx, m) -> int:
        return ctx.world.modules["bank"].balance(ctx.state, m["account"])

    def _history(self, st, gname, i) -> np.ndarray:
        h = np.zeros((HIST, OBS_DIM), np.float32)
        ex = sorted(st.find_ids("pop_strat_exp", "member", int(i)))[-HIST:]
        for k, e in enumerate(ex):
            h[HIST - len(ex) + k] = st.tables["pop_strat_exp"][e]["obs"]
        return h

    def _carry_out(self, ctx: Ctx, i: int, m: dict, act: dict, rng) -> dict:
        """Execute the chosen actions with ordinary tools as the participant."""
        p = ids.pid(i)
        w = ctx.world
        acct, org = m["account"], m["org"]
        host = f"ent{i}"
        changed = (act["topic"], act["price"], act["content"]) != (m["topic"], m["price"], m["content"])
        out = {"topic": act["topic"], "price": act["price"], "content": act["content"]}
        try:
            if m.get("host") is None:
                rid = ctx.call("code.create_repo", {"name": host, "owner": org, "visibility": "public",
                                                    "description": f"{TOPICS[act['topic']]} shop"}, as_principal=p)["repo"]
                files = sites.app_files(_site_spec(i, act["topic"], act["price"], act["content"], rng), acct)
                ctx.call("code.commit", {"repo": rid, "files": files, "message": "launch"}, as_principal=p)
                prov = sorted(ctx.state.ids("providers"))[0]
                inst = ctx.call("cloud.rent", {"provider": prov, "cores": 1, "account": acct, "owner": org},
                                as_principal=p)
                ctx.call("cloud.deploy", {"repo": rid, "host": host, "instance": inst["instance"]}, as_principal=p)
                ctx.call("id.grant", {"grantee": f"svc:{host}", "cap": "bank.read", "owner": org, "scope": "*"},
                         as_principal=p)
                out.update(host=host, repo=rid, instance=inst["instance"])
            elif changed:
                files = sites.app_files(_site_spec(i, act["topic"], act["price"], act["content"], rng), acct)
                ctx.call("code.commit", {"repo": m["repo"], "files": files, "message": "update"}, as_principal=p)
                ctx.call("cloud.deploy", {"repo": m["repo"], "host": host, "instance": m["instance"]}, as_principal=p)
            if act["promote"] and (m.get("host") or out.get("host")) and "forum" in w.modules:
                boards = sorted(ctx.state.tables["forum_boards"])
                if boards:
                    words = text.TOPICS[TOPICS[act["topic"]]]
                    ctx.call("forum.post", {"board": boards[0], "title": f"New {words[0]} {words[1]} shop",
                                            "body": sites.compose(rng, words, 25) + f" Visit {host}.crucible today."},
                             as_principal=p)
        except CrucibleError as e:
            out["error"] = f"{e.code}: {str(e)[:120]}"
        return out

    def _learn(self, ctx, grp: StrategicGroup) -> None:
        st = ctx.state
        done = sorted(k for k in st.find_ids("pop_strat_exp", "group", grp.name)
                      if st.tables["pop_strat_exp"][k].get("reward") is not None and not st.tables["pop_strat_exp"][k].get("used"))
        if len(done) < int(self.cfg["min_examples"]):
            return
        meta = st.get("pop_policies", f"strategic:{grp.name}")
        step = int(meta["step"]) + 1
        prev = grp.weights_hash()
        seed = int(ctx.world.rng_for("pop_strategic.learn", grp.name, step).integers(0, 2**31 - 1))
        b0 = float(grp.baseline)
        out = grp.learn(*self._batch(st, done), steps=int(self.cfg["steps"]), seed=seed)
        ck = step % int(self.cfg["checkpoint_every"]) == 0
        self._update_event(ctx, grp, step, out, prev, done, ck,
                           {"steps": int(self.cfg["steps"]), "seed": seed, "baseline_before": b0,
                            "baseline_after": float(grp.baseline)})
        ctx.emit("pop_strategic.used", {"ids": done, "step": step}, vis="internal")

    # ------------------------------------------------------------------ system interventions
    @endpoint("freeze_copy", S.obj({"group": S.s_str(pattern=r"[A-Za-z0-9@_-]{1,24}", maxLength=24),
                                    "from_group": S.s_str(maxLength=24), "share": S.s_num(minimum=0, maximum=1)},
                                   ["group", "from_group", "share"]), cost_s=0)
    def ep_freeze_copy(self, ctx, group, from_group, share):
        """System only: a frozen historical copy of a strategic group's current weights for a share of its members."""
        if ctx.actor != SYSTEM:
            raise Forbidden("only the simulator operator can create policy groups")
        st = ctx.state
        if st.get("pop_strat_groups", group) is not None:
            raise InvalidAction("group exists")
        src = st.get("pop_strat_groups", from_group)
        if src is None:
            raise NotFound("no such source group")
        code = 100 + len(st.tables["pop_strat_groups"])
        rec = {**src, "group": group, "learning": False, "code": code, "copy_of": from_group}
        bus.emit(ctx, "pop_strategic.group", rec, vis="internal")
        grp = self._new_group(ctx.world, rec)
        grp.load(self.group(ctx.world, from_group).arrays16())
        self._update_event(ctx, grp, 0, {"loss": 0.0, "loss_before": 0.0, "n": 0},
                           self.group(ctx.world, from_group).weights_hash(), [], True, {"copy_of": from_group})
        w = ctx.world
        members = np.flatnonzero(columns.col(w, TABLE, "policy_group") == int(src["code"]))
        move = members[ctx.rng("freeze").random(members.size) < float(share)]
        if move.size:
            columns.batch(ctx, TABLE, "pop_strategic.assign", move, {"policy_group": np.full(move.size, code, np.int16)})
        return {"group": group, "members": int(move.size), "weights_hash": grp.weights_hash()}

    # ------------------------------------------------------------------ appliers
    @applier("pop_strategic.group")
    def ap_group(self, ev):
        self.world.state.put("pop_strat_groups", ev.payload["group"], dict(ev.payload))

    @applier("pop_strategic.member")
    def ap_member(self, ev):
        p = dict(ev.payload)
        self.world.state.put("pop_strat_members", str(p["member"]), p)

    @applier("pop_strategic.exp")
    def ap_exp(self, ev):
        p = ev.payload
        self.world.state.put("pop_strat_exp", p["id"], {**p, "reward": None, "used": None})

    @applier("pop_strategic.reward")
    def ap_reward(self, ev):
        st = self.world.state
        r = st.get("pop_strat_exp", ev.payload["id"])
        if r is not None:
            st.put("pop_strat_exp", ev.payload["id"], {**r, "reward": ev.payload["reward"]})

    @applier("pop_strategic.used")
    def ap_used(self, ev):
        st = self.world.state
        for e in ev.payload["ids"]:
            r = st.get("pop_strat_exp", e)
            if r is not None:
                st.put("pop_strat_exp", e, {**r, "used": ev.payload["step"]})

    def check_invariants(self, world):
        errs = []
        st = world.state
        for name, rec in st.tables["pop_strat_groups"].items():
            meta = st.get("pop_policies", f"strategic:{name}")
            g = self._rt(world).get(name)
            if meta is None:
                errs.append(f"strategic group {name} has no policy record")
            elif g is not None and g.weights_hash() != meta["weights_hash"]:
                errs.append(f"strategic group {name}: runtime weights differ from the recorded hash")
        return errs
