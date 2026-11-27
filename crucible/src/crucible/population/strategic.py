"""Class 3: strategic neural participants (MASTER_SPEC Part 3.2 class 3; module `pop_strategic`).

Strategic participants are individuals of the population (rows of `pop` with class 3) who run ventures on the
population's internet: each founds a company with its own savings (identity `create_org`, a bank account) and then,
decision after decision, BUILDS its business with ordinary world tools, as itself and under the same schemas and
authorisation as any agent (`code.create_repo`, `code.read`, `code.commit`, `cloud.rent`, `cloud.deploy`,
`cloud.undeploy`, `cloud.release`, `id.grant`, `forum.post`, `bank.balance`, `popweb.visits`, `search.query`).

The policy's action space is the set of things such a participant can do, composed over time - not a fixed site
template with a few knobs (the class-3 design the recovery audit rejected):

* `launch` a site on a market topic (at most two live sites): it writes the first articles itself
  (`population.writing`: real corpus sentences of the topic; coherence and length depend on its own writing skill and
  on the length it chooses), optionally a product at a chosen price, and deploys the site on a rented instance;
* `write` another article for one of its sites (topic and length of its choice: a site may drift off its topic);
* `product`: put a product on sale at a chosen price; `reprice` its products (price 0 withdraws them);
* `promote` a site with a forum post written about the chosen topic;
* `link` to another venture's site found through public search on a topic - a cooperative act: links let the
  visitors of one site discover the other;
* `close` a site (undeploy, release the instance: hosting costs stop);
* `wait`.

Writing costs the participant attention (minutes from its daily budget at its writing speed); hosting costs money
(cloud billing). Nothing is scored by the generator: a site's quality is the text its participant wrote, and the
population's visitors judge it (popweb, shared semantic space, legibility); revenue comes only from independent
individuals' purchases through the site's own checkout.

Observation (through tools, as the member): its balance; per own site the change since its last decision of visits,
first visits, returns, purchases and revenue (`popweb.visits`, operator only), articles, products, price level,
live flag; per market topic how many public pages compete (`search.query`) and whether it has a site there; the
time; its last reward and last action. A GRU group also sees its last six observations.

Objectives differ between groups (`objective`): `profit` (change of the venture's bank balance, in $10 units) or
`reach` (log of the visits plus half the log of the returning visits its sites got) - unrelated objectives in one
market. The reward of a decision is the objective's change until the member's next decision, computed by the member
from what it observes.

Policy groups (config `groups`): each group is ONE network trained only on its own members' experience, with its own
architecture and algorithm. Defaults: `A` MLP (~2.3M parameters), REINFORCE with standardised rewards and a running
baseline (Adam), objective profit; `A0` a FROZEN copy of A's initial weights (identical initialisation: the control
for whether A's change of behaviour is learned); `B` GRU over the member's history (~2.3M), advantage actor-critic
(RMSprop), objective reach; `C` MLP (~2.1M), PPO with a clipped probability ratio (Adam), objective profit. Members of
all groups compete in the same market (topics, buyers, the population's attention) and may cooperate (links).
`pop_strategic.freeze_copy` (system only) creates a frozen historical copy of a group's current weights for a share
of its members. Every update emits `pop.policy_update`; checkpoints (float16 weights the live model continues from)
are array blobs every `checkpoint_every` updates; loading rebuilds the weights from the last checkpoint plus the
logged updates (deterministic) and checks every hash.

Decisions run inside the population's own tick (`population.register_class_runtime(world, 3, ...)`): every batch a
decision records on the population table is listed by that session (decision provenance). Roles covered:
entrepreneurs, creators/publishers, competitors and collaborators; the spec's researchers, engineers, employees,
organisations and investors are not class-3 roles here (limits in FIDELITY_REGISTER `pop.decision.strategic_policy`).
"""
from __future__ import annotations

import json
import math

import numpy as np

from ..core import schema as S
from ..core.errors import CrucibleError, Forbidden, InvalidAction, NotFound, ProtocolError
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, register_module
from ..substrate import artifacts, attention, bus, columns, ids, inference
from ..substrate import inference_batch as IB
from . import semantics, sites, writing
from .schema import TABLE

DAY = 86400
N_TOPICS = 16
ACTIONS = ("wait", "launch", "write", "product", "reprice", "promote", "link", "close")
LENGTHS = (3, 7, 14)                       # sentences per article
PRICES = (0, 300, 900, 2500, 6000)         # cents (0: no product / withdraw products)
MAX_SITES = 2
HEADS = {"act": len(ACTIONS), "topic": N_TOPICS, "site": MAX_SITES, "length": len(LENGTHS), "price": len(PRICES)}
SITE_F = 9
OBS_DIM = 2 + MAX_SITES * SITE_F + 2 * N_TOPICS + 3 + 1 + len(ACTIONS)
HIST = 6
WORDS_PER_MIN = 12.0
OBJECTIVES = ("profit", "reach")


def market_topics() -> list[int]:
    """The 16 most read topics of the population's corpus (semantics.popularity): the market strategic participants
    choose among (indices of semantics.topic_names())."""
    p = semantics.popularity()
    return [int(t) for t in np.argsort(-p, kind="stable")[:N_TOPICS]]


def feasible(m: dict) -> np.ndarray:
    """Which action kinds a member can take given its own ventures (what it knows about itself): launching needs a
    free site slot, everything that works on a site needs a live one."""
    live = sum(1 for s in m.get("sites") or [] if s.get("live"))
    f = np.ones(len(ACTIONS), bool)
    if live >= MAX_SITES:
        f[ACTIONS.index("launch")] = False
    if live == 0:
        for a in ("write", "product", "reprice", "promote", "link", "close"):
            f[ACTIONS.index(a)] = False
    return f


def venture_host(member: int, k: int) -> str:
    return f"v{int(member)}s{int(k)}"


def host_member(host: str) -> int | None:
    """The member that operates a venture host `v<i>s<k>` (None for any other host)."""
    if not isinstance(host, str) or not host.startswith("v") or "s" not in host[1:]:
        return None
    a, _, b = host[1:].partition("s")
    return int(a) if a.isdigit() and b.isdigit() else None


bus.event_type("pop_strategic.exp", S.obj({"id": {"type": "string"}, "group": {"type": "string"},
                                           "member": {"type": "integer"}, "t": {"type": "integer"},
                                           "obs": {"type": "array"}, "hist": {"type": "array"},
                                           "act": {"type": "object", "additionalProperties": True},
                                           "logp": {"type": "number"}, "balance": {"type": "integer"},
                                           "outcome": {"type": "string"}, "detail": {"type": "string"},
                                           "mask": {"type": "array"}},
                                          ["id", "group", "member", "t", "obs", "act", "balance"]),
               owner="pop_strategic", doc="one strategic decision (observation, chosen actions, what happened)")
bus.event_type("pop_strategic.reward", S.obj({"id": {"type": "string"}, "reward": {"type": "number"}},
                                             ["id", "reward"]), owner="pop_strategic",
               doc="the realised reward of a strategic decision (the group objective's change until the next decision)")
bus.event_type("pop_strategic.group", S.obj({"group": {"type": "string"}, "arch": {"type": "string"},
                                             "hidden": {"type": "integer"}, "algo": {"type": "string"},
                                             "lr": {"type": "number"}, "learning": {"type": "boolean"},
                                             "seed": {"type": "integer"}, "code": {"type": "integer"},
                                             "objective": {"type": "string"},
                                             "copy_of": {"type": ["string", "null"]}},
                                            ["group", "arch", "learning", "code"]), owner="pop_strategic",
               doc="a strategic policy group")

DEFAULT_GROUPS = [
    {"group": "A", "arch": "mlp", "hidden": 1024, "algo": "reinforce", "lr": 3e-4, "seed": 11, "objective": "profit"},
    {"group": "A0", "arch": "mlp", "hidden": 1024, "algo": "reinforce", "lr": 3e-4, "seed": 11, "objective": "profit",
     "learning": False, "copy_of": "A"},
    {"group": "B", "arch": "gru", "hidden": 576, "algo": "a2c", "lr": 1e-4, "seed": 23, "objective": "reach"},
    {"group": "C", "arch": "mlp", "hidden": 1000, "algo": "ppo", "lr": 3e-4, "seed": 37, "objective": "profit"},
]


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

    def learn(self, obs, hist, acts: dict, reward, logp_old=None, mask=None, *, steps: int, seed: int) -> dict:
        """One update of the group's policy on its completed decisions (deterministic). Rewards are standardised
        over the batch. REINFORCE: advantage = reward minus a running baseline; A2C: reward minus the value head
        (critic loss on the standardised reward); PPO: clipped ratio against the logged decision log-probabilities,
        the value head as baseline. Entropy bonus 0.02; gradient norm clipped at 1."""
        import torch
        from . import nets
        rng = np.random.default_rng(seed)
        n = len(reward)
        lr = float(self.rec["lr"])
        algo = self.rec["algo"]
        r_raw = np.asarray(reward, np.float64)
        mu, sd = float(r_raw.mean()), float(r_raw.std())
        r = torch.from_numpy(((r_raw - mu) / (sd if sd > 1e-8 else 1.0)).astype(np.float32))
        lp_old = torch.from_numpy(np.asarray(logp_old if logp_old is not None else np.zeros(n), np.float32))
        with nets.deterministic():
            self.net.train()
            opt = (torch.optim.RMSprop if algo == "a2c" else torch.optim.Adam)(self.net.parameters(), lr=lr)
            o = torch.from_numpy(np.asarray(obs, np.float32))
            h = None if (hist is None or self.arch == "mlp") else torch.from_numpy(np.asarray(hist, np.float32))
            a = {k: torch.from_numpy(np.asarray(v, np.int64)) for k, v in acts.items()}
            neg = None if mask is None else torch.from_numpy(np.where(np.asarray(mask, bool), 0.0, -1e9
                                                                       ).astype(np.float32))

            def loss_fn(ix):
                out, val = self.net(o[ix], None if h is None else h[ix])
                if neg is not None:                    # infeasible actions were masked when the member decided
                    out = {**out, "act": out["act"] + neg[ix]}
                logp = sum(torch.log_softmax(out[k], -1).gather(1, a[k][ix][:, None])[:, 0] for k in HEADS)
                ent = sum(-(torch.softmax(out[k], -1) * torch.log_softmax(out[k], -1)).sum(-1) for k in HEADS)
                if algo == "reinforce":
                    adv = r[ix] - float(self.baseline)
                    return -(logp * adv).mean() - 0.02 * ent.mean()
                adv = (r[ix] - val).detach()
                if algo == "ppo":
                    ratio = torch.exp(torch.clamp(logp - lp_old[ix], -20.0, 20.0))
                    pg = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
                    return pg + 0.5 * ((val - r[ix]) ** 2).mean() - 0.02 * ent.mean()
                return -(logp * adv).mean() + 0.5 * ((val - r[ix]) ** 2).mean() - 0.02 * ent.mean()
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
            self.baseline = 0.8 * self.baseline + 0.2 * float(r.mean().item())
        self._hash = None
        return {"loss": after, "loss_before": before, "weights_hash": self.weights_hash(), "n": n,
                "reward_mean": mu, "reward_sd": sd}


def _site_spec(host: str, topic_name: str, articles: list, products: list, links: list, about: str) -> dict:
    title = " ".join(w.capitalize() for w in topic_name.replace("_", " ").split()[:3]) or "Site"
    return {"name": host, "title": f"{title} - {host}", "topic": topic_name,
            "category": "shop" if products else "blog", "behaviour": "operated", "articles": articles,
            "products": products, "links": links, "about": about}


@register_module
class PopStrategic(Module):
    """Strategic policy groups of class 3. Config: {groups: [{group, arch, hidden, algo, lr, seed, objective,
    learning, copy_of}], decide_every_s: 43200, endowment_cents: 500000, steps: 8, checkpoint_every: 3,
    min_examples: 4}."""
    name = "pop_strategic"
    host = "pop_strategic"
    version = "2"
    depends = ("pop", "popweb", "identity", "bank", "code", "cloud", "net", "search", "artifacts", "cols")

    def install(self, world):
        st = world.state
        st.declare_table("pop_strat_groups")
        st.declare_table("pop_strat_exp", index=("group", "member"))
        st.declare_table("pop_strat_members")
        self.cfg = {"decide_every_s": DAY // 2, "endowment_cents": 500_000, "steps": 8, "checkpoint_every": 3,
                    "min_examples": 4, **self.config}
        self.cfg["groups"] = [dict(g) for g in (self.config.get("groups") or DEFAULT_GROUPS)]
        self.topics = market_topics()
        bus.enforce(world, owner=self.name)
        from . import core
        core.register_class_runtime(world, 3, self.decide, owner=self.name)

    def _rt(self, world) -> dict:
        return world.runtime.setdefault("pop_strategic.nets", {})

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        w = ctx.world
        members = np.flatnonzero(columns.col(w, TABLE, "class") == 3)
        groups = self.cfg["groups"]
        for k, g in enumerate(groups):
            rec = {"group": g["group"], "arch": g.get("arch", "mlp"), "hidden": int(g.get("hidden", 1024)),
                   "algo": g.get("algo", "reinforce"), "lr": float(g.get("lr", 3e-4)),
                   "learning": bool(g.get("learning", True)), "seed": int(g.get("seed", 11 + k)), "code": 100 + k,
                   "objective": str(g.get("objective", "profit")), "copy_of": g.get("copy_of")}
            if rec["objective"] not in OBJECTIVES:
                raise ProtocolError(f"pop_strategic: objective must be one of {OBJECTIVES}")
            bus.emit(ctx, "pop_strategic.group", rec, vis="internal")
            grp = self._new_group(w, rec)
            self._update_event(ctx, grp, step=0, out={"loss": 0.0, "loss_before": 0.0, "n": 0}, prev="", exp=[],
                               checkpoint=True, data={})
        if members.size:
            codes = np.array([100 + (k % len(groups)) for k in range(members.size)], np.int16)
            columns.batch(ctx, TABLE, "pop_strategic.assign", members, {"policy_group": codes})
        bank = w.modules["bank"]
        idm = w.modules["identity"]
        for i in members.tolist():
            # the entrepreneur founds a company with its savings; the company operates its sites
            p = ids.ensure_principal(ctx, i, account=True)
            oid = idm.create_org(ctx, f"ent{i}", f"Venture {i}", p, "company", {})
            acct = bank.open_account(ctx, oid, "operating")
            bank.move(ctx, "acct_genesis", acct, int(self.cfg["endowment_cents"]), "endowment", "founder savings")
            ctx.emit("pop_strategic.member", {"member": i, "sites": [], "seen": {}, "last_exp": None,
                                              "org": oid, "account": acct, "balance": bank.balance(ctx.state, acct),
                                              "last_reward": 0.0, "last_act": -1}, vis="internal")

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
            grp.learn(*b, steps=lg["data"]["steps"], seed=lg["data"]["seed"])
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

    def _group_of_code(self, st, code: int) -> str | None:
        for name, rec in st.tables["pop_strat_groups"].items():
            if int(rec["code"]) == int(code):
                return name
        return None

    def _batch(self, st, exp_ids):
        recs = [st.tables["pop_strat_exp"][e] for e in exp_ids]
        obs = np.array([r["obs"] for r in recs], np.float32)
        hist = np.array([r["hist"] for r in recs], np.float32)
        acts = {k: np.array([r["act"][k] for r in recs], np.int64) for k in HEADS}
        rew = np.array([r["reward"] for r in recs], np.float32)
        lp = np.array([float(r.get("logp") or 0.0) for r in recs], np.float32)
        mk = np.array([r.get("mask") or [True] * len(ACTIONS) for r in recs], bool)
        return obs, hist, acts, rew, lp, mk

    # ------------------------------------------------------------------ observing (through tools, as the member)
    def _market(self, ctx: Ctx, p: str) -> np.ndarray:
        """Competing public pages per market topic, from public search (the first member deciding in a tick asks;
        the results are the same for everyone deciding at that time)."""
        cache = ctx.world.runtime.setdefault("pop_strategic.market", {})
        if cache.get("t") == ctx.now:
            return cache["v"]
        names = semantics.topic_names()
        v = np.zeros(N_TOPICS, np.float32)
        for k, t in enumerate(self.topics):
            q = " ".join(names[t].replace("_", " ").split()[:2])
            try:
                v[k] = float(ctx.call("search.query", {"q": q, "kind": "page", "limit": 1}, as_principal=p)["matched"])
            except CrucibleError:
                v[k] = 0.0
        cache.update(t=ctx.now, v=v)
        return v

    def _observe(self, ctx: Ctx, i: int, m: dict) -> tuple[np.ndarray, int, dict, float]:
        p = ids.pid(i)
        bal = int(ctx.call("bank.balance", {"account": m["account"]}, as_principal=p)["balance"])
        seen = dict(m.get("seen") or {})
        o = np.zeros(OBS_DIM, np.float32)
        o[0] = math.log1p(max(bal, 0) / 100.0) / 10.0
        live = [s for s in m["sites"] if s.get("live")]
        o[1] = len(live) / MAX_SITES
        new_seen, reach = {}, 0.0
        for k, sref in enumerate(live[:MAX_SITES]):
            tot = {"visits": 0, "first_visits": 0, "return_visits": 0, "purchases": 0, "revenue_cents": 0}
            try:
                tot.update(ctx.call("popweb.visits", {"host": sref["host"]}, as_principal=p).get("total") or {})
            except CrucibleError:
                pass
            prev = seen.get(sref["host"]) or {}
            d = {q: max(0, int(tot.get(q, 0)) - int(prev.get(q, 0))) for q in tot}
            new_seen[sref["host"]] = {q: int(tot.get(q, 0)) for q in tot}
            reach += math.log1p(d["visits"]) + 0.5 * math.log1p(d["return_visits"])
            b = 2 + k * SITE_F
            o[b:b + SITE_F] = [math.log1p(d["visits"]), math.log1p(d["first_visits"]),
                               d["return_visits"] / max(1, d["visits"]), math.log1p(d["purchases"]),
                               math.log1p(d["revenue_cents"] / 100.0), sref.get("articles", 0) / 20.0,
                               sref.get("products", 0) / 5.0, sref.get("price", 0) / (len(PRICES) - 1), 1.0]
        off = 2 + MAX_SITES * SITE_F
        o[off:off + N_TOPICS] = np.log1p(self._market(ctx, p)) / 5.0
        for sref in live:
            if int(sref.get("tk", -1)) >= 0:
                o[off + N_TOPICS + int(sref["tk"])] = 1.0
        off += 2 * N_TOPICS
        day = ctx.now / DAY
        o[off:off + 3] = [min(day / 60.0, 2.0), math.sin(2 * math.pi * day / 7), math.cos(2 * math.pi * day / 7)]
        o[off + 3] = float(np.clip(m.get("last_reward", 0.0), -5, 5))
        if int(m.get("last_act", -1)) >= 0:
            o[off + 4 + int(m["last_act"])] = 1.0
        return o, bal, new_seen, reach

    def _history(self, st, gname, i) -> np.ndarray:
        h = np.zeros((HIST, OBS_DIM), np.float32)
        ex = sorted(st.find_ids("pop_strat_exp", "member", int(i)),
                    key=lambda e: (st.tables["pop_strat_exp"][e]["t"], e))[-HIST:]
        for k, e in enumerate(ex):
            h[HIST - len(ex) + k] = st.tables["pop_strat_exp"][e]["obs"]
        return h

    # ------------------------------------------------------------------ deciding (the population's tick, class 3)
    def decide(self, ctx: Ctx, idx: np.ndarray, now: int) -> np.ndarray:
        """Class-3 runtime: the due members decide with their group's policy, act through tools, and the groups
        learn from completed decisions. Returns each member's next activation time."""
        w = ctx.world
        st = ctx.state
        rng = ctx.rng("pop_strategic.decide")
        groups = columns.col(w, TABLE, "policy_group")
        # the next decision on the group's common grid (multiples of decide_every_s): every member of a group that is
        # due decides in the same forward pass from its second decision on (batched inference, MASTER_SPEC 13.3)
        E = int(self.cfg["decide_every_s"])
        nxt = np.full(idx.size, (now // E + 1) * E, np.int64)
        by: dict[str, list] = {}
        for i in idx.tolist():
            g = self._group_of_code(st, int(groups[i]))
            if g is None or st.get("pop_strat_members", str(i)) is None:
                continue
            by.setdefault(g, []).append(int(i))
        for gname in sorted(by):
            grp = self.group(w, gname)
            obj = grp.rec.get("objective", "profit")
            mem = by[gname]
            obs, hist, bals, seens, reaches = [], [], [], [], []
            for i in mem:
                m = st.tables["pop_strat_members"][str(i)]
                o, bal, sn, rc = self._observe(ctx, i, m)
                obs.append(o)
                bals.append(bal)
                seens.append(sn)
                reaches.append(rc)
                hist.append(self._history(st, gname, i))
            O, H = np.stack(obs), np.stack(hist)
            out = IB.decide(ctx, IB.strategic_request(grp, O, H, model_id=f"strategic:{gname}"))
            for j, i in enumerate(mem):
                m = dict(st.tables["pop_strat_members"][str(i)])
                if m.get("last_exp"):                 # the previous decision's reward is now known
                    rew = (bals[j] - int(m["balance"])) / 1000.0 if obj == "profit" else float(reaches[j])
                    bus.emit(ctx, "pop_strategic.reward", {"id": m["last_exp"], "reward": float(rew)}, vis="internal")
                    m["last_reward"] = float(rew)
                act, logp = {}, 0.0
                feas = feasible(m)
                for k in sorted(HEADS):
                    lg = out[k][j].astype(np.float64)
                    if k == "act":
                        lg = np.where(feas, lg, -1e9)
                    pr = np.exp(lg - lg.max())
                    pr = pr / pr.sum()
                    act[k] = int(rng.choice(len(pr), p=pr))
                    logp += math.log(max(float(pr[act[k]]), 1e-12))
                eid = f"{gname}|{i}|{now}"
                res = self._carry_out(ctx, i, m, act, rng)
                bus.emit(ctx, "pop_strategic.exp", {"id": eid, "group": gname, "member": int(i), "t": int(now),
                                                    "obs": [round(float(x), 5) for x in O[j]],
                                                    "hist": [[round(float(x), 5) for x in r] for r in H[j]],
                                                    "act": act, "logp": round(float(logp), 6), "balance": int(bals[j]),
                                                    "mask": [bool(x) for x in feas],
                                                    "outcome": res.pop("outcome", "ok"),
                                                    "detail": str(res.pop("detail", ""))}, vis="internal")
                m.update(res)
                m.update({"last_exp": eid, "balance": int(self._balance(ctx, m["account"])), "seen": seens[j],
                          "last_act": int(act["act"])})
                ctx.emit("pop_strategic.member", {"member": i, **{k: v for k, v in m.items() if k != "member"}},
                         vis="internal")
            if grp.rec["learning"]:
                self._learn(ctx, grp)
        return nxt

    def _balance(self, ctx, acct) -> int:
        return ctx.world.modules["bank"].balance(ctx.state, acct)

    # ------------------------------------------------------------------ acting with ordinary tools
    def _write_articles(self, ctx: Ctx, i: int, topic: int, n_articles: int, length: int, rng) -> list[dict]:
        """Articles the member writes (real corpus sentences, `population.writing`), limited by the attention it has
        left today (writing speed WORDS_PER_MIN); its writing skill grows with practice."""
        skill = float(columns.col(ctx.world, TABLE, "skills")[i, 1])
        out = []
        for _ in range(int(n_articles)):
            title, body = writing.compose(rng, topic, LENGTHS[length], skill)
            mins = len(body.split()) / WORDS_PER_MIN
            got = float(attention.spend(ctx, np.array([i]), np.array([mins]), activity="strategic.write")[0])
            if got + 1e-6 < mins:
                if got < 1.0:
                    break
                sents = body.split(". ")
                body = ". ".join(sents[:max(1, int(len(sents) * got / mins))])
            out.append({"title": title, "body": body})
        if out:
            sk = columns.col(ctx.world, TABLE, "skills")[np.array([i])].astype(np.float32)
            sk[0, 1] = min(1.0, float(sk[0, 1]) + 0.01 * len(out))
            from .api import record_decisions
            from .core import running_as
            with running_as(ctx, "pop"):
                record_decisions(ctx, "pop_strategic.practice", np.array([i]), {"skills": sk.astype(np.float16)},
                                 model_id="strategic_policy", model_version="2",
                                 fidelity_id="pop.decision.strategic_policy", extra={"articles": len(out)})
        return out

    def _read_spec(self, ctx, p, repo) -> dict:
        r = ctx.call("code.read", {"repo": repo, "path": "site.json"}, as_principal=p)
        return json.loads(r.get("content") or r.get("text") or "{}")

    def _deploy(self, ctx, p, sref, spec, acct, message) -> None:
        files = sites.app_files(spec, acct)
        files["site.json"] = json.dumps(spec, sort_keys=True)
        ctx.call("code.commit", {"repo": sref["repo"], "files": files, "message": message}, as_principal=p)
        ctx.call("cloud.deploy", {"repo": sref["repo"], "host": sref["host"], "instance": sref["instance"]},
                 as_principal=p)

    def _carry_out(self, ctx: Ctx, i: int, m: dict, act: dict, rng) -> dict:
        """Execute the chosen action with ordinary tools as the participant. Returns the member-state changes, an
        `outcome` label (ok, invalid: <why>, error: <tool error>) and a `detail`."""
        p = ids.pid(i)
        w = ctx.world
        acct, org = m["account"], m["org"]
        names = semantics.topic_names()
        a = ACTIONS[act["act"]]
        tk = int(act["topic"])
        topic = self.topics[tk]
        sites_ = [dict(s) for s in m["sites"]]
        live = [s for s in sites_ if s.get("live")]
        target = live[act["site"] % len(live)] if live else None
        skill = float(columns.col(w, TABLE, "skills")[i, 1])
        out: dict = {"sites": sites_}
        try:
            if a == "wait":
                out["outcome"] = "ok"
            elif a == "launch":
                if len(live) >= MAX_SITES:
                    out["outcome"] = "invalid: two sites already"
                    return out
                host = venture_host(i, len(sites_))
                arts = self._write_articles(ctx, i, topic, 2, act["length"], rng)
                if not arts:
                    out["outcome"] = "invalid: no attention left to write"
                    return out
                prods = []
                if PRICES[act["price"]] > 0:
                    _, about = writing.compose(rng, topic, 2, skill)
                    prods = [{"name": f"{names[topic].replace('_', ' ').title()} guide", "price": PRICES[act["price"]],
                              "about": about}]
                _, about = writing.compose(rng, topic, 2, skill)
                spec = _site_spec(host, names[topic], arts, prods, [], about)
                rid = ctx.call("code.create_repo", {"name": host, "owner": org, "visibility": "public",
                                                    "description": f"{names[topic]} by venture {i}"},
                               as_principal=p)["repo"]
                prov = sorted(ctx.state.ids("providers"))[0]
                inst = ctx.call("cloud.rent", {"provider": prov, "cores": 1, "account": acct, "owner": org},
                                as_principal=p)
                sref = {"host": host, "repo": rid, "instance": inst["instance"], "tk": tk, "articles": len(arts),
                        "products": len(prods), "price": int(act["price"]) if prods else 0, "live": True, "links": []}
                self._deploy(ctx, p, sref, spec, acct, "launch")
                ctx.call("id.grant", {"grantee": f"svc:{host}", "cap": "bank.read", "owner": org, "scope": "*"},
                         as_principal=p)
                sites_.append(sref)
                out.update(outcome="ok", detail=host)
            elif target is None:
                out["outcome"] = f"invalid: {a} without a live site"
            elif a == "write":
                arts = self._write_articles(ctx, i, topic, 1, act["length"], rng)
                if not arts:
                    out["outcome"] = "invalid: no attention left to write"
                    return out
                spec = self._read_spec(ctx, p, target["repo"])
                spec["articles"] = (spec["articles"] + arts)[-40:]
                self._deploy(ctx, p, target, spec, acct, "new article")
                target["articles"] = len(spec["articles"])
                out.update(outcome="ok", detail=target["host"])
            elif a in ("product", "reprice"):
                spec = self._read_spec(ctx, p, target["repo"])
                price = PRICES[act["price"]]
                if a == "product":
                    if price <= 0 or len(spec["products"]) >= 5:
                        out["outcome"] = "invalid: no price or too many products"
                        return out
                    _, about = writing.compose(rng, topic, 2, skill)
                    spec["products"].append({"name": f"{names[topic].replace('_', ' ').title()} "
                                                     f"{('kit', 'pack', 'pass', 'box', 'set')[len(spec['products'])]}",
                                             "price": int(price), "about": about})
                else:
                    if not spec["products"]:
                        out["outcome"] = "invalid: nothing to reprice"
                        return out
                    if price <= 0:
                        spec["products"] = []
                    else:
                        for pr in spec["products"]:
                            pr["price"] = int(price)
                spec["category"] = "shop" if spec["products"] else "blog"
                self._deploy(ctx, p, target, spec, acct, a)
                target["products"] = len(spec["products"])
                target["price"] = int(act["price"]) if spec["products"] else 0
                out.update(outcome="ok", detail=target["host"])
            elif a == "promote":
                boards = sorted(ctx.state.tables.get("forum_boards", {})) if "forum" in w.modules else []
                if not boards:
                    out["outcome"] = "invalid: no forum board"
                    return out
                ttl, body = writing.compose(rng, topic, LENGTHS[act["length"]], skill)
                ctx.call("forum.post", {"board": boards[0], "title": ttl[:120],
                                        "body": (body + f" More at {target['host']}.crucible.")[:3900]},
                         as_principal=p)
                out.update(outcome="ok", detail=target["host"])
            elif a == "link":
                words = " ".join(names[topic].replace("_", " ").split()[:2])
                res = ctx.call("search.query", {"q": words, "kind": "page", "limit": 10}, as_principal=p)
                mine = {s["host"] for s in sites_}
                cand = []
                for r_ in res.get("results") or []:
                    ref = str(r_.get("ref") or "")
                    h = ref.split(".crucible", 1)[0] if ".crucible" in ref else ((r_.get("hosts") or [None])[0])
                    if h and h not in mine and host_member(h) is not None and h not in cand:
                        cand.append(h)
                spec = self._read_spec(ctx, p, target["repo"])
                cand = [h for h in cand if h not in spec["links"]]
                if not cand:
                    out["outcome"] = "invalid: no other venture site to link"
                    return out
                spec["links"] = (spec["links"] + [cand[0]])[-6:]
                self._deploy(ctx, p, target, spec, acct, "link")
                target["links"] = list(spec["links"])
                out.update(outcome="ok", detail=cand[0])
            elif a == "close":
                ctx.call("cloud.undeploy", {"host": target["host"]}, as_principal=p)
                ctx.call("cloud.release", {"instance": target["instance"]}, as_principal=p)
                target["live"] = False
                out.update(outcome="ok", detail=target["host"])
        except CrucibleError as e:
            out["outcome"] = f"error: {e.code}: {str(e)[:100]}"
        return out

    def _learn(self, ctx, grp: StrategicGroup) -> None:
        st = ctx.state
        done = sorted(k for k in st.find_ids("pop_strat_exp", "group", grp.name)
                      if st.tables["pop_strat_exp"][k].get("reward") is not None and
                      not st.tables["pop_strat_exp"][k].get("used"))
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
                            "baseline_after": float(grp.baseline), "reward_mean": out["reward_mean"],
                            "reward_sd": out["reward_sd"]})
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


# ================================================================================ evaluator side (SP-3.2c, G.4)
def behaviour(world) -> dict:
    """What a world's class 3 did, from its recorded state and log (evaluator side):

    * compete - market topics on which members of two or more different groups had live sites at the same time
      (from the members' site histories) and the visits they shared;
    * cooperate - links from one venture's site to another member's venture site (a cooperative act: it sends
      visitors to the other), mutual link pairs, and links across groups;
    * unrelated objectives - distinct objectives of the groups whose members decided;
    * adapt over the lifetime - a learning group whose weights changed and whose POLICY changed: the mean total
      variation distance between its action distributions on its own recorded observations under its first
      checkpoint and under its current weights is at least 0.05 (the frozen identical-init control A0 stays at 0);
      the chi-square test of the chosen actions, first vs second half of the decisions (which also moves with the
      situations met), and second-half mean rewards are reported alongside;
    * policy copies - groups whose final weights equal another group's."""
    from scipy.stats import chi2_contingency
    st = world.state
    exps = sorted(st.tables.get("pop_strat_exp", {}).values(), key=lambda r: (r["t"], r["id"]))
    groups = st.tables.get("pop_strat_groups", {})
    members = st.tables.get("pop_strat_members", {})
    code_of = {int(r["code"]): g for g, r in groups.items()}
    pg = columns.col(world, TABLE, "policy_group")
    group_of = {int(m): code_of.get(int(pg[int(m)])) for m in members}
    # live sites per topic over time: launches and closes in the decision log
    open_sites: dict[str, tuple] = {}
    overlap = set()
    for r in exps:
        a = ACTIONS[int(r["act"]["act"])] if r.get("outcome") == "ok" else None
        if a == "launch":
            open_sites[r.get("detail", "")] = (int(r["act"]["topic"]), r["group"])
        elif a == "close":
            open_sites.pop(r.get("detail", ""), None)
        by_topic: dict[int, set] = {}
        for tk, g in open_sites.values():
            by_topic.setdefault(tk, set()).add(g)
        overlap |= {tk for tk, gs in by_topic.items() if len(gs) >= 2}
    links = [(int(r["member"]), host_member(r.get("detail", ""))) for r in exps
             if r.get("outcome") == "ok" and ACTIONS[int(r["act"]["act"])] == "link"]
    links = [(a, b) for a, b in links if b is not None and b != a]
    pairs = {(a, b) for a, b in links}
    mutual = {tuple(sorted(p)) for p in pairs if (p[1], p[0]) in pairs}
    across = sum(1 for a, b in pairs if group_of.get(a) != group_of.get(b))
    ups = [e.payload for e in world.log if e.type == "pop.policy_update" and str(e.payload["group"]).startswith(
        "strategic")]
    hashes: dict[str, list] = {}
    for u in ups:
        hashes.setdefault(u["group"].split(":", 1)[1], []).append(u["weights_hash"])
    final = {g: h[-1] for g, h in hashes.items()}
    copies = sum(1 for g, h in final.items() if any(h == h2 for g2, h2 in final.items() if g2 != g))
    adapt = {}
    stm = world.modules.get("pop_strategic")
    for g, rec in groups.items():
        own = [r for r in exps if r["group"] == g]
        acts = [int(r["act"]["act"]) for r in own]
        changed = len(set(hashes.get(g, []))) > 1
        if len(acts) < 20:
            continue
        h = len(acts) // 2
        t = np.zeros((2, len(ACTIONS)))
        for a_ in acts[:h]:
            t[0, a_] += 1
        for a_ in acts[h:]:
            t[1, a_] += 1
        t = t[:, t.sum(axis=0) > 0]
        p = float(chi2_contingency(t)[1]) if t.shape[1] >= 2 else 1.0
        # the policy's own change: its action distribution on the SAME observations (the group's recorded decisions,
        # with their feasibility masks) under its initial weights and its weights now - a change of the network, not
        # of the situations it met
        tv = _policy_shift(world, stm, g, own) if stm is not None else None
        adapt[g] = {"learning": bool(rec.get("learning")), "weights_changed": changed, "policy_shift_tv": tv,
                    "pvalue": p, "decisions": len(acts), "updates": len(hashes.get(g, [])) - 1,
                    "first_half": [int(x) for x in t[0]], "second_half": [int(x) for x in t[1]]}
    rewards: dict[str, list] = {}
    for r in exps:
        if r.get("reward") is not None:
            rewards.setdefault(r["group"], []).append((r["t"], float(r["reward"])))
    second_half = {}
    for g, lst in rewards.items():
        lst.sort()
        second_half[g] = float(np.mean([x for _, x in lst[len(lst) // 2:]])) if lst else 0.0
    objs = sorted({groups[g].get("objective", "profit") for g in {r["group"] for r in exps} if g in groups})
    outcomes: dict[str, int] = {}
    for r in exps:
        k = str(r.get("outcome", "")).split(":")[0]
        outcomes[k] = outcomes.get(k, 0) + 1
    acts_all = {a: sum(1 for r in exps if r.get("outcome") == "ok" and int(r["act"]["act"]) == j)
                for j, a in enumerate(ACTIONS)}
    kinds = {"compete": len(overlap) > 0, "cooperate": len(pairs) > 0, "unrelated_objectives": len(objs) >= 2,
             "adapt_over_lifetime": any(v["learning"] and v["weights_changed"] and (v["policy_shift_tv"] or 0) >= 0.05
                                        for v in adapt.values())}
    return {"kinds": kinds, "kind_count": int(sum(kinds.values())), "competing_topics": sorted(overlap),
            "link_pairs": len(pairs), "mutual_link_pairs": len(mutual), "cross_group_links": across,
            "objectives": objs, "adapt": adapt, "identical_policy_copies": copies, "final_hashes": final,
            "second_half_mean_reward": second_half, "outcomes": outcomes, "actions_ok": acts_all,
            "decisions": len(exps)}


def _policy_shift(world, stm, g: str, own: list) -> float | None:
    """Mean total-variation distance between a group's action distributions (the `act` head, feasibility-masked) on
    its own recorded observations under its first checkpoint and under its current weights."""
    st = world.state
    meta = st.get("pop_policies", f"strategic:{g}")
    if meta is None or not meta.get("checkpoints") or not own:
        return None
    rec = st.get("pop_strat_groups", g)
    first = StrategicGroup(rec, seed=0)
    first.load(artifacts.get_array_blob(world, meta["checkpoints"][0][1]))
    now = stm.group(world, g)
    O = np.array([r["obs"] for r in own], np.float32)
    H = np.array([r["hist"] for r in own], np.float32)
    M = np.array([r.get("mask") or [True] * len(ACTIONS) for r in own], bool)

    def dist(grp):
        lg = grp.logits(O, H)["act"].astype(np.float64)
        lg = np.where(M, lg, -1e9)
        e = np.exp(lg - lg.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return float(0.5 * np.abs(dist(first) - dist(now)).sum(axis=1).mean())
