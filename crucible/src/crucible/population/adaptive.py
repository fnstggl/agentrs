"""Class 2: small adaptive neural participants (MASTER_SPEC Part 3.2 class 2, contract section 2).

`AdaptivePolicy` is ONE shared network per policy group (`nets.AdaptiveNet`, ~0.2M parameters by default) used by
every member of the group; each member keeps its own memory in the `pop.memory` column. A decision for a cohort is
one batched forward pass:

    action_i ~ policy_theta_k(observation_i, memory_i, preferences_i, goals_i, resources_i)

`act_batch(features, memory)` returns the chosen candidate and the new memory (a GRU of the old memory and the
current observation, which includes the outcome of the previous visit). The decision is the resource-visit choice of
the `web` activity (module `popweb`): which known resource to visit, or to explore something new.

Learning is online, from each member's own experience. After every visit the realised utility is paired with the
decision's inputs into an example blob (content-addressed, referenced by event `pop_adaptive.examples`). The
process `pop_adaptive.learn` (default every 6 simulated hours) trains the group's network on its unused examples
(regression of predicted on realised utility; Adam, a fresh optimizer each round) and records `pop.policy_update
{group, step, loss_before, loss, weights_hash, n_examples, examples, checkpoint?}`. The update is deterministic (CPU,
one thread, minibatches drawn from a generator keyed by group and step), so loading a snapshot rebuilds the weights
from the last checkpoint plus the logged updates and checks every weights hash. Inference goes through
`substrate.inference.recorded` (regeneration does not rerun it).

Groups (config `groups`): [{id, share, learning, memory, hidden, seed}] - several independently learning groups,
frozen groups (`learning: false`), groups without personal memory (`memory: false`), and `pop_adaptive.freeze_copy`
(system only) creates a frozen historical copy of another group's current weights.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import Forbidden, InvalidAction, NotFound, ProtocolError
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from ..substrate import artifacts, bus, columns, inference
from . import text
from .schema import RES_TABLE, TABLE

N_KINDS = 7


def _torch():
    from . import nets
    return nets


bus.event_type("pop_adaptive.examples", S.obj({"group": {"type": "integer"}, "ref": {"type": "string"},
                                               "n": {"type": "integer"}, "t": {"type": "integer"}},
                                              ["group", "ref", "n", "t"]), owner="pop_adaptive",
               doc="training examples (decision inputs + realised utility) of one adaptive policy group")
bus.event_type("pop_adaptive.group", S.obj({"group": {"type": "integer"}, "learning": {"type": "boolean"},
                                            "memory": {"type": "boolean"}, "hidden": {"type": "integer"},
                                            "seed": {"type": "integer"}, "copy_of": {"type": ["integer", "null"]},
                                            "share": {"type": "number"}}, ["group", "learning"]),
               owner="pop_adaptive", doc="an adaptive policy group's settings")


class AdaptivePolicy:
    """Shared weights of one policy group (contract: act_batch(features, memory) -> (actions, new_memory);
    learn(batch) -> {loss, weights_hash, n})."""

    def __init__(self, group: int, obs_dim: int, cand_dim: int, mem_dim: int, hidden: int = 256, seed: int = 0,
                 memory: bool = True):
        nets = _torch()
        self.group, self.obs_dim, self.cand_dim, self.mem_dim = group, obs_dim, cand_dim, mem_dim
        self.use_memory = memory
        self.net = nets.AdaptiveNet(obs_dim, cand_dim, mem_dim, hidden)
        nets.init_params(self.net, np.random.default_rng(seed))
        self.net.eval()
        self._hash = None
        self.hidden = hidden

    @property
    def params(self) -> int:
        return _torch().n_params(self.net)

    def weights_hash(self) -> str:
        if self._hash is None:
            self._hash = inference.weights_hash(self.net)
        return self._hash

    def arrays(self) -> dict[str, np.ndarray]:
        return _torch().state_arrays(self.net)

    def load(self, arrays: dict[str, np.ndarray]) -> None:
        _torch().load_arrays(self.net, arrays)
        self._hash = None

    def scores(self, obs: np.ndarray, mem: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        nets = _torch()
        with torch.no_grad(), nets.deterministic():
            m = torch.from_numpy(mem.astype(np.float32)) if self.use_memory else torch.zeros(len(obs), self.mem_dim)
            s, m2 = self.net(torch.from_numpy(obs.astype(np.float32)), m, torch.from_numpy(cand.astype(np.float32)))
        m2 = m2.numpy() if self.use_memory else np.zeros((len(obs), self.mem_dim), np.float32)
        return s.numpy().astype(np.float32), m2.astype(np.float32)

    def act_batch(self, features: dict, memory: np.ndarray, rng: np.random.Generator | None = None,
                  temperature: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        """features {obs [B, obs_dim], cand [B, K, cand_dim], mask [B, K]} -> (chosen candidate [B], new memory)."""
        s, m2 = self.scores(features["obs"], memory, features["cand"])
        g = (rng or np.random.default_rng(0)).gumbel(size=s.shape) * temperature
        s = np.where(features["mask"], s + g, -np.inf)
        return np.argmax(s, axis=1), m2

    def loss_on(self, batch: dict) -> float:
        import torch
        nets = _torch()
        with torch.no_grad(), nets.deterministic():
            return float(self._loss(batch, np.arange(len(batch["reward"]))).item())

    def _loss(self, batch, ix):
        import torch
        obs = torch.from_numpy(batch["obs"][ix].astype(np.float32))
        mem = torch.from_numpy(batch["mem"][ix].astype(np.float32)) if self.use_memory else \
            torch.zeros(len(ix), self.mem_dim)
        cand = torch.from_numpy(batch["cand"][ix].astype(np.float32))[:, None, :]
        r = torch.from_numpy(batch["reward"][ix].astype(np.float32))
        h, m2 = self.net.memory(obs, mem)
        if not self.use_memory:
            m2 = torch.zeros_like(m2)
        pred = self.net.score(h, m2, cand)[:, 0]
        return torch.mean((pred - r) ** 2)

    def learn(self, batch: dict, steps: int = 16, batch_size: int = 256, lr: float = 1e-3, seed: int = 0) -> dict:
        """Train on `batch` {obs, mem, cand (chosen candidate features), reward}: `steps` Adam steps on minibatches
        drawn by a numpy generator from `seed`. Deterministic. Returns {loss, loss_before, weights_hash, n, steps}."""
        import torch
        nets = _torch()
        n = len(batch["reward"])
        rng = np.random.default_rng(seed)
        with nets.deterministic():
            self.net.train()
            opt = torch.optim.Adam(self.net.parameters(), lr=lr)
            with torch.no_grad():
                before = float(self._loss(batch, np.arange(n)).item())
            for _ in range(int(steps)):
                ix = rng.choice(n, size=min(batch_size, n), replace=n < batch_size)
                opt.zero_grad()
                loss = self._loss(batch, ix)
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                after = float(self._loss(batch, np.arange(n)).item())
        self._hash = None
        return {"loss": after, "loss_before": before, "weights_hash": self.weights_hash(), "n": n, "steps": int(steps)}


def _cand_features(w, feats: dict, K: int) -> tuple[np.ndarray, np.ndarray]:
    """[B, K+1, 9 + N_KINDS] candidate features (K known resources + explore) and the mask."""
    B = feats["known"].shape[0]
    R = columns.rows(w, RES_TABLE)
    kinds = np.zeros((B, K + 1), np.int64)
    if R:
        kc = columns.col(w, RES_TABLE, "kind")
        kinds[:, :K] = np.where(feats["valid"], kc[np.clip(feats["known"], 0, R - 1)], 0)
        sug = feats["suggest"]
        kinds[:, K] = np.where(sug >= 0, kc[np.clip(sug, 0, R - 1)], 0)
    has_sug = (feats["suggest"] >= 0).astype(np.float32)
    f = np.zeros((B, K + 1, 9 + N_KINDS), np.float32)
    f[:, :K, 0] = feats["match"]
    f[:, :K, 1] = feats["known_sat"]
    f[:, :K, 2] = np.log1p(feats["known_n"]) / 3.0
    f[:, :K, 3] = feats["recency"]
    f[:, :K, 4] = feats["social"]
    f[:, :K, 5] = feats["valid"]
    f[:, K, 6] = 1.0
    f[:, K, 7] = has_sug
    f[:, K, 8] = feats["trust"] * has_sug
    f[np.arange(B)[:, None], np.arange(K + 1)[None, :], 9 + np.clip(kinds, 0, N_KINDS - 1)] = 1.0
    mask = np.concatenate([feats["valid"], np.ones((B, 1), bool)], axis=1)
    return f, mask


def obs_features(w, idx: np.ndarray) -> np.ndarray:
    """[B, obs_dim] observation of each participant: interests, preferences, beliefs, satisfaction, attention,
    time of day, last utility and the kind of the last resource visited."""
    c = lambda name: columns.col(w, TABLE, name)[idx]  # noqa: E731
    bud = c("attention_budget_min").astype(np.float32)
    hour = (w.now % 86400) / 3600.0
    R = columns.rows(w, RES_TABLE)
    last = c("recent")[:, 0].astype(np.int64)
    lk = np.zeros((idx.size, N_KINDS), np.float32)
    if R:
        k = np.where(last >= 0, columns.col(w, RES_TABLE, "kind")[np.clip(last, 0, R - 1)], 0)
        lk[np.arange(idx.size), np.clip(k, 0, N_KINDS - 1)] = 1.0
    return np.concatenate([
        c("interests").astype(np.float32), c("prefs").astype(np.float32), c("beliefs").astype(np.float32),
        c("satisfaction").astype(np.float32)[:, None], (c("attention_left_min") / np.maximum(bud, 1e-3))[:, None],
        np.full((idx.size, 1), math.sin(2 * math.pi * hour / 24), np.float32),
        np.full((idx.size, 1), math.cos(2 * math.pi * hour / 24), np.float32),
        c("last_utility").astype(np.float32)[:, None], lk], axis=1).astype(np.float32)


@register_module
class PopAdaptive(Module):
    """Shared adaptive policies of class 2 and their online learning. Config: {groups: [{id, share, learning,
    memory, hidden, seed}], learn_every_s: 21600, steps: 16, batch: 256, lr: 1e-3, min_examples: 64,
    checkpoint_every: 4, temperature: 0.15}."""
    name = "pop_adaptive"
    host = "pop_adaptive"
    version = "1"
    depends = ("pop", "popweb", "artifacts", "cols")

    def install(self, world):
        pop = world.modules["pop"]
        self.K = pop.d["K"]
        self.M = pop.d["M"]
        self.obs_dim = pop.d["D"] + pop.d["P"] + pop.d["B"] + 5 + N_KINDS
        self.cand_dim = 9 + N_KINDS
        world.state.declare_table("pop_examples", index=("group",))
        world.state.declare_table("pop_groups")
        self.cfg = {"learn_every_s": 21600, "steps": 16, "batch": 256, "lr": 1e-3, "min_examples": 64,
                    "checkpoint_every": 4, "temperature": 0.15, "hidden": 256, **self.config}
        self.cfg["groups"] = list(self.config.get("groups") or [{"id": 0, "share": 1.0}])
        bus.enforce(world, owner=self.name)

    def _rt(self, world) -> dict:
        return world.runtime.setdefault("pop_adaptive.nets", {})

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        w = ctx.world
        members = np.flatnonzero(columns.col(w, TABLE, "class") == 2)
        groups = self.cfg["groups"]
        shares = np.array([float(g.get("share", 1.0)) for g in groups])
        if members.size and len(groups) > 1:
            rng = w.rng_for("pop_adaptive-assign")
            gid = np.array([int(g["id"]) for g in groups])[rng.choice(len(groups), size=members.size,
                                                                      p=shares / shares.sum())]
            columns.batch(ctx, TABLE, "pop_adaptive.assign", members, {"policy_group": gid.astype(np.int16)})
        elif members.size:
            columns.batch(ctx, TABLE, "pop_adaptive.assign", members,
                          {"policy_group": np.full(members.size, int(groups[0]["id"]), np.int16)})
        for g in groups:
            gid = int(g["id"])
            rec = {"group": gid, "learning": bool(g.get("learning", True)), "memory": bool(g.get("memory", True)),
                   "hidden": int(g.get("hidden", self.cfg["hidden"])), "seed": int(g.get("seed", 1000 + gid)),
                   "copy_of": None, "share": float(g.get("share", 1.0))}
            bus.emit(ctx, "pop_adaptive.group", rec, vis="internal")
            pol = self._new_policy(w, rec)
            self._checkpoint(ctx, gid, pol, step=0, loss=0.0, n=0, examples=[], prev="")
        ctx.schedule(ctx.now + int(self.cfg["learn_every_s"]), "pop_adaptive.learn", {}, prio=60,
                     period=int(self.cfg["learn_every_s"]), key="pop_adaptive.learn")

    def _new_policy(self, w, rec) -> AdaptivePolicy:
        pol = AdaptivePolicy(rec["group"], self.obs_dim, self.cand_dim, self.M, rec["hidden"],
                             seed=int(rec["seed"]) + 7919 * int(w.spec.seed), memory=rec["memory"])
        self._rt(w)[rec["group"]] = pol
        return pol

    def _checkpoint(self, ctx, gid, pol, step, loss, n, examples, prev, loss_before=None, data=None,
                    force=True) -> None:
        ref = artifacts.put_array_blob(ctx, pol.arrays()) if force else None
        bus.emit(ctx, "pop.policy_update", {
            "group": f"adaptive:{gid}", "step": int(step), "loss": float(loss), "weights_hash": pol.weights_hash(),
            "n_examples": int(n), "prev_hash": prev, "params": pol.params, "examples": list(examples),
            "checkpoint": ref, "loss_before": float(loss if loss_before is None else loss_before),
            "arch": f"gru-bandit h{pol.hidden} m{pol.mem_dim}", "optimizer_steps": int((data or {}).get("steps", 0)),
            "data": dict(data or {})}, vis="internal")

    # ------------------------------------------------------------------ loading: rebuild the shared weights
    def on_load(self, world):
        self._rt(world).clear()
        st = world.state
        for gid_s, rec in sorted(st.tables["pop_groups"].items()):
            self._rebuild(world, rec)

    def _rebuild(self, world, rec) -> AdaptivePolicy:
        st = world.state
        gid = int(rec["group"])
        pol = self._new_policy(world, rec)
        meta = st.get("pop_policies", f"adaptive:{gid}")
        if meta is None:
            return pol
        cks = meta.get("checkpoints") or []
        if not cks:
            raise ProtocolError(f"adaptive group {gid} has no checkpoint")
        step0, ref, h0 = cks[-1]
        pol.load(artifacts.get_array_blob(world, ref))
        if pol.weights_hash() != h0:
            raise ProtocolError(f"adaptive group {gid}: checkpoint {ref[:20]} does not match its hash")
        for key in sorted(st.find_ids("pop_policy_log", "group", f"adaptive:{gid}")):
            lg = st.tables["pop_policy_log"][key]
            if lg["step"] <= step0:
                continue
            batch = self._load_examples(world, lg["examples"])
            out = pol.learn(batch, **lg["data"]["learn"])
            if out["weights_hash"] != lg["weights_hash"]:
                raise ProtocolError(f"adaptive group {gid} step {lg['step']}: rebuilt weights differ from the log")
        return pol

    def policy(self, world, gid: int) -> AdaptivePolicy:
        rec = world.state.get("pop_groups", str(int(gid)))
        if rec is None:
            raise NotFound(f"no adaptive policy group {gid}")
        pol = self._rt(world).get(int(gid))
        if pol is None:
            pol = self._rebuild(world, rec)
        pol.use_memory = bool(rec.get("memory", True))      # the group's settings are state
        return pol

    # ------------------------------------------------------------------ deciding (called by popweb)
    def choose(self, ctx: Ctx, idx: np.ndarray, feats: dict, rng) -> np.ndarray:
        """Resource choice of adaptive participants `idx`: returns candidate indices (K = explore). Records the
        inference, writes the new memories and keeps the decision inputs until the visit's utility is known."""
        w = ctx.world
        K = self.K
        obs = obs_features(w, idx)
        cand, mask = _cand_features(w, feats, K)
        mem = columns.col(w, TABLE, "memory")[idx].astype(np.float32)
        groups = columns.col(w, TABLE, "policy_group")[idx].astype(np.int64)
        pick = np.full(idx.size, K, np.int64)
        new_mem = mem.copy()
        stats = self._stats(w)
        for gid in sorted(set(groups.tolist())):
            sel = np.flatnonzero(groups == gid)
            pol = self.policy(w, gid)
            inp = {"obs": obs[sel], "mem": mem[sel], "cand": cand[sel], "mask": mask[sel]}
            out = inference.recorded(ctx, f"pop.adaptive.{gid}", f"adaptive:{gid}", pol.weights_hash(), inp,
                                     lambda pol=pol, inp=inp: dict(zip(("scores", "memory"),
                                                                       pol.scores(inp["obs"], inp["mem"], inp["cand"]))))
            s = np.where(mask[sel], out["scores"] + rng.gumbel(size=out["scores"].shape) *
                         float(self.cfg["temperature"]), -np.inf)
            pick[sel] = np.argmax(s, axis=1)
            new_mem[sel] = out["memory"]
            stats["max_batch"] = max(stats["max_batch"], int(sel.size))
            stats["calls"] += 1
            stats["decisions"] += int(sel.size)
        columns.batch(ctx, TABLE, "pop_adaptive.memory", idx, {"memory": new_mem.astype(np.float16)},
                      extra={"t": int(ctx.now), "n": int(idx.size)})
        pend = w.runtime.setdefault("pop_adaptive.pending", {})
        pend.clear()
        pend["seq"] = w.seq
        pend["rows"] = {int(i): (obs[j], mem[j], cand[j, pick[j]], int(groups[j])) for j, i in enumerate(idx)}
        return pick

    def _stats(self, w) -> dict:
        return w.runtime.setdefault("pop_adaptive.stats", {"max_batch": 0, "calls": 0, "decisions": 0})

    def reward(self, ctx: Ctx, idx: np.ndarray, U: np.ndarray) -> None:
        """Pair the pending decisions of `idx` with their realised utility; one example blob per group."""
        pend = ctx.world.runtime.get("pop_adaptive.pending") or {}
        rows = pend.get("rows") or {}
        if not rows:
            return
        by: dict[int, list] = {}
        for j, i in enumerate(np.asarray(idx).tolist()):
            r = rows.pop(int(i), None)
            if r is not None:
                by.setdefault(r[3], []).append((int(i), r, float(U[j])))
        for gid, lst in sorted(by.items()):
            arr = {"ind": np.array([x[0] for x in lst], np.int32), "obs": np.stack([x[1][0] for x in lst]),
                   "mem": np.stack([x[1][1] for x in lst]), "cand": np.stack([x[1][2] for x in lst]),
                   "reward": np.array([x[2] for x in lst], np.float32), "t": np.full(len(lst), ctx.now, np.int64)}
            ref = artifacts.put_array_blob(ctx, arr)
            bus.emit(ctx, "pop_adaptive.examples", {"group": int(gid), "ref": ref, "n": len(lst), "t": int(ctx.now)},
                     vis="internal")

    def _load_examples(self, world, refs: list[str]) -> dict:
        parts = [artifacts.get_array_blob(world, r) for r in refs]
        return {k: np.concatenate([p[k] for p in parts]) for k in ("obs", "mem", "cand", "reward", "ind")}

    # ------------------------------------------------------------------ learning
    @process("learn")
    def pr_learn(self, ctx):
        w = ctx.world
        st = ctx.state
        for gid_s, rec in sorted(st.tables["pop_groups"].items()):
            if not rec["learning"]:
                continue
            gid = int(rec["group"])
            ex = sorted((r["t"], k) for k in st.find_ids("pop_examples", "group", gid)
                        for r in [st.tables["pop_examples"][k]] if not r.get("used"))
            refs = [k for _, k in ex]
            n = sum(st.tables["pop_examples"][k]["n"] for k in refs)
            if n < int(self.cfg["min_examples"]):
                continue
            pol = self.policy(w, gid)
            meta = st.get("pop_policies", f"adaptive:{gid}")
            step = int(meta["step"]) + 1
            prev = pol.weights_hash()
            batch = self._load_examples(w, refs)
            seed = int(w.rng_for("pop_adaptive.learn", gid, step).integers(0, 2**31 - 1))
            lk = {"steps": int(self.cfg["steps"]), "batch_size": int(self.cfg["batch"]), "lr": float(self.cfg["lr"]),
                  "seed": seed}
            out = pol.learn(batch, **lk)
            ck = step % int(self.cfg["checkpoint_every"]) == 0
            self._checkpoint(ctx, gid, pol, step, out["loss"], out["n"], refs, prev, out["loss_before"],
                             data={"learn": lk, "steps": lk["steps"]}, force=ck)
            ctx.emit("pop_adaptive.used", {"refs": refs, "step": step}, vis="internal")

    # ------------------------------------------------------------------ system interventions (experiments)
    @endpoint("set_learning", S.obj({"group": S.s_int(minimum=0, maximum=10_000), "learning": S.s_bool()},
                                    ["group", "learning"]), cost_s=0)
    def ep_set_learning(self, ctx, group, learning):
        """System only: switch online learning of an adaptive policy group on or off (controlled experiments)."""
        if ctx.actor != SYSTEM:
            raise Forbidden("only the simulator operator can change a policy group")
        rec = ctx.state.get("pop_groups", str(group))
        if rec is None:
            raise NotFound("no such group")
        bus.emit(ctx, "pop_adaptive.group", {**{k: rec[k] for k in ("group", "memory", "hidden", "seed", "copy_of",
                                                                       "share")}, "learning": bool(learning)},
                 vis="internal")
        return {"group": group, "learning": bool(learning)}

    @endpoint("freeze_copy", S.obj({"group": S.s_int(minimum=0, maximum=10_000),
                                    "from_group": S.s_int(minimum=0, maximum=10_000),
                                    "share": S.s_num(minimum=0, maximum=1)}, ["group", "from_group", "share"]),
              cost_s=0)
    def ep_freeze_copy(self, ctx, group, from_group, share):
        """System only: create a frozen historical copy `group` of `from_group`'s current weights and move a share of
        `from_group`'s members to it."""
        if ctx.actor != SYSTEM:
            raise Forbidden("only the simulator operator can create policy groups")
        st = ctx.state
        if st.get("pop_groups", str(group)) is not None:
            raise InvalidAction("group exists")
        src = st.get("pop_groups", str(from_group))
        if src is None:
            raise NotFound("no such source group")
        w = ctx.world
        srcpol = self.policy(w, from_group)
        rec = {"group": int(group), "learning": False, "memory": src["memory"], "hidden": src["hidden"],
               "seed": src["seed"], "copy_of": int(from_group), "share": float(share)}
        bus.emit(ctx, "pop_adaptive.group", rec, vis="internal")
        pol = self._new_policy(w, rec)
        pol.load(srcpol.arrays())
        self._checkpoint(ctx, int(group), pol, step=0, loss=0.0, n=0, examples=[], prev=srcpol.weights_hash(),
                         data={"copy_of": int(from_group), "source_step": int(st.get("pop_policies",
                                                                                     f"adaptive:{from_group}")["step"])})
        members = np.flatnonzero(columns.col(w, TABLE, "policy_group") == int(from_group))
        rng = ctx.rng("freeze_copy")
        move = members[rng.random(members.size) < float(share)]
        if move.size:
            columns.batch(ctx, TABLE, "pop_adaptive.assign", move, {"policy_group": np.full(move.size, int(group),
                                                                                             np.int16)})
        return {"group": group, "members": int(move.size), "weights_hash": pol.weights_hash()}

    # ------------------------------------------------------------------ appliers
    @applier("pop_adaptive.examples")
    def ap_examples(self, ev):
        p = ev.payload
        self.world.state.put("pop_examples", p["ref"], {"group": p["group"], "n": p["n"], "t": p["t"], "used": None})

    @applier("pop_adaptive.used")
    def ap_used(self, ev):
        st = self.world.state
        for r in ev.payload["refs"]:
            rec = st.get("pop_examples", r)
            if rec is not None:
                st.put("pop_examples", r, {**rec, "used": ev.payload["step"]})

    @applier("pop_adaptive.group")
    def ap_group(self, ev):
        p = ev.payload
        self.world.state.put("pop_groups", str(p["group"]), dict(p))

    def check_invariants(self, world):
        errs = []
        st = world.state
        for gid_s, rec in st.tables["pop_groups"].items():
            meta = st.get("pop_policies", f"adaptive:{rec['group']}")
            if meta is None:
                errs.append(f"adaptive group {gid_s} has no policy record")
                continue
            pol = self._rt(world).get(int(rec["group"]))
            if pol is not None and pol.weights_hash() != meta["weights_hash"]:
                errs.append(f"adaptive group {gid_s}: runtime weights differ from the recorded hash")
        return errs
