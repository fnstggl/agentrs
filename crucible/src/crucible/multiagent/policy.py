"""Neural tool-level policies and their learning rules (the learned self-play participants of Gate Q).

A `PolicyGroup` is one independently updated policy: its own network, optimiser (state persisted across updates),
random generators and experience buffer. One group may drive several identities (a shared training system: their
experience is pooled into the one network) or one. Groups share nothing - no tensor, optimiser, generator, buffer
or knowledge object (checked by `crucible.multiagent.integrity.independence`).

The policy acts with the ordinary tools of the world it is in (`crucible.experience`): each decision it scores
every tool it can currently use (its schema's required slots can be filled from what the identity has seen,
`crucible.multiagent.grammar`) plus ``wait``, samples one, then samples one candidate per argument slot. The
network sees

* a state vector built from the identity's own observation and its private `Knowledge` (balances and their change,
  counts of visible event types since its last decision, how many things of each kind it knows and owns, which
  tools it has used successfully so far (hashed), its last result, the clock);
* per tool (scored within its host, after the host is chosen: log pi(tool) = log pi(host) + log pi(tool | host)):
  a hashed bag of the tool's name tokens (so a tool deployed by another agent is scored by what its name
  says), whether it is read-only, its time cost, its required slots, and the identity's own record with it (never
  tried / succeeded / last attempt rejected); one learnable scalar starts state-changing tools at `ACT_BIAS` log-odds
  above read-only ones (an inductive bias toward acting, which learning can undo);
* per candidate value: a hashed bag of the slot name, the candidate's source (seen, own, fresh name, enum, numeric
  bin, omitted, grammar variant) and id kind, plus its recency, ownership and bin position.

Architectures: ``mlp`` (feed-forward state encoder) and ``gru`` (recurrent state encoder over the identity's
decisions; its hidden state is the identity's own). Learning rules (all on the group's own experience; the
training signal is computed by each identity's learner-side objective from its own observations,
`crucible.multiagent.objectives` - the environment's reward is 0.0 in observation-only mode and is never used):

* ``reinforce``  Monte-Carlo policy gradient with a running-mean baseline;
* ``a2c``        advantage actor-critic with n-step bootstrapped returns and a learned value head;
* ``ppo``        clipped surrogate objective with generalised advantage estimation, several epochs per batch.

Evidence-scaled updates (audit K skeptic S.3/S.7): advantages are divided by a running scale of the signal (never
re-normalised per batch, so a batch of tiny differences is not blown up to unit size); a batch whose signals are
all zero is not trained on at all, so a no-signal control's parameters - and behaviour - do not change; an
update counts only when the parameters changed (its weights hash differs).
"""
from __future__ import annotations

import hashlib
import io
import math
import re
import zlib
from typing import Any

import numpy as np

from . import grammar as G
from .knowledge import Knowledge

BUCKETS = 1024
EMB = 16
HIDDEN = 64
TOOL_FLAGS = 7
CAND_NUM = 6
N_EVENT_BUCKETS = 32
KINDS = ("person", "organisation", "account", "artifact", "service", "repo", "deal", "prd", "plan", "lst", "inv",
         "pkg", "proj", "task", "ord", "sub", "ent", "dep", "inst", "entity")
N_DONE_BUCKETS = 48         # what it has accomplished: its own accepted uses of each tool (hashed by tool name)
STATE_DIM = 12 + N_EVENT_BUCKETS + 2 * len(KINDS) + 8 + N_DONE_BUCKETS
WAIT = {"name": "wait", "description": "do nothing this turn", "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True, "costSeconds": 0}}
ALGOS = ("reinforce", "a2c", "ppo")
ACT_BIAS = 1.5           # initial log-odds preference for a state-changing tool over a read-only one (learnable)
ARCHS = ("mlp", "gru")
HP_DEFAULT = {"lr": 1e-3, "bc_lr": 3e-3, "gamma": 0.9, "lam": 0.9, "clip": 0.2, "epochs": 4, "ent": 0.01, "vf": 0.5,
              "update_every": 4, "max_grad": 1.0, "scale_decay": 0.98, "scale_floor": 1e-3}


def h32(s: str) -> int:
    return zlib.crc32(s.encode()) % BUCKETS


def tool_tokens(name: str) -> list[int]:
    parts = re.split(r"[._]", name)
    toks = [f"t:{name}", f"h:{parts[0]}"] + [f"w:{p}" for p in parts[1:] if p]
    return [h32(x) for x in toks]


def field_tokens(tool: str, name: str) -> list[int]:
    return [h32(f"f:{name}"), h32(f"fh:{tool.split('.', 1)[0]}:{name}")]


def cand_tokens(c: G.Cand, fname: str) -> list[int]:
    return [h32(f"s:{c.source}"), h32(f"k:{c.kind}"), h32(f"sf:{c.source}:{fname}"), h32(f"kf:{c.kind}:{fname}")]


def cand_num(c: G.Cand) -> list[float]:
    return [float(c.recency), 1.0 if c.mine else 0.0, float(c.bin), math.log1p(c.n_seen) / 5.0,
            1.0 if c.value is G.OMIT else 0.0, 1.0 if c.meta.get("addressed") else 0.0]


def _symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def state_vector(obs: dict, kn: Knowledge, prev_total: float | None, last: dict | None, signal_ema: float) -> np.ndarray:
    """The identity's state (see the module docstring); only its own observation and knowledge are read."""
    v = np.zeros(STATE_DIM, dtype=np.float32)
    bal = sum(int(x) for k, x in (obs.get("numbers") or {}).items() if k.startswith("bank.balances."))
    res = obs.get("resources") or {}
    v[0] = _symlog(bal) / 15.0
    v[1] = 0.0 if prev_total is None else _symlog(bal - prev_total) / 10.0
    v[2] = math.log1p(len(obs.get("events") or [])) / 5.0
    v[3] = math.log1p(int(res.get("executes", 0))) / 6.0
    v[4] = float(res.get("rejected", 0)) / max(1.0, float(res.get("executes", 0)))
    v[5] = 0.0 if not last else (1.0 if last.get("ok") else -1.0)
    t = int(obs.get("t", 0))
    v[6] = math.sin(2 * math.pi * (t % 86400) / 86400)
    v[7] = math.cos(2 * math.pi * (t % 86400) / 86400)
    v[8] = min(1.0, t / (30 * 86400))
    v[9] = math.log1p(len(kn.orgs)) / 2.0
    v[10] = math.log1p(len(kn.mine)) / 4.0
    v[11] = float(np.tanh(signal_ema))
    k = 12
    me = obs.get("agent")
    for e in obs.get("events") or []:
        b = zlib.crc32(str(e.get("type", "")).encode()) % N_EVENT_BUCKETS
        v[k + b] += 1.0
    v[k:k + N_EVENT_BUCKETS] = np.log1p(v[k:k + N_EVENT_BUCKETS]) / 3.0
    k += N_EVENT_BUCKETS
    cnt = kn.counts()
    for i, kind in enumerate(KINDS):
        v[k + i] = math.log1p(cnt["known"].get(kind, 0)) / 4.0
        v[k + len(KINDS) + i] = math.log1p(cnt["mine"].get(kind, 0)) / 3.0
    k += 2 * len(KINDS)
    evs = obs.get("events") or []
    v[k] = math.log1p(sum(1 for e in evs if e.get("actor") not in (None, me) and str(e.get("actor", "")).startswith(
        ("p:", "o:")))) / 4.0
    v[k + 1] = math.log1p(sum(1 for e in evs if e.get("cause"))) / 4.0
    v[k + 2] = math.log1p(len(kn.words)) / 6.0
    v[k + 3] = math.log1p(len(kn.packages)) / 2.0
    k += 8
    for tool, st in kn.tool_stats.items():          # its own record: which tools it has used successfully
        if st.get("ok"):
            v[k + zlib.crc32(tool.encode()) % N_DONE_BUCKETS] += math.log1p(st["ok"]) / 2.0
    return v


def tool_flags(tool: dict, kn: Knowledge) -> list[float]:
    ann = tool.get("annotations") or {}
    st = kn.tool_stats.get(tool["name"]) or {}
    req = len((tool.get("inputSchema") or {}).get("required") or [])
    return [1.0 if ann.get("readOnlyHint") else 0.0, math.log1p(float(ann.get("costSeconds") or 0)) / 6.0,
            req / 4.0, 1.0 if not st else 0.0, math.log1p(st.get("ok", 0)) / 3.0,
            1.0 if st and st.get("last_error") else 0.0, 1.0 if tool["name"] == "wait" else 0.0]


# ====================================================================== network
def _torch():
    import torch
    return torch


class PolicyNet:
    """Factory for the torch module (kept lazy so that importing this package does not import torch)."""

    @staticmethod
    def build(arch: str, seed: int):
        torch = _torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.arch = arch
                if arch == "gru":
                    self.cell = nn.GRUCell(STATE_DIM, HIDDEN)
                else:
                    self.enc = nn.Sequential(nn.Linear(STATE_DIM, 96), nn.Tanh(), nn.Linear(96, HIDDEN), nn.Tanh())
                self.tool_emb = nn.EmbeddingBag(BUCKETS, EMB, mode="mean")
                self.field_emb = nn.EmbeddingBag(BUCKETS, EMB, mode="mean")
                self.cand_emb = nn.EmbeddingBag(BUCKETS, EMB, mode="mean")
                self.tool_head = nn.Sequential(nn.Linear(HIDDEN + EMB + TOOL_FLAGS, 48), nn.Tanh(), nn.Linear(48, 1))
                self.host_head = nn.Sequential(nn.Linear(HIDDEN + EMB, 32), nn.Tanh(), nn.Linear(32, 1))
                # inductive bias (learnable): a preference for state-changing tools over reading, from the start
                self.act_bias = nn.Parameter(torch.tensor(ACT_BIAS))
                self.slot_head = nn.Sequential(nn.Linear(HIDDEN + 3 * EMB + CAND_NUM, 48), nn.Tanh(),
                                               nn.Linear(48, 1))
                self.value = nn.Sequential(nn.Linear(HIDDEN, 32), nn.Tanh(), nn.Linear(32, 1))
                for head in (self.tool_head, self.slot_head, self.host_head):
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)

            def encode(self, x, h=None):
                if self.arch == "gru":
                    h = torch.zeros(x.shape[0], HIDDEN) if h is None else h
                    z = self.cell(x, h)
                    return z, z
                z = self.enc(x)
                return z, None

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            return Net()


def _bag(emb, toks: list[list[int]]):
    torch = _torch()
    flat = [t for ts in toks for t in ts]
    offs, o = [], 0
    for ts in toks:
        offs.append(o)
        o += len(ts)
    return emb(torch.tensor(flat, dtype=torch.long), torch.tensor(offs, dtype=torch.long))


def host_of(names: list[str]) -> tuple[list[list[int]], list[int]]:
    """Hosts (modules) of a tool list: their token bags and each tool's host index (the policy first chooses a host,
    then a tool of that host: log pi(tool) = log pi(host) + log pi(tool | host))."""
    hosts: dict[str, int] = {}
    idx = []
    for n in names:
        h = n.split(".", 1)[0]
        if h not in hosts:
            hosts[h] = len(hosts)
        idx.append(hosts[h])
    return [[h32(f"h:{h}"), h32(f"hh:{h}")] for h in hosts], idx


def tool_log_probs(net, z, ttoks, tflags, htoks, hidx):
    """Log-probabilities of every candidate tool under the hierarchical (host, then tool) policy."""
    torch = _torch()
    n = len(ttoks)
    te = _bag(net.tool_emb, ttoks)
    tl = net.tool_head(torch.cat([z.expand(n, -1), te, tflags], 1)).squeeze(1) + net.act_bias * (1.0 - tflags[:, 0])
    he = _bag(net.tool_emb, htoks)
    hl = net.host_head(torch.cat([z.expand(len(htoks), -1), he], 1)).squeeze(1)
    hi = torch.tensor(hidx, dtype=torch.long)
    m = tl.max().detach()
    sums = torch.zeros(len(htoks)).index_add(0, hi, (tl - m).exp())
    within = tl - (sums.log() + m)[hi]
    return torch.log_softmax(hl, 0)[hi] + within, te


def weights_hash(net) -> str:
    torch = _torch()
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ====================================================================== the group
class PolicyGroup:
    """One independently updated policy (see the module docstring)."""

    def __init__(self, gid: str, algo: str = "reinforce", arch: str = "mlp", seed: int = 0, hp: dict | None = None,
                 lineage: str | None = None):
        if algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}")
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}")
        torch = _torch()
        self.gid, self.algo, self.arch, self.seed = gid, algo, arch, int(seed)
        self.lineage = lineage or gid
        self.hp = dict(HP_DEFAULT, **(hp or {}))
        self.net = PolicyNet.build(arch, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.hp["lr"])
        self.gen = torch.Generator().manual_seed(int(seed) * 7919 + 17)
        self.rng = np.random.default_rng([int(seed), 0x5E1F])
        self.buffer: dict[str, list[dict]] = {}
        self.frozen = False
        self.version = 0
        self.hash = weights_hash(self.net)
        self.parent_hash: str | None = None
        self.updates = 0                      # parameter-changing updates
        self.skipped_no_signal = 0
        self.update_log: list[dict] = []
        self.baseline = 0.0
        self.scale = 0.0
        self.decisions = 0
        self.identities: list[str] = []
        self.on_update = None                 # callback(group, record) set by the harness (versioning)
        self.controls: list = []              # no-signal controls: twins trained on this group's batches, signal 0

    # ------------------------------------------------------------------ deciding
    def candidates(self, obs: dict, kn: Knowledge, rng: np.random.Generator) -> list[tuple[dict, list]]:
        out = [(WAIT, [])]
        for tool in obs.get("tools") or []:
            slots = G.tool_slots(tool, kn, rng)
            if slots is not None:
                out.append((tool, slots))
        return out

    def decide(self, identity: str, obs: dict, kn: Knowledge, state: np.ndarray, h_prev, explore: bool = True,
               chooser=None):
        """Sample one action. With `chooser` (a scripted demonstrator, `crucible.multiagent.demonstrator`) the
        choices are the chooser's - among exactly the candidates the policy itself would score - and the record is
        a demonstration (its log-probability under the policy is still recorded)."""
        torch = _torch()
        cands = self.candidates(obs, kn, self.rng)
        with torch.no_grad():
            x = torch.from_numpy(state[None, :])
            hp = None if h_prev is None else torch.from_numpy(h_prev[None, :])
            z, h = self.net.encode(x, hp)
            names = [t["name"] for t, _ in cands]
            ttoks = [tool_tokens(n) for n in names]
            htoks, hidx = host_of(names)
            tfl = torch.tensor([tool_flags(t, kn) for t, _ in cands], dtype=torch.float32)
            lp_all, te = tool_log_probs(self.net, z, ttoks, tfl, htoks, hidx)
            pi = lp_all.exp()
            if chooser is not None:
                ti = int(chooser.choose_tool([t for t, _ in cands]))
            else:
                ti = int(torch.multinomial(pi / pi.sum(), 1, generator=self.gen))   # a stochastic policy
            logp = float(lp_all[ti])
            value = float(self.net.value(z).squeeze())
            tool, slots = cands[ti]
            args, slot_recs, meta = {}, [], {}
            te_i = te[ti:ti + 1]
            props = (tool.get("inputSchema") or {}).get("properties") or {}
            for sl in slots:
                cs = sl.cands
                if sl.name in G.FILL_LATER:
                    cs = G.slot_cands(tool["name"], sl.name, props.get(sl.name, {}), sl.required, kn, self.rng, args)
                    if not cs:
                        if sl.required:
                            break
                        continue
                if len(cs) == 0:
                    continue
                ftok = field_tokens(tool["name"], sl.name)
                ctoks = [cand_tokens(c, sl.name) for c in cs]
                cnum = torch.tensor([cand_num(c) for c in cs], dtype=torch.float32)
                fe = _bag(self.net.field_emb, [ftok]).expand(len(cs), -1)
                ce = _bag(self.net.cand_emb, ctoks)
                sl_logits = self.net.slot_head(torch.cat([z.expand(len(cs), -1), te_i.expand(len(cs), -1), fe, ce,
                                                          cnum], 1)).squeeze(1)
                sp = torch.softmax(sl_logits, 0)
                if chooser is not None:
                    ci = int(chooser.choose_value(tool["name"], sl.name, cs, args))
                else:
                    ci = int(torch.multinomial(sp, 1, generator=self.gen))
                logp += float(torch.log(sp[ci] + 1e-12))
                slot_recs.append({"ftok": ftok, "ctoks": ctoks, "cnum": cnum.numpy(), "ci": ci})
                if cs[ci].value is not G.OMIT:
                    args[sl.name] = cs[ci].value
                    meta.update({k: v for k, v in cs[ci].meta.items() if k != "addressed"})
        self.decisions += 1
        rec = {"identity": identity, "t": int(obs.get("t", 0)), "x": state, "h": h_prev, "names": names,
               "ttoks": ttoks, "tflags": tfl.numpy(), "htoks": htoks, "hidx": hidx, "ti": ti, "slots": slot_recs, "logp": logp, "v": value,
               "r": 0.0, "tool": tool["name"], "version": self.version, "n_tools": len(cands), "meta": meta}
        action = None if tool["name"] == "wait" else {"tool": tool["name"], "args": args}
        h_out = None if h is None else h.numpy()[0].copy()
        return action, rec, h_out

    def tool_distribution(self, obs: dict, kn: Knowledge, state: np.ndarray, h_prev=None) -> dict[str, float]:
        """The policy's probability of each tool it can use in this situation (behavioural probes; no sampling)."""
        torch = _torch()
        rng = np.random.default_rng(0)
        cands = self.candidates(obs, kn, rng)
        with torch.no_grad():
            z, _ = self.net.encode(torch.from_numpy(state[None, :]),
                                   None if h_prev is None else torch.from_numpy(h_prev[None, :]))
            names = [t["name"] for t, _ in cands]
            htoks, hidx = host_of(names)
            tfl = torch.tensor([tool_flags(t, kn) for t, _ in cands], dtype=torch.float32)
            lp, _ = tool_log_probs(self.net, z, [tool_tokens(n) for n in names], tfl, htoks, hidx)
        return {n: float(p) for n, p in zip(names, lp.exp())}

    # ------------------------------------------------------------------ experience
    def record(self, rec: dict) -> None:
        if self.frozen:
            return
        self.buffer.setdefault(rec["identity"], []).append(rec)

    def add_signal(self, identity: str, r: float) -> None:
        buf = self.buffer.get(identity)
        if buf:
            buf[-1]["r"] = float(r)

    def pending(self) -> int:
        return sum(len(b) for b in self.buffer.values())

    # ------------------------------------------------------------------ learning
    def _logp_value(self, recs: list[dict]):
        """Log-probabilities, entropies and values of recorded decisions under the current parameters."""
        torch = _torch()
        x = torch.from_numpy(np.stack([r["x"] for r in recs]))
        hs = None
        if self.arch == "gru":
            hs = torch.from_numpy(np.stack([r["h"] if r["h"] is not None else np.zeros(HIDDEN, np.float32)
                                            for r in recs]))
        z, _ = self.net.encode(x, hs)
        logps, ents = [], []
        for i, r in enumerate(recs):
            zi = z[i:i + 1]
            lsm, te = tool_log_probs(self.net, zi, r["ttoks"], torch.from_numpy(r["tflags"]), r["htoks"], r["hidx"])
            lp = lsm[r["ti"]]
            ent = -(lsm.exp() * lsm).sum()
            te_i = te[r["ti"]:r["ti"] + 1]
            for s in r["slots"]:
                n = len(s["ctoks"])
                fe = _bag(self.net.field_emb, [s["ftok"]]).expand(n, -1)
                ce = _bag(self.net.cand_emb, s["ctoks"])
                sl = self.net.slot_head(torch.cat([zi.expand(n, -1), te_i.expand(n, -1), fe, ce,
                                                   torch.from_numpy(s["cnum"])], 1)).squeeze(1)
                sls = torch.log_softmax(sl, 0)
                lp = lp + sls[s["ci"]]
                ent = ent + (-(sls.exp() * sls).sum())
            logps.append(lp)
            ents.append(ent)
        v = self.net.value(z).squeeze(1)
        return torch.stack(logps), torch.stack(ents), v

    def maybe_update(self, force: bool = False) -> dict:
        if self.frozen:
            return {"updates": 0, "changed": False, "reason": "frozen"}
        if not force and self.pending() < int(self.hp["update_every"]) * max(1, len(self.buffer)):
            return {"updates": 0, "changed": False, "reason": "collecting"}
        return self.update()

    def update(self) -> dict:
        torch = _torch()
        hp = self.hp
        streams = {k: v for k, v in self.buffer.items() if v}
        if not streams:
            return {"updates": 0, "changed": False, "reason": "empty"}
        recs, adv, ret = [], [], []
        rs = np.array([r["r"] for v in streams.values() for r in v], dtype=np.float64)
        for c in self.controls:               # the same experience with every signal set to zero
            c.buffer = {k: [dict(r, r=0.0) for r in v] for k, v in streams.items()}
            c.update()
        info = {"identities": sorted(streams), "transitions": int(len(rs)), "signal_sum": float(rs.sum()),
                "signal_abs_max": float(np.abs(rs).max()) if len(rs) else 0.0,
                "decisions": [[r["identity"], r["t"]] for v in streams.values() for r in v],
                "version_before": self.version, "params_hash_before": self.hash}
        self.buffer = {}
        if not len(rs) or float(np.abs(rs).max()) == 0.0:
            self.skipped_no_signal += 1
            info.update(updates=0, changed=False, reason="no signal")
            self.update_log.append({k: v for k, v in info.items() if k != "decisions"})
            return info
        # evidence scale: a running root-mean-square of the signal (not per-batch normalisation)
        d = hp["scale_decay"]
        self.scale = math.sqrt(d * self.scale ** 2 + (1 - d) * float(np.mean(rs ** 2))) if self.scale else \
            float(np.sqrt(np.mean(rs ** 2)))
        scale = max(self.scale, hp["scale_floor"])
        for ident, v in streams.items():
            r = np.array([x["r"] for x in v], dtype=np.float64)
            vals = np.array([x["v"] for x in v], dtype=np.float64)
            g = hp["gamma"]
            if self.algo == "reinforce":
                G_ = np.zeros(len(r))
                acc = 0.0
                for i in range(len(r) - 1, -1, -1):
                    acc = r[i] + g * acc
                    G_[i] = acc
                a = G_ - self.baseline
                self.baseline = 0.9 * self.baseline + 0.1 * float(G_.mean())
                rt = G_
            elif self.algo == "a2c":
                boot = vals[-1]
                rt = np.zeros(len(r))
                acc = boot
                for i in range(len(r) - 1, -1, -1):
                    acc = r[i] + g * acc
                    rt[i] = acc
                a = rt - vals
            else:                                   # ppo: GAE
                nxt = np.append(vals[1:], vals[-1])
                delta = r + g * nxt - vals
                a = np.zeros(len(r))
                acc = 0.0
                for i in range(len(r) - 1, -1, -1):
                    acc = delta[i] + g * hp["lam"] * acc
                    a[i] = acc
                rt = a + vals
            recs += v
            adv.append(a / scale)
            ret.append(rt / scale)
        A = torch.tensor(np.concatenate(adv), dtype=torch.float32)
        R = torch.tensor(np.concatenate(ret), dtype=torch.float32)
        old = torch.tensor([r["logp"] for r in recs], dtype=torch.float32)
        epochs = int(hp["epochs"]) if self.algo == "ppo" else 1
        loss_v = 0.0
        for _ in range(epochs):
            logp, ent, val = self._logp_value(recs)
            if self.algo == "ppo":
                ratio = (logp - old).exp()
                pg = -torch.min(ratio * A, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * A).mean()
            else:
                pg = -(logp * A).mean()
            vl = ((val - R) ** 2).mean() if self.algo in ("a2c", "ppo") else torch.zeros(())
            loss = pg + hp["vf"] * vl - hp["ent"] * ent.mean()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), hp["max_grad"])
            self.opt.step()
            loss_v = float(loss.detach())
        new_hash = weights_hash(self.net)
        changed = new_hash != self.hash
        if changed:
            self.parent_hash, self.hash = self.hash, new_hash
            self.version += 1
            self.updates += 1
        info.update(updates=int(changed), changed=changed, loss=loss_v, params_hash_after=self.hash,
                    version_after=self.version, scale=scale, algo=self.algo, epochs=epochs,
                    adv_abs_mean=float(A.abs().mean()))
        self.update_log.append({k: v for k, v in info.items() if k != "decisions"})
        if self.on_update is not None:
            self.on_update(self, info)
        return info

    # ------------------------------------------------------------------ imitation (behaviour cloning)
    def imitate(self, demos: list[dict], *, epochs: int = 4, batch: int = 64, lr: float | None = None) -> dict:
        """Supervised pre-training on recorded demonstrations (records of `decide(..., chooser=...)`): maximise
        the log-probability of the demonstrated tool and argument choices. Uses the group's own optimiser and
        random generator; counts as updates (versions) like any other parameter change."""
        torch = _torch()
        if not demos:
            return {"updates": 0, "changed": False}
        opt = torch.optim.Adam(self.net.parameters(), lr=lr or self.hp.get("bc_lr", self.hp["lr"]))
        order_rng = np.random.default_rng([self.seed, 0xBC])
        losses, steps = [], 0
        h0 = self.hash
        for _ in range(int(epochs)):
            idx = order_rng.permutation(len(demos))
            for k in range(0, len(idx), int(batch)):
                recs = [demos[i] for i in idx[k:k + int(batch)]]
                logp, _, _ = self._logp_value(recs)
                loss = -logp.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.hp["max_grad"])
                opt.step()
                losses.append(float(loss.detach()))
                steps += 1
        new = weights_hash(self.net)
        changed = new != h0
        info = {"updates": int(changed), "changed": changed, "steps": steps, "demos": len(demos),
                "loss_first": losses[0], "loss_last": float(np.mean(losses[-max(1, len(losses) // 8):])),
                "params_hash_before": h0, "params_hash_after": new, "version_before": self.version,
                "algo": "behaviour_cloning", "transitions": len(demos), "signal_sum": 0.0, "signal_abs_max": 0.0,
                "identities": sorted({d["identity"] for d in demos}),
                "decisions": [[d["identity"], d["t"]] for d in demos]}
        if changed:
            self.parent_hash, self.hash = h0, new
            self.version += 1
            info["version_after"] = self.version
            if self.on_update is not None:
                self.on_update(self, info)
        self.update_log.append({k: v for k, v in info.items() if k != "decisions"})
        return info

    # ------------------------------------------------------------------ checkpoints
    def state_bytes(self) -> bytes:
        torch = _torch()
        buf = io.BytesIO()
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict(), "algo": self.algo,
                    "arch": self.arch, "hp": self.hp, "version": self.version, "hash": self.hash,
                    "baseline": self.baseline, "scale": self.scale, "gen": self.gen.get_state(),
                    "rng": self.rng.bit_generator.state, "lineage": self.lineage, "seed": self.seed}, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, gid: str, data: bytes, *, frozen: bool = True, seed: int | None = None) -> "PolicyGroup":
        torch = _torch()
        d = torch.load(io.BytesIO(data), weights_only=False)
        g = cls(gid, d["algo"], d["arch"], d["seed"] if seed is None else seed, d["hp"], lineage=d.get("lineage"))
        g.net.load_state_dict(d["net"])
        g.opt.load_state_dict(d["opt"])
        g.version, g.baseline, g.scale = int(d["version"]), float(d["baseline"]), float(d["scale"])
        if seed is None:
            g.gen.set_state(d["gen"])
            g.rng.bit_generator.state = d["rng"]
        g.hash = weights_hash(g.net)
        if g.hash != d["hash"]:
            raise ValueError(f"checkpoint of {gid}: weights hash {g.hash[:12]} != recorded {d['hash'][:12]}")
        g.frozen = bool(frozen)
        return g

    def clone(self, gid: str) -> "PolicyGroup":
        """An exact copy (parameters, optimiser, generators, counters) that shares nothing with this group."""
        g = PolicyGroup.from_bytes(gid, self.state_bytes(), frozen=self.frozen)
        g.updates, g.skipped_no_signal, g.lineage = self.updates, self.skipped_no_signal, self.lineage
        return g

    def tool_probs_of(self, rec: dict):
        """This policy's tool distribution in a recorded situation (a decision record): {tool: probability}."""
        torch = _torch()
        with torch.no_grad():
            x = torch.from_numpy(rec["x"][None, :])
            h = None if rec.get("h") is None or self.arch != "gru" else torch.from_numpy(rec["h"][None, :])
            z, _ = self.net.encode(x, h)
            lp, _ = tool_log_probs(self.net, z, rec["ttoks"], torch.from_numpy(rec["tflags"]), rec["htoks"],
                                   rec["hidx"])
        names = rec.get("names") or [str(i) for i in range(len(rec["ttoks"]))]
        return {n: float(p) for n, p in zip(names, lp.exp())}

    def frozen_copy(self, gid: str, seed: int | None = None) -> "PolicyGroup":
        return PolicyGroup.from_bytes(gid, self.state_bytes(), frozen=True, seed=seed)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))
