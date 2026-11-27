"""`ReferenceLearner`: a small recurrent neural learner over structured observations, for every tier.

It exists to prove the interface, not to be good: it consumes `Observation`s, emits ``{"tool", "args"}`` actions
for any tool list, stores experience, performs real gradient updates and checkpoints exactly. It is architecture
independent in the sense that matters here: it uses only the public `crucible.experience` interface (the same one
documented for CAM and any other learner in docs/v5/LEARNER_GUIDE.md).

Model (about 0.2M parameters with the defaults)::

    x_t   = hashed features of the observation (numbers, arrays, events, tools, last result, resources, missing, t)
    h_t   = GRUCell([x_t, e(a_{t-1})], h_{t-1})
    tool  ~ softmax_i( q(h_t) . e(tool_i) )          e(.) = EmbeddingBag over hashed name / host / category tokens;
                                                       option 0 is "wait" (no action this step)
    args  : filled from the tool's JSON schema - bounded numbers lo + sigmoid(z)(hi - lo), z ~ N(mu(h, e(tool))[slot],
            sigma); enums and id-like strings ~ softmax over candidates (entity ids; ids seen in its own results
            and visible events under the same key - a ``listing`` argument draws from the ``listing`` ids it has
            seen; names seen in results);
            maps of numbers (e.g. rig.set settings) over the parameter specs (``{name, min, max}``) seen in results
    V(h_t), and a world-model head that predicts x_{t+1} from (h_t, e(a_t))

Update: PPO-clip on the stored transitions (log-probabilities of the actions actually taken are recomputed for the
current parameters), a value loss, an entropy bonus and the self-supervised next-feature loss. The world-model loss
gives real parameter updates even when no reward exists.

Learning signal (`signal`): ``"reward"`` uses `env.reward(agent)` (legacy or verified modes); ``"observed"`` is the
learner's *own* evaluation computed from legitimate observations - by default the change of its own balances
(``bank.balances.*``) - which is how this learner trains in observation-only mode; ``"none"`` trains the world model
only. ``"auto"`` picks ``"observed"`` in observation-only mode and ``"reward"`` otherwise. Choosing what to value is
the learner's business (MASTER_SPEC 11.3); Crucible computes no shaped reward for it.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .features import is_id, id_kind
from .modes import OBSERVATION_ONLY

VERSION = "reference/1"


def _h(s: str, n: int) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") % n


def _symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


_NUM_RE = re.compile(r"\d+")


def norm_key(k: str) -> str:
    """A feature path with ids replaced by their kind and digits collapsed, so features carry across worlds."""
    parts = []
    for seg in re.split(r"([.\[\]])", k):
        if is_id(seg):
            parts.append(f"<{id_kind(seg)}>")
        else:
            parts.append(_NUM_RE.sub("#", seg))
    return "".join(parts)


@dataclass
class Config:
    feat_dim: int = 256
    hidden: int = 96
    emb: int = 32
    tok_buckets: int = 1024
    num_slots: int = 16
    lr: float = 3e-4
    gamma: float = 0.97
    lam: float = 0.9
    clip: float = 0.2
    epochs: int = 2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    wm_coef: float = 0.2
    init_log_std: float = -0.5
    signal: str = "auto"
    signal_keys: str = r"^bank\.balances\."
    signal_scale: float = 1e-4
    max_events: int = 200
    threads: int = 1
    wait_bias: float = 0.0


class ReferenceLearner:
    name = "reference"
    kind = "rl"
    version = VERSION

    def __init__(self, seed: int = 0, reward_mode: str | None = None, device: str = "cpu", **cfg):
        import torch
        from .modes import check_mode
        self.cfg = Config(**cfg)
        torch.set_num_threads(int(self.cfg.threads))
        self.seed = int(seed)
        self.reward_mode = check_mode(reward_mode)      # observation_only unless named (Clarification 2)
        self.device = device
        torch.manual_seed(self.seed)
        self.net = _Net(self.cfg).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.gen = torch.Generator().manual_seed(self.seed + 1)
        self.rng = np.random.default_rng([self.seed, 17])
        self.frozen = False
        self.updates = 0
        self.steps = 0
        self.memory: dict[str, dict] = {}       # parameter specs seen in results: name -> {min, max, integer}
        # ids seen in its own results and visible events, by the key they appeared under (most recent last); an
        # episodic memory of the current world: checkpointed with the learner's state, never part of its weights
        self.ids: dict[str, list[str]] = {}
        self.reset()

    # ------------------------------------------------------------------ context
    def reset(self, actor: str | None = None, rng=None) -> None:
        """Start of a new life, episode, agent or world: clears the recurrent state (never weights). Transitions
        already buffered are kept for the next `update`; the next one is marked as starting from a fresh recurrent
        state, so the update replays the sequence with the same resets the learner experienced."""
        import torch
        self.actor = actor
        self.h = torch.zeros(1, self.cfg.hidden)
        self.prev_tokens: list[int] = []
        if not getattr(self, "buf", None):
            self.buf: list[dict] = []
            self.h0 = self.h.clone()
            self._boundary = False
        else:
            self._boundary = True
        self._pending: dict | None = None
        self._last_value: float | None = None

    def params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def params_hash(self) -> str:
        """SHA-256 over the sorted parameter tensors (name, dtype, shape, bytes)."""
        h = hashlib.sha256()
        for k, v in sorted(self.net.state_dict().items()):
            a = v.detach().cpu().contiguous().numpy()
            h.update(k.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def info(self) -> dict:
        return {"name": self.name, "version": self.version, "params": self.params(), "params_hash": self.params_hash(),
                "updates": self.updates}

    # ------------------------------------------------------------------ features
    def features(self, obs: dict) -> np.ndarray:
        d = self.cfg.feat_dim
        v = np.zeros(d, np.float32)

        def add(key: str, x: float) -> None:
            if math.isfinite(x):
                v[_h(key, d)] += _symlog(x)

        for k, x in (obs.get("numbers") or {}).items():
            nk = norm_key(k)
            add("n:" + nk, float(x))
            add("p:" + nk, 1.0)
        for k, arr in (obs.get("arrays") or {}).items():
            flat = np.asarray(arr, dtype=np.float64).ravel()
            if flat.size:
                nk = norm_key(k)
                add("a:" + nk + ":mean", float(flat.mean()))
                add("a:" + nk + ":last", float(flat[-1]))
                add("a:" + nk + ":n", float(flat.size))
        evs = obs.get("events") or []
        for e in evs[-self.cfg.max_events:]:
            t = e.get("type", "?")
            add("e:" + t, 1.0)
            if e.get("cause"):
                add("e:caused:" + t, 1.0)
            pl = e.get("payload")
            if isinstance(pl, dict):
                for k2, x in pl.items():
                    if isinstance(x, (int, float)) and not isinstance(x, bool):
                        add(f"e:{t}.{k2}", float(x))
        add("e:#", float(len(evs)))
        tools = obs.get("tools") or []
        add("tools:#", float(len(tools)))
        for t in tools:
            add("tc:" + str((t.get("annotations") or {}).get("category")), 1.0)
        lr = obs.get("last_result")
        if lr:
            add(f"r:{lr.get('tool')}:{'ok' if lr.get('ok') else (lr.get('error') or {}).get('code')}", 1.0)
            add("r:ok" if lr.get("ok") else "r:err", 1.0)
        for k, x in (obs.get("resources") or {}).items():
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                add("res:" + k, float(x))
        for k, x in (obs.get("analytics") or {}).items():
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                add("an:" + norm_key(k), float(x))
        add("missing:#", float(len(obs.get("missing") or [])))
        t = int(obs.get("t", 0))
        v[_h("t:sin", d)] += math.sin(2 * math.pi * (t % 86400) / 86400)
        v[_h("t:cos", d)] += math.cos(2 * math.pi * (t % 86400) / 86400)
        add("t:hours", t / 3600.0)
        return v

    def _signal_value(self, obs: dict) -> float:
        """The learner's own valuation of a state: its money as the observation shows it - the numbers matching
        `signal_keys` (kernel and mechanism worlds: ``bank.balances.*``), else its own resources' money (rapid
        worlds report ``resources.cash_cents``)."""
        pat = re.compile(self.cfg.signal_keys)
        hits = [float(x) for k, x in (obs.get("numbers") or {}).items() if pat.search(k)]
        if hits:
            return float(sum(hits))
        res = obs.get("resources") or {}
        for k in ("balance_cents", "cash_cents"):
            v = res.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return 0.0

    def signal_mode(self) -> str:
        s = self.cfg.signal
        if s == "auto":
            return "observed" if self.reward_mode == OBSERVATION_ONLY else "reward"
        return s

    # ------------------------------------------------------------------ memory of parameter specs and ids
    MAX_ID_KEYS = 512
    IDS_PER_KEY = 64

    def _remember_id(self, key: str, v: str) -> None:
        lst = self.ids.get(key)
        if lst is None:
            if len(self.ids) >= self.MAX_ID_KEYS:
                return
            lst = self.ids[key] = []
        elif v in lst:
            lst.remove(v)
        lst.append(v)
        if len(lst) > self.IDS_PER_KEY:
            del lst[0]

    def _harvest(self, x: Any, depth: int = 0, key: str | None = None) -> None:
        """Remember parameter specs (``{name, min, max}``) and the ids a result or event shows, by key: ``{"listing":
        "xl_5"}`` under ``listing``; an ``id`` inside a list under ``orders`` also under ``order``; a list of ids
        under its key's singular."""
        if depth > 6:
            return
        if isinstance(x, dict):
            if isinstance(x.get("name"), str) and isinstance(x.get("min"), (int, float)) and \
                    isinstance(x.get("max"), (int, float)) and not isinstance(x.get("min"), bool):
                self.memory[x["name"]] = {"min": float(x["min"]), "max": float(x["max"]),
                                          "integer": bool(x.get("integer", False))}
            for k, v in x.items():
                if isinstance(v, str):
                    if is_id(v):
                        self._remember_id(str(k), v)
                        if k == "id" and key:
                            self._remember_id(_singular(key), v)
                else:
                    self._harvest(v, depth + 1, str(k))
        elif isinstance(x, list):
            for v in x[:256]:
                if isinstance(v, str):
                    if key and is_id(v):
                        self._remember_id(_singular(key), v)
                else:
                    self._harvest(v, depth + 1, key)

    def _harvest_events(self, obs: dict) -> None:
        for e in (obs.get("events") or [])[-50:]:
            if isinstance(e.get("payload"), dict):
                self._harvest(e["payload"], 1)

    # ------------------------------------------------------------------ acting
    def tool_tokens(self, t: dict) -> list[int]:
        n = self.cfg.tok_buckets
        name = t.get("name", "")
        host, _, ep = name.partition(".")
        cat = str((t.get("annotations") or {}).get("category"))
        return [_h("tool:" + name, n), _h("host:" + host, n), _h("ep:" + ep, n), _h("cat:" + cat, n)]

    def act(self, obs: dict, tools: list[dict] | None = None, explore: bool = True) -> dict | None:
        """Choose an action (or None: wait) for `obs`. `tools` defaults to the observation's tool list."""
        import torch
        tools = list(tools if tools is not None else obs.get("tools") or [])
        self._harvest(obs.get("last_result"))
        self._harvest_events(obs)
        x = torch.from_numpy(self.features(obs)).unsqueeze(0)
        with torch.no_grad():
            h = self.net.step(x, self.prev_tokens, self.h)
            action, logp, ent, toks = _Decider(self, h, obs, tools, explore=explore).decide()
            value = float(self.net.value(h))
        self._pending = {"obs": obs, "tools": tools, "action": action, "logp": float(logp), "value": value,
                         "prev_tokens": list(self.prev_tokens), "signal0": self._signal_value(obs),
                         "h_after": h.clone(), "tokens_after": list(toks), "reset": bool(self._boundary)}
        self._boundary = False
        self.h = h
        self.prev_tokens = toks
        self.steps += 1
        return action

    def observe(self, reward: float, result: dict | None = None, next_obs: dict | None = None,
                done: bool = False) -> None:
        """Complete the transition started by the last `act`: the reward the environment gave (0.0 in
        observation-only mode), the action's result and the next observation."""
        p, self._pending = self._pending, None
        if p is None:
            return
        if result is not None:
            self._harvest(result.get("result") if isinstance(result, dict) else None)
        mode = self.signal_mode()
        if mode == "reward":
            sig = float(reward)
        elif mode == "observed" and next_obs is not None:
            sig = (self._signal_value(next_obs) - p["signal0"]) * self.cfg.signal_scale
        else:
            sig = 0.0
        if self.frozen:
            return
        p.update(signal=float(sig), done=bool(done), next_obs=next_obs)
        self.buf.append(p)

    # Policy interface (agents.base.Policy / scale.rollout workers)
    def observe_result(self, action: dict, result: dict) -> None:
        self._harvest(result.get("result") if isinstance(result, dict) else None)

    def freeze(self, frozen: bool = True) -> None:
        self.frozen = bool(frozen)

    # ------------------------------------------------------------------ learning
    def update(self) -> dict:
        """PPO-clip + value + entropy + world-model step(s) on the buffered transitions; clears the buffer."""
        import torch
        if self.frozen or len(self.buf) < 2:
            return {"updates": 0, "transitions": len(self.buf)}
        before = self.params_hash()
        buf = self.buf
        T = len(buf)
        sig = np.array([b["signal"] for b in buf], np.float64)
        # V(s_{t+1}) for every t: the next transition's value when it continues this one; otherwise (the last one, or
        # one followed by a recurrent reset) the value of its own next observation under the state it left behind
        nxt = np.zeros(T)
        for t in range(T):
            b = buf[t]
            if b["done"]:
                continue
            if t + 1 < T and not buf[t + 1].get("reset"):
                nxt[t] = buf[t + 1]["value"]
            else:
                nxt[t] = self._bootstrap(b)
        vals = np.array([b["value"] for b in buf], np.float64)
        adv = np.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            nonterm = 0.0 if buf[t]["done"] else 1.0
            cont = 0.0 if (t + 1 >= T or buf[t + 1].get("reset")) else 1.0
            delta = sig[t] + self.cfg.gamma * nxt[t] * nonterm - vals[t]
            last = delta + self.cfg.gamma * self.cfg.lam * nonterm * cont * last
            adv[t] = last
        ret = adv + vals
        if T > 1 and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        X = torch.from_numpy(np.stack([self.features(b["obs"]) for b in buf]))
        Xn = torch.from_numpy(np.stack([self.features(b["next_obs"]) if b.get("next_obs") is not None
                                        else self.features(b["obs"]) for b in buf]))
        adv_t, ret_t = torch.tensor(adv, dtype=torch.float32), torch.tensor(ret, dtype=torch.float32)
        old = torch.tensor([b["logp"] for b in buf], dtype=torch.float32)
        stats = {}
        n_up = 0
        for _ in range(self.cfg.epochs):
            h = self.h0
            logps, ents, values, preds = [], [], [], []
            for t, b in enumerate(buf):
                if b.get("reset"):
                    h = torch.zeros(1, self.cfg.hidden)
                h = self.net.step(X[t:t + 1], b["prev_tokens"], h)
                lp, ent, toks = _Decider(self, h, b["obs"], b["tools"]).score(b["action"])
                logps.append(lp)
                ents.append(ent)
                values.append(self.net.value(h))
                preds.append(self.net.predict(h, toks))
            logp = torch.stack(logps)
            ratio = torch.exp(logp - old)
            pg = -torch.min(ratio * adv_t, torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv_t).mean()
            vloss = ((torch.stack(values).reshape(-1) - ret_t) ** 2).mean()
            wm = ((torch.cat(preds) - Xn) ** 2).mean()
            ent = torch.stack(ents).mean()
            loss = pg + self.cfg.vf_coef * vloss + self.cfg.wm_coef * wm - self.cfg.ent_coef * ent
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            n_up += 1
            stats = {"loss": loss.item(), "pg": pg.item(), "value_loss": vloss.item(), "world_model_loss": wm.item(),
                     "entropy": ent.item()}
        self.updates += n_up
        self.buf = []
        self.h0 = self.h.detach().clone()
        self._boundary = False
        after = self.params_hash()
        return {"updates": n_up, "transitions": T, "signal_sum": float(sig.sum()), "params_hash_before": before,
                "params_hash_after": after, "changed": before != after, **stats}

    def learn_from_records(self, records, batch: int = 64) -> dict:
        """Train on stored experience records (`ExperienceStore.iterate()` output, any learner's): consecutive
        records of one agent in one world become transitions; the stored action is scored under the current
        parameters. The recurrent state is reset whenever the agent or the world changes."""
        out = {"updates": 0, "transitions": 0, "batches": 0, "changed": False, "resets": 0}
        prev = None
        self.buf = []
        self.reset()
        first_hash = self.params_hash()
        for r in records:
            obs = r["observation"]
            if prev is not None and prev["agent"] == r["agent"] and prev["env"] == r["env"] and \
                    not prev.get("done") and prev.get("episode", 0) == r.get("episode", 0):
                self._push_record(prev, obs)
            elif prev is not None:
                self._push_record(prev, None)            # the last record of that agent/world: no successor
                self._reset_recurrent()
                out["resets"] += 1
            prev = r
            if len(self.buf) >= batch:
                u = self.update()
                out["updates"] += u["updates"]
                out["transitions"] += u["transitions"]
                out["batches"] += 1
        if prev is not None:
            self._push_record(prev, None)
        if len(self.buf) >= 2:
            u = self.update()
            out["updates"] += u["updates"]
            out["transitions"] += u["transitions"]
            out["batches"] += 1
        out["changed"] = self.params_hash() != first_hash
        out["params_hash"] = self.params_hash()
        return out

    def _reset_recurrent(self) -> None:
        import torch
        self.h = torch.zeros(1, self.cfg.hidden)
        self.prev_tokens = []
        self._boundary = True

    def _bootstrap(self, b: dict) -> float:
        """V(s_{t+1}) of a transition from its stored next observation and the recurrent state its action left."""
        import torch
        if b.get("next_obs") is None or b.get("h_after") is None:
            return 0.0
        with torch.no_grad():
            x = torch.from_numpy(self.features(b["next_obs"])).unsqueeze(0)
            h1 = self.net.step(x, b["tokens_after"], b["h_after"])
            return float(self.net.value(h1))

    def _push_record(self, r: dict, next_obs: dict | None) -> None:
        import torch
        obs = r["observation"]
        tools = obs.get("tools") or []
        self._harvest((r.get("result") or {}).get("result"))
        x = torch.from_numpy(self.features(obs)).unsqueeze(0)
        with torch.no_grad():
            h = self.net.step(x, self.prev_tokens, self.h)
            lp, _, toks = _Decider(self, h, obs, tools).score(r.get("action"))
            value = float(self.net.value(h))
        mode = self.signal_mode()
        if mode == "reward":
            sig = float(r.get("reward", 0.0))
        elif mode == "observed" and next_obs is not None:
            sig = (self._signal_value(next_obs) - self._signal_value(obs)) * self.cfg.signal_scale
        else:
            sig = 0.0
        self.buf.append({"obs": obs, "tools": tools, "action": r.get("action"), "logp": float(lp), "value": value,
                         "prev_tokens": list(self.prev_tokens), "signal": sig, "done": next_obs is None,
                         "next_obs": next_obs, "h_after": h.clone(), "tokens_after": list(toks),
                         "reset": bool(self._boundary)})
        self._boundary = False
        self.h, self.prev_tokens = h, toks

    # ------------------------------------------------------------------ checkpoints
    def state(self) -> dict:
        return {"format": VERSION, "config": asdict(self.cfg), "seed": self.seed, "reward_mode": self.reward_mode,
                "net": self.net.state_dict(), "opt": self.opt.state_dict(), "gen": self.gen.get_state(),
                "rng": self.rng.bit_generator.state, "memory": dict(self.memory),
                "ids": {k: list(v) for k, v in self.ids.items()}, "updates": self.updates,
                "steps": self.steps,
                # the learner's context: a mid-episode save resumes with the same recurrent state, the same
                # buffered transitions and the same transition in flight
                "context": {"h": self.h, "h0": self.h0, "prev_tokens": list(self.prev_tokens), "buf": list(self.buf),
                            "pending": self._pending, "boundary": bool(self._boundary), "actor": self.actor,
                            "frozen": bool(self.frozen)}}

    def save(self, path: str) -> None:
        import torch
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(self.state(), tmp)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        import torch
        self._load_state(torch.load(path, map_location="cpu", weights_only=True))

    def _load_state(self, st: dict) -> None:
        if st.get("format") != VERSION:
            raise ValueError(f"not a {VERSION} checkpoint")
        if asdict(self.cfg) != st["config"]:
            self.cfg = Config(**st["config"])
            self.net = _Net(self.cfg)
            import torch
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.net.load_state_dict(st["net"])
        self.opt.load_state_dict(st["opt"])
        self.gen.set_state(st["gen"])
        self.rng.bit_generator.state = st["rng"]
        self.memory = dict(st["memory"])
        self.ids = {k: list(v) for k, v in (st.get("ids") or {}).items()}
        self.updates, self.steps = int(st["updates"]), int(st["steps"])
        self.seed, self.reward_mode = int(st["seed"]), st["reward_mode"]
        c = st.get("context")
        if c is not None:
            self.h, self.h0 = c["h"], c["h0"]
            self.prev_tokens, self.buf = list(c["prev_tokens"]), list(c["buf"])
            self._pending, self._boundary = c["pending"], bool(c["boundary"])
            self.actor, self.frozen = c["actor"], bool(c["frozen"])

    def params_blob(self) -> bytes:
        import torch
        b = io.BytesIO()
        torch.save({"config": asdict(self.cfg), "net": self.net.state_dict(), "memory": dict(self.memory)}, b)
        return b.getvalue()

    def load_params(self, blob: bytes) -> None:
        import torch
        if not blob:
            return
        st = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
        if asdict(self.cfg) != st["config"]:
            self.cfg = Config(**st["config"])
            self.net = _Net(self.cfg)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.net.load_state_dict(st["net"])
        self.memory.update(st.get("memory") or {})


# ====================================================================== the network
def _make_net_class():
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self, cfg: Config):
            super().__init__()
            self.cfg = cfg
            H, E = cfg.hidden, cfg.emb
            self.tok = nn.EmbeddingBag(cfg.tok_buckets, E, mode="mean")
            self.inp = nn.Linear(cfg.feat_dim + E, H)
            self.gru = nn.GRUCell(H, H)
            self.q = nn.Linear(H, E)
            self.wait = nn.Parameter(torch.zeros(E))
            self.num = nn.Linear(H + E, cfg.num_slots)
            self.log_std = nn.Parameter(torch.full((cfg.num_slots,), float(cfg.init_log_std)))
            self.v = nn.Linear(H, 1)
            self.wm = nn.Sequential(nn.Linear(H + E, H), nn.Tanh(), nn.Linear(H, cfg.feat_dim))

        def embed(self, tokens: list[list[int]]):
            if not tokens:
                return torch.zeros(0, self.cfg.emb)
            flat = torch.tensor([t for ts in tokens for t in ts], dtype=torch.long)
            offs = torch.tensor(np.cumsum([0] + [len(ts) for ts in tokens[:-1]]), dtype=torch.long)
            return self.tok(flat, offs)

        def step(self, x, prev_tokens: list[int], h):
            e = self.embed([prev_tokens]) if prev_tokens else torch.zeros(1, self.cfg.emb)
            z = torch.tanh(self.inp(torch.cat([x, e], dim=1)))
            return self.gru(z, h)

        def value(self, h):
            return self.v(h).reshape(())

        def predict(self, h, tokens: list[int]):
            e = self.embed([tokens]) if tokens else torch.zeros(1, self.cfg.emb)
            return self.wm(torch.cat([h, e], dim=1))

    return Net


_NET = None


def _Net(cfg: Config):
    global _NET
    if _NET is None:
        _NET = _make_net_class()
    return _NET(cfg)


# ====================================================================== deciding and scoring actions
class _Decider:
    """Samples an action (`decide`) or scores a given one (`score`) with the same distribution, so stored actions
    (this learner's or anyone's) can be trained on."""

    def __init__(self, L: ReferenceLearner, h, obs: dict, tools: list[dict], explore: bool = True):
        self.L, self.h, self.obs, self.tools, self.explore = L, h, obs, tools, explore
        self.lps: list = []
        self.ents: list = []

    # -- distributions
    def _cat(self, logits, choice: int | None):
        import torch
        dist = torch.distributions.Categorical(logits=logits)
        if choice is None:
            if self.explore:
                choice = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=self.L.gen))
            else:
                choice = int(torch.argmax(logits))
        self.lps.append(dist.log_prob(torch.tensor(choice)))
        self.ents.append(dist.entropy())
        return choice

    def _gauss(self, mu, log_std, z: float | None):
        import torch
        std = torch.exp(log_std)
        if z is None:
            eps = torch.randn((), generator=self.L.gen) if self.explore else torch.zeros(())
            z = float(mu + std * eps)
        dist = torch.distributions.Normal(mu, std)
        self.lps.append(dist.log_prob(torch.tensor(float(z))))
        self.ents.append(dist.entropy())
        return z

    # -- the tool
    def _tool_logits(self):
        import torch
        net = self.L.net
        emb = net.embed([self.L.tool_tokens(t) for t in self.tools])
        cand = torch.cat([net.wait.unsqueeze(0), emb], dim=0)
        logits = (cand @ net.q(self.h).reshape(-1)) / math.sqrt(self.L.cfg.emb)
        return logits + torch.tensor([self.L.cfg.wait_bias] + [0.0] * len(self.tools)), emb

    def decide(self):
        logits, emb = self._tool_logits()
        k = self._cat(logits, None)
        if k == 0:
            return None, self._total(), self._ent(), []
        tool = self.tools[k - 1]
        args = self._fill(tool, emb[k - 1:k], None)
        return {"tool": tool["name"], "args": args}, self._total(), self._ent(), self.L.tool_tokens(tool)

    def score(self, action: dict | None):
        import torch
        logits, emb = self._tool_logits()
        if action is None:
            self._cat(logits, 0)
            return self._total(), self._ent(), []
        names = [t.get("name") for t in self.tools]
        if action.get("tool") not in names:
            # a tool that is not offered in this observation (e.g. a stored action from another world): no
            # probability mass to train; it contributes a constant
            return torch.zeros(()), torch.zeros(()), []
        k = names.index(action["tool"]) + 1
        self._cat(logits, k)
        tool = self.tools[k - 1]
        self._fill(tool, emb[k - 1:k], action.get("args") or {})
        return self._total(), self._ent(), self.L.tool_tokens(tool)

    def _total(self):
        import torch
        return torch.stack(self.lps).sum() if self.lps else torch.zeros(())

    def _ent(self):
        import torch
        return torch.stack(self.ents).sum() if self.ents else torch.zeros(())

    # -- arguments
    def _fill(self, tool: dict, temb, given: dict | None) -> dict:
        import torch
        net = self.L.net
        self.mu = net.num(torch.cat([self.h, temb], dim=1)).reshape(-1)
        self.tool_name = tool.get("name", "")
        schema = tool.get("inputSchema") or {}
        return self._obj(schema, "", given, top=True)

    def _slot(self, path: str) -> int:
        return _h(f"{self.tool_name}|{path}", self.L.cfg.num_slots)

    def _obj(self, schema: dict, path: str, given: dict | None, top: bool = False) -> dict:
        props = schema.get("properties") or {}
        req = set(schema.get("required") or [])
        out = {}
        for k in sorted(props):
            if given is not None and k not in given:
                continue                            # scoring a given action: only what it actually contains
            if given is None and k not in req:
                continue
            v = self._value(props[k], f"{path}.{k}", k, None if given is None else given.get(k))
            if v is not _SKIP:
                out[k] = v
        addl = schema.get("additionalProperties")
        if not props and isinstance(addl, dict) and addl.get("type") in ("number", "integer"):
            # a map of numbers (e.g. lever settings): the parameter specs remembered from results
            keys = sorted(given) if given is not None else sorted(self.L.memory)
            for key in keys:
                sp = self.L.memory.get(key, {"min": 0.0, "max": 1.0, "integer": addl.get("type") == "integer"})
                v = self._num(sp["min"], sp["max"], sp.get("integer", False), f"{path}[{key}]",
                              None if given is None else given.get(key))
                out[key] = v
        return out

    def _value(self, s: dict, path: str, key: str, given: Any):
        t = s.get("type")
        t = t[0] if isinstance(t, list) else t
        if "enum" in s:
            vals = list(s["enum"])
            logits = self._choice_logits([str(v) for v in vals], path)
            idx = vals.index(given) if given is not None and given in vals else None
            if given is not None and idx is None:
                return given
            return vals[self._cat(logits, idx)]
        if t in ("number", "integer"):
            lo, hi = s.get("minimum"), s.get("maximum")
            if lo is not None and hi is not None:
                return self._num(float(lo), float(hi), t == "integer", path, given)
            if given is None and "default" in s:
                return s["default"]
            return self._unbounded(lo, hi, t == "integer", path, given)
        if t == "boolean":
            import torch
            logit = self.mu[self._slot(path)]
            idx = None if given is None else int(bool(given))
            return bool(self._cat(torch.stack([torch.zeros(()), logit]), idx))
        if t == "string":
            cands = self._string_candidates(key)
            if not cands:
                return given if given is not None else (s.get("default") or self.L.actor or "x")
            idx = cands.index(given) if given is not None and given in cands else None
            if given is not None and idx is None:
                return given
            return cands[self._cat(self._choice_logits(cands, path), idx)]
        if t == "object":
            return self._obj(s, path, given if isinstance(given, dict) else None)
        if t == "array":
            return given if given is not None else []
        return given if given is not None else _SKIP

    def _choice_logits(self, cands: list[str], path: str):
        net, n = self.L.net, self.L.cfg.tok_buckets
        emb = net.embed([[_h(f"c:{path}:{c}", n), _h(f"k:{id_kind(c) if is_id(c) else c}", n)] for c in cands])
        return (emb @ net.q(self.h).reshape(-1)) / math.sqrt(self.L.cfg.emb)

    def _num(self, lo: float, hi: float, integer: bool, path: str, given: Any):
        k = self._slot(path)
        z = None
        if given is not None:
            if hi <= lo:
                return given
            u = min(1 - 1e-4, max(1e-4, (float(given) - lo) / (hi - lo)))
            z = math.log(u / (1 - u))
        z = self._gauss(self.mu[k], self.L.net.log_std[k], z)
        if given is not None:
            return given
        v = lo + (hi - lo) / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        return float(round(v)) if integer else float(v)

    def _unbounded(self, lo, hi, integer: bool, path: str, given: Any):
        k = self._slot(path)
        z = None if given is None else _symlog(float(given)) / 2.0
        z = self._gauss(self.mu[k], self.L.net.log_std[k], z)
        if given is not None:
            return given
        v = math.copysign(math.expm1(min(20.0, abs(2.0 * z))), z)
        if lo is not None:
            v = max(float(lo), v)
        if hi is not None:
            v = min(float(hi), v)
        return int(round(v)) if integer else float(v)

    def _string_candidates(self, key: str) -> list[str]:
        obs = self.obs
        ids = {e["id"] for e in obs.get("entities") or [] if isinstance(e.get("id"), str)}
        for k in (obs.get("numbers") or {}):
            for seg in k.split("."):
                if is_id(seg):
                    ids.add(seg)
        want = {"account": "account", "from_account": "account", "to_account": "account", "org": "organisation",
                "owner": None, "to": None, "principal": None, "seller": None}.get(key, "-")
        seen = self.L.ids.get(key) or []
        if want == "-":
            names = sorted(self.L.memory) if key in ("name", "lever", "target") else []
            return (names + sorted(set(seen) - set(names)))[:64]
        out = sorted({i for i in ids if want is None or id_kind(i) == want} |
                     {i for i in seen if want is None or id_kind(i) == want})
        return out[:64]


_SKIP = object()


def _singular(key: str) -> str:
    if key.endswith("ies") and len(key) > 4:
        return key[:-3] + "y"
    if key.endswith("s") and not key.endswith("ss") and len(key) > 2:
        return key[:-1]
    return key
