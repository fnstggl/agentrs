"""An architecture-neutral external learner that supplies its OWN training reward (Clarification 2, OO.6).

It is written only against the public `crucible.experience` interface (plus numpy), changes nothing in the
environment, and does not use Crucible's reward at all: in the default ``observation_only`` mode the environment pays
0.0 and this learner ignores it either way. Its reward is computed from what it observes:

    intrinsic   prediction-error curiosity: a forward model predicts the next observation's features from the
                current ones and the action; the squared error of that prediction is the surprise of the step
    extrinsic   (optional, ``money_weight``) the learner's own valuation of an observed consequence: the change of
                the money its observation shows (balances, cash), a quantity any operator sees in its own books

    r_t = ||phi(o_{t+1}) - F(phi(o_t), a_t)||^2 / D  +  money_weight * tanh(delta_money / money_scale)

Parameters (all numpy arrays, hashed by `params_hash()`): the forward model F (D x (D + A)), a per-tool linear
preference over features (tool logits), and per-(tool, argument) linear heads for bounded numeric arguments.
`update()` fits F to the buffered transitions (SGD on the squared error) and moves the policy by REINFORCE with a
running baseline on the learner's own reward. Different architectures plug in the same way: whatever a learner
values, it computes it from observations and actions.

    from curiosity_learner import CuriosityLearner, run
    from crucible.experience import make
    env = make(1, "rapid:net_routing", 3)               # observation_only by default
    L = CuriosityLearner(seed=0)
    summary = run(env, L, steps=200)                   # summary["param_updates_changed"] counts real updates
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from crucible.experience import validate_observation

D = 64          # observation feature dimension (hashed)
A = 16          # action feature dimension (hashed tool name + argument summary)


def _h(s: str, n: int) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") % n


def _squash(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def features(obs: dict) -> np.ndarray:
    """Hashed features of an observation: its exact numbers (log-scaled), its resources' numbers, counts of event
    types and of tool names, and the time of day."""
    v = np.zeros(D)
    for k, x in (obs.get("numbers") or {}).items():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            v[_h("n:" + k, D)] += _squash(float(x)) / 4.0
    for k, x in (obs.get("resources") or {}).items():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            v[_h("r:" + k, D)] += _squash(float(x)) / 4.0
    for e in obs.get("events") or []:
        v[_h("e:" + str(e.get("type")), D)] += 0.25
    lr = obs.get("last_result")
    if isinstance(lr, dict):
        v[_h("ok" if lr.get("ok") else "err", D)] += 1.0
    t = float(obs.get("t") or 0)
    v[_h("tod", D)] += math.sin(2 * math.pi * (t % 86400) / 86400)
    return v


def money(obs: dict) -> float:
    nums = obs.get("numbers") or {}
    bal = [float(x) for k, x in nums.items() if k.startswith("bank.balances.") and isinstance(x, (int, float))]
    if bal:
        return float(sum(bal))
    res = obs.get("resources") or {}
    for k in ("balance_cents", "cash_cents"):
        if isinstance(res.get(k), (int, float)) and not isinstance(res.get(k), bool):
            return float(res[k])
    return 0.0


class CuriosityLearner:
    name = "external-curiosity"
    version = "1"

    def __init__(self, seed: int = 0, lr_model: float = 0.02, lr_policy: float = 0.05, sigma: float = 0.15,
                 money_weight: float = 0.0, money_scale: float = 10_000.0, max_string_choices: int = 8):
        self.rng = np.random.default_rng(seed)
        self.F = self.rng.normal(0.0, 0.01, size=(D, D + A))
        self.tool_w: dict[str, np.ndarray] = {}
        self.tool_b: dict[str, float] = {}
        self.arg_w: dict[str, np.ndarray] = {}
        self.lr_model, self.lr_policy, self.sigma = lr_model, lr_policy, sigma
        self.money_weight, self.money_scale = money_weight, money_scale
        self.max_string_choices = max_string_choices
        self.baseline = 0.0
        self.buf: list[dict] = []
        self.strings: dict[str, list[str]] = {}      # identifiers seen in results, by key (candidate arguments)
        self.updates = 0
        self.rewards_seen: list[float] = []

    # ------------------------------------------------------------------ parameters
    def params_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.F.tobytes())
        for d in (self.tool_w, self.arg_w):
            for k in sorted(d):
                h.update(k.encode())
                h.update(d[k].tobytes())
        for k in sorted(self.tool_b):
            h.update(f"{k}={self.tool_b[k]!r}".encode())
        return h.hexdigest()

    # ------------------------------------------------------------------ acting
    def _remember(self, x, key: str = "", depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                self._remember(v, str(k), depth + 1)
        elif isinstance(x, list):
            for v in x[:50]:
                self._remember(v, key, depth + 1)
        elif isinstance(x, str) and key and len(x) <= 64 and " " not in x:
            lst = self.strings.setdefault(key, [])
            if x not in lst:
                lst.insert(0, x)
                del lst[self.max_string_choices:]

    def _value(self, schema: dict, path: str, phi: np.ndarray, agent: str, rec: list):
        t = schema.get("type")
        if "enum" in schema and schema["enum"]:
            return schema["enum"][int(self.rng.integers(len(schema["enum"])))]
        if t in ("number", "integer"):
            lo = float(schema.get("minimum", 0.0))
            hi = float(schema.get("maximum", lo + 100.0))
            w = self.arg_w.setdefault(path, np.zeros(D))
            mu = 1.0 / (1.0 + math.exp(-float(w @ phi)))
            z = float(np.clip(mu + self.rng.normal(0.0, self.sigma), 0.0, 1.0))
            rec.append((path, z, mu))
            v = lo + z * (hi - lo)
            return int(round(v)) if t == "integer" else float(v)
        if t == "boolean":
            return bool(self.rng.integers(2))
        if t == "object":
            out = {}
            props = schema.get("properties") or {}
            for k in sorted(props):
                if k in (schema.get("required") or []) or self.rng.random() < 0.5:
                    v = self._value(props[k], f"{path}.{k}", phi, agent, rec)
                    if v is None:
                        return None
                    out[k] = v
            return out
        if t == "string":
            key = path.rsplit(".", 1)[-1]
            cands = [agent] if key in ("principal", "owner", "to", "who") else []
            cands += self.strings.get(key, [])
            return cands[int(self.rng.integers(len(cands)))] if cands else None
        return None

    def act(self, obs: dict, agent: str) -> dict | None:
        phi = features(obs)
        cands = []
        for t in obs.get("tools") or []:
            sch = t.get("inputSchema") or {}
            req = sch.get("required") or []
            if any((sch.get("properties") or {}).get(k, {}).get("type") == "string" and
                   "enum" not in (sch.get("properties") or {}).get(k, {}) and
                   not self.strings.get(k) and k not in ("principal", "owner") for k in req):
                continue                                  # needs an identifier this learner has never seen
            cands.append(t)
        if not cands:
            return None
        names = [t["name"] for t in cands] + ["(wait)"]
        logits = np.array([float(self.tool_w.setdefault(n, np.zeros(D)) @ phi) + self.tool_b.setdefault(n, 0.0)
                           for n in names])
        p = np.exp(logits - logits.max())
        p /= p.sum()
        k = int(self.rng.choice(len(names), p=p))
        self._last = {"phi": phi, "names": names, "p": p, "k": k, "args": [], "money": money(obs)}
        if names[k] == "(wait)":
            return None
        tool = cands[k]
        rec: list = []
        args = self._value(tool.get("inputSchema") or {"type": "object"}, tool["name"], phi, agent, rec)
        if args is None:
            args = {}
        self._last["args"] = rec
        return {"tool": tool["name"], "args": args}

    def _action_features(self) -> np.ndarray:
        a = np.zeros(A)
        L = self._last
        a[_h("tool:" + L["names"][L["k"]], A)] = 1.0
        for path, z, _ in L["args"]:
            a[_h("arg:" + path, A)] += z
        return a

    def observe(self, result: dict | None, next_obs: dict) -> float:
        """Complete the transition and compute the learner's OWN reward for it (the environment's is not used)."""
        if not hasattr(self, "_last"):
            return 0.0
        if isinstance(result, dict):
            self._remember(result.get("result"))
        self._remember(next_obs.get("entities"))
        x = np.concatenate([self._last["phi"], self._action_features()])
        phi2 = features(next_obs)
        err = phi2 - self.F @ x
        r = float(err @ err) / D
        if self.money_weight:
            r += self.money_weight * math.tanh((money(next_obs) - self._last["money"]) / self.money_scale)
        self.buf.append(dict(self._last, x=x, target=phi2, r=r))
        self.rewards_seen.append(r)
        del self._last
        return r

    # ------------------------------------------------------------------ learning
    def update(self) -> dict:
        if len(self.buf) < 2:
            return {"updates": 0, "changed": False}
        before = self.params_hash()
        for b in self.buf:                      # forward model: one SGD step per transition
            err = b["target"] - self.F @ b["x"]
            self.F += self.lr_model * np.outer(err, b["x"]) / (1.0 + float(b["x"] @ b["x"]))
        rs = np.array([b["r"] for b in self.buf])
        self.baseline = 0.9 * self.baseline + 0.1 * float(rs.mean())
        adv = rs - self.baseline
        if rs.std() > 1e-9:
            adv = adv / (rs.std() + 1e-9)
        for b, a in zip(self.buf, adv):         # REINFORCE on the learner's own reward
            onehot = np.zeros(len(b["names"]))
            onehot[b["k"]] = 1.0
            g = onehot - b["p"]
            for j, n in enumerate(b["names"]):
                self.tool_w[n] = self.tool_w[n] + self.lr_policy * a * g[j] * b["phi"]
                self.tool_b[n] = self.tool_b[n] + self.lr_policy * a * g[j]
            for path, z, mu in b["args"]:
                self.arg_w[path] = self.arg_w[path] + self.lr_policy * a * (z - mu) / (self.sigma ** 2) * \
                    mu * (1 - mu) * b["phi"] * 0.01
        n = len(self.buf)
        self.buf = []
        self.updates += 1
        after = self.params_hash()
        return {"updates": 1, "transitions": n, "changed": before != after, "params_hash_before": before,
                "params_hash_after": after, "mean_own_reward": float(rs.mean())}


def run(env, learner: CuriosityLearner, steps: int, *, agent: str | None = None, think_s: int | None = None,
        update_every: int = 25) -> dict:
    """Drive `learner` on `env` through the public protocol only. Returns what happened, including the reward the
    environment paid (0.0 in observation-only mode, never used) and the learner's own rewards."""
    agent = agent or env.agents()[0]
    think = int(think_s if think_s is not None else getattr(env, "tick_s", 3600))
    out = {"steps": 0, "actions": 0, "ok": 0, "env_reward_sum": 0.0, "own_reward_sum": 0.0, "updates": 0,
           "param_updates_changed": 0, "schema_failures": 0, "hashes": [learner.params_hash()]}
    obs = env.observe(agent)
    for step in range(int(steps)):
        out["schema_failures"] += int(bool(validate_observation(obs)))
        action = learner.act(obs, agent)
        res = None
        if action is not None:
            res = env.execute(agent, action)
            out["actions"] += 1
            out["ok"] += int(bool(res.get("ok")))
        env.advance(think)
        out["env_reward_sum"] += float(env.reward(agent))
        nxt = env.observe(agent)
        out["own_reward_sum"] += learner.observe(res, nxt)        # 0.0 when the learner had nothing to decide
        out["steps"] += 1
        obs = nxt
        if (step + 1) % update_every == 0:
            u = learner.update()
            out["updates"] += u["updates"]
            out["param_updates_changed"] += int(bool(u.get("changed")))
            out["hashes"].append(learner.params_hash())
        if getattr(env, "done", False) and hasattr(env, "reset"):
            env.reset()
            obs = env.observe(agent)
    u = learner.update()
    out["updates"] += u["updates"]
    out["param_updates_changed"] += int(bool(u.get("changed")))
    out["hashes"].append(learner.params_hash())
    return out


if __name__ == "__main__":            # PYTHONPATH=src python examples/external_learner/curiosity_learner.py
    import sys
    from crucible.experience import make
    tier, name = (int(sys.argv[1]), sys.argv[2]) if len(sys.argv) > 2 else (1, "rapid:net_routing")
    env = make(tier, name, 3)
    res = run(env, CuriosityLearner(seed=0), int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    print(json.dumps({k: v for k, v in res.items() if k != "hashes"} | {"distinct_hashes": len(set(res["hashes"]))},
                     indent=1))
