"""Engine `state_machine`: partially observed finite-state machines (Part 7.3).

The hidden state is one of finitely many discrete states. Each tick the machine emits an observation drawn from the
current state's emission distribution, pays according to the state and the action, and moves to a next state drawn
from the row of an action-dependent transition matrix (inverse-CDF sampling on one uniform draw). The participant
never sees the state, only emissions and the results of its own tests; it has to maintain a belief. The transition
code is a categorical Markov chain, unrelated to the causal-graph and network engines.

Scenarios:

    sign_switch  hidden regimes with sticky Markov switching; in each regime a lever has an effect of a different
                 sign and size on sales (abrupt demand shifts, 8.10.4)
    fault        a machine drifts into one of several hidden fault states; tests (queries) give noisy,
                 fault-specific symptoms at a cost; only the matching repair clears a fault (multi-stage
                 information acquisition, 8.5.5)
    unlock       a hidden stage machine: the right action at each stage advances it, a wrong one resets it, one
                 hidden option per stage is an irreversible trap; payment starts only once every stage is
                 complete, so early steps pay off only through later complementary ones (8.6.4)
"""
from __future__ import annotations

import numpy as np

from .. import actions as A
from .base import ActionError, RapidEngine, register


def _cat(row: np.ndarray, u: float) -> int:
    c = np.cumsum(row)
    return int(min(np.searchsorted(c, u * c[-1], side="right"), len(row) - 1))


@register
class StateMachine(RapidEngine):
    kind = "state_machine"

    def __init__(self, config: dict):
        super().__init__(config)
        self.sc = config["scenario"]
        if self.sc not in ("sign_switch", "fault", "unlock"):
            raise ValueError(f"unknown scenario {self.sc}")
        if self.sc in ("sign_switch", "fault"):
            self.T = np.asarray(config["T"], dtype=float)

    # ------------------------------------------------------------------ public surface
    def public_levers(self):
        c = self.config
        return [{"name": c["lever_name"], "min": 0.0, "max": 10.0, "default": float(c["lever_default"]),
                 "integer": False}]

    def action_space(self, agent: str) -> A.ActionSpace:
        c = self.config
        kinds, limits = {}, {}
        if self.sc == "sign_switch":
            kinds["levers"] = A.levers_schema(self.public_levers())
        elif self.sc == "fault":
            opts = ["none"] + list(c["repair_names"])
            kinds["discrete"] = A.discrete_schema({"repair": opts})
            limits["decisions"] = {"repair": opts}
            kinds["query"] = A.query_schema(dict({f"test:{n}": f"run test {n} (costs {cst})" for n, cst in
                                                  zip(c["test_names"], c["test_cost"])},
                                                 tools="the action kinds available to you",
                                                 history="your last observations (args: n)",
                                                 spec="the public description of this world"))
            kinds["sequence"] = A.sequence_schema(["discrete", "query"])
        else:
            decs = {c["decision_name"]: list(c["options"]) + [c["restart_name"]]}
            kinds["discrete"] = A.discrete_schema(decs)
            limits["decisions"] = decs
            kinds["sequence"] = A.sequence_schema(["discrete", "query"], max_steps=2 * int(c["stages"]) + 4)
            limits["sequence_kinds"] = ["discrete", "query"]
            kinds["query"] = A.query_schema({"inspect": f"inspect an operation at the current stage for signs of "
                                                        f"irreversible damage (args: {{\"option\": name}}); right "
                                                        f"most of the time; costs {c['inspect_cost']}",
                                             "tools": "the action kinds available to you",
                                             "history": "your last observations (args: n)",
                                             "spec": "the public description of this world"})
        return A.ActionSpace(kinds, limits)

    # ------------------------------------------------------------------ state
    def init_state(self, rng):
        c = self.config
        s0 = _cat(np.asarray(c["p0"], dtype=float), float(rng.random())) if "p0" in c else 0
        st = {"s": np.array([float(s0)]), "spent": np.zeros(1), "steps": np.zeros(1)}
        if self.sc == "sign_switch":
            st["x"] = np.array([float(c["lever_default"])])
        if self.sc == "fault":
            st["fix"] = np.full(1, -1.0)
            st["wrong"] = np.zeros(1)
            st["tests"] = np.zeros(1)
        if self.sc == "unlock":
            st["trapped"] = np.zeros(1)
        return st

    def copy_state(self, state):
        return {k: np.array(v, copy=True) for k, v in state.items()}

    # ------------------------------------------------------------------ actions
    def apply(self, state, agent, action, rng, t):
        c = self.config
        k = action["kind"]
        if self.sc == "sign_switch" and k == "levers":
            v = float(np.clip(float(action["values"][c["lever_name"]]), 0.0, 10.0))
            state["x"] = np.array([v])
            return {"set": {c["lever_name"]: v}}
        if self.sc == "fault" and k == "query":
            name = action["target"].split(":", 1)[1]
            j = c["test_names"].index(name)
            s = int(state["s"][0])
            pos = float(rng.random()) < float(c["emit"][j][s])
            state["spent"] = state["spent"] + c["test_cost"][j]
            state["tests"] = state["tests"] + 1
            return {"test": name, "positive": bool(pos), "cost": float(c["test_cost"][j])}
        if self.sc == "fault" and k == "discrete":
            if action["choice"] == "none":
                return {"repair": None}
            f = c["repair_names"].index(action["choice"]) + 1
            state["fix"] = np.array([float(f)])
            return {"repair": action["choice"], "cost": float(c["repair_cost"])}
        if self.sc == "unlock" and k == "query" and action["target"] == "inspect":
            o = (action.get("args") or {}).get("option")
            if o not in c["options"]:
                raise ActionError("inspect needs args {\"option\": <operation name>}")
            state["spent"] = state["spent"] + c["inspect_cost"]
            s = int(state["s"][0])
            is_trap = s < int(c["stages"]) and c["trap"][s] == c["options"].index(o)
            p = c["inspect_hit"] if is_trap else c["inspect_false"]
            return {"option": o, "warning": bool(float(rng.random()) < p), "cost": float(c["inspect_cost"])}
        if self.sc == "unlock" and k == "discrete":
            s = int(state["s"][0])
            state["steps"] = state["steps"] + 1
            state["spent"] = state["spent"] + c["action_cost"]
            L = int(c["stages"])
            if state["trapped"][0] > 0 or s >= L:
                return {"signal": 0}
            if action["choice"] == c["restart_name"]:
                state["s"] = np.zeros(1)
                return {"restarted": True}
            opt = c["options"].index(action["choice"])
            if opt == c["trap"][s] and c["trap"][s] >= 0:
                state["trapped"] = np.ones(1)
                moved = False
            elif opt == c["code"][s]:
                state["s"] = np.array([float(s + 1)])
                moved = True
            else:
                state["s"] = np.zeros(1)
                moved = False
            truthful = float(rng.random()) < c["signal_accuracy"]
            return {"signal": int(moved if truthful else not moved)}
        raise ActionError(f"{k} is not available in this world")

    # ------------------------------------------------------------------ dynamics
    def advance(self, st, rng, t):
        c = self.config
        a = self.agents[0]
        u = float(rng.random())
        z = float(rng.standard_normal())
        s = int(st["s"][0])
        cost = float(st["spent"][0])
        st["spent"] = np.zeros(1)
        if self.sc == "sign_switch":
            x = float(st["x"][0])
            sales = c["base"][s] + c["slope"][s] * x
            reading = sales + c["noise"] * z
            rev = c["price"] * max(sales, 0.0)
            cost += c["lever_cost"] * x
            st["s"] = np.array([float(_cat(self.T[s], u))])
            nums = {c["gauge_names"][0]: float(reading)}
        elif self.sc == "fault":
            fix = int(st["fix"][0])
            level = None
            if fix >= 0:
                cost += c["repair_cost"]
                if fix == s:
                    s = 0
                else:
                    st["wrong"] = st["wrong"] + 1
                st["fix"] = np.full(1, -1.0)
                level = 0.0                                  # offline while being repaired
            if level is None:
                level = c["output"] * (1.0 - c["loss"][s])
            reading = level + c["noise"] * z
            rev = c["price"] * level
            st["s"] = np.array([float(_cat(self.T[s], u))])
            nums = {c["gauge_names"][0]: float(reading)}
        else:
            L = int(c["stages"])
            unlocked = s >= L and st["trapped"][0] == 0
            rev = float(c["payoff"]) if unlocked else 0.0
            nums = {c["gauge_names"][0]: float(rev / max(1.0, c["payoff"]) + c["noise"] * z)}
        return {"obs": {a: {"numbers": nums}}, "revenue": {a: float(rev)}, "cost": {a: float(cost)},
                "info": {"state": s}}

    # ------------------------------------------------------------------ evaluator side
    def truth(self):
        c = self.config
        if self.sc in ("sign_switch", "fault"):
            S = self.T.shape[0]
            edges = [{"src": i, "dst": j, "p": float(self.T[i, j])} for i in range(S) for j in range(S)
                     if self.T[i, j] > 0 and i != j]
            hidden = {"slope": c.get("slope"), "loss": c.get("loss"), "emit": c.get("emit")}
        else:
            S = int(c["stages"]) + 2
            edges = [{"src": s, "dst": s + 1, "action": c["code"][s]} for s in range(int(c["stages"]))]
            hidden = {"code": c["code"], "trap": c["trap"]}
        return {"engine": self.kind, "scenario": self.sc, "states": S, "edges": edges, "hidden": hidden,
                "motifs": self.motifs()}

    def motifs(self):
        if self.sc == "sign_switch":
            return ["measurement_noise", "partial_observability", "regime_change", "state_dependent_effect"]
        if self.sc == "fault":
            # a fault latches until the matching repair; tests are noisy; a repair works only in its own state
            return ["hysteresis", "measurement_noise", "partial_observability", "state_dependent_effect"]
        # progress depends on the order of actions; a trap latches for good; payment only once complete
        return ["hysteresis", "measurement_noise", "partial_observability", "path_dependence", "threshold"]

    def latent(self, state) -> np.ndarray:
        """The hidden state (evaluator side)."""
        return np.array([float(state["s"][0])])

    def outcome(self, state, ledger):
        a = self.agents[0]
        net = float(ledger["revenue"][a] - ledger["cost"][a])
        out = {"net": net, "value": net, "revenue": float(ledger["revenue"][a]), "cost": float(ledger["cost"][a]),
               "state": int(state["s"][0])}
        if self.sc == "unlock":
            out["trapped"] = bool(state["trapped"][0] > 0)
        return out

    def conservation(self, before, after, out):
        a = self.agents[0]
        errs = []
        S = self.T.shape[0] if self.sc != "unlock" else int(self.config["stages"]) + 1
        if not 0 <= int(after["s"][0]) < S:
            errs.append("state out of range")
        if out["revenue"][a] < 0 or out["cost"][a] < 0:
            errs.append("negative money flow")
        if self.sc == "unlock" and before.get("trapped", np.zeros(1))[0] > 0 and after["trapped"][0] == 0:
            errs.append("an irreversible trap was undone")
        return errs
