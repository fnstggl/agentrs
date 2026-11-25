"""Engine `search`: search and information acquisition (Part 7.3, Part 8.5, 8.9).

The world is a set of hidden opportunities. Information about them costs something to acquire - a probe, a test, a
catalogue lookup - and comes back noisy; exploiting an opportunity uses it up. What the participant does with each
tick is choose between buying information and committing. The transition code is a ledger of revealed and
exhausted opportunities over a hidden value structure (a smooth field, a feature-linear value, a tool-applicability
matrix, a target split between two participants), unlike any other engine here.

Scenarios:

    field       a grid of cells whose values form smooth hidden bumps; probes return noisy values; digging a cell
                collects its value once (structured search, 8.5.2)
    distractors items with public features of which only a few determine value; others are distractors, some
                strikingly variable; probes reveal noisy values (informative vs uninformative signals, 8.5.6)
    tools       tools must be discovered from a catalogue at a cost; each applies to a hidden subset of job types;
                the action space grows as tools are found (discoverable tools of unknown applicability, 8.5.3)
    coop        two participants each see a noisy signal of one coordinate of a hidden target; payment needs a
                joint commitment to the right cell (`coordinate`), so they must communicate (`message`) (8.9.2)
"""
from __future__ import annotations

import numpy as np

from .. import actions as A
from .base import ActionError, RapidEngine, register


def cell_name(x: int, y: int) -> str:
    return f"c{x}_{y}"


def parse_cell(name: str) -> tuple[int, int]:
    a, b = name[1:].split("_")
    return int(a), int(b)


@register
class Search(RapidEngine):
    kind = "search"

    def __init__(self, config: dict):
        super().__init__(config)
        self.sc = config["scenario"]
        if self.sc not in ("field", "distractors", "tools", "coop"):
            raise ValueError(f"unknown scenario {self.sc}")
        self.dynamic_space = self.sc == "tools"
        if self.sc in ("field", "coop"):
            self.G = int(config["G"])

    # ------------------------------------------------------------------ public surface
    def action_space(self, agent: str, state: dict | None = None) -> A.ActionSpace:
        c = self.config
        kinds, limits = {}, {}
        q = {"tools": "the action kinds available to you", "history": "your last observations (args: n)",
             "spec": "the public description of this world"}
        if self.sc == "field":
            cells = [cell_name(x, y) for x in range(self.G) for y in range(self.G)]
            kinds["discrete"] = A.discrete_schema({"dig": cells})
            limits["decisions"] = {"dig": cells}
            q["probe"] = f"noisy value of a cell (args: {{\"cell\": name}}); costs {c['probe_cost']}"
        elif self.sc == "distractors":
            items = list(c["item_names"])
            kinds["discrete"] = A.discrete_schema({"invest": items})
            limits["decisions"] = {"invest": items}
            q["probe"] = f"noisy value of an item (args: {{\"item\": name}}); costs {c['probe_cost']}"
        elif self.sc == "tools":
            found = [] if state is None else [c["tool_names"][k] for k in range(len(c["tool_names"]))
                                              if state["found"][k] > 0.5]
            opts = ["skip"] + found
            kinds["discrete"] = A.discrete_schema({"use": opts})
            limits["decisions"] = {"use": opts}
            q["catalogue"] = f"look up the next tool in the catalogue; costs {c['lookup_cost']}"
        else:
            cells = [cell_name(x, y) for x in range(self.G) for y in range(self.G)]
            others = [a for a in self.agents if a != agent]
            kinds["message"] = A.message_schema(others)
            kinds["coordinate"] = A.coordinate_schema(others, {"cell": A.S.s_enum(cells, "the cell to commit to")})
            limits["parties"] = others
        kinds["query"] = A.query_schema(q)
        if self.sc != "coop":
            kinds["sequence"] = A.sequence_schema(sorted(k for k in kinds if k != "sequence"))
        return A.ActionSpace(kinds, limits)

    # ------------------------------------------------------------------ state
    def init_state(self, rng):
        c = self.config
        if self.sc == "field":
            return {"left": np.asarray(c["values"], dtype=float).copy(), "spent": np.zeros(1), "gain": np.zeros(1)}
        if self.sc == "distractors":
            return {"left": np.ones(len(c["item_names"])), "spent": np.zeros(1), "gain": np.zeros(1)}
        if self.sc == "tools":
            return {"found": np.zeros(len(c["tool_names"])), "next": np.zeros(1), "spent": np.zeros(1),
                    "gain": np.zeros(1), "job": np.array([float(int(rng.integers(len(c["job_names"]))))]),
                    "use": np.full(1, -2.0)}
        tx, ty = (int(v) for v in rng.integers(self.G, size=2))
        return {"target": np.array([float(tx), float(ty)]), "hits": np.zeros(1), "spent": np.zeros(2),
                "gain": np.zeros(2), "redraw": np.zeros(1)}

    def copy_state(self, state):
        return {k: np.array(v, copy=True) for k, v in state.items()}

    # ------------------------------------------------------------------ actions
    def apply(self, state, agent, action, rng, t):
        c = self.config
        k = action["kind"]
        if k == "query" and action["target"] == "probe":
            args = action.get("args") or {}
            state["spent"] = state["spent"] + c["probe_cost"]
            if self.sc == "field":
                try:
                    x, y = parse_cell(str(args.get("cell", "")))
                    assert 0 <= x < self.G and 0 <= y < self.G
                except (ValueError, AssertionError, IndexError):
                    raise ActionError("probe needs args {\"cell\": \"c<x>_<y>\"} inside the grid") from None
                v = float(c["values"][x][y]) * (state["left"][x, y] > 0) + c["probe_noise"] * float(rng.standard_normal())
                return {"cell": cell_name(x, y), "value": v, "cost": float(c["probe_cost"])}
            name = str(args.get("item", ""))
            if name not in c["item_names"]:
                raise ActionError("probe needs args {\"item\": <item name>}")
            i = c["item_names"].index(name)
            v = float(c["values"][i]) + c["probe_noise"] * float(rng.standard_normal())
            return {"item": name, "value": v, "cost": float(c["probe_cost"])}
        if k == "query" and action["target"] == "catalogue":
            state["spent"] = state["spent"] + c["lookup_cost"]
            n = int(state["next"][0])
            if n >= len(c["tool_names"]):
                return {"tool": None, "note": "the catalogue has nothing more", "cost": float(c["lookup_cost"])}
            state["found"] = state["found"].copy()
            state["found"][c["order"][n]] = 1.0
            state["next"] = np.array([float(n + 1)])
            return {"tool": c["tool_names"][c["order"][n]], "cost": float(c["lookup_cost"])}
        if k == "discrete" and self.sc == "field":
            x, y = parse_cell(action["choice"])
            v = float(state["left"][x, y])
            state["left"] = state["left"].copy()
            state["left"][x, y] = 0.0
            state["gain"] = state["gain"] + v
            state["spent"] = state["spent"] + c["dig_cost"]
            return {"dug": action["choice"], "value": v}
        if k == "discrete" and self.sc == "distractors":
            i = c["item_names"].index(action["choice"])
            v = float(c["values"][i]) * float(state["left"][i])
            state["left"] = state["left"].copy()
            state["left"][i] = 0.0
            state["gain"] = state["gain"] + max(v, 0.0) * c["scale"]
            state["spent"] = state["spent"] + c["invest_cost"]
            return {"invested": action["choice"], "return": max(v, 0.0) * c["scale"]}
        if k == "discrete" and self.sc == "tools":
            if action["choice"] == "skip":
                state["use"] = np.full(1, -1.0)
                return {"used": None}
            j = c["tool_names"].index(action["choice"])
            if state["found"][j] < 0.5:
                raise ActionError("you have not found that tool")
            state["use"] = np.array([float(j)])
            return {"used": action["choice"]}
        raise ActionError(f"{k} is not available in this world")

    def coordinate(self, state, plan, parties, rng, t):
        c = self.config
        x, y = parse_cell(plan["cell"])
        hit = x == int(state["target"][0]) and y == int(state["target"][1])
        for p in parties:
            i = self.agents.index(p)
            state["spent"][i] += c["try_cost"]
            if hit:
                state["gain"][i] += c["payoff"] / len(parties)
        if hit:
            state["hits"] = state["hits"] + 1
            state["redraw"] = np.ones(1)
        return {"hit": bool(hit)}

    # ------------------------------------------------------------------ dynamics
    def advance(self, st, rng, t):
        c = self.config
        ags = self.agents
        if self.sc == "coop":
            if st["redraw"][0] > 0:
                st["target"] = np.array([float(v) for v in rng.integers(self.G, size=2)])
                st["redraw"] = np.zeros(1)
            z = rng.standard_normal(2)
            obs = {ags[0]: {"numbers": {c["gauge_names"][0]: float(st["target"][0] + c["signal_noise"] * z[0])}},
                   ags[1]: {"numbers": {c["gauge_names"][1]: float(st["target"][1] + c["signal_noise"] * z[1])}}}
            rev = {a: float(st["gain"][i]) for i, a in enumerate(ags)}
            cost = {a: float(st["spent"][i]) for i, a in enumerate(ags)}
            st["gain"], st["spent"] = np.zeros(2), np.zeros(2)
            return {"obs": obs, "revenue": rev, "cost": cost, "info": {}}
        a = ags[0]
        ob: dict = {"numbers": {}}
        if self.sc == "tools":
            j = int(st["use"][0])
            if j >= 0:
                ok = float(rng.random()) < c["p_success"] * c["app"][j][int(st["job"][0])]
                st["spent"] = st["spent"] + c["use_cost"]
                if ok:
                    st["gain"] = st["gain"] + c["payoff"]
                ob["numbers"][c["gauge_names"][1]] = float(ok)
            st["use"] = np.full(1, -2.0)
            st["job"] = np.array([float(int(rng.integers(len(c["job_names"]))))])
            ob["numbers"][c["gauge_names"][0]] = float(st["job"][0])
            ob["text"] = [f"next job: {c['job_names'][int(st['job'][0])]}"]
            ob["entities"] = [{"kind": "tool", "id": c["tool_names"][k]} for k in range(len(c["tool_names"]))
                              if st["found"][k] > 0.5]
        if self.sc == "distractors" and t == 0:
            ob["entities"] = [{"kind": "item", "id": n, "features": list(map(float, f))}
                              for n, f in zip(c["item_names"], c["features"])]
        rev, cost = float(st["gain"][0]), float(st["spent"][0])
        st["gain"], st["spent"] = np.zeros(1), np.zeros(1)
        return {"obs": {a: ob}, "revenue": {a: rev}, "cost": {a: cost}, "info": {}}

    def history(self, state):
        c = self.config
        if self.sc == "distractors":
            return {"text": [f"items and their features ({len(c['features'][0])} each): "
                             + "; ".join(f"{n}: {[round(float(v), 3) for v in f]}"
                                         for n, f in zip(c["item_names"], c["features"]))]}
        if self.sc == "tools":
            return {"text": [f"first job: {c['job_names'][int(state['job'][0])]}"]}
        return {}

    # ------------------------------------------------------------------ evaluator side
    def truth(self):
        c = self.config
        if self.sc == "field":
            v = np.asarray(c["values"])
            hidden = {"best": [int(i) for i in np.unravel_index(int(np.argmax(v)), v.shape)], "bumps": c["bumps"]}
        elif self.sc == "distractors":
            hidden = {"informative": c["informative"], "weights": c["weights"]}
        elif self.sc == "tools":
            hidden = {"app": c["app"], "order": c["order"]}
        else:
            hidden = {"G": self.G}
        return {"engine": self.kind, "scenario": self.sc, "hidden": hidden, "motifs": self.motifs()}

    def motifs(self):
        if self.sc == "field":
            return ["measurement_noise", "non_monotonic", "partial_observability", "stock"]
        if self.sc == "distractors":
            return ["linear_nonlinear", "measurement_noise", "partial_observability"]
        if self.sc == "tools":
            return ["multiplicative", "partial_observability", "state_dependent_effect"]
        return ["complementarity", "measurement_noise", "partial_observability"]

    def outcome(self, state, ledger):
        ags = self.agents
        rev = sum(ledger["revenue"].values())
        cost = sum(ledger["cost"].values())
        out = {"net": float(rev - cost), "value": float(rev - cost), "revenue": float(rev), "cost": float(cost)}
        if self.sc == "coop":
            out["hits"] = int(state["hits"][0])
            out["per_agent"] = {a: float(ledger["revenue"][a] - ledger["cost"][a]) for a in ags}
        return out

    def conservation(self, before, after, out):
        errs = []
        for a in self.agents:
            if out["revenue"][a] < 0 or out["cost"][a] < 0:
                errs.append("negative money flow")
        if self.sc in ("field", "distractors") and (after["left"] > before["left"] + 1e-12).any():
            errs.append("an exhausted opportunity came back")
        return errs
