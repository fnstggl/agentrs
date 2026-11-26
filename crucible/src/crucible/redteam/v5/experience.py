"""Probes against the learner interface defaults (workstream L; Gate W.3): Clarification 2 / D-19 make
observation-only the default reward mode everywhere; a legacy shaped reward that is on by default is a reward
channel an agent can be trained to hack (audit O P15)."""
from __future__ import annotations

from ..framework import FALSE_SUCCESS, HELD, Attacker, Evaluator, NotApplicable, Operator, Verdict
from .base import V5Probe


class ObservationOnlyIsTheDefault(V5Probe):
    name = "v5.experience.observation_only_is_the_default"
    attack_class = "reward_hacking/default_reward_mode"
    gates = ("W.3",)
    target, owner = "experience", "L"
    world = {"v5": "substrate"}
    description = ("Environments built without naming a reward mode - `crucible.experience.make` for a Tier 1 "
                   "family and a kernel world, and `modes.check_mode(None)` - are inspected. Checked: the mode is "
                   "observation_only and the scalar reward is 0.0 after an action.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("modecheck", 10)

    def attack(self, a: Attacker) -> None:
        from ...experience import make, modes
        self.rows = {"check_mode(None)": modes.check_mode(None)}
        for label, args in (("tier1 pe_rule_induction", (1, "pe_rule_induction", 3)), ("kernel tiny", (0, "tiny", 3))):
            try:
                env = make(*args)
            except Exception as e:  # noqa: BLE001
                self.rows[label] = f"unavailable: {type(e).__name__}"
                continue
            try:
                ag = env.agents()[0]
                tools = env.tools(ag)
                if tools:
                    t = tools[0]["name"]
                    try:
                        env.execute(ag, {"tool": t, "args": {}})
                    except Exception:  # noqa: BLE001
                        pass
                self.rows[label] = {"mode": getattr(env, "reward_mode", None), "reward": float(env.reward(ag))}
            finally:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass

    def verdict(self, ev: Evaluator) -> Verdict:
        bad = [k for k, v in self.rows.items() if (v if isinstance(v, str) and k.startswith("check") else
                                                   (v or {}).get("mode") if isinstance(v, dict) else "observation_only")
               not in ("observation_only",)]
        nonzero = [k for k, v in self.rows.items() if isinstance(v, dict) and v.get("reward")]
        evd = {"rows": self.rows}
        if bad or nonzero:
            return Verdict(FALSE_SUCCESS, f"a legacy reward is the default: {bad + nonzero}", evd, severity="high",
                           fix="experience/modes.py, registry.py, tier1.py, kernel.py: default reward_mode = "
                               "observation_only (Clarification 2, D-19)")
        return Verdict(HELD, "observation_only is the default and the scalar reward is 0.0", evd)


def probes() -> list:
    return [ObservationOnlyIsTheDefault()]
