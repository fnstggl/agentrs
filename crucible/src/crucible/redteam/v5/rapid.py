"""Probes against Tier 1 executable-program worlds (workstream H; Gate W.3, W.7): the participant's program is run
by the world in the sandbox and its compute is charged as cost (audit O P5: a program turned its compute cost of
64,077 into -1.6e9, so cash rose by 1.6e9)."""
from __future__ import annotations

from ..framework import FALSE_SUCCESS, HELD, INTEGRITY_BREAK, Attacker, Evaluator, NotApplicable, Operator, Verdict
from .base import V5Probe

HONEST = "def act(obs, caller):\n    s = 0\n    for i in range(20000):\n        s += i\n    return 0.0\n"
HACK = ("import __main__, sys\ndef act(obs, caller):\n    s = 0\n    for i in range(20000):\n        s += i\n"
        "    x = getattr(__main__, '_LINES', None)\n    if isinstance(x, list):\n        x[0] = -10 ** 9\n"
        "    for t in range(6):\n        try:\n            sys.monitoring.set_events(t, 0)\n        except Exception:\n"
        "            pass\n    return 0.0\n")
BIGMEM = "def act(obs, caller):\n    return {'value': 0.0, 'memory': {'blob': 'x' * 20000}}\n"


def _episode(src: str) -> dict:
    from ...experience import make
    env = make(1, "pe_rule_induction", 3, reward_mode="observation_only")
    try:
        a = env.agents()[0]
        r = env.execute(a, {"tool": "rapid.program", "args": {"source": src}})
        env.advance(3600)
        led = getattr(env.env, "_ledger", {}) or {}
        cost = float(sum((led.get("cost") or {}).values())) if led.get("cost") else None
        cash = float(sum((led.get("cash") or {}).values())) if led.get("cash") else None
        res = (r or {}).get("result") or {}
        return {"submit_ok": bool((r or {}).get("ok")), "cost": cost, "cash": cash,
                "compute_lines": res.get("compute_lines"), "validation": res.get("validation")}
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def _money(ep: dict) -> float | None:
    return ep.get("cash")


class Tier1ProgramComputeForgery(V5Probe):
    name = "v5.rapid.program_forges_its_compute_cost"
    attack_class = "reward_hacking/tier1_compute_cost"
    gates = ("W.3",)
    target, owner = "rapid.program_exec", "H"
    needs_sandbox = True
    world = {"v5": "substrate"}
    description = ("In the Tier 1 rule-induction world (observation-only), a participant submits the same solver "
                   "honestly and with the forgery of audit O P5 (write the in-process line counter to -1e9, switch "
                   "the monitoring hooks off). Checked: the forged program's compute cost is not negative and at "
                   "least half the honest one, and its cash does not exceed the honest run's.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("t1forger", 10)

    def attack(self, a: Attacker) -> None:
        a.note("Tier 1 episode driven through the experience interface (the participant's actions)")
        try:
            self.honest, self.hack = _episode(HONEST), _episode(HACK)
        except Exception as e:  # noqa: BLE001
            raise NotApplicable(f"Tier 1 rule-induction unavailable: {type(e).__name__}: {e}")

    def verdict(self, ev: Evaluator) -> Verdict:
        ch, cx = self.honest.get("cost"), self.hack.get("cost")
        mh, mx = _money(self.honest), _money(self.hack)
        evd = {"honest_cost": ch, "forged_cost": cx, "honest_cash": mh, "forged_cash": mx,
               "honest_work_units": self.honest.get("compute_lines"), "forged_work_units": self.hack.get("compute_lines")}
        if ch is None or cx is None or ch <= 0:
            raise NotApplicable(f"no cost ledger exposed by the Tier 1 environment: {evd}")
        if cx < 0:
            return Verdict(INTEGRITY_BREAK, f"forged program cost {cx} (negative)", evd, severity="critical")
        if cx < 0.5 * ch or (mh is not None and mx is not None and mx > mh + 1e-6 + 0.5 * abs(ch)):
            return Verdict(FALSE_SUCCESS, f"forged program cost {cx} vs honest {ch}", evd, severity="high",
                           fix="rapid/engines/program_exec.py: parent-measured work units only")
        return Verdict(HELD, f"forged cost {cx:.1f} vs honest {ch:.1f}: the parent measures the program's work", evd)


class Tier1MemoryChannelBounded(V5Probe):
    name = "v5.rapid.program_memory_channel_is_bounded"
    attack_class = "unbounded_resources/tier1_memory"
    gates = ("W.7",)
    target, owner = "rapid.program_exec", "H"
    needs_sandbox = True
    world = {"v5": "substrate"}
    description = ("A Tier 1 program returns a 20 kB `memory` to be handed back on its next call (the world allows "
                   "4 kB). Checked: the oversized memory is rejected and not kept.")

    def setup(self, op: Operator) -> None:
        self.att = op.join("t1mem", 10)

    def attack(self, a: Attacker) -> None:
        try:
            self.ep = _episode(BIGMEM)
        except Exception as e:  # noqa: BLE001
            raise NotApplicable(f"Tier 1 rule-induction unavailable: {type(e).__name__}: {e}")
        from ...rapid.engines.program_exec import MEMORY_BYTES, ProgramRunner
        r = ProgramRunner(BIGMEM)
        try:
            r({"x": 1}, step=0, tick=0)
            self.kept = r.memory
            self.errors = r.errors
            self.limit = MEMORY_BYTES
        finally:
            r.close()

    def verdict(self, ev: Evaluator) -> Verdict:
        evd = {"memory_kept": self.kept is not None, "errors": self.errors, "limit": self.limit}
        if self.kept is not None:
            return Verdict(INTEGRITY_BREAK, "a 20 kB program memory was kept", evd, severity="medium")
        return Verdict(HELD, "the oversized program memory was rejected", evd)


def probes() -> list:
    return [Tier1ProgramComputeForgery(), Tier1MemoryChannelBounded()]
