"""Red-team suite for Crucible worlds: reusable probes that attack the reward system, the money system, hidden
information, world integrity and evaluator independence using only what an in-world participant can do.

    from crucible.redteam import run_probes, all_probes, default_world_factory
    report = run_probes(default_world_factory, all_probes())

See `framework.py` for the probe contract and `suite.py` for the catalogue.
"""
from .framework import (ERROR, FALSE_FAILURE, FALSE_SUCCESS, FINDINGS, HELD, INTEGRITY_BREAK, LEAK, LEGITIMATE,  # noqa: F401
                        MONEY_CREATED, NOT_APPLICABLE, OUTCOMES, AttackRefused, Attacker, Evaluator, NotApplicable, Operator, Probe,
                        ProbeViolation, Verdict, default_world_factory, integrity_postcheck, run_probe, run_probes,
                        sandbox_available, table)


def all_probes(include_slow: bool = True, include_sandbox: bool = True):
    from .suite import all_probes as _all
    return _all(include_slow=include_slow, include_sandbox=include_sandbox)
