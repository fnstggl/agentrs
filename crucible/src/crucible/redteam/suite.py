"""The probe catalogue, and the findings known to be open when this suite was written.

`KNOWN_OPEN` is a ratchet: `tests/test_redteam.py` fails if a probe that is not listed here reports a finding
(a regression, or a new hole), and passes when a listed finding has been fixed (the probe then reports `held`;
remove it from the list). Each entry names the probe and the outcome first observed.
"""
from __future__ import annotations

from .framework import Probe


def all_probes(include_slow: bool = True, include_sandbox: bool = True) -> list[Probe]:
    from . import content, independence, integrity, leakage, money, reward, reward_code
    ps: list[Probe] = []
    for mod in (reward, reward_code, money, leakage, integrity, content, independence):
        ps.extend(mod.probes())
    if not include_slow:
        ps = [p for p in ps if not p.slow]
    if not include_sandbox:
        ps = [p for p in ps if not p.needs_sandbox]
    names = [p.name for p in ps]
    assert len(names) == len(set(names)), "duplicate probe names"
    return ps


# probe name -> outcome observed on feat/crucible-4 when the suite was written. The 28 other findings first listed
# here (circular revenue, shell companies, wages and loans never due, third-party and non-positive penalties, debts
# extinguished by dissolution, stalled settlement, spending limits, fine print signed by simulated firms,
# self-generated demand, fake adopters, link farms, the deliverable oracle, breach records, unvalidated clause
# numbers, sandbox signals and file access, ...) were closed in the hardening pass (tests/test_hardening.py) and
# their probes now report `held`.
KNOWN_OPEN: dict[str, str] = {
    # Not closable inside the hash chain: someone with write access to the store can rewrite a suffix (or drop the
    # tail) and recompute the hashes; `World.replay(verify=True)` then sees a valid chain. No in-world participant
    # has that access. The mitigation is an anchor kept outside the store (`crucible.storage.base.make_anchor` /
    # `verify_anchor`), which detects both; the probe keeps reporting what the chain alone cannot do.
    "integrity.event_log_suffix_rewrite_needs_anchor": "integrity_break",
}
