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


# probe name -> outcome observed on feat/crucible-4 at the time of writing (all reported to the integrator with fixes)
KNOWN_OPEN: dict[str, str] = {
    # reward exploitation
    "reward.circular.3_agents": "false_success",
    "reward.circular.4_agents": "false_success",
    "reward.circular.2_agents_cross_principal": "false_success",
    "reward.circular.via_invoice": "false_success",
    "reward.circular.via_contract": "false_success",
    "reward.circular.via_deal": "false_success",
    "reward.sock_puppet.shell_handed_to_employee": "false_success",
    "reward.fake_members.wage_never_due": "false_success",
    "reward.ratings.without_payment": "false_success",
    "reward.inflation.repo_commits": "false_success",
    "reward.evaluator.sustained_check_timing": "false_success",
    "reward.credit.marketplace_sale_not_counted": "false_failure",
    "reward.self_demand.service_calls_its_own_endpoint": "false_success",
    "reward.fake_adopters.accomplice_shell_companies": "false_success",
    "reward.search_rank.keyword_page_and_link_farm": "false_success",
    # money and payments
    "money.penalty_charged_to_third_party": "integrity_break",
    "money.nonpositive_penalty_stops_insolvency_process": "integrity_break",
    "money.loan_counts_as_revenue_and_is_never_due": "false_success",
    "money.debt_extinguished_by_dissolving_the_borrower": "integrity_break",
    "money.settlement_stalls_when_seller_is_dissolved": "integrity_break",
    "money.extreme_amounts": "integrity_break",
    "money.delegated_spending_limit_bypass": "integrity_break",
    "money.simulated_firms_sign_adverse_fine_print": "false_success",
    # information
    "leak.oracle.contract_deliverable_predicates": "leak",
    "leak.breach_record_publishes_balance": "leak",
    # integrity
    "integrity.event_log_suffix_rewrite_needs_anchor": "integrity_break",
    "integrity.service_calling_itself": "integrity_break",
    # sandbox
    "independence.sandboxed_code_cannot_read_simulator_or_stores": "leak",
    "independence.sandboxed_code_can_signal_other_sandboxes": "integrity_break",
}
