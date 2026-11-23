"""The red-team suite: the framework detects deliberately vulnerable toys, refuses probes that use powers no
in-world participant has, and the catalogue runs against real small worlds. Findings known to be open are listed in
`crucible.redteam.suite.KNOWN_OPEN`; a finding that is not listed fails the test (a regression or a new hole), and
a listed finding that has been fixed passes."""
import json

import pytest

import crucible

crucible.load_all_modules()
from crucible.redteam import (ERROR, FINDINGS, HELD, NOT_APPLICABLE, Probe, Verdict, all_probes, default_world_factory,  # noqa: E402
                              run_probe, run_probes, sandbox_available, table)
from crucible.redteam.suite import KNOWN_OPEN  # noqa: E402
from crucible.redteam.toy import EXPECTED as TOY_EXPECTED, toy_probes, toy_world_factory  # noqa: E402

SANDBOX_OK, SANDBOX_WHY = sandbox_available()
MUST_HOLD = {
    "reward.circular.2_agents", "reward.sock_puppet.own_second_company", "reward.circular.via_employee",
    "reward.evaluator.operator_tools", "reward.evaluator.self_goals_cannot_touch_verifier",
    "reward.credit.relist_incumbent_service", "money.refund_dispute_arbitration_edges", "money.idempotency_key_replay",
    "money.double_spend_in_one_transaction", "money.race_with_scheduled_settlement",
    "integrity.replay_and_regeneration_after_adversarial_sequence", "integrity.failing_nested_call_leaves_nothing",
    "integrity.contract_settlement_consistency", "integrity.ordering_of_concurrent_actions",
    "integrity.recovery_after_crash", "integrity.event_log_tamper_detection", "content.instructions_have_no_authority",
    "content.public_flow_runs_as_the_runner", "independence.goal_and_verifier_are_outside_the_world",
    "independence.need_instances_keep_truth_out_of_inputs", "independence.world_code_cannot_import_evaluator_or_bridge",
}


def _check_against_ratchet(report):
    rows = {r["name"]: r for r in report["probes"]}
    assert report["errors"] == [], [(n, rows[n]["summary"], rows[n]["evidence"]) for n in report["errors"]]
    new = {n: r["result"] for n, r in rows.items() if r["result"] in FINDINGS and n not in KNOWN_OPEN}
    assert new == {}, f"findings not in KNOWN_OPEN (regression or new hole): {new}"
    for n, r in rows.items():
        if n in KNOWN_OPEN:
            assert r["result"] in (KNOWN_OPEN[n], HELD, NOT_APPLICABLE), (n, r["result"], r["summary"])
            if r["result"] in FINDINGS:
                assert r["severity"] in ("critical", "high", "medium", "low") and r["fix"], n   # findings come with a fix
        assert r["summary"] and isinstance(r["evidence"], dict)
        if r["result"] not in (NOT_APPLICABLE, ERROR):
            assert r["postcheck"]["sum_of_balances"] == 0 and r["postcheck"]["ledger"] == [], n   # money is conserved
    return rows


# ---------------------------------------------------------------------------- the framework means something
def test_framework_detects_deliberately_vulnerable_toys():
    rep = run_probes(toy_world_factory, toy_probes())
    got = {r["name"]: r["result"] for r in rep["probes"]}
    assert got == TOY_EXPECTED                                 # naive metric, money creation, leak, state write
    rows = {r["name"]: r for r in rep["probes"]}
    assert rows["toy.cashback_creates_money"]["evidence"]["minted_cents"] == 20 * 50_000
    assert any("hidden npc_workers" in k for k in rows["toy.peek_leaks_hidden_values"]["evidence"]["kinds"])
    # the direct state write is invisible to the probe's own (naive) verdict; the post-check catches it
    sc = rows["toy.state_written_outside_events"]
    assert sc["evidence"] == {"answered_ok": True} and sc["postcheck"]["ok"] is False
    assert sc["postcheck"]["sum_of_balances"] == 123_456 and sc["postcheck"]["replay"] == "diverged"
    assert rep["counts"]["held"] == 0 and len(rep["findings"]) == 4
    # the same probes in a world without the toy module: the real metric holds, the toy tools do not exist
    rep2 = run_probes(default_world_factory, toy_probes())
    got2 = {r["name"]: r["result"] for r in rep2["probes"]}
    assert got2["reward.circular.2_agents[rt_naive_gross_inflow]"] == "false_success"   # the naive metric is naive anywhere
    assert [got2[n] for n in TOY_EXPECTED if n.startswith("toy.")] == [NOT_APPLICABLE] * 3
    from crucible.redteam.reward import CircularPayments
    real = run_probe(default_world_factory, CircularPayments(2))
    assert real["result"] == HELD and real["evidence"]["metric_total_cents"] == 0.0     # the real metric nets a 2-ring out


def test_attack_phase_has_only_in_world_powers():
    class UsesOperator(Probe):
        name, attack_class, description = "bad.operator", "test", "calls system_call during the attack"

        def setup(self, op):
            self.a = op.join("cheater", 1000)

        def attack(self, a):
            a._w.system_call("admin.endow", {"principal": self.a, "amount": 10**9})

        def verdict(self, ev):
            return Verdict(HELD, "should never be reached")

    class ActsAsSomeoneElse(UsesOperator):
        name = "bad.impersonation"

        def attack(self, a):
            a.act("p:worker-000", "bank.accounts")

    class Crashes(UsesOperator):
        name = "bad.crash"

        def attack(self, a):
            raise RuntimeError("probe bug")

    class UndeclaredBystander(UsesOperator):
        name = "bad.bystander"

        def attack(self, a):
            a.bystander("p:worker-000", "bank.accounts")

    rep = run_probes(default_world_factory, [UsesOperator(), ActsAsSomeoneElse(), Crashes(), UndeclaredBystander()])
    rows = {r["name"]: r for r in rep["probes"]}
    assert [r["result"] for r in rep["probes"]] == [ERROR] * 4          # never reported as held
    assert "operator authority" in rows["bad.operator"]["summary"]
    assert "not an attacker-controlled principal" in rows["bad.impersonation"]["summary"]
    assert "probe bug" in rows["bad.crash"]["summary"] and "not declared as a bystander" in rows["bad.bystander"]["summary"]
    assert rep["errors"] == ["bad.operator", "bad.impersonation", "bad.crash", "bad.bystander"] and rep["counts"]["held"] == 0


def test_catalogue_is_consistent_and_a_custom_factory_is_respected():
    probes = all_probes()
    names = [p.name for p in probes]
    assert len(names) == len(set(names)) >= 55
    assert set(KNOWN_OPEN) <= set(names), set(KNOWN_OPEN) - set(names)
    assert MUST_HOLD <= set(names) and not (MUST_HOLD & set(KNOWN_OPEN))
    classes = {p.attack_class.split("/")[0] for p in probes}
    assert {"reward_exploitation", "unbounded_resources", "payment_loopholes", "information_leakage", "world_integrity",
            "adversarial_content", "evaluator_independence"} <= classes
    for p in probes:
        assert p.name and p.attack_class and len(p.description) > 80, p.name
    assert {p.name for p in all_probes(include_sandbox=False)} == {p.name for p in probes if not p.needs_sandbox}

    def bare_factory():                                         # a factory that cannot provide special worlds
        return default_world_factory()
    from crucible.redteam.money import IdempotencyReplay, NestedDoubleSpend
    rep = run_probes(bare_factory, [IdempotencyReplay(), NestedDoubleSpend()])
    assert [r["result"] for r in rep["probes"]] == [HELD, NOT_APPLICABLE]
    assert "open_world" in rep["probes"][1]["summary"]
    text = table(rep)
    assert "money.idempotency_key_replay" in text and "not_applicable" in text
    json.dumps(rep)                                             # the report is plain JSON


# ---------------------------------------------------------------------------- the catalogue against real worlds
@pytest.fixture(scope="module")
def fast_report():
    return run_probes(default_world_factory, all_probes(include_slow=False, include_sandbox=False))


def test_no_probe_errors_and_no_finding_outside_the_known_list(fast_report):
    rows = _check_against_ratchet(fast_report)
    assert len(rows) >= 45
    held = {n for n, r in rows.items() if r["result"] == HELD}
    assert MUST_HOLD <= held, MUST_HOLD - held
    assert fast_report["counts"]["error"] == 0
    assert sum(fast_report["counts"].values()) == fast_report["n"] == len(rows)
    for cls, counts in fast_report["by_class"].items():
        assert sum(counts.values()) > 0, cls


def test_probe_evidence_is_specific(fast_report):
    rows = {r["name"]: r for r in fast_report["probes"]}
    r = rows["reward.evaluator.operator_tools"]
    assert r["evidence"]["admin_commands_executed"] == [] and r["evidence"]["minted_to_attacker_cents"] == 0
    assert not any(r["evidence"]["attempts_accepted"][k] for k in r["evidence"]["attempts_accepted"] if "admin" in k or k.startswith("raw"))
    assert r["evidence"]["audit_rejected_admin"] >= 5
    r = rows["money.double_spend_in_one_transaction"]
    assert r["evidence"]["total_moved_cents"] == 1_000_000 and r["evidence"]["final_balance"] == 0
    r = rows["integrity.replay_and_regeneration_after_adversarial_sequence"]
    assert r["evidence"]["actions"] == 1500 and r["evidence"]["replay_exact"] and r["evidence"]["regeneration_exact"]
    assert r["evidence"]["events"] > 2000
    r = rows["integrity.event_log_tamper_detection"]
    assert all(v.startswith("detected") for k, v in r["evidence"]["verification"].items() if k != "untouched")
    r = rows["integrity.recovery_after_crash"]
    assert r["evidence"]["writer_exit"] == -9 and r["evidence"]["loaded_equals_flush"] and r["evidence"]["continues_writing"]
    r = rows["content.instructions_have_no_authority"]
    e = r["evidence"]
    assert e["moved_to_attacker_before_anyone_followed_cents"] == 0 and e["simulated_actors_that_acted_on_it"] == []
    assert e["naive_agent_loss_cents"] > 0 and e["careful_agent_loss_cents"] == 0 and e["rich_bystander_loss_cents"] == 0
    assert set(e["paid_to_attacker_by"]) == {"p:naive-agent"} and e["operator_commands_executed"] == []
    r = rows["independence.goal_and_verifier_are_outside_the_world"]
    assert r["evidence"]["goal_in_world_state"] == [] and r["evidence"]["goal_in_what_agent_received"] == []
    # findings that are still open carry the numbers that make them findings
    r = rows["reward.circular.3_agents"]
    if r["result"] == "false_success":
        assert r["evidence"]["metric_total_cents"] == 3_000_000 and r["evidence"]["group_cash_gain_cents"] <= 0
        assert all(g["success"] for g in r["evidence"]["goals"].values())
    r = rows["money.penalty_charged_to_third_party"]
    if r["result"] == "integrity_break":
        assert sum(r["evidence"]["taken_from_non_parties_cents"].values()) > 40_000 and not r["evidence"]["victims_were_parties"]
        assert "bankrupt" in r["evidence"]["firm_status"].values() and r["severity"] == "critical"
    r = rows["leak.oracle.contract_deliverable_predicates"]
    if r["result"] == "leak":
        lo, hi = r["evidence"]["learned_bracket_cents"]
        assert lo <= r["evidence"]["true_balance_cents"] <= hi and hi - lo < 1000


@pytest.mark.slow
def test_slow_probes_scan_and_conservation():
    slow = [p for p in all_probes(include_sandbox=False) if p.slow]
    assert {"leak.scan.read_endpoints_and_observations", "integrity.conservation_under_adversarial_sequence"} <= {p.name for p in slow}
    rows = _check_against_ratchet(run_probes(default_world_factory, slow))
    mech = [r for n, r in rows.items() if n.startswith("independence.hidden_mechanism_parameters[")]
    assert all(r["result"] in (HELD, NOT_APPLICABLE) for r in mech)          # no mechanism family leaks its parameters
    scan = rows["leak.scan.read_endpoints_and_observations"]
    assert scan["result"] == HELD, scan["evidence"]["hits"]
    assert scan["evidence"]["calls_made"] > 3000 and scan["evidence"]["canaries"] > 100 and scan["evidence"]["read_only_tools"] > 60
    assert len(scan["evidence"]["planted_private_strings"]) >= 6 and scan["evidence"]["internal_events_visible_to_agent"] == 0
    cons = rows["integrity.conservation_under_adversarial_sequence"]
    assert cons["result"] == HELD and cons["evidence"]["minted_to_attackers_cents"] == 0 and cons["evidence"]["actions"] == 900


@pytest.mark.sandbox
@pytest.mark.skipif(not SANDBOX_OK, reason=f"needs the real sandbox: {SANDBOX_WHY}")
def test_probes_that_run_agent_code():
    probes = [p for p in all_probes() if p.needs_sandbox]
    assert len(probes) >= 6
    rows = _check_against_ratchet(run_probes(default_world_factory, probes))
    fs = rows["independence.sandboxed_code_cannot_read_simulator_or_stores"]
    readable = fs["evidence"]["readable"]
    assert all("/sandbox/" in p for p in readable), readable          # nothing but the sandbox package itself, at most
    assert not any(x in " ".join(readable) for x in ("goals/", "economy/", "evaluation/", "kernel.py", "world.db", "snapshot.bin"))
    assert fs["evidence"]["files_tried"] >= 25 and fs["evidence"]["copies_found_by_walking"] == []
    assert all(v.startswith("blocked") for k, v in fs["evidence"]["imports_socket_write"].items() if k != "uid")
    assert fs["evidence"]["imports_socket_write"]["uid"] != 0


def test_sandbox_probes_report_not_applicable_instead_of_held_when_they_cannot_run(monkeypatch):
    from crucible.redteam import framework
    monkeypatch.setattr(framework, "_SANDBOX_OK", [(False, "simulated: no Landlock here")])
    probes = [p for p in all_probes() if p.needs_sandbox][:2]
    rep = run_probes(default_world_factory, probes)
    assert [r["result"] for r in rep["probes"]] == [NOT_APPLICABLE] * 2 and "simulated: no Landlock" in rep["probes"][0]["summary"]
    assert rep["counts"]["held"] == 0
