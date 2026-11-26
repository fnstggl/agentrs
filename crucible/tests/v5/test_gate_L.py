"""Gate L - endogenous opportunities, in the integrated civilisation (workstream G, wave 3b; D-18).

Worlds and arms: `_invention_kit` (`innovation.civexp`: factual, garbage, ban_novel, ban_published) - civilisations
in which the population's autonomous developers invent (no scripted agent). Needs are labelled evaluator-side by
the one relation graph (`innovation.relations`), never shown in-world; developers and individuals react only to
consequences (their own hand work, rejected results, failures, public requests, prices). Definitions:
`crucible.innovation.gates` (module docstring).

* L.1 capability_effects_measured - pre-existing conditions an invention changes, by counterfactual removal;
* L.2 emerged_needs - needs first labelled after genesis (e.g. the second step of a pipeline becoming the
  bottleneck once a developer's service automates the first; unreliable providers creating a need for an
  alternative);
* L.3 needs_addressed_by_new_services - needs whose target was later served by a provider created after the need
  appeared, which holders adopted;
* L.4 independent_chains / min_chain_stages - chains mined generically from the labelled-need history and the event
  log (distinct first changes; >= 3 stages across >= 2 subsystems; among the top three an agent artifact, social
  activity and technical activity; no scripted stage);
* L.5 counterfactual_removals_confirming - chains whose later stages do not happen in the arm where the chain's
  first change was removed;
* L.n1 special_case_bonuses - credits from system accounts to creators after admission (any ledger kind; the
  population's own income deposits are verified against its deposit records), plus a static scan of every module
  for success-conditioned grants from system accounts.
"""
import ast
import os

import pytest

from _invention_kit import A as arms
from crucible.acceptance.v5.evidence import record
from crucible.innovation import gates as G

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10800)]
LABELS = {"autonomous_population_creators": True, "scripted": False, "world": "civilisation"}


@pytest.fixture(scope="module")
def L():
    return G.l_measure(arms())


def _chains_note(L) -> str:
    return "; ".join(f"{c['chain']} ({c['n_stages']} stages: " + " -> ".join(s["name"] for s in c["stages"]
                                                                            if s.get("t") is not None) + ")"
                     for c in L["independent"])


def test_L1_agent_capabilities_alter_existing_conditions(L):
    record("L.1", dict(L["L.1"]), notes=f"effects (removal arm minus factual) {L['effects']}", labels=LABELS)
    assert L["L.1"]["capability_effects_measured"] >= 1


def test_L2_new_participant_needs_arise(L):
    record("L.2", dict(L["L.2"]), notes=f"emerged {L['emerged']}", labels=LABELS)
    assert L["L.2"]["emerged_needs"] >= 1


def test_L3_new_services_address_needs(L):
    record("L.3", dict(L["L.3"]), notes=_chains_note(L), labels=LABELS)
    assert L["L.3"]["needs_addressed_by_new_services"] >= 1


def test_L4_multi_stage_chains(L):
    record("L.4", dict(L["L.4"]), notes=_chains_note(L), labels=LABELS)
    m = L["L.4"]
    assert m["independent_chains"] >= 3 and m["min_chain_stages"] >= 3
    assert m["top3_covers_artifact_social_technical"] and m["scripted_stages"] == 0


def test_L5_counterfactual_removal_confirms(L):
    record("L.5", dict(L["L.5"]), notes="; ".join(f"{c} (removed in {a}): {'confirmed' if ok else 'not confirmed'}"
                                                   for c, a, ok in L["confirmations"]), labels=LABELS)
    assert L["L.5"]["counterfactual_removals_confirming"] >= 3


def _system_grants_in_code() -> list[str]:
    """Static scan of every module: bank moves out of a system account inside a function that is not world
    generation (genesis/install), admission, income or exogenous-economy plumbing."""
    root = os.path.join(os.path.dirname(__file__), "..", "..", "src", "crucible")
    allowed_fn = ("genesis", "install", "ep_join", "ep_endow", "money_deposit", "pr_income", "_income", "_pay_income",
                  "pr_daily", "pr_tick", "_wages", "_endow", "_seed", "_spawn")
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            src = open(path).read()
            if "acct_genesis" not in src:
                continue
            tree = ast.parse(src)
            for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                for node in ast.walk(func):
                    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "move" and node.args[1:2] and \
                            isinstance(node.args[1], ast.Constant) and node.args[1].value == "acct_genesis":
                        kinds = [a.value for a in node.args[4:5] if isinstance(a, ast.Constant)]
                        if kinds and kinds[0] in ("bonus", "reward", "prize", "grant", "subsidy") and \
                                not func.name.startswith(allowed_fn):
                            out.append(f"{os.path.relpath(path, root)}:{func.name}")
    return sorted(set(out))


def test_Ln1_no_special_case_success_bonuses(L):
    static = _system_grants_in_code()
    m = dict(L["L.n1"])
    m.update(special_case_bonuses=int(m["runtime_bonuses"] + len(static)), static_success_grants=len(static))
    record("L.n1", m, notes=f"static findings {static}", labels=LABELS)
    assert m["runtime_bonuses"] == 0 and not static and m["creators"] >= 1
