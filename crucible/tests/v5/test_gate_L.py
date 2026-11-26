"""Gate L - endogenous opportunities (workstream F+G; D-18). The multi-stage chains are finished in the integrated
civilization by workstream CIV; here they are measured in the ECON creation worlds on the one population.

Worlds and arms: `innovation.experiments` (see `_creation_kit`): the factual arm, the arm without the SaaS products,
the arm without the generated-category service. All agents are LABELLED SCRIPTED and act only on public
observables (help requests on the forum, the catalogue, their own books); needs are labelled evaluator-side by the
relation graph (`innovation.relations`), never shown in-world.

* L.1 capability_effects_measured - pre-existing conditions an agent-created capability changes, by counterfactual
  removal: the population's hand-work minutes, incumbents' live subscriptions, incumbents' prices;
* L.2 emerged_needs - needs first labelled after genesis (e.g. the second step of a pipeline becoming the
  bottleneck once a provider automates the first);
* L.3 needs_addressed_by_new_services - needs whose target was later served by a provider created after the need
  appeared, which holders adopted, after which the need was relieved;
* L.4 independent_chains / min_chain_stages - chains mined generically from the labelled-need history and the event
  log, of at least three stages across subsystems, involving an agent artifact, social activity and technical
  activity among them;
* L.5 counterfactual_removals_confirming - chains whose later stages are absent in the arm where the chain's first
  change was removed;
* L.n1 special_case_bonuses - credits from system accounts to creators after admission (any ledger kind), plus a
  static scan of every module for success-conditioned grants from system accounts.
"""
import ast
import os

import pytest

from _creation_kit import arms
from crucible.acceptance.v5.evidence import record
from crucible.innovation import measure as M

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]
DAY = 86400


@pytest.fixture(scope="module")
def A():
    return arms()


def _good_chains(chains):
    return [c for c in chains if c["n_stages"] >= 3 and c["ordered"] and len(c["subsystems"]) >= 2]


def _removal_arm(chain) -> str:
    return "no_saas" if ("saasco" in chain["chain"] or any("saasco" in s["name"] for s in chain["stages"])
                         or any(s["name"].startswith("adoption of a") for s in chain["stages"])) else "no_novelty"


def _confirmed(A, chain) -> bool:
    """The chain's later stages do not happen when its first change is removed."""
    arm = A[_removal_arm(chain)]
    if chain["chain"].startswith("need:"):
        nid = chain["chain"][5:]
        later = [s for s in chain["stages"] if s["name"] in ("adoption by holders", "need relieved")]
        if any(s["name"].startswith("adoption of a") for s in chain["stages"]):
            # adoption created the need: without the earlier adoption, the need never arises
            return nid not in arm["outcome"]["needs_ever"]
        # without the artifact the need is never relieved by anything: still there at the end
        return bool(later) and nid in arm["outcome"]["needs"]
    if chain["chain"].startswith("competition:"):
        return arm["outcome"]["incumbent_price_changes"] < A["factual"]["outcome"]["incumbent_price_changes"] or \
            not any(c["chain"] == chain["chain"] and c["n_stages"] >= 3 for c in arm["chains"])
    return False


def test_L1_agent_capabilities_alter_existing_conditions(A):
    f = A["factual"]["outcome"]
    effects = {}
    for arm in ("no_saas", "no_novelty"):
        o = A[arm]["outcome"]
        effects[f"hand_work_minutes_{arm}"] = o["diy_minutes"] - f["diy_minutes"]
    inc_f = sum(f["incumbent_live_subs"].values())
    inc_r = sum(A["no_saas"]["outcome"]["incumbent_live_subs"].values())
    effects["incumbent_subscriptions_no_saas"] = inc_r - inc_f
    effects["incumbent_price_changes"] = f["incumbent_price_changes"] - A["no_saas"]["outcome"]["incumbent_price_changes"]
    n = sum(1 for k, v in effects.items() if v > 0)
    record("L.1", {"capability_effects_measured": n, **{k: float(v) for k, v in effects.items()}},
           notes="positive = the condition is changed by the agent-created capability (removal arm minus factual)")
    assert n >= 1


def test_L2_new_participant_needs_arise(A):
    h = A["factual"]["history"]
    t0 = h[0]["t"]
    first = {}
    for x in h:
        for nid, n in x["needs"].items():
            first.setdefault(nid, (x["t"], n))
    emerged = {k: v for k, v in first.items() if v[0] > t0 + DAY}
    record("L.2", {"emerged_needs": len(emerged)}, notes="; ".join(
        f"{k}: rule {v[1]['rule']}, first day {v[0] // DAY}, {v[1]['holders']} holders" for k, v in sorted(emerged.items())))
    assert emerged


def test_L3_new_services_address_needs(A):
    addressed = [c for c in A["factual"]["chains"] if c["chain"].startswith("need:")
                 and any(s["name"] == "need relieved" for s in c["stages"] if s["t"] is not None)
                 and any(s["name"].startswith("agent artifact") for s in c["stages"] if s["t"] is not None)]
    record("L.3", {"needs_addressed_by_new_services": len(addressed)},
           notes="; ".join(f"{c['chain']}: " + " -> ".join(s["name"] for s in c["stages"] if s["t"] is not None)
                           for c in addressed))
    assert addressed


def test_L4_multi_stage_chains(A):
    good = _good_chains(A["factual"]["chains"])
    top = sorted(good, key=lambda c: -c["n_stages"])
    kinds = {"agent_artifact": any(c["agent_artifact"] for c in top), "social": any(c["social"] for c in top),
             "technical": any(c["technical"] for c in top)}
    record("L.4", {"independent_chains": len(good), "min_chain_stages": min((c["n_stages"] for c in top[:3]), default=0),
                   **{f"has_{k}": v for k, v in kinds.items()}, "scripted_agents": True},
           notes="; ".join(f"{c['chain']} ({c['n_stages']}): " + " -> ".join(
               f"{s['name']} [{s['subsystem']}, d{(s['t'] or 0) // DAY}]" for s in c["stages"] if s["t"] is not None)
               for c in top))
    assert len(good) >= 3 and min(c["n_stages"] for c in top[:3]) >= 3 and all(kinds.values())


def test_L5_counterfactual_removal_confirms(A):
    good = _good_chains(A["factual"]["chains"])
    conf = [(c["chain"], _removal_arm(c), _confirmed(A, c)) for c in good]
    n = sum(1 for _, _, ok in conf if ok)
    record("L.5", {"counterfactual_removals_confirming": n, "chains_tested": len(conf)},
           notes="; ".join(f"{c} (removed in {a}): {'confirmed' if ok else 'not confirmed'}" for c, a, ok in conf))
    assert n >= 3


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


def test_Ln1_no_special_case_success_bonuses(A):
    creators = set()
    for a in ("factual", "garbage", "no_saas", "no_novelty"):
        r = A[a]
        for ag in (r["maker"], r["saas"]):
            if ag is not None and ag.org:
                creators.add((a, ag.me))
                creators.add((a, ag.org))
    total = 0
    per = {}
    for a in ("factual", "garbage", "no_saas", "no_novelty"):
        w = A[a]["world"]
        res = M.special_case_bonuses(w, sorted(c for arm, c in creators if arm == a))
        per[a] = res["bonuses"]
        total += res["bonuses"]
    static = _system_grants_in_code()
    record("L.n1", {"special_case_bonuses": int(total + len(static)), "runtime_bonuses": int(total),
                    "static_success_grants": len(static)},
           notes=f"per arm {per}; static findings {static}")
    assert total == 0 and not static
