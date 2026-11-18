"""Automatic reward-contract audit with adversarial scripted policies.

For every environment family, scripted policies that (a) terminate early, (b) waste resources,
(c) preserve resources without making progress, (d) reach intermediate milestones but fail and
(e) succeed (an oracle that reads the hidden simulator state) are rolled out on fixed validation
seeds. The *training objective* each would receive — the expected discounted return under the reward
contract, J = E[sum_t (prod_{k<t} discount_k) r_t] — is computed, and orderings that any sound
objective must satisfy are checked:

  1. the successful strategy has the highest objective;
  2. terminating early is never better than preserving resources or reaching milestones;
  3. wasting resources is never better than preserving them;
  4. with a positive constraint multiplier, ending by a hard-constraint violation is strictly worse
     than preserving resources.

These scripted policies are test fixtures only: they are never shown to, or imitated by, CAM.

    PYTHONPATH=src python -m agentrs.cam.audit --mode legacy      # the original reward (fails)
    PYTHONPATH=src python -m agentrs.cam.audit --mode contract --lam 0.5
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json

import numpy as np

from agentrs.cam.contract import RewardContract, RewardSpec
from agentrs.cam.envs import ENVS

# ---------------------------------------------------------------------------- product fixtures
from agentrs.cam.envs.product import CHANNELS, FEATS, PRICES, SEGS  # noqa: E402


def _p_early(e, st):
    i = st.setdefault("i", 0)
    st["i"] += 1
    return ["CAMPAIGN", CHANNELS[i], "B4"] if i < 3 else ["WAIT", "W1"]


def _p_waste(e, st):
    i = st.setdefault("i", 0)
    st["i"] += 1
    if i < len(FEATS):
        return ["BUILD", FEATS[i]] if not (e.built[i] or i in e.building) else ["RESEARCH", "S0"]
    return ["ONBOARD"] if e.onb < 3 else ["RESEARCH", SEGS[i % 4]]


def _p_preserve(e, st):
    return ["WAIT", "W1"]


def _p_milestone(e, st):
    i = st.setdefault("i", 0)
    st["i"] += 1
    if i < 4:
        return ["RESEARCH", SEGS[i]]
    if "b" not in st:
        cnt = {}
        for top, _ in e.research.values():
            for k, f in enumerate(top):
                cnt[f] = cnt.get(f, 0) + 2 - k
        st["b"] = sorted(cnt, key=lambda f: -cnt[f])[:2]
    if st["b"]:
        return ["BUILD", st["b"].pop(0)]
    return ["CAMPAIGN", "ads", "B1"] if e.spend_left[0] == 0 else ["WAIT", "W1"]


def _p_succeed(e, st):
    """Oracle: reads the hidden market, but pays for and sequences everything through the real actions."""
    if "plan" not in st:
        w = (e.needs * e.sizes[:, None]).sum(0)
        st["plan"] = [["BUILD", FEATS[int(f)]] for f in np.argsort(-w)[:3]]
        st["ch"] = int(np.argmax((e.aff[:, :3] * e.sizes[:, None]).sum(0) * e.eff[:3]))
        st["onb"], st["priced"] = 2, False
    if st["plan"]:
        return st["plan"].pop(0)
    if e.spend_left[st["ch"]] == 0:
        return ["CAMPAIGN", CHANNELS[st["ch"]], "B2"]
    if not st["priced"] and not e.building:
        st["priced"] = True
        ee = copy.deepcopy(e)
        best = max(range(len(PRICES)), key=lambda q: (setattr(ee, "price", q),
                                                      (ee._accept() * ee.sizes).sum() * max(1.0, PRICES[q]) ** 0.3)[1])
        return ["SETPRICE", f"P{best}"]
    if st["onb"] > 0 and e.onb < 3:
        st["onb"] -= 1
        return ["ONBOARD"]
    return ["WAIT", "W1"]


# ---------------------------------------------------------------------------- negotiation fixtures
from agentrs.cam.envs.negotiation import ALL, ISSUES, LEVELS  # noqa: E402


def _offer(x, arg="NOARG"):
    return ["PROPOSE", *[LEVELS[int(v)] for v in x], arg]


def _n_early(e, st):
    return ["WALK"]


def _n_waste(e, st):  # gives away everything: the worst offer for itself, accepted at once
    return _offer(ALL[int(np.argmin(e.u_a_all))])


def _n_preserve(e, st):  # never commits; keeps goodwill by only asking
    return ["ASK", ISSUES[e.round % 4]]


def _n_milestone(e, st):  # gets counteroffers, then accepts the first one (an agreement, usually below target)
    if e.last_cp is not None:
        return ["ACCEPT"]
    return _offer(ALL[int(np.argmax(e.u_a_all))])


def _n_succeed(e, st):
    """Oracle: knows the counterparty's utility and its acceptance threshold for the next round."""
    e.round += 1
    thr = e._threshold()
    e.round -= 1
    feas = e.u_c_all >= thr
    if feas.any():
        j = int(np.argmax(np.where(feas, e.u_a_all, -1)))
        if e.u_a_all[j] >= e.target:
            return _offer(ALL[j], e.concern)
    ok = e.u_a_all >= e.target
    if e.cp_type == "titfortat" and ok.any():  # concede within what still meets the target
        j = int(np.argmax(np.where(ok, e.u_c_all, -1)))
        return _offer(ALL[j], e.concern)
    return ["ASK", ISSUES[e.round % 4]]


# ---------------------------------------------------------------------------- bizdev fixtures
from agentrs.cam.envs.bizdev import NP, PARTNERS, SHARES  # noqa: E402


def _terms(e, i):
    share = e.min_share[i] + (2 if e.want[i] == "share" else 0)
    return share, "EXCL" if e.want[i] == "exclusive" else "NOEXCL", "FAST" if e.want[i] == "speed" else "SLOW"


def _b_waste(e, st):
    for i in range(NP):
        if e.status[i] in ("UNKNOWN", "OBJECTION", "INTERESTED"):
            return ["PITCH", PARTNERS[i], "N10", "EXCL", "FAST"]
    return ["WAIT"]


def _b_preserve(e, st):
    return ["WAIT"]


def _b_milestone(e, st):
    i = st.setdefault("i", 0)
    st["i"] += 1
    if i < NP:
        return ["RESEARCH", PARTNERS[i]]
    if "p" not in st:
        st["p"] = next((j for j in range(NP) if e.anchor[j] is None and e.want[j] != "proof"), 0)
    p = st["p"]
    if e.status[p] in ("UNKNOWN", "OBJECTION"):
        s, x, f = _terms(e, p)
        return ["PITCH", PARTNERS[p], SHARES[min(10, s)], x, f]
    return ["FOLLOWUP", PARTNERS[p]] if e.status[p] == "PENDING" else ["WAIT"]


def _b_succeed(e, st):
    """Oracle: knows every partner's terms, anchors and rivals; pitches a feasible set in a feasible order."""
    if "order" not in st:
        st["order"] = []
        best = None
        for combo in itertools.permutations(range(NP), e.k):
            tot = sum(_terms(e, i)[0] for i in combo)
            if tot > e.cap:
                continue
            ok = all((e.anchor[i] is None or e.anchor[i] in combo[:p]) and (e.rival[i] not in combo[:p])
                     and (e.want[i] != "proof" or p > 0) and not any(e.rival[j] == i for j in combo[p + 1:])
                     for p, i in enumerate(combo))
            if ok and (best is None or tot < best[0]):
                best = (tot, combo)
        st["order"] = list(best[1]) if best else []
    for i in st["order"]:
        if e.status[i] == "COMMITTED":
            continue
        if e.status[i] == "PENDING":
            return ["FOLLOWUP", PARTNERS[i]]
        if e.anchor[i] is not None and e.status[e.anchor[i]] != "COMMITTED":
            return ["WAIT"]
        if e.want[i] == "proof" and not any(s == "COMMITTED" for s in e.status):
            return ["WAIT"]
        s, x, f = _terms(e, i)
        return ["PITCH", PARTNERS[i], SHARES[min(10, s)], x, f]
    return ["WAIT"]


POLICIES = {
    "product": {"early_terminate": _p_early, "waste": _p_waste, "preserve": _p_preserve, "milestone": _p_milestone,
                "succeed": _p_succeed},
    "negotiation": {"early_terminate": _n_early, "waste": _n_waste, "preserve": _n_preserve, "milestone": _n_milestone,
                    "succeed": _n_succeed},
    "bizdev": {"waste": _b_waste, "preserve": _b_preserve, "milestone": _b_milestone, "succeed": _b_succeed},
}


def rollout(fam: str, policy, rc: RewardContract, seed: int, split: str = "val", lam: float = 0.0) -> dict:
    env = ENVS[fam]()
    env.reset(seed, split)
    c = env.contract()
    s0 = env.task_status()
    st: dict = {}
    J, acc, n = 0.0, 1.0, 0
    while not env.done and n < 2 * env.max_steps:
        res = env.step(policy(env, st))
        n += 1
        s1 = env.task_status()
        tr = rc.transition(c, s0, s1, res.done, bool(res.success), lam, n, env.max_steps)
        J += acc * tr.reward
        acc *= tr.discount
        s0 = s1
    fs = env.task_status()
    return {"J": J, "success": bool(env.success), "violation": fs.violated is not None, "clock": fs.clock,
            "steps": n, "milestones": sum(bool(v) for v in fs.milestones.values()), "cause": env.failure_cause()}


def audit(rc: RewardContract, families=("product", "negotiation", "bizdev"), n: int = 40, lam: float | dict = 0.0,
          tol: float = 1e-6) -> dict:
    out = {"reward": vars(rc.spec), "families": {}, "violations": []}
    for fam in families:
        lf = lam.get(fam, 0.0) if isinstance(lam, dict) else lam
        res = {}
        for name, pol in POLICIES[fam].items():
            rs = [rollout(fam, pol, rc, 50_000 + s, "val", lf) for s in range(n)]
            res[name] = {"J": float(np.mean([r["J"] for r in rs])), "success": float(np.mean([r["success"] for r in rs])),
                         "violation": float(np.mean([r["violation"] for r in rs])),
                         "clock_used": float(np.mean([r["clock"] for r in rs])),
                         "milestones": float(np.mean([r["milestones"] for r in rs]))}
        J = {k: v["J"] for k, v in res.items()}
        bad = []
        for k in J:
            if k != "succeed" and J[k] >= J["succeed"] - tol:
                bad.append(f"{fam}: '{k}' (success {res[k]['success']:.2f}) scores >= 'succeed' "
                           f"(success {res['succeed']['success']:.2f}): {J[k]:.4f} vs {J['succeed']:.4f}")
        if "early_terminate" in J:
            for k in ("preserve", "milestone"):
                if J["early_terminate"] > J[k] + tol:
                    bad.append(f"{fam}: terminating early beats '{k}': {J['early_terminate']:.4f} > {J[k]:.4f}")
        if J["waste"] > J["preserve"] + tol:
            bad.append(f"{fam}: wasting resources beats preserving them: {J['waste']:.4f} > {J['preserve']:.4f}")
        if lf > 0:
            for k, v in res.items():
                if v["violation"] > 0.5 and v["success"] == 0 and J[k] >= J["preserve"] - tol:
                    bad.append(f"{fam}: constraint-violating '{k}' is not worse than 'preserve' despite lam={lf}")
        out["families"][fam] = {"policies": res, "violations": bad}
        out["violations"] += bad
    out["passed"] = not out["violations"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="contract")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)
    r = audit(RewardContract(RewardSpec(mode=a.mode)), n=a.n, lam=a.lam)
    for fam, v in r["families"].items():
        print(f"== {fam}")
        for k, p in v["policies"].items():
            print(f"  {k:16s} J {p['J']:+.4f}  success {p['success']:.2f}  violation {p['violation']:.2f}  "
                  f"clock {p['clock_used']:.1f}  milestones {p['milestones']:.1f}")
    print("PASSED" if r["passed"] else "FAILED:\n  " + "\n  ".join(r["violations"]))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
