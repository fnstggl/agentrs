"""G2-learn probe: D's credit and continual-learning machinery inside the INTEGRATED agent (CamLiveAgent driven by
the real LiveRuntime against LabWorld with the signing LabVerifier; tiny backbone + debug core, CPU) — the setting
D's status file still describes as "not integrated".

    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_agent_credit.py [--seeds 11 12 13]

Checks on the agent's own store and logs after the lifetimes (no code under test is modified):
  RC-01  every matured declared outcome is linked to its decision; per outcome the link weights sum to <= 1;
  RC-07  no link to a decision made after its outcome; no stored outcome dated before its decision; the linker's
         leakage audit is empty;
  RC-06  a verified objective verdict reaches the store (put_objective_outcome) and versioned credit is written for the
         earlier decisions of closed objectives; the retrain queue is fed and drained;
  RC-05  COCOA support status of the agent's decisions (open-ended / tracked proposals);
  RC-08  the estimator selection: what evidence exists inside the agent (holdout_experiments=False there);
  CL-02 / CL-03 / CL-08  frequent / slow / fast updates in the learning log, slow-step guard verdicts (how many slow
         steps ran without a held-out verdict), reverted steps;
  CL-05  promotion attempts and their decisions.
"""
import collections
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import TESTS, WORK, check, finish, info  # noqa: E402

sys.path.insert(0, os.path.join(TESTS, "integration"))
from int_helpers import Recorder, agent_config, run_lifetime  # noqa: E402

import torch  # noqa: E402

from cam_live.agent import CamLiveAgent  # noqa: E402
from cam_live.contracts import OutcomeStatus, RunFiles, SimClock  # noqa: E402
from cam_live.credit.providers import candidate_set_open_ended, proposals_tracked  # noqa: E402
from cam_live.runtime.logs import iter_jsonl  # noqa: E402

seeds = [11, 12, 13]
if "--seeds" in sys.argv:
    i = sys.argv.index("--seeds") + 1
    seeds = []
    while i < len(sys.argv) and not sys.argv[i].startswith("--"):
        seeds.append(int(sys.argv[i]))
        i += 1
torch.set_num_threads(2)
root = os.path.join(WORK, "agent_credit")
shutil.rmtree(root, ignore_errors=True)
os.makedirs(root)
clock = SimClock()
agent = CamLiveAgent(agent_config(policy_update_every=6), os.path.join(root, "agent"), clock=clock)
rec = Recorder(agent)
lab = {"outcome_horizon_s": 1800.0, "mean_delay_s": (300.0, 900.0)}
lts = [run_lifetime(rec, clock, root, s, max_steps=40, max_decisions=22, lab_kw=lab) for s in seeds]
info("lifetimes (status)", [lt.status for lt in lts])
st = agent.store
recs = st.all()
info("records", len(recs))

# ---- RC-01 / RC-07 ------------------------------------------------------------------------------------------------
matured, unlinked = 0, []
for r in recs:
    qids = {q.query_id for q in list(r.pending_queries) + (list(r.result.expected_outcomes) if r.result else [])}
    for o in r.outcomes:
        if o.query_id in qids and o.status == OutcomeStatus.OBSERVED and o.dedup_key and not o.dedup_key.startswith("self:"):
            matured += 1
            if not any(l["outcome_key"] for l in st.links(exp_id=r.exp_id)):
                unlinked.append(r.exp_id)
links = [l for r in recs for l in st.links(exp_id=r.exp_id)]
by_key = collections.defaultdict(float)
for l in links:
    by_key[l["outcome_key"]] += float(l["weight"])
over = {k: v for k, v in by_key.items() if v > 1.0 + 1e-9}
info("matured declared outcomes / links / distinct linked outcomes", (matured, len(links), len(by_key)))
check("RC-01: every matured declared outcome of the agent's records is linked", matured > 0 and not unlinked,
      f"{matured} matured, unlinked records {unlinked[:5]}")
check("RC-01: per outcome the link weights sum to at most 1", not over, f"{len(over)} outcomes over 1: {list(over.items())[:3]}")
dec_ts = {r.exp_id: r.decision.ts for r in recs}
later = [l for l in links if float(l["decision_ts"]) > float(l["outcome_ts"]) + 1e-9]
check("RC-07: no link to a decision made after its outcome", not later, f"{len(later)}")
before_dec = [(r.exp_id, o.query_id, o.observed_at, r.decision.ts) for r in recs for o in r.outcomes
              if o.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED) and o.observed_at
              and o.observed_at + 1e-9 < r.decision.ts]
check("RC-07: no stored outcome is dated before the decision it is attached to", not before_dec, f"{before_dec[:3]}")
leaks = agent.linker.audit_leakage()
check("RC-07: the linker's leakage audit finds no context item dated after its decision", not leaks, f"{leaks[:3]}")

# ---- RC-06 --------------------------------------------------------------------------------------------------------
objs = sorted({r.objective_id for r in recs if r.objective_id})
closed = [o for o in objs if st.objective_outcome(o) is not None]
info("objectives / closed in the store", (len(objs), len(closed)))
check("RC-06: verified objective verdicts reach the store (put_objective_outcome by the agent)", len(closed) >= 1,
      f"{len(closed)} of {len(objs)} closed")
hist = st.credit_history()
by_est = collections.Counter(h["estimator"] for h in hist if h["value"] is not None)
info("credit history entries by estimator", dict(by_est))
in_closed = {r.exp_id for r in recs if r.objective_id in closed}
learned = {h["exp_id"] for h in hist if h["estimator"] in ("rudder", "cocoa") and h["value"] is not None}
cov = len(learned & in_closed) / max(1, len(in_closed))
check("RC-06: learned (RUDDER / COCOA) credit is written retrospectively for decisions of closed objectives",
      len(learned & in_closed) > 0, f"{len(learned & in_closed)} of {len(in_closed)} records of closed objectives "
      f"({cov:.2f})")
versions = collections.Counter((h["exp_id"], h["estimator"]) for h in hist)
info("records whose credit was re-versioned (more than one version)", sum(1 for v in versions.values() if v > 1))
info("retrain queue entries left at the end", st.retrain_count())

# ---- RC-05 --------------------------------------------------------------------------------------------------------
oe = [candidate_set_open_ended(r) for r in recs]
tr = [proposals_tracked(r) for r in recs]
info("decisions: open-ended candidate sets / tracked proposals / COCOA-supportable (closed or tracked)",
     (sum(oe), sum(tr), sum(1 for a, b in zip(oe, tr) if (not a) or b)))
cocoa_on_oe = [r.exp_id for r, a, b in zip(recs, oe, tr) if a and not b and r.exp_id in
               {h["exp_id"] for h in hist if h["estimator"] == "cocoa" and h["value"] is not None}]
check("RC-05: no COCOA credit value for a decision with an open-ended untracked candidate set", not cocoa_on_oe,
      f"{len(cocoa_on_oe)} such records carry COCOA credit: {cocoa_on_oe[:3]}")

# ---- RC-08 --------------------------------------------------------------------------------------------------------
sel = os.path.join(agent.dir, "credit_selection.jsonl")
sel_rows = list(iter_jsonl(sel)) if os.path.exists(sel) else []
info("estimator selection log rows", len(sel_rows))
if sel_rows:
    last = sel_rows[-1]
    info("last selection (default, holdout, excluded)", json.dumps({k: last.get(k) for k in
                                                                    ("default_after", "holdout", "excluded_not_held_out",
                                                                     "reasons")}, default=str)[:600])
# can the agent compare its estimators with its own randomised experiments at all?
from cam_live.credit.links import record_arm  # noqa: E402
exps = [e for e in st.experiments() if getattr(e, "assignment", "") == "randomized" and len(e.arms) >= 2]
by_exp = collections.defaultdict(set)
for r in recs:
    if r.decision.experiment_id:
        by_exp[r.decision.experiment_id].add(record_arm(r))
info("randomised experiments: registered in the experience store / experiment ids on the agent's decisions",
     (len(exps), len(by_exp)))
cmp_out = []
for eid, arms in list(by_exp.items())[:6]:
    arms = sorted(a for a in arms if a)
    if len(arms) < 2:
        cmp_out.append((eid[:14], f"arms on the decisions: {arms}"))
        continue
    try:
        ev = agent.assigner.compare_with_experiment(eid, treatment=arms[0], control=arms[1])
        cmp_out.append((eid[:14], None if ev is None else {k: getattr(ev, k, None) for k in
                                                            ("held_out", "n_excluded_trained", "randomized")}))
    except Exception as ex:  # noqa: BLE001
        cmp_out.append((eid[:14], f"{type(ex).__name__}: {str(ex)[:80]}"))
info("compare_with_experiment on the agent's own experiments (None = fewer than two matured held-out units per arm)",
     cmp_out)
trained = set(getattr(agent.learner, "trained_objectives", set()))
exp_objs = {r.objective_id for r in recs if r.decision.experiment_id}
info("objectives with randomised decisions / of them trained on by the learner (holdout_experiments="
     f"{agent.learner.settings.holdout_experiments})", (len(exp_objs), len(exp_objs & trained)))
# KF-E01 (later commits): does the stored credit reach the policy update?
cf = getattr(getattr(getattr(agent, "meta", None), "inner", None), "credit_fn", None)
if cf is not None:
    used = collections.Counter()
    for r in recs:
        try:
            c = cf(r)
        except Exception:  # noqa: BLE001
            c = None
        used["none" if c is None else f"{c[1]}:{'return_like' if c[2] else 'per_decision'}"] += 1
    info("records whose stored credit the policy update's goal term would use (agent.meta.inner.credit_fn)", dict(used))
    check("RC-02: stored credit reaches the policy update for some decisions (credit_fn returns a value)",
          sum(v for k, v in used.items() if k != "none") > 0, f"{dict(used)}")
else:
    info("agent.meta.inner.credit_fn", "absent (stored credit does not reach the policy update at this commit)")
info("credit default used by the agent's assigner", getattr(agent.assigner, "default", None) if not callable(
    getattr(agent.assigner, "default", None)) else agent.assigner.default())

# ---- CL-02 / CL-03 / CL-08 / CL-05 --------------------------------------------------------------------------------
# the learner's own log (ContinualLearner log_path); the runtime's copy (RunFiles.LEARNING) is checked for integrity
ups = []
for line in open(os.path.join(agent.dir, "learner_updates.jsonl")):
    d = json.loads(line)
    ups.append(d.get("data", d))


def notes_of(u):
    n = u.get("notes") or "{}"
    return json.loads(n) if isinstance(n, str) else n


ts = collections.Counter(u["timescale"] for u in ups)
info("learning updates by timescale (learner log)", dict(ts))
slow = [u for u in ups if u["timescale"] == "slow_adapters"]
applied = [u for u in slow if (u.get("weight_delta") or {}).get("semantic_adapters", 0.0) > 0 or
           (notes_of(u).get("guard") or {}).get("reverted")]
unguarded = [u for u in applied if notes_of(u).get("guard") is None]
reverted = [u for u in applied if (notes_of(u).get("guard") or {}).get("reverted")]
info("slow updates: attempted / applied-or-reverted / without a guard verdict / reverted",
     (len(slow), len(applied), len(unguarded), len(reverted)))
info("guard verdicts of the guarded slow steps", [{k: notes_of(u)["guard"].get(k) for k in
                                                   ("accepted", "n_chunks", "paired_se", "tolerance", "paired_change")}
                                                  for u in applied if notes_of(u).get("guard")])
check("CL-03: adapter (slow) updates occur in the integrated agent", len(applied) > 0, f"{len(applied)}")
check("CL-03: every applied slow step in the integrated agent carries a held-out guard verdict", not unguarded,
      f"{len(unguarded)} of {len(applied)} applied slow steps ran without any held-out check (no validation cells)")
freq = [u for u in ups if u["timescale"] == "frequent" and (u.get("weight_delta") or {}).get("experiential", 0) > 0]
check("CL-02: frequent updates change experiential weights in the integrated agent", len(freq) > 0, f"{len(freq)}")
beh = sum(int(bool(notes_of(u).get("behaviour_changed"))) for u in freq)
info("frequent updates whose behavioural probe changed", (beh, len(freq)))
check("CL-02: frequent updates of the integrated agent change predictions or behaviour (probe)", beh > 0,
      f"{beh} of {len(freq)}")
trig = collections.Counter(u.get("trigger", "")[:40] for u in ups if u["timescale"] == "frequent")
info("frequent update triggers", dict(trig.most_common(6)))
val = agent.learner.validate()
info("learner.validate() at the end (held-out losses available to the slow guard and to promotion)", val)
# runtime copy of the learning log: parseable notes?
bad_notes = 0
n_rt = 0
for lt in lts:
    for e in iter_jsonl(os.path.join(lt.run_dir, RunFiles.LEARNING)):
        n_rt += 1
        try:
            notes_of(e["data"])
        except Exception:  # noqa: BLE001
            bad_notes += 1
info("runtime learning log (RunFiles.LEARNING): records whose notes JSON no longer parses", (bad_notes, n_rt))
info("promotions (decision, reasons)", [(p.get("promoted"), (p.get("reasons") or [])[:2]) for p in agent.promotions][:6])
agent.close()
sys.exit(finish())
