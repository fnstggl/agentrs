"""G2-learn probe: attacks on D's fixes of G-cl M2 (append-only history), M3 (attribution amounts) and M4 (slow-step
guard) that the earlier probes did not try.

    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_credit_gaps.py [--only store,links,guard]

store  (CL-01, RC-07)  update() may append outcomes: is the add_outcome() refusal of an outcome dated before its
                       query also applied there? protected decision diagnostics; the approval placeholder is
                       replaceable once only; a tampered index column is detected on read; items are immutable.
links  (RC-01)         attribution maps: unknown ids, booleans, overflow; weights of one outcome never exceed 1 and the
                       credited amount never exceeds the outcome.
guard  (CL-03, CL-08)  the slow-step guard on the real model class (debug scale, D's own lab): a degrading step is
                       undone (control); what happens without held-out records; whether the aggregate over providers
                       can hide one provider's degradation.
"""
import json
import math
import os
import shutil
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import TESTS, WORK, check, finish, info  # noqa: E402

sys.path.insert(0, os.path.join(TESTS, "credit"))
from conftest import build_record, make_lab  # noqa: E402

import torch  # noqa: E402

from cam_live.contracts import (ActionResult, ExperienceItem, ItemKind, Observation, OutcomeObservation,  # noqa: E402
                                OutcomeStatus, Provenance, Quantity, ResultStatus, SimClock)
from cam_live.credit.links import OutcomeLinker  # noqa: E402
from cam_live.credit.replay import ImmutableHistoryError, SqliteExperienceStore, StoreIntegrityError  # noqa: E402

ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else ["store", "links", "guard"]


def fresh(name):
    p = os.path.join(WORK, "credit_gaps", name)
    shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p)
    return p


def refused(fn):
    try:
        fn()
        return False, ""
    except (ImmutableHistoryError, ValueError, KeyError) as e:
        return True, f"{type(e).__name__}: {str(e)[:120]}"


# ================================================================================================== store
if "store" in ONLY:
    st = SqliteExperienceStore(os.path.join(fresh("store"), "s.sqlite"))
    r = build_record(1)                      # decision at 1010, query issued at 1015 (ts + duration)
    st.append(r)
    q = r.pending_queries[0]
    early = OutcomeObservation(query_id=q.query_id, status=OutcomeStatus.OBSERVED, value=5.0, observed_at=900.0,
                               dedup_key="early")
    res = st.add_outcome(early)
    check("control: add_outcome refuses an outcome observed before its query was issued", not res.accepted,
          f"accepted={res.accepted} reason={res.reason!r} (query issued {q.issued_at})")
    rec = st.get(r.exp_id)
    rec.outcomes.append(early)
    ok, why = refused(lambda: st.update(rec))
    stored = [o for o in st.get(r.exp_id).outcomes if o.query_id == q.query_id]
    check("update() applies the same refusal to an appended outcome dated before its query (RC-07 guard)", ok,
          f"refused={ok} {why}; stored outcomes of the query now {[(o.value, o.observed_at) for o in stored]}")
    # an appended outcome dated before the DECISION itself
    r2 = build_record(2)
    st.append(r2)
    rec2 = st.get(r2.exp_id)
    rec2.outcomes.append(OutcomeObservation(query_id=r2.pending_queries[0].query_id, status=OutcomeStatus.OBSERVED,
                                            value=7.0, observed_at=r2.decision.ts - 500.0, dedup_key="before_decision"))
    ok, why = refused(lambda: st.update(rec2))
    check("update() refuses an appended outcome observed before the decision was made", ok, f"refused={ok} {why}")
    # protected diagnostics
    r3 = build_record(3)
    st.append(r3)
    rec3 = st.get(r3.exp_id)
    rec3.decision.diagnostics["tracked_proposals"] = True
    ok, why = refused(lambda: st.update(rec3))
    check("update() refuses adding a protected decision diagnostic after the fact (tracked_proposals)", ok, why)
    rec3 = st.get(r3.exp_id)
    rec3.decision.probabilities = [0.9, 0.05, 0.05]
    ok, why = refused(lambda: st.update(rec3))
    check("update() refuses rewriting behaviour probabilities", ok, why)
    # approval placeholder: replaceable once
    r4 = build_record(4, status=ResultStatus.NEEDS_APPROVAL)
    st.append(r4)
    rec4 = st.get(r4.exp_id)
    rec4.result = ActionResult(action_id=rec4.decision.chosen.action_id, status=ResultStatus.OK, started=1050.0,
                               ended=1055.0)
    ok1, why1 = refused(lambda: st.update(rec4))
    rec4 = st.get(r4.exp_id)
    rec4.result = ActionResult(action_id=rec4.decision.chosen.action_id, status=ResultStatus.FAILED, started=1050.0,
                               ended=1056.0)
    ok2, why2 = refused(lambda: st.update(rec4))
    check("a needs_approval placeholder is replaced once by the executed result, then the result is immutable",
          (not ok1) and ok2, f"first replacement refused={ok1} {why1}; second refused={ok2} {why2}")
    # tampered index column
    st.close()
    con = sqlite3.connect(os.path.join(WORK, "credit_gaps", "store", "s.sqlite"))
    con.execute("UPDATE experiences SET provenance='staging' WHERE exp_id=?", (r3.exp_id,))
    con.commit()
    con.close()
    st = SqliteExperienceStore(os.path.join(WORK, "credit_gaps", "store", "s.sqlite"))
    try:
        st.get(r3.exp_id)
        caught, why = False, "returned the record"
    except StoreIntegrityError as e:
        caught, why = True, str(e)[:120]
    check("a tampered index column (provenance) is detected on read", caught, why)
    try:
        live = st.sample(8, loss="world", provenance="live", seed=0)
        bad = [x.exp_id for x in live if x.provenance != "live"]
        check("sample(provenance='live') never returns a record whose stored provenance differs", not bad,
              f"{len(live)} sampled, mismatching {bad}")
    except StoreIntegrityError as e:
        check("sample(provenance='live') never returns a record whose stored provenance differs", True,
              f"refused: {str(e)[:100]}")
    # items are immutable
    it = ExperienceItem(kind=ItemKind.OBSERVATION, text="v1", ts=1.0, ref="item1")
    st.put_item(it)
    try:
        st.put_item(ExperienceItem(kind=ItemKind.OBSERVATION, text="v2 rewritten", ts=1.0, ref="item1"))
    except Exception as e:  # noqa: BLE001
        info("put_item of a changed item with an existing ref", f"{type(e).__name__}: {str(e)[:80]}")
    check("a stored context item is never rewritten", st.items(["item1"])[0].text == "v1",
          f"stored text {st.items(['item1'])[0].text!r}")
    st.close()

# ================================================================================================== links
if "links" in ONLY:
    st = SqliteExperienceStore(os.path.join(fresh("links"), "s.sqlite"))
    clock = SimClock(5000.0)
    recs = [build_record(i) for i in range(1, 4)]
    for r in recs:
        st.append(r)
    lk = OutcomeLinker(st, clock=clock)
    a, b = recs[0].decision.chosen.action_id, recs[1].decision.chosen.action_id

    def ev(i, data, value=2.0):
        return Observation(obs_id=f"ev{i}", ts=2000.0 + i, kind="event", text="", data={"metric": "m", "value": value, **data},
                           quantities=[Quantity("m", value)], provenance=Provenance(source="env", trust="environment"))

    cases = [("unknown id among valid amounts", {"by_action": {a: 3.0, "nobody": 1.0}}),
             ("boolean amount", {"by_action": {a: True, b: 1.0}}),
             ("string amount", {"by_action": {a: "3", b: 1.0}}),
             ("amounts that overflow when summed", {"by_action": {a: 1e308, b: 1e308}}),
             ("one valid positive and one zero amount", {"by_action": {a: 2.0, b: 0.0}})]
    for i, (name, data) in enumerate(cases):
        res = lk.link_event(ev(i, data))
        tot = sum(l.weight for l in res.links)
        cred = sum(l.weight * (l.value or 0.0) for l in res.links)
        ok = tot <= 1.0 + 1e-9 and cred <= 2.0 + 1e-9 and all(l.weight >= 0 for l in res.links)
        check(f"attribution map ({name}): weights of one outcome <= 1 and credit <= the outcome", ok,
              f"links {[(l.exp_id, round(l.weight, 4)) for l in res.links]}, total {tot:.4f}, credited {cred:.4f}, "
              f"reason {res.reason!r}")
        if name.startswith("amounts that overflow"):
            check("attribution map whose amounts overflow when summed still links the outcome to the named decisions",
                  len(res.links) == 2 and abs(tot - 1.0) < 1e-9,
                  f"{len(res.links)} links, total weight {tot:.4f} (both decisions named with equal amounts)")
    st.close()

# ================================================================================================== guard
if "guard" in ONLY:
    import hashlib

    from cam_live.config import CamLiveConfig
    from cam_live.credit.learner import LearnerSettings
    from cam_live.learning_api import TIMESCALE_SLOW, UpdatePlan
    from cam_live.nn.api import GROUP_ADAPTERS

    torch.set_num_threads(1)

    class Probe:
        """Slow-timescale loss with a known effect of one step on its held-out value (same construction as D's own
        regression test): training minimises X = 10 + s * mean(adapters); evaluation reports sign * X."""
        timescale = TIMESCALE_SLOW
        needs_latents = False

        def __init__(self, model, name, s, eval_sign, spread=0.5):
            self.model, self.name, self.s, self.eval_sign, self.weight, self.spread = model, name, s, eval_sign, 1.0, spread

        def eligible(self, record):
            return True

        def loss(self, model, batch, store):
            ps = model.param_groups()[GROUP_ADAPTERS]
            n = sum(p.numel() for p in ps)
            x = 10.0 + self.s * sum(p.sum() for p in ps) / n
            f = sum(int(hashlib.sha1(r.exp_id.encode()).hexdigest()[:4], 16) / 65536.0 for r in batch.records)
            x = 10.0 + (x - 10.0) * (1.0 + self.spread * (f / max(len(batch.records), 1) - 0.5))
            return (x if model.training else self.eval_sign * x), {}

    def lab_with(name, **settings):
        cfg = CamLiveConfig.debug()
        cfg.optim.lr_adapters = 1e-4
        stt = dict(slow_guard=True, val_fraction=0.4, val_batch=32, slow_batch_size=4)
        stt.update(settings)
        torch.manual_seed(0)
        return make_lab(__import__("pathlib").Path(fresh(name)), n_worlds=8, n_decisions=6, cfg=cfg, world=False,
                        policy=False, credit=False, settings=LearnerSettings.from_cfg(cfg, **stt))

    def slow(L):
        rec = L.learn(UpdatePlan(trigger="probe", timescales=(TIMESCALE_SLOW,)))[-1]
        return rec, json.loads(rec.notes or "{}")

    # control: a degrading step is undone
    lab = lab_with("guard_control")
    L, m = lab.learner, lab.model
    L.register(Probe(m, "probe", 100.0, -1.0, spread=0.0))
    L.refresh_eligibility(full=True)
    rec, notes = slow(L)
    g = notes.get("guard") or {}
    check("control: a slow step that raises the held-out loss is undone", bool(g) and g.get("reverted") is True,
          f"paired change {g.get('paired_change')}, tolerance {g.get('tolerance')}, reverted {g.get('reverted')}")
    lab.close()

    # 1. no held-out records at all (e.g. early in a lifetime): the same degrading step
    lab = lab_with("guard_noval", val_fraction=0.0)
    L, m = lab.learner, lab.model
    L.register(Probe(m, "probe", 100.0, -1.0, spread=0.0))
    L.refresh_eligibility(full=True)
    w0 = [p.detach().clone() for p in m.param_groups()[GROUP_ADAPTERS]]
    rec, notes = slow(L)
    changed = any(not torch.equal(a, b) for a, b in zip(w0, m.param_groups()[GROUP_ADAPTERS]))
    info("no held-out records: notes of the slow update", {k: notes.get(k) for k in ("guard", "skipped")})
    check("without held-out records a slow step is not kept unguarded (it is skipped or refused)",
          not changed, f"adapters changed={changed}, guard verdict={notes.get('guard')}, timescale record param_groups="
          f"{rec.param_groups}")
    lab.close()

    # 2. two providers: one improves on held-out data, the other degrades by more than the tolerance
    lab = lab_with("guard_mask")
    L, m = lab.learner, lab.model
    L.register(Probe(m, "improves", 100.0, +1.0, spread=0.5))
    L.register(Probe(m, "degrades", 50.0, -1.0, spread=0.5))
    L.refresh_eligibility(full=True)
    before = L._validation_cells()
    rec, notes = slow(L)
    g = notes.get("guard") or {}
    if g.get("reverted"):
        after = before
    else:
        after = L._validation_cells()
    per = {}
    for (prov, j), v in before.items():
        if (prov, j) in after:
            per.setdefault(prov, []).append(after[(prov, j)] - v)
    per = {k: sum(v) / len(v) for k, v in per.items()}
    info("per-provider paired held-out change of the step", per)
    deg = per.get("degrades", 0.0)
    check("a slow step is not kept when one provider's held-out loss rises by more than the tolerance, whatever another "
          "provider gains", not (g.get("accepted") and deg > (g.get("tolerance") or 0.0)),
          f"accepted={g.get('accepted')}, aggregate paired change {g.get('paired_change')}, tolerance "
          f"{g.get('tolerance')}, 'degrades' provider change {deg:.4g}")
    lab.close()

sys.exit(finish())
