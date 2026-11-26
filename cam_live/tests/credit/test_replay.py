"""SqliteExperienceStore: contract conformance, durability, de-duplication, provenance strata, balanced sampling."""
import collections
import json
import os
import signal
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from cam_live.contracts import (ExperienceItem, ExperienceStore, ExperimentRecord, ItemKind, OutcomeKind,
                                OutcomeObservation, OutcomeStatus, PredictionRecord, Distribution, ResultStatus)
from cam_live.credit.replay import DEFAULT_STRATA, SqliteExperienceStore, StoreLockedError
from cam_live.learning_api import matured_outcomes

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _store(tmp_path, **kw):
    kw.setdefault("durable", False)
    return SqliteExperienceStore(str(tmp_path / "experience.sqlite"), **kw)


def _obs(rec, value, at, **kw):
    q = rec.pending_queries[0]
    return OutcomeObservation(query_id=q.query_id, status=OutcomeStatus.OBSERVED, value=value, observed_at=at, **kw)


def test_implements_the_contract_and_round_trips(tmp_path, make_record):
    st = _store(tmp_path)
    assert isinstance(st, ExperienceStore)
    assert st._q("PRAGMA journal_mode")[0][0] == "wal"
    rec = make_record(0)
    rec.seq = 0
    st.append(rec)
    assert rec.seq == 1 and st.count() == 1
    assert st.get("e0").to_dict() == rec.to_dict()
    assert st.by_decision("d0").exp_id == "e0" and st.by_action("a0_0").exp_id == "e0"
    assert st.get("nope") is None and st.by_decision("nope") is None
    with pytest.raises(ValueError):
        st.append(rec)
    with pytest.raises(KeyError):
        st.update(make_record(99))
    rec.intrinsic_reward, rec.credit["simple"], rec.credit_uncertainty["simple"] = 0.25, 1.5, 0.2
    st.update(rec)
    got = st.get("e0")
    assert got.to_dict() == rec.to_dict() and got.seq == 1
    r2 = make_record(1)
    st.append(r2)
    assert r2.seq == 2 and [r.exp_id for r in st.iter_records()] == ["e0", "e1"]
    st.flush()
    st.close()
    st2 = _store(tmp_path)
    assert st2.count() == 2 and st2.get("e0").credit == {"simple": 1.5} and st2.integrity_ok()


def test_items_are_ordered_immutable_and_never_silently_missing(tmp_path):
    st = _store(tmp_path)
    items = [ExperienceItem(kind=ItemKind.OBSERVATION, text=f"t{i}", ts=float(i), ref=f"o{i}") for i in range(5)]
    st.put_items(items)
    st.put_item(items[2])                                      # identical content: no-op
    assert [x.ref for x in st.items(["o3", "o0", "o3"])] == ["o3", "o0", "o3"]
    assert st.items(["o1"])[0].to_dict() == items[1].to_dict()
    with pytest.raises(ValueError):                            # rewriting history is refused
        st.put_item(ExperienceItem(kind=ItemKind.OBSERVATION, text="changed", ts=9.0, ref="o1"))
    with pytest.raises(KeyError):
        st.items(["o1", "ghost"])
    assert len(st.items(["o1", "ghost"], missing_ok=True)) == 1
    anon = ExperienceItem(kind=ItemKind.GOAL, text="g")
    st.put_item(anon)
    assert anon.ref.startswith("item_") and st.items([anon.ref])[0].text == "g"


def test_pending_due_times_and_outcome_deduplication(tmp_path, make_record):
    st = _store(tmp_path)
    a, b = make_record(0, horizon=100.0), make_record(1, horizon=500.0)
    st.append(a), st.append(b)
    assert st.pending(now=1000.0) == [] and st.next_due() == a.pending_queries[0].due_at
    due = st.pending(now=a.pending_queries[0].due_at)
    assert [(r.exp_id, q.query_id) for r, q in due] == [("e0", "q0")]
    assert len(st.pending(now=1e9)) == 2
    # a PENDING observation is not an outcome
    assert not st.add_outcome(OutcomeObservation("q0", OutcomeStatus.PENDING)).accepted
    r = st.add_outcome(_obs(a, 3.0, 1200.0, dedup_key="ev-1"))
    assert r.accepted and r.exp_id == "e0" and r.final
    dup = st.add_outcome(_obs(a, 3.0, 1300.0, dedup_key="ev-1"))          # the same event again
    assert not dup.accepted and dup.duplicate
    got = st.get("e0")
    assert len(got.outcomes) == 1 and got.outcomes[0].value == 3.0
    assert [q.query_id for _, q in st.pending(now=1e9)] == ["q1"] and st.next_due() == b.pending_queries[0].due_at
    # the query definition survives, so the shared helper still pairs query and outcome
    assert [(q.query_id, o.value) for q, o in matured_outcomes(got)] == [("q0", 3.0)]
    # a verified measurement of the same event replaces the unverified one, once
    up = st.add_outcome(_obs(a, 4.0, 1400.0, dedup_key="ev-1", verified=True))
    assert up.accepted and up.upgraded
    got = st.get("e0")
    assert len(got.outcomes) == 1 and got.outcomes[0].verified and got.outcomes[0].value == 4.0
    assert not st.add_outcome(_obs(a, 5.0, 1500.0, dedup_key="ev-1", verified=True)).accepted
    # unobservable resolves the query but is never a matured target
    un = st.add_outcome(OutcomeObservation("q1", OutcomeStatus.UNOBSERVABLE, observed_at=2000.0))
    assert un.accepted and st.pending(now=1e9) == [] and st.next_due() is None
    assert matured_outcomes(st.get("e1")) == []
    # outcomes for unknown queries, or observed before the query existed, are refused
    assert not st.add_outcome(OutcomeObservation("zz", OutcomeStatus.OBSERVED, 1.0, observed_at=5.0)).accepted
    c = make_record(2)
    st.append(c)
    assert not st.add_outcome(_obs(c, 1.0, c.pending_queries[0].issued_at - 50.0)).accepted
    assert [e["dedup_key"] for e in st.outcome_events()] == ["ev-1"]
    assert st.pending_pointer()["n_pending"] == 1


def test_update_deduplicates_outcomes_inside_a_record(tmp_path, make_record):
    st = _store(tmp_path)
    rec = make_record(0)
    st.append(rec)
    rec.outcomes += [_obs(rec, 2.0, 1200.0, dedup_key="k"), _obs(rec, 2.0, 1201.0, dedup_key="k")]
    st.update(rec)
    assert len(st.get("e0").outcomes) == 1 and st.labels("e0")["n_pending"] == 0


def test_provenance_strata_are_never_mixed_unless_asked(tmp_path, make_record):
    st = _store(tmp_path)
    with st.batch():
        for i in range(60):
            st.append(make_record(i, provenance=("live", "staging", "diagnostic")[i % 3]))
    with pytest.raises(ValueError):
        st.append(make_record(999, provenance="imagined"))
    assert st.provenance_counts() == {"live": 20, "staging": 20, "diagnostic": 20}
    for prov in ("live", "staging", "diagnostic"):
        got = st.sample(20, loss="world", provenance=prov, seed=1)
        assert len(got) == 20 and {r.provenance for r in got} == {prov}
    # default: the stratum of the newest record only (here: diagnostic), never a mixture
    assert {r.provenance for r in st.sample(40, loss="world", seed=2)} == {"diagnostic"}
    mixed = st.sample(60, loss="world", provenance=("live", "staging"), seed=3)
    assert {r.provenance for r in mixed} == {"live", "staging"} and len(mixed) == 40
    st.close()
    live_only = _store(tmp_path, provenance="live")
    assert {r.provenance for r in live_only.sample(40, loss="world", seed=2)} == {"live"}


def test_eligibility_is_per_loss_and_fail_closed(tmp_path, make_record):
    st = _store(tmp_path)
    with st.batch():
        for i in range(30):
            st.append(make_record(i, eligible=("world",) if i % 2 else ("world", "policy")))
    assert len(st.sample(100, loss="world", seed=0)) == 30
    pol = st.sample(100, loss="policy", seed=0)
    assert len(pol) == 15 and all(r.eligible.get("policy") for r in pol)
    assert st.sample(10, loss="never_set", seed=0) == []
    assert len(st.sample(100, loss="*", seed=0)) == 30
    assert st.set_eligible(["e1", "e3"], "policy", True) == 2 and st.count(loss="policy") == 17
    rep = st.refresh_eligibility("rudder", lambda r: r.seq <= 10)
    assert rep == {"checked": 30, "eligible": 10, "changed": 30} and st.count(loss="rudder") == 10
    assert st.set_eligible(["e1"], "policy", False) == 1 and st.count(loss="policy") == 16


def _lifetime(st, make_record, n_old=1800, n_recent=200):
    """An old phase with failures, delayed consequences and three objective families, then a recent run of nothing
    but immediate successes in one family."""
    with st.batch():
        for i in range(n_old + n_recent):
            old = i < n_old
            fam = ("famA", "famB", "famC")[i % 3] if old else "famD"
            failed = old and i % 4 == 0
            rec = make_record(i, family=fam, status=ResultStatus.FAILED if failed else ResultStatus.OK,
                              kind=OutcomeKind.BINARY)
            st.append(rec)
            if failed:
                continue
            if old and i % 5 == 0:          # a consequence that arrived long after the action: delayed success
                st.add_outcome(_obs(rec, 1.0, rec.decision.ts + 5000.0))
            elif old and i % 7 == 0:        # resolved negative
                st.add_outcome(_obs(rec, 0.0, rec.decision.ts + 5.5))
            elif not old:                   # recent: immediate success
                st.add_outcome(_obs(rec, 1.0, rec.decision.ts + 5.5))
            if old and i % 50 == 1:
                st.set_surprise(rec.exp_id, 25.0)


def test_sampler_does_not_collapse_onto_recent_successes(tmp_path, make_record):
    st = _store(tmp_path, recent_window=200)
    _lifetime(st, make_record)
    head = st.head_seq()
    recent_cut = head - 200
    tot = collections.Counter()
    fams = collections.Counter()
    old_ids, n = set(), 0
    for b in range(200):
        ids = st.sample_ids(32, loss="world", seed=b)
        assert len(ids) == len(set(ids)) == 32
        for e in ids:
            lab = st.labels(e)
            n += 1
            tot["recent"] += lab["seq"] > recent_cut
            tot["success"] += lab["success"] == 1
            tot["failure"] += lab["success"] == 0
            tot["delayed"] += lab["delayed"] == 1
            tot["recent_success"] += lab["seq"] > recent_cut and lab["success"] == 1
            fams[lab["family"]] += 1
            if lab["seq"] <= recent_cut:
                old_ids.add(e)
    frac = {k: v / n for k, v in tot.items()}
    # the recent successes are 10 % of the lifetime and may not take over the batch
    assert frac["recent_success"] < 0.55, frac
    assert frac["failure"] > 0.15 and frac["delayed"] > 0.15 and frac["recent"] > 0.25, frac
    assert 1 - frac["recent"] > 0.40, frac
    assert min(fams[f] for f in ("famA", "famB", "famC", "famD")) / n > 0.08, fams
    # old experience is covered broadly (no reservoir that keeps replacing it, no fixed top-k)
    assert len(old_ids) > 0.65 * 1800, len(old_ids)
    # a recency-only sampler, for comparison, is all recent successes
    only_recent = st.sample_ids(32, loss="world", seed=0, strata={"recent": 1.0})
    assert all(st.labels(e)["seq"] > recent_cut and st.labels(e)["success"] == 1 for e in only_recent)
    assert st.last_sample_info["strata"] == {"recent": 32}
    print("REPLAY_BALANCE", json.dumps({**{k: round(v, 4) for k, v in frac.items()}, "old_distinct": len(old_ids),
                                        "families": {k: round(v / n, 4) for k, v in fams.items()}}))


def test_informative_stratum_prefers_stored_surprise_and_credit(tmp_path, make_record):
    st = _store(tmp_path)
    with st.batch():
        for i in range(1000):
            st.append(make_record(i))
        for i in range(0, 500, 25):
            st.set_surprise(f"e{i}", 40.0)
        for i in range(10, 500, 25):
            st.set_credit(f"e{i}", "simple", -30.0)
    marked = {f"e{i}" for i in range(0, 500, 25)} | {f"e{i}" for i in range(10, 500, 25)}
    hits = sum(e in marked for b in range(100) for e in st.sample_ids(16, loss="world", seed=b,
                                                                       strata={"informative": 1.0}))
    assert hits / 1600 > 0.15        # 40 of 750 older records (5.3 %) receive > 15 % of the informative draws
    assert all(st.labels(e)["seq"] <= 750 for e in st.sample_ids(64, loss="world", seed=1, strata={"informative": 1.0}))


def test_sampling_is_seeded_and_the_sampler_state_checkpoints(tmp_path, make_record):
    st = _store(tmp_path, seed=7)
    _lifetime(st, make_record, n_old=300, n_recent=60)
    assert st.sample_ids(16, loss="world", seed=5) == st.sample_ids(16, loss="world", seed=5)
    assert st.sample_ids(16, loss="world", seed=5) != st.sample_ids(16, loss="world", seed=6)
    first = st.sample_ids(16, loss="world")
    state = st.state_dict()
    assert state["draws"] == 1 and state["head_seq"] == 360 and state["strata"] == DEFAULT_STRATA
    cont = [st.sample_ids(16, loss="world") for _ in range(3)]
    assert cont[0] != first
    st.close()
    st2 = _store(tmp_path, seed=123)                # a new process: the stream position comes from the database
    assert st2.state_dict()["draws"] == 4 and st2.state_dict()["seed"] == 7
    diff = st2.load_state_dict(json.loads(json.dumps(state)))
    assert diff["records_after_checkpoint"] == 0
    assert [st2.sample_ids(16, loss="world") for _ in range(3)] == cont


def test_large_parts_go_to_blobs_and_updates_do_not_copy_the_static_part(tmp_path, make_record):
    st = _store(tmp_path, blob_threshold=4096)
    rec = make_record(0, text_pad=20000)
    st.append(rec)
    blobs = [f for _, _, fs in os.walk(st.blob_dir) for f in fs]
    assert len(blobs) == 1
    assert st._q("SELECT static_body FROM experiences")[0][0].startswith("blob:")
    assert st.get("e0").to_dict() == rec.to_dict()
    packed = []
    orig = st._pack
    st._pack = lambda text: (packed.append(len(text)), orig(text))[1]
    for k in range(20):
        st.set_credit("e0", "simple", float(k))
    st.add_outcome(_obs(rec, 2.0, 1200.0))
    assert len(packed) == 21 and max(packed) < 4096      # only the small dynamic part was rewritten each time
    assert len([f for _, _, fs in os.walk(st.blob_dir) for f in fs]) == 1
    ref = st.put_blob(b"\x00\x01" * 5000)
    assert st.get_blob(ref) == b"\x00\x01" * 5000 and st.put_blob(b"\x00\x01" * 5000) == ref


def test_single_writer_lock_and_read_only_readers(tmp_path, make_record):
    st = _store(tmp_path)
    st.append(make_record(0))
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {SRC!r})
        from cam_live.credit.replay import SqliteExperienceStore, StoreLockedError
        try:
            SqliteExperienceStore({st.path!r})
            print("OPENED")
        except StoreLockedError:
            print("LOCKED")
        r = SqliteExperienceStore({st.path!r}, readonly=True)
        print(r.count(), r.get("e0").exp_id)
        try:
            r.set_surprise("e0", 1.0)
        except PermissionError:
            print("RO")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.split() == ["LOCKED", "1", "e0", "RO"], out.stdout + out.stderr
    with pytest.raises(StoreLockedError):
        SqliteExperienceStore(st.path)
    st.close()
    SqliteExperienceStore(st.path).close()               # released on close


WRITER = """
import sys, time
sys.path.insert(0, {src!r}); sys.path.insert(0, {tests!r})
from conftest import build_record
from cam_live.credit.replay import SqliteExperienceStore
from cam_live.contracts import OutcomeObservation, OutcomeStatus
st = SqliteExperienceStore({path!r}, blob_threshold=8192)
i = 0
while True:
    rec = build_record(i, text_pad=30000 if i % 5 == 0 else 10)
    st.append(rec)
    print("A", rec.exp_id, rec.content_hash(), flush=True)          # acknowledged only after append returned
    if i % 3 == 0:
        r = st.add_outcome(OutcomeObservation(rec.pending_queries[0].query_id, OutcomeStatus.OBSERVED, float(i),
                                              observed_at=rec.decision.ts + 50.0, dedup_key="k%d" % i))
        assert r.accepted
        print("O", rec.exp_id, i, flush=True)
    i += 1
"""


@pytest.mark.parametrize("n_ack", [40, 137])
def test_kill_9_loses_no_acknowledged_record(tmp_path, n_ack):
    path = str(tmp_path / "experience.sqlite")
    code = WRITER.format(src=SRC, tests=os.path.dirname(__file__), path=path)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    acked, outcomes = {}, {}
    try:
        while len(acked) < n_ack:
            line = proc.stdout.readline()
            assert line, proc.stderr.read()
            parts = line.split()
            if parts[0] == "A":
                acked[parts[1]] = parts[2]
            else:
                outcomes[parts[1]] = float(parts[2])
    finally:
        os.kill(proc.pid, signal.SIGKILL)                 # no chance to flush, close or checkpoint the WAL
        proc.wait()
    assert proc.returncode == -signal.SIGKILL
    st = SqliteExperienceStore(path, blob_threshold=8192)   # the dead writer's lock is gone with its process
    assert st.integrity_ok() and st.count() >= len(acked)
    for exp_id, digest in acked.items():
        rec = st.get(exp_id)
        assert rec is not None, f"acknowledged record {exp_id} was lost"
        if exp_id in outcomes:
            assert [o.value for o in rec.outcomes] == [outcomes[exp_id]]
            rec.outcomes = []
        assert rec.content_hash() == digest                 # including the large parts stored as blobs
    assert [r.seq for r in st.iter_records()] == list(range(1, st.count() + 1))
    st.append(__import__("conftest").build_record(10_000))  # and the store keeps working
    assert st.get("e10000").seq == st.count()


def test_windows_versions_predictions_and_experiments(tmp_path, make_record):
    st = _store(tmp_path)
    with st.batch():
        for i in range(12):
            st.append(make_record(i, model_version="m1" if i < 8 else "m2", experiment_id="x1" if i >= 10 else ""))
    assert st.assign_window([f"e{i}" for i in range(0, 6)], "w_support", kind="support") == 6
    assert st.assign_window([f"e{i}" for i in range(6, 12)], "w_query", kind="query") == 6
    assert [r.exp_id for r in st.window("w_query")] == [f"e{i}" for i in range(6, 12)]
    assert [(w["window_id"], w["n"], w["kind"]) for w in st.windows()] == [("w_support", 6, "support"),
                                                                             ("w_query", 6, "query")]
    v = {(d["kind"], d["version"]): d["n_records"] for d in st.versions()}
    assert v == {("model", "m1"): 8, ("model", "m2"): 4, ("reward", "r1"): 12}
    assert [d["first_seq"] for d in st.versions("model")] == [1, 9]
    q = make_record(0).pending_queries[0]
    p = PredictionRecord(prediction_id="p1", query=q, dist=Distribution("poisson", {"rate": 2.0}, mean=2.0), made_at=1.0,
                         decision_id="d0")
    st.put_prediction(p)
    assert st.predictions(query_id="q0", unresolved_only=True)[0].to_dict() == p.to_dict()
    p.resolved_status, p.resolved_value, p.nll = OutcomeStatus.OBSERVED, 3.0, 1.7
    st.put_prediction(p)
    assert st.predictions(unresolved_only=True) == [] and st.predictions(decision_id="d0")[0].nll == 1.7
    ex = ExperimentRecord(experiment_id="x1", created_at=5.0, question="q", hypothesis="h", arms=["a", "b"],
                          assignment="randomized")
    st.put_experiment(ex)
    assert st.experiment("x1").to_dict() == ex.to_dict() and len(st.experiments()) == 1
    assert [r.exp_id for r in st.experiment_records("x1")] == ["e10", "e11"]


def test_retrain_queue(tmp_path, make_record):
    st = _store(tmp_path)
    for i in range(4):
        st.append(make_record(i))
    assert st.mark_for_retraining(["e0", "e1"], "outcome_matured", outcome_key="k1", now=10.0) == 2
    assert st.mark_for_retraining(["e1", "e2"], "credit_revised", outcome_key="k2", now=11.0) == 2
    assert st.mark_for_retraining(["e0"], "outcome_matured", outcome_key="k1", now=12.0) == 0      # idempotent
    qu = st.retrain_queue()
    assert [d["exp_id"] for d in qu] == ["e0", "e1", "e2"] and qu[1]["reasons"] == ["outcome_matured", "credit_revised"]
    assert st.retrain_count() == 3 and st.pop_retrain(["e0", "e1"]) == 3 and st.retrain_count() == 1
