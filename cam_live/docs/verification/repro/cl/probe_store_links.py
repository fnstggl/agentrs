"""G-cl probe: experience store (CL-01) and outcome linking (RC-01, RC-07).

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/cl/probe_store_links.py [--kills 12]
Every check prints [PASS] / [FAIL] (a FAIL is a refuted claim); exits 1 if any check fails. Work directory under
the session scratchpad (or $CL_WORK).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "cam_live/src")
sys.path.insert(0, "cam_live/tests/credit")
from conftest import build_record                                            # noqa: E402  (contract record builder)
from cam_live.contracts import (ExperienceItem, Observation, OutcomeObservation, OutcomeStatus, Provenance,  # noqa: E402
                                Quantity, SimClock)
from cam_live.credit.replay import SqliteExperienceStore, StoreLockedError, DEFAULT_STRATA  # noqa: E402
from cam_live.credit.links import OutcomeLinker                              # noqa: E402
from cam_live.credit.simple import credit_from_links                         # noqa: E402

WORK = os.environ.get("CL_WORK", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G_cl/work_store")
FAILS, OUT = [], {}


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def fresh(name):
    p = os.path.join(WORK, name)
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p)
    return os.path.join(p, "store.sqlite")


# ------------------------------------------------------------------------------ kill -9 child
CHILD = r'''
import sys, os
sys.path.insert(0, "cam_live/src"); sys.path.insert(0, "cam_live/tests/credit")
from conftest import build_record
from cam_live.credit.replay import SqliteExperienceStore
from cam_live.contracts import OutcomeObservation
path, start = sys.argv[1], int(sys.argv[2])
st = SqliteExperienceStore(path, blob_threshold=2048)
i = start
while True:
    r = build_record(i, text_pad=(3000 if i % 3 == 0 else 0))     # every third record goes to a blob
    st.append(r)
    os.write(1, f"A {r.exp_id}\n".encode())
    if i % 2 == 0:
        st.add_outcome(OutcomeObservation(query_id=f"q{i}", status="observed", value=float(i), observed_at=r.decision.ts + 20.0))
        os.write(1, f"O {r.exp_id}\n".encode())
    i += 1
'''


def kill9(n_kills: int):
    path = fresh("kill9")
    child = os.path.join(WORK, "kill9_child.py")
    open(child, "w").write(CHILD)
    acked, outs, start = [], [], 0
    rng = random.Random(0)
    for k in range(n_kills):
        p = subprocess.Popen([sys.executable, child, path, str(start)], stdout=subprocess.PIPE, cwd=os.getcwd())
        time.sleep(1.0 + rng.random() * 1.5)
        os.kill(p.pid, signal.SIGKILL)
        out = p.communicate()[0].decode().split("\n")
        a = [l.split()[1] for l in out if l.startswith("A ")]
        o = [l.split()[1] for l in out if l.startswith("O ")]
        acked += a
        outs += o
        if a:
            start = int(a[-1][1:]) + 1
        # the record being written at the kill may or may not be there; the next child continues after the last ack
        st = SqliteExperienceStore(path, readonly=True)
        head = st.head_seq()
        start = max(start, int(st.iter_records(since_seq=head - 1)[-1].exp_id[1:]) + 1 if head else start)
        st.close()
    st = SqliteExperienceStore(path)
    have = {r.exp_id for r in st.iter_records()}
    missing = [e for e in acked if e not in have]
    lost_out = [e for e in outs if not st.get(e).outcomes]
    ok_int = st.integrity_ok()
    blobs_ok = True
    try:
        for r in st.iter_records():
            pass
    except Exception as ex:                                               # noqa: BLE001
        blobs_ok = False
    seqs = [r.seq for r in st.iter_records()]
    st.close()
    OUT["kill9"] = {"kills": n_kills, "acknowledged_records": len(acked), "missing": len(missing),
                    "acknowledged_outcomes": len(outs), "lost_outcomes": len(lost_out), "integrity_ok": ok_int}
    check(f"kill -9 x{n_kills}: every acknowledged append survives", not missing and len(acked) > 50,
          f"{len(acked)} acked, {len(missing)} missing")
    check(f"kill -9 x{n_kills}: every acknowledged add_outcome survives", not lost_out, f"{len(outs)} acked, {len(lost_out)} lost")
    check("kill -9: integrity_check ok, blob-backed records decode", ok_int and blobs_ok)
    check("kill -9: seq strictly increasing, no duplicates", seqs == sorted(set(seqs)))


def single_writer():
    path = fresh("lock")
    a = SqliteExperienceStore(path)
    try:
        SqliteExperienceStore(path)
        check("second writer in the same process is refused", False)
    except StoreLockedError:
        check("second writer in the same process is refused", True)
    code = ("import sys; sys.path.insert(0,'cam_live/src'); from cam_live.credit.replay import SqliteExperienceStore, "
            "StoreLockedError\ntry:\n SqliteExperienceStore(sys.argv[1]); print('OPENED')\nexcept StoreLockedError: print('LOCKED')")
    out = subprocess.run([sys.executable, "-c", code, path], capture_output=True, text=True).stdout.strip()
    check("second writer in another process is refused", out == "LOCKED", out)
    r = SqliteExperienceStore(path, readonly=True)
    a.append(build_record(1))
    check("read-only reader sees the committed record while the writer is open", r.get("e1") is not None)
    try:
        r.append(build_record(2))
        check("read-only reader cannot write", False)
    except Exception:                                                     # noqa: BLE001
        check("read-only reader cannot write", True)
    # a second path to the same database file (symlinked directory) takes a different lock file
    link = os.path.join(WORK, "lock_alias")
    if os.path.lexists(link):
        os.unlink(link)
    os.symlink(os.path.dirname(path), link)
    try:
        b = SqliteExperienceStore(os.path.join(link, "store.sqlite"))
        b.append(build_record(3))
        both = a.get("e3") is not None
        b.close()
        check("a symlinked path to the same store is refused as a second writer", False,
              f"second writer opened through {link}; it wrote e3 (visible to the first writer: {both})")
    except StoreLockedError:
        check("a symlinked path to the same store is refused as a second writer", True)
    a.close(); r.close()


def immutability():
    path = fresh("immut")
    st = SqliteExperienceStore(path)
    st.put_item(ExperienceItem(ref="it1", kind="observation", ts=900.0, text="original"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.put_item(ExperienceItem(ref="it1", kind="observation", ts=2000.0, text="rewritten after the decision"))
    check("put_item does not replace a stored item", st.items(["it1"])[0].text == "original")
    rec = build_record(1)
    rec.context_item_refs = ["it1"]
    st.append(rec)
    # record-level history: can update() rewrite the original decision / its context / its provenance?
    r2 = st.get("e1")
    r2.decision.probabilities = [0.9, 0.05, 0.05]
    r2.decision.ts = 5.0
    st.put_item(ExperienceItem(ref="late", kind="observation", ts=5000.0, text="future"))
    r2.context_item_refs = ["it1", "late"]
    r2.provenance = "staging"
    try:
        st.update(r2)
        accepted = True
    except Exception as ex:                                               # noqa: BLE001
        accepted = False
    after = st.get("e1")
    rewritten = accepted and after.decision.probabilities == [0.9, 0.05, 0.05] and after.context_item_refs == ["it1", "late"]
    check("update() refuses to rewrite the stored decision / context / provenance of a record", not rewritten,
          f"accepted={accepted}; stored probabilities now {after.decision.probabilities}, ts {after.decision.ts}, "
          f"context {after.context_item_refs}, provenance {after.provenance}")
    # index columns vs body after such an update
    by_prov = st.sample_ids(5, loss="*", provenance="live", seed=0)
    body_prov = [st.get(e).provenance for e in by_prov]
    OUT["update_rewrite"] = {"accepted": accepted, "sample(provenance='live') returned": by_prov,
                             "their body provenance": body_prov}
    check("sample(provenance='live') never returns a record whose stored provenance is not 'live'",
          all(p == "live" for p in body_prov), f"ids {by_prov} have provenance {body_prov}")
    st.close()


def sampling():
    path = fresh("sampling")
    st = SqliteExperienceStore(path, durable=False, recent_window=100, recent_fraction=0.1)
    rng = random.Random(1)
    fams = ["f0"] * 600 + ["f1"] * 250 + ["f2"] * 100 + ["f3"] * 50     # imbalanced families
    rng.shuffle(fams)
    with st.batch():
        for i in range(1000):
            r = build_record(i, family=fams[i], provenance="live")
            st.append(r)
        for i in range(200):
            st.append(build_record(5000 + i, family="f9", provenance="diagnostic", prefix="x"))
    with st.batch():
        for i in range(1000):
            u = rng.random()
            if u < 0.2:
                st.set_success(f"e{i}", True)
            elif u < 0.35:
                st.set_success(f"e{i}", False)
            else:
                st.set_success(f"e{i}", None)
            if rng.random() < 0.1:
                st.add_outcome(OutcomeObservation(query_id=f"q{i}", status="observed", value=1.0,
                                                  observed_at=1000.0 + 10 * i + 5 + 500.0))    # delayed
            st.set_surprise(f"e{i}", rng.expovariate(1.0))
    labels = {f"e{i}": st.labels(f"e{i}") for i in range(1000)}
    n, B = 32, 3000
    dup, mixed = 0, 0
    origin_tot = {}
    frac = {"recent": [], "success": [], "failure": [], "delayed": []}
    recent_ids = {f"e{i}" for i in range(900, 1000)}
    newest = st._resolve_provenance(None)
    d_ids = st.sample_ids(n, loss="*", seed=0)
    check("default provenance (newest record = diagnostic) gives a pure diagnostic sample",
          newest == ("diagnostic",) and all(e.startswith("xe") for e in d_ids), f"{newest}")
    for s in range(B):
        ids = st.sample_ids(n, loss="*", seed=s, provenance="live")
        if len(set(ids)) != len(ids):
            dup += 1
        if any(e.startswith("xe") for e in ids):
            mixed += 1
        for k, v in st.last_sample_info["strata"].items():
            origin_tot[k] = origin_tot.get(k, 0) + v
        frac["recent"].append(sum(e in recent_ids for e in ids) / n)
        frac["success"].append(sum(labels[e]["success"] is True or labels[e]["success"] == 1 for e in ids) / n)
        frac["failure"].append(sum(labels[e]["success"] is False or labels[e]["success"] == 0 for e in ids) / n)
        frac["delayed"].append(sum(bool(labels[e]["delayed"]) for e in ids) / n)
    check(f"no repeated record inside a sample ({B} samples of {n})", dup == 0, f"{dup} samples with repeats")
    check("provenance='live' samples never contain diagnostic records", mixed == 0, f"{mixed} mixed samples")
    q = SqliteExperienceStore._quotas(n, DEFAULT_STRATA)
    share = {k: origin_tot.get(k, 0) / (B * n) for k in list(DEFAULT_STRATA) + ["fill"]}
    OUT["sampling_origin_share"] = share
    OUT["sampling_quota_share"] = {k: v / n for k, v in q.items()}
    worst = max(abs(share[k] - q[k] / n) for k in DEFAULT_STRATA)
    check("per-stratum draw counts equal the configured quotas", worst < 1e-9, f"max deviation {worst:.4f}; {share}")
    # batch composition by property
    base = {"recent": 100 / 1000, "success": sum(1 for l in labels.values() if l["success"] in (True, 1)) / 1000,
            "failure": sum(1 for l in labels.values() if l["success"] in (False, 0)) / 1000,
            "delayed": sum(1 for l in labels.values() if l["delayed"]) / 1000}
    comp = {k: float(np.mean(v)) for k, v in frac.items()}
    OUT["sampling_population_share"] = base
    OUT["sampling_batch_share"] = comp
    check("batches over-represent failures and delayed consequences relative to the population",
          comp["failure"] > base["failure"] and comp["delayed"] > base["delayed"], f"batch {comp} vs pool {base}")
    check("recent records are not more than recent quota + their share elsewhere (no recency collapse)",
          comp["recent"] < 0.30 + 0.1 + 0.05, f"{comp['recent']:.3f}")
    # within-stratum distributions: family stratum alone must be family-balanced; informative alone ~ priority
    fam_counts = {}
    for s in range(2000):
        for e in st.sample_ids(8, loss="*", strata={"family": 1.0}, seed=10_000 + s, provenance="live"):
            f = fams[int(e[1:])]
            fam_counts[f] = fam_counts.get(f, 0) + 1
    tot = sum(fam_counts.values())
    chi2 = sum((c - tot / 4) ** 2 / (tot / 4) for c in fam_counts.values())
    OUT["family_stratum_counts"] = fam_counts
    check("family stratum is balanced across imbalanced families (chi2, 3 dof, p>0.001)", chi2 < 16.27,
          f"counts {fam_counts}, chi2={chi2:.1f}")
    # informative stratum: single draws ~ uniform_floor/N + (1-floor) * (s + s0)^alpha / sum
    ix = st._index("*", ("live",))
    st.sample_ids(1, loss="*", strata={"informative": 1.0}, seed=0, provenance="live")
    p = ix.cache["p"]["informative"]
    inf = ix.cache["members"]["informative"]
    cnt = np.zeros(len(inf))
    pos = {int(j): k for k, j in enumerate(inf)}
    R = 60000
    id2i = {e: i for i, e in enumerate(ix.exp_id)}
    for s in range(R):
        e = st.sample_ids(1, loss="*", strata={"informative": 1.0}, seed=50_000 + s, provenance="live")[0]
        cnt[pos[id2i[e]]] += 1
    # compare in 10 bins of p
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    obs = np.array([cnt[b].sum() for b in bins]); exp = np.array([p[b].sum() * R for b in bins])
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    OUT["informative_bins_obs_over_exp"] = (obs / exp).round(3).tolist()
    check("informative stratum draws follow the configured priority (chi2 over 10 bins, p>0.001)", chi2 < 27.9,
          f"chi2={chi2:.1f}, obs/exp={OUT['informative_bins_obs_over_exp']}")
    # pure strata contain only their members
    for name, pred in (("success", lambda e: labels[e]["success"] in (True, 1)),
                       ("failure", lambda e: labels[e]["success"] in (False, 0)),
                       ("delayed", lambda e: bool(labels[e]["delayed"])), ("recent", lambda e: e in recent_ids)):
        ids = [e for s in range(50) for e in st.sample_ids(8, loss="*", strata={name: 1.0}, seed=s, provenance="live")]
        bad = sum(not pred(e) for e in ids)
        check(f"strata={{'{name}': 1}} returns only {name} records", bad == 0, f"{bad}/{len(ids)} not {name}")
    ids = st.sample_ids(64, loss="*", seed=1, provenance=["live", "diagnostic"])
    check("naming two provenances explicitly is the only way to mix", any(e.startswith("xe") for e in ids) and
          all(not e.startswith("xe") for e in st.sample_ids(64, loss="*", seed=1, provenance="live")))
    st.close()


def outcomes_and_links():
    path = fresh("links")
    clock = SimClock(10_000.0)
    st = SqliteExperienceStore(path)
    lk = OutcomeLinker(st, clock=clock)
    for i in range(6):
        st.append(build_record(i, ts=1000.0 + 100 * i, horizon=50.0))
    # duplicates and verified upgrade
    o = OutcomeObservation(query_id="q0", status="observed", value=3.0, observed_at=1080.0, dedup_key="ev-1")
    r1 = st.add_outcome(o)
    r2 = st.add_outcome(OutcomeObservation(query_id="q0", status="observed", value=3.0, observed_at=1080.0, dedup_key="ev-1"))
    r3 = st.add_outcome(OutcomeObservation(query_id="q0", status="observed", value=4.0, observed_at=1090.0, dedup_key="ev-2"))
    r4 = st.add_outcome(OutcomeObservation(query_id="q0", status="observed", value=5.0, observed_at=1095.0, verified=True,
                                           dedup_key="ev-1"))
    rec = st.get("e0")
    check("same event twice is de-duplicated", r1.accepted and r2.duplicate, f"{r2}")
    check("a second unverified answer to an answered query is dropped", not r3.accepted)
    check("a verified copy replaces the unverified one", r4.upgraded and len(rec.outcomes) == 1 and rec.outcomes[0].verified
          and rec.outcomes[0].value == 5.0, f"outcomes {[(x.value, x.verified) for x in rec.outcomes]}")
    r5 = st.add_outcome(OutcomeObservation(query_id="q1", status="observed", value=1.0, observed_at=10.0))
    check("an outcome dated before its query is refused", not r5.accepted, r5.reason)
    # a decision after the outcome never gets credit: event with attribution key naming a later decision
    ev = Observation(obs_id="o1", ts=1150.0, kind="event", text="joined",
                     data={"metric": "m", "value": 1.0, "by_decision": {"d0": 1, "d1": 1, "d3": 1}})
    rep = lk.reconcile(events=[ev])
    w = {l.exp_id: l.weight for l in rep.links}
    check("a decision made after the outcome is refused and its share stays unattributed",
          "e3" not in w and abs(sum(w.values()) - 2 / 3) < 1e-12, f"weights {w}, refused {rep.results[0].refused}")
    # ambiguous attribution keeps uncertainty: two decisions declared attribution windows on the same metric
    for i, t in ((20, 3000.0), (21, 3010.0)):
        r = build_record(i, ts=t, horizon=500.0)
        r.pending_queries[0].metric = "server.members"
        st.append(r)
    ev2 = Observation(obs_id="o2", ts=3200.0, kind="event", text="a member joined",
                      data={"metric": "server.members", "value": 1.0})
    rep2 = lk.reconcile(events=[ev2])
    us = [(l.exp_id, l.weight, l.uncertainty, l.evidence) for l in rep2.links]
    OUT["ambiguous_window_links"] = us
    check("ambiguous (window) evidence -> equal fractional weights, recency ignored",
          len(us) == 2 and all(abs(x[1] - 0.5) < 1e-12 for x in us), str(us))
    check("ambiguous attribution carries a non-zero uncertainty", all(x[2] > 0 for x in us), str(us))
    cv = credit_from_links(rep2.links)
    check("simple credit of an ambiguous outcome keeps its uncertainty", all(v.uncertainty > 0 for v in cv.values()),
          str({k: (v.value, v.uncertainty) for k, v in cv.items()}))
    # negative amounts in an attribution map: weights > 1 and fabricated credit
    ev3 = Observation(obs_id="o3", ts=1590.0, kind="event", text="net change",
                      data={"metric": "members", "value": 2.0, "by_action": {"a4_0": 3, "a5_0": -1}})
    rep3 = lk.reconcile(events=[ev3])
    ws = {l.exp_id: (l.weight, l.amount) for l in rep3.links}
    tot_w = sum(v[0] for v in ws.values())
    cred = credit_from_links(rep3.links)
    OUT["negative_amount_links"] = {"links": ws, "credit": {k: v.value for k, v in cred.items()}, "event_value": 2.0}
    check("the weights of one outcome never sum to more than 1 (negative attribution amounts)", tot_w <= 1 + 1e-12,
          f"links {ws}: total weight {tot_w}; credited {sum(v.value for v in cred.values())} for an event of value 2.0")
    # audit_leakage finds planted leaks
    st.put_item(ExperienceItem(ref="past", kind="observation", ts=900.0, text="p"))
    st.put_item(ExperienceItem(ref="future", kind="observation", ts=99999.0, text="f"))
    r = build_record(50, ts=2000.0)
    r.context_item_refs = ["past", "future"]
    st.append(r)
    r = build_record(51, ts=2100.0)
    r.context_item_refs = ["past"]
    st.append(r)
    leaks = lk.audit_leakage()
    check("audit_leakage reports exactly the planted leak", [x["exp_id"] for x in leaks] == ["e50"], json.dumps(leaks))
    # retraining queue: the linked decision is queued when its outcome matures
    rep6 = lk.reconcile(outcomes=[OutcomeObservation(query_id="q2", status="observed", value=7.0, observed_at=1260.0)])
    check("a matured declared outcome links to its decision and queues it", [l.exp_id for l in rep6.links] == ["e2"]
          and "e2" in [q["exp_id"] for q in st.retrain_queue()])
    # an outcome event with no evidence links to nothing (not the latest action)
    ev4 = Observation(obs_id="o4", ts=1700.0, kind="event", text="unknown cause", data={"value": 1.0})
    rep7 = lk.reconcile(events=[ev4])
    check("no evidence -> no link (never the latest action)", not rep7.links)
    st.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kills", type=int, default=12)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    parts = {"kill9": lambda: kill9(a.kills), "lock": single_writer, "immut": immutability, "sampling": sampling,
             "links": outcomes_and_links}
    for k, f in parts.items():
        if not a.only or k in a.only.split(","):
            f()
    print(json.dumps(OUT, indent=1, default=str))
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)


if __name__ == "__main__":
    main()
