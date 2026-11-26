"""crucible.civ.determinism and the exact cross-process checks built on it (Gate P.3, SP-10.0).

* A new process that loads a civilisation from a copy of its store taken at a snapshot and re-executes the live
  run's recorded continuation through the handlers (recorded external outcomes injected) reaches exactly the live
  world - the exact resume check of crucible.civ.longrun (`resume-regen`).
* A fresh process regenerating a recorded civilisation from genesis (`World.regenerate` with the store as blob
  source) reaches the recorded hash and state digest - the independent replication of crucible.civ.gates.
* `first_divergence` attributes a divergence to a recorded non-reproducible outcome only when that transaction's
  `ext` record differs, and to the world's logic otherwise.
"""
import json
import os

import pytest

import crucible

crucible.load_all_modules()

from crucible.civ import determinism as D, gates, generator, longrun  # noqa: E402
from crucible.storage.sqlite import SQLiteStore  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
H = 3600


def _ev(seq, typ, h, payload=None):
    return {"seq": seq, "type": typ, "hash": h, "payload": payload or {}}


def test_first_divergence_attributes_the_origin():
    base = [_ev(1, "proc", "a"), _ev(2, "pop.visit", "b"), _ev(3, "ext", "c", {"outcomes": [{"kind": "infer.x",
                                                                                         "out": 1}]})]
    same = [dict(e) for e in base]
    assert D.first_divergence(base, same) == {"equal": True, "first_seq": None, "origin": None, "len_a": 3,
                                              "len_b": 3}
    # the decision event differs and so does its transaction's recorded inference outcome
    inf = [base[0], _ev(2, "pop.visit", "B"), _ev(3, "ext", "C", {"outcomes": [{"kind": "infer.x", "out": 2}]})]
    d = D.first_divergence(base, inf)
    assert d["first_seq"] == 2 and d["origin"] == "recorded_outcome" and d["ext_kinds"] == ["infer.x"]
    # the decision differs although the recorded outcome is identical: the world's logic (or a fork) is the origin
    logic = [base[0], _ev(2, "pop.visit", "B"), _ev(3, "ext", "C", base[2]["payload"])]
    assert D.first_divergence(base, logic)["origin"] == "logic"
    # a transaction without recorded outcomes
    plain = [_ev(1, "proc", "a"), _ev(2, "pop.visit", "b")]
    assert D.first_divergence(plain, [plain[0], _ev(2, "pop.visit", "x")])["origin"] == "logic"


def test_exact_resume_and_regeneration_reproduce_a_civilisation(tmp_path):
    path = str(tmp_path / "live.sqlite")
    store = SQLiteStore(path)
    w = generator.generate("balanced", 3, store=store, world_id="det")
    w.snapshot()
    w.advance_time(H)
    w.snapshot()
    w.flush()
    cp = str(tmp_path / "copy.sqlite")
    longrun._backup(store, cp)
    at = w.seq
    w.advance_time(2 * H)
    w.flush()
    live = {"hash": w.last_hash, "digest": w.digest(), "seq": w.seq}
    assert live["seq"] > at
    rg = longrun._regen_resume_child(cp, "det", at, path, w.seq)
    assert rg["ok"], rg.get("error")
    assert (rg["hash"], rg["digest"], rg["seq"]) == (live["hash"], live["digest"], live["seq"]), (rg, live)
    assert rg["events"] == live["seq"] - at
    # from genesis, in a fresh process, from the store alone
    full = gates._regenerate_child(path, "det", w.seq)
    assert full["ok"], full.get("error")
    assert (full["hash"], full["digest"], full["seq"]) == (live["hash"], live["digest"], live["seq"]), (full, live)
    w.close()
    store.close()


def test_resume_compare_counts_the_exact_check(tmp_path):
    """`_resume_compare` on a real pending resume: the free child and the exact child both ran on the copy."""
    path = str(tmp_path / "live.sqlite")
    store = SQLiteStore(path)
    w = generator.generate("balanced", 3, store=store, world_id="rc")
    w.snapshot()
    w.advance_time(H)
    w.snapshot()
    w.flush()
    cp = str(tmp_path / "copy.sqlite")
    longrun._backup(store, cp)
    evo = cp + ".events.json"
    child = longrun._resume_child(cp, "rc", 2 * H, events_out=evo)
    assert child["ok"], child.get("error")
    pend = {"day": 1, "until": 2 * H, "child": child, "copy": cp, "events": evo, "at_seq": w.seq}
    w.advance_time(2 * H)
    out = longrun._resume_compare(w, store, path, "rc", pend, 2)
    assert out["regen"] and out["regen"]["ok"], out
    assert out["ok"], out
    if not out["free_ok"]:
        # a free resume may part from the live run only through a recorded non-reproducible outcome
        assert out["free_divergence"]["origin"] == "recorded_outcome", out["free_divergence"]
    assert not os.path.exists(cp) and not os.path.exists(evo)
    s = longrun.summarize({"daily": [], "resume": [out], "replay": {"segments": 0}})
    assert s["resume_checks"] == 1 and s["resume_mismatches"] == 0 and s["resume_exact_checks"] == 1
    json.dumps(out)
    w.close()
    store.close()
