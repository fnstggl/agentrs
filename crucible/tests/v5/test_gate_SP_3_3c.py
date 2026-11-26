"""SP-3.3c: semantic participants have persistent identity and state in the standard profile, bound to rows of the
shared population (MASTER_SPEC 3.1, 3.3).

A standard civilisation (`crucible.civ`, population "standard") runs a day and a half with its class-4 individuals
bound to the semantic participant runtime. We count participants whose mind-state row (`semp_state`) belongs to a
class-4 row of the population with that row's own principal; then snapshot the world, resume it from the snapshot
blob (a new kernel world, and a fresh process), and check that every participant's state (persona, goals, memory,
history, known identifiers, counters) comes back exactly and that the resumed participants continue from it: their
next activations add to the restored memory and history. Both continuations run the same half day; their digests
are compared (equal when the models give the same outputs on the same device).

    PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 6000 --ram-mb 24000 --timeout 10800 \
        --label "D: SP-3.3c" -- python -m pytest -q tests/v5/test_gate_SP_3_3c.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from crucible.acceptance.v5.evidence import record

pytestmark = [pytest.mark.slow, pytest.mark.timeout(10800)]

RESUME = r"""
import json, sys
import crucible
crucible.load_all_modules()
from crucible.core.kernel import World
from crucible.semantic import generalists as G
from crucible.storage.sqlite import SQLiteStore
# the snapshot blob plus the artifact bytes it references (policy weights, texts), exported into a store
w = World.from_blob(open(sys.argv[1], "rb").read(), world_id="sp33c-proc", store=SQLiteStore(sys.argv[2]))
before = json.dumps(G.members(w), sort_keys=True, default=str)
w.advance_time(w.now + 43200)
print(json.dumps({"before": before, "digest": w.digest(), "rows": G.members(w)}, default=str))
"""


def test_SP_3_3c_semantic_participants_persist_and_are_bound_to_population_rows():
    import crucible
    from crucible.core.kernel import World
    from crucible.semantic import generalists as G
    from crucible.semantic.generalists import verify
    crucible.load_all_modules()
    w = G.civ_world("balanced", 2, population="standard")
    w.advance_time(w.now + 36 * 3600)          # snapshot at noon of day 1: the continuation covers waking hours
    bound = verify.bound(w)
    rows0 = G.members(w)
    blob = w.snapshot_blob()
    r = World.from_blob(blob, world_id="sp33c-resumed")
    restored = G.members(r) == rows0
    w.advance_time(w.now + 43200)
    r.advance_time(r.now + 43200)
    rows_w, rows_r = G.members(w), G.members(r)
    def continued_from(a, b):
        """b is a's state continued: more activations, a's goals kept (by id), a's latest memory item still
        remembered (unless memory is at its bound) and a's history a prefix of b's recent history."""
        goals_kept = {g["id"] for g in a["goals"]} <= {g["id"] for g in b["goals"]} or len(b["goals"]) >= 24
        mem_kept = (not a["memory"]) or a["memory"][-1] in b["memory"] or len(b["memory"]) >= 40
        hist_kept = (not a["history"]) or a["history"][-1] in b["history"] or len(b["history"]) >= 80
        return int(b["counts"].get("activations", 0)) > int(a["counts"].get("activations", 0)) and goals_kept \
            and mem_kept and hist_kept
    continued = sum(1 for a, b in zip(rows0, rows_r) if continued_from(a, b))
    from crucible.storage.sqlite import SQLiteStore
    from crucible.substrate import artifacts
    tmp = tempfile.mkdtemp(prefix="sp33c-")
    path, db = os.path.join(tmp, "world.blob"), os.path.join(tmp, "artifacts.sqlite")
    with open(path, "wb") as fh:
        fh.write(blob)
    st = SQLiteStore(db)
    exported = artifacts.export(r, st)          # bytes a new process needs (they live in this process's pool)
    st.close()
    env = dict(os.environ)
    p = subprocess.run([sys.executable, "-c", RESUME, path, db], capture_output=True, text=True, env=env,
                       timeout=7200)
    shutil.rmtree(tmp, ignore_errors=True)
    assert p.returncode == 0, p.stderr[-3000:]
    proc = json.loads(p.stdout.strip().splitlines()[-1])
    m = {"semantic_participants_bound_to_population": bound["bound"], "class4_rows": bound["class4_rows"],
         "state_rows": bound["state_rows"], "state_restored_exactly": restored,
         "state_restored_in_new_process": proc["before"] == json.dumps(rows0, sort_keys=True, default=str),
         "artifacts_exported_for_new_process": int(exported.get("copied", 0)) + int(exported.get("present", 0)),
         "artifacts_missing_for_new_process": len(exported.get("missing", [])),
         "participants_continued_after_resume": continued,
         "resumed_digest_equal_in_process": r.digest() == w.digest(),
         "resumed_digest_equal_new_process": proc["digest"] == w.digest(),
         "activations_before_snapshot": sum(int(x["counts"].get("activations", 0)) for x in rows0),
         "activations_after": sum(int(x["counts"].get("activations", 0)) for x in rows_w)}
    record("SP-3.3c", m, command="PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_3_3c.py",
           notes="Standard population; snapshot after 36 h, resumed in-process and in a fresh process, 12 h each.")
    w.close()
    r.close()
    assert m["semantic_participants_bound_to_population"] >= 8
    assert m["state_restored_exactly"] and m["state_restored_in_new_process"]
    assert m["participants_continued_after_resume"] >= 8
