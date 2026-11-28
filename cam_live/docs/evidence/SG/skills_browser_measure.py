"""SG: skills acquired from the agent's own *grounded* browser traces versus primitive-only operation, on the local
staging sites with the real structured browser and the real authorization service (spec 12A.17 comparison, CU-33).

This measures the procedural mechanism, not CAM's decisions: the primitive actions are issued here in the exact form
the agent's grounded candidates take (ref + obs_id + tab of the observation they were read from). Which action CAM
chooses is measured by the real-model runs (docs/SG.md §5).

    PYTHONPATH=cam_live/src .venv/bin/python cam_live/docs/evidence/SG/skills_browser_measure.py --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time

from cam_live.contracts import ActionSpec, ResultStatus, WallClock, new_id
from cam_live.harness.skills import SkillExecutor
from cam_live.validation import StagingComputer


class Bench:
    def __init__(self, root, key):
        self.sc = StagingComputer(os.path.join(root, "cu"), WallClock())
        self.task = self.sc.task(key, max_decisions=10_000)
        self.env, self.policy, _ = self.sc.open(self.task, self.sc.start_url(key))
        self.tools = {t.name: t for t in self.env.discover_tools()}
        self.skills = next(e for e in self.env._executors.values() if isinstance(e, SkillExecutor))
        self.decisions = 0
        self.rate_waits = 0

    def run(self, op, args):
        """One action; a refusal by the authorization service's rate limit (60 operations per 60 s per site) is
        waited out and the action issued again (the wait is not counted as latency; ``rate_waits`` counts it)."""
        t = self.tools[op]
        for _ in range(3):
            a = ActionSpec(action_id=new_id("act"), operation=op, modality=t.modality, level=t.level,
                           arguments=args, auth_scope=t.auth_scope)
            t0 = time.perf_counter()
            r = self.env.execute(a, self.policy.authorize(a, self.task, {}))
            dt = time.perf_counter() - t0
            m = re.search(r"rate limit .* retry in (\d+) s", r.failure_detail or "")
            if r.status == ResultStatus.DENIED and m:
                self.rate_waits += 1
                time.sleep(int(m.group(1)) + 1)
                continue
            break
        self.decisions += 1
        return a, r, dt

    @staticmethod
    def page(r):
        return next((o for o in reversed(r.observations) if o.data.get("elements")), None)

    def grounded(self, op, o, name, **extra):
        e = next(x for x in o.data["elements"] if x.get("name") == name)
        return self.run(op, dict({"ref": e["ref"], "obs_id": o.obs_id, "tab": o.data["tab"]}, **extra))


def store_primitive(b, name, label):
    """inspect, fill, select, click: one decision each (what the agent does without a skill)."""
    ids, ops, secs, ok = [], 0, 0.0, True
    a, r, s = b.run("browser.inspect", {})
    o = b.page(r)
    ops, secs = ops + len(r.ops), secs + s
    for op, el, extra in (("browser.fill", "Designation", {"text": name}),
                          ("browser.select_option", "Group", {"label": label}),
                          ("browser.click", "Store entry", {})):
        a, r, s = b.grounded(op, o, el, **extra)
        ids.append(a.action_id)
        ops, secs, ok = ops + len(r.ops), secs + s, ok and r.status == ResultStatus.OK
        o = b.page(r) or o
    return ids, {"decisions": 4, "primitive_ops": ops, "latency_s": secs, "ok": ok}


def record_primitive(b, name, cat):
    """The three-page wizard of the records application: navigate, fill, select, Continue, Continue, Confirm."""
    base = b.sc.sites.urls["structured"].rstrip("/")
    ids, ops, secs, ok = [], 0, 0.0, True
    a, r, s = b.run("browser.navigate", {"url": base + "/items/new"})
    ids.append(a.action_id)
    o = b.page(r)
    ops, secs = ops + len(r.ops), secs + s
    if o is None:                                   # the navigation's observation listed no controls: look again
        a, r, s = b.run("browser.inspect", {})
        o = b.page(r)
        ops, secs = ops + len(r.ops), secs + s
    if o is None:
        return ids, {"decisions": len(ids), "primitive_ops": ops, "latency_s": secs, "ok": False,
                     "detail": f"no controls observed: {r.status} {r.failure_detail[:120]}"}
    for op, el, extra in (("browser.fill", "Item name", {"text": name}),
                          ("browser.select_option", "Category", {"label": cat}),
                          ("browser.click", "Continue", {}), ("browser.click", "Continue", {}),
                          ("browser.click", "Confirm", {})):
        a, r, s = b.grounded(op, o, el, **extra)
        ids.append(a.action_id)
        ops, secs, ok = ops + len(r.ops), secs + s, ok and r.status == ResultStatus.OK
        o = b.page(r) or o
    return ids, {"decisions": 6, "primitive_ops": ops, "latency_s": secs, "ok": ok}


def summary(rows):
    return {"n": len(rows), "success": sum(1 for x in rows if x["ok"]),
            "decisions_per_task": statistics.mean(x["decisions"] for x in rows),
            "primitive_ops_per_task": statistics.mean(x["primitive_ops"] for x in rows),
            "latency_s_mean": statistics.mean(x["latency_s"] for x in rows),
            "latency_s_median": statistics.median(x["latency_s"] for x in rows)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(argv)
    root = tempfile.mkdtemp(prefix="sg_measure_")
    out = {"what": __doc__.strip().splitlines()[0], "n_per_arm": a.n, "tasks": {}}
    labels = ["Tools", "Parts", "Misc"]

    # ---- store an entry (entry book, layout A) ------------------------------------------------------------------
    b = Bench(root + "/s", "store_entry")
    prim = [store_primitive(b, f"Primitive {i}", labels[i % 3])[1] for i in range(a.n)]
    t1, _ = store_primitive(b, "Source one", "Parts")
    t2, _ = store_primitive(b, "Source two", "Misc")
    _, r, s = b.run("skill.induce", {"traces": [t1, t2]})
    sid = r.artifacts.get("skill_id")
    acq = {"ok": r.status == ResultStatus.OK, "latency_s": s, "primitive_ops": len(r.ops), "skill_id": sid,
           "source_decisions": 8, "detail": r.failure_detail[:300]}
    print("store_entry acquisition", acq, flush=True)
    if not sid:
        out["tasks"]["store_entry"] = {"primitive_only": summary(prim), "acquisition": acq}
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=str)
        return 1
    sk = []
    for i in range(a.n):
        _, r, s = b.run("skill.run", {"skill_id": sid, "arguments": {"text": f"Skill {i}", "label": labels[i % 3]}})
        sk.append({"decisions": 1, "primitive_ops": len(r.ops), "latency_s": s,
                   "ok": r.status == ResultStatus.OK and r.verified is True})
    entries = b.sc.sites.variants.entries
    names = [e["name"] for e in entries]
    stored_ok = all(f"Skill {i}" in names for i in range(a.n))
    # unfamiliar: the same form at a path the skill never saw; then a changed interface (layout B)
    base = b.sc.sites.urls["variants"].rstrip("/")
    b.run("browser.navigate", {"url": base + "/app/"})
    _, r, _ = b.run("skill.run", {"skill_id": sid, "arguments": {"text": "Unfamiliar path", "label": "Tools"}})
    app_path = {"ok": r.status == ResultStatus.OK, "termination": r.artifacts.get("termination")}
    b.sc.sites.variants.set_layout("b")
    changed = []
    for i in range(3):
        b.run("browser.navigate", {"url": base + "/app/"})
        _, r, _ = b.run("skill.run", {"skill_id": sid, "arguments": {"text": f"After change {i}", "label": "Tools"}})
        changed.append({"status": r.status, "termination": r.artifacts.get("termination"),
                        "skill_status": b.skills.library.get(sid).status})
    out["tasks"]["store_entry"] = {"primitive_only": summary(prim), "skill_based": summary(sk), "acquisition": acq,
                                   "every_skill_entry_stored": stored_ok,
                                   "unfamiliar_path_same_layout": app_path, "interface_changed_to_layout_b": changed,
                                   "skill": {"operations": b.skills.library.get(sid).operations(),
                                             "params": sorted(b.skills.library.get(sid).arg_schema["properties"]),
                                             "applicability_n": b.skills.library.get(sid).applicability_model().n},
                                   "rate_limit_waits": b.rate_waits}
    b.sc.close()

    # ---- record an item (three-page wizard) ---------------------------------------------------------------------
    b = Bench(root + "/r", "record_item")
    prim = [record_primitive(b, f"Primitive item {i}", labels[i % 3])[1] for i in range(a.n)]
    t1, _ = record_primitive(b, "Source item one", "Parts")
    t2, _ = record_primitive(b, "Source item two", "Misc")
    _, r, s = b.run("skill.induce", {"traces": [t1, t2]})
    sid = r.artifacts.get("skill_id")
    acq = {"ok": r.status == ResultStatus.OK, "latency_s": s, "primitive_ops": len(r.ops), "skill_id": sid,
           "source_decisions": 12, "detail": r.failure_detail[:300]}
    sk = []
    if sid:
        for i in range(a.n):
            _, r, s = b.run("skill.run", {"skill_id": sid, "arguments": {"text": f"Skill item {i}",
                                                                         "label": labels[i % 3]}})
            sk.append({"decisions": 1, "primitive_ops": len(r.ops), "latency_s": s,
                       "ok": r.status == ResultStatus.OK and r.verified is True, "detail": r.failure_detail[:120]})
    items = [i["name"] for i in b.sc.sites.structured.items]
    out["tasks"]["record_item"] = {"primitive_only": summary(prim), "skill_based": summary(sk) if sk else None,
                                   "acquisition": acq, "every_skill_item_recorded":
                                       bool(sk) and all(f"Skill item {i}" in items for i in range(a.n)),
                                   "skill": {"operations": b.skills.library.get(sid).operations(),
                                             "params": sorted(b.skills.library.get(sid).arg_schema["properties"])}
                                   if sid else None, "failures": [x for x in sk if not x["ok"]][:3],
                                   "rate_limit_waits": b.rate_waits}
    b.sc.close()
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
