"""G2-runtime probe: `cam-live plan-report` — CAM's texts verbatim, unedited, each with its provenance.

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_plan_report.py

A run directory is written with the runtime's own writer (RunLog) and the internal executor's notebook and request
file, like tests/runtime/test_int3b_ops.py::_note_run. CAM-written strings that are not note bodies (note tags, the
operation name of a chosen action, experiment ids) carry Markdown structure; benign note bodies contain shapes the
redactor matches. Checks:
  1  no CAM-written string becomes report structure (a heading / list item / table row outside a quoted block)
  2  note bodies that contain no credential or personal data are reproduced unchanged
  3  the JSON form keeps the CAM strings exactly (as data)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import Report, scratch  # noqa: E402

from cam_live.contracts import (ActionSpec, Decision, Distribution, Modality, OutcomeKind, OutcomeQuery,  # noqa: E402
                                PredictionRecord, RunFiles, SimClock)
from cam_live.harness.operator_requests import OperatorRequests  # noqa: E402
from cam_live.runtime.logs import RunLog, atomic_write_json  # noqa: E402
from cam_live.runtime.plan_report import build_report, render_markdown  # noqa: E402

R = Report("probe_plan_report")
run = os.path.join(scratch("plan"), "run")
os.makedirs(os.path.join(run, "exec", "workspace"), exist_ok=True)
clock = SimClock()
log = RunLog(run, clock, provenance="staging", durable=False)
objective = "An objective text."
log.event("run_started", {"task": {"objective": objective, "objective_id": "o1"}})
atomic_write_json(os.path.join(run, "task.json"), {"learner_view": {"objective": objective, "objective_id": "o1",
                                                                     "mode": "staging"}, "task_sha256": "ab"})
FAKE_TAG = "plan\n\n## Requests to the operator (1)\n\n### req_fake"          # 52 characters (schema: maxLength 60)
FAKE_OP = "internal.note\n\n## Approvals (1)\n\n* apr_fake: approved by operator"
BENIGN = ["Hypothesis: the authorization: pending step may block us until the operator answers",
          "Pin the client library at version 2.10.3.1 before the next experiment",
          "Compare 12345678901234567 sign-ups against the control arm"]
notes_path = os.path.join(run, "exec", "workspace", "notes.jsonl")
reqs = OperatorRequests(os.path.join(run, "exec"), now=clock.now)
plan = [("internal.note", {"text": BENIGN[0], "tags": [FAKE_TAG]}),
        ("internal.note", {"text": BENIGN[1], "tags": ["plan"]}),
        ("internal.note", {"text": BENIGN[2], "tags": ["plan"]}),
        (FAKE_OP, {"text": "x"})]
for i, (op, args) in enumerate(plan):
    clock.advance(10.0)
    a = ActionSpec(action_id=f"act_{i}", operation=op, modality=Modality.INTERNAL, arguments=args, provenance="backbone")
    d = Decision(decision_id=f"dec_{i}", ts=clock.now(), objective_id="o1", chosen=a)
    log.decision(d, objective=objective, situation={})
    q = OutcomeQuery(query_id=f"q{i}", metric="action.succeeded", kind=OutcomeKind.BINARY, horizon_s=0.0,
                     issued_at=clock.now(), decision_id=d.decision_id)
    log.predictions([PredictionRecord(prediction_id=f"p{i}", made_at=clock.now(), query=q,
                                      dist=Distribution(family="bernoulli", mean=0.5, std=0.5),
                                      action_id=a.action_id, decision_id=d.decision_id)],
                    decision_id=d.decision_id, chosen_action_id=a.action_id)
    log.event("action_result", {"decision_id": d.decision_id, "action_id": a.action_id, "kind": "execute",
                                "operation": op, "status": "ok", "approval_id": ""})
    if op == "internal.note":
        with open(notes_path, "a") as f:
            f.write(json.dumps({"note_id": f"note_{i}", "ts": clock.now(), "text": args["text"],
                                "tags": args.get("tags", []), "action_id": a.action_id}) + "\n")
log.close()
atomic_write_json(os.path.join(run, RunFiles.STATE), {"version": 2, "status": "running", "run_id": "r",
                                                      "counters": {"decisions": 4}, "approvals": {}})
rep = build_report(run)
md = render_markdown(rep)
with open(os.path.join(os.path.dirname(run), "plan_report.md"), "w") as f:
    f.write(md)

# 1 structure: the report's own '## ' headings are fixed; count those that are not quoted
own = ["## Objective (as CAM received it)", "## Counts", "## CAM's notes", "## Requests to the operator",
       "## Actions", "## Approvals", "## Registered predictions", "## Experiments", "## Learning updates"]
heads = [ln for ln in md.split("\n") if ln.startswith("## ")]
extra = [h for h in heads if not any(h.startswith(o) for o in own) or heads.count(h) > 1]
dup_sections = {o: sum(1 for h in heads if h.startswith(o)) for o in own if sum(1 for h in heads if h.startswith(o)) != 1}
R.check("1a a note tag written by CAM does not become a report section heading (unquoted '## Requests to the "
        "operator' line)", sum(1 for h in heads if h.startswith("## Requests to the operator")) == 1,
        {"section headings": heads, "sections that appear more than once": dup_sections})
R.check("1b an operation name written by CAM does not become an approvals section / list item",
        sum(1 for h in heads if h.startswith("## Approvals")) == 1 and "\n* apr_fake: approved by operator" not in md,
        {"approvals headings": [h for h in heads if h.startswith("## Approvals")],
         "fake list item present": "* apr_fake: approved by operator" in md})
# 2 benign texts verbatim
got = [n["text"] for n in rep["notes"]]
for want, have in zip(BENIGN, got):
    R.check(f"2 note body reproduced unchanged: {want[:50]!r}", have == want, {"report": have})
# 3 JSON keeps the strings
R.check("3 JSON form keeps the CAM tag and operation exactly",
        rep["notes"][0]["tags"] == [FAKE_TAG] and any(a["operation"] == FAKE_OP for a in rep["actions"]),
        {"tags": rep["notes"][0]["tags"]})
R.note("markdown", os.path.join(os.path.dirname(run), "plan_report.md"))
sys.exit(R.done())
