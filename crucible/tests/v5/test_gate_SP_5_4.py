"""SP-5.4: a fidelity record for every important behavioural mechanism, with the population, platform and period of
its evidence and its limits (owner workstream E).

The important mechanisms are computed at test time (`crucible.calibration.fidelity.important_mechanisms`): every
registered fitted model, every in-world mechanism a card says it grounds, every behavioural mechanism of the Crucible
4.0 register, every kernel module a v5 behaviour package registers, every population decision model, every parameter
group of the social consumption engine and every runtime decision model. Records come from
`docs/v5/FIDELITY_REGISTER.md` (every table of it) and the 4.0 register; a record covers exactly the mechanism whose
id it carries, and every claimed in-world use of a fitted model is verified."""
import os

from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.acceptance.v5.evidence import record
from crucible.calibration import fidelity as FI
from crucible.calibration import list_models


def test_SP_5_4_fidelity_record_for_every_important_mechanism():
    m = FI.sp54_metrics()
    metrics = {k: v for k, v in m.items() if not k.startswith("_")}
    record("SP-5.4", metrics, command="PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-5.4",
           artifacts=[FI.REGISTER_MD, os.path.join(os.path.dirname(FI.REGISTER_MD), "..", "..", "src", "crucible",
                                                   "evaluation", "fidelity.py")],
           notes=f"missing: {m['_missing']}; incomplete: {m['_incomplete']}; in-world claims: "
                 f"{ {k: v['ok'] for k, v in m['_claims'].items()} }")
    # every v5 row is well formed, ids are unique, and every claimed in-world use verifies
    assert not m["_row_problems"], m["_row_problems"]
    assert m["inworld_claims_unverified"] == 0, m["_claims"]
    assert m["empirically_fitted_rows"] >= 1
    # the register is read completely: every row of every table, including those after blank lines and headings
    n_lines = sum(1 for line in open(FI.REGISTER_MD, encoding="utf-8")
                  if line.strip().startswith("|") and not line.strip().startswith(("| id ", "|---")))
    assert m["v5_rows"] == n_lines
    # each fitted model and each population decision has its own record
    imp = {x["id"] for x in FI.important_mechanisms()}
    assert {f"model:{n}" for n in list_models()} <= imp
    assert {i for i in imp if i.startswith("pop.decision.")}, "population decisions are important mechanisms"
    crit = next(e for e in MF.load()["requirements"] if e["id"] == "SP-5.4")["criteria"]
    fails = [msg for c in crit for ok, msg in [criterion_met(c, metrics)] if not ok]
    assert not fails, (fails, m["_missing"], m["_incomplete"])


def test_SP_5_4_register_rejects_incomplete_records():
    good = {"id": "x", "mechanism": "x", "module": "`pop`", "category": "empirically fitted",
            "evidence": "kuairand_pure: Kuaishou featured feed, 27,285 users, 2022-04-22..05-08",
            "heldout": "log loss 0.44 vs 0.48", "limits": "one app, one month", "inworld": "none"}
    assert FI.record_problems(good) == []
    assert FI.record_problems(dict(good, category="calibrated"))            # not a v5 category
    assert FI.record_problems(dict(good, evidence="our own survey 2022"))    # no registered dataset
    assert FI.record_problems(dict(good, evidence="kuairand_pure"))          # no period
    assert FI.record_problems(dict(good, heldout=""))                        # empirical without held-out result
    assert FI.record_problems(dict(good, limits=""))
    assert FI.record_problems(dict(good, inworld=""))                        # in-world use must be stated
    assert FI.record_problems({"id": "y", "mechanism": "y", "module": "`pop`", "category": "plausible, unvalidated",
                               "evidence": "none", "heldout": "", "limits": "assumed", "inworld": "none"}) == []


def _register(tmp_path, name: str, body: str) -> str:
    p = tmp_path / name
    head = ("| id | mechanism | module | category | evidence (dataset, period) | held-out result | limits | "
            "in-world use |\n|---|---|---|---|---|---|---|---|\n")
    p.write_text(body.replace("@@HEAD@@", head), encoding="utf-8")
    return str(p)


def test_SP_5_4_parser_reads_every_table_and_reports_malformed_rows(tmp_path):
    row = "| {i} | m | `pop` | plausible, unvalidated | none | none | assumed | none |\n"
    p = _register(tmp_path, "a.md", "# R\n\n@@HEAD@@" + row.format(i="a") + "\n" + row.format(i="b") +
                  "\n## more\n\n@@HEAD@@" + row.format(i="c") + "| d | too | few |\n")
    rows = FI.v5_rows(p)
    assert [r["id"] for r in rows] == ["a", "b", "c", "d"]        # rows after blank lines and headings are read
    probs = FI.register_problems(rows)
    assert "d" in probs and "a" not in probs                       # a malformed row is reported, not dropped
    dup = FI.v5_rows(_register(tmp_path, "b.md", "@@HEAD@@" + row.format(i="a") + row.format(i="a")))
    assert any("duplicate" in x for x in FI.register_problems(dup)["a"])


def test_SP_5_4_false_inworld_claims_are_rejected(tmp_path):
    base = "| {i} | m | `feed` | {cat} | mind_small, Microsoft News, 2019-11-09..15 | ll 0.15 | limits | {inw} |\n"
    rows = FI.v5_rows(_register(tmp_path, "c.md", "@@HEAD@@" +
                                base.format(i="fit_none", cat="empirically fitted", inw="none") +
                                base.format(i="bad_module", cat="empirically fitted",
                                            inw="opt-in: `nosuchmodule` news_click@1") +
                                base.format(i="unreferenced", cat="empirically fitted",
                                            inw="opt-in: `feed` session_rate@1") +
                                base.format(i="default_no_probe", cat="empirically fitted",
                                            inw="default: `feed` news_click@1") +
                                base.format(i="ok", cat="empirically fitted", inw="opt-in: `feed` news_click@1")))
    probs = FI.register_problems(rows)
    # an in-world mechanism labelled fitted while nothing in the world uses the model
    assert any("in-world use: none" in x for x in probs["fit_none"])
    assert any("not registered" in x for x in probs["bad_module"])
    assert any("references session_rate@1" in x for x in probs["unreferenced"])   # the module never loads it
    assert any("probe" in x for x in probs["default_no_probe"])                   # default use needs a probe
    assert "ok" not in probs


def test_SP_5_4_one_module_row_does_not_cover_its_decisions():
    ids = {m["id"] for m in FI.important_mechanisms()}
    decisions = {i for i in ids if i.startswith("pop.decision.")}
    assert len(decisions) >= 9 and "module:pop" in ids
    # with only the module row, every decision of the module is missing
    rows = [r for r in FI.v5_rows() if r["id"] == "module:pop"]
    by_id = {r["id"] for r in rows}
    assert by_id == {"module:pop"} and not (decisions & by_id)
