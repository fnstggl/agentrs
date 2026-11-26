"""Unified perception (spec 12A.5): exact quantities, provenance, untrusted trust, compact differences."""
import io
import json

from cam_live.contracts import Observation, PrimitiveOp, Quantity, ResultStatus, SimClock, Trust
from cam_live.harness.perception import (Perception, diff_page_states, digest, parse_number, quantities_from_table,
                                         _QBag)

P = Perception(SimClock(1000.0))


def q(obs):
    return {x.name: x for x in obs.quantities}


def test_parse_number_is_exact_and_keeps_units():
    cases = {"12": (12.0, ""), "1,234.50": (1234.5, ""), "-0.125": (-0.125, ""), "40%": (40.0, "%"),
             "211.930 credits": (211.93, "credits"), "$3.20": (3.2, "$"), "1.5k": (1500.0, ""), "2M": (2e6, ""),
             "7 kg": (7.0, "kg"), "5m": (5.0, "m"), "3e2": (300.0, ""), "−4": (-4.0, ""),
             "9007199254740993": (float(9007199254740993), "")}
    for text, (value, unit) in cases.items():
        got = parse_number(text)
        assert got is not None, text
        assert got[0] == value and got[1] == unit and got[2] == text, (text, got)
    for text in ("REF-0004", "J-101", "2026-11-26", "12 of 15", "", "abc", "1.2.3", True, None):
        assert parse_number(text) is None, text


def test_table_cells_become_named_quantities():
    bag = _QBag()
    quantities_from_table({"caption": "Items", "headers": ["Name", "Quantity", "Unit price", "Status"],
                           "rows": [["Alpha widget", "12", "3.50", "active"], ["Beta gasket", "240", "0.125", "active"]]},
                          "page.items", bag)
    got = {x.name: x.value for x in bag.items}
    assert got == {"page.items.items.alpha_widget.quantity": 12.0, "page.items.items.alpha_widget.unit_price": 3.5,
                   "page.items.items.beta_gasket.quantity": 240.0, "page.items.items.beta_gasket.unit_price": 0.125}
    assert bag.literal["page.items.items.alpha_widget.unit_price"] == "3.50"      # the literal text is kept too


def test_html_document_numbers_units_and_provenance():
    html = ("<html><head><title>Report</title><script>var x = 999</script></head><body><h1>Report</h1>"
            "<table><caption>Summary</caption><tr><th>Active items</th><td>3</td></tr>"
            "<tr><th>Total value</th><td>1,211.930 credits</td></tr><tr><th>Archived share</th><td>12.5%</td></tr>"
            "</table><p>Open tickets: 17</p></body></html>")
    o = P.document(html, uri="http://example.test/report.html")
    assert o.kind == "document" and o.data["format"] == "html" and o.data["title"] == "Report"
    got = q(o)
    assert got["doc.report_html.summary.active_items"].value == 3.0
    assert got["doc.report_html.summary.total_value"].value == 1211.93
    assert got["doc.report_html.summary.total_value"].unit == "credits"
    assert got["doc.report_html.summary.archived_share"].unit == "%"
    assert got["doc.report_html.open_tickets"].value == 17.0
    assert "999" not in o.text                                   # script content is not page text
    assert o.provenance.trust == Trust.UNTRUSTED and o.provenance.uri.endswith("report.html")
    assert len(o.provenance.content_sha256) == 64 and o.provenance.retrieved_at == 1000.0 and o.ts == 1000.0


def test_csv_json_text_documents():
    o = P.document("name,quantity,unit_price\nAlpha,12,3.50\nBeta,240,0.125\n", uri="/w/report.csv")
    assert o.data["format"] == "csv" and o.data["rows"] == 2 and o.data["headers"] == ["name", "quantity", "unit_price"]
    assert q(o)["doc.report_csv.t0.beta.unit_price"].value == 0.125
    o = P.document(json.dumps({"metrics": {"count": 5, "ratio": 0.3333333333333333, "ok": True, "tags": ["a"]},
                               "series": [1, 2.5]}), uri="/w/m.json")
    got = q(o)
    assert o.data["format"] == "json"
    assert got["doc.m_json.metrics.count"].value == 5.0 and got["doc.m_json.metrics.ratio"].value == 0.3333333333333333
    assert got["doc.m_json.series.1"].value == 2.5 and "doc.m_json.metrics.ok" not in got     # booleans are not numbers
    o = P.document("Latency: 12.5 ms\nStatus: ready\nRetries = 3", uri="/w/notes.txt")
    got = q(o)
    assert got["doc.notes_txt.latency"].value == 12.5 and got["doc.notes_txt.latency"].unit == "ms"
    assert got["doc.notes_txt.retries"].value == 3.0 and "doc.notes_txt.status" not in got


def _minimal_pdf(text):
    stream = f"BT /F1 18 Tf 40 100 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, o in enumerate(objs, 1):
        offs.append(out.tell())
        out.write(b"%d 0 obj\n" % i + o + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1))
    for off in offs:
        out.write(b"%010d 00000 n \n" % off)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref))
    return out.getvalue()


def test_pdf_text_layer():
    o = P.document(_minimal_pdf("Total units: 4200"), uri="/w/sheet.pdf")
    assert o.data["format"] == "pdf" and o.data["pages"] == 1
    assert "Total units: 4200" in o.text and q(o)["doc.sheet_pdf.total_units"].value == 4200.0


def test_image_dimensions_and_reference_without_a_model(tmp_path):
    from PIL import Image
    path = tmp_path / "shot.png"
    Image.new("RGB", (321, 123), (10, 20, 30)).save(path)
    o = P.image(str(path))
    assert o.kind == "image" and o.data["width"] == 321 and o.data["height"] == 123 and o.data["format"] == "PNG"
    assert q(o)["image.width"].value == 321.0 and o.refs["image"] == str(path)
    assert P.file(str(path)).kind == "image"
    assert P.image(b"not an image").data["error"]


def test_tool_results_traces_events_metrics():
    o = P.tool_result("sandbox.python", {"exit_code": 0, "stdout": "ok", "stats": {"rows": 1234567, "mean": 0.1}})
    got = q(o)
    assert got["tool.sandbox_python.stats.rows"].value == 1234567.0 and got["tool.sandbox_python.stats.mean"].value == 0.1
    assert o.kind == "tool_result" and o.provenance.trust == Trust.UNTRUSTED
    ops = [PrimitiveOp("browser.click", "structured_browser", {"ref": "e1"}, 10.0, 10.5),
           PrimitiveOp("browser.fill", "structured_browser", {"ref": "e2"}, 10.5, 11.0, ResultStatus.FAILED,
                       "stale_observation", note="gone")]
    t = P.trace(ops, action_id="act_1", status="failed")
    got = q(t)
    assert got["trace.ops"].value == 2 and got["trace.failed_ops"].value == 1 and got["trace.seconds"].value == 1.0
    assert "browser.fill" in t.text and "stale_observation" in t.text and t.parent_action_id == "act_1"
    e = P.event("job.finished", "job 7 finished", quantities=[Quantity("job.rows", 10)], data={"job": 7})
    assert e.kind == "event" and e.data["event"] == "job.finished" and e.quantities[0].value == 10
    m = P.metrics([Quantity("a.b", 3, "items")])
    assert m.kind == "metric" and "a.b=3 items" in m.text
    for o in (o, t, e, m, P.error("boom", failure_class="unknown")):
        assert isinstance(o, Observation) and o.provenance.trust == Trust.UNTRUSTED and o.provenance.imagined is False
        assert Observation.from_dict(json.loads(o.to_json())).to_dict() == o.to_dict()       # JSON round trip


def _state(text, elements, url="http://h/p", doc="d1", tables=()):
    return {"url": url, "title": "T", "doc": doc, "text": text, "elements": elements, "tables": list(tables), "tabs": []}


def test_page_observation_full_then_compact_diff():
    els = [{"ref": "e1", "role": "textbox", "name": "Name", "value": ""}, {"ref": "e2", "role": "button", "name": "Save"}]
    table = {"caption": "Figures", "headers": [], "rows": [["Reading A", "100"], ["Reading B", "2.50 units"]]}
    s1 = _state("Board\nStatus: loading", els, tables=[table])
    full = P.page(s1, None, mode="auto")
    assert full.data["full"] is True and '[e1] textbox "Name"' in full.text and "Status: loading" in full.text
    assert q(full)["page.p.figures.reading_b"].value == 2.5 and q(full)["page.p.figures.reading_b"].unit == "units"
    assert full.refs == {"e1": "textbox:Name", "e2": "button:Save"}
    els2 = [{"ref": "e1", "role": "textbox", "name": "Name", "value": "abc"}, {"ref": "e3", "role": "button", "name": "Undo"}]
    table2 = {"caption": "Figures", "headers": [], "rows": [["Reading A", "107"], ["Reading B", "2.50 units"]]}
    s2 = _state("Board\nStatus: ready", els2, tables=[table2])
    d = P.page(s2, s1, mode="auto", base_obs_id=full.obs_id)
    assert d.data["full"] is False and d.data["base_obs_id"] == full.obs_id
    diff = d.data["diff"]
    assert diff["text_added"] == ["Status: ready"] and diff["text_removed"] == ["Status: loading"]
    assert [e["ref"] for e in diff["elements_added"]] == ["e3"] and [e["ref"] for e in diff["elements_removed"]] == ["e2"]
    assert diff["elements_changed"] == [{"ref": "e1", "role": "textbox", "name": "Name", "field": "value",
                                         "before": "", "after": "abc"}]
    assert {"name": "page.p.figures.reading_a", "before": 100.0, "after": 107.0} in diff["quantities_changed"]
    assert [x.name for x in d.quantities] == ["page.p.figures.reading_a"]          # only what changed is re-reported
    assert "\nBoard" not in d.text and "Reading B" not in d.text          # unchanged content is not re-encoded
    same = P.page(s2, s2, mode="auto")
    assert same.data["diff"]["unchanged"] is True and "(no change)" in same.text and same.quantities == []
    assert diff_page_states(None, s1)["initial"] is True


def test_digest_is_bounded_and_says_what_was_omitted():
    long = "x" * 50000
    d = digest(long, 4000)
    assert len(d) <= 4000 + 60 and "characters omitted" in d
    o = P.document(long, uri="/w/big.txt")
    assert len(o.text) <= 4000 + 60 and o.data["text_len"] == 50000
