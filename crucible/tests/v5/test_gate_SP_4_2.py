"""Gate SP-4.2 (MASTER_SPEC Part 4.2): actual content artifacts of ten kinds can be created, edited, rendered,
published, updated, removed and inspected with sandboxed tools.

For each of the ten formats (text, HTML, image, audio, video, executable application, interactive page, PDF
document, educational product, software package) one creator runs the whole life cycle through tools only:

    create   a studio tool makes the artifact (bytes validated against the format)
    edit     a studio tool derives a modified version (provenance: the original is its parent)
    render   a studio tool renders it (a preview image; an app is executed and a package tested in the program
             sandbox: uid nobody, Landlock, seccomp, rlimits, no network)
    publish  content.publish through the platform that carries it; another participant opens it
    update   content.update to the edited version (the item keeps its history)
    remove   content.remove (gone from listings and search)
    inspect  content.inspect (creator analytics) and studio.probe (facts measured from the bytes)

artifact_kinds counts the formats whose seven steps all succeeded and were verified; content_tools the distinct
content tools that succeeded.
"""
from __future__ import annotations

import pytest

from crucible.acceptance.v5 import evidence
from social_helpers import Actor, social_world, assert_met

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_4_2.py"

APP_FILES = {"app.py": "import numpy as np\n\ndef main(x, caller):\n    a = np.array(x['values'], dtype=float)\n"
                       "    return {'mean': float(a.mean()), 'n': int(a.size), 'by': caller}\n"}
PKG_FILES = {"tidecalc/__init__.py": "", "tidecalc/core.py": "def range_m(high, low):\n    return high - low\n",
             "tests/test_core.py": "from tidecalc.core import range_m\n\ndef test_range():\n    assert range_m(5, 2) == 3\n",
             "README.md": "# tidecalc\nTide range helpers."}


@pytest.mark.timeout(900)
def test_SP_4_2_ten_artifact_kinds_full_life_cycle():
    w, P = social_world(seed=42)
    A, B = Actor(w, P["alice"]), Actor(w, P["bob"])
    tools_ok: set[str] = set()

    def ok(actor, tool, **args):
        out = actor.ok(tool, **args)
        tools_ok.add(tool)
        return out

    lessons = [{"title": "Reading a tide table", "body": "High water, low water, and the **rule of twelfths**."}]
    exercises = [{"question": "How many tides a day?", "choices": ["1", "2"], "answer": "2"}]
    img0 = ok(A, "studio.image", text="Tide chart", width=320)["ref"]
    snd0 = ok(A, "studio.audio", kind="tone", seconds=2.0, freq=330.0)["ref"]
    plan = {
        "text": (lambda: ok(A, "studio.write", format="markdown", title="Tides", body="The tide rises twice."),
                 lambda r: ok(A, "studio.edit", ref=r, replace=[{"find": "twice", "with": "twice a day"}]),
                 "posts", "post"),
        "html": (lambda: ok(A, "studio.write", format="html", title="Moorings", body="East quay, **berth 4**."),
                 lambda r: ok(A, "studio.edit", ref=r, append="<p>Water and power on the pontoon.</p>"),
                 "blog", "article"),
        "image": (lambda: ok(A, "studio.image", shapes=[{"shape": "ellipse", "xy": [40, 40, 200, 200],
                                                         "fill": "teal"}], width=256, height=256),
                  lambda r: ok(A, "studio.edit_image", ref=r, ops=[{"op": "grayscale"}, {"op": "border", "width": 4}]),
                  "pics", "image"),
        "audio": (lambda: ok(A, "studio.audio", kind="melody", notes=["E4", "G4", "B4", None, "E5"], bpm=120),
                  lambda r: ok(A, "studio.edit_audio", ref=r, effects=[{"effect": "echo"}, {"effect": "fade"}]),
                  "pods", "audio"),
        "video": (lambda: ok(A, "studio.video", images=[img0], audio=snd0, seconds_per_image=3),
                  lambda r: ok(A, "studio.edit_video", ref=r, start_s=0.5, duration_s=2.0, width=320),
                  "shorts", "video_short"),
        "app": (lambda: ok(A, "studio.app", name="tidemean", entry="app:main", files=APP_FILES,
                           description="mean of tide readings"),
                lambda r: ok(A, "studio.revise", ref=r, files={"README.md": "Averages readings."}),
                "dev", "app"),
        "page": (lambda: ok(A, "studio.page", title="Tide clock", html="<button id='b'>Now</button><p id='o'></p>",
                            script="document.getElementById('b').onclick=()=>{document.getElementById('o')"
                                   ".textContent=new Date().toISOString()}"),
                 lambda r: ok(A, "studio.edit", ref=r, append="<p>Times are local.</p>"),
                 "dev", "app"),
        "document": (lambda: ok(A, "studio.write", format="document", title="Harbour rules",
                                body="1. Keep to starboard.\n2. Five knots inside the wall."),
                     lambda r: ok(A, "studio.edit", ref=r, append="\n3. No wash."),
                     "blog", "article"),
        "course": (lambda: ok(A, "studio.course", title="Tides 101", lessons=lessons, exercises=exercises),
                   lambda r: ok(A, "studio.course", title="Tides 101", parent=r, exercises=exercises,
                                lessons=lessons + [{"title": "Springs and neaps", "body": "Moon and sun align."}]),
                   "kb", "article"),
        "package": (lambda: ok(A, "studio.package", name="tidecalc", version="1.0.0", files=PKG_FILES),
                    lambda r: ok(A, "studio.revise", ref=r, version="1.0.1",
                                 files={"tidecalc/core.py": "def range_m(high, low):\n    return abs(high - low)\n"}),
                    "dev", "package"),
    }
    done: dict[str, dict] = {}
    for fmt, (create, edit, platform, kind) in plan.items():
        steps = {}
        made = create()
        steps["create"] = made["format"] == fmt and made["facts"]["format"] == fmt
        ed = edit(made["ref"])
        info = ok(A, "studio.probe", ref=ed["ref"])
        steps["edit"] = ed["format"] == fmt and made["ref"] in info["parents"] and ed["ref"] != made["ref"]
        # render: a preview of every format; apps run and packages are tested in the program sandbox
        prev = ok(A, "studio.preview", ref=made["ref"], width=240)
        steps["render"] = prev["format"] == "image" and prev["facts"]["width"] <= 240
        if fmt == "app":
            run = ok(A, "studio.run", ref=ed["ref"], input={"values": [1.0, 2.0, 6.0]})
            steps["render"] &= run["ok"] and run["result"]["mean"] == 3.0 and run["result"]["by"] == P["alice"]
        if fmt == "package":
            t = ok(A, "studio.test", ref=made["ref"])
            steps["render"] &= bool(t["ok"])
        pub = ok(A, "content.publish", ref=made["ref"], kind=kind, platform=platform, title=f"{fmt} item",
                 tags=[fmt])
        cid = pub["id"]
        opened = B.ok("content.open", id=cid)
        steps["publish"] = opened["format"] == fmt and opened["ref"] == made["ref"]
        up = ok(A, "content.update", id=cid, ref=ed["ref"], title=f"{fmt} item, revised")
        rec = w.state.tables["content"][cid]
        steps["update"] = up["version"] == 2 and rec["ref"] == ed["ref"] and \
            [v["ref"] for v in rec["versions"]] == [made["ref"], ed["ref"]]
        ins = ok(A, "content.inspect", id=cid)
        probe = ok(A, "studio.probe", ref=ed["ref"])
        steps["inspect"] = ins["item"]["version"] == 2 and probe["facts"]["format"] == fmt and \
            "audience" in ins and B("content.inspect", id=cid).ok is False
        ok(A, "content.remove", id=cid, reason="life-cycle test")
        listed = B.ok("content.list", platform=platform, limit=50)
        steps["remove"] = w.state.tables["content"][cid]["removed"] and \
            all(it["id"] != cid for it in listed["items"]) and not w.state.has("search_docs", cid) and \
            B("content.open", id=cid).ok is False
        done[fmt] = steps
    ok(A, "studio.drafts", limit=200)
    ok(A, "studio.formats")
    ok(A, "content.kinds")
    assert w.check_invariants() == []
    kinds_ok = [f for f, s in done.items() if all(s.values())]
    metrics = {"artifact_kinds": len(kinds_ok), "content_tools": len(tools_ok), "formats_tested": len(done),
               "tools": ",".join(sorted(tools_ok))}
    for f, s in done.items():
        metrics[f"{f}_steps_ok"] = sum(1 for v in s.values() if v)
    evidence.record("SP-4.2", metrics, command=CMD, notes=(
        "Ten formats, seven life-cycle steps each, through the studio.* and content.* tools; edits keep provenance, "
        "renders are previews (apps executed and packages tested in the program sandbox), publication is checked "
        "by another participant opening the item, removal by listings, search and open."))
    assert_met('SP-4.2', metrics)
    assert sorted(kinds_ok) == sorted(plan), done
