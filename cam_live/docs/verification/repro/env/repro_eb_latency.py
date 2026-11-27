"""G-env: re-run workstream EB's latency measurement WITHOUT overwriting docs/evidence/EB/latency.json.

    cd /workspace/camlive && .venv/bin/python cam_live/docs/verification/repro/env/repro_eb_latency.py [out.json]

Executes cam_live/docs/evidence/EB/measure_latency.py unchanged except for the output path, then prints the
re-measured values next to the values recorded in docs/status/EB.json.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CAM = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ORIG = os.path.join(CAM, "docs", "evidence", "EB", "measure_latency.py")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.mkdtemp(prefix="g_env_lat"), "latency.json")

src = open(ORIG).read()
needle = 'path = os.path.join(os.path.dirname(__file__), "latency.json")'
assert needle in src, "measure_latency.py changed: output line not found"
src = src.replace(needle, f"path = {out!r}")
g = {"__name__": "__main__", "__file__": ORIG}
exec(compile(src, ORIG, "exec"), g)

mine = json.load(open(out))
claimed = json.load(open(os.path.join(CAM, "docs", "status", "EB.json")))["measured"]
pairs = [("first_action_including_chromium_launch_ms", mine["first_action_including_chromium_launch_ms"]),
         ("navigate_median_ms", mine["navigate"][0]), ("click_with_navigation_median_ms", mine["click_with_navigation"][0]),
         ("fill_median_ms", mine["fill"][0]), ("inspect_full_with_a11y_median_ms", mine["inspect_full_with_a11y"][0]),
         ("inspect_diff_unchanged_median_ms", mine["inspect_diff_unchanged"][0]),
         ("wait_schedule_median_ms", mine["wait_schedule"][0]), ("rpc_ping_median_ms", mine["rpc_ping"][0]),
         ("rpc_inspect_full_median_ms", mine["rpc_inspect_full_with_a11y"][0]),
         ("observation_full_text_chars", mine["observation_chars"]["full_text"]),
         ("observation_diff_unchanged_text_chars", mine["observation_chars"]["diff_unchanged_text"]),
         ("observation_diff_after_fill_text_chars", mine["observation_chars"]["diff_after_fill_text"])]
print("\n== G-env reproduction (claimed in status/EB.json vs re-measured) ==")
for k, v in pairs:
    print(f"{k:45s} claimed={claimed.get(k)!s:>8}  remeasured={v}")
