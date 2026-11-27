"""Regenerate every number of workstream D's evidence from one *committed* tree (fix of G-cl m5).

    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/D/regenerate.py [--commit HEAD]
        [--only measured,diagnostics,real,status] [--work DIR]

The commit is exported with ``git archive`` into ``DIR/tree`` (so uncommitted edits of any workstream cannot enter)
and every experiment runs there, with ``CAM_LIVE_SOURCE_COMMIT`` set so each evidence file records the commit.

    measured     the default credit test suite with ``-s``: the JSON block each test prints under an upper-case tag
                 (EXACT_NKD, MC_UNBIASED <mdp>, KTD_VARIANCE, LONG_DELAY_SNR, COCOA_*, RUDDER_KEY, SELECTION,
                 REPLAY_BALANCE) -> measured.json; plus bench_store.py (machine-dependent throughput)
    diagnostics  ``python -m cam_live.credit.diagnose`` (all debug-scale experiments, one process, fixed seeds and
                 threads) -> diagnostics.json, written fresh (never merged with an older file)
    real         ``python -m cam_live.credit.diagnose --real`` (Qwen3-4B + 50m core, shared GPU lock) -> real_update.json
    status       ``docs/status/D.json["measured"]`` recomputed from the three files above and the JUnit results
                 (docs/evidence/test_results/D_fix*.xml); nothing in that block is typed by hand

Default: measured, diagnostics, status (``real`` needs the GPU and is run on its own).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # .../cam_live
REPO = os.path.dirname(PROJECT)                                           # the git worktree
PY = os.path.join(REPO, ".venv", "bin", "python")
STATUS = os.path.join(PROJECT, "docs", "status", "D.json")
RESULTS = os.path.join(PROJECT, "docs", "evidence", "test_results")
TAGS = {"EXACT_NKD": "exact_nkd", "KTD_VARIANCE": "ktd_variance", "LONG_DELAY_SNR": "long_delay_snr",
        "COCOA_KTD": "cocoa_ktd", "COCOA_LEARNED_BIAS": "cocoa_learned_bias", "COCOA_ELAPSED": "cocoa_elapsed",
        "RUDDER_KEY": "rudder_key", "SELECTION": "selection", "REPLAY_BALANCE": "replay_balance"}


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def export(commit: str, work: str) -> str:
    tree = os.path.join(work, "tree")
    shutil.rmtree(tree, ignore_errors=True)
    os.makedirs(tree)
    arch = subprocess.run(["git", "archive", commit, "cam_live"], cwd=REPO, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", tree], input=arch, check=True)
    return tree


def env_for(commit: str) -> dict:
    e = dict(os.environ)
    e["CAM_LIVE_SOURCE_COMMIT"] = commit
    e.setdefault("PYTHONHASHSEED", "0")
    return e


def run_measured(tree: str, commit: str, work: str) -> dict:
    t0 = time.time()
    p = subprocess.run([PY, "-m", "pytest", "cam_live/tests/credit", "-q", "--rootdir=cam_live", "-s", "-p",
                        "no:cacheprovider"], cwd=tree, env=env_for(commit), capture_output=True, text=True)
    out = {"_commands": {"tests": f"(git archive {commit}) .venv/bin/python -m pytest cam_live/tests/credit -q "
                                  f"--rootdir=cam_live -s", "note": "every block except store_benchmark is the JSON a "
                                  "test prints (tag in upper case) during that run",
                         "store_benchmark": ".venv/bin/python cam_live/docs/evidence/D/bench_store.py",
                         "regenerate": "cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/D/regenerate.py"},
           "_code": {"commit": commit, "source": f"git archive {commit}"}}
    for line in p.stdout.splitlines():
        m = re.search(r"\b(EXACT_NKD|MC_UNBIASED|KTD_VARIANCE|LONG_DELAY_SNR|COCOA_KTD|COCOA_LEARNED_BIAS|"
                      r"COCOA_ELAPSED|RUDDER_KEY|SELECTION|REPLAY_BALANCE)\s+(\S+\s+)?(\{.*\})\s*$", line)
        if not m:
            continue
        tag, extra, blob = m.group(1), (m.group(2) or "").strip(), m.group(3)
        key = f"mc_unbiased_{extra}" if tag == "MC_UNBIASED" else TAGS[tag]
        out[key] = json.loads(blob.replace("Infinity", "1e999").replace("NaN", "null"))
    summary = [l for l in p.stdout.splitlines() if re.search(r"\d+ (passed|failed)", l)]
    out["_tests"] = {"summary": summary[-1] if summary else "", "returncode": p.returncode,
                     "seconds": round(time.time() - t0, 1)}
    b = subprocess.run([PY, os.path.join(tree, "cam_live", "docs", "evidence", "D", "bench_store.py")], cwd=tree,
                       env=env_for(commit), capture_output=True, text=True)
    try:
        out["store_benchmark"] = json.loads(b.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        out["store_benchmark"] = {"error": b.stderr[-2000:]}
    return out


def run_diagnose(tree: str, commit: str, work: str, real: bool) -> dict:
    name = "real_update.json" if real else "diagnostics.json"
    tmp = os.path.join(work, name)
    if os.path.exists(tmp):
        os.remove(tmp)
    cmd = [PY, "-m", "cam_live.credit.diagnose", "--out", tmp, "--work", os.path.join(work, "diag")]
    if real:
        cmd.append("--real")
    env = env_for(commit)
    env["PYTHONPATH"] = os.path.join(tree, "cam_live", "src")
    subprocess.run(cmd, cwd=tree, env=env, check=True)
    return json.load(open(tmp))


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _junit(name: str) -> dict:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return {}
    r = ET.parse(path).getroot()
    s = r if r.tag == "testsuite" else r.find("testsuite")
    return {"tests": int(s.get("tests", 0)), "failures": int(s.get("failures", 0)), "errors": int(s.get("errors", 0)),
            "skipped": int(s.get("skipped", 0)), "seconds": float(s.get("time", 0.0)), "file": f"docs/evidence/test_results/{name}"}


def status_measured() -> dict:
    """D.json["measured"] from the evidence files (every value has a source path)."""
    ev = os.path.join(PROJECT, "docs", "evidence", "D")
    me = json.load(open(os.path.join(ev, "measured.json")))
    dg = json.load(open(os.path.join(ev, "diagnostics.json")))
    rl = json.load(open(os.path.join(ev, "real_update.json"))) if os.path.exists(os.path.join(ev, "real_update.json")) else {}
    m: dict = {"_source": "docs/evidence/D/regenerate.py status (from measured.json, diagnostics.json, real_update.json, "
                          "test_results/D_fix*.xml)",
               "_commits": {"measured.json": _get(me, "_code", "commit"), "diagnostics.json": _get(dg, "_code", "commit"),
                            "real_update.json": _get(rl, "_code", "commit")}}
    for suite, f in (("default", "D_fix.xml"), ("slow", "D_fix_slow.xml"), ("gpu", "D_fix_gpu.xml")):
        j = _junit(f)
        if j:
            m[f"tests_{suite}"] = j
    nkd = me.get("exact_nkd", {})
    unb = [v["rel_bias"] for k, v in nkd.items() if k not in ("hca_return", "return_conditioned") and "rel_bias" in v]
    m["exact_rel_bias_max_unbiased_estimators_enumerated"] = max(unb) if unb else None
    for k in ("reinforce", "cocoa", "rudder_immediate", "all_actions"):
        m[f"exact_rel_var_{k}"] = _get(nkd, k, "rel_var")
    m["exact_rel_bias_return_conditioned_hindsight"] = _get(nkd, "hca_return", "rel_bias") or \
        _get(nkd, "return_conditioned", "rel_bias")
    zs = [v.get("max_abs_z") for k in ("mc_unbiased_key_to_door", "mc_unbiased_random_tree") for v in me.get(k, {}).values()]
    es = [v.get("rel_error") for k in ("mc_unbiased_key_to_door", "mc_unbiased_random_tree") for v in me.get(k, {}).values()]
    m["mc_max_abs_z_all_estimators"] = max([z for z in zs if z is not None], default=None)
    m["mc_max_rel_error_all_estimators"] = max([e for e in es if e is not None], default=None)
    for k in ("reinforce", "cocoa", "rudder_immediate"):
        m[f"key_to_door_key_decision_var_{k}"] = _get(me, "ktd_variance", "var", k) or \
            _get(me, "ktd_variance", "var", k + "_exact")
    for T in ("16", "1024"):
        for k in ("reinforce", "cocoa"):
            m[f"long_delay_snr_{k}_T{T}"] = _get(me, "long_delay_snr", T, k)
    rk = me.get("rudder_key", {})
    m.update({"rudder_key_abs_err": rk.get("key_abs_err"), "rudder_other_steps_abs_mean": rk.get("other_abs_mean"),
              "rudder_key_is_argmax_fraction": rk.get("key_is_argmax")})
    ck, cb, ce = me.get("cocoa_ktd", {}), me.get("cocoa_learned_bias", {}), me.get("cocoa_elapsed", {})
    m.update({"cocoa_learned_w_weighted_abs_err_key_to_door": ck.get("weighted_abs_err"),
              "cocoa_learned_w_key_door": ck.get("w_key_door"), "cocoa_exact_w_key_door": ck.get("w_key_door_exact"),
              "cocoa_learned_gradient_rel_bias": cb.get("rel_bias"), "cocoa_w0_key_rel_bias": cb.get("key_rel_bias_w0"),
              "cocoa_hindsight_tv_with_elapsed": ce.get("tv_with_elapsed"),
              "cocoa_hindsight_tv_without_elapsed": ce.get("tv_without_elapsed")})
    sel = me.get("selection", {})
    m["selection_synthetic"] = {"validation": sel.get("validation"), "default_clear": sel.get("default_clear"),
                                "default_ambiguous": sel.get("default_ambiguous"),
                                "max_abs_err_vs_true_effect": sel.get("max_abs_err_vs_true_effect"),
                                "scores": sel.get("scores")}
    rb = me.get("replay_balance", {})
    m.update({"replay_fraction_recent_successes_in_batches": rb.get("recent_success"),
              "replay_fraction_failures": rb.get("failure"), "replay_fraction_delayed": rb.get("delayed"),
              "replay_old_records_covered_of_1800": rb.get("old_distinct")})
    m["store_benchmark"] = me.get("store_benchmark")
    # debug-scale diagnostics
    ln = dg.get("learning", {})
    nll = ln.get("heldout_prediction_nll", {})
    m["debug_heldout_nll"] = {k: nll.get(k) for k in ("before", "after", "change", "se", "n", "median_before",
                                                      "median_after", "share_improved")}
    m["debug_frequent_updates"] = _get(ln, "counters", "frequent")
    m["debug_total_weight_delta"] = ln.get("total_weight_delta")
    sa = ln.get("slow_adapters", {})
    m["debug_slow_updates"] = {"attempted": sa.get("n"), "applied": sa.get("applied"),
                               "reverted_by_guard": sa.get("reverted_by_guard")}
    cr = dg.get("credit", {})
    m["credit_experiments"] = len(cr.get("experiments", []))
    m["credit_holdout"] = cr.get("holdout")
    m["credit_max_conservation_error"] = _get(cr, "assignment", "max_conservation_error")
    m["credit_links_to_later_decision"] = cr.get("links_to_a_later_decision")
    m["credit_context_leaks"] = cr.get("context_leaks")
    m["credit_rmse_of_claiming_no_effect"] = cr.get("rmse_of_claiming_no_effect")
    m["credit_scores"] = {k: {x: v.get(x) for x in ("n", "rmse", "mean_sq_z", "sign_agreement", "passes", "n_refused",
                                                     "n_paired", "reasons")}
                          for k, v in _get(cr, "selection", "scores", default={}).items()}
    m["credit_selected_default"] = _get(cr, "selection", "default_after")
    m["credit_selection_reasons"] = _get(cr, "selection", "reasons")
    m["credit_first_decision_mean_credit"] = cr.get("first_decision_mean_credit")
    rt = dg.get("retention", {}).get("summary", {})
    m["retention"] = rt
    rs = dg.get("restart", {})
    m["restart"] = {k: {x: rs[k].get(x) for x in ("all_identical", "updates_compared", "final_hashes_identical")}
                    for k in ("kill_during_update", "kill_during_checkpoint") if k in rs}
    ex = dg.get("export", {})
    m["export"] = {"all_bit_exact": ex.get("all_bit_exact"), "max_abs_difference": ex.get("max_abs_difference"),
                   "export_bytes": ex.get("export_bytes")}
    m["diagnostics_seconds"] = {k: _get(dg, k, "_seconds") for k in ("learning", "credit", "retention", "restart", "export")}
    if rl:
        fr, sl = rl.get("frequent", {}), rl.get("slow_adapters", {})
        m["real"] = {"parameters": rl.get("parameters"), "load_seconds": rl.get("load_seconds"),
                     "gpu_gb_after_load": rl.get("gpu_gb_after_load"),
                     "frequent_grad_norms": fr.get("grad_norms"), "frequent_weight_delta": fr.get("weight_delta"),
                     "frequent_prediction_change": fr.get("prediction_change"),
                     "slow_grad_norms": sl.get("grad_norms"), "slow_weight_delta": sl.get("weight_delta"),
                     "slow_guard": _get(sl, "notes", "guard"), "slow_param_groups": sl.get("param_groups"),
                     "slow_adapter_grad_norm": _get(sl, "notes", "adapter_grad_norm"),
                     "slow_adapter_delta_norm": _get(sl, "notes", "adapter_delta_norm")}
    return json.loads(json.dumps(m, default=str).replace("NaN", "null"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--only", default="measured,diagnostics,status")
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    parts = [p for p in a.only.split(",") if p]
    commit = sh(["git", "rev-parse", a.commit], cwd=REPO).strip()
    work = os.path.abspath(a.work or os.path.join("/tmp", f"camlive_D_regen_{commit[:8]}"))
    os.makedirs(work, exist_ok=True)
    ev = os.path.join(PROJECT, "docs", "evidence", "D")
    tree = export(commit, work) if any(p in parts for p in ("measured", "diagnostics", "real")) else ""
    if "measured" in parts:
        me = run_measured(tree, commit, work)
        json.dump(me, open(os.path.join(ev, "measured.json"), "w"), indent=1, sort_keys=True)
        print("measured:", me.get("_tests"), flush=True)
    if "diagnostics" in parts:
        dg = run_diagnose(tree, commit, work, real=False)
        json.dump(dg, open(os.path.join(ev, "diagnostics.json"), "w"), indent=1, sort_keys=True, default=str)
        print("diagnostics:", dg.get("_code"), flush=True)
    if "real" in parts:
        rl = run_diagnose(tree, commit, work, real=True)
        json.dump(rl, open(os.path.join(ev, "real_update.json"), "w"), indent=1, sort_keys=True, default=str)
        print("real:", rl.get("_code"), flush=True)
    if "status" in parts:
        st = json.load(open(STATUS))
        st["measured"] = status_measured()
        st["measured_commands"] = {"regenerate_all": "cd /workspace/camlive && .venv/bin/python "
                                                     "cam_live/docs/evidence/D/regenerate.py",
                                   "real_configuration": "cd /workspace/camlive && .venv/bin/python "
                                                         "cam_live/docs/evidence/D/regenerate.py --only real,status"}
        json.dump(st, open(STATUS, "w"), indent=1, ensure_ascii=False)
        open(STATUS, "a").write("\n")
        print("status: measured block regenerated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
