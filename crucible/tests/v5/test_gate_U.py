"""Gate U - Generalization research protocols (MASTER_SPEC Part 15, Gate U; workstream N).

Every number is measured on the v5 registry with observation-only worlds; learners learn from their own observed
consequences. Heavy parts run through the slot scheduler, e.g.

    python -m crucible.scale.slots --cpu 6 -- env CRUCIBLE_RESEARCH_JOBS=6 python -m pytest tests/v5/test_gate_U.py

The control battery (`instrument.controls_result`) is cached by source digest under $CRUCIBLE_CACHE/research_v5 so
U.3/U.4 and the SP-15.x gates read one confirmation run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import comparison as CMP
from crucible.research.v5 import curves as CV
from crucible.research.v5 import distribution as D
from crucible.research.v5 import evaluator as EV
from crucible.research.v5 import fresh as FR
from crucible.research.v5 import hstats as HS
from crucible.research.v5 import instrument as I
from crucible.research.v5 import learners as LR
from crucible.research.v5 import levels as LV
from crucible.research.v5.episodes import lever_entries, run_episode
from crucible.research.v5.sampler import build_env

pytestmark = [pytest.mark.slow, pytest.mark.timeout(14400)]


@pytest.fixture(scope="module")
def controls():
    return I.controls_result()


def test_U1_executable_levels_1_to_5():
    reg = D.build_registry()
    runs = {lv: LV.run_level(LV.build_level(lv, reg), seeds=(0, 1, 2), reg=reg) for lv in (1, 2, 3, 4, 5)}
    executed = [lv for lv, r in runs.items() if r["executed"]]
    why = {lv: r.get("why") or r.get("failure") for lv, r in runs.items() if not r["executed"]}
    gains = {lv: round(r["result"]["transfer_gain"]["estimate"], 4) for lv, r in runs.items()
             if r["executed"] and "transfer_gain" in r.get("result", {})}
    l5 = runs[5].get("result") or {}
    l5_gain = (l5.get("transfer_gain_standardised") or {}).get("estimate")
    record("U.1", {"executable_levels": len(executed), "levels_attempted": 5,
                   "level1_executed": 1 in executed, "level5_executed": 5 in executed,
                   "level5_scored_worlds": int(l5.get("scored_worlds", 0)),
                   "level5_test_tiers": len(l5.get("test_tiers") or [])},
           notes=f"Executed levels {executed}; crossed transfer gains (weights learner, 3 seeds) {gains}; Level 5 "
                 f"(reference learner trained on Tier 1, paired untrained copy): test tiers {l5.get('test_tiers')}, "
                 f"engines {l5.get('test_engines')}, standardised transfer gain {l5_gain}, world-changing actions "
                 f"{l5.get('world_changing_actions')}; not executed: {why}")
    for lv, r in runs.items():
        assert r["split"]["ok"], (lv, r["split"]["checks"])
    assert len(executed) >= 5, why


def _write_auth(path: str, base: str, expires: str) -> None:
    doc = {"acknowledgement": "I own or am explicitly authorised to test every base URL listed in this document.",
           "authorised_by": "gate U.2 <operator>", "experiment": "level6-interop", "purpose": "adapter interoperability",
           "base_urls": [base], "methods": ["GET"], "limits": {"max_requests": 20, "max_rate_per_s": 50.0,
                                                               "max_bytes": 100000, "max_wall_s": 120},
           "expires": expires, "approval_required": True, "allow_private_addresses": ["127.0.0.1"]}
    with open(path, "w") as fh:
        json.dump(doc, fh)
    os.chmod(path, 0o600)


def test_U2_level6_adapter_is_authorisation_gated(tmp_path):
    import datetime as dt
    refusals = {}
    for name, kw, env in (("default", {}, None), ("enable_only", {"enable": True}, None),
                          ("env_no_document", {"enable": True}, "1")):
        old = os.environ.pop("CRUCIBLE_LEVEL6", None)
        if env:
            os.environ["CRUCIBLE_LEVEL6"] = env
        try:
            LV.Level6Adapter(**kw)
            refusals[name] = False
        except LV.Level6Disabled:
            refusals[name] = True
        finally:
            os.environ.pop("CRUCIBLE_LEVEL6", None)
            if old is not None:
                os.environ["CRUCIBLE_LEVEL6"] = old
    # an expired document and a document without the acknowledgement are refused (in a fresh process: the bridge
    # refuses processes that host synthetic worlds)
    exp = (dt.datetime.utcnow() + dt.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    script = textwrap.dedent(f"""
        import sys, os, json, threading, http.server
        sys.path.insert(0, sys.argv[1])
        os.environ["CRUCIBLE_LEVEL6"] = "1"
        from crucible.research.v5.levels import Level6Adapter, Level6Disabled
        from crucible.realworld.authorization import AuthorizationError
        class H(http.server.BaseHTTPRequestHandler):
            n = 0
            def do_GET(self):
                H.n += 1
                body = json.dumps({{"count": H.n}}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *a): pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        out = {{}}
        tmp = sys.argv[2]
        base = f"http://127.0.0.1:{{port}}/api/"
        def doc(path, **over):
            d = {{"acknowledgement": "I own or am explicitly authorised to test every base URL listed in this document.",
                 "authorised_by": "gate U.2 <operator>", "experiment": "level6-interop", "purpose": "interop",
                 "base_urls": [base], "methods": ["GET"], "limits": {{"max_requests": 20, "max_rate_per_s": 50.0,
                 "max_bytes": 100000, "max_wall_s": 120}}, "expires": "{exp}", "approval_required": True,
                 "allow_private_addresses": ["127.0.0.1"]}}
            d.update(over)
            open(path, "w").write(json.dumps(d)); os.chmod(path, 0o600)
            return path
        for name, over in (("expired", {{"expires": "2020-01-01T00:00:00Z"}}), ("no_ack", {{"acknowledgement": "ok"}})):
            try:
                Level6Adapter(doc(os.path.join(tmp, name + ".json"), **over), log_path=os.path.join(tmp, name + ".log"),
                              enable=True)
                out[name] = False
            except (AuthorizationError, Level6Disabled, PermissionError):
                out[name] = True
        a = Level6Adapter(doc(os.path.join(tmp, "ok.json")), log_path=os.path.join(tmp, "ok.log"), enable=True)
        import time
        def pol(reader, k):
            time.sleep(0.1)                      # within the document's request rate
            return {{"method": "GET", "url": base + "ping"}} if k < 3 else None
        def meas(reader):
            time.sleep(0.1)
            return json.loads(reader.get(base + "count").body)["count"]
        rec = a.run(pol, meas, name="served-requests", steps=5)
        out["interop_value"] = rec["value"]
        out["claim"] = a.claim()["claim"]
        print(json.dumps(out))
    """)
    p = subprocess.run([sys.executable, "-I", "-c", script, EV.src_root(), str(tmp_path)], capture_output=True,
                       env=EV.isolated_env(), timeout=300)
    sub = json.loads(p.stdout.decode().strip().splitlines()[-1]) if p.returncode == 0 else {}
    disabled = refusals["default"] and refusals["enable_only"]
    requires = refusals["env_no_document"] and sub.get("expired") is True and sub.get("no_ack") is True
    methodology = os.path.exists(os.path.join(EV.src_root(), "..", LV.METHODOLOGY))
    record("U.2", {"level6_disabled_by_default": bool(disabled), "level6_requires_authorization": bool(requires),
                   "level6_interoperable_value": float(sub.get("interop_value") or -1),
                   "level6_claim_without_enough_experiments": sub.get("claim") is not None,
                   "methodology_documented": bool(methodology)},
           notes=f"Refusals in this process: {refusals}; fresh-process run: {sub or p.stderr.decode()[-500:]}. The "
                 "authorised run used a local test service on 127.0.0.1 listed in the document; no real-world "
                 "action was taken and no generalisation claim was produced.")
    assert disabled and requires and methodology
    assert sub.get("interop_value") == 4 and sub.get("claim") is None


def test_U3_fresh_process_weights_only(tmp_path):
    reg = D.build_registry()
    fams = I.default_design(reg).train_families
    worlds = [{"entry": f, "seed": 1_000_601 + i} for i, f in enumerate(fams)]
    rng = np.random.default_rng(3)
    out = {}
    for arch in ("weights", "stateful"):
        L = LR.make(arch, 0)
        for i in range(24):
            f = fams[i % len(fams)]
            for _ in range(3):
                run_episode(build_env(reg[f], 710_000 + i, {}, {"mode": "assigned"}), L, learn=True, rng=rng,
                            verify=False)
        ip = FR.in_process_eval(L, worlds)
        man = FR.save_checkpoint(L, str(tmp_path / arch))
        fo = FR.weights_only_eval(man, worlds)
        out[arch] = {"in_process": float(np.mean([s["score"] for s in ip["scores"]])),
                     "fresh": float(np.mean([s["score"] for s in fo["scores"]])), "checks": fo["checks"],
                     "ok": fo["ok"], "manifest": man}
    base = FR.weights_only_eval(FR.save_checkpoint(LR.make("weights", 0), str(tmp_path / "untrained")), worlds)
    untrained = float(np.mean([s["score"] for s in base["scores"]]))
    checks = out["weights"]["checks"]
    passed = [k for k in FR.CHECKS if checks.get(k)]
    adversarial = {}
    for kind in ("memory_in_params", "pickle", "arch_edit"):
        r = FR.weights_only_eval(FR.tampered_copy(out["weights"]["manifest"], kind, str(tmp_path / kind)), worlds[:1])
        adversarial[kind] = (not r["ok"]) and bool(r.get("refused"))
    st = out["stateful"]
    stash_detected = st["in_process"] - st["fresh"] > 0.02 and abs(st["fresh"] - untrained) < 1e-9
    record("U.3", {"fresh_process_weights_only_checks": len(passed), "checks_defined": len(FR.CHECKS),
                   "weights_gain_fresh": out["weights"]["fresh"] - untrained,
                   "stateful_in_process_gain": st["in_process"] - untrained, "stateful_fresh_gain": st["fresh"] - untrained,
                   "process_stash_detected": bool(stash_detected),
                   "adversarial_checkpoints_refused": sum(adversarial.values())},
           notes=f"Checks passed in the child: {passed}. Adversarial checkpoints: {adversarial}. A learner keeping "
                 "what it learned in process memory gains in process and not in the fresh weights-only process.")
    assert len(passed) >= 9 and all(checks.values())
    assert all(adversarial.values()) and stash_detected


def test_U4_positive_controls(controls):
    c = controls["controls"]
    kinds = {k: c[a]["check"]["correct"] for a, k in I.KIND_OF.items()}
    neg_ok = c["transfer_sham"]["check"]["correct"]
    labels = {a: c[a]["labels"] for a in c}
    record("U.4", {"positive_control_kinds_classified_correctly": sum(kinds.values()),
                   "positive_control_kinds": len(kinds), "known_negative_transfer_control_correct": bool(neg_ok),
                   "learner_seeds": len(controls["design"]["seeds"])},
           notes=f"Labels found: {labels}. Families (capabilities established on independent seeds): "
                 f"{controls['design']['train_families']}; transfer targets {controls['design']['transfer_families']}. "
                 f"Prespecified effects {controls['effects']}.")
    assert sum(kinds.values()) >= 5 and neg_ok, {a: c[a]["check"] for a in c}


def test_U5_matched_budget_modes():
    reg = D.build_registry()
    res, matched = {}, 0
    for mode, budget in (("experience", 1500), ("compute", 1.5)):
        r = CMP.compare(arms=("A", "B"), learner="meta", mode=mode, budget=budget, seeds=(0, 1, 2), registry=reg)
        res[mode] = r
        a, b = r["arms"]["A"], r["arms"]["B"]
        if mode == "experience":
            spent = [t["ticks"] for t in a["training"] + b["training"]]
            per_ep = max(max(t["ticks"] / t["episodes"] for t in a["training"] + b["training"]), 1)
            ok = min(spent) >= budget and max(spent) - budget <= per_ep
        else:
            spent = [t["cpu_s"] for t in a["training"] + b["training"]]
            ok = min(spent) >= budget and max(spent) <= budget * 1.5
        matched += int(bool(ok) and a["executed"] and b["executed"])
    d = res["experience"]["differences"].get("B-A", {})
    record("U.5", {"matched_budget_modes": matched, "modes_defined": len(CMP.BUDGET_MODES),
                   "experience_B_minus_A": float(d.get("estimate", 0.0)),
                   "compute_B_minus_A": float(res["compute"]["differences"].get("B-A", {}).get("estimate", 0.0))},
           notes=f"Arms A (original six 4.0 families) and B (v5 rapid families), identical meta learner, 3 seeds; "
                 f"experience budget 1500 ticks, compute budget 1.5 CPU-s; differences {res['experience']['differences']} "
                 f"/ {res['compute']['differences']}")
    assert matched >= 2


def test_U6_learning_curves():
    reg = D.build_registry()
    d = I.default_design(reg)
    rng = np.random.default_rng(6)
    T = LR.make("transfer", 3)
    for i in range(12):
        f = d.train_families[i % len(d.train_families)]
        for _ in range(4):
            run_episode(build_env(reg[f], 720_000 + i, {}, {"mode": "assigned"}), T, learn=True, rng=rng, verify=False)
    runs = []
    for f in d.transfer_families[:2]:
        T.clear_memory()
        runs.append(CV.practice_curve(reg, {"entry": f, "seed": 1_000_651}, T, rng=rng, q=d.q))
    pts = [p for p in runs[0]["curve"]]
    exact = [p["budget"] for p in pts if p["interactions_used"] == p["budget"]]
    record("U.6", {"learning_curve_points": len(pts), "points_at_exact_budget": len(exact),
                   "unfamiliar_tasks": len(runs), "score_at_0": float(pts[0]["score"]), "score_at_64":
                   float(next(p["score"] for p in pts if p["budget"] == 64))},
           notes=f"Curves (budget, score) on held-out families: "
                 f"{[[(p['budget'], round(p['score'], 3)) for p in r['curve']] for r in runs]}")
    assert [p["budget"] for p in pts][:8] == list(CV.SCHEDULE) and len(pts) >= 7
    assert len(exact) >= 7


def test_U7_independent_evaluator():
    ev = EV.Evaluator.start()
    try:
        ping = ev.must("ping")
        reg_local = D.build_registry()
        same_registry = ev.must("registry")["digest"] == reg_local.digest
        fam = lever_entries(reg_local, "train")[0]
        rng = np.random.default_rng(7)
        t = ev.open_episode(fam, 1_000_701)
        rec = EV.run_remote_episode(ev, t, LR.make("adapter", 0), rng)
        refusals = {}
        for op in ("result", "open_episode", "declare_sealed", "open_sealed", "preregister", "log", "registry"):
            refusals[op] = not ev.part.call(op, t).get("ok")
        refusals["pickle_bytes"] = not ev.part.raw(b"\x80\x04\x95garbage\n").get("ok")
        refusals["self_report"] = not ev.part.call("execute", t, "a0", {"tool": "report_success", "args": {}}).get("ok") \
            or not ev.part.call("execute", t, "a0", {"tool": "report_success", "args": {}})["value"].get("ok", True)
        again = ev.result(t)
        unchanged = again["value"] == rec["value"]
        independent = ping["pid"] != os.getpid() and ping["isolated"] and same_registry and all(refusals.values()) \
            and unchanged and rec["verifier"].startswith("ExperienceEnv.verify")
    finally:
        ev.close()
    record("U.7", {"independent_evaluator_used": bool(independent), "evaluator_pid_differs": ping["pid"] != os.getpid(),
                   "participant_refusals": sum(refusals.values()), "participant_probes": len(refusals),
                   "outcome_unchanged_by_learner_reports": bool(unchanged), "score": float(rec["score"])},
           notes=f"Outcome computed by the evaluator process from the world's verifier; the learner held only the "
                 f"participant channel. Refusals: {refusals}.")
    assert independent, refusals


def test_U8_statistics_account_for_worlds_and_seeds():
    reg = D.build_registry()
    d = I.default_design(reg)
    worlds = [(f, 1_000_801 + k) for f in d.train_families for k in range(3)]
    pool = []
    for s in range(12):                      # one learner, 12 independent learner seeds: a real null pool
        L = LR.make("weights", s)
        rng = np.random.default_rng(800 + s)
        for i in range(12):
            f = d.train_families[i % len(d.train_families)]
            run_episode(build_env(reg[f], 730_000 + i, {}, {"mode": "assigned"}), L, learn=True, rng=rng, verify=False)
        L.frozen = True
        from crucible.research.v5.episodes import score_episode
        pool.append([score_episode(reg, f, w, L, learn=False, rng=rng)["score"] for f, w in worlds])
    pool = np.array(pool)
    real = HS.null_calibration("crossed", trials=400, n_seeds=3, n_worlds=12, pool=pool, seed=1)
    real_naive = HS.null_calibration("naive", trials=400, n_seeds=3, n_worlds=12, pool=pool, seed=1)
    sim = HS.null_calibration("crossed", trials=600, n_seeds=3, n_worlds=12, seed=2, sd_seed=0.15)
    sim_naive = HS.null_calibration("naive", trials=600, n_seeds=3, n_worlds=12, seed=2, sd_seed=0.15)
    tol = 0.03
    ok = real["false_positive_rate"] <= 0.05 + tol and sim["false_positive_rate"] <= 0.05 + tol
    seed_sd = float(np.std(pool.mean(axis=1), ddof=1))
    record("U.8", {"intervals_account_for_worlds_and_seeds": bool(ok), "learner_seeds": 3,
                   "null_fpr_real_pool": real["false_positive_rate"], "null_fpr_real_pool_naive": real_naive["false_positive_rate"],
                   "null_fpr_simulated": sim["false_positive_rate"], "null_fpr_simulated_naive": sim_naive["false_positive_rate"],
                   "pool_seed_sd": seed_sd, "pool_seeds": int(pool.shape[0]), "pool_worlds": int(pool.shape[1])},
           notes="Null calibration of the crossed learner-seed x world interval: identical learners split into two "
                 "arms of 3 seeds on 12 worlds (400 random splits of a 12-seed real pool; 600 simulated nulls with "
                 "seed sd 0.15, the audit's case). The 4.0 analysis (seeds averaged, worlds bootstrapped) is shown "
                 "for comparison.")
    assert ok, (real, sim)
    assert sim_naive["false_positive_rate"] > 0.2
