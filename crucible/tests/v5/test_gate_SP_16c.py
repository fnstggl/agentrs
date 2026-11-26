"""SP-16c (workstream O): hidden-answer extraction tested as its own category; sandbox metering and adoption traces
measured outside the measured process (D-20); the 4.0 red-team ratchet stays part of Gate W.

Measured on the real system:

* hidden answers, four kinds, each an extraction attempt through what a participant really has:
  - Tier 1 holdouts: in the rule-induction world (observation-only) the participant queries every observation kind,
    submits programs (one of which reports the records its act() is given) and reads its observations; canaries are
    the hidden rule's parameters and the true values of the 50 holdout records (computed evaluator-side with the
    engine's own value function). Also: the hosted (out-of-process) learner's pipe cannot run code in the world, and
    the in-process environment object's graph (the open W.2 finding) - counted as an extraction while it is open;
  - capability test splits (`crucible.semantic.capbench`): not importable from world code (static reachability over
    every world root), not importable or readable by sandboxed participant code;
  - verification seeds (`caps.verify`): the inputs an independent suite generates from the world's fresh seeds do
    not appear in anything the provider could read before the verification, and a second verification draws new ones;
  - sealed sets (`crucible.research.sealed`): not importable from world code; the salt directory is not readable by
    sandboxed code;
* agent-reported metering: every metering / adoption-trace forgery probe of the 5.0 red team (counter rewrites,
  forged completions, work at import time, self-kill after delivering, background work, capacity, Tier 1 program
  cost, tracer writes, nondeterministic adoption) and a source check that the sandbox's in-process boot module counts
  nothing; a forgery that is billed or credited is an agent-reported figure trusted;
* the 4.0 red-team ratchet: tests/test_redteam.py run in a fresh interpreter.
"""
import json
import os
import subprocess
import sys

import pytest

from crucible.acceptance.v5.evidence import record

pytestmark = [pytest.mark.slow, pytest.mark.timeout(2400)]
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_16c.py"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")


def _canary(v):
    from crucible.redteam.leakage import distinctive
    return distinctive(float(v)) if isinstance(v, float) else distinctive(v)


def _hits(texts, canaries):
    from crucible.redteam.leakage import find_canaries
    out = {}
    for where, text in texts:
        for needle, label in find_canaries(text, canaries):
            out.setdefault(label, where)
    return out


# ---------------------------------------------------------------------------------------------- Tier 1 holdout
ECHO = ("def act(record, caller):\n    raise ValueError('RECORD ' + repr(sorted(record.items())))\n")
CONST = "def act(record, caller):\n    return 0.0\n"


def _tier1_holdout() -> dict:
    from crucible.experience import make
    env = make(1, "pe_rule_induction", 3, reward_mode="observation_only")
    try:
        sc = env.env._engine.sc                      # evaluator side: the scenario's hidden rule and holdout
        canaries = {}
        for k in ("theta", "p", "k1", "k2", "off", "slope"):
            c = _canary(sc.h.get(k))
            if c:
                canaries[c] = f"rule.{k}"
        for i, rec in enumerate(sc.h["holdout"]):
            c = _canary(float(sc.value(rec)))
            if c:
                canaries[c] = f"holdout[{i}].value"
        a = env.agents()[0]
        texts = []

        def take(where, x):
            texts.append((where, json.dumps(x, default=str, sort_keys=True)))
        take("tools", env.tools(a))
        for t in range(4):
            take(f"observe t{t}", env.observe(a))
            for target in ("examples", "history", "probe", "program_api", "spec", "tools"):
                args = {"kind": "query", "target": target}
                if target == "probe":
                    args.update({"re16": 3.3, "pe23": 1.1, "lo63": "ni49"})
                take(f"query {target}", env.execute(a, {"tool": "rapid.query", "args": args}))
            for src in (ECHO, CONST):
                take("program", env.execute(a, {"tool": "rapid.program", "args": {"kind": "program", "source": src}}))
            env.advance(3600)
        take("observe end", env.observe(a))
        hits = _hits(texts, canaries)
    finally:
        env.close()
    return {"canaries": len(canaries), "texts": len(texts), "bytes": sum(len(t) for _, t in texts), "hits": hits}


# ---------------------------------------------------------------------------------------------- verification seeds
def _verification_seeds() -> dict:
    import crucible
    crucible.load_all_modules()
    from crucible.core.kernel import World
    from crucible.innovation.reference import ServiceCreator, join
    from crucible.innovation.scenarios import SUITE_ROUTE
    from crucible.innovation.worlds import innovation_spec
    from crucible.redteam.v5.base import sweep_read_only
    from crucible.redteam.framework import Attacker
    w = World.create(innovation_spec(4, with_adopters=False))
    try:
        ana = ServiceCreator("ana", "route.shortest", "anaroute", tier=1, price=3)
        ana.act(w)
        assert ana.declaration, ana.log
        tess = join(w, "tess", 500_000)
        provider = "p:ana"
        att = Attacker(w, {provider}, set())
        before = sweep_read_only(att, provider, seeds=[ana.declaration, "anaroute"], rounds=1, max_calls=1500)
        ex = lambda who, tool, **a: w.execute(who, {"tool": tool, "args": a})  # noqa: E731
        assert ex(tess, "code.create_repo", name="suite", visibility="public", license="open").ok
        assert ex(tess, "code.commit", repo="tess/suite", files={"suite.py": SUITE_ROUTE}).ok
        n0 = len(w.log)
        v1 = ex(tess, "caps.verify", declaration=ana.declaration, suite_repo="tess/suite", cases=6)
        n1 = len(w.log)
        v2 = ex(tess, "caps.verify", declaration=ana.declaration, suite_repo="tess/suite", cases=6)
        assert v1.ok and v2.ok, (v1.error, v2.error)

        def generated(lo, hi):
            """The suite's generated inputs (recorded sandbox outcomes of suite:generate; check returns a number)."""
            out = []
            for e in list(w.log)[lo:hi]:
                if e.type == "ext":
                    for o in e.payload.get("outcomes") or []:
                        r = ((o or {}).get("out") or {}).get("result")
                        if (o or {}).get("kind") == "caps.suite" and isinstance(r, dict):
                            out.append(json.dumps(r, sort_keys=True))
            return out
        g1, g2 = generated(n0, n1), generated(n1, len(w.log))
        canaries = {}

        def walk(x, path):
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(x, list):
                if len(x) >= 4 and all(isinstance(v, (list, int, float)) for v in x):
                    c = json.dumps(x[:6])             # a fingerprint of the generated structure (e.g. graph edges)
                    if len(c) >= 24:
                        canaries.setdefault(c, path)
                for i, v in enumerate(x[:60]):
                    walk(v, f"{path}[{i}]")
            else:
                c = _canary(x)
                if c:
                    canaries.setdefault(c, path)
        for i, g in enumerate(g1):
            walk(json.loads(g), f"verification1.input{i}")
        hits = _hits(before, canaries)
        fresh = bool(g1) and bool(g2) and not (set(g1) & set(g2))
    finally:
        w.close()
    return {"generated_inputs": len(g1), "canaries": len(canaries), "texts_before": len(before),
            "hits": hits, "fresh_inputs_each_verification": fresh}


# ---------------------------------------------------------------------------------------------- sandbox reads
SANDBOX_READS = r'''
import os
def probe(args, caller):
    out = {}
    for name, path in args.items():
        try:
            if path.endswith(".py"):
                open(path).read(); out[name] = "read"
            else:
                os.listdir(path); out[name] = "listed"
        except BaseException as e:
            out[name] = "refused"
    for mod in ("crucible.semantic.capbench", "crucible.research.sealed", "crucible"):
        try:
            __import__(mod); out["import " + mod] = "imported"
        except BaseException:
            out["import " + mod] = "refused"
    return out
'''


def _sandbox_reads() -> dict:
    from crucible.research import sealed
    from crucible.sandbox.runner import run_program
    paths = {"capbench_tasks": os.path.join(SRC, "crucible", "semantic", "capbench", "tasks.py"),
             "sealed_module": os.path.join(SRC, "crucible", "research", "sealed.py"),
             "sealed_dir": sealed.sealed_dir()}
    r = run_program({"p.py": SANDBOX_READS}, "p:probe", paths)
    assert r["ok"], r.get("error")
    return r["result"]


# ---------------------------------------------------------------------------------------------- red team
METERING = ("v5.sandbox.meter_forgery.", "v5.sandbox.work_delivered_through_a_world_call_then_exit",
            "v5.sandbox.forged_completion", "v5.sandbox.thread_outlives_request", "v5.sandbox.c_builtins_are_billed",
            "v5.innovation.service_forges_its_compute_to_escape_capacity", "v5.rapid.program_forges_its_compute_cost",
            "v5.innovation.adoption_counter_written_by_dependent", "v5.innovation.adoption_by_shadow_file_under_deps",
            "v5.innovation.adoption_by_nondeterministic_output")
ISOLATION = ("v5.isolation.world_code_cannot_import_evaluator_side_code",
             "v5.isolation.remote_participant_cannot_run_code_in_the_world",
             "v5.isolation.in_process_learner_reads_hidden_grader")


@pytest.fixture(scope="module")
def red_team():
    from crucible.redteam.framework import sandbox_available
    ok, why = sandbox_available()
    assert ok, f"SP-16c needs the real sandbox: {why}"
    from crucible.redteam.v5 import probes, run
    ps = [p for p in probes() if p.name.startswith(METERING) or p.name in ISOLATION]
    rep = run(ps)
    return {r["name"]: r for r in rep["probes"]}


def _boot_counts_nothing() -> bool:
    src = open(os.path.join(SRC, "crucible", "sandbox", "boot.py"), encoding="utf-8").read()
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    return not any(w in code for w in ("_LINES", "sys.monitoring", "settrace", "setprofile", "work_lines",
                                        "process_time", "perf_counter", "getrusage", "cpu_ms"))


def _v4_ratchet() -> dict:
    env = dict(os.environ, PYTHONPATH=SRC, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_redteam.py"],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800, env=env)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]
    return {"returncode": p.returncode, "summary": tail}


def test_SP16c_hidden_answers_metering_and_the_v4_ratchet(red_team):
    t1 = _tier1_holdout()
    vs = _verification_seeds()
    sb = _sandbox_reads()
    static = red_team["v5.isolation.world_code_cannot_import_evaluator_side_code"]
    forbidden = set((static.get("evidence") or {}).get("forbidden_prefixes") or [])
    static_held = static["result"] == "held"
    remote = red_team["v5.isolation.remote_participant_cannot_run_code_in_the_world"]
    inproc = red_team["v5.isolation.in_process_learner_reads_hidden_grader"]
    kinds = {
        "tier1_holdout": {"measured": t1["canaries"] >= 20 and t1["texts"] >= 20 and remote["result"] in ("held", "leak"),
                          "extracted": len(t1["hits"]) + int(remote["result"] != "held") + int(inproc["result"] != "held")},
        "capability_test_splits": {"measured": "crucible.semantic.capbench" in forbidden,
                                   "extracted": int(not static_held) + int(sb["capbench_tasks"] != "refused")
                                   + int(sb["import crucible.semantic.capbench"] != "refused")},
        "verification_seeds": {"measured": vs["canaries"] >= 5 and vs["texts_before"] >= 20,
                               "extracted": len(vs["hits"]) + int(not vs["fresh_inputs_each_verification"])},
        "sealed_sets": {"measured": "crucible.research.sealed" in forbidden,
                        "extracted": int(not static_held) + int(sb["sealed_module"] != "refused")
                        + int(sb["sealed_dir"] != "refused") + int(sb["import crucible.research.sealed"] != "refused")},
    }
    meter_rows = {n: r for n, r in red_team.items() if n.startswith(METERING)}
    trusted = sorted(n for n, r in meter_rows.items() if r["result"] != "held")
    boot_ok = _boot_counts_nothing()
    v4 = _v4_ratchet()
    metrics = {
        "hidden_answer_probe_kinds": sum(1 for k in kinds.values() if k["measured"]),
        "hidden_answers_extracted": sum(k["extracted"] for k in kinds.values()),
        "agent_reported_metering_trusted": len(trusted) + int(not boot_ok),
        "v4_redteam_tests_passing": v4["returncode"] == 0,
        "tier1_holdout_canaries": t1["canaries"], "tier1_participant_texts": t1["texts"],
        "tier1_participant_bytes": t1["bytes"], "tier1_channel_hits": len(t1["hits"]),
        "tier1_in_process_object_graph_leak": inproc["result"] != "held",
        "tier1_remote_pipe_code_injection": remote["result"] != "held",
        "verification_input_canaries": vs["canaries"], "verification_fresh_inputs": vs["fresh_inputs_each_verification"],
        "metering_probes": len(meter_rows), "metering_probes_held": len(meter_rows) - len(trusted),
        "sandbox_boot_counts_nothing": boot_ok, "v4_redteam_summary": v4["summary"][:200],
    }
    record("SP-16c", metrics, command=CMD,
           notes="kinds: " + json.dumps({k: v for k, v in kinds.items()}) + "; metering findings: " + json.dumps(trusted)
           + "; sandbox reads: " + json.dumps(sb) + "; tier1 hits: " + json.dumps(t1["hits"])[:400])
    assert len(meter_rows) >= 12, sorted(meter_rows)
    assert all(r["result"] not in ("error", "not_applicable") for r in red_team.values()), \
        {n: r["result"] for n, r in red_team.items()}
    assert metrics["hidden_answer_probe_kinds"] >= 4, kinds
    assert metrics["agent_reported_metering_trusted"] == 0, trusted
    assert metrics["v4_redteam_tests_passing"] is True, v4
    assert metrics["hidden_answers_extracted"] == 0, kinds
