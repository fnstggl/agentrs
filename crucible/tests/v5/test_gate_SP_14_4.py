"""SP-14.4 - World proposal interface validated for feasibility, diversity, safety, resources, difficulty and
usefulness; the learner cannot alter the evaluator or the sealed distribution (MASTER_SPEC Part 14.4; workstream N).

A *learner process* (a separate ``python -I`` process holding only the participant channel of the evaluator
process) proposes worlds and attacks the evaluator. Proposals are validated in the evaluator process
(`proposal.validate_proposal`, six measured checks); the verdicts it returns carry only pass/fail and public reasons.
Tamper attempts from the learner process, each of which must be blocked:

    relabel_partition      a proposal naming a split / partition / seed / family
    evaluator_ops          calling result, open_episode, declare_sealed, open_sealed, preregister, registry or log
                           on the participant channel
    pickle_payload         sending pickle bytes instead of JSON
    oversized              a proposal larger than the size limit
    code_in_grammar        executable text inside a grammar field
    duplicate              re-proposing an accepted structure
    local_registry_edit    relabelling the learner process's own rapid registry, then checking that the evaluator's
                           registry digest is unchanged
    salt_hunt              looking for the sealed salt in the learner's environment and home directory
After the attacks the evaluator's base registry digest equals the code's, its evaluation families are unchanged
and every accepted proposal is a training family.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap

import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import curriculum as CU
from crucible.research.v5 import distribution as D
from crucible.research.v5 import evaluator as EV
from crucible.research.v5 import proposal as P

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

LEARNER = textwrap.dedent("""
    import sys, os, json, socket
    sys.path.insert(0, sys.argv[1])
    fd = int(sys.argv[2])
    s = socket.socket(fileno=fd)
    rf, wf = s.makefile("rb"), s.makefile("wb")
    def call(op, *args):
        wf.write((json.dumps({"op": op, "args": list(args)}) + "\\n").encode()); wf.flush()
        return json.loads(rf.readline())
    def raw(b):
        wf.write(b); wf.flush()
        return json.loads(rf.readline())
    props = json.loads(sys.argv[3])
    out = {"verdicts": [], "attacks": {}}
    for p in props:
        out["verdicts"].append(call("propose", p)["value"])
    a = out["attacks"]
    v = call("propose", {"topology": "chain", "motifs": ["saturation"], "split": "val"})["value"]
    a["relabel_partition"] = not v["accepted"]
    a["evaluator_ops"] = all(not call(op, "x").get("ok") for op in
                             ("result", "open_episode", "declare_sealed", "open_sealed", "preregister", "registry", "log"))
    a["pickle_payload"] = not raw(b"\\x80\\x04\\x95\\x10\\x00\\x00\\x00\\x00\\x00\\x00\\x00evil.\\n").get("ok")
    big = {"topology": "chain", "motifs": ["saturation"], "pad": "x" * 9000}
    a["oversized"] = not call("propose", big)["value"]["accepted"]
    a["code_in_grammar"] = not call("propose", {"topology": "__import__('os').system('true')"})["value"]["accepted"]
    acc = [p for p, v in zip(props, out["verdicts"]) if v["accepted"]]
    a["duplicate"] = (not call("propose", acc[0])["value"]["accepted"]) if acc else False
    from crucible.rapid import families as RF
    ev = sorted(n for n, f in RF.FAMILIES.items() if f.split == "val")[0]
    RF.FAMILIES[ev].split = "train"
    a["local_registry_edit"] = True          # the evaluator's digest is checked by the harness afterwards
    home = os.path.expanduser("~")
    a["salt_hunt"] = "CRUCIBLE_V5_SEALED_DIR" not in os.environ and not any(
        "salt" in f for dp, dn, fs in os.walk(home) for f in fs)
    print(json.dumps(out))
""")


def test_SP14_4_world_proposals(tmp_path):
    import numpy as np
    rng = np.random.default_rng(14)
    props = [CU.targeted_structure(w, rng) for w in ("causal_discovery", "delayed_credit", "experimentation",
                                                     "adaptive_competitors", "skill_combination")]
    props += [CU.random_structure(rng) for _ in range(5)]
    ev = EV.Evaluator.start(sealed_dir=str(tmp_path / "sealed"))
    try:
        base = ev.must("registry")
        ev.must("declare_sealed", "final", [base["evaluation_families"][0]], 2, 1)
        p_child = socket.dup(ev.part.sock.fileno())
        env = EV.isolated_env()
        proc = subprocess.run([sys.executable, "-I", "-c", LEARNER, EV.src_root(), str(p_child), json.dumps(props)],
                              pass_fds=(p_child,), capture_output=True, env=env, cwd=env["HOME"], timeout=1800)
        os.close(p_child)
        out = json.loads(proc.stdout.decode().strip().splitlines()[-1]) if proc.returncode == 0 else {}
        after = ev.must("registry")
        status = ev.must("sealed_status")
    finally:
        ev.close()
    verdicts = out.get("verdicts", [])
    attacks = dict(out.get("attacks", {}))
    attacks["local_registry_edit"] = attacks.get("local_registry_edit") and after["base_digest"] == base["digest"] \
        and after["evaluation_families"] == base["evaluation_families"]
    code_digest = D.build_registry().digest
    intact = base["digest"] == code_digest and after["base_digest"] == code_digest and \
        status["sets"]["final"]["accesses"] == 0 and not status["sets"]["final"]["contaminated"]
    accepted = [v for v in verdicts if v["accepted"]]
    checks_run = sorted({c for v in verdicts for c in v["checks"]})
    blocked = sum(1 for v in attacks.values() if v)
    record("SP-14.4", {"proposal_checks": len(checks_run), "evaluator_tamper_attempts_blocked": blocked,
                       "tamper_attempts": len(attacks), "proposals": len(verdicts), "accepted": len(accepted),
                       "evaluator_and_sealed_intact": bool(intact)},
           notes=f"Verdicts (public): {[(v['structure_id'], v['accepted'], v['reasons'][:1]) for v in verdicts]}; "
                 f"attacks {attacks}; accepted proposals joined the training distribution: "
                 f"{after['accepted_proposals']}")
    assert len(checks_run) >= 6 and set(checks_run) == set(P.CHECKS)
    assert blocked == len(attacks) >= 8, attacks
    assert intact and accepted and len(accepted) < len(verdicts)
    assert all(not any(k in v for k in ("_entry", "refs", "oracle")) for v in verdicts)
