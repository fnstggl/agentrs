"""W2 real end-to-end check (redesign plan 2026-11-26 §3 W2): the real ``dom_dev`` CamUnifiedAgent (8 x 384 core,
Qwen3-0.6B, built by `stage1.system.build_system` from `config.make_preset("dom_dev")`) on a process-isolated ow_tiny
OpenWorld from `factory.make_env` (observation_only, result delivery on), driven by the lifetime harness for 12
decisions. Checks:

* the decision after ``world.hosts`` (decision 0 is offered only that tool, so the real decoder must call it) holds a
  TOOL_RESULT whose ``caused_by`` is that action's id - and every later decision holds its predecessor's result;
* observation_only: every reward the agent receives is 0.0 (`Transition.rewards is None` needs W1's canonical loop:
  reported, not asserted) and the agent's state (core Z and FastState) is bit-identical after every decision when the
  evaluator's payout is altered (payout x7 + 3.5 per clock stop in run B; the operator's legacy total differs);
* canaries planted in the task's private params, in the evaluator's objective state / status and in the label
  channel (a verified-metric name) never appear in what the agent received, in its archive / traces / decision logs
  or in the harness transcript; no evaluator-status key appears in the agent's inputs.

GPU: ~10 GB. Run as a script for the evidence JSON, under the shared lock (never run pytest under the lock):

    flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock python tests/cam_unified/envs/test_env_real_e2e.py \\
        --out docs/evidence/dom/W2/env_e2e_dom_dev.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import fields

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CRUCIBLE = os.path.abspath(os.path.join(HERE, "../../../../crucible/src"))
SRC = os.path.abspath(os.path.join(HERE, "../../../src"))
for p in (CRUCIBLE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

DAY = 86400.0
FIRST_TOOLS = ("world.hosts",)


# Declared deviation from the dom_dev preset: stage1.build_system (W1) builds a Consolidator over model.retriever,
# which is None with use_memory=False (AttributeError at seam-0), so memory stays on until W1's build_agent lands.
DEVIATIONS = {"use_memory": True}


def dom_dev_system(run_dir: str, seed: int = 0):
    from superlearner.cam_unified.config import make_config, make_preset
    from superlearner.cam_unified.stage1.system import SystemConfig, build_system
    pre, base = make_preset("dom_dev", **DEVIATIONS), make_config("dev")
    over = {f.name: getattr(pre, f.name) for f in fields(pre)
            if f.name != "scale" and getattr(pre, f.name) != getattr(base, f.name)}
    sc = SystemConfig(scale="dev", overrides=over, sem_model=pre.sem_model, device="cuda", seed=seed,
                      deterministic=True)
    return build_system(run_dir, sc), over


class Recorder:
    """The real agent, recorded: its act() inputs, its actions, the transitions it received, its state after every
    decision. Decision 0 is offered only `first_tools` (an environment choice, not an agent change)."""

    def __init__(self, inner, first_tools=FIRST_TOOLS):
        self.inner, self.first_tools = inner, tuple(first_tools)
        self.inputs, self.actions, self.transitions, self.states, self.act_s = [], [], [], [], []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def begin(self, env_ids, goals):
        self.goals = list(goals)
        self.inner.begin(env_ids, goals)

    def act(self, obs, tools):
        if not self.actions and self.first_tools:
            tools = [[t for t in ts if t.name in self.first_tools] for ts in tools]
        self.inputs.append((obs, tools))
        t = time.perf_counter()
        acts = self.inner.act(obs, tools)
        self.act_s.append(time.perf_counter() - t)
        self.actions.append(list(acts))
        return acts

    def observe(self, tr):
        self.transitions.append(tr)
        out = self.inner.observe(tr)
        self.states.append(snapshot(self.inner.agent.state))
        return out

    def update(self):
        return self.inner.update()


def snapshot(state) -> dict:
    import torch
    out = {"Z": state.Z.detach().cpu().clone()}
    fast = getattr(state, "fast", None)
    if fast is not None:
        for f in fields(fast):
            v = getattr(fast, f.name)
            out[f"fast.{f.name}"] = v.detach().cpu().clone() if torch.is_tensor(v) else json.dumps(v, default=str)
    return out


def same_state(a: dict, b: dict) -> bool:
    import torch
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if torch.is_tensor(x):
            if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
                return False
        elif x != y:
            return False
    return True


def _norm(text: str) -> str:
    import re
    text = re.sub(r"ow\d+-[0-9a-f]{10}", "ENV", text)
    text = re.sub(r"a_[0-9a-f]{32}", "ACT", text)
    return re.sub(r"g_[0-9a-f]{32}", "GOAL", text)


def one_run(root: str, tag: str, canaries: dict, payout_scale: float, payout_bias: float, n: int,
            update_every: int, seed: int = 0) -> dict:
    import torch
    from superlearner.cam_unified.envs import integrity as I
    from superlearner.cam_unified.envs.factory import make_env
    from superlearner.cam_unified.envs.harness import run_lifetime
    from superlearner.cam_unified.envs.labels import LabelStore
    from superlearner.cam_unified.envs.protocols.ow1 import open_economy_task
    from superlearner.cam_unified.stage1.harness_agent import HarnessAgent
    run_dir = os.path.join(root, tag)
    os.makedirs(run_dir, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    system, over = dom_dev_system(run_dir, seed)
    build_s = time.time() - t0
    rec = Recorder(HarnessAgent(system))
    task = open_economy_task(seed=100_000, days=1)
    task.params = {**task.params, "canary": canaries["task"]}
    task.private_note = canaries["task_attr"]
    task.horizon_s = int(n * 600)                     # the episode resolves (failure label) at the last decision
    es = make_env("openworld", tasks=[task], isolated=True, seed=seed, max_wait_s=600,
                  evaluator_kw={"objective_factory": I.canary_objective_factory(
                      {"state": canaries["state"], "status": canaries["status"], "label": canaries["label"]},
                      payout_scale=payout_scale, payout_bias=payout_bias)},
                  store_url=f"sqlite:///{os.path.join(root, tag + '_world.db')}")   # operator side: outside run_dir
    ad, rop, rev = es
    store = LabelStore(os.path.join(run_dir, "labels"))
    labels = []

    def on_step(k, tr):
        for lab in rev.take_labels():
            store.append(lab)
            labels.append(lab)
    t1 = time.time()
    try:
        res = run_lifetime(rec, ad, rev, max_decisions=n, think_s=600, update_every=update_every,
                           transcript=os.path.join(run_dir, "transcript.jsonl"),
                           model_version=lambda: rec.inner.model_version, on_step=on_step)
        details = rev.evaluate_goal(0).details
    finally:
        system.recorder.flush()
        system.archive.flush()
        es.close()
    loop_s = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    inputs_dump = I.actor_dump(*[x for pair in rec.inputs for x in pair], rec.goals)
    recs_dump = I.actor_dump(*[tr.records for tr in rec.transitions], *[tr.next_obs for tr in rec.transitions])
    delivered = []
    for k in range(len(rec.actions) - 1):
        a = rec.actions[k][0]
        obs = rec.inputs[k + 1][0].observations[0]
        hits = [o for o in obs if o.caused_by == a.action_id and o.kind.value in ("tool_result", "error")]
        delivered.append({"k": k, "tool": a.tool, "ok": bool(rec.transitions[k].records[0].result.ok),
                          "result_in_next_input": bool(hits), "kinds": sorted({o.kind.value for o in hits})})
    cans = list(canaries.values())
    out = {
        "tag": tag, "payout_scale": payout_scale, "payout_bias": payout_bias, "decisions": res.steps,
        "updates": len(res.updates), "build_s": round(build_s, 2), "loop_s": round(loop_s, 2),
        "act_s": [round(x, 3) for x in rec.act_s], "gpu_peak_gb": round(peak, 2),
        "actions": [[a.tool, a.args] for acts in rec.actions for a in acts if a is not None],
        "first_tool": rec.actions[0][0].tool if rec.actions else None,
        "delivered": delivered,
        "rewards_seen": sorted({float(r) for tr in rec.transitions for r in tr.rewards}),
        "transition_rewards_is_none": all(tr.rewards is None for tr in rec.transitions),
        "harness_returns": res.returns, "operator_legacy_total": details.get("return_total"),
        "labels": [{"status": lab.status, "kind": lab.kind, "t_resolved_days": lab.t_resolved_days,
                    "label_canary_in_label": canaries["label"] in json.dumps(lab.meta)} for lab in labels],
        "canary_hits_inputs": I.canary_hits(inputs_dump, cans),
        "canary_hits_records": I.canary_hits(recs_dump, cans),
        "canary_scan_run_dir": I.scan_run_dir(run_dir, cans),
        "canary_positive_controls": {
            "status_in_evaluator_details": details.get("canary") == canaries["status"],
            "label_in_label_store": I.scan_run_dir(run_dir, [canaries["label"]], exclude=())["hits"] != {}},
        "privileged_keys_in_inputs": I.privileged_keys([x for pair in rec.inputs for x in pair] + [rec.goals]),
        "core_hash_after": _core_hash(system.model),
        "dom_dev_overrides": sorted(over),
        "_states": rec.states, "_inputs": _norm(inputs_dump), "_recs": _norm(recs_dump),
    }
    del system, rec
    torch.cuda.empty_cache()
    return out


def _core_hash(model) -> str:
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(model.core.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def run_check(root: str, n: int = 12, update_every: int = 6) -> dict:
    from superlearner.cam_unified.envs import integrity as I
    canaries = {k: I.new_canary(k) for k in ("task", "task_attr", "state", "status", "label")}
    a = one_run(root, "A_payout_x1", canaries, 1.0, 0.0, n, update_every)
    b = one_run(root, "B_payout_x7_plus3.5", canaries, 7.0, 3.5, n, update_every)
    same = [same_state(x, y) for x, y in zip(a["_states"], b["_states"])]
    rep = {"runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in (a, b)],
           "state_identical_per_decision": same,
           "inputs_identical": a["_inputs"] == b["_inputs"], "records_identical": a["_recs"] == b["_recs"],
           "actions_identical": a["actions"] == b["actions"],
           "core_hash_identical": a["core_hash_after"] == b["core_hash_after"],
           "legacy_payout_differs": a["operator_legacy_total"] != b["operator_legacy_total"]}
    if not all(same):                                    # a determinism control tells invariance from GPU noise
        c = one_run(root, "C_payout_x1_repeat", canaries, 1.0, 0.0, n, update_every)
        rep["determinism_control_identical_per_decision"] = [same_state(x, y)
                                                             for x, y in zip(a["_states"], c["_states"])]
        rep["runs"].append({k: v for k, v in c.items() if not k.startswith("_")})
    r0 = rep["runs"][0]
    rep["checks"] = {
        "decisions_12": all(r["decisions"] == n for r in rep["runs"]),
        "first_decision_world_hosts": r0["first_tool"] == "world.hosts",
        "world_hosts_result_in_next_input": bool(r0["delivered"]) and r0["delivered"][0]["result_in_next_input"],
        "every_result_in_next_input": all(d["result_in_next_input"] for r in rep["runs"] for d in r["delivered"]),
        "rewards_all_zero": all(r["rewards_seen"] == [0.0] for r in rep["runs"]),
        "state_bit_identical_under_altered_payout": all(same) and len(same) == n,
        "inputs_identical_under_altered_payout": rep["inputs_identical"],
        "payout_really_altered": rep["legacy_payout_differs"],
        "no_canary_in_actor_inputs_records": all(not r["canary_hits_inputs"] and not r["canary_hits_records"]
                                                 for r in rep["runs"]),
        "no_canary_in_run_dir": all(not r["canary_scan_run_dir"]["hits"] for r in rep["runs"]),
        "canary_positive_controls": all(all(r["canary_positive_controls"].values()) for r in rep["runs"]),
        "no_privileged_keys_in_inputs": all(not r["privileged_keys_in_inputs"] for r in rep["runs"]),
        "label_emitted": all(len(r["labels"]) == 1 for r in rep["runs"]),
    }
    rep["preset"] = {"name": "dom_dev", "deviations": dict(DEVIATIONS)}
    rep["not_asserted"] = {"transition_rewards_is_none": "requires W1's runtime.run_lifetime (the legacy harness "
                                                         "shim passes the RewardGate's 0.0 list)"}
    return rep


def test_real_dom_dev_agent_on_ow_tiny(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    rep = run_check(str(tmp_path))
    failed = [k for k, v in rep["checks"].items() if not v]
    assert not failed, (failed, json.dumps(rep["checks"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--update-every", type=int, default=6)
    args = ap.parse_args()
    root = args.root or tempfile.mkdtemp(prefix="w2_e2e_")
    t0 = time.time()
    rep = run_check(root, args.n, args.update_every)
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["root"] = root
    rep["command"] = " ".join([os.path.basename(sys.executable)] + sys.argv)
    try:
        rep["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
        rep["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=SRC,
                                                    text=True).strip())
    except Exception:  # noqa: BLE001
        rep["commit"] = None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps(rep["checks"], indent=1))
    print("passed" if all(rep["checks"].values()) else "FAILED", args.out)


if __name__ == "__main__":
    main()
