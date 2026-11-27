"""W2 real end-to-end check (redesign plan 2026-11-26 §3 W2): the real ``dom_dev`` CamUnifiedAgent (8 x 384 core,
Qwen3-0.6B, built by `stage1.system.build_system` from `config.make_preset("dom_dev")`) on a process-isolated ow_tiny
OpenWorld from `factory.make_env` (observation_only, result delivery on), driven by the lifetime harness for 12
decisions with 2 updates. Checks:

* the decision after ``world.hosts`` (decision 0 is offered only that tool, so the real decoder must call it) holds a
  TOOL_RESULT whose ``caused_by`` is that action's id - and every later decision holds its predecessor's result;
* observation_only: every reward the agent receives is 0.0 (`Transition.rewards is None` needs W1's canonical loop:
  reported, not asserted). Payout invariance, bit-exact: a twin world B (same task and seeds, its evaluator pays
  x7 + 3.5 per clock stop; the operator's legacy totals differ) is stepped in lock-step with the agent's own world A
  using the agent's actions; at every decision (i) B delivers exactly what A delivered (results, next observation
  batch, rewards), and (ii) the real model's belief update (`belief_runtime.observe_outcomes`: core outcome trackers,
  particles, Kalman layer = the FastState) computed on cloned states from B's transition is bit-identical to the one
  from A's transition (a repeat of A's is the noise control). Two separately built agents cannot be compared bit by
  bit on this GPU stack (measured: a same-payout rebuild diverges from decision 0, and its actions after the first
  update - non-deterministic attention backward), so invariance is checked within one process on identical calls;
* canaries planted in the task's private params, in the evaluator's objective state / status and in the label
  channel (a verified-metric name) never appear in what the agent received, in its archive / traces / decision logs
  or in the harness transcript; no evaluator-status key appears in the agent's inputs.

GPU: ~12 GB. Run as a script for the evidence JSON, under the shared lock (never run pytest under the lock):

    flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock python tests/cam_unified/envs/test_env_real_e2e.py \\
        --out docs/evidence/dom/W2/env_e2e_dom_dev.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
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


def snapshot(state) -> dict:
    """Bitwise-comparable copy of a CamState's core state Z and FastState (tensors on CPU)."""
    import torch
    out = {"Z": state.Z.detach().cpu().clone()}
    for name in ("fast", "fast_prev"):
        fast = getattr(state, name, None)
        if fast is None:
            continue
        for f in fields(fast):
            v = getattr(fast, f.name)
            out[f"{name}.{f.name}"] = v.detach().cpu().clone() if torch.is_tensor(v) else json.dumps(v, default=str)
    return out


def bit_equal(x, y) -> bool:
    """Bitwise tensor equality where NaN equals NaN (FastState keeps NaN placeholders, e.g. empty surprise slots)."""
    import torch
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.is_floating_point():
        nx, ny = torch.isnan(x), torch.isnan(y)
        return bool(torch.equal(nx, ny) and torch.equal(x[~nx], y[~ny]))
    return bool(torch.equal(x, y))


def same_state(a: dict, b: dict) -> bool:
    import torch
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if torch.is_tensor(x):
            if not torch.is_tensor(y) or not bit_equal(x, y):
                return False
        elif x != y:
            return False
    return True


def state_diff(a: dict, b: dict) -> dict:
    """{key: max |a - b|} for the keys whose values differ (non-tensor keys: "differs")."""
    import torch
    out = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if torch.is_tensor(x) and torch.is_tensor(y) and x.shape == y.shape:
            if not bit_equal(x, y):
                d = (x.double() - y.double()).abs()
                d = d[torch.isfinite(d)]
                out[k] = float(d.max()) if d.numel() else float("nan")
        elif x != y if not (torch.is_tensor(x) or torch.is_tensor(y)) else True:
            out[k] = "differs"
    return out


def _norm(text: str) -> str:
    text = re.sub(r"ow\d+-[0-9a-f]{10}", "ENV", text)
    return re.sub(r"g_[0-9a-f]{32}", "GOAL", text)


class Recorder:
    """The real agent, recorded: its act() inputs, its actions and the transitions it received. Decision 0 is offered
    only `first_tools` (an environment choice, not an agent change). With a `twin` (adapter, evaluator) of a world
    whose evaluator pays differently, every executed action is also executed there in lock-step (same Action
    objects, same requested durations), and the real model's belief update is recomputed on cloned states from the
    twin's transition (see the module docstring)."""

    def __init__(self, inner, first_tools=FIRST_TOOLS, twin=None):
        self.inner, self.first_tools, self.twin = inner, tuple(first_tools), twin
        self.inputs, self.actions, self.transitions, self.act_s, self.twin_rows = [], [], [], [], []

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
        if self.twin is not None:
            self.twin_rows.append(self._twin_step(tr))
        return self.inner.observe(tr)

    def update(self):
        return self.inner.update()

    def _twin_step(self, tr) -> dict:
        from superlearner.cam_unified.core.belief_runtime import observe_outcomes
        from superlearner.cam_unified.envs import integrity as I
        ad_b, ev_b = self.twin
        acts = [None if r is None else r.action for r in tr.records]
        ad_b.available_tools()                                  # the same calls world A received this round
        res_b = ad_b.execute(acts)
        nb = ad_b.advance_time([float(r.extra.get("dt_requested_days", 0.0)) for r in tr.records])
        rew_b = ev_b.take_rewards()
        res_a = [None if r is None else r.result for r in tr.records]
        row = {"results_identical": _norm(I.actor_dump(res_a)) == _norm(I.actor_dump(res_b)),
               "next_obs_identical": _norm(I.actor_dump(tr.next_obs.observations)) == _norm(
                   I.actor_dump(nb.observations)),
               "t_done_identical": list(tr.next_obs.t) == list(nb.t) and list(tr.next_obs.done) == list(nb.done),
               "rewards_a": list(tr.rewards), "rewards_b": list(rew_b)}
        import random

        import numpy as np
        import torch
        agent = self.inner.agent
        rows, dec = agent._pending[0], agent._pending[1]
        model = agent.model
        sub = agent.state.select(rows)
        # the particle update samples (rejuvenation noise, systematic resampling) from the global RNGs: every
        # counterfactual starts from the same RNG state, and the real update afterwards from the untouched one
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())

        def set_rng():
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state_all(rng[1])
            np.random.set_state(rng[2])
            random.setstate(rng[3])

        def update(res, nxt, rew):
            st = copy.deepcopy(sub)
            set_rng()
            observe_outcomes(model, st, dec, [res[b] for b in rows], [list(nxt[b]) for b in rows],
                             [float(rew[b]) for b in rows])
            return snapshot(st)
        try:
            cf_w = update(res_a, tr.next_obs.observations, tr.rewards)     # warm-up (fills encoder caches)
            cf_a2 = update(res_a, tr.next_obs.observations, tr.rewards)
            cf_b = update(res_b, nb.observations, rew_b)
            cf_a3 = update(res_a, tr.next_obs.observations, tr.rewards)
        finally:
            set_rng()
        row["fast_state_identical"] = same_state(cf_a2, cf_b) and same_state(cf_b, cf_a3)
        row["fast_state_repeatable"] = same_state(cf_a2, cf_a3)
        row["warm_equals_cold"] = same_state(cf_w, cf_a2)
        row["fast_state_keys"] = len(cf_b)
        row["maxdiff_a2_b"] = state_diff(cf_a2, cf_b)
        row["maxdiff_a2_a3"] = state_diff(cf_a2, cf_a3)
        row["maxdiff_cold_warm"] = state_diff(cf_w, cf_a2)
        return row


def _core_hash(model) -> str:
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(model.core.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _task(canaries: dict, n: int):
    from superlearner.cam_unified.envs.protocols.ow1 import open_economy_task
    task = open_economy_task(seed=100_000, days=1)
    task.params = {**task.params, "canary": canaries["task"]}
    task.private_note = canaries["task_attr"]
    task.horizon_s = int(n * 600)                     # the episode resolves (failure label) at the last decision
    return task


def _world(root: str, tag: str, canaries: dict, n: int, seed: int, payout_scale: float, payout_bias: float):
    from superlearner.cam_unified.envs import integrity as I
    from superlearner.cam_unified.envs.factory import make_env
    return make_env("openworld", tasks=[_task(canaries, n)], isolated=True, seed=seed, max_wait_s=600,
                    evaluator_kw={"objective_factory": I.canary_objective_factory(
                        {"state": canaries["state"], "status": canaries["status"], "label": canaries["label"]},
                        payout_scale=payout_scale, payout_bias=payout_bias)},
                    store_url=f"sqlite:///{os.path.join(root, tag + '_world.db')}")   # operator side: not run_dir


def run_check(root: str, n: int = 12, update_every: int = 6, seed: int = 0) -> dict:
    import torch
    from superlearner.cam_unified.envs import integrity as I
    from superlearner.cam_unified.envs.harness import run_lifetime
    from superlearner.cam_unified.envs.labels import LabelStore
    from superlearner.cam_unified.stage1.harness_agent import HarnessAgent
    canaries = {k: I.new_canary(k) for k in ("task", "task_attr", "state", "status", "label")}
    run_dir = os.path.join(root, "run")
    os.makedirs(run_dir, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    system, over = dom_dev_system(run_dir, seed)
    build_s = time.time() - t0
    core0 = _core_hash(system.model)
    es_a = _world(root, "A", canaries, n, seed, 1.0, 0.0)
    es_b = _world(root, "B", canaries, n, seed, 7.0, 3.5)
    ad_a, _, rev_a = es_a
    ad_b, _, rev_b = es_b
    ad_b.reset()
    rec = Recorder(HarnessAgent(system), twin=(ad_b, rev_b))
    store = LabelStore(os.path.join(run_dir, "labels"))
    labels, labels_b = [], []

    def on_step(k, tr):
        for lab in rev_a.take_labels():
            store.append(lab)
            labels.append(lab)
        labels_b.extend(rev_b.take_labels())
    t1 = time.time()
    try:
        res = run_lifetime(rec, ad_a, rev_a, max_decisions=n, think_s=600, update_every=update_every,
                           transcript=os.path.join(run_dir, "transcript.jsonl"),
                           model_version=lambda: rec.inner.model_version, on_step=on_step)
        det_a, det_b = rev_a.evaluate_goal(0).details, rev_b.evaluate_goal(0).details
    finally:
        system.recorder.flush()
        system.archive.flush()
        es_a.close()
        es_b.close()
    loop_s = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    inputs = [x for pair in rec.inputs for x in pair] + [rec.goals]
    inputs_dump = I.actor_dump(*inputs)
    recs_dump = I.actor_dump(*[tr.records for tr in rec.transitions], *[tr.next_obs for tr in rec.transitions])
    delivered = []
    for k in range(len(rec.actions) - 1):
        a = rec.actions[k][0]
        obs = rec.inputs[k + 1][0].observations[0]
        hits = [o for o in obs if o.caused_by == a.action_id and o.kind.value in ("tool_result", "error")]
        delivered.append({"k": k, "tool": a.tool, "ok": bool(rec.transitions[k].records[0].result.ok),
                          "result_in_next_input": bool(hits), "kinds": sorted({o.kind.value for o in hits})})
    cans = list(canaries.values())
    scan = I.scan_run_dir(run_dir, cans)
    twin = rec.twin_rows
    rep = {
        "preset": {"name": "dom_dev", "deviations": dict(DEVIATIONS), "overrides": sorted(over)},
        "decisions": res.steps, "updates": len(res.updates), "build_s": round(build_s, 2),
        "loop_s": round(loop_s, 2), "act_s": [round(x, 3) for x in rec.act_s], "gpu_peak_gb": round(peak, 2),
        "core_hash_before": core0, "core_hash_after": _core_hash(system.model),
        "actions": [[a.tool, a.args] for acts in rec.actions for a in acts if a is not None],
        "first_tool": rec.actions[0][0].tool if rec.actions else None,
        "delivered": delivered,
        "rewards_seen": sorted({float(r) for tr in rec.transitions for r in tr.rewards}),
        "transition_rewards_is_none": all(tr.rewards is None for tr in rec.transitions),
        "harness_returns": res.returns,
        "operator_legacy_total": {"A_payout_x1": det_a.get("return_total"),
                                  "B_payout_x7_plus3.5": det_b.get("return_total")},
        "twin_per_decision": twin,
        "labels": [{"status": lab.status, "kind": lab.kind, "t_resolved_days": lab.t_resolved_days,
                    "label_canary_in_label": canaries["label"] in json.dumps(lab.meta)} for lab in labels],
        "labels_twin": [{"status": lab.status, "t_resolved_days": lab.t_resolved_days} for lab in labels_b],
        "canary_hits_inputs": I.canary_hits(inputs_dump, cans),
        "canary_hits_records": I.canary_hits(recs_dump, cans),
        "canary_scan_run_dir": {"hits": scan["hits"], "files": scan["files"], "unscanned": scan["unscanned"]},
        "canary_positive_controls": {
            "status_in_evaluator_details": det_a.get("canary") == canaries["status"],
            "label_in_label_store": I.scan_run_dir(run_dir, [canaries["label"]], exclude=())["hits"] != {}},
        "privileged_keys_in_inputs": I.privileged_keys(inputs),
    }
    rep["checks"] = {
        "decisions_12": res.steps == n,
        "core_hash_changed_by_updates": rep["core_hash_after"] != core0 and rep["updates"] == n // update_every,
        "first_decision_world_hosts": rep["first_tool"] == "world.hosts",
        "world_hosts_result_in_next_input": bool(delivered) and delivered[0]["result_in_next_input"],
        "every_result_in_next_input": bool(delivered) and all(d["result_in_next_input"] for d in delivered),
        "rewards_all_zero": rep["rewards_seen"] == [0.0],
        "twin_stream_identical_under_altered_payout": len(twin) == n and all(
            r["results_identical"] and r["next_obs_identical"] and r["t_done_identical"]
            and r["rewards_a"] == r["rewards_b"] == [0.0] for r in twin),
        "fast_state_bit_identical_under_altered_payout": len(twin) == n and all(
            r["fast_state_identical"] for r in twin),
        "payout_really_altered": det_a.get("return_total") != det_b.get("return_total"),
        "no_canary_in_actor_inputs_records": not rep["canary_hits_inputs"] and not rep["canary_hits_records"],
        "no_canary_in_run_dir": not scan["hits"] and scan["files"] > 0,
        "canary_positive_controls": all(rep["canary_positive_controls"].values()),
        "no_privileged_keys_in_inputs": not rep["privileged_keys_in_inputs"],
        "label_emitted": len(labels) == 1,
    }
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
        rep["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain", "--", ".", SRC], cwd=HERE,
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
