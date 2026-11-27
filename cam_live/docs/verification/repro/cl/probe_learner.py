"""G-cl probe: continual learner (NA-08, CL-02, CL-03, CL-08), promotion / rollback (CL-05) and restart / export
(CL-06) on the real model class at debug scale (CPU), through D's own lab wiring.

Parts (``--only``):
  learn   trigger (independent units, not rows), frequent update changes experiential + heads only, slow update
          adapters only, reward group untouched, behaviour probe changes, per-group clipping applied to what the
          optimizer sees, fast update changes no parameter, guarded slow-update revert (weights AND optimizer state)
  promote promotion gates refuse a forgetting / missing-metric checkpoint and the refusal rolls the learner back
  kill    kill -9 at random times while checkpoints are being written: the newest intact checkpoint loads
  export  weights-only export in this process, reload in a fresh process: predictions bit-exact; optimizer state
          restore exact (capture -> fresh learner -> restore -> state tensors equal)

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/cl/probe_learner.py [--only ...]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time

import torch

sys.path.insert(0, "cam_live/src")
from cam_live.config import CamLiveConfig                                       # noqa: E402
from cam_live.credit import checkpoint as ckpt                                  # noqa: E402
from cam_live.credit.labkit import build_lab, collect_family                    # noqa: E402
from cam_live.credit.learner import LearnerSettings                            # noqa: E402
from cam_live.credit.retention import retention_metrics                        # noqa: E402
from cam_live.diagnostics.world import LabConfig                               # noqa: E402
from cam_live.learning_api import TIMESCALE_SLOW, UpdatePlan, encode_decisions  # noqa: E402

WORK = os.environ.get("CL_WORK", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G_cl/work_learner")
FAILS, OUT = [], {}
FAM = LabConfig(n_channels=4, family="lab/A", target=30)


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def fresh(name):
    p = os.path.join(WORK, name)
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p)
    return p


def make_lab(name, *, collect=True, n_worlds=12, **skw):
    torch.manual_seed(0)
    cfg = CamLiveConfig.debug()
    cfg.optim.adapter_every = skw.pop("adapter_every", 3)
    settings = LearnerSettings.from_cfg(cfg, val_fraction=0.2, batch_size=12, **skw)
    lab = build_lab(os.path.join(fresh(name), "store.sqlite"), cfg, bridge=True, settings=settings)
    if collect:
        collect_family(lab.store, lab.linker, FAM, n_worlds, 6, seed=4, clock=lab.clock)
    return lab


def snap(model, group):
    return {f"{i}": p.detach().clone() for i, p in enumerate(model.param_groups().get(group, []))}


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def part_learn():
    lab = make_lab("learn", adapter_every=3)
    L, m, st = lab.learner, lab.model, lab.store
    try:
        groups = list(m.param_groups())
        OUT["groups"] = groups
        # ---- trigger: rows vs independent units
        q = st.retrain_queue()
        plan = L.poll()
        OUT["trigger_state"] = dict(L.trigger_state)
        check("poll(): a plan exists after matured outcomes, counting independent units",
              plan is not None and L.trigger_state["independent_units"] <= L.trigger_state["matured_rows"],
              json.dumps(L.trigger_state))
        # same rows again, only a 'return_changed' reason -> no trigger (not a matured outcome)
        lab2 = make_lab("trigger", collect=False)
        try:
            check("poll(): no plan without matured outcomes (time / rows alone do not trigger)", lab2.learner.poll() is None)
        finally:
            lab2.close()
        # ---- one frequent update
        reward0, adapt0 = snap(m, "reward"), snap(m, "semantic_adapters")
        exp0, head0 = snap(m, "experiential"), snap(m, "heads")
        seen = {}
        for g, opt in L.optimizers.items():
            orig = opt.step

            def wrapped(*a, _g=g, _orig=orig, _opt=opt, **k):
                ps = [p for pg in _opt.param_groups for p in pg["params"] if p.grad is not None]
                seen.setdefault(_g, []).append(float(torch.norm(torch.stack([p.grad.norm() for p in ps]))) if ps else 0.0)
                return _orig(*a, **k)
            opt.step = wrapped
        L.clip["experiential"] = 1e-3        # far below the natural norm: clipping must bind
        recs = L.learn(UpdatePlan(trigger="replay", max_steps=1))
        fr = [r for r in recs if r.timescale == "frequent"][0]
        OUT["frequent_record"] = {"param_groups": fr.param_groups, "grad_norms": fr.grad_norms,
                                  "weight_delta": fr.weight_delta, "prediction_change": fr.prediction_change}
        check("frequent update changed experiential and heads", not same(exp0, snap(m, "experiential")) and
              not same(head0, snap(m, "heads")), str(fr.param_groups))
        check("frequent update left semantic adapters unchanged", same(adapt0, snap(m, "semantic_adapters")))
        check("reward group unchanged and without .grad", same(reward0, snap(m, "reward")) and
              all(p.grad is None for p in m.param_groups().get("reward", [])))
        check("per-group clipping: the experiential optimizer stepped on a gradient of norm <= clip",
              seen.get("experiential") and seen["experiential"][-1] <= 1e-3 * (1 + 1e-4),
              f"pre-clip {fr.grad_norms.get('experiential'):.4g}, seen by optimizer {seen.get('experiential')}")
        check("behaviour probe changed after the frequent update", fr.prediction_change.get("max_abs", 0) > 1e-7,
              str(fr.prediction_change))
        L.clip["experiential"] = lab.cfg.optim.clip_experiential
        # ---- slow schedule: adapter_every=3 frequent updates -> one slow update
        recs = L.learn(UpdatePlan(trigger="replay", max_steps=5, budget_s=1e9))
        ts = [r.timescale for r in recs]
        slow = [r for r in recs if r.timescale == TIMESCALE_SLOW]
        OUT["schedule"] = ts
        # 1 frequent update happened before: frequent #2, #3 -> slow, #4, #5, #6 -> slow
        expect = ["frequent", "frequent", TIMESCALE_SLOW, "frequent", "frequent", "frequent", TIMESCALE_SLOW]
        check("slow adapter update runs after every adapter_every (=3) applied frequent updates", ts == expect,
              f"timescales {ts}, counters {L.counters}")
        if slow:
            s = slow[0]
            check("slow update changed only semantic_adapters", s.param_groups == ["semantic_adapters"], str(s.param_groups))
        # ---- fast update changes no parameter
        allp = {g: snap(m, g) for g in m.param_groups()}
        recs_m = [r for r in st.all() if r.outcomes][:2]
        b = encode_decisions(m, st, recs_m, grad=False)
        from cam_live.learning_api import matured_outcomes
        from cam_live.credit.labkit import _canonical
        qs = [[_canonical(matured_outcomes(r)[0][0])] for r in recs_m]
        phi = m.features(b.z_chosen[:, None], m.encode_queries(qs)[0])[:, 0, 0]
        before_fast = {k: v["mu"].clone() for k, v in L.fast_state.state_dict()["post"].items()}
        L.fast_update(phi[0], float(matured_outcomes(recs_m[0])[0][1].value), "count", exposure=qs[0][0].exposure)
        after_fast = L.fast_state.state_dict()["post"]
        check("fast update: no persistent parameter changed", all(same(allp[g], snap(m, g)) for g in allp))
        check("fast update: the fast belief changed", any(not torch.equal(before_fast[k], after_fast[k]["mu"]) for k in before_fast))
        # ---- guarded slow update that is reverted: are weights AND optimizer state restored?
        L.settings.slow_guard_tol = -1e9 if False else -0.999999       # any slow step that is not a 99.9999 % improvement is undone
        opt = L.optimizers["semantic_adapters"]
        a_before = snap(m, "semantic_adapters")
        st_before = {id(p): {k: (v.clone() if torch.is_tensor(v) else v) for k, v in opt.state.get(p, {}).items()}
                     for pg in opt.param_groups for p in pg["params"]}
        r = [x for x in L.learn(UpdatePlan(trigger="probe", timescales=(TIMESCALE_SLOW,))) if x.timescale == TIMESCALE_SLOW][-1]
        notes = json.loads(r.notes)
        OUT["guard"] = notes.get("guard")
        reverted = bool(notes.get("guard", {}).get("reverted"))
        check("guarded slow update was reverted (setup)", reverted, json.dumps(notes.get("guard")))
        if reverted:
            check("revert restores the adapter weights bit-exactly", same(a_before, snap(m, "semantic_adapters")))
            changed = 0
            for pg in opt.param_groups:
                for p in pg["params"]:
                    sb, sa = st_before.get(id(p), {}), opt.state.get(p, {})
                    for k, v in sa.items():
                        if torch.is_tensor(v) and (k not in sb or not torch.equal(sb[k], v)):
                            changed += 1
                            break
            OUT["guard_optimizer_tensors_changed_after_revert"] = changed
            check("revert also restores the adapter optimizer state (Adam moments / step)", changed == 0,
                  f"{changed} adapter parameters carry Adam state from the rejected step")
    finally:
        lab.close()


def part_promote():
    lab = make_lab("promote")
    L = lab.learner
    try:
        root = os.path.join(WORK, "promote", "ckpt")
        mgr = ckpt.CheckpointManager(root)
        d0, i0, _ = ckpt.promote_learner(mgr, L, retention={"lab/A/neg_nll": -2.0})
        check("first checkpoint with validation + retention is promoted", d0.promoted, str(d0.reasons))
        h0 = L.model.group_hashes()
        o0 = {g: {k: v.clone() for k, v in o.state_dict()["state"].get(0, {}).items() if torch.is_tensor(v)}
              for g, o in L.optimizers.items()}
        L.learn(UpdatePlan(trigger="replay", max_steps=2, budget_s=1e9))
        changed = L.model.group_hashes() != h0
        d1, _, rb = ckpt.promote_learner(mgr, L, retention={"lab/A/neg_nll": -2.5})
        check("a checkpoint that forgot (retention -0.5) is refused", not d1.promoted, str(d1.reasons))
        check("refusal rolls the learner back to the promoted weights", changed and L.model.group_hashes() == h0 and rb is not None)
        o1 = {g: {k: v for k, v in o.state_dict()["state"].get(0, {}).items() if torch.is_tensor(v)}
              for g, o in L.optimizers.items()}
        same_opt = all(torch.equal(o0[g][k], o1[g][k]) for g in o0 for k in o0[g]) and all(not o1[g] for g in o1 if not o0[g])
        check("... and to the promoted optimizer state", same_opt)
        d2 = mgr.promote(mgr.latest().checkpoint_id, validation={"heldout/neg_world": -1.0}, retention={})
        check("no retention metric -> refused", not d2.promoted, str(d2.reasons))
        d3 = mgr.promote(mgr.latest().checkpoint_id, validation={"intrinsic_reward": 99.0},
                         retention={"lab/A/neg_nll": -2.0})
        check("internal reward alone cannot promote", not d3.promoted, str(d3.reasons))
        d4 = mgr.promote(mgr.latest().checkpoint_id, validation={"heldout/neg_reward_return": 99.0, **{k: v for k, v in
                         d0.validation.items()}}, retention={"lab/A/neg_nll": -2.0})
        OUT["promote_metric_named_reward_return"] = {"promoted": d4.promoted, "ignored": d4.ignored}
        print(f"       a metric named 'heldout/neg_reward_return' is ignored? {d4.ignored}")
    finally:
        lab.close()


KILL_CHILD = r'''
import os, sys, torch
sys.path.insert(0, "cam_live/src")
from cam_live.credit import checkpoint as ckpt
root = sys.argv[1]
torch.manual_seed(int(sys.argv[2]))
w = {"experiential": {f"w{i}": torch.randn(512, 1024) for i in range(24)}, "heads": {"h": torch.randn(64, 64)}}
i = 0
while True:
    st = ckpt.CheckpointState(weights=w, schedules={"i": i}, model_version=f"v{i}")
    info = ckpt.save_checkpoint(root, st, notes=str(i))
    os.write(1, f"SAVED {info.checkpoint_id} {i}\n".encode())
    for t in w["experiential"].values():
        t.add_(1.0)
    i += 1
'''


def part_kill():
    d = fresh("kill")
    child = os.path.join(d, "child.py")
    open(child, "w").write(KILL_CHILD)
    root = os.path.join(d, "ckpt")
    rng = random.Random(3)
    results = []
    for k in range(10):
        p = subprocess.Popen([sys.executable, child, root, str(k)], stdout=subprocess.PIPE, cwd=os.getcwd())
        time.sleep(4.0 + rng.random() * 3.0)
        os.kill(p.pid, signal.SIGKILL)
        saved = [l.split() for l in p.communicate()[0].decode().splitlines() if l.startswith("SAVED")]
        rec = ckpt.recover(root)
        info = ckpt.latest_valid(root)
        ok = info is not None
        problems = ckpt.verify_checkpoint(info.path) if ok else ["none"]
        stt = ckpt.load_checkpoint(root) if ok else None
        last_ack = saved[-1][1] if saved else None
        # the newest intact checkpoint must be at least the last acknowledged one
        newest_ok = ok and (last_ack is None or info.checkpoint_id >= last_ack or
                            ckpt._seq_of(info.checkpoint_id) >= ckpt._seq_of(last_ack))
        consistent = ok and stt is not None and stt.model_version == f"v{stt.schedules['i']}"
        results.append({"acked": len(saved), "latest": info.checkpoint_id if ok else None, "last_ack": last_ack,
                        "problems": problems, "newest_ok": newest_ok, "consistent": consistent,
                        "debris": sorted(x for x in os.listdir(root) if x.startswith(".tmp"))})
    OUT["kill_checkpoint"] = results
    check("kill -9 during checkpoint writes (10 kills): newest intact checkpoint >= last acknowledged, loads, verifies",
          all(r["newest_ok"] and not r["problems"] and r["consistent"] for r in results) and sum(r["acked"] for r in results) > 5,
          f"acked per run {[r['acked'] for r in results]}")


EXPORT_CHILD = r'''
import json, sys, torch
sys.path.insert(0, "cam_live/src")
from cam_live.credit import checkpoint as ckpt
from cam_live.credit.labkit import register_lab_heads
from cam_live.credit.replay import SqliteExperienceStore
from cam_live.learning_api import encode_decisions
torch.set_num_threads(1)
exp_dir, store_path, ids, out = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4]
holder = {}
def reg(model):
    holder.update(register_lab_heads(model, None, world=True, policy=True, credit=True, seed=0))
em = ckpt.load_exported_model(exp_dir, register_heads=reg)
st = SqliteExperienceStore(store_path, readonly=True)
recs = st.get_many(ids)
em.model.eval()
with torch.no_grad():
    b = encode_decisions(em.model, st, recs, grad=False)
    pw = holder["world"].predict(em.model, b)
    pp = holder["policy"](b.z, b.a_mask, b.b).probs
torch.save({"b": b.b, "z": b.z_chosen, "nll": pw["nll"], "probs": pp}, out)
print("OK", em.report.summary() if hasattr(em.report, "summary") else "")
'''


def part_export():
    lab = make_lab("export")
    L, m, st = lab.learner, lab.model, lab.store
    try:
        L.learn(UpdatePlan(trigger="replay", max_steps=4, budget_s=1e9))
        exp_dir = os.path.join(WORK, "export", "weights")
        ckpt.export_learner_weights(L, exp_dir)
        ids = [r.exp_id for r in st.all() if r.outcomes][:6]
        recs = st.get_many(ids)
        m.eval()
        with torch.no_grad():
            b = encode_decisions(m, st, recs, grad=False)
            ref = {"b": b.b, "z": b.z_chosen, "nll": lab.world.predict(m, b)["nll"], "probs": lab.policy(b.z, b.a_mask, b.b).probs}
        st.flush()
        child = os.path.join(WORK, "export", "child.py")
        open(child, "w").write(EXPORT_CHILD)
        outp = os.path.join(WORK, "export", "child_out.pt")
        r = subprocess.run([sys.executable, child, exp_dir, st.path, json.dumps(ids), outp], capture_output=True, text=True,
                           cwd=os.getcwd(), env={**os.environ, "OMP_NUM_THREADS": "1"})
        ok = r.returncode == 0
        check("fresh process loads the weights-only export", ok, (r.stdout + r.stderr)[-600:])
        if ok:
            got = torch.load(outp)
            diffs = {k: float((ref[k].double() - got[k].double()).abs().max()) for k in ref}
            OUT["export_max_abs_diff"] = diffs
            check("fresh-process predictions are bit-exact (belief, decision latent, world NLL, action probs)",
                  all(torch.equal(ref[k], got[k]) for k in ref), str(diffs))
        files = sorted(os.listdir(exp_dir))
        keys = ckpt.export_keys(exp_dir)
        OUT["export_files"] = files
        OUT["export_has_optimizer_or_store"] = [k for k in keys if "exp_avg" in k or "store" in k]
        check("export holds parameters and configuration only", not OUT["export_has_optimizer_or_store"]
              and not any(f.endswith(".sqlite") for f in files), str(files))
        # optimizer state restore exact
        state = ckpt.capture_learner(L)
        lab2 = make_lab("export2", collect=False)
        try:
            rep = ckpt.restore_learner(lab2.learner, state)
            diffs = 0
            for g, o in L.optimizers.items():
                o2 = lab2.learner.optimizers[g]
                n1 = {n: p for n, p in m.named_parameters()}
                n2 = {n: p for n, p in lab2.model.named_parameters()}
                inv1 = {id(p): n for n, p in n1.items()}
                for pg in o.param_groups:
                    for p in pg["params"]:
                        s1 = o.state.get(p, {})
                        s2 = o2.state.get(n2[inv1[id(p)]], {})
                        for k, v in s1.items():
                            if torch.is_tensor(v) and not torch.equal(v.cpu(), s2.get(k, torch.tensor(float("nan"))).cpu()):
                                diffs += 1
            check("optimizer state restored exactly into a fresh learner (by parameter name)", diffs == 0, f"{diffs} differ")
            check("weights restored exactly", lab2.model.group_hashes() == m.group_hashes())
        finally:
            lab2.close()
    finally:
        lab.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="learn,promote,kill,export")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("CL_THREADS", "1")))
    for k in a.only.split(","):
        {"learn": part_learn, "promote": part_promote, "kill": part_kill, "export": part_export}[k]()
    print(json.dumps(OUT, indent=1, default=str))
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)


if __name__ == "__main__":
    main()
