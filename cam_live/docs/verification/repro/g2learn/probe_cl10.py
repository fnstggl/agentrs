"""G2-learn round 2 probe: CL-10 — does a checkpoint of the INTEGRATED agent hold every item of spec 15.2, on the
CPU, and come back in a fresh process?

    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_cl10.py write DIR
    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_cl10.py read DIR      (a second, fresh process)

write: the integrated agent (tiny backbone, CPU) runs two LabWorld lifetimes through the real runtime, then
``agent.save(DIR/ckpt)``; the checkpoint directory is inspected item by item (15.2: model weights, semantic adapters,
intrinsic reward weights, optimizer state, fast adaptive state, current task, replay indices, experiment records,
pending outcomes, learning schedules, random seeds, authorization configuration, model versions), tensors must be on
the CPU; reference values are saved. read: a fresh agent on a copy of the run directory loads the checkpoint and
must reproduce the trainable state, fast state, counters and RNG states exactly.
"""
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import TESTS, check, finish, info  # noqa: E402

sys.path.insert(0, os.path.join(TESTS, "integration"))
from int_helpers import Recorder, agent_config, run_lifetime  # noqa: E402

import torch  # noqa: E402

from cam_live.agent import CamLiveAgent  # noqa: E402
from cam_live.contracts import SimClock  # noqa: E402

mode, D = sys.argv[1], sys.argv[2]       # write | read | tamper
torch.set_num_threads(2)


def flat_state(agent):
    return {k: v.detach().cpu().clone() for k, v in agent.model.trainable_state().items()}


def fast_flat(agent):
    sd = agent.learner.fast_state.state_dict()
    return {f"{k}.{n}": v for k, p in sd["post"].items() for n, v in p.items() if torch.is_tensor(v)}


if mode == "write":
    shutil.rmtree(D, ignore_errors=True)
    os.makedirs(D)
    cfg = agent_config(policy_update_every=6)
    agent = CamLiveAgent(cfg, os.path.join(D, "run", "agent"), clock=SimClock())
    rec = Recorder(agent)
    lab = {"outcome_horizon_s": 1800.0, "mean_delay_s": (300.0, 900.0)}
    run_lifetime(rec, agent.clock, os.path.join(D, "run"), 11, max_steps=40, max_decisions=22, lab_kw=lab)
    run_lifetime(rec, agent.clock, os.path.join(D, "run"), 12, max_steps=4, max_decisions=4, lab_kw=lab)   # stops with outcomes pending
    ck = os.path.join(D, "ckpt")
    cid = agent.save(ck)
    files = sorted(os.path.relpath(p, ck) for p in glob.glob(os.path.join(ck, "**", "*"), recursive=True)
                   if os.path.isfile(p))
    info("checkpoint files", files)
    stj = next((p for p in glob.glob(os.path.join(ck, "**", "state.json"), recursive=True)), None)
    st = json.load(open(stj)) if stj else {}
    info("state.json keys", sorted(st))
    reg_exps = len(agent.experiments.records())     # the agent keeps its experiments in predict.registry
    store_exps = len(agent.store.experiments())
    pend = agent.store.pending_all()
    items = {
        "model weights / adapters / reward weights": any("weights" in f for f in files),
        "optimizer state": any(f.startswith("optim") or "optim" in f for f in files),
        "fast adaptive state": any("fast" in f for f in files),
        "current task": bool(st.get("current_task")),
        "replay indices": bool(st.get("replay_state") or st.get("replay")),
        "experiment records": "experiments" in st,
        "pending outcomes": "pending_outcomes" in st,
        "learning schedules": bool(st.get("schedules")),
        "random seeds": any("rng" in f for f in files) or bool(st.get("rng")),
        "authorization configuration": bool(st.get("auth_config_hash")),
        "model versions": bool(st.get("model_version")),
    }
    for k, v in items.items():
        check(f"CL-10: the integrated agent's checkpoint holds: {k}", v)
    info("experiments: agent registry / experience store / checkpoint", (reg_exps, store_exps, len(st.get("experiments") or [])))
    check("CL-10: the checkpoint's experiment records are the agent's experiments", reg_exps is not None and
          len(st.get("experiments") or []) == reg_exps, f"registry {reg_exps}, checkpoint {len(st.get('experiments') or [])}")
    info("pending outcome queries: store / checkpoint", (len(pend), len(st.get("pending_outcomes") or [])))
    check("CL-10: the checkpoint's pending outcomes are the store's unresolved queries (and there are some)",
          len(st.get("pending_outcomes") or []) == len(pend) and len(pend) > 0, f"{len(pend)} vs {len(st.get('pending_outcomes') or [])}")
    info("auth_config_hash recorded", repr(st.get("auth_config_hash")))
    # every tensor file loads to the CPU
    devs = set()
    for f in files:
        p = os.path.join(ck, f)
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file
            devs |= {str(t.device) for t in load_file(p).values()}
        elif f.endswith(".pt"):
            obj = torch.load(p, map_location=None, weights_only=False)
            stack = [obj]
            while stack:
                o = stack.pop()
                if torch.is_tensor(o):
                    devs.add(str(o.device))
                elif isinstance(o, dict):
                    stack += list(o.values())
                elif isinstance(o, (list, tuple)):
                    stack += list(o)
    check("CL-10: every tensor in the checkpoint is stored on the CPU", devs <= {"cpu"}, f"{sorted(devs)}")
    torch.save({"state": flat_state(agent), "fast": fast_flat(agent), "counters": dict(agent.learner.counters),
                "gen": agent.gen.get_state(), "n_decisions": agent.n_decisions, "cid": cid}, os.path.join(D, "ref.pt"))
    agent.close()
    shutil.copytree(os.path.join(D, "run"), os.path.join(D, "run_copy"))
elif mode == "tamper":
    # DX-2: the authorization hash is "checked on load". A fresh agent whose task's authorization fields differ
    # (another interface list / limits) loads the checkpoint: is the difference refused, reported, or silent?
    from cam_live.credit.checkpoint import auth_config_hash, load_checkpoint, restore_learner
    from cam_live.contracts import TaskSpec
    st = json.load(open(next(p for p in glob.glob(os.path.join(D, "ckpt", "**", "state.json"), recursive=True))))
    task = TaskSpec.from_dict(st["current_task"])
    info("recorded auth_config_hash / hash of the recorded task", (st.get("auth_config_hash"), auth_config_hash(task)))
    check("CL-10: the agent checkpoint records the authorization hash of its task", bool(st.get("auth_config_hash"))
          and st.get("auth_config_hash") == auth_config_hash(task))
    for k in (1, 2):                              # each agent loads into its own copy of the run directory
        shutil.rmtree(os.path.join(D, f"run_tamper{k}"), ignore_errors=True)
        shutil.copytree(os.path.join(D, "run"), os.path.join(D, f"run_tamper{k}"))
    cfg = agent_config(policy_update_every=6)
    # (1) fresh process, no task yet (what a restart does): is any check made?
    a1 = CamLiveAgent(cfg, os.path.join(D, "run_tamper1", "agent"), clock=SimClock())
    info("fresh agent before load: task", None if a1.task is None else a1.task.objective_id)
    a1.load(os.path.join(D, "ckpt"))
    info("(1) restart without a task: loaded, task now", a1.task.objective_id if a1.task else None)
    a1.close()
    # (2) an agent whose task's authorization fields differ
    t2 = TaskSpec.from_dict(st["current_task"])
    t2.limits.max_decisions = int(t2.limits.max_decisions or 0) + 999
    a2 = CamLiveAgent(cfg, os.path.join(D, "run_tamper2", "agent"), clock=SimClock())
    a2.set_objective(t2)
    h2 = auth_config_hash(t2)
    raised = None
    try:
        a2.load(os.path.join(D, "ckpt"))
    except Exception as e:  # noqa: BLE001
        raised = f"{type(e).__name__}: {str(e)[:120]}"
    rep = restore_learner(a2.learner, load_checkpoint(os.path.join(D, "ckpt")), expected_auth_hash=h2) if raised is None else {}
    info("(2) changed authorization fields: agent.load raised / restore report auth_config_changed / task after load",
         (raised, rep.get("auth_config_changed"), a2.task.limits.max_decisions if a2.task else None))
    check("CL-10 (DX-2): loading a checkpoint under a changed authorization configuration is refused or at least "
          "reported by the agent", raised is not None, "agent.load completed silently; the restore report's "
          "auth_config_changed is computed but the agent discards the report" if raised is None else raised)
    a2.close()
else:
    ref = torch.load(os.path.join(D, "ref.pt"), weights_only=False)
    cfg = agent_config(policy_update_every=6)
    agent = CamLiveAgent(cfg, os.path.join(D, "run_copy", "agent"), clock=SimClock())
    before = flat_state(agent)
    agent.load(os.path.join(D, "ckpt"))
    now = flat_state(agent)
    diff = [k for k in ref["state"] if k not in now or not torch.equal(ref["state"][k], now[k])]
    moved = sum(1 for k in before if k in now and not torch.equal(before[k], now[k]))
    check("CL-10 recovery: a fresh agent restores every trainable tensor bit-exactly", not diff,
          f"{len(diff)} differ of {len(ref['state'])}; {moved} tensors changed by the load")
    fd = [k for k in ref["fast"] if not torch.equal(ref["fast"][k], fast_flat(agent).get(k, torch.empty(0)))]
    check("CL-10 recovery: the fast adaptive state comes back exactly", not fd, f"{fd[:3]}")
    check("CL-10 recovery: learner counters (schedules) come back", dict(agent.learner.counters) == ref["counters"],
          f"{dict(agent.learner.counters)} vs {ref['counters']}")
    check("CL-10 recovery: the agent's random generator comes back", torch.equal(agent.gen.get_state(), ref["gen"]))
    check("CL-10 recovery: the current task comes back", agent.task is not None, f"{getattr(agent.task, 'objective_id', None)}")
    info("experience store of the run copy: experiments / pending queries after the load",
         (len(agent.store.experiments()), len(agent.store.pending_all())))
    agent.close()
sys.exit(finish())
