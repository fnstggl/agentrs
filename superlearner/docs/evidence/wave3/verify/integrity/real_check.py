"""Independent verifier (H) real-model integrity check: dom_dev via build_agent + IntegratedLearner(+MetaWindowDriver)
through runtime.run_lifetime on real Crucible rapid worlds, observation_only. Canaries in evaluator refs and labels;
psi gradient hooks; meta-gradient per-parameter norms; gc reachability from the agent."""
import gc, json, os, sys, time, types, tempfile
sys.path.insert(0, "/workspace/agentrs/crucible/src")
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/verify_integrity/real_run.json"
ROUNDS = int(os.environ.get("ROUNDS", "40"))
from superlearner.cam_unified.build import build_agent, SystemConfig
from superlearner.cam_unified.core.bench import rapid_lever_specs
from superlearner.cam_unified.envs.experience import ExperienceOperator, WorldSpec
from superlearner.cam_unified.runtime import run_lifetime, GuardedEnv
from superlearner.cam_unified.stage1.dom_e2e import canary_scan

res = {}
run_dir = tempfile.mkdtemp(prefix="hverify_", dir="/workspace/agentrs/runs")
sc = SystemConfig(deterministic=False, grad_probe_every=1, archive_autoflush=False, fresh_max_steps=64,
                  checkpoint_semantic_cache=False)
sysm = build_agent("dom_dev", env_kind="mixed", device="cuda", run_dir=run_dir,
                   overrides={"grad_checkpoint": True, "update_every": 2, "meta_every": 1}, system=sc, seed=0)
agent, model, learner, rm = sysm.agent, sysm.model, sysm.learner, sysm.reward_model
res["learner"] = type(learner).__name__
res["meta_trainer"] = type(getattr(learner, "meta_trainer", None)).__name__
res["reward_model_stub"] = bool(getattr(rm, "stub", None))
res["model_reward_model_attached"] = model.__dict__.get("reward_model") is not None
res["cfg"] = {k: getattr(sysm.cfg, k) for k in ("reward_mode", "meta_every", "update_every", "meta_inner_steps",
                                                 "reward_in_beliefs", "n_intent")}

# ---- canaries: evaluator-side references (numbers) + label meta string
C_NULL, C_ORACLE = 1234.56789123, 98765.4321987
specs = [WorldSpec(s.tier, s.name, s.seed, (("horizon", 12),)) for s in rapid_lever_specs(4, seed0=3000)]
op = ExperienceOperator(specs)
op._references = lambda w: {"null": C_NULL, "oracle": C_ORACLE, "how": "canary"}
for w in op.worlds:
    w.refs = op._references(w)
C_LABEL = "CANARYLBL7f3e9a1c"
released = []


class CanaryLabels:
    def __init__(self, src):
        self.src = src

    def episode_of(self, e):
        return self.src.episode_of(e)

    def take_labels(self):
        out = self.src.take_labels()
        for r in out:
            r.meta = {**(r.meta or {}), "note": C_LABEL}
            released.append({"env": r.env_id, "ep": r.episode, "u": r.utility, "v": r.value, "status": r.status})
        return out


labels = CanaryLabels(op)
env = GuardedEnv(op.adapter(deliver_results=True))

# ---- gc reachability from the agent to operator-side objects
def reach(root, targets, limit=3_000_000):
    tid = {id(t): n for n, t in targets.items()}
    mod_dicts = {id(m.__dict__) for m in list(sys.modules.values()) if m is not None and hasattr(m, "__dict__")}
    seen, stack, hits, n = set(), [root], {}, 0
    while stack and n < limit:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o)); n += 1
        if id(o) in tid and o is not root:
            hits[tid[id(o)]] = hits.get(tid[id(o)], 0) + 1
            continue
        if isinstance(o, (types.ModuleType, type, types.FrameType)) or id(o) in mod_dicts:
            continue
        if torch.is_tensor(o):
            continue
        for r in gc.get_referents(o):
            if isinstance(r, (int, float, str, bytes, bool)) or r is None:
                continue
            stack.append(r)
    return hits, n

tg = {"operator": op, "label_source": labels, "label_store": sysm.label_store, "learner": learner,
      "world_env0": op.worlds[0].env, "adapter": env}
res["reach_from_agent_before"] = reach(agent, tg)

# ---- psi hooks
psi = list(rm.meta_parameters())
psi_names = {id(p): n for n, p in rm.named_parameters() if any(p is q for q in psi)}
phase = {"name": "act"}
fires = {}
for p in psi:
    def hk(g, n=psi_names[id(p)]):
        k = (phase["name"], n)
        fires[k] = fires.get(k, 0) + 1
    p.register_hook(hk)
supervised = list(rm.supervised_parameters())

def flat(ps):
    return torch.cat([p.detach().float().reshape(-1).cpu() for p in ps])

psi_hist = []
step_log = []
inner = learner


class Wrap:
    reward_model = rm

    def on_step(self, tr):
        inner.on_step(tr)

    def step(self):
        phase["name"] = "slow_step"
        b = flat(psi); s0 = flat(supervised)
        out = inner.step()
        phase["name"] = "act"
        a = flat(psi); s1 = flat(supervised)
        step_log.append({"updated": bool(out.get("updated")), "reason": out.get("reason"),
                         "psi_changed": bool((a != b).any()), "phi_vl_changed": bool((s1 != s0).any()),
                         "psi_grads": [None if p.grad is None else float(p.grad.norm()) for p in psi][:3],
                         "actor_psi_grad_norm": out.get("actor_psi_grad_norm"),
                         "reward": {k: v for k, v in (out.get("reward") or {}).items() if k in (
                             "mode", "psi_version", "stub", "mean", "std", "n", "info_gain_mean", "cost_mean")},
                         "losses": {k: v for k, v in (out.get("losses") or {}).items()},
                         "label_join": out.get("label_join"), "meta": out.get("meta")})
        return out

    def meta_step(self):
        phase["name"] = "meta_open"
        out = inner.meta_step()
        phase["name"] = "act"
        return out

    def state_dict(self):
        return inner.state_dict()


mt = inner.meta_trainer
meta_log = []
if mt is not None:
    tr_ = mt.trainer
    mg0 = tr_.mrl.meta_gradient

    def mg(fast, q, retain_graph=False, score_baseline=None):
        phase["name"] = "meta_outer"
        g = mg0(fast, q, retain_graph=retain_graph, score_baseline=score_baseline)
        phase["name"] = "act"
        meta_log.append({"per_param_grad_norm": {psi_names[id(p)]: float(gi.norm()) for p, gi in zip(tr_.mrl.meta_params, g)},
                         "query_U": [float(x) for x in q.utility.reshape(-1)], "n_query_steps": int(q.mask.sum()),
                         "n_scores": len(tr_.mrl._scores)})
        return g
    tr_.mrl.meta_gradient = mg
    is0 = tr_.inner_step

    def ist(fast, batch, on_policy=True):
        phase["name"] = "meta_inner"
        out = is0(fast, batch, on_policy=on_policy)
        phase["name"] = "act"
        return out
    tr_.inner_step = ist

seen_dump = []
act0 = agent.act

def act(obs, tools):
    seen_dump.append(json.dumps([[o.to_dict() for o in ob] for ob in obs.observations], default=str))
    seen_dump.append(json.dumps([[t.__dict__ for t in ts] for ts in tools], default=str))
    return act0(obs, tools)
agent.act = act
rew_none, rec_zero, ep_mismatch, ep_checked, stamps = [], [], [], 0, []

def on_step(k, tr):
    global ep_checked
    rew_none.append(tr.rewards is None)
    rec_zero.append(all(r.reward == 0.0 for r in tr.records))
    seen_dump.append(json.dumps([r.to_dict() for r in tr.records], default=str))
    seen_dump.append(json.dumps([[o.to_dict() for o in ob] for ob in tr.next_obs.observations], default=str))
    for r in tr.records:
        if r.action is None:
            continue
        le = r.extra.get("label_episode")
        mine = inner.episode_ordinal.get((str(r.env_id), str(r.episode_id)))
        ep_checked += 1
        if le is not None and mine is not None and int(le) != int(mine):
            ep_mismatch.append((r.env_id, r.episode_id, le, mine))
        st = (r.action.meta or {}).get("cam_eval") if hasattr(r.action, "meta") else None
        if st is not None and len(stamps) < 3:
            stamps.append({k_: st.get(k_) for k_ in ("psi_version", "versions", "policy_version")})

t0 = time.time()
life = run_lifetime(agent, env, labels=labels, label_store=sysm.label_store, learner=Wrap(), max_decisions=ROUNDS,
                    reset_on_done=True, transcript=os.path.join(run_dir, "transcript.jsonl"), on_step=on_step)
agent.act = act0
res["wall_s"] = time.time() - t0
res["rounds"] = life.steps
res["meta_updates_runtime"] = life.meta_updates
res["meta_driver_log"] = getattr(mt, "log", None)
res["meta_trainer_log"] = getattr(getattr(mt, "trainer", None), "log", None)
res["meta_gradient_log"] = meta_log
res["psi_hook_fires"] = {f"{a}|{b}": c for (a, b), c in fires.items()}
res["psi_hook_fires_by_phase"] = {}
for (a, b), c in fires.items():
    res["psi_hook_fires_by_phase"][a] = res["psi_hook_fires_by_phase"].get(a, 0) + c
res["steps"] = step_log
res["rewards_none_all"] = all(rew_none)
res["record_rewards_zero_all"] = all(rec_zero)
res["label_episode_checked"] = ep_checked
res["label_episode_mismatch"] = ep_mismatch[:10]
res["n_label_episode_mismatch"] = len(ep_mismatch)
res["stamps"] = stamps
res["released_labels"] = released
res["psi_version_final"] = rm.version
# canary scan: actor-side dumps, archive, traces, transcript
nums = [repr(C_NULL)[:10], repr(C_ORACLE)[:10], "1234.567", "98765.43"]
nums += [repr(float(l["u"]))[:9] for l in released if l["u"] is not None and abs(l["u"]) > 1e-6 and abs(l["u"] - 1) > 1e-6]
cans = [C_LABEL] + sorted(set(nums))
blob = "\n".join(seen_dump)
res["canaries"] = cans
res["actor_dump_bytes"] = len(blob)
res["actor_dump_hits"] = {c: blob.count(c) for c in cans if c in blob}
tp = os.path.join(run_dir, "transcript.jsonl")
tb = open(tp).read() if os.path.exists(tp) else ""
res["transcript_hits"] = {c: tb.count(c) for c in cans if c in tb}
res["archive_trace_scan"] = canary_scan(sysm, cans)
# positive control: label store holds C_LABEL
ls = [json.dumps(l.__dict__, default=str) for l in sysm.label_store.iterate()]
res["label_store_has_canary"] = any(C_LABEL in s for s in ls)
res["n_labels_in_store"] = len(ls)
res["reach_from_agent_after"] = reach(agent, tg)
res["run_dir"] = run_dir
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("rounds", "wall_s", "psi_hook_fires_by_phase", "rewards_none_all",
                                     "actor_dump_hits", "transcript_hits", "label_store_has_canary",
                                     "reach_from_agent_after", "model_reward_model_attached", "meta_trainer",
                                     "n_label_episode_mismatch", "psi_version_final")}, default=str, indent=1))
print("meta_driver_log", json.dumps(res["meta_driver_log"], default=str)[:2000])
print("meta_gradient_log", json.dumps(meta_log, default=str)[:3000])
