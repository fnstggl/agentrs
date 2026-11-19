"""Pre-registration check for H12: per-decision pointer entropy and score on cam/bizdev validation tasks for scratch,
experienced sources, and experienced sources with the action heads reset (init_reset). Run with the frozen v16 src on
PYTHONPATH. Gate disabled in effect (fam_n set huge so g = 1): this measures the raw preferences the network carries."""
import json, math, sys
from pathlib import Path
import torch
from agentrs.sl import worlds as W
from agentrs.sl.model import SLConfig, SLCore
from agentrs.sl.runner import Runner
from agentrs.sl.world import RewardContract, RewardSpec

RESET = ["query.", "slot.", "hi.", "term."]
ref = json.loads(Path("/workspace/agentrs/runs/sl/T_B16_scratch_s1/state.json").read_text())
c = ref["config"]
fam = "cam/bizdev"
W.resolve(fam)
tasks = W.eval_tasks(fam, "val", 48)


def build(src, reset, seed=1):
    torch.manual_seed(seed)
    m = SLCore(SLConfig(**c["model"]))
    if src:
        sd = torch.load(Path(src) / "weights.pt", map_location="cpu")
        own = m.state_dict()
        ld = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape and not any(k.startswith(p) for p in reset)}
        m.load_state_dict(ld, strict=False)
    m.fam_n.fill_(1e15)  # g = 1: raw preferences
    m.eval()
    return m


def probe(name, m):
    ents, hents = [], []
    raw_gate = SLCore.gate_logits

    def rec(raw, msk, g):
        lg = raw.masked_fill(~msk, -1e9)
        p = torch.softmax(lg, -1)
        e = -(p * torch.log_softmax(lg, -1)).masked_fill(~msk, 0).sum(-1)
        n = msk.sum(-1).float()
        for ei, ni in zip(e.tolist(), n.tolist()):
            if ni > 1:
                ents.append((ei, math.log(ni)))
        return raw_gate(raw, msk, g)
    m.gate_logits = rec
    r = Runner(m, {fam: 1.0}, W.make_world, W.sample_task, 48, trial_episodes=int(c["trial_episodes"]), split="val", seed=12345,
               mode="full", contract=RewardContract(RewardSpec(**c["reward"])), device=torch.device("cpu"),
               max_episode_steps=c["max_episode_steps"], fixed_tasks=list(tasks), dt_ref=dict(ref.get("dt_ref", {})))
    with torch.no_grad():
        eps = r.run_trials()
    r.close()
    h = sum(e for e, _ in ents) / len(ents)
    hmax = sum(x for _, x in ents) / len(ents)
    print(f"{name:28s} decisions {len(ents):6d}  entropy {h:.3f}  max {hmax:.3f}  ratio {h / hmax:.3f}  "
          f"score {sum(e['score'] for e in eps) / len(eps):.3f}  mean steps {sum(e['steps'] for e in eps) / len(eps):.1f}", flush=True)


P = "/workspace/agentrs/runs/sl/{}/ckpt/life_1000000"
arms = [("scratch", None, []), ("P_N_s1", P.format("P_N_s1"), []), ("P_N_s1 reset heads", P.format("P_N_s1"), RESET),
        ("P_NP_s1", P.format("P_NP_s1"), []), ("P_NP_s1 reset heads", P.format("P_NP_s1"), RESET),
        ("P_N_s2", P.format("P_N_s2"), []), ("P_NP_s2", P.format("P_NP_s2"), [])]
for name, src, reset in arms:
    probe(name, build(src, reset))
