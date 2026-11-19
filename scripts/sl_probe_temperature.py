"""Zero-shot cam/bizdev validation under a policy temperature T (offline, frozen weights): mean policy entropy (nats,
normalised by log #options) and score. Diagnoses whether an experienced init is merely overconfident on unfamiliar actions."""
import json, math, sys
from pathlib import Path
import torch
from agentrs.sl import worlds as W
from agentrs.sl.model import SLConfig, SLCore
from agentrs.sl.runner import Runner
from agentrs.sl.world import RewardContract, RewardSpec

run, ckpt, fam = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
temps = [float(x) for x in sys.argv[4].split(",")]
st = json.loads((run / "state.json").read_text()); c = st["config"]
m = SLCore(SLConfig(**c["model"])); m.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu")); m.eval()
W.resolve(fam); tasks = W.eval_tasks(fam, "val", 48)
Cat = torch.distributions.Categorical
for T in temps:
    ent = []
    class TCat(Cat):
        def __init__(self, logits=None, **k):
            super().__init__(logits=logits / T, **k)
            valid = (logits > -1e8).sum(-1).clamp(min=2).float()
            ent.extend((self.entropy() / valid.log()).tolist())
    torch.distributions.Categorical = TCat
    r = Runner(m, {fam: 1.0}, W.make_world, W.sample_task, 48, trial_episodes=int(c["trial_episodes"]), split="val", seed=12345,
               mode="full", contract=RewardContract(RewardSpec(**c["reward"])), device=torch.device("cpu"),
               max_episode_steps=c["max_episode_steps"], fixed_tasks=list(tasks), dt_ref=dict(st.get("dt_ref", {})))
    with torch.no_grad():
        eps = r.run_trials()
    r.close()
    torch.distributions.Categorical = Cat
    K = int(c["trial_episodes"])
    first = [e["score"] for e in eps if e["episode"] == 0]; last = [e["score"] for e in eps if e["episode"] == K - 1]
    print(f"{run.name} {ckpt.name} {fam} T={T:g}: entropy {sum(ent)/len(ent):.3f}  score ep1 {sum(first)/len(first):.3f}  "
          f"epK {sum(last)/len(last):.3f}  success {sum(e['success'] for e in eps)/len(eps):.3f}", flush=True)
