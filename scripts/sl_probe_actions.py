"""What does a checkpoint do on cam/bizdev validation tasks? Counts of first action tokens, end reasons, violations."""
import collections, json, sys
from pathlib import Path
import torch
from agentrs.sl import worlds as W
from agentrs.sl.model import SLConfig, SLCore
from agentrs.sl.runner import Runner
from agentrs.sl.world import RewardContract, RewardSpec

run, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
st = json.loads((run / "state.json").read_text()); c = st["config"]
m = SLCore(SLConfig(**c["model"])); m.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu")); m.eval()
acts = collections.Counter(); fam = sys.argv[3] if len(sys.argv) > 3 else "cam/bizdev"
W.resolve(fam)
def mk(f, *a, **k):
    w = W.make_world(f, *a, **k); ex = w.execute
    def execute(agent, act):
        acts[act[0] if isinstance(act, list) else act] += 1
        return ex(agent, act)
    w.execute = execute
    return w
tasks = W.eval_tasks(fam, "val", 48)
r = Runner(m, {fam: 1.0}, mk, W.sample_task, 48, trial_episodes=int(c["trial_episodes"]), split="val", seed=12345, mode="full",
           contract=RewardContract(RewardSpec(**c["reward"])), device=torch.device("cpu"), max_episode_steps=c["max_episode_steps"],
           fixed_tasks=list(tasks), dt_ref=dict(st.get("dt_ref", {})))
with torch.no_grad():
    eps = r.run_trials()
r.close()
tot = sum(acts.values())
print(run.name, ckpt.name, "episodes", len(eps), "mean steps", round(sum(e["steps"] for e in eps) / len(eps), 1),
      "score", round(sum(e["score"] for e in eps) / len(eps), 3), "violations", collections.Counter(e["violated"] for e in eps).most_common(4))
print("  actions:", [(a, round(n / tot, 3)) for a, n in acts.most_common(10)])
