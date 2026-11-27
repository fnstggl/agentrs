"""INT2-8 probe: the support batch a MetaLearner update event cached vs the batch rebuilt from the store later (what a
restored agent uses). Usage: python inner_batch_rebuild.py <scratch dir>"""
import sys, os, tempfile, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tests", "integration"))
from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task
from int_helpers import DriverState, lab_config, agent_config, drive
root = tempfile.mkdtemp(dir=sys.argv[1])
clock = SimClock(); world = LabWorld(lab_config(51), clock); verifier = LabVerifier(world)
agent = CamLiveAgent(agent_config(seed=5), os.path.join(root, "a"), clock=clock, verifier=verifier)
agent.set_objective(lab_task(world, "obj_saveload"))
state = DriverState()
drive(agent, world, clock, verifier, state, 12)
m = agent.meta
print("events", len(m.events))
for e in m.events:
    if e.ib is None: print(e.event_id, "no cached batch"); continue
    recs = [r for r in (agent.store.get(i) for i in e.exp_ids) if r is not None]
    nb = m.inner.build_batch(agent.store, recs)
    diffs = []
    for f in ("b", "e_a", "signals", "r_boot", "r_goal", "z_pred", "gamma"):
        x, y = getattr(e.ib, f), getattr(nb, f)
        if (x is None) != (y is None) or (x is not None and (x.shape != y.shape or not torch.equal(x, y))):
            diffs.append(f)
    ns = [k for k in set(e.ib.norm_state) | set(nb.norm_state)
          if k not in e.ib.norm_state or k not in nb.norm_state or not torch.equal(e.ib.norm_state[k], nb.norm_state[k])]
    print(e.event_id, "rows", len(e.exp_ids), "differs:", diffs, "norm_state differs:", ns[:4],
          "goal_source", e.ib.goal_source, "->", nb.goal_source)
agent.close()
