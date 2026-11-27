import sys; sys.path.insert(0, "/workspace/agentrs/crucible/src")
from superlearner.cam_unified.runtime import run_lifetime, ActionNotAuthorised
from superlearner.cam_unified.decision.condition_g import TwoGoalEnv
from superlearner.cam_unified.interfaces import Action
from superlearner.cam_unified.config import make_preset
class Dumb:
    def __init__(self, cfg=None):
        if cfg is not None: self.config = cfg
        self.n = 0
    def act(self, obs, tools):
        self.n += 1
        return [Action(tool=tools[b][0].name, args={}) for b in range(len(tools))]
    def observe(self, *a, **k): pass
for cfg in (None, make_preset("dom_dev")):
    env = TwoGoalEnv(2, seed=0); env.reset([1, 2])
    ag = Dumb(cfg)
    try:
        life = run_lifetime(ag, env, max_decisions=2, update_every=0)
        print("cfg" if cfg else "no-cfg", "EXECUTED unstamped actions:", ag.n, "authorisation:", life.authorisation)
    except ActionNotAuthorised as e:
        print("cfg" if cfg else "no-cfg", "REFUSED:", str(e)[:150])
    except Exception as e:
        print("cfg" if cfg else "no-cfg", "other", type(e).__name__, str(e)[:300])
