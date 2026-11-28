"""RW-1 / RW-2 on a patched copy: an agent checkpoint written without the validity state (as the pre-RW code wrote
it) restores, the restore report names what started fresh, and status() carries the validity summary."""
import json, os, sys, tempfile
from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.validation import staging_config
d = tempfile.mkdtemp(dir=os.environ["RW_TMP"])
cfg = staging_config(None, False, 0)
a = CamLiveAgent(cfg, os.path.join(d, "a"), clock=SimClock())
orig = a.meta.state_dict
a.meta.state_dict = lambda: {k: v for k, v in orig().items() if k not in ("validity", "validity_registered")}
p = os.path.join(d, "ckpt")
a.save(p)
b = CamLiveAgent(cfg, os.path.join(d, "b"), clock=SimClock())
b.load(p)
out = {"restore_report_meta_restored_fresh": b.restore_report.get("meta_restored_fresh"),
       "status_reward_validity": b.status().get("reward_validity")}
print(json.dumps(out, default=str))
assert out["restore_report_meta_restored_fresh"] == ["validity (not in the checkpoint)"]
assert out["status_reward_validity"]["pairs"] == 0
print("RW-1 / RW-2 hold on the patched copy")
