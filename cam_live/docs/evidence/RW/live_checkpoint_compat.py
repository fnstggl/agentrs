"""Does the RW meta-learner load the live planning run's newest checkpoint (written by the pre-RW code)? Read-only:
the checkpoint's aux state is read from the live run, the outcome ledger from a copy in the scratch directory."""
import json, os, sys, torch
from cam_live.config import CamLiveConfig
from cam_live.nn.model import build_model
from cam_live.diagnostics.memstore import MemoryStore
from cam_live.reward.metalearner import MetaLearner
ck, ledger = sys.argv[1], sys.argv[2]
aux = torch.load(os.path.join(ck, "aux.pt"), map_location="cpu", weights_only=False)
meta_sd = aux["agent"]["meta"] if "agent" in aux else aux["aux_state"]["agent"]["meta"]
cfg = CamLiveConfig.debug()
ml = MetaLearner(build_model(cfg), MemoryStore(), cfg, ledger_path=ledger)
ml.load_state_dict(meta_sd)
print(json.dumps({"checkpoint": os.path.basename(ck), "keys_in_checkpoint": sorted(meta_sd),
                  "restored_fresh": ml.restored_fresh, "mix": ml.mixer.mix, "events": len(ml.events),
                  "ledger_entries": len(ml.ledger), "promotion_rule": ml.mc.promotion_rule,
                  "estimator": ml.mc.estimator, "utility": ml.mc.utility,
                  "meta_records": len(ml.records), "validity_units": ml.validity.n_units}, indent=1))
