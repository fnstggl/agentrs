"""Read-only check: the live planning run's newest checkpoint state (learner schedules + credit assigner state, JSON)
loads into CR's learner / assigner (debug-scale objects; the JSON parts are model-independent)."""
import copy, glob, json, os, sys, tempfile
from cam_live.config import CamLiveConfig
from cam_live.credit.labkit import build_lab
from cam_live.credit.learner import ContinualLearner, LearnerSettings
L = sorted(glob.glob("/workspace/camlive/runs/cam_live/live/discord-100-planning-*/checkpoints/ckpt_*/state.json"))[-1]
d = json.load(open(L))
out = {"checkpoint": L}
for split in ("record", "segment"):
    tmp = tempfile.mkdtemp(dir=sys.argv[1])
    cfg = CamLiveConfig.debug()
    lab = build_lab(os.path.join(tmp, "s.sqlite"), cfg, world=False, policy=False, credit=True,
                    settings=LearnerSettings.from_cfg(cfg, val_split=split, holdout_experiments=False))
    try:
        rep = lab.learner.load_state_dict(copy.deepcopy(d["schedules"]))
        lab.assigner.load_state_dict(copy.deepcopy(d["extra"]["credit"]))
        out[split] = {"load_report": rep, "default": lab.assigner.default, "selector_units": len(lab.assigner.selector.units),
                      "evidence": len(lab.assigner.selector.evidence), "revised": len(lab.assigner._revised),
                      "trained_objectives": len(lab.learner.trained_objectives),
                      "trained_units_superset": set(lab.learner.trained_objectives) <= lab.learner.trained_units,
                      "segment_split_from_seq": lab.learner.segment_split_from_seq,
                      "state_roundtrip_keys": sorted(set(lab.learner.state_dict()) - set(d["schedules"])),
                      "assigner_new_keys": sorted(set(lab.assigner.state_dict()) - set(d["extra"]["credit"]))}
    finally:
        lab.close()
print(json.dumps(out, indent=1, default=str))
