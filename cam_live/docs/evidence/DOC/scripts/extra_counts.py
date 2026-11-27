"""Frozen randomised-prior buffers of the world-model ensemble (not parameters, not counted by count_parameters) and
the size of the meta-learner's adaptable subset, for the default (50m: d_model 768, F 128) configuration."""
import json, sys
from cam_live.config import CamLiveConfig
from cam_live.predict.heads import PredictConfig
from cam_live.predict.world import WorldConfig, WorldHeads
out = {}
for label, cfg in (("default", CamLiveConfig.load()), ("debug", CamLiveConfig.debug())):
    from cam_live.nn.presets import resolve_core
    core = resolve_core(cfg.core)
    p = PredictConfig.from_cfg(cfg)
    w = WorldHeads(WorldConfig(d_model=core.d_model, n_features=core.n_features if p.use_phi else 0, hidden=p.hidden,
                               n_members=p.n_members, n_hazard_bins=p.n_hazard_bins, max_categories=p.max_categories,
                               prior_scale=p.prior_scale, prior_hidden=max(16, p.hidden // 3), seed=p.seed))
    out[label] = {"world_trainable_params": sum(x.numel() for x in w.parameters()),
                  "world_frozen_prior_buffer_numbers": sum(b.numel() for n, b in w.named_buffers() if n.startswith("prior.")),
                  "predict_config": {"hidden": p.hidden, "n_members": p.n_members, "transition_hidden": p.transition_hidden,
                                     "transition_members": p.transition_members, "n_hazard_bins": p.n_hazard_bins,
                                     "max_categories": p.max_categories}}
print(json.dumps(out, indent=1))
