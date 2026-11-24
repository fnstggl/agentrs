"""Experiment 2 worlds: hidden mechanisms built from Crucible's own SCM motif generator, never used for training.

Crucible's families with confounders, feedback and regime switches are sealed test families and stay untouched. These
two families compose the same generator's motifs in new combinations, so every property the specification lists for
active causal discovery is present:

* `x2_confound_gate`: hidden confounders (an exogenous driver moves a gauge and the outcome; the gauge causes nothing, so
  passive correlation misleads), feedback through a lag, gates (an effect exists only while a third node is above a
  hidden level), product interactions, delays, distractor levers (uninformative interventions) and costly levers.
* `x2_regime_mix`: gates and delays plus a regime switch at a hidden tick (some causal weights change strength or sign,
  so the best settings change), interactions, thresholds, distractors.

Registered with split "val" (public validation range of seeds); CAM 7.0 is never trained on them. Importing this module
registers them in the running process only; Crucible's files and historical families are unchanged.
"""
from __future__ import annotations

from crucible.mechanisms.base import MECH_FAMILIES, mech_family
from crucible.mechanisms.scm import _COMMON

FAMILIES = ("x2_confound_gate", "x2_regime_mix")

if "x2_confound_gate" not in MECH_FAMILIES:
    @mech_family("x2_confound_gate", "scm", "val", "structural",
                 "CAM 7.0 Experiment 2: hidden confounders, lagged feedback, gates, interactions and delays with "
                 "distractor levers. Passive correlation misleads; only well-chosen interventions identify what pays.",
                 motifs=_COMMON + ("confounder", "feedback", "gate", "interaction", "delay", "lagged_gauge"))
    def fam_x2_confound_gate(rng, o):
        return {"required": ["confounder", "gate"], "pool": {"feedback": 1, "delay": 1, "interaction": 1, "direct": 2,
                                                             "chain": 1, "fork": 1},
                "n_blocks": [4, 5], "gauge_lag_p": 0.2, "distractors": [1, 3], "horizon": 144,
                "seats": int(o.get("seats", 1))}

if "x2_regime_mix" not in MECH_FAMILIES:
    @mech_family("x2_regime_mix", "scm", "val", "structural",
                 "CAM 7.0 Experiment 2: gates and delays with a regime switch at a hidden tick (the best settings change), "
                 "interactions, thresholds and distractor levers.",
                 motifs=_COMMON + ("gate", "delay", "interaction", "threshold", "regime"))
    def fam_x2_regime_mix(rng, o):
        return {"required": ["gate", "delay"], "pool": {"interaction": 1, "threshold": 1, "direct": 2, "chain": 1},
                "n_blocks": [3, 5], "regime": True, "distractors": [1, 3], "horizon": 192,
                "seats": int(o.get("seats", 1))}
