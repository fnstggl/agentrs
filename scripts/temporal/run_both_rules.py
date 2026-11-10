"""Run an experiment script unchanged, scoring every evaluation with both held-out scoring rules.

``evaluate_temporal`` results get the current rule; the rule used before the audit of
``event_truth`` (decided from the label alone; see ``audit_event_truth.py``) is added under the key
``legacy_rule`` (also inside ``summarize`` output), from the same predictions. Results are written
with the prefix ``audit_event_truth_`` so the original result files are left untouched. Training is
unchanged (the temporal model's training never used ``event_truth``).

    python scripts/temporal/run_both_rules.py exp_b_scheduling.py --steps 200 --forecaster-steps 150

Scripts that call ``event_truth`` directly (Experiment E) are run once per rule instead:

    python scripts/temporal/run_both_rules.py --rule old exp_e_causal.py --seed 20
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from audit_event_truth import RULES  # noqa: E402

import agentrs.temporal.temporal_evaluation as te  # noqa: E402

_evaluate, _summarize, _save = te.evaluate_temporal, te.summarize, common.save_json


def evaluate_both(predict, examples, horizons, events, counts=()):
    examples = list(examples)
    cache = {}

    def once(exs, hs):
        key = tuple(hs)
        if key not in cache:
            cache[key] = predict(exs, hs)
        return cache[key]

    te.event_truth = RULES["old"]
    old = _evaluate(once, examples, horizons, events, counts)
    te.event_truth = RULES["new"]
    new = _evaluate(once, examples, horizons, events, counts)
    if new:
        new["legacy_rule"] = old
    return new


def summarize_both(res, events):
    s = _summarize(res, events)
    if "legacy_rule" in res:
        s["legacy_rule"] = _summarize(res["legacy_rule"], events)
    return s


def main():
    args = sys.argv[1:]
    if args[0] == "--rule":
        rule, args = args[1], args[2:]
        te.event_truth = RULES[rule]  # picked up by scripts that import it by name
        prefix = f"audit_event_truth_{rule}_"
    else:
        te.evaluate_temporal, te.summarize = evaluate_both, summarize_both
        prefix = "audit_event_truth_"
    common.save_json = lambda name, obj: _save(prefix + name, obj)
    script = str(HERE / args[0])
    sys.argv = [script, *args[1:]]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
