# Reproducing Crucible 5.0

Run everything from `crucible/` with `PYTHONPATH=src` (the package is not installed in the shared environment).
Commands that need more than one CPU thread or any GPU run through the slot scheduler.

## Status

```bash
PYTHONPATH=src python -m crucible.acceptance.v5 --verify-all      # exit 0 only when every requirement is VERIFIED
PYTHONPATH=src python -m crucible.acceptance.v5 report --gate B   # per-requirement status and problems
```

## Regression baseline (Gate A)

```bash
python -m pytest -q -p no:cacheprovider --junitxml=docs/v5/evidence/gate_A/regression_g0.xml <group 0 files>
```
The baseline used three groups split by file size (see `docs/v5/evidence/gate_A/regression_baseline.json`): 1,076
tests, 1,074 passed, 2 skipped, 694 s wall with three groups in parallel, one thread each, GPU hidden.

## Gate evidence

```bash
PYTHONPATH=src python -m crucible.acceptance.v5 run --gate B           # runs tests/v5/test_gate_B.py, records evidence
PYTHONPATH=src python -m crucible.acceptance.v5 review --requirement B.1 --reviewer ws-P --verdict approved --notes "..."
```

## Compute scheduler

```bash
PYTHONPATH=src python -m crucible.scale.slots --status
PYTHONPATH=src python -m crucible.scale.slots --cpu 2 --gpu-mb 4000 --label "what" -- <command>
```

Further commands are added here by each workstream as its components land.

## Learner interface (workstream L)

```bash
PYTHONPATH=src python -m crucible.experience tiers                       # which tiers this checkout has
PYTHONPATH=src python -m crucible.experience run --tier 1 --name mech:scm_additive --seed 0 --steps 120 \
    --reward-mode observation_only --store /tmp/exp --checkpoint /tmp/ref.pt --out /tmp/run.json   # reproducible
CRUCIBLE_V5_AUTHOR=ws-L PYTHONPATH=src python -m pytest -q tests/v5/test_gate_R.py tests/v5/test_gate_SP_11_*.py
PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "ws-L: gate S" -- \
    env CRUCIBLE_V5_AUTHOR=ws-L python -m pytest -q tests/v5/test_gate_S.py   # S.7 starts four worker processes
```
See docs/v5/LEARNER_GUIDE.md and docs/v5/components/L.md.
