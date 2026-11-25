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

## Substrate (workstream A)

```bash
PYTHONPATH=src python -m pytest -q tests/test_v5_substrate_*.py                  # unit and integration tests
PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_2_3.py tests/v5/test_gate_SP_2_1.py
PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --gpu-mb 0 --label "substrate bench" -- \
    env PYTHONPATH=src python -m crucible.substrate.bench --rows 1000000 --table-rows 10000000
# the 4.0 byte-identity digests in tests/v5_kernel_scenarios.py were recorded on the untouched 4.0 tree:
#   git archive c07098b crucible/src | tar -x -C /tmp/k40
#   PYTHONPATH=/tmp/k40/crucible/src python tests/v5_kernel_scenarios.py
```

## Calibration E2: commercial response and software adoption

```bash
# data (about 1.5 GB under $CRUCIBLE_DATA; GH Archive is streamed: 9.8 GB downloaded and deleted hour by hour)
PYTHONPATH=src python -c "from crucible.calibration import registry_e2 as r; r.acquire('criteo_uplift_v2.1')"
PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "E2 acquire gharchive" -- \
    env CRUCIBLE_ACQUIRE_WORKERS=4 PYTHONPATH=src python -c "from crucible.calibration import registry_e2 as r; r.acquire('gharchive_2024-10-07_96h')"
# fits (the plans in docs/v5/calibration/plans/ were committed before the fits)
PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "E2 fit commercial" -- env PYTHONPATH=src python -m crucible.calibration.fit_commercial
PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --label "E2 fit software" -- env PYTHONPATH=src python -m crucible.calibration.fit_software --threads 1
# tests (load the checkpoints through the registry and recompute held-out metrics from the data)
PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --label "E2 tests" -- \
    env PYTHONPATH=src python -m pytest -q tests/test_v5_calibration_commercial.py tests/test_v5_calibration_software.py
```

## Learner interface (workstream L)

PYTHONPATH=src python -m crucible.experience tiers                       # which tiers this checkout has
PYTHONPATH=src python -m crucible.experience run --tier 1 --name mech:scm_additive --seed 0 --steps 120 \
    --reward-mode observation_only --store /tmp/exp --checkpoint /tmp/ref.pt --out /tmp/run.json   # reproducible
CRUCIBLE_V5_AUTHOR=ws-L PYTHONPATH=src python -m pytest -q tests/v5/test_gate_R.py tests/v5/test_gate_SP_11_*.py
PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "ws-L: gate S" -- \
    env CRUCIBLE_V5_AUTHOR=ws-L python -m pytest -q tests/v5/test_gate_S.py   # S.7 starts four worker processes
```
See docs/v5/LEARNER_GUIDE.md and docs/v5/components/L.md.
