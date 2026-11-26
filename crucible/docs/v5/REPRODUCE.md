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

## Gate evidence (official runs and independent reviews, decision D-24)

```bash
# official run: clones the committed HEAD into a fresh directory, runs the gate's test files there, installs evidence,
# the run record and the JUnit under docs/v5/evidence/ (uncommitted work is never measured); then commit them
PYTHONPATH=src python -m crucible.acceptance.v5 run --gate B --agent ws-B
git add docs/v5/evidence && git commit -m "evidence: gate B"
# independent review (another agent and session): re-executes the tests in another fresh clone of the evidence commit
# and compares the metrics within the declared tolerance; writes docs/v5/reviews/; then commit it
PYTHONPATH=src python -m crucible.acceptance.v5 review --requirement B.1 --agent qa-1 --verdict approved --notes "..."
git add docs/v5/reviews && git commit -m "review: B.1"
# release check: everything above plus a fresh re-execution of every VERIFIED requirement
PYTHONPATH=src python -m crucible.acceptance.v5 --verify-all --reproduce --agent release-check
# after merging a workstream branch: rebuild the manifest and merge the workstream reports' file lists
PYTHONPATH=src python -m crucible.acceptance.v5 build-manifest [--decision D-<n>]
PYTHONPATH=src python -m crucible.acceptance.v5 merge-workstreams
PYTHONPATH=src python -m crucible.acceptance.v5 final-report            # docs/v5/FINAL_REPORT.md from the checker
```
Gate tests that need more than one CPU or the GPU run under the slot scheduler, e.g. Gate A with the GPU visible:
`python -m crucible.scale.slots --cpu 3 --gpu-mb 4000 --label "P: gate A" -- env CRUCIBLE_GATE_A_WORKERS=3 PYTHONPATH=src
python -m crucible.acceptance.v5 run --gate A --agent ws-P`.

## Clean-checkout reproduction (Gate X.3)

`crucible.acceptance.v5.qa.clean_checkout` clones the committed tree into a fresh directory and runs every line of the
block below there (each must exit 0), then re-executes every VERIFIED requirement's tests in fresh clones. The block is
machine-read; keep each command on one line.

```clean-checkout
python -m crucible.acceptance.v5 progress validate
python -m crucible.acceptance.v5 report
python -m pytest -q -p no:cacheprovider tests/test_kernel.py tests/test_invariants.py
python -m pytest -q -p no:cacheprovider tests/test_v5_acceptance_checker.py
```
Calibration tests read `$CRUCIBLE_DATA` (default /workspace/data/crucible, not committed): re-acquire the datasets
first (Calibration section below) for a clean-room run of Gate F.

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

## Calibration (workstream E: E1 and E2, one registry)

One registry serves all nine models and all four datasets (`docs/v5/components/E.md`); E1's fits are described in
`components/E1.md`, E2's in `components/E2.md`.

```bash
# data under $CRUCIBLE_DATA: KuaiRand-Pure and MIND-small about 0.45 GB, Criteo and the GH Archive extract about 1.3 GB
# (GH Archive is streamed: 9.8 GB downloaded and deleted hour by hour)
PYTHONPATH=src python -c "from crucible.calibration import acquire as a; [a(n) for n in ('kuairand_pure', 'mind_small', 'criteo_uplift_v2.1')]"
PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "E acquire gharchive" -- \
    env CRUCIBLE_ACQUIRE_WORKERS=4 PYTHONPATH=src python -c "from crucible.calibration import acquire; acquire('gharchive_2024-10-07_96h')"
# fits (the plans in docs/v5/calibration/plans/ were committed before the fits)
PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "E1 fit" -- \
    env OMP_NUM_THREADS=4 PYTHONPATH=src python -m crucible.calibration.fit_e1 --models content,watch,return,session,news
PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "E2 fit commercial" -- env PYTHONPATH=src python -m crucible.calibration.fit_commercial
PYTHONPATH=src python -m crucible.scale.slots --cpu 1 --label "E2 fit software" -- env PYTHONPATH=src python -m crucible.calibration.fit_software --threads 1
# tests and gates (load every checkpoint through the registry and recompute held-out metrics from the data)
PYTHONPATH=src python -m crucible.scale.slots --cpu 2 --label "E: calibration tests" -- \
    env OMP_NUM_THREADS=2 PYTHONPATH=src python -m pytest -q tests/test_v5_calibration_core.py \
    tests/test_v5_calibration_commercial.py tests/test_v5_calibration_software.py \
    tests/v5/test_gate_F.py tests/v5/test_gate_SP_5_4.py tests/v5/test_gate_SP_5_5.py
```

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
