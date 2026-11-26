# E: the calibration package and its one registry

Workstream E's two halves were built in parallel: E1 (content consumption and browsing, `components/E1.md`) and E2
(commercial response and software adoption, `components/E2.md`). Each shipped its own section-5 registry. They are
now one registry. `crucible.calibration.load("<name>@<version>")` loads all nine fitted models, and
`crucible.calibration.datasets()` serves all four datasets. The gates (F, SP-5.4, SP-5.5) measure every model by
the same rules, whichever toolchain fitted it. No model was refitted and no card was rewritten: every fitted
parameter, held-out metric and checkpoint hash is unchanged.

## Wave 3 (continuation): what changed and where

| piece | files | what it does |
|---|---|---|
| advertising redesign (audit E skeptic S5) | `fit_e5.py`, `predictors_e5.py`, plan `plans/E5_ad_exposure_plan.json`, model `models/ad_exposure_response/`, report `reports/ad_exposure_response.json`, runtime `AdDecision` (`ad_response@2`) | Criteo eligibility as the instrument for display; joint three-arm likelihood per stratum with profiled nuisances; kept one log-odds shift 1.39 (CI 1.22-1.61) of the individual's own organic visit probability; attention per display (relevance, repetition) through a registered attention model; no negative effects; incremental visits convert 2.76x; RUNTIME.md "Advertising response" |
| in-world validation harness | `inworld.py` | model-only driver (runtime models forward-simulated from KuaiRand held-out users' history) and in-world driver (decisions a world recorded with fitted provenance); four reports (activity distribution, retention by prior activity, heterogeneity, long-horizon drift) with prespecified tolerances; random-exposure and ad-holdout experiment analyses; SP-5.0 provenance metrics; static card-scope check |
| SP-5.1 / SP-5.3 measurements | `audit_sp5.py` | licence records, raw-identifier check of every GH Archive id value on disk, uncertainty, holdout kinds, identification labels; rule comparisons, selection with stability and cost, coverage of the six Part 5.3 behaviours |
| fidelity register | `FIDELITY_REGISTER.md`, `fidelity.py` | repaired after the wave-2 union merge; decision-recording modules and provenance fidelity ids are important mechanisms |
| out-of-scope uses removed | `economy/work.py`, `domains/common.py` (minimal edits) | no Criteo type table for search clicks; no news model with rank mapping for lesson search |

Gate tests: `tests/v5/test_gate_F.py` (all ten pass on this tree, F.3 includes the E5 reproducer),
`test_gate_SP_5_1.py`, `test_gate_SP_5_3.py`, `test_gate_SP_5_4.py`, `test_gate_SP_5_5.py` (pass),
`test_gate_SP_5_0.py` and `test_gate_SP_5_5b.py` (slow; fail until the population wires the runtime models and
`civ.generate` builds again - they never skip). Unit tests: `tests/test_v5_calibration_ad_exposure.py` (estimator on
synthetic trials with a known shift and with none, predictor, runtime `ad_response@2`, harness).

## World code: one entry point

```python
from crucible.calibration import load, datasets
m = load("content_consumption@1")            # verifies the SHA-256 of every checkpoint file; NumPy only, no pickle
m.predict(X)                                 # X: [n, len(m.features)] ordered as m.features
m.domain, m.identification, m.datasets       # canonical vocabulary (registry.DOMAINS, registry.IDENTIFICATION)
m.feature_map                                # how a simulator quantity maps onto each input
m.metrics, m.baseline_metrics, m.limits, m.provenance

ad = load("commercial_response@1")           # a CommercialResponseModel (a FittedModel subclass)
types = ad.sample_types(rng, n); r = ad.respond(rng, types, treated, ad.transport(control_visit_rate=0.05))
growth = load("software_repo_growth@1")      # a RepoGrowthModel
growth.sample(X, rng)                        # same argument order as every distributional model
```

Errors are `CalibrationError`s. `ModelNotFoundError` is also a `KeyError` (unknown name or version).
`ChecksumError` is also an `IOError` (a checkpoint file is missing or altered). `CardError` is also a `ValueError`
(the registry refuses the card: unknown schema, kind or loader, or a loader outside `crucible.calibration`).

## The registry (`src/crucible/calibration/registry.py`)

A model is a directory `docs/v5/calibration/models/<name>/` that holds `model_card.json` and the checkpoint files
the card hashes. `list_models()` returns every such directory. Two card schemas are registered:

| schema | written by | how the model is built | checkpoint | models |
|---|---|---|---|---|
| `crucible.v5.model_card/1` (`SCHEMA`) | `registry.save_model` via `fitlib` (E1) | ``kind``: a portable predictor in `predictors.KINDS` (optional ``module`` to import first) | ``files`` {file: sha256}, ``checkpoint`` | the six E1 models |
| `crucible.v5.calibration.model_card/1` (`LOADER_SCHEMA`) | `fitmeta_e2` (E2) | ``loader``: `crucible.calibration.<module>:<Class>`, a `FittedModel` subclass that receives the checkpoint's arrays | ``checkpoint`` {file, sha256, bytes} | the three E2 models |

`card_view(card)` reads either schema into one description. The audit, the fidelity register and `FittedModel` use
only this view:

* identity: ``schema``, ``name``, ``version``, ``id``;
* vocabulary: ``domain`` and ``identification``, mapped onto the canonical terms, with the card's own term kept as
  ``domain_label`` / ``identification_label``;
* data and checkpoint: ``datasets`` and ``files``;
* documents: ``report``, ``limits``, ``mechanisms`` (E1 ``mechanisms``, E2 ``fidelity_mechanisms``) and
  ``feature_map`` (E1 ``feature_map``; E2 ``simulator_mapping`` per input, including grouped keys such as
  ``actor_*``);
* ``plans``: the evaluation plan, then each amendment (E2 ``plan_amendments``), each with
  {path, sha256, commit, role};
* ``fit``: commit, and start time where the card records one;
* ``selection``: the selection record (below).

The selection record holds ``rule``, ``primary_metric``, ``lower_is_better``, ``validation`` and ``test`` (the
primary metric of each candidate), ``baselines``, ``candidates``, ``kept``, ``kept_is_baseline``, ``complexity``,
the paired-bootstrap ``comparisons`` and ``worse_model_kept``. For E2 cards these come from the card's
``candidates`` table. The primary metric is the one named by the card's bootstrap comparisons (``"log_loss
(test)"``). A composite primary metric that the plan defines in words is stored under a key that starts with
``primary_`` (``primary_pooled_logloss_visit_plus_conversion``).

The two toolchains used different labels for two vocabulary terms, which the view maps as follows:

| card label (E2) | canonical | why |
|---|---|---|
| domain `commercial_intervention_response` | `advertising_response` | MASTER_SPEC 5.3 domain list |
| domain `software_collaboration_and_adoption` | `software_adoption` | same |
| identification `randomized_experiment` | `randomized_controlled_trial` | Criteo randomised eligibility for targeting. `randomized_exposure` is reserved for randomly inserted content (KuaiRand) |

`card_problems(card)` lists what a card of either schema lacks: the schema's required fields
(`REQUIRED_CARD_FIELDS`), a canonical domain and identification, a dataset, hashed checkpoint files, limits, a plan
and a primary metric. All nine committed cards return `[]`.

### Registered models

| model | card schema | domain (card label) | identification (card label) | dataset | primary metric: kept vs best baseline (test) |
|---|---|---|---|---|---|
| `commercial_response@1` | loader | advertising_response (commercial_intervention_response) | randomized_controlled_trial (randomized_experiment) | criteo_uplift_v2.1 | primary_pooled_logloss_visit_plus_conversion 0.1143 (t_gbdt_reg) vs 0.2091 (constant_effect) |
| `content_consumption@1` | kind | content_selection | randomized_exposure | kuairand_pure | log_loss 0.4426 (R_gbdt) vs 0.4779 (B1_popularity) |
| `content_consumption_feed@1` | kind | content_selection | observational | kuairand_pure | log_loss 0.6238 (O_gbdt) vs 0.6650 (B1_popularity) |
| `news_click@1` | kind | content_selection | observational | mind_small | log_loss 0.1465 (M2_gbdt) vs 0.1539 (B1_popularity) |
| `return_daily@1` | kind | browsing_frequency | observational | kuairand_pure | log_loss 0.5617 (M2_gbdt) vs 0.6103 (B1_own_rate) |
| `session_rate@1` | kind | browsing_frequency | observational | kuairand_pure | poisson_deviance 3.3015 (M1_gamma_poisson) vs 3.7246 (B0_global_mean) |
| `software_deep_engagement@1` | loader | software_adoption (software_collaboration_and_adoption) | observational | gharchive_2024-10-07_96h | log_loss 0.1269 (gbdt) vs 0.1439 (constant_rate) |
| `software_repo_growth@1` | loader | software_adoption (software_collaboration_and_adoption) | observational | gharchive_2024-10-07_96h | mean_poisson_deviance 0.5361 (gbdt_poisson_reg) vs 0.7318 (proportional) |
| `watch_duration@1` | kind | watch_duration | observational | kuairand_pure | pinball 0.1294 (M2_quantile_mlp) vs 0.1378 (B1_duration_bins) |

## Datasets (`src/crucible/calibration/datasets.py`)

`datasets()`, `acquire(name)`, `verify(name)` and `validate_schema(name)` serve datasets from three sources:

* the recipes in `datasets.py`: `kuairand_pure` and `mind_small`, plain checksummed downloads with safe archive
  extraction;
* committed manifests `docs/v5/calibration/datasets/<name>.json` (schema `crucible.v5.calibration.dataset/1`):
  `criteo_uplift_v2.1` and `gharchive_2024-10-07_96h`. `_from_manifest` keeps every manifest field and adds the
  entry fields the registry uses: ``files[].path``/``size``, ``license.name``/``text_location``, ``extracted``,
  ``schema`` and ``manifest``;
* `register(name, entry)` from any module `crucible.calibration.datasets_<x>`, for future workstreams.

A dataset that is not a plain download names an ``acquirer`` module. That module provides
`acquire_dataset(entry, dest)`, `verify_dataset(entry, dest)` and `validate_dataset(entry, dest)`:

* Criteo (`criteo.py`) verifies the downloaded CSV and the converted arrays, and checks the conversion record's
  schema checks and the array shapes.
* GH Archive (`gharchive.py`) deletes each raw hour after extraction. So each of the 96 hours is verified by two
  values against the manifest: the raw SHA-256 recorded while the hour was streamed, and the content digest of the
  extracted table. The validation checks every hour's columns, dtypes and event count.

The manifests gained the provenance fields the registry requires of every dataset: publisher, citation, platform,
period and retrieval date. They also gained a title, a description and the acquirer. `criteo.manifest()` and
`gharchive.manifest()` now emit the same fields. No checksum or file entry changed. Criteo's collection period is
not disclosed by the publisher, and the manifest says so ("released with Diemert et al., AdKDD 2018").

## The gate audit (`src/crucible/calibration/audit.py`)

* `examine(name)`, `all_records()` and `gate_f_metrics()` (Gate F) read cards through `card_view`. They read
  reports through `report_view`. E1 reports already have the audit's layout. An E2 report
  (`crucible.v5.calibration.report/1`) is re-indexed into it:
  * ``heldout_splits``: every split except validation;
  * ``splits[s].rows``: the kept candidate's n;
  * ``baseline_comparison``: the card's kept and baseline test values, ``difference`` = kept - baseline, and the
    report's paired-bootstrap ``ci95``;
  * ``calibration``: the kept candidate's test calibration block;
  * ``extrapolation_limits``: the list joined into text.

  Only ``difference`` is computed; every other number is the report's or the card's own.
* Calibration counts both scalar measures and reliability tables. The scalars are ECE, Brier, PIT KS, coverage and
  count calibration error, per outcome where there are several (``visit_ece``). The tables are bins of mean
  predicted vs mean observed (``calibration_by_decile``).
* `plan_check(card)` (SP-5.5) checks the plan and each amendment:
  * it is in git at the recorded commit with the recorded SHA-256, and unchanged in the working tree;
  * it is in HEAD's history and contained in the fit commit;
  * it was committed before the commit that first added the card, and before the fit started where the card records
    the start (E2's `fit.started_at`).

  It then checks the plan's content, in either layout:
  * fitlib plans (``models`` keyed by name): primary metric, secondary metrics reported, candidate set;
  * E2 plans (``models`` list): the plan names the model, its ``(primary)`` metric or composite
    ``primary_metric``, every planned candidate was evaluated, the baselines are the plan's, and any candidate absent
    from the plan is named by an amendment that lists the plan in ``amends``. This is how `t_gbdt_reg` and
    `gbdt_poisson_reg` are accounted for by amendment 1.
* `selection_check(card)` checks the selection:
  * the kept model is the validation argmin (ties go to the lower complexity) unless a simple baseline beat it on
    test;
  * it is no worse than any baseline on the primary test metric;
  * the card's metrics are the kept candidate's test metrics;
  * for E2 cards: every other candidate is compared with the kept model on test, and none is significantly better
    (``worse_model_kept`` = 0 and agrees with the bootstrap intervals).

## Compatibility (`src/crucible/calibration/registry_e2.py`)

`registry_e2` delegates every call to `registry`, `datasets` or `audit`. It keeps E2's names working for the fitting
scripts (`fit_commercial`, `fit_software`, `fitmeta_e2`), including the `CRUCIBLE_E2_CAL_DIR` redirect for smoke
runs:

* `load`, `model_card`, `verify_checkpoint`, `report` and `list_models` delegate to `registry`. Names must have the
  form `name@version`.
* `prespecified_before_fit` delegates to `audit.plan_check`.
* `datasets` returns E2's datasets only.
* `gate_metrics` returns the Gate F and SP-5.5 quantities over E2's models, counting only the datasets those models
  cite. Its `empirical_domains` are now canonical (`advertising_response`, `software_adoption`).

`commercial.py` and `software.py` subclass `registry.FittedModel` directly. World code should never import
`registry_e2`.

## Measured (this tree, data under `/workspace/data/crucible`)

Gate F over the unified registry:

| quantity | value |
|---|---|
| registered / qualifying models | 9 / 9 |
| empirical domains | 5 (content_selection, browsing_frequency, watch_duration, advertising_response, software_adoption) |
| content consumption / randomised commercial response fitted | true / true |
| held-out reports, baseline comparisons, calibration reports, limit sections | 9, 9, 9, 9 |
| checkpoints saved / hash mismatches | 9 / 0 (largest 830,690 bytes) |
| datasets with checksums / complete provenance | 4 / 4 |
| synthetic-only domains | 0 |

F.3 recomputes the held-out metric of all nine models from the data and the checkpoints through `load`:

* the six E1 models on their test splits;
* `software_deep_engagement` (83,304 test stars) and `software_repo_growth` (321,303 test repo-days) on their whole
  test splits;
* `commercial_response` on the report's seeded audit subsample of 100,000 test users.

Across the nine models the largest relative difference between the reported and the recomputed metric is 1.9e-16.

SP-5.5 checks 9 models: every plan is prespecified before its fit, `worse_model_kept` is 0, and 9 learned models
were kept. SP-5.4 finds 31 important mechanisms, all with a complete record, and 12 v5 register rows with no
problems. These numbers come from a run of the six test modules at commit 1426ada on a clean tree: 59 tests passed in
7 min 18 s with 2 CPUs.

## Reproducing

```bash
cd crucible
PYTHONPATH=src python -c "from crucible.calibration import acquire as a; [a(n) for n in ('kuairand_pure', 'mind_small', 'criteo_uplift_v2.1')]"
PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "E acquire gharchive" -- \
    env CRUCIBLE_ACQUIRE_WORKERS=4 PYTHONPATH=src python -c "from crucible.calibration import acquire; acquire('gharchive_2024-10-07_96h')"
PYTHONPATH=src python -m crucible.scale.slots --cpu 2 --label "E: calibration tests" -- \
    env OMP_NUM_THREADS=2 PYTHONPATH=src python -m pytest -q tests/test_v5_calibration_core.py \
    tests/test_v5_calibration_commercial.py tests/test_v5_calibration_software.py \
    tests/v5/test_gate_F.py tests/v5/test_gate_SP_5_4.py tests/v5/test_gate_SP_5_5.py
```

The fits themselves are in `components/E1.md` and `components/E2.md`; they write cards in their own schema, and
both are loaded as above.

## Adding a model

Write the card in either schema under `docs/v5/calibration/models/<name>/`. The simplest path is `fitlib.finish`
or `registry.save_model`. A model that needs simulator-facing methods can instead name a `FittedModel` subclass in
``loader``. The card needs:

* a domain that is in `DOMAINS` or mapped by `DOMAIN_ALIASES`;
* datasets that `datasets()` serves (a committed manifest, or `register` from `datasets_<x>.py`);
* an evaluation plan committed before the fit;
* a held-out report.

The gates count the model as soon as it is merged. F.3 requires a reproducer in `reproduce.REPRODUCERS` for every
qualifying model.

## Limitations

* The two card schemas coexist; the cards were not migrated. Changing a card would change the files the gates and
  the superseded run-1 disclosure point to. The cost is that two readers must stay in step: `_view_kind_card` and
  `_view_loader_card` in `registry.py`, and `_spec_problems_keyed` and `_spec_problems_listed` in `audit.py`.
* For E2's one-model plan (`E2_commercial_response.plan.json`) the primary metric is a definition in words. The audit
  accepts the card's `primary_*` key as that metric. It cannot check the key's name against the words.
* The secondary metrics of E2 plans are prose, not metric keys, so `plan_check` does not check that they were
  reported. The E2 unit tests recompute them from the data.
* `new_repository_adoption_curve` (listed in `software_repo_growth@1`'s `fidelity_mechanisms`) is a measured
  descriptive reference: it is not fitted, has no held-out evaluation, and no simulator mechanism consumes it. Its
  register row is therefore "plausible, unvalidated" for the simulator side. The measured curve is in the evidence
  column.
* F.3 recomputes `commercial_response`'s primary metric on the 100,000-user audit subsample, not on all 2,795,919
  test users. The full-test number is the fit's own.
* The register rows for E2 now use the cards' mechanism ids. Their categories are exactly one of the six; the
  qualifiers (observational, randomised experiment, validated on held-out users) moved to the evidence column.
