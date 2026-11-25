"""Crucible 5.0 - Empirical calibration: dataset acquisition, behavioural model fitting and held-out evaluation, the fitted-model registry.

Owned by workstream E (docs/v5/WORKSTREAMS.md). Kernel modules of this package live in
`calibration/modules.py` or are imported from it; `crucible.load_all_modules()` imports that file when it exists.

Interface (INTERFACE_CONTRACTS section 5):

    datasets() -> dict[str, dict]       # name -> {url, license, sha256 per file, size, schema, provenance}
    acquire(name, dest=None) -> str     # downloads, verifies checksums, returns the directory
    load(model_name) -> FittedModel     # e.g. "content_consumption@1"; verifies checkpoint hashes

World code depends only on `load`; it never reads datasets. Fitting code (`crucible.calibration.fit_e1`, ...) is run
offline and writes checkpoints, model cards and held-out reports under `docs/v5/calibration/`.
"""
from .datasets import acquire, datasets
from .registry import FittedModel, list_models, load

__all__ = ["FittedModel", "acquire", "datasets", "list_models", "load"]
