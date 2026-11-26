"""Kernel modules of the Tier 2 domains (imported by `crucible.load_all_modules()`): the participant seats (`seat`)
and every module file of this package, so that every domain kernel module (`sci`, `fleet`, `learn`, `firms`, and
whatever a domain file registers) exists in a fresh process and a stored domain world can be created and replayed
without importing anything else (audit I_domains section 2, item 4). Domain files must therefore be cheap to import
(definitions only: never build a world at import)."""
from . import load_all, seats, science  # noqa: F401

load_all()
