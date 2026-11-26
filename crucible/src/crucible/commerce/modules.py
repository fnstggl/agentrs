"""Kernel modules of `crucible.commerce` (imported by `crucible.load_all_modules()`).

Workstream F1: products (`products`/prod), subscriptions (`subs`), digital marketplaces (`xchg`), business
lifecycle (`ventures`/venture) and independent customers (`patrons`); evaluator metrics in `revenue`.
Workstream F2: institutions (`inst`), escrowed commitments (`commit`) and the labour market (`labor`).
"""
from . import products, subscriptions, marketplace, business, customers, revenue  # noqa: F401  (F1)
from . import contracts, institutions, labor  # noqa: F401  (F2: modules inst, commit, labor)
from ..economy import work  # noqa: F401  (5.0 ECON: population work activities, the demand side)
from . import rivals  # noqa: F401  (5.0 ECON: adaptive incumbents)
