"""Kernel modules of `crucible.commerce` (imported by `crucible.load_all_modules()`).

Workstream F1: products (`products`/prod), subscriptions (`subs`), digital marketplaces (`xchg`), business
lifecycle (`ventures`/venture) and independent customers (`patrons`); evaluator metrics in `revenue`.
"""
from . import products, subscriptions, marketplace, business, customers, revenue  # noqa: F401
"""Kernel modules of the commerce package (imported by `crucible.load_all_modules()`).

Workstream F2: institutions (`inst`), escrowed commitments (`commit`) and the labour market (`labor`).
from . import contracts, institutions, labor  # noqa: F401  (F2: modules inst, commit, labor)
