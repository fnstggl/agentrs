"""Kernel modules of the commerce package (imported by `crucible.load_all_modules()`).

Workstream F2: institutions (`inst`), escrowed commitments (`commit`) and the labour market (`labor`).
"""
from . import contracts, institutions, labor  # noqa: F401  (F2: modules inst, commit, labor)
