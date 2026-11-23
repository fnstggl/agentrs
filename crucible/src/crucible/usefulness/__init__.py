"""The general-purpose usefulness substrate (Crucible 4.0).

In 3.0 the valuable work of the economy was eight hand-written problem formats. Here the set of valuable work is
open: `category` is the extension interface (a `NeedCategory` describes one category of digital need completely; a
`CategoryFamily` generates differently-ruled categories per world; `compose` builds cross-category needs),
`families` holds the built-in procedural families, and `module` is the kernel module `use` (discovery tools, the
paid acceptance lab and the usefulness ledger). The environment always knows whether an artifact works - by
executing it on real requests and scoring the output with the category's exact evaluator - and no LLM judges.
"""
from .category import (FAMILIES, GENERATED, CategoryFamily, DerivedEdge, Externalities, NeedCategory,  # noqa: F401
                       Population, category_families, category_family, category_of, compose, from_config, instantiate,
                       load_families, materialize, measure_tiers, register_category, scoped_registry,
                       unregister_generated)

load_families()
