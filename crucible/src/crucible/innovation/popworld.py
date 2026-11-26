"""Worlds for open-ended creation on the one population (Gates J and L, SP-6.3; D-15, D-18).

`creation_spec(seed, ...)` composes the startup economy of `commerce.worlds` (published work formats with incumbent
SaaS firms, the 4.0 internet, the 5.0 commerce stack, the population and its work activities) and adds work
families of *procedurally generated categories* (`usefulness` families, a fresh hidden rule per world): tasks that
no format registry documents, that no firm serves at genesis and that individuals do by hand. Whether anything
will ever serve them depends on whether some agent notices (the individuals' public requests for help are the only
signal), builds a service whose declared interface accepts the individuals' inputs and whose outputs their
employers accept.

Optional second steps (`then`) make a family a two-step pipeline: automating the first step changes where the
individual's time goes, which is how adoption creates new needs.

Workstream CIV composes the same modules into `crucible.civ.generate`; these specs are the ECON slice used by the
gate tests (integration note).
"""
from __future__ import annotations

from ..core.kernel import WorldSpec


def creation_spec(seed: int, *, n: int = 1200, published: tuple = ("schedule.jobs",), novel: tuple = ("allocate",),
                  novel_share: float = 1.0, then: dict | None = None, work: dict | None = None,
                  extra_modules: tuple = (), params: dict | None = None) -> WorldSpec:
    from ..commerce.worlds import startup_spec
    spec = startup_spec(seed, n=n, formats=list(published), n_formats=len(published), work=work, params=params)
    mods = [dict(m) for m in spec.modules]
    for m in mods:
        if m["name"] == "work":
            cfg = dict(m["config"])
            fams = [dict(f) for f in cfg["families"]]
            for k, fam in enumerate(novel):
                fams.append({"category": {"family": fam, "seed": int(seed) * 100 + k},
                             "share": float(novel_share), "tasks_day": [1.0, 2.5], "diy_min": [15.0, 45.0],
                             "size": [0.2, 0.5], "standard": [0.9, 0.99], "budget_min": [60.0, 200.0]})
            for f in fams:
                if then and f.get("need") in then:
                    f["then"] = then[f["need"]]
            cfg["families"] = fams
            m["config"] = cfg
    for name in extra_modules:
        if not any(m["name"] == name for m in mods):
            mods.insert(len(mods) - 3, {"name": name, "config": {}})
    return WorldSpec("creation", int(seed), mods, dict(spec.params))
