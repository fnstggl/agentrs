"""Kernel modules of workstream G, registered when `crucible.load_all_modules()` imports this file.

* `caps`  - capability registry v5 (subclass of the 4.0 registry; typed declarations, search, trials, adoption)
* `needs` - needs graph with six relation types and the `needs.update` process
* `pkg`   - software ecosystem (packages, releases, dependencies, adoption by execution, PRs, reviews, advisories)
* `adopt` - need-driven participants that discover, test and adopt capabilities
* `infra` - compute providers created at genesis (world construction helper)
"""
from . import caps, needs, pkg, adopt, worlds  # noqa: F401
