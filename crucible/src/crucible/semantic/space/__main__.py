"""Command line of the shared space (run through the slot scheduler; see each command's module).

    python -m crucible.semantic.space info
    python -m crucible.semantic.space fit --report out.json           # image alignment map (writes space/maps/)
    python -m crucible.semantic.space build-prior [--report out.json] # MIND interest prior for this SPACE_VERSION
    python -m crucible.semantic.space validate --report out.json      # fine-grained, locality, shortcuts, alignment
"""
from __future__ import annotations

import argparse
import json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crucible.semantic.space")
    ap.add_argument("cmd", choices=["info", "fit", "build-prior", "validate"])
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    from . import DIM, SPACE_ID, SPACE_VERSION, layout, preprocessing
    if a.cmd == "info":
        print(json.dumps({"space": SPACE_ID, "version": SPACE_VERSION, "dim": DIM, "layout": layout(),
                          "preprocessing": preprocessing()}, indent=1, default=str))
    elif a.cmd == "fit":
        from . import fit
        r = fit.run(report=a.report, save=True)
        print(json.dumps({k: v for k, v in r.items() if k != "images"}, indent=1))
    elif a.cmd == "build-prior":
        from . import prior
        print(json.dumps(prior.build(SPACE_VERSION, report=a.report), indent=1))
    else:
        from . import validate
        out = {"finegrained_test": validate.finegrained("test"), "finegrained_dev": validate.finegrained("dev"),
               "long_document": validate.long_document_locality(), "shortcuts": validate.shortcut_resistance()}
        if a.report:
            with open(a.report, "w") as fh:
                json.dump(out, fh, indent=1)
        print(json.dumps(out, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
