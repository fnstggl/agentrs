"""OW-1 ... OW-7 task protocols (spec Part 16) and the open-economy smoke protocol. `PROTOCOLS[name]` is a
`base.TaskProtocol` with `calibrate()`."""
from __future__ import annotations

import importlib

NAMES = ("ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7")


def load(name: str):
    if name == "open_economy":
        return importlib.import_module(f"{__name__}.ow1").OPEN_ECONOMY
    return importlib.import_module(f"{__name__}.{name}").PROTOCOL


def all_protocols() -> dict:
    out = {n: load(n) for n in NAMES}
    out["open_economy"] = load("open_economy")
    return out
