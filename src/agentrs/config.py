"""Typed YAML configuration with dotted command-line overrides.

    cfg = load_config(ExperimentConfig, "configs/experiments/dev_D.yaml", ["rl.algo=ppo", "seed=1"])

YAML files may ``inherit: <relative path>`` another file; keys are deep-merged.
Every value is validated by the Pydantic model, so typos fail loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("inherit", None)
    if parent:
        data = _deep_merge(load_yaml(path.parent / parent), data)
    return data


def apply_overrides(d: dict, overrides: Sequence[str]) -> dict:
    d = dict(d)
    for ov in overrides or ():
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        parsed: Any = yaml.safe_load(val)
        cur = d
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise ValueError(f"cannot set {key}: {p} is not a mapping")
        cur[parts[-1]] = parsed
    return d


def load_config(model: type[T], path: Optional[str | Path] = None, overrides: Sequence[str] = ()) -> T:
    d = load_yaml(path) if path else {}
    return model.model_validate(apply_overrides(d, overrides))


def dump_config(cfg: BaseModel, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
