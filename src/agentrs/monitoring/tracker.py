"""Experiment tracking: JSONL metrics + TensorBoard + SQLite run metadata."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


class Tracker:
    def __init__(self, run_dir: str | os.PathLike, use_tensorboard: bool = True, run_name: Optional[str] = None,
                 config: Optional[dict] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception:  # pragma: no cover - tensorboard optional
                self._tb = None
        self.run_name = run_name or self.run_dir.name
        self.t0 = time.time()
        if config is not None:
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        RunRegistry(self.run_dir.parent / "runs.sqlite").register(self.run_name, str(self.run_dir), config or {})

    def log(self, metrics: dict[str, Any], step: int, prefix: str = "") -> None:
        rec = {"step": step, "time": round(time.time() - self.t0, 3)}
        for k, v in metrics.items():
            key = f"{prefix}{k}"
            if isinstance(v, (int, float)) and v == v:  # skip NaN for TB
                rec[key] = v
                if self._tb is not None:
                    self._tb.add_scalar(key, v, step)
            else:
                rec[key] = v if isinstance(v, (str, bool, type(None), list, dict)) else str(v)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()


class RunRegistry:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS runs (name TEXT, dir TEXT, started REAL, config TEXT)")

    def register(self, name: str, run_dir: str, config: dict) -> None:
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT INTO runs VALUES (?,?,?,?)", (name, run_dir, time.time(), json.dumps(config,
                                                                                                default=str)))

    def list(self) -> list[tuple]:
        with sqlite3.connect(self.path) as c:
            return list(c.execute("SELECT name, dir, started FROM runs ORDER BY started"))


def read_metrics(path: str | os.PathLike) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        p = p / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
