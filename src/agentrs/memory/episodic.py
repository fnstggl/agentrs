"""Persistent episodic memory (separate from neural weights).

Stores task histories (goal, family, verified outcome, abstracted action
sequence, errors encountered and how they were resolved) in SQLite and
retrieves similar episodes with a hashed bag-of-words embedding. Retrieved
memories are inserted into the belief state's *inferred* section as hints;
they are context, not learned capability.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from agentrs.actions.schema import try_parse_action
from agentrs.data.schemas import EpisodeRecord
from agentrs.environments.base import PublicTask

DIM = 256


def embed(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    toks = re.findall(r"[a-z]+", text.lower())
    # quoted spans carry instance-specific values; down-weight them via the token set itself
    for tok in toks:
        h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
        v[h % DIM] += 1.0 if (h >> 9) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def goal_template(goal: str) -> str:
    """Remove instance-specific quoted values so retrieval matches task structure."""
    return re.sub(r"[\"“][^\"”]*[\"”]", '"X"', goal)


def abstract_steps(ep: EpisodeRecord) -> list[str]:
    """Id-free description of an episode's actions, e.g. 'type into "Email"' / 'click "Submit"'."""
    out = []
    for t in ep.transitions:
        a = try_parse_action(t.action)
        if a is None:
            continue
        el = t.observation.element(a.target) if a.target else None
        name = f' "{el.name}"' if el is not None else ""
        verb = {"type": "type into", "select": "choose in"}.get(a.type.value, a.type.value)
        s = f"{verb}{name}" + (" -> error" if not t.action_valid or t.error else "")
        out.append(s)
    return out


class EpisodicMemory:
    def __init__(self, path: str | Path, max_hints: int = 2, only_successful_hints: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_hints = max_hints
        self.only_successful = only_successful_hints
        with sqlite3.connect(self.path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, task_id TEXT, family TEXT,
                         split TEXT, goal TEXT, template TEXT, success INTEGER, steps TEXT, errors TEXT,
                         emb BLOB)""")

    def add_episode(self, ep: EpisodeRecord) -> None:
        if ep.split != "train":  # never memorise held-out evaluation tasks
            return
        steps = abstract_steps(ep)
        errors = [t.error for t in ep.transitions if t.error]
        tmpl = goal_template(ep.goal)
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (ep.episode_id, ep.task_id, ep.family, ep.split, ep.goal, tmpl, int(bool(ep.success)),
                       json.dumps(steps), json.dumps(errors[:5]), embed(tmpl).tobytes()))

    def retrieve(self, goal: str, k: int = 3, successful_only: Optional[bool] = None) -> list[dict]:
        so = self.only_successful if successful_only is None else successful_only
        q = embed(goal_template(goal))
        with sqlite3.connect(self.path) as c:
            rows = c.execute("SELECT family, goal, success, steps, errors, emb FROM episodes"
                             + (" WHERE success=1" if so else "")).fetchall()
        scored = []
        for fam, g, s, steps, errs, emb in rows:
            sim = float(np.dot(q, np.frombuffer(emb, dtype=np.float32)))
            scored.append((sim, {"family": fam, "goal": g, "success": bool(s), "steps": json.loads(steps),
                                 "errors": json.loads(errs), "similarity": sim}))
        scored.sort(key=lambda x: -x[0])
        # de-duplicate identical abstract solutions
        out, seen = [], set()
        for _, r in scored:
            key = tuple(r["steps"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= k:
                break
        return out

    def hints_for(self, task: PublicTask) -> list[str]:
        hints = []
        for r in self.retrieve(task.goal, k=self.max_hints):
            if r["similarity"] < 0.5:
                continue
            hints.append("similar task solved by: " + "; ".join(r["steps"][:6]))
        return hints

    def __len__(self) -> int:
        with sqlite3.connect(self.path) as c:
            return c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

