"""Registry of authorised sources, revocation, and coverage-driven source selection.

Sources are chosen for diversity and value per cost, not raw event volume: ``discover`` ranks
candidate sources by how much they add to the current coverage (environments / action types not
yet covered), their reliability and their cost, and refuses any candidate whose declaration does
not authorise training use.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from agentrs.sensors.permissions import check_source
from agentrs.sensors.source_adapter import SourceAdapter, SourceDeclaration


class SourceEntry(BaseModel):
    declaration: SourceDeclaration
    status: str = "active"  # active | revoked | paused
    registered_wall: float = Field(default_factory=time.time)
    revoked_reason: str | None = None
    coverage_tags: list[str] = Field(default_factory=list)  # e.g. ["forum", "create_post", "socialweblab"]
    cost_per_observation: float = 1.0
    reliability: float = 1.0


class SourceRegistry:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.entries: dict[str, SourceEntry] = {}
        self.adapters: dict[str, SourceAdapter] = {}
        if self.path and self.path.exists():
            for d in json.loads(self.path.read_text()):
                e = SourceEntry.model_validate(d)
                self.entries[e.declaration.source_id] = e

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([e.model_dump() for e in self.entries.values()], indent=1))

    def register(self, adapter: SourceAdapter, coverage_tags: list[str] | None = None, cost: float = 1.0,
                 reliability: float = 1.0) -> SourceEntry:
        decl = adapter.declaration
        ok, why = check_source(decl)
        if not ok:
            raise PermissionError(f"source {decl.source_id} refused: {'; '.join(why)}")
        e = SourceEntry(declaration=decl, coverage_tags=coverage_tags or [], cost_per_observation=cost,
                        reliability=reliability)
        prev = self.entries.get(decl.source_id)
        if prev is not None and prev.status == "revoked":
            raise PermissionError(f"source {decl.source_id} was revoked ({prev.revoked_reason}); re-authorise explicitly")
        self.entries[decl.source_id] = e
        self.adapters[decl.source_id] = adapter
        self._save()
        return e

    def active(self) -> list[str]:
        out = []
        for sid, e in self.entries.items():
            if e.status != "active" or sid not in self.adapters:
                continue
            ok, _ = check_source(e.declaration)
            if ok:
                out.append(sid)
        return out

    def revoke(self, source_id: str, reason: str) -> None:
        e = self.entries.get(source_id)
        if e is None:
            return
        e.status = "revoked"
        e.revoked_reason = reason
        self.adapters.pop(source_id, None)
        self._save()

    def declaration(self, source_id: str) -> SourceDeclaration:
        return self.entries[source_id].declaration

    @staticmethod
    def discover(candidates: list[SourceEntry], covered: dict[str, int], budget: int) -> list[tuple[str, float, str]]:
        """Rank candidates by marginal coverage value / cost. ``covered``: tag -> examples already held.
        Returns [(source_id, score, reason)] for the chosen ones (unauthorised ones get score 0)."""
        ranked = []
        for e in candidates:
            ok, why = check_source(e.declaration)
            if not ok:
                ranked.append((e.declaration.source_id, 0.0, "refused: " + "; ".join(why)))
                continue
            novelty = sum(1.0 / (1.0 + covered.get(t, 0)) for t in e.coverage_tags) or 0.01
            score = novelty * e.reliability / max(1e-6, e.cost_per_observation)
            ranked.append((e.declaration.source_id, score, f"coverage gain {novelty:.2f}"))
        ranked.sort(key=lambda x: -x[1])
        return [r for r in ranked if r[1] > 0][:budget] + [r for r in ranked if r[1] == 0]
